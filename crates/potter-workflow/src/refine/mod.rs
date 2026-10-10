use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use jcode_sdk::api::{ApiEvent, ToolConfiguration};
use jcode_sdk::{
    ConnectOptions, CreateSessionOptions, JcodeClient, LaunchOptions, RunOptions,
    RunStructuredOptions, StructuredEventCallback, TurnResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::potter_bridge;
use potter_core::cli::PreviewMode;

mod prompts;

/// Model from reference images, then render, review, and repair until no blocking differences remain.
///
/// Loop: render the model from several views, have a fresh reviewer list blocking differences
/// from the references, and let the modeler repair exactly those. Stops when no blocking findings
/// remain, progress stalls (--stall-limit), or --max-iterations is reached.
///
/// Writes the potter project to <OUT>/scene and each iteration's renders, review.json, and
/// fix-report.json to <OUT>/iter-NN. Exits 0 when converged and 2 otherwise.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// Reference image (png, jpg, jpeg, webp, gif); repeat for more views.
    #[arg(short = 'i', long = "image", required = true, value_name = "PATH")]
    images: Vec<PathBuf>,

    /// Output directory [default: <first image stem>-refine in the working directory, numbered when taken].
    #[arg(short, long, value_name = "DIR")]
    out: Option<PathBuf>,

    /// Render views in order.
    #[arg(
        long,
        value_delimiter = ',',
        value_parser = parse_view,
        default_value = "front,back,left,right,top,iso"
    )]
    views: Vec<String>,

    /// Preview rendering mode. `beauty` needs scene lighting: a freshly initialized scene renders black.
    #[arg(long, value_enum, default_value = "solid")]
    mode: PreviewMode,

    /// Render size in pixels.
    #[arg(long, default_value_t = 512)]
    size: u32,

    /// Maximum number of reviews.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    max_iterations: u64,

    /// Stop as `stalled` after this many consecutive reviews without fewer findings than the best review so far.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u64).range(1..))]
    stall_limit: u64,

    /// jcode provider that serves --model, e.g. openai, openai-api, claude-oauth, copilot,
    /// openrouter.
    #[arg(short = 'p', long, value_name = "NAME")]
    provider: String,

    /// Model ID used for the modeler and each reviewer.
    #[arg(short = 'm', long, value_name = "ID")]
    model: String,
}

fn parse_view(view: &str) -> Result<String, String> {
    match view {
        "front" | "back" | "right" | "left" | "top" | "bottom" | "iso" => Ok(view.to_owned()),
        _ => Err(format!(
            "unknown view {view:?}; choose front, back, right, left, top, bottom, or iso"
        )),
    }
}

/// `<first image stem>-refine` in the working directory, suffixed `-2`, `-3`, ... when taken.
fn default_out(first_image: &Path) -> PathBuf {
    let stem = first_image
        .file_stem()
        .map_or_else(|| "model".into(), |stem| stem.to_string_lossy());
    let base = format!("{stem}-refine");
    let mut out = PathBuf::from(&base);
    let mut suffix = 1;
    while out.exists() {
        suffix += 1;
        out = PathBuf::from(format!("{base}-{suffix}"));
    }
    out
}

/// One reviewer verdict. Only `findings` block convergence; `non_findings` record decided
/// observations so later reviews do not raise them again.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Review {
    summary: String,
    findings: Vec<Finding>,
    resolved: Vec<String>,
    non_findings: Vec<NonFinding>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Finding {
    id: String,
    status: FindingStatus,
    category: FindingCategory,
    views: Vec<String>,
    description: String,
    fix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FindingStatus {
    New,
    Persists,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FindingCategory {
    RenderProblem,
    MissingPart,
    ExtraPart,
    Shape,
    Proportion,
    Placement,
    ColorMaterial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NonFinding {
    item: String,
    classification: NonFindingClass,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NonFindingClass {
    BelowTolerance,
    RenderArtifact,
    NotInReference,
    AcceptedBlocker,
}

/// The modeler's account of one repair turn, handed to the next reviewer.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixReport {
    summary: String,
    results: Vec<FixResult>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixResult {
    id: String,
    status: FixStatus,
    note: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FixStatus {
    Fixed,
    Blocked,
}

#[derive(Serialize)]
struct ReviewHistory<'a> {
    previous_review: &'a Review,
    modeler_report: &'a FixReport,
    /// Every non-finding from every earlier review, so a decision outlives the review after it.
    decided_non_findings: &'a [NonFinding],
}

/// Loop monitor: counts consecutive reviews that fail to report fewer findings than the best
/// review so far.
#[derive(Default)]
struct Progress {
    fewest: Option<usize>,
    stale_reviews: u64,
}

impl Progress {
    fn record(&mut self, findings: usize) -> u64 {
        if self.fewest.is_none_or(|fewest| findings < fewest) {
            self.fewest = Some(findings);
            self.stale_reviews = 0;
        } else {
            self.stale_reviews += 1;
        }
        self.stale_reviews
    }
}

/// Gives every finding a stable ID: a `persists` finding keeps its previous ID, anything else
/// (new, unknown, or duplicated IDs) gets the next `F<n>`.
fn assign_finding_ids(findings: &mut [Finding], previous: &[Finding], last_id: &mut u32) {
    let mut carried_ids = HashSet::new();
    for finding in findings {
        let carried = finding.status == FindingStatus::Persists
            && previous.iter().any(|old| old.id == finding.id)
            && carried_ids.insert(finding.id.clone());
        if !carried {
            *last_id += 1;
            finding.id = format!("F{last_id}");
            finding.status = FindingStatus::New;
        }
    }
}

pub(crate) fn run(args: Args) -> anyhow::Result<crate::Finished> {
    let reference_images = load_references(&args.images)?;
    let out = match args.out {
        Some(out) => out,
        None => default_out(args.images.first().context("no --image given")?),
    };
    let cwd = std::env::current_dir().context("getting current working directory")?;
    let client = JcodeClient::launch(LaunchOptions {
        working_dir: Some(cwd),
        client_name: concat!("potter-workflow-refine/", env!("CARGO_PKG_VERSION")).into(),
        // A fresh private home has no cached jcode build, so the auto-updater would download the
        // latest release and restart the runtime mid-turn: the turn drops with "harness connection
        // closed" and the restarted server outlives this process.
        env: [("JCODE_NO_AUTO_UPDATE".into(), "1".into())].into(),
        ..LaunchOptions::default()
    })
    .context("launching private jcode runtime")?;
    if !client.supports("session_tools") {
        bail!("the jcode runtime is too old: it does not support session_tools; upgrade jcode");
    }
    // The bridge attaches one session per connection, and session tool policies are dropped when a
    // session is re-attached, so reviewers get their own connection and the modeler keeps its tools.
    let reviewer_client = JcodeClient::connect(ConnectOptions {
        socket_path: Some(client.socket_path().to_path_buf()),
        client_name: concat!(
            "potter-workflow-refine-reviewer/",
            env!("CARGO_PKG_VERSION")
        )
        .into(),
        ensure_runtime: false,
        ..ConnectOptions::default()
    })
    .context("connecting reviewer to the private jcode runtime")?;
    std::fs::create_dir_all(&out)
        .with_context(|| format!("creating output directory {}", out.display()))?;
    eprintln!("output: {}", out.display());
    let scene = out.join("scene");
    potter_bridge::init_scene(&scene)?;
    // jcode rejects relative session working directories.
    let scene = std::fs::canonicalize(&scene)
        .with_context(|| format!("resolving scene path {}", scene.display()))?;

    // jcode routes `<provider>:<model>` to that provider.
    let model = format!("{}:{}", args.provider, args.model);
    let mut agents = JcodeAgents::new(client, reviewer_client, scene.clone(), model)?;
    let outcome = refine_loop(
        &mut agents,
        &reference_images,
        &LoopSettings {
            scene: &scene,
            out: &out,
            views: &args.views,
            mode: args.mode,
            size: args.size,
            max_iterations: args.max_iterations,
            stall_limit: args.stall_limit,
        },
    )?;

    Ok(crate::Finished {
        summary: json!({
            "status": outcome.status.as_str(),
            "iterations": outcome.iterations,
            "remaining_findings": outcome.remaining_findings,
            "scene": scene.to_string_lossy(),
            "out": out.to_string_lossy(),
            "last_review": outcome.last_review.to_string_lossy(),
        }),
        reached_goal: outcome.status == Status::Converged,
    })
}

trait Agents {
    /// Initial modeling turn; the modeler edits the scene through its potter tools.
    fn build(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<()>;
    /// One review in a fresh, tool-less reviewer session.
    fn review(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<Review>;
    /// One repair turn in the persistent modeler session.
    fn fix(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<FixReport>;
}

struct JcodeAgents {
    modeler: JcodeClient,
    reviewer: JcodeClient,
    modeler_session: String,
    scene: PathBuf,
    model: String,
}

impl JcodeAgents {
    fn new(
        modeler: JcodeClient,
        reviewer: JcodeClient,
        scene: PathBuf,
        model: String,
    ) -> anyhow::Result<Self> {
        let modeler_session = create_session(&modeler, &scene, prompts::modeler_system())?;
        let tool_definitions = potter_bridge::tool_definitions()?;
        let enabled_tools = tool_definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect();
        modeler
            .configure_tools(
                &modeler_session,
                ToolConfiguration {
                    enabled: Some(enabled_tools),
                    disabled: Vec::new(),
                    custom: tool_definitions,
                },
            )
            .context("configuring modeler tools")?;
        set_model(&modeler, &modeler_session, &model)?;

        Ok(Self {
            modeler,
            reviewer,
            modeler_session,
            scene,
            model,
        })
    }
}

impl Agents for JcodeAgents {
    fn build(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<()> {
        let initial_turn = self
            .modeler
            .run(
                &self.modeler_session,
                prompt,
                RunOptions {
                    images,
                    on_event: {
                        let handler = tool_handler(self.modeler.clone(), self.scene.clone());
                        Some(Box::new(move |event: &ApiEvent| handler(event)))
                    },
                    auto_approve: false,
                },
            )
            .context("running initial modeling turn")?;
        report_modeler_turn(&initial_turn)
    }

    fn review(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<Review> {
        let reviewer_session = create_session(
            &self.reviewer,
            &self.scene,
            prompts::reviewer_system().into(),
        )?;
        self.reviewer
            .configure_tools(
                &reviewer_session,
                ToolConfiguration {
                    enabled: Some(Vec::new()),
                    disabled: Vec::new(),
                    custom: Vec::new(),
                },
            )
            .context("disabling tools for reviewer")?;
        set_model(&self.reviewer, &reviewer_session, &self.model)?;

        Ok(self
            .reviewer
            .run_structured::<Review>(
                &reviewer_session,
                prompt,
                RunStructuredOptions {
                    images,
                    ..RunStructuredOptions::new(review_schema())
                },
            )
            .context("reviewing rendered model")?
            .data)
    }

    fn fix(&mut self, prompt: &str, images: Vec<(String, String)>) -> anyhow::Result<FixReport> {
        Ok(self
            .modeler
            .run_structured::<FixReport>(
                &self.modeler_session,
                prompt,
                RunStructuredOptions {
                    images,
                    on_event: Some(tool_handler(self.modeler.clone(), self.scene.clone())),
                    ..RunStructuredOptions::new(fix_report_schema())
                },
            )?
            .data)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Converged,
    Stalled,
    MaxIterationsReached,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::Stalled => "stalled",
            Self::MaxIterationsReached => "max_iterations_reached",
        }
    }
}

struct LoopSettings<'a> {
    scene: &'a Path,
    out: &'a Path,
    views: &'a [String],
    mode: PreviewMode,
    size: u32,
    max_iterations: u64,
    stall_limit: u64,
}

struct Outcome {
    status: Status,
    iterations: u64,
    remaining_findings: usize,
    last_review: PathBuf,
}

fn refine_loop(
    agents: &mut impl Agents,
    references: &[(String, String)],
    settings: &LoopSettings<'_>,
) -> anyhow::Result<Outcome> {
    agents.build(
        &prompts::initial_modeling(references.len()),
        references.to_vec(),
    )?;

    let mut status = Status::MaxIterationsReached;
    let mut iterations = 0;
    let mut last_review_path = None;
    let mut remaining_findings = 0;
    let mut progress = Progress::default();
    let mut last_finding_id = 0;
    let mut history: Option<(Review, FixReport)> = None;
    let mut decided_non_findings = Vec::new();

    for iteration in 1..=settings.max_iterations {
        iterations = iteration;
        let iteration_dir = settings.out.join(format!("iter-{iteration:02}"));
        std::fs::create_dir_all(&iteration_dir)
            .with_context(|| format!("creating iteration directory {}", iteration_dir.display()))?;

        let rendered = potter_bridge::render_views(
            settings.scene,
            &iteration_dir,
            settings.views,
            settings.mode,
            settings.size,
        )?;
        let floating_parts = potter_bridge::floating_parts(settings.scene)?;
        let render_images = rendered
            .pngs
            .iter()
            .map(|path| encode_png(path))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut review_images = references.to_vec();
        review_images.extend(render_images.iter().cloned());
        let previous = history
            .as_ref()
            .map(|(previous_review, modeler_report)| {
                serde_json::to_string_pretty(&ReviewHistory {
                    previous_review,
                    modeler_report,
                    decided_non_findings: &decided_non_findings,
                })
            })
            .transpose()?;
        let review_prompt = prompts::review(
            references.len(),
            settings.views,
            &rendered.warnings,
            &floating_parts,
            previous.as_deref(),
        )?;
        let mut review = agents.review(&review_prompt, review_images)?;
        let previous_findings = history.as_ref().map_or(&[][..], |(old, _)| &old.findings);
        assign_finding_ids(
            &mut review.findings,
            previous_findings,
            &mut last_finding_id,
        );
        decided_non_findings.extend(review.non_findings.iter().cloned());

        remaining_findings = review.findings.len();
        eprintln!(
            "iteration {iteration}: {remaining_findings} finding(s), {} resolved, {} non-finding(s), {} floating part(s)",
            review.resolved.len(),
            review.non_findings.len(),
            floating_parts.len()
        );
        eprintln!("summary: {}", review.summary);
        let review_path = iteration_dir.join("review.json");
        write_json(&review_path, &review)?;
        last_review_path = Some(review_path);

        if review.findings.is_empty() {
            status = Status::Converged;
            break;
        }
        if progress.record(remaining_findings) >= settings.stall_limit {
            status = Status::Stalled;
            break;
        }
        if iteration == settings.max_iterations {
            break;
        }

        let fix_report = agents
            .fix(
                &prompts::fix(
                    iteration,
                    &serde_json::to_string_pretty(&review.findings)?,
                    settings.views,
                ),
                render_images,
            )
            .context("running modeler fix turn")?;
        eprintln!("modeler: {}", fix_report.summary);
        write_json(&iteration_dir.join("fix-report.json"), &fix_report)?;
        history = Some((review, fix_report));
    }

    Ok(Outcome {
        status,
        iterations,
        remaining_findings,
        last_review: last_review_path.context("no review was produced")?,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
}

fn review_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary", "findings", "resolved", "non_findings"],
        "properties": {
            "summary": { "type": "string" },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "status", "category", "views", "description", "fix"],
                    "properties": {
                        "id": { "type": "string" },
                        "status": { "type": "string", "enum": ["new", "persists"] },
                        "category": {
                            "type": "string",
                            "enum": [
                                "render_problem", "missing_part", "extra_part", "shape",
                                "proportion", "placement", "color_material"
                            ]
                        },
                        "views": { "type": "array", "items": { "type": "string" } },
                        "description": { "type": "string" },
                        "fix": { "type": "string" }
                    }
                }
            },
            "resolved": { "type": "array", "items": { "type": "string" } },
            "non_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["item", "classification", "reason"],
                    "properties": {
                        "item": { "type": "string" },
                        "classification": {
                            "type": "string",
                            "enum": [
                                "below_tolerance", "render_artifact", "not_in_reference",
                                "accepted_blocker"
                            ]
                        },
                        "reason": { "type": "string" }
                    }
                }
            }
        }
    })
}

fn fix_report_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary", "results"],
        "properties": {
            "summary": { "type": "string" },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "status", "note"],
                    "properties": {
                        "id": { "type": "string" },
                        "status": { "type": "string", "enum": ["fixed", "blocked"] },
                        "note": { "type": "string" }
                    }
                }
            }
        }
    })
}

fn load_references(paths: &[PathBuf]) -> anyhow::Result<Vec<(String, String)>> {
    paths
        .iter()
        .map(|path| {
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            let media_type = match extension.as_deref() {
                Some("png") => "image/png",
                Some("jpg" | "jpeg") => "image/jpeg",
                Some("webp") => "image/webp",
                Some("gif") => "image/gif",
                _ => bail!(
                    "unsupported reference image extension for {} (expected png, jpg, jpeg, webp, or gif)",
                    path.display()
                ),
            };
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading reference image {}", path.display()))?;
            Ok((media_type.to_owned(), STANDARD.encode(bytes)))
        })
        .collect()
}

fn encode_png(path: &Path) -> anyhow::Result<(String, String)> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading rendered PNG {}", path.display()))?;
    Ok(("image/png".to_owned(), STANDARD.encode(bytes)))
}

fn create_session(
    client: &JcodeClient,
    scene: &Path,
    system_prompt: String,
) -> anyhow::Result<String> {
    let session = client
        .create_session_with_options(CreateSessionOptions {
            working_dir: Some(scene.to_string_lossy().into_owned()),
            system_prompt: Some(system_prompt),
        })
        .context("creating agent session")?;
    Ok(session.session_id)
}

fn set_model(client: &JcodeClient, session_id: &str, model: &str) -> anyhow::Result<()> {
    client
        .set_model(session_id, model)
        .with_context(|| format!("setting session model to {model}"))?;
    Ok(())
}

fn report_modeler_turn(turn: &TurnResult) -> anyhow::Result<()> {
    eprintln!("modeler: {}", turn.final_text.trim());
    if let Some(reason) = &turn.stop_reason {
        bail!(
            "agent turn stopped ({reason:?}): {}",
            turn.stop_message
                .as_deref()
                .unwrap_or("no stop message provided")
        );
    }
    Ok(())
}

fn tool_handler(client: JcodeClient, scene: PathBuf) -> StructuredEventCallback {
    Arc::new(move |event: &ApiEvent| {
        if let ApiEvent::ToolCall {
            session_id,
            call_id,
            name,
            input,
        } = event
        {
            let potter_bridge::ToolOutcome { output, error } =
                potter_bridge::execute_tool(&scene, name, input);
            if let Some(message) = error.as_deref() {
                eprintln!("[tool] {name}: error: {message} (input: {input})");
            } else {
                eprintln!("[tool] {name}: ok");
            }
            if let Err(submit_error) =
                client.submit_tool_result(session_id, call_id, &output, error)
            {
                eprintln!("[tool] {name}: failed to submit result: {submit_error}");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};

    use anyhow::Context as _;
    use serde_json::{Value, json};

    use super::{
        Agents, Finding, FindingCategory, FindingStatus, FixReport, LoopSettings, Progress, Review,
        Status, assign_finding_ids, fix_report_schema, refine_loop, review_schema,
    };

    fn finding(id: &str, status: FindingStatus) -> Finding {
        Finding {
            id: id.to_owned(),
            status,
            category: FindingCategory::Shape,
            views: vec!["front".to_owned()],
            description: String::new(),
            fix: String::new(),
        }
    }

    #[test]
    fn persisting_findings_keep_ids_and_everything_else_gets_fresh_ones() {
        let previous = [
            finding("F1", FindingStatus::New),
            finding("F2", FindingStatus::New),
        ];
        let mut current = [
            finding("F2", FindingStatus::Persists),
            finding("F9", FindingStatus::Persists),
            finding("new", FindingStatus::New),
            finding("F2", FindingStatus::Persists),
        ];
        let mut last_id = 2;
        assign_finding_ids(&mut current, &previous, &mut last_id);

        let ids: Vec<_> = current.iter().map(|f| (f.id.as_str(), f.status)).collect();
        assert_eq!(
            ids,
            [
                ("F2", FindingStatus::Persists),
                ("F3", FindingStatus::New),
                ("F4", FindingStatus::New),
                ("F5", FindingStatus::New),
            ]
        );
        assert_eq!(last_id, 5);
    }

    #[test]
    fn progress_counts_reviews_since_the_fewest_findings() {
        let mut progress = Progress::default();
        let stale: Vec<_> = [5, 4, 4, 6, 3, 3]
            .into_iter()
            .map(|findings| progress.record(findings))
            .collect();
        assert_eq!(stale, [0, 0, 1, 2, 0, 1]);
    }

    struct ScriptedAgents {
        reviews: VecDeque<Review>,
        build_count: usize,
        review_prompts: Vec<(String, usize)>,
        fix_prompts: Vec<(String, usize)>,
    }

    impl ScriptedAgents {
        fn new(reviews: Vec<Review>) -> Self {
            Self {
                reviews: reviews.into(),
                build_count: 0,
                review_prompts: Vec::new(),
                fix_prompts: Vec::new(),
            }
        }
    }

    impl Agents for ScriptedAgents {
        fn build(&mut self, _prompt: &str, _images: Vec<(String, String)>) -> anyhow::Result<()> {
            self.build_count += 1;
            Ok(())
        }

        fn review(
            &mut self,
            prompt: &str,
            images: Vec<(String, String)>,
        ) -> anyhow::Result<Review> {
            self.review_prompts.push((prompt.to_owned(), images.len()));
            self.reviews
                .pop_front()
                .context("scripted review exhausted")
        }

        fn fix(
            &mut self,
            prompt: &str,
            images: Vec<(String, String)>,
        ) -> anyhow::Result<FixReport> {
            self.fix_prompts.push((prompt.to_owned(), images.len()));
            Ok(FixReport {
                summary: format!("fix marker {}", self.fix_prompts.len()),
                results: Vec::new(),
            })
        }
    }

    fn scripted_review(findings: Vec<Value>, resolved: &[&str]) -> anyhow::Result<Review> {
        Ok(serde_json::from_value(json!({
            "summary": "scripted review",
            "findings": Value::Array(findings),
            "resolved": resolved,
            "non_findings": [],
        }))?)
    }

    fn scripted_finding(id: &str, status: &str) -> Value {
        json!({
            "id": id,
            "status": status,
            "category": "shape",
            "views": ["front"],
            "description": "shape differs",
            "fix": "adjust shape",
        })
    }

    fn test_scene(root: &Path) -> anyhow::Result<PathBuf> {
        let scene = root.join("scene");
        crate::potter_bridge::init_scene(&scene)?;
        let outcome = crate::potter_bridge::execute_tool(
            &scene,
            "pot_apply",
            &json!({
                "batch": {
                    "schema_version": 1,
                    "base_revision": 0,
                    "operations": [{
                        "op": "node.create",
                        "id": "box",
                        "kind": "box",
                        "params": {"size": 1}
                    }]
                }
            }),
        );
        anyhow::ensure!(
            outcome.error.is_none(),
            "adding test box failed: {}",
            outcome.output
        );
        Ok(scene)
    }

    fn references() -> Vec<(String, String)> {
        vec![("image/png".to_owned(), "AAAA".to_owned())]
    }

    #[test]
    fn refine_loop_converges_after_one_repair() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let scene = test_scene(temp.path())?;
        let out = temp.path().join("out");
        let views = vec!["front".to_owned()];
        let settings = LoopSettings {
            scene: &scene,
            out: &out,
            views: &views,
            mode: potter_core::cli::PreviewMode::Solid,
            size: 64,
            max_iterations: 5,
            stall_limit: 2,
        };
        let mut agents = ScriptedAgents::new(vec![
            scripted_review(vec![scripted_finding("new", "new")], &[])?,
            scripted_review(Vec::new(), &["F1"])?,
        ]);

        let outcome = refine_loop(&mut agents, &references(), &settings)?;

        assert_eq!(outcome.status, Status::Converged);
        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.remaining_findings, 0);
        assert_eq!(outcome.last_review, out.join("iter-02/review.json"));
        assert_eq!(agents.build_count, 1);
        assert_eq!(agents.fix_prompts.len(), 1);
        assert_eq!(agents.fix_prompts[0].1, 1);
        assert_eq!(
            agents
                .review_prompts
                .iter()
                .map(|(_, image_count)| *image_count)
                .collect::<Vec<_>>(),
            [2, 2]
        );
        let second_review = agents
            .review_prompts
            .get(1)
            .context("second review prompt was not recorded")?;
        assert!(second_review.0.contains("F1"));
        assert!(second_review.0.contains("fix marker 1"));
        let saved_review: Value =
            serde_json::from_slice(&std::fs::read(out.join("iter-01/review.json"))?)?;
        assert_eq!(saved_review["findings"][0]["id"], "F1");
        assert!(out.join("iter-01/fix-report.json").is_file());
        assert!(!out.join("iter-02/fix-report.json").exists());
        Ok(())
    }

    #[test]
    fn refine_loop_keeps_decided_non_findings_beyond_the_next_review() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let scene = test_scene(temp.path())?;
        let out = temp.path().join("out");
        let views = vec!["front".to_owned()];
        let settings = LoopSettings {
            scene: &scene,
            out: &out,
            views: &views,
            mode: potter_core::cli::PreviewMode::Solid,
            size: 64,
            max_iterations: 3,
            stall_limit: 5,
        };
        let first: Review = serde_json::from_value(json!({
            "summary": "scripted review",
            "findings": [scripted_finding("new", "new")],
            "resolved": [],
            "non_findings": [{
                "item": "decided seam marker",
                "classification": "render_artifact",
                "reason": "shading, not geometry"
            }],
        }))?;
        let mut agents = ScriptedAgents::new(vec![
            first,
            scripted_review(vec![scripted_finding("F1", "persists")], &[])?,
            scripted_review(vec![scripted_finding("F1", "persists")], &[])?,
        ]);

        refine_loop(&mut agents, &references(), &settings)?;

        let third_review = agents
            .review_prompts
            .get(2)
            .context("third review prompt was not recorded")?;
        assert!(third_review.0.contains("decided seam marker"));
        Ok(())
    }

    #[test]
    fn refine_loop_stops_at_max_iterations_without_a_final_repair() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let scene = test_scene(temp.path())?;
        let out = temp.path().join("out");
        let views = vec!["front".to_owned()];
        let settings = LoopSettings {
            scene: &scene,
            out: &out,
            views: &views,
            mode: potter_core::cli::PreviewMode::Solid,
            size: 64,
            max_iterations: 2,
            stall_limit: 5,
        };
        let mut agents = ScriptedAgents::new(vec![
            scripted_review(vec![scripted_finding("new", "new")], &[])?,
            scripted_review(vec![scripted_finding("F1", "persists")], &[])?,
        ]);

        let outcome = refine_loop(&mut agents, &references(), &settings)?;

        assert_eq!(outcome.status, Status::MaxIterationsReached);
        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.remaining_findings, 1);
        assert_eq!(agents.build_count, 1);
        assert_eq!(agents.fix_prompts.len(), 1);
        assert!(out.join("iter-02/review.json").is_file());
        assert!(!out.join("iter-02/fix-report.json").exists());
        Ok(())
    }

    #[test]
    fn refine_loop_stops_when_stalled() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let scene = test_scene(temp.path())?;
        let out = temp.path().join("out");
        let views = vec!["front".to_owned()];
        let settings = LoopSettings {
            scene: &scene,
            out: &out,
            views: &views,
            mode: potter_core::cli::PreviewMode::Solid,
            size: 64,
            max_iterations: 5,
            stall_limit: 2,
        };
        let mut agents = ScriptedAgents::new(vec![
            scripted_review(
                vec![
                    scripted_finding("first", "new"),
                    scripted_finding("second", "new"),
                ],
                &[],
            )?,
            scripted_review(
                vec![
                    scripted_finding("third", "new"),
                    scripted_finding("fourth", "new"),
                ],
                &[],
            )?,
            scripted_review(
                vec![
                    scripted_finding("fifth", "new"),
                    scripted_finding("sixth", "new"),
                ],
                &[],
            )?,
        ]);

        let outcome = refine_loop(&mut agents, &references(), &settings)?;

        assert_eq!(outcome.status, Status::Stalled);
        assert_eq!(outcome.iterations, 3);
        assert_eq!(outcome.remaining_findings, 2);
        assert_eq!(agents.build_count, 1);
        assert_eq!(agents.review_prompts.len(), 3);
        assert_eq!(agents.fix_prompts.len(), 2);
        Ok(())
    }

    fn schema_enum<'a>(schema: &'a Value, pointer: &str) -> anyhow::Result<&'a [Value]> {
        schema
            .pointer(pointer)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .context("schema enum is missing")
    }

    fn assert_review_schema_round_trip(
        validator: &jsonschema::Validator,
        sample: &Value,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            validator.is_valid(sample),
            "sample review does not match schema: {sample}"
        );
        let review: Review = serde_json::from_value(sample.clone())?;
        anyhow::ensure!(
            serde_json::to_value(review)? == *sample,
            "review round trip changed the sample"
        );
        Ok(())
    }

    #[test]
    fn review_schema_enum_values_round_trip_through_review() -> anyhow::Result<()> {
        let schema = review_schema();
        let validator = jsonschema::validator_for(&schema)?;
        let categories = schema_enum(
            &schema,
            "/properties/findings/items/properties/category/enum",
        )?;
        let statuses = schema_enum(&schema, "/properties/findings/items/properties/status/enum")?;
        for category in categories {
            for status in statuses {
                assert_review_schema_round_trip(
                    &validator,
                    &json!({
                        "summary": "complete review",
                        "findings": [{
                            "id": "F1",
                            "status": status,
                            "category": category,
                            "views": ["front"],
                            "description": "shape differs",
                            "fix": "adjust shape"
                        }],
                        "resolved": [],
                        "non_findings": []
                    }),
                )?;
            }
        }

        for classification in schema_enum(
            &schema,
            "/properties/non_findings/items/properties/classification/enum",
        )? {
            assert_review_schema_round_trip(
                &validator,
                &json!({
                    "summary": "complete review",
                    "findings": [],
                    "resolved": [],
                    "non_findings": [{
                        "item": "small seam",
                        "classification": classification,
                        "reason": "within visual tolerance"
                    }]
                }),
            )?;
        }
        Ok(())
    }

    #[test]
    fn fix_report_schema_status_values_round_trip_through_fix_report() -> anyhow::Result<()> {
        let schema = fix_report_schema();
        let validator = jsonschema::validator_for(&schema)?;
        for status in schema_enum(&schema, "/properties/results/items/properties/status/enum")? {
            let sample = json!({
                "summary": "complete fix report",
                "results": [{
                    "id": "F1",
                    "status": status,
                    "note": "updated the shape"
                }]
            });
            anyhow::ensure!(
                validator.is_valid(&sample),
                "sample fix report does not match schema: {sample}"
            );
            let report: FixReport = serde_json::from_value(sample.clone())?;
            anyhow::ensure!(
                serde_json::to_value(report)? == sample,
                "fix report round trip changed the sample"
            );
        }
        Ok(())
    }
}

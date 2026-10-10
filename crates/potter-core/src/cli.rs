use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};

use crate::{commands, response::Envelope};

/// The scene commands of `pot`. The `pot` binary parses them as top-level subcommands and owns
/// argument parsing and `--json` envelope emission.
#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(about = "Create a new project scene.")]
    Init(ScenePath),
    #[command(about = "List committed project history.")]
    History(ScenePath),
    #[command(about = "Restore an earlier committed project state.")]
    Undo(UndoArgs),
    #[command(about = "Reapply a later committed project state.")]
    Redo(UndoArgs),
    #[command(about = "Apply a typed operation batch atomically.")]
    Apply(ApplyArgs),
    #[command(about = "Import a supported scene interchange file.")]
    Import(ImportArgs),
    #[command(about = "Inspect scene structure, state, and features.")]
    Inspect(InspectArgs),
    #[command(about = "Render selected scene views to preview images.")]
    Preview(PreviewArgs),
    #[command(about = "Pick an object or element from a preview.")]
    Pick(PickArgs),
    #[command(about = "Validate a scene and optional export format.")]
    Validate(ValidateArgs),
    #[command(about = "Export a scene to a supported interchange format.")]
    Export(ExportArgs),
    #[command(about = "Render frames to an image sequence or movie.")]
    Render(RenderArgs),
    #[command(about = "Bake simulation, texture, geometry, or animation data.")]
    Bake(BakeArgs),
    #[command(about = "List and verify project assets.")]
    Assets(AssetsArgs),
    #[command(about = "Print a JSON Schema and optional typed catalog.")]
    Schema(SchemaArgs),
}

#[derive(Debug, Args)]
pub struct ScenePath {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
}

#[derive(Debug, Args)]
pub struct UndoArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, help = "Expected current project revision.")]
    pub base_revision: u64,
    #[arg(
        long,
        default_value_t = 1,
        help = "Number of history steps to traverse."
    )]
    pub steps: usize,
}

#[derive(Debug, Args)]
pub struct ApplyArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(
        long,
        required = true,
        help = "Operation batch JSON file, or - for stdin."
    )]
    pub file: PathBuf,
    #[arg(long, help = "Render preview views after applying the batch.")]
    pub preview: Option<String>,
    #[arg(long, help = "Square preview image size in pixels.")]
    pub size: Option<u32>,
    #[arg(long, help = "Validate the batch without committing changes.")]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, required = true, help = "Interchange file to import.")]
    pub file: PathBuf,
    #[arg(long, required = true, value_enum, help = "Input format.")]
    pub format: ExchangeFormat,
    #[arg(long, required = true, help = "Expected current project revision.")]
    pub base_revision: u64,
    #[arg(long, value_enum, default_value_t = ImportMode::Append, help = "Append to or replace the project.")]
    pub mode: ImportMode,
    #[arg(long, value_enum, default_value_t = AssetPolicy::Copy, help = "Copy imported assets or keep external links.")]
    pub asset_policy: AssetPolicy,
    #[arg(long, help = "Accept reported data loss during import.")]
    pub allow_lossy: bool,
    #[arg(long, help = "Validate the import without committing changes.")]
    pub dry_run: bool,
    #[arg(long, help = "Blender executable used only for .blend exchange.")]
    pub blender: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct Context {
    #[arg(long, help = "Scene ID to evaluate; defaults to the active scene.")]
    pub scene_id: Option<String>,
    #[arg(long, help = "View Layer ID to evaluate.")]
    pub view_layer: Option<String>,
    #[arg(long, help = "Finite evaluation frame, including subframes.")]
    pub frame: Option<f64>,
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, group = "filter", help = "Inspect one entity by ID.")]
    pub id: Option<String>,
    #[arg(long, group = "filter", help = "Inspect entities matching a tag.")]
    pub tag: Option<String>,
    #[arg(long, help = "Include the feature capability catalog.")]
    pub features: bool,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct PreviewArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, group = "view", help = "Comma-separated preview views.")]
    pub views: Option<String>,
    #[arg(long, group = "view", help = "Camera ID for the preview.")]
    pub camera: Option<String>,
    #[arg(long, value_enum, default_value_t = PreviewMode::Solid, help = "Preview shading and output mode.")]
    pub mode: PreviewMode,
    #[arg(
        long,
        default_value_t = 768,
        help = "Square preview image size in pixels."
    )]
    pub size: u32,
    #[arg(long, help = "Output file or directory for preview images.")]
    pub out: Option<PathBuf>,
    #[arg(long, help = "Replace existing preview outputs.")]
    pub overwrite: bool,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct PickArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(
        long,
        required = true,
        help = "Preview manifest produced by pot preview."
    )]
    pub render: PathBuf,
    #[arg(long, required = true, value_parser = parse_pixel, help = "Pixel coordinate as x,y.")]
    pub pixel: (u32, u32),
    #[arg(long, value_enum, default_value_t = PickDomain::Object, help = "Entity or element domain to pick.")]
    pub domain: PickDomain,
}

#[derive(Debug, Args)]
pub struct ValidateArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, help = "Fail when validation reports any issue.")]
    pub strict: bool,
    #[arg(long, value_enum, help = "Check loss conditions for an export format.")]
    pub format: Option<ExchangeFormat>,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct ExportArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, required = true, value_enum, help = "Output interchange format.")]
    pub format: ExchangeFormat,
    #[arg(long, required = true, help = "Output file path.")]
    pub out: PathBuf,
    #[arg(long, help = "Allow export when loss conditions are reported.")]
    pub allow_lossy: bool,
    #[arg(long, help = "Pack supported external assets into the output.")]
    pub pack: bool,
    #[arg(long, help = "Replace existing output files.")]
    pub overwrite: bool,
    #[arg(long, help = "Blender executable used only for .blend exchange.")]
    pub blender: Option<PathBuf>,
    #[arg(long, help = "Orthographic view for SVG or PDF export.")]
    pub view: Option<String>,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct RenderArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, help = "Camera ID used for rendering.")]
    pub camera: Option<String>,
    #[arg(long, value_enum, help = "Rendering engine.")]
    pub engine: Option<RenderEngine>,
    #[arg(long, value_enum, default_value_t = RenderDevice::Cpu, help = "Rendering device.")]
    pub device: RenderDevice,
    #[arg(long, help = "Frame range as start:end[:step].")]
    pub frames: Option<String>,
    #[arg(
        long,
        required = true,
        value_enum,
        help = "Output image or movie format."
    )]
    pub format: RenderFormat,
    #[arg(long, required = true, help = "Output directory.")]
    pub out: PathBuf,
    #[arg(long, help = "Replace existing output files.")]
    pub overwrite: bool,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct BakeArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, required = true, value_enum, help = "Data kind to bake.")]
    pub kind: BakeKind,
    #[arg(long, help = "Target entity ID, when required by the bake kind.")]
    pub target: Option<String>,
    #[arg(long, help = "Frame range as start:end[:step].")]
    pub frames: Option<String>,
    #[arg(long, required = true, help = "Output directory.")]
    pub out: PathBuf,
    #[arg(long, help = "Replace existing output files.")]
    pub overwrite: bool,
    #[command(flatten)]
    pub context: Context,
}

#[derive(Debug, Args)]
pub struct AssetsArgs {
    #[arg(help = "Project directory.")]
    pub scene: PathBuf,
    #[arg(long, help = "Rehash and verify referenced project assets.")]
    pub check: bool,
}

#[derive(Debug, Args)]
pub struct SchemaArgs {
    #[arg(long, value_enum, default_value_t = SchemaKind::Operations, help = "Schema or typed catalog kind.")]
    pub kind: SchemaKind,
    #[arg(long, help = "Operation name for an individual operation schema.")]
    pub op: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ExchangeFormat {
    Blend,
    Glb,
    Gltf,
    Usda,
    Usdc,
    Usd,
    Usdz,
    Alembic,
    Fbx,
    #[value(name = "fbx-binary")]
    FbxBinary,
    Obj,
    Ply,
    Stl,
    Bvh,
    Svg,
    Pdf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ImportMode {
    Append,
    Replace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AssetPolicy {
    Copy,
    Link,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PreviewMode {
    Solid,
    Beauty,
    Wire,
    Normal,
    Depth,
    Id,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PickDomain {
    Object,
    Vertex,
    Edge,
    Face,
    Bone,
    Stroke,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RenderEngine {
    Path,
    Realtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RenderDevice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RenderFormat {
    Png,
    Exr,
    Openexr,
    Jpg,
    Jpeg,
    Tiff,
    Tif,
    Webp,
    Mp4,
    Webm,
    Mov,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BakeKind {
    Simulation,
    Texture,
    Geometry,
    Animation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SchemaKind {
    Scene,
    Operations,
    Preview,
    Response,
    Capabilities,
    Formats,
}

fn parse_pixel(value: &str) -> std::result::Result<(u32, u32), String> {
    let Some((x, y)) = value.split_once(',') else {
        return Err("pixel must be x,y".to_owned());
    };
    if y.contains(',') {
        return Err("pixel must be x,y".to_owned());
    }
    let x = x
        .parse::<u32>()
        .map_err(|_| "pixel x must be an unsigned integer".to_owned())?;
    let y = y
        .parse::<u32>()
        .map_err(|_| "pixel y must be an unsigned integer".to_owned())?;
    Ok((x, y))
}

impl CliCommand {
    fn name(&self) -> &'static str {
        match self {
            Self::Init(_) => "init",
            Self::History(_) => "history",
            Self::Undo(_) => "undo",
            Self::Redo(_) => "redo",
            Self::Apply(_) => "apply",
            Self::Import(_) => "import",
            Self::Inspect(_) => "inspect",
            Self::Preview(_) => "preview",
            Self::Pick(_) => "pick",
            Self::Validate(_) => "validate",
            Self::Export(_) => "export",
            Self::Render(_) => "render",
            Self::Bake(_) => "bake",
            Self::Assets(_) => "assets",
            Self::Schema(_) => "schema",
        }
    }

    /// Runs the command and wraps its outcome in the response envelope `pot` prints.
    #[must_use]
    pub fn execute(self) -> Envelope {
        let name = self.name();
        match dispatch(self) {
            Ok((scene, result)) => Envelope::success(name, scene, result),
            Err(error) => Envelope::failure(Some(name.to_owned()), error),
        }
    }
}

fn dispatch(
    command: CliCommand,
) -> crate::error::Result<(Option<crate::response::SceneInfo>, serde_json::Value)> {
    match command {
        CliCommand::Init(args) => commands::init::run(args),
        CliCommand::History(args) => commands::history::run(args),
        CliCommand::Undo(args) => commands::undo::run(args),
        CliCommand::Redo(args) => commands::redo::run(args),
        CliCommand::Apply(args) => commands::apply::run(args),
        CliCommand::Inspect(args) => commands::inspect::run(args),
        CliCommand::Import(args) => commands::import::run(&args),
        CliCommand::Preview(args) => commands::preview::run(args),
        CliCommand::Pick(args) => commands::pick::run(args),
        CliCommand::Validate(args) => commands::validate::run(args),
        CliCommand::Export(args) => commands::export::run(&args),
        CliCommand::Render(args) => commands::render::run(args),
        CliCommand::Bake(args) => commands::bake::run(args),
        CliCommand::Assets(args) => commands::assets::run(args),
        CliCommand::Schema(args) => commands::schema::run(&args),
    }
}

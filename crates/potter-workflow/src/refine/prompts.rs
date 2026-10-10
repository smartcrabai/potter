use crate::potter_bridge::FloatingPart;

pub(crate) fn modeler_system() -> String {
    format!(
        "You are an expert 3D modeler driving potter through its custom tools.\n\n\
         ## Potter modeling reference\n\n{}\n\n\
         ## Tool adaptation\n\
         The CLI is replaced by the tools pot_apply, pot_inspect, pot_schema, and pot_undo.\n\
         pot_apply maps to `pot apply`, pot_inspect to `pot inspect`, pot_schema to `pot schema --op <op>`, and pot_undo to `pot undo`.\n\
         There is no preview or export tool: the workflow renders the model and a separate reviewer compares the renders with the reference.\n\
         Always call pot_schema for an operation before its first use. For every mutation, take base_revision from the latest tool envelope's `scene.revision`.\n\n\
         ## Modeling rules\n\
         - Build one model centered on the origin, resting on z=0, at a plausible real-world size in meters.\n\
         - Model only the object's geometry and base colors: shadows, highlights, reflections, and shading in the reference are lighting, never extra geometry or decals.\n\
         - Parts that touch in the reference must touch in the model: check contact with pot_inspect bounds after every change.\n\
         - Give parts materials whose base_color matches the reference.\n\n\
         ## Review loop\n\
         After the initial build, each turn lists the reviewer's blocking findings by ID.\n\
         - Repair exactly those findings. Do not change parts no finding mentions: unrequested changes cause regressions.\n\
         - If a finding cannot be fixed by modeling (for example it describes a rendering artifact or a potter limitation), leave the scene as is and report it as blocked with the concrete reason. Never claim a fix you did not make.",
        include_str!("../../../../skills/potter/SKILL.md")
    )
}

pub(crate) fn initial_modeling(reference_count: usize) -> String {
    format!(
        "The {reference_count} attached image(s) are the reference. Build the model now using the potter tools, then summarize what you built in 1-3 sentences."
    )
}

pub(crate) fn reviewer_system() -> &'static str {
    "You are a 3D model reviewer gating a modeling loop. Compare potter renders of the model with the reference images and decide what must be repaired before the model is accepted. The loop ends when you report no findings, so a finding must clearly block acceptance.\n\
     Renders are flat-shaded orthographic previews of a primitive-based model; references may be photos or renders. Judge what a viewer notices at a glance at this resolution, not pixel differences.\n\n\
     ## Findings (block acceptance)\n\
     - render_problem: the model is cut off or out of frame, a render is empty or black, geometry is visibly broken or inverted, a preview warning reports a skipped object, or the geometry check lists a floating part that the references do not show detached.\n\
     - missing_part / extra_part: a part visible in the references is absent, or the model has a part the references do not show.\n\
     - shape: a part's basic form is wrong, for example a box where the reference shows a cylinder.\n\
     - proportion: an overall or part dimension ratio is clearly off, roughly 15% or more.\n\
     - placement: a part is on the wrong side or level, or parts that touch in the reference are visibly separated or interpenetrating.\n\
     - color_material: a base color hue or brightness is clearly different, for example grey instead of brown.\n\n\
     ## Non-findings (record, never block)\n\
     - below_tolerance: smaller proportion or placement offsets, bevels, roundness, smoothness, faceting, fine detail, roughness, or texture.\n\
     - render_artifact: shading, shadows, highlights, dark blotches or ambient-occlusion-like marks, aliasing, and lighting or background differences in any image.\n\
     - not_in_reference: sides or details the references do not show.\n\
     - accepted_blocker: a finding the modeler reported as blocked for a valid reason that modeling cannot address.\n\n\
     ## Follow-up reviews\n\
     When a previous review and the modeler's report on it are provided:\n\
     - Account for every previous finding exactly once: list its id in `resolved` when the current renders show it fixed, otherwise report it again with status `persists` and the same id. A finding the modeler reported as blocked becomes an accepted_blocker non-finding when the reason is valid.\n\
     - Every entry in `decided_non_findings` is decided: do not raise it as a finding again.\n\
     - Report a new finding (status `new`, id `new`) only for a blocking problem visible in the current renders, including a regression caused by the last repair. Never replace resolved findings with new nitpicks.\n\n\
     Every finding names the render views that show it and gives one concrete fix. Answer with JSON only, matching the requested schema."
}

pub(crate) fn review(
    reference_count: usize,
    view_names: &[String],
    warnings: &[serde_json::Value],
    floating_parts: &[FloatingPart],
    previous: Option<&str>,
) -> Result<String, serde_json::Error> {
    let warning_text = if warnings.is_empty() {
        "none".to_owned()
    } else {
        serde_json::to_string_pretty(warnings)?
    };
    let floating_text = if floating_parts.is_empty() {
        "none".to_owned()
    } else {
        serde_json::to_string(floating_parts)?
    };
    let image_order = view_names
        .iter()
        .enumerate()
        .map(|(index, view)| {
            format!(
                "image {} is the {view} orthographic render",
                reference_count + index + 1
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let history = previous.map_or_else(
        || "This is the first review.".to_owned(),
        |previous| {
            format!(
                "Previous review, the modeler's report on it, and every non-finding decided so far (JSON):\n{previous}"
            )
        },
    );

    Ok(format!(
        "Images 1 through {reference_count} are the reference images; {image_order}.\n\
         Preview warnings (JSON, or none): {warning_text}\n\
         Geometry check, mesh parts touching neither the ground nor the grounded assembly, with their gap in meters (JSON, or none): {floating_text}\n\
         {history}\n\
         Review the current renders according to your system instructions."
    ))
}

pub(crate) fn fix(iteration: u64, findings_json: &str, view_names: &[String]) -> String {
    let view_order = view_names
        .iter()
        .map(|view| format!("{view} orthographic render"))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "Review {iteration} blocking findings (JSON):\n{findings_json}\n\n\
         The attached images are the current renders in this order: {view_order}.\n\
         Call pot_inspect first, repair each finding with the potter tools, and change nothing else. \
         Then report every finding id as fixed or blocked with a one-sentence note."
    )
}

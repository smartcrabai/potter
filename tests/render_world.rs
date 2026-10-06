use std::{error::Error, fs, io::Cursor, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn assert_cli_success(output: &Output) -> Result<Value, Box<dyn Error>> {
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "CLI exited with {:?}: {}{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    json(output)
}

fn init_scene(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert_cli_success(&output)?;
    Ok(())
}

fn apply_operations(scene: &Path, operations: &[Value]) -> Result<Value, Box<dyn Error>> {
    let directory = scene
        .parent()
        .ok_or_else(|| std::io::Error::other("scene directory has no parent"))?;
    let batch_path = directory.join("operations.json");
    fs::write(
        &batch_path,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch_path)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)
}

fn decode_rgba(image: &[u8]) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let mut reader = png::Decoder::new(Cursor::new(image)).read_info()?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels)?;
    if info.color_type != png::ColorType::Rgba {
        return Err("renderer PNG is not RGBA".into());
    }
    pixels.truncate(info.buffer_size());
    Ok((info.width, info.height, pixels))
}

fn render_png(
    scene: &Path,
    output_directory: &Path,
    engine: Option<&str>,
) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let mut command = pot();
    command.arg("render").arg(scene);
    if let Some(engine) = engine {
        command.args(["--engine", engine]);
    }
    let output = command
        .args(["--format", "png", "--out"])
        .arg(output_directory)
        .arg("--json")
        .output()?;
    assert_cli_success(&output)?;
    let image = fs::read(output_directory.join("frame_0001.png"))?;
    decode_rgba(&image)
}

fn render_png_at_frame(
    scene: &Path,
    output_directory: &Path,
    engine: &str,
    frame_range: &str,
) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let output = pot()
        .arg("render")
        .arg(scene)
        .args([
            "--engine",
            engine,
            "--frames",
            frame_range,
            "--format",
            "png",
            "--out",
        ])
        .arg(output_directory)
        .arg("--json")
        .output()?;
    let result = assert_cli_success(&output)?;
    let image_path = result["result"]["frames"][0]["path"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("rendered frame path is missing"))?;
    decode_rgba(&fs::read(image_path)?)
}

fn pixel(
    pixels: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Result<[u8; 4], Box<dyn Error>> {
    if x >= width || y >= height {
        return Err("pixel coordinate is outside the rendered image".into());
    }
    let offset = usize::try_from(y)?
        .checked_mul(usize::try_from(width)?)
        .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
        .and_then(|index| index.checked_mul(4))
        .ok_or_else(|| std::io::Error::other("pixel offset exceeds platform limits"))?;
    let channels = pixels
        .get(offset..offset + 4)
        .ok_or_else(|| std::io::Error::other("rendered pixel is missing"))?;
    Ok([channels[0], channels[1], channels[2], channels[3]])
}

fn luminance(pixel: [u8; 4]) -> u16 {
    u16::from(pixel[0]) + u16::from(pixel[1]) + u16::from(pixel[2])
}

fn append_world_texture_graph(
    operations: &mut Vec<Value>,
    graph_id: &str,
    texture_type: &str,
    texture_inputs: &Value,
    texture_properties: &Value,
) {
    operations.extend([
        json!({"op":"graph.create","id":graph_id,"kind":"shader"}),
        json!({
            "op":"graph.node_add","graph":graph_id,"id":"world_texture",
            "type":texture_type,"inputs":texture_inputs,"properties":texture_properties
        }),
        json!({
            "op":"graph.node_add","graph":graph_id,"id":"background",
            "type":"ShaderNodeBackground","inputs":{"Strength":1.0}
        }),
        json!({
            "op":"graph.node_add","graph":graph_id,"id":"world_output",
            "type":"ShaderNodeOutputWorld","properties":{"is_active_output":true}
        }),
        json!({
            "op":"graph.link","graph":graph_id,"from_node":"world_texture",
            "from_socket":"Color","to_node":"background","to_socket":"Color"
        }),
        json!({
            "op":"graph.link","graph":graph_id,"from_node":"background",
            "from_socket":"Background","to_node":"world_output","to_socket":"Surface"
        }),
        json!({"op":"world.create","id":"world_main","node_tree":graph_id}),
    ]);
}

fn append_camera_and_render(
    operations: &mut Vec<Value>,
    transform: &Value,
    camera_settings: &Value,
    render_settings: &Value,
    world: Option<&str>,
) {
    let mut camera = json!({
        "op":"camera.create","id":"camera_main","transform":transform
    });
    if let (Some(camera_object), Some(settings)) =
        (camera.as_object_mut(), camera_settings.as_object())
    {
        camera_object.extend(
            settings
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    operations.push(camera);

    let mut scene_settings = json!({"camera":"camera_main"});
    if let (Some(settings), Some(world)) = (scene_settings.as_object_mut(), world) {
        settings.insert("world".to_owned(), json!(world));
    }
    operations.push(json!({"op":"scene.update","target":{"id":"scene_main"},"set":scene_settings}));
    operations
        .push(json!({"op":"render.update","target":{"id":"scene_main"},"set":render_settings}));
}

fn make_scene(scene: &Path, operations: &[Value]) -> Result<Value, Box<dyn Error>> {
    init_scene(scene)?;
    apply_operations(scene, operations)
}

#[test]
fn analytic_sky_brightness_changes_with_sun_elevation() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let mut mean_brightness = Vec::new();
    for (name, sun_direction) in [
        ("low_sun", json!([1.0, 0.0, 0.0])),
        (
            "high_sun",
            json!([
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2
            ]),
        ),
    ] {
        let scene = directory.path().join(name);
        let mut operations = Vec::new();
        append_world_texture_graph(
            &mut operations,
            "sky_graph",
            "ShaderNodeTexSky",
            &json!({"Sun Direction":sun_direction}),
            &json!({}),
        );
        append_camera_and_render(
            &mut operations,
            &json!({"translation":[0.0,0.0,0.0],"rotation":[1.0,0.0,0.0,0.0]}),
            &json!({"projection":"perspective","lens_mm":50.0}),
            &json!({"resolution_x":24,"resolution_y":24,"samples":1,"seed":0,"max_bounces":0}),
            Some("world_main"),
        );
        make_scene(&scene, &operations)?;
        let (width, height, pixels) = render_png(
            &scene,
            &directory.path().join(format!("{name}_render")),
            Some("path"),
        )?;
        mean_brightness.push(luminance(pixel(
            &pixels,
            width,
            height,
            width / 2,
            height / 2,
        )?));
    }

    let low = *mean_brightness.first().ok_or("low-sun render is missing")?;
    let high = *mean_brightness.get(1).ok_or("high-sun render is missing")?;
    assert!(
        high > low + 100,
        "raising the sun should brighten the sky: {low} vs {high}"
    );
    Ok(())
}

#[test]
fn constant_white_environment_lights_a_diffuse_sphere_without_channel_tint()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = vec![
        json!({"op":"image.create","id":"environment","width":2,"height":1,"colorspace":"linear","fill_color":[1.0,1.0,1.0,1.0]}),
        json!({"op":"material.create","id":"diffuse_white","base_color":[1.0,1.0,1.0,1.0],"roughness":1.0}),
        json!({"op":"node.create","id":"sphere","kind":"sphere","params":{"radius":0.8,"segments":24,"ring_count":12},"material":"diffuse_white"}),
    ];
    append_world_texture_graph(
        &mut operations,
        "environment_graph",
        "ShaderNodeTexEnvironment",
        &json!({}),
        &json!({"image":"environment","interpolation":"closest"}),
    );
    append_camera_and_render(
        &mut operations,
        &json!({"translation":[0.0,0.0,4.0]}),
        &json!({"projection":"orthographic","ortho_scale":2.5}),
        &json!({"resolution_x":32,"resolution_y":32,"samples":32,"seed":11,"max_bounces":2}),
        Some("world_main"),
    );
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("path"))?;
    let center = pixel(&pixels, width, height, width / 2, height / 2)?;
    let darkest = *center[..3].iter().min().ok_or("RGB channels are missing")?;
    let brightest = *center[..3].iter().max().ok_or("RGB channels are missing")?;
    assert!(
        darkest >= 245,
        "white environment should light the diffuse sphere: {center:?}"
    );
    assert!(
        u16::from(brightest - darkest) <= 2,
        "a neutral environment should not tint the sphere by more than 1%: {center:?}"
    );
    let (realtime_width, realtime_height, realtime_pixels) =
        render_png(&scene, &directory.path().join("realtime"), Some("realtime"))?;
    let realtime_center = pixel(
        &realtime_pixels,
        realtime_width,
        realtime_height,
        realtime_width / 2,
        realtime_height / 2,
    )?;
    let realtime_range = u16::from(
        *realtime_center[..3]
            .iter()
            .max()
            .ok_or("RGB channels are missing")?,
    ) - u16::from(
        *realtime_center[..3]
            .iter()
            .min()
            .ok_or("RGB channels are missing")?,
    );
    assert!(
        realtime_range <= 2 && realtime_center[0] >= 245,
        "realtime environment lighting should preserve a neutral white sky: {realtime_center:?}"
    );
    Ok(())
}

#[test]
fn homogeneous_volume_cube_transmittance_matches_beer_lambert() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let density = 0.7_f64;
    let scene = directory.path().join("scene");
    let mut operations = vec![json!({
    "op":"volume.create","id":"fog","grids":[{
        "dims":[2,2,2],"voxel_size":1.0,"origin":[-0.5,-0.5,-0.5],
        "values":[density,density,density,density,density,density,density,density]
    }]})];
    append_camera_and_render(
        &mut operations,
        &json!({"translation":[0.0,0.0,3.0]}),
        &json!({"projection":"orthographic","ortho_scale":2.0}),
        &json!({"resolution_x":24,"resolution_y":24,"samples":1,"seed":0,"max_bounces":0,"film_transparent":true}),
        None,
    );
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("path"))?;
    let alpha = f64::from(pixel(&pixels, width, height, width / 2, height / 2)?[3]) / 255.0;
    let measured_transmittance = 1.0 - alpha;
    let expected_transmittance = (-density).exp();
    assert!(
        (measured_transmittance - expected_transmittance).abs() < 0.01,
        "one unit of homogeneous density {density} should transmit e^(-density): {measured_transmittance} vs {expected_transmittance}"
    );
    Ok(())
}

#[test]
fn material_volume_output_transmits_according_to_beer_lambert() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let density = 0.7_f64;
    let operations = vec![
        json!({"op":"graph.create","id":"volume_graph","kind":"shader"}),
        json!({"op":"graph.node_add","graph":"volume_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
        json!({"op":"graph.node_add","graph":"volume_graph","id":"volume","type":"ShaderNodeVolumePrincipled","inputs":{"Density":density,"Color":[0.0,0.0,0.0,1.0],"Anisotropy":0.0}}),
        json!({"op":"graph.link","graph":"volume_graph","from_node":"volume","from_socket":"Volume","to_node":"output","to_socket":"Volume"}),
        json!({"op":"material.create","id":"fog","graph":"volume_graph"}),
        json!({"op":"node.create","id":"fog_cube","kind":"box","params":{"size":1.0},"material":"fog"}),
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,3.0]},"projection":"orthographic","ortho_scale":2.0}),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":24,"resolution_y":24,"samples":1,"seed":0,"max_bounces":2}}),
    ];
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("path"))?;
    let center = pixel(&pixels, width, height, width / 2, height / 2)?;
    let encoded = f64::from(center[0]) / 255.0;
    let measured_transmittance = if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    };
    let expected_transmittance = (-density).exp();
    assert!(
        (measured_transmittance - expected_transmittance).abs() < 0.02,
        "material volume should transmit e^(-density*length): {measured_transmittance} vs {expected_transmittance}"
    );
    Ok(())
}

#[test]
fn true_displacement_subdivides_and_moves_the_rendered_silhouette() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let mut foreground_counts = Vec::new();
    for method in ["bump", "displacement"] {
        let scene = directory.path().join(method);
        let operations = vec![
            json!({"op":"graph.create","id":"displacement_graph","kind":"shader"}),
            json!({"op":"graph.node_add","graph":"displacement_graph","id":"output","type":"ShaderNodeOutputMaterial"}),
            json!({"op":"graph.node_add","graph":"displacement_graph","id":"displacement","type":"ShaderNodeDisplacement","inputs":{"Height":0.5,"Midlevel":0.0,"Scale":1.0}}),
            json!({"op":"graph.link","graph":"displacement_graph","from_node":"displacement","from_socket":"Displacement","to_node":"output","to_socket":"Displacement"}),
            json!({"op":"material.create","id":"clay","graph":"displacement_graph","displacement_method":method,"base_color":[1.0,1.0,1.0,1.0],"emission_color":[1.0,1.0,1.0],"emission_strength":2.0}),
            json!({"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"material":"clay"}),
            json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,-4.0,4.0],"rotation":[std::f64::consts::FRAC_PI_8.sin(),0.0,0.0,std::f64::consts::FRAC_PI_8.cos()]},"projection":"orthographic","ortho_scale":4.0}),
            json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}}),
            json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":64,"samples":1,"seed":0,"max_bounces":0}}),
        ];
        make_scene(&scene, &operations)?;
        let (_, _, pixels) = render_png(
            &scene,
            &directory.path().join(format!("{method}_render")),
            Some("path"),
        )?;
        foreground_counts.push(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|pixel| pixel[0] > 150)
                .count(),
        );
    }
    let bump = *foreground_counts.first().ok_or("bump render is missing")?;
    let displaced = *foreground_counts
        .get(1)
        .ok_or("displacement render is missing")?;
    assert_ne!(
        displaced, bump,
        "true displacement should alter the subdivided render silhouette"
    );
    Ok(())
}
#[test]
fn realtime_occluder_darkens_the_surface_behind_it() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let mut images = Vec::new();
    for blocked in [false, true] {
        let name = if blocked { "blocked" } else { "unblocked" };
        let scene = directory.path().join(name);
        let mut operations = vec![
            json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
            json!({"op":"material.create","id":"white","base_color":[0.9,0.9,0.9,1.0],"roughness":0.9}),
            json!({"op":"node.create","id":"floor","kind":"box","params":{"size":4.0},"transform":{"translation":[0.0,0.0,-0.05],"scale":[1.0,1.0,0.025]},"material":"white"}),
        ];
        if blocked {
            operations.push(json!({
                "op":"node.create","id":"occluder","kind":"box","params":{"size":0.3},
                "transform":{"translation":[0.25,0.0,1.0]},"material":"white"
            }));
        }
        operations.extend([
            json!({
                "op":"camera.create","id":"camera_main",
                "transform":{"translation":[0.0,-4.0,4.0],"rotation":[std::f64::consts::FRAC_PI_8.sin(),0.0,0.0,std::f64::consts::FRAC_PI_8.cos()]},
                "projection":"orthographic","ortho_scale":5.0
            }),
            json!({"op":"light.create","id":"key","light_type":"point","energy":20.0,"radius":0.05,"transform":{"translation":[0.5,0.0,2.0]}}),
            json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
            json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1,"seed":0,"max_bounces":0}}),
        ]);
        make_scene(&scene, &operations)?;
        images.push(render_png(
            &scene,
            &directory.path().join(format!("{name}_render")),
            Some("realtime"),
        )?);
    }

    let (unblocked_width, unblocked_height, unblocked) =
        images.first().ok_or("unblocked render is missing")?;
    let (blocked_width, blocked_height, blocked) =
        images.get(1).ok_or("blocked render is missing")?;
    assert_eq!(unblocked_width, blocked_width);
    assert_eq!(unblocked_height, blocked_height);
    let mut darkened_pixels = 0_usize;
    for index in (0..unblocked.len()).step_by(4) {
        let Some(unblocked_pixel) = unblocked.get(index..index + 3) else {
            continue;
        };
        let Some(blocked_pixel) = blocked.get(index..index + 3) else {
            continue;
        };
        let unblocked_luminance = unblocked_pixel
            .iter()
            .map(|channel| u16::from(*channel))
            .sum::<u16>();
        let blocked_luminance = blocked_pixel
            .iter()
            .map(|channel| u16::from(*channel))
            .sum::<u16>();
        if blocked_luminance + 20 < unblocked_luminance {
            darkened_pixels += 1;
        }
    }
    assert!(
        darkened_pixels > 0,
        "a realtime occluder should darken floor pixels behind it"
    );
    Ok(())
}

#[test]
fn equirectangular_panorama_pixels_follow_their_camera_ray_directions() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let longitude_colors = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut operations = vec![
        json!({"op":"image.create","id":"panorama","width":8,"height":1,"colorspace":"linear","fill_color":[0.0,0.0,0.0,1.0]}),
        json!({"op":"image.set_pixels","id":"panorama","width":8,"height":1,"pixels":longitude_colors}),
    ];
    append_world_texture_graph(
        &mut operations,
        "panorama_graph",
        "ShaderNodeTexEnvironment",
        &json!({}),
        &json!({"image":"panorama","interpolation":"closest"}),
    );
    append_camera_and_render(
        &mut operations,
        &json!({"translation":[0.0,0.0,0.0]}),
        &json!({"projection":"panorama","panorama_type":"equirectangular"}),
        &json!({"resolution_x":64,"resolution_y":32,"samples":1,"seed":0,"max_bounces":0}),
        Some("world_main"),
    );
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("path"))?;
    let toward_positive_x = pixel(&pixels, width, height, 48, 12)?;
    let toward_negative_x = pixel(&pixels, width, height, 16, 12)?;
    assert!(
        toward_positive_x[1] > 240 && toward_positive_x[2] > 240 && toward_positive_x[0] < 15,
        "the positive-X panorama ray should sample the equirectangular u=0.5 texel: {toward_positive_x:?}"
    );
    assert!(
        toward_negative_x[0] < 15 && toward_negative_x[1] < 15 && toward_negative_x[2] < 15,
        "the negative-X panorama ray should sample the opposite longitude texel: {toward_negative_x:?}"
    );
    Ok(())
}

fn checker_contrast(
    pixels: &[u8],
    width: u32,
    height: u32,
    patch_center_x: f64,
) -> Result<u16, Box<dyn Error>> {
    let mut white_total = 0_u32;
    let mut black_total = 0_u32;
    let mut white_count = 0_u32;
    let mut black_count = 0_u32;
    for row in 0..8 {
        for column in 0..8 {
            let world_x = patch_center_x + (f64::from(column) - 3.5) * 0.2;
            let world_y = (3.5 - f64::from(row)) * 0.2;
            let x = ((world_x + 2.0) * f64::from(width) / 4.0).floor() as u32;
            let y = ((2.0 - world_y) * f64::from(height) / 4.0).floor() as u32;
            let value = u32::from(luminance(pixel(pixels, width, height, x, y)?));
            if (row + column) % 2 == 0 {
                white_total += value;
                white_count += 1;
            } else {
                black_total += value;
                black_count += 1;
            }
        }
    }
    let white_mean = white_total / white_count;
    let black_mean = black_total / black_count;
    Ok(u16::try_from(white_mean.saturating_sub(black_mean))?)
}

#[test]
fn depth_of_field_blurs_the_out_of_focus_checker_but_keeps_focus_sharp()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = vec![
        json!({"op":"material.create","id":"black","base_color":[0.0,0.0,0.0,1.0],"emission_color":[0.0,0.0,0.0],"emission_strength":1.0}),
        json!({"op":"material.create","id":"white","base_color":[1.0,1.0,1.0,1.0],"emission_color":[1.0,1.0,1.0],"emission_strength":1.0}),
    ];
    for (layer, (patch_center_x, z)) in [(-0.9, 0.0), (0.9, -2.0)].into_iter().enumerate() {
        for row in 0..8 {
            for column in 0..8 {
                let material = if (row + column) % 2 == 0 {
                    "white"
                } else {
                    "black"
                };
                operations.push(json!({
                    "op":"node.create","id":format!("tile_{layer}_{row}_{column}"),"kind":"box",
                    "params":{"size":0.2},
                    "transform":{"translation":[patch_center_x+(f64::from(column)-3.5)*0.2,(3.5-f64::from(row))*0.2,z],"scale":[1.0,1.0,0.1]},
                    "material":material
                }));
            }
        }
    }
    operations.extend([
        json!({
            "op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,4.0]},
            "projection":"orthographic","ortho_scale":4.0,"dof_enabled":true,
            "focus_distance":4.0,"lens_mm":50.0,"f_stop":0.1
        }),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":96,"resolution_y":96,"samples":32,"seed":19,"max_bounces":0}}),
    ]);
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("path"))?;
    let in_focus_contrast = checker_contrast(&pixels, width, height, -0.9)?;
    let out_of_focus_contrast = checker_contrast(&pixels, width, height, 0.9)?;
    assert!(
        in_focus_contrast > 300,
        "checker cells at the focus distance should remain sharp: contrast {in_focus_contrast}"
    );
    assert!(
        out_of_focus_contrast < in_focus_contrast * 4 / 5,
        "the farther checker should have visibly lower contrast: {out_of_focus_contrast} vs {in_focus_contrast}"
    );
    Ok(())
}

#[test]
fn ambient_occlusion_darkens_a_corner_relative_to_open_surface() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let operations = vec![
        json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
        json!({"op":"material.create","id":"white","base_color":[1.0,1.0,1.0,1.0],"roughness":1.0}),
        json!({"op":"node.create","id":"floor","kind":"box","params":{"size":4.0},"transform":{"translation":[0.0,0.0,-0.05],"scale":[1.0,1.0,0.025]},"material":"white"}),
        json!({"op":"node.create","id":"wall","kind":"box","params":{"size":4.0},"transform":{"translation":[-0.05,0.0,0.75],"scale":[0.025,1.0,0.375]},"material":"white"}),
        json!({
            "op":"camera.create","id":"camera_main",
            "transform":{"translation":[0.0,-4.0,4.0],"rotation":[std::f64::consts::FRAC_PI_8.sin(),0.0,0.0,std::f64::consts::FRAC_PI_8.cos()]},
            "projection":"orthographic","ortho_scale":4.0
        }),
        json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
        json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":64,"samples":1,"seed":0,"max_bounces":0}}),
    ];
    make_scene(&scene, &operations)?;

    let (width, height, pixels) =
        render_png(&scene, &directory.path().join("render"), Some("realtime"))?;
    let corner = luminance(pixel(&pixels, width, height, 34, height / 2)?);
    let open_surface = luminance(pixel(&pixels, width, height, 53, height / 2)?);
    assert!(
        open_surface > 0,
        "the open surface should receive world illumination"
    );
    assert!(
        corner < open_surface,
        "ambient occlusion should darken the floor near the wall corner: {corner} vs {open_surface}"
    );
    Ok(())
}

#[test]
fn motion_blur_samples_animated_transforms_inside_the_shutter() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let mut visible_pixel_counts = Vec::new();
    for motion_blur in [false, true] {
        let name = if motion_blur { "blurred" } else { "still" };
        let scene = directory.path().join(name);
        let operations = vec![
            json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
            json!({"op":"material.create","id":"glow","base_color":[1.0,1.0,1.0,1.0],"emission_color":[1.0,1.0,1.0],"emission_strength":2.0}),
            json!({"op":"node.create","id":"moving","kind":"box","params":{"size":0.25},"material":"glow"}),
            json!({"op":"action.create","id":"move_action","name":"Move"}),
            json!({"op":"node.update","target":{"id":"moving"},"set":{"action":"move_action"}}),
            json!({"op":"keyframe.insert","target":{"id":"moving"},"path":"transform.translation","index":0,"frame":1.0,"value":-1.0,"interpolation":"linear"}),
            json!({"op":"keyframe.insert","target":{"id":"moving"},"path":"transform.translation","index":0,"frame":3.0,"value":1.0,"interpolation":"linear"}),
            json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,4.0]},"projection":"orthographic","ortho_scale":3.0}),
            json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
            json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":32,"samples":1,"seed":0,"max_bounces":0,"motion_blur":motion_blur,"shutter":1.0,"motion_blur_samples":7}}),
        ];
        make_scene(&scene, &operations)?;
        let (width, _, pixels) = render_png_at_frame(
            &scene,
            &directory.path().join(format!("{name}_render")),
            "path",
            "2:2",
        )?;
        let count = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|pixel| pixel[0] > 10)
            .count();
        visible_pixel_counts.push((width, count));
    }
    let (width, still_count) = *visible_pixel_counts
        .first()
        .ok_or("still render is missing")?;
    let (blurred_width, blurred_count) = *visible_pixel_counts
        .get(1)
        .ok_or("blurred render is missing")?;
    assert_eq!(width, blurred_width);
    assert!(
        blurred_count > still_count * 2,
        "subframe motion should increase the animated object's exposure footprint: {still_count} vs {blurred_count}"
    );
    Ok(())
}

#[test]
fn stereo_side_by_side_and_anaglyph_separate_left_and_right_views() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    for stereo_mode in ["side_by_side", "anaglyph"] {
        let scene = directory.path().join(stereo_mode);
        let operations = vec![
            json!({"op":"world.create","id":"world_main","color":[0.0,0.0,0.0],"strength":0.0}),
            json!({"op":"material.create","id":"left_red","base_color":[1.0,0.0,0.0,1.0],"emission_color":[1.0,0.0,0.0],"emission_strength":2.0}),
            json!({"op":"material.create","id":"right_blue","base_color":[0.0,0.0,1.0,1.0],"emission_color":[0.0,0.0,1.0],"emission_strength":2.0}),
            json!({"op":"node.create","id":"left_object","kind":"box","params":{"size":0.35},"transform":{"translation":[-0.5,0.0,0.0],"scale":[1.0,1.0,0.285_714_285_714_285_7]},"material":"left_red"}),
            json!({"op":"node.create","id":"right_object","kind":"box","params":{"size":0.35},"transform":{"translation":[0.5,0.0,0.0],"scale":[1.0,1.0,0.285_714_285_714_285_7]},"material":"right_blue"}),
            json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,4.0]},"projection":"orthographic","ortho_scale":2.0,"stereo_mode":stereo_mode,"interocular_distance":1.0}),
            json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
            json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1,"seed":0,"max_bounces":0}}),
        ];
        make_scene(&scene, &operations)?;
        let (width, height, pixels) = render_png(
            &scene,
            &directory.path().join(format!("{stereo_mode}_render")),
            Some("path"),
        )?;
        if stereo_mode == "side_by_side" {
            let left = pixel(&pixels, width, height, width / 4, height / 2)?;
            let right = pixel(&pixels, width, height, 3 * width / 4, height / 2)?;
            assert!(
                left[0] > left[2],
                "left stereo image should contain the red object: {left:?}"
            );
            assert!(
                right[2] > right[0],
                "right stereo image should contain the blue object: {right:?}"
            );
        } else {
            let center = pixel(&pixels, width, height, width / 2, height / 2)?;
            assert!(
                center[0] > 200 && center[2] > 200,
                "anaglyph channels should combine the two eyes: {center:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn realtime_blends_alpha_and_respects_alpha_clip_thresholds() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    for alpha_mode in ["blend", "clip"] {
        let scene = directory.path().join(alpha_mode);
        let operations = vec![
            json!({"op":"world.create","id":"world_main","color":[1.0,1.0,1.0],"strength":1.0}),
            json!({"op":"material.create","id":"red","base_color":[1.0,0.0,0.0,1.0],"roughness":1.0}),
            json!({"op":"material.create","id":"blue","base_color":[0.0,0.0,1.0,0.4],"roughness":1.0,"alpha_mode":alpha_mode,"alpha_threshold":0.6}),
            json!({"op":"node.create","id":"back","kind":"box","params":{"size":1.0},"transform":{"scale":[1.0,1.0,0.1]},"material":"red"}),
            json!({"op":"node.create","id":"front","kind":"box","params":{"size":0.8},"transform":{"translation":[0.0,0.0,1.0],"scale":[1.0,1.0,0.125]},"material":"blue"}),
            json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,4.0]},"projection":"orthographic","ortho_scale":2.0}),
            json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}}),
            json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1,"seed":0}}),
        ];
        make_scene(&scene, &operations)?;
        let (width, height, pixels) = render_png(
            &scene,
            &directory.path().join(format!("{alpha_mode}_render")),
            Some("realtime"),
        )?;
        let center = pixel(&pixels, width, height, width / 2, height / 2)?;
        if alpha_mode == "blend" {
            assert!(
                center[0] > 100 && center[2] > 100,
                "semi-transparent blue should blend over the red surface: {center:?}"
            );
        } else {
            assert!(
                center[0] > center[2] * 2,
                "sub-threshold clip alpha should reveal the red background surface: {center:?}"
            );
        }
    }
    Ok(())
}

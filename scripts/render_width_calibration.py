import blenderproc as bproc

"""Calibrate reflection apparent width against the frozen C3 crack target."""

import csv
import json
import math
from pathlib import Path
import sys

import bpy
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_ambiguity_calibration import (
    BASE_COLOR,
    BOLT_PATH,
    CAM_LOCATION,
    CAM_LOOK_AT,
    CRACK_RENDER_DIR,
    DEFAULT_KEY_POS,
    FEATURE_ANGLE_DEG,
    FEATURE_CENTER,
    FEATURE_LENGTH,
    FILL_COLOR,
    FILL_ENERGY,
    FILL_POS,
    FILL_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    KEY_COLOR,
    KEY_ENERGY,
    KEY_SIZE_X,
    KEY_SIZE_Y,
    OUTPUT_DIR,
    PLATE_PATH,
    SIDE_KEY_POS,
    TARGET_CENTER,
    apply_flat_shading,
    apply_smooth_shading,
    configure_color_management,
    configure_cycles,
    create_reflection_material,
    create_uniform_metal_material,
    import_stl,
    luminance,
    move_key_light,
    orient_area_toward,
    project_point,
    save_frame,
    set_area_dimensions,
)


WIDTH_OUTPUT_DIR = REPO_ROOT = Path(__file__).resolve().parents[1] / "renders" / "width_calibration"
REFLECTION_DIR = WIDTH_OUTPUT_DIR / "reflection"
C3_DEFAULT = CRACK_RENDER_DIR / "C3_default.png"
C3_SIDE = CRACK_RENDER_DIR / "C3_side.png"
METRICS_JSON = WIDTH_OUTPUT_DIR / "width_metrics.json"
METRICS_CSV = WIDTH_OUTPUT_DIR / "width_metrics.csv"
CONTACT_SHEET = WIDTH_OUTPUT_DIR / "contact_sheet.png"

REFLECTIONS = {
    "RW1": {"width_mm": 0.6, "polished_roughness": 0.08, "metallic": 1.0},
    "RW2": {"width_mm": 0.9, "polished_roughness": 0.08, "metallic": 1.0},
    "RW3": {"width_mm": 1.2, "polished_roughness": 0.08, "metallic": 1.0},
    "RW4": {"width_mm": 1.5, "polished_roughness": 0.08, "metallic": 1.0},
}

FIXED_THRESHOLD_LEVELS = 10.0
MAX_CROSS_SECTION_RADIUS_PX = 12.0
CROSS_SECTION_STEP_PX = 0.25
T_START = 0.30
T_END = 0.86
T_SAMPLES = 48


def projected_feature(scene, plate_obj):
    top_z = max(float(v[2]) for v in plate_obj.bound_box) + 0.02
    angle = math.radians(FEATURE_ANGLE_DEG)
    along = np.array([math.cos(angle), math.sin(angle)])
    center = np.array(FEATURE_CENTER)
    p0 = project_point(scene, plate_obj, *(center - along * FEATURE_LENGTH / 2.0), top_z)
    p1 = project_point(scene, plate_obj, *(center + along * FEATURE_LENGTH / 2.0), top_z)
    return p0, p1


def bilinear_sample(image, xs, ys):
    """Sample luminance at subpixel coordinates without changing the image."""
    h, w = image.shape
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = xs - x0
    wy = ys - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def apparent_width_measurement(image, p0, p1):
    """Measure contiguous contrast width around the projected centreline.

    For 48 centreline locations over t=0.30..0.86, sample a perpendicular
    cross-section from -12 to +12 px at 0.25 px spacing.  The local plate
    reference is the mean of the two +/-10 px samples.  A contiguous response
    is counted when absolute luminance difference is at least 10/255, starting
    at the centreline.  Bolt and endpoints are excluded by the t interval.
    """
    normalized_luminance = luminance(image[..., :3].astype(np.float32) / 255.0)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    along = delta / length
    normal = np.array([-along[1], along[0]])
    distances = np.arange(-MAX_CROSS_SECTION_RADIUS_PX, MAX_CROSS_SECTION_RADIUS_PX + CROSS_SECTION_STEP_PX / 2.0, CROSS_SECTION_STEP_PX)
    center_index = int(np.argmin(np.abs(distances)))
    widths = []
    contrasts = []
    signed_contrasts = []
    bright_fractions = []
    dark_fractions = []

    for t in np.linspace(T_START, T_END, T_SAMPLES):
        center = p0 + delta * t
        section_points = center[None, :] + distances[:, None] * normal[None, :]
        values = bilinear_sample(normalized_luminance, section_points[:, 0], section_points[:, 1])
        ref_points = np.vstack((center + normal * 10.0, center - normal * 10.0))
        reference = float(bilinear_sample(normalized_luminance, ref_points[:, 0], ref_points[:, 1]).mean())
        signed = values - reference
        active = np.abs(signed) >= FIXED_THRESHOLD_LEVELS / 255.0
        if not active[center_index]:
            continue

        left = center_index
        while left > 0 and active[left - 1]:
            left -= 1
        right = center_index
        while right < len(active) - 1 and active[right + 1]:
            right += 1
        width = float(distances[right] - distances[left])
        widths.append(width)
        center_contrast = float(abs(signed[center_index]))
        contrasts.append(center_contrast)
        signed_contrasts.append(float(signed[center_index]))
        bright_fractions.append(float((signed[left : right + 1] >= FIXED_THRESHOLD_LEVELS / 255.0).mean()))
        dark_fractions.append(float((signed[left : right + 1] <= -FIXED_THRESHOLD_LEVELS / 255.0).mean()))

    if not widths:
        return {
            "median_width_px": 0.0,
            "mean_width_px": 0.0,
            "width_std_px": 0.0,
            "width_p10_px": 0.0,
            "width_p90_px": 0.0,
            "number_of_valid_cross_sections": 0,
            "default_local_contrast_levels_mean": 0.0,
            "default_local_contrast_levels_median": 0.0,
            "signed_contrast_levels_mean": 0.0,
            "bright_fraction": 0.0,
            "dark_fraction": 0.0,
        }
    return {
        "median_width_px": float(np.median(widths)),
        "mean_width_px": float(np.mean(widths)),
        "width_std_px": float(np.std(widths)),
        "width_p10_px": float(np.percentile(widths, 10)),
        "width_p90_px": float(np.percentile(widths, 90)),
        "number_of_valid_cross_sections": len(widths),
        "default_local_contrast_levels_mean": float(np.mean(contrasts) * 255.0),
        "default_local_contrast_levels_median": float(np.median(contrasts) * 255.0),
        "signed_contrast_levels_mean": float(np.mean(signed_contrasts) * 255.0),
        "bright_fraction": float(np.mean(bright_fractions)),
        "dark_fraction": float(np.mean(dark_fractions)),
    }


def matched_response_change(default_image, side_image, p0, p1):
    """Return signed centreline response change in intensity levels."""
    def sample_centerline(image):
        lum = luminance(image[..., :3].astype(np.float32) / 255.0)
        delta = p1 - p0
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        values = []
        for t in np.linspace(T_START, T_END, T_SAMPLES):
            center = p0 + delta * t
            refs = np.vstack((center + normal * 10.0, center - normal * 10.0))
            value = bilinear_sample(lum, np.array([center[0]]), np.array([center[1]]))[0]
            reference = bilinear_sample(lum, refs[:, 0], refs[:, 1]).mean()
            values.append((value - reference) * 255.0)
        return np.asarray(values)

    default_response = sample_centerline(default_image)
    side_response = sample_centerline(side_image)
    return {
        "side_response_change_levels_mean_abs": float(np.mean(np.abs(side_response - default_response))),
        "default_signed_response_levels_mean": float(default_response.mean()),
        "side_signed_response_levels_mean": float(side_response.mean()),
        "side_minus_default_signed_response_levels_mean": float((side_response - default_response).mean()),
    }


def render_reflections(scene, key_light, plate_obj, bolt_obj, materials):
    REFLECTION_DIR.mkdir(parents=True, exist_ok=True)
    plate_obj.hide_render = False
    bolt_obj.hide_render = False
    for name, material in materials.items():
        plate_obj.data.materials.clear()
        plate_obj.data.materials.append(material)
        move_key_light(key_light, DEFAULT_KEY_POS)
        save_frame(REFLECTION_DIR / f"{name}_default.png")
        move_key_light(key_light, SIDE_KEY_POS)
        save_frame(REFLECTION_DIR / f"{name}_side.png")


def make_contact_sheet():
    tile_w, tile_h = IMAGE_WIDTH, IMAGE_HEIGHT
    margin_x, margin_y = 20, 20
    label_w, label_h = 180, 34
    image_x = margin_x + label_w
    sheet = Image.new("RGB", (image_x + tile_w * 2 + margin_x, margin_y * 2 + label_h * 6 + tile_h * 5), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        small = font
    y = margin_y
    draw.text((image_x + tile_w // 2 - 65, y), "DEFAULT", fill=(0, 0, 0), font=font)
    draw.text((image_x + tile_w + tile_w // 2 - 35, y), "SIDE", fill=(0, 0, 0), font=font)
    y += label_h
    rows = [(name, REFLECTION_DIR, name) for name in REFLECTIONS]
    rows.append(("C3 target", CRACK_RENDER_DIR, "C3"))
    for label, directory, name in rows:
        draw.text((margin_x, y + tile_h // 2 - 12), label, fill=(0, 0, 0), font=small)
        for column, condition in enumerate(("default", "side")):
            image_path = directory / f"{name}_{condition}.png"
            sheet.paste(Image.open(image_path).convert("RGB"), (image_x + column * tile_w, y))
        y += tile_h + label_h
    sheet.save(CONTACT_SHEET)


def write_metrics(measurements, p0, p1):
    target_default = measurements["C3"]["default"]
    target_side = measurements["C3"]["side"]
    rows = []
    for name, parameters in REFLECTIONS.items():
        default = measurements[name]["default"]
        side = measurements[name]["side"]
        rows.append({
            "candidate": name,
            "physical_width_mm": parameters["width_mm"],
            "default_apparent_width_px": default["median_width_px"],
            "width_error_px": abs(default["median_width_px"] - target_default["median_width_px"]),
            "default_local_contrast_levels": default["default_local_contrast_levels_mean"],
            "side_light_response_change_levels": side["response_change"]["side_response_change_levels_mean_abs"],
            "default_width_std_px": default["width_std_px"],
            "default_valid_cross_sections": default["number_of_valid_cross_sections"],
            "default_signed_contrast_levels": default["signed_contrast_levels_mean"],
            "default_bright_fraction": default["bright_fraction"],
            "default_dark_fraction": default["dark_fraction"],
        })
    rows.sort(key=lambda row: (row["width_error_px"], row["default_local_contrast_levels"]))
    for rank, row in enumerate(rows, start=1):
        row["width_rank"] = rank

    payload = {
        "measurement_rule": {
            "description": "Contiguous cross-section width around the projected centreline.",
            "t_range": [T_START, T_END],
            "cross_sections": T_SAMPLES,
            "cross_section_radius_px": MAX_CROSS_SECTION_RADIUS_PX,
            "cross_section_step_px": CROSS_SECTION_STEP_PX,
            "local_reference": "mean luminance at +/-10 px perpendicular to the centreline",
            "threshold_intensity_levels": FIXED_THRESHOLD_LEVELS,
            "valid_cross_section": "centreline contrast reaches the threshold",
            "width": "contiguous thresholded run around centreline, measured in image pixels",
        },
        "fixed_feature": {"center_mm": list(FEATURE_CENTER), "length_mm": FEATURE_LENGTH, "angle_degrees": FEATURE_ANGLE_DEG, "projected_endpoint_a_px": p0.tolist(), "projected_endpoint_b_px": p1.tolist()},
        "target": {"name": "C3", "default": target_default, "side": target_side},
        "ranking_by_default_width_error": rows,
        "held_constant": ["C3 geometry and existing C3 renders", "camera and intrinsics", "plate and bolt", "feature centre, length, and angle", "default/side key light poses, energy, size, and colour", "fill light", "world/background", "base material", "Cycles settings", "AgX/exposure", "resolution 640x480"],
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ["width_rank", "candidate", "physical_width_mm", "default_apparent_width_px", "width_error_px", "default_local_contrast_levels", "side_light_response_change_levels", "default_width_std_px", "default_valid_cross_sections", "default_signed_contrast_levels", "default_bright_fraction", "default_dark_fraction"]
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(f"Saved {METRICS_JSON}")
    print(f"Saved {METRICS_CSV}")
    for row in rows:
        print(f"  {row['width_rank']}: {row['candidate']} width={row['default_apparent_width_px']:.3f}px error={row['width_error_px']:.3f}px contrast={row['default_local_contrast_levels']:.2f} side_change={row['side_light_response_change_levels']:.2f}")


def main():
    if not C3_DEFAULT.exists() or not C3_SIDE.exists():
        raise FileNotFoundError(f"Expected existing C3 renders: {C3_DEFAULT} and {C3_SIDE}")
    WIDTH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    configure_cycles(scene)
    configure_color_management(scene)
    plate_obj = import_stl(PLATE_PATH, "WidthCalibrationPlate")
    bolt_obj = import_stl(BOLT_PATH, "Bolt")
    apply_flat_shading(plate_obj)
    apply_smooth_shading(bolt_obj)
    bolt_obj.data.materials.append(create_uniform_metal_material("WidthCalibrationBolt"))
    materials = {name: create_reflection_material(f"Width_{name}", **parameters) for name, parameters in REFLECTIONS.items()}
    plate_obj.data.materials.append(materials["RW1"])

    rotation_matrix = bproc.camera.rotation_from_forward_vec(CAM_LOOK_AT - CAM_LOCATION)
    cam2world = bproc.math.build_transformation_mat(CAM_LOCATION, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    scene.camera.data.lens = 50.0
    scene.camera.data.sensor_width = 36.0

    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location(FILL_POS)
    fill_light.set_energy(FILL_ENERGY)
    fill_light.set_color(FILL_COLOR)
    set_area_dimensions(fill_light.blender_obj, FILL_SIZE, FILL_SIZE)
    orient_area_toward(fill_light.blender_obj, TARGET_CENTER)

    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Strength"].default_value = 0.08
    background.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)

    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_energy(KEY_ENERGY)
    key_light.set_color(KEY_COLOR)
    set_area_dimensions(key_light.blender_obj, KEY_SIZE_X, KEY_SIZE_Y)
    p0, p1 = projected_feature(scene, plate_obj)
    print(f"Projected feature: {p0.tolist()} -> {p1.tolist()}")
    render_reflections(scene, key_light, plate_obj, bolt_obj, materials)

    measurements = {"C3": {}}
    c3_default = imageio.imread(str(C3_DEFAULT))
    c3_side = imageio.imread(str(C3_SIDE))
    measurements["C3"]["default"] = apparent_width_measurement(c3_default, p0, p1)
    measurements["C3"]["side"] = apparent_width_measurement(c3_side, p0, p1)
    measurements["C3"]["side"]["response_change"] = matched_response_change(c3_default, c3_side, p0, p1)
    for name in REFLECTIONS:
        default_image = imageio.imread(str(REFLECTION_DIR / f"{name}_default.png"))
        side_image = imageio.imread(str(REFLECTION_DIR / f"{name}_side.png"))
        measurements[name] = {
            "default": apparent_width_measurement(default_image, p0, p1),
            "side": apparent_width_measurement(side_image, p0, p1),
        }
        measurements[name]["default"]["response_change"] = matched_response_change(default_image, side_image, p0, p1)
        measurements[name]["side"]["response_change"] = measurements[name]["default"]["response_change"]

    make_contact_sheet()
    write_metrics(measurements, p0, p1)
    print(f"Saved contact sheet: {CONTACT_SHEET}")
    print("Width calibration complete: 8 new reflection renders; C3 target reused.")


if __name__ == "__main__":
    main()

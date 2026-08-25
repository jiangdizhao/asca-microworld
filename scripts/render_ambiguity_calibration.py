import blenderproc as bproc

"""Render and score the controlled 3-by-3 reflection/crack calibration.

This script produces exactly twelve final candidate images: two light
conditions for each of three BRDF-only reflections and three subtractive
CadQuery crack plates.  Pair scores are computed afterwards from the same
fixed local ROI; no pair-specific renders are made.
"""

import csv
import json
import math
from pathlib import Path
import bpy
import imageio.v2 as imageio
import mathutils
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_PATH = REPO_ROOT / "outputs" / "plate.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
CRACK_DIR = REPO_ROOT / "outputs" / "calibration_cracks"
OUTPUT_DIR = REPO_ROOT / "renders" / "calibration"
REFLECTION_DIR = OUTPUT_DIR / "reflection"
CRACK_RENDER_DIR = OUTPUT_DIR / "crack"
PAIR_JSON = OUTPUT_DIR / "pair_scores.json"
PAIR_CSV = OUTPUT_DIR / "pair_scores.csv"
CONTACT_SHEET = OUTPUT_DIR / "contact_sheet.png"

DEFAULT_KEY_POS = np.array([-120.0, 156.0, 100.0])
SIDE_KEY_POS = np.array([140.0, -140.0, 100.0])
TARGET_CENTER = np.array([10.0, 8.0, 6.0])
KEY_ENERGY = 120000.0
KEY_SIZE_X = 25.0
KEY_SIZE_Y = 12.0
KEY_COLOR = np.array([1.0, 0.95, 0.9])

FILL_POS = np.array([-80.0, -60.0, 60.0])
FILL_ENERGY = 8000.0
FILL_SIZE = 80.0
FILL_COLOR = np.array([0.9, 0.95, 1.0])

CAM_LOCATION = np.array([140.0, -140.0, 100.0])
CAM_LOOK_AT = np.array([0.0, 0.0, 10.0])
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

FEATURE_CENTER = (10.0, 8.0)
FEATURE_LENGTH = 36.0
FEATURE_ANGLE_DEG = 35.0
FIXED_ROI = (280, 230, 466, 285)

BASE_COLOR = (0.75, 0.75, 0.75, 1.0)
BASE_METALLIC = 0.7
BASE_ROUGHNESS = 0.30

REFLECTIONS = {
    "R1": {"width_mm": 1.5, "polished_roughness": 0.08, "metallic": 1.0},
    "R2": {"width_mm": 2.5, "polished_roughness": 0.05, "metallic": 1.0},
    "R3": {"width_mm": 3.5, "polished_roughness": 0.03, "metallic": 1.0},
}
CRACKS = {
    "C1": {"top_width_mm": 0.6, "depth_mm": 0.12, "bottom_width_mm": 0.05, "bottom_offset_mm": 0.10},
    "C2": {"top_width_mm": 0.8, "depth_mm": 0.18, "bottom_width_mm": 0.06, "bottom_offset_mm": 0.16},
    "C3": {"top_width_mm": 1.0, "depth_mm": 0.25, "bottom_width_mm": 0.08, "bottom_offset_mm": 0.22},
}


def apply_flat_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = False


def apply_smooth_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = True


def create_uniform_metal_material(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = BASE_COLOR
    bsdf.inputs["Metallic"].default_value = BASE_METALLIC
    bsdf.inputs["Roughness"].default_value = BASE_ROUGHNESS
    return mat


def create_reflection_material(name, width_mm, polished_roughness, metallic):
    """Create a material-only spatial BRDF strip; there is no extra geometry."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = BASE_COLOR

    geo = nodes.new("ShaderNodeNewGeometry")
    sub = nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (FEATURE_CENTER[0], FEATURE_CENTER[1], 0.0)
    rotate = nodes.new("ShaderNodeVectorRotate")
    rotate.rotation_type = "Z_AXIS"
    rotate.inputs["Angle"].default_value = math.radians(-FEATURE_ANGLE_DEG)
    separate = nodes.new("ShaderNodeSeparateXYZ")

    abs_u = nodes.new("ShaderNodeMath")
    abs_u.operation = "ABSOLUTE"
    abs_v = nodes.new("ShaderNodeMath")
    abs_v.operation = "ABSOLUTE"
    inside_length = nodes.new("ShaderNodeMath")
    inside_length.operation = "LESS_THAN"
    inside_length.inputs[1].default_value = FEATURE_LENGTH / 2.0
    inside_width = nodes.new("ShaderNodeMath")
    inside_width.operation = "LESS_THAN"
    inside_width.inputs[1].default_value = width_mm / 2.0
    mask = nodes.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"

    roughness = nodes.new("ShaderNodeMixRGB")
    roughness.inputs["Color1"].default_value = (BASE_ROUGHNESS,) * 3 + (1.0,)
    roughness.inputs["Color2"].default_value = (polished_roughness,) * 3 + (1.0,)
    metallic_node = nodes.new("ShaderNodeMixRGB")
    metallic_node.inputs["Color1"].default_value = (BASE_METALLIC,) * 3 + (1.0,)
    metallic_node.inputs["Color2"].default_value = (metallic,) * 3 + (1.0,)
    base_color = nodes.new("ShaderNodeRGB")
    base_color.outputs["Color"].default_value = BASE_COLOR

    links.new(geo.outputs["Position"], sub.inputs[0])
    links.new(sub.outputs[0], rotate.inputs["Vector"])
    links.new(rotate.outputs[0], separate.inputs["Vector"])
    links.new(separate.outputs["X"], abs_u.inputs[0])
    links.new(separate.outputs["Y"], abs_v.inputs[0])
    links.new(abs_u.outputs[0], inside_length.inputs[0])
    links.new(abs_v.outputs[0], inside_width.inputs[0])
    links.new(inside_length.outputs[0], mask.inputs[0])
    links.new(inside_width.outputs[0], mask.inputs[1])
    links.new(mask.outputs[0], roughness.inputs["Fac"])
    links.new(mask.outputs[0], metallic_node.inputs["Fac"])
    links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(metallic_node.outputs["Color"], bsdf.inputs["Metallic"])
    links.new(base_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def configure_cycles(scene):
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.samples = 256
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.01
    cycles.max_bounces = 8
    cycles.diffuse_bounces = 4
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    gpu_enabled = False
    device_summary = []
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        selected_backend = None
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                usable = [d for d in prefs.devices if d.type in ("OPTIX", "CUDA")]
                if usable:
                    for dev in prefs.devices:
                        dev.use = dev.type in ("OPTIX", "CUDA")
                    selected_backend = backend
                    gpu_enabled = True
                    break
            except Exception:
                continue
        if gpu_enabled:
            cycles.device = "GPU"
        device_summary = [f"{d.name}:{d.type}:use={d.use}" for d in prefs.devices]
        print(f"Cycles device backend: {selected_backend or 'CPU fallback'}")
        print("Cycles devices: " + (", ".join(device_summary) or "CPU"))
    except Exception as exc:
        cycles.device = "CPU"
        print(f"Cycles GPU setup unavailable; using CPU fallback ({exc})")
    return gpu_enabled, device_summary


def configure_color_management(scene):
    view = scene.view_settings
    view.view_transform = "AgX"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False


def orient_area_toward(light_obj, target):
    direction = target - np.array(light_obj.location)
    direction = direction / np.linalg.norm(direction)
    rotation = bproc.camera.rotation_from_forward_vec(direction)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = light_obj.location
    light_obj.matrix_world = mathutils.Matrix(transform.tolist())


def set_area_dimensions(light_obj, size_x, size_y):
    light_obj.data.shape = "RECTANGLE"
    light_obj.data.size = size_x
    light_obj.data.size_y = size_y


def move_key_light(light, position):
    light.set_location(position)
    orient_area_toward(light.blender_obj, TARGET_CENTER)


def import_stl(path, name):
    try:
        bpy.ops.wm.stl_import(filepath=str(path))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(path))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def save_frame(path):
    data = bproc.renderer.render()
    imageio.imwrite(str(path), data["colors"][0])
    print(f"  Saved: {path}")


def project_point(scene, plate_obj, x, y, z):
    world = plate_obj.matrix_world @ mathutils.Vector((float(x), float(y), float(z)))
    co = world_to_camera_view(scene, scene.camera, world)
    return np.array([float(co.x * IMAGE_WIDTH), float((1.0 - co.y) * IMAGE_HEIGHT)])


def projected_feature(scene, plate_obj):
    top_z = max(float(v[2]) for v in plate_obj.bound_box) + 0.02
    angle = math.radians(FEATURE_ANGLE_DEG)
    along = np.array([math.cos(angle), math.sin(angle)])
    center = np.array(FEATURE_CENTER)
    p0 = project_point(scene, plate_obj, *(center - along * FEATURE_LENGTH / 2.0), top_z)
    p1 = project_point(scene, plate_obj, *(center + along * FEATURE_LENGTH / 2.0), top_z)
    return p0, p1


def line_masks(p0, p1):
    x0, y0, x1, y1 = FIXED_ROI
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length_sq = dx * dx + dy * dy
    t = ((xx - p0[0]) * dx + (yy - p0[1]) * dy) / length_sq
    tc = np.clip(t, 0.0, 1.0)
    nearest_x = p0[0] + tc * dx
    nearest_y = p0[1] + tc * dy
    distance = np.hypot(xx - nearest_x, yy - nearest_y)
    active = (t >= 0.10) & (t <= 0.98)
    line = active & (distance <= 2.6)
    reference = active & (distance >= 8.0) & (distance <= 15.0)
    return line, reference, nearest_x, nearest_y


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def image_descriptor(image, p0, p1):
    x0, y0, x1, y1 = FIXED_ROI
    rgb = image[y0:y1, x0:x1, :3].astype(np.float32) / 255.0
    lum_full = luminance(image[..., :3].astype(np.float32) / 255.0)
    lum = lum_full[y0:y1, x0:x1]
    line, reference, nearest_x, nearest_y = line_masks(p0, p1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    seg_len = math.hypot(dx, dy)
    nx, ny = -dy / seg_len, dx / seg_len
    ax = np.clip(np.rint(nearest_x + nx * 10.0).astype(int), 0, IMAGE_WIDTH - 1)
    ay = np.clip(np.rint(nearest_y + ny * 10.0).astype(int), 0, IMAGE_HEIGHT - 1)
    bx = np.clip(np.rint(nearest_x - nx * 10.0).astype(int), 0, IMAGE_WIDTH - 1)
    by = np.clip(np.rint(nearest_y - ny * 10.0).astype(int), 0, IMAGE_HEIGHT - 1)
    local_reference = (lum_full[ay, ax] + lum_full[by, bx]) / 2.0
    line_values = lum[line]
    signed = line_values - local_reference[line]
    contrast = np.abs(signed)
    gy, gx = np.gradient(lum)
    gradient = np.hypot(gx, gy)
    def frac(values, predicate):
        return float(predicate(values).mean()) if len(values) else 0.0
    values = {
        "mean_luminance": float(lum.mean()),
        "std_luminance": float(lum.std()),
        "local_contrast_p95_minus_p50": float(np.percentile(lum, 95) - np.percentile(lum, 50)),
        "bright_fraction": frac(lum, lambda v: v >= 0.90),
        "dark_fraction": frac(lum, lambda v: v <= 0.25),
        "gradient_mean": float(gradient.mean()),
        "gradient_p95": float(np.percentile(gradient, 95)),
        "line_extent_fraction": frac(contrast, lambda v: v >= 10.0 / 255.0),
        "line_abs_contrast_mean": float(contrast.mean()) if len(contrast) else 0.0,
        "line_abs_contrast_p95": float(np.percentile(contrast, 95)) if len(contrast) else 0.0,
        "line_mean_luminance": float(line_values.mean()) if len(line_values) else 0.0,
        "line_luminance_std": float(line_values.std()) if len(line_values) else 0.0,
        "line_signed_contrast_mean": float(signed.mean()) if len(signed) else 0.0,
        "line_bright_fraction": frac(signed, lambda v: v >= 10.0 / 255.0),
        "line_dark_fraction": frac(signed, lambda v: v <= -10.0 / 255.0),
        "line_pixel_count": int(line.sum()),
    }
    return values


DESCRIPTOR_SCALES = {
    "mean_luminance": 1.0,
    "std_luminance": 1.0,
    "local_contrast_p95_minus_p50": 1.0,
    "bright_fraction": 1.0,
    "dark_fraction": 1.0,
    "gradient_mean": 0.20,
    "gradient_p95": 0.50,
    "line_extent_fraction": 1.0,
    "line_abs_contrast_mean": 1.0,
    "line_abs_contrast_p95": 1.0,
    "line_mean_luminance": 1.0,
    "line_luminance_std": 1.0,
    "line_signed_contrast_mean": 1.0,
    "line_bright_fraction": 1.0,
    "line_dark_fraction": 1.0,
}


def descriptor_distance(a, b):
    terms = {}
    for key, scale in DESCRIPTOR_SCALES.items():
        terms[key] = abs(float(a[key]) - float(b[key])) * scale
    return float(math.sqrt(sum(value * value for value in terms.values()))), terms


def pair_distance(ref_img, crack_img, p0, p1):
    ref = ref_img[FIXED_ROI[1]:FIXED_ROI[3], FIXED_ROI[0]:FIXED_ROI[2], :3].astype(np.float32) / 255.0
    crack = crack_img[FIXED_ROI[1]:FIXED_ROI[3], FIXED_ROI[0]:FIXED_ROI[2], :3].astype(np.float32) / 255.0
    ref_lum = luminance(ref)
    crack_lum = luminance(crack)
    mae = float(np.abs(ref - crack).mean())
    correlation = float(np.corrcoef(ref_lum.ravel(), crack_lum.ravel())[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    ref_desc = image_descriptor(ref_img, p0, p1)
    crack_desc = image_descriptor(crack_img, p0, p1)
    desc_distance, terms = descriptor_distance(ref_desc, crack_desc)
    # All terms are local and normalized. Descriptor distance carries the
    # interpretable feature diagnostics; MAE/correlation guard against a
    # visually obvious pixel-level mismatch.
    score = 0.65 * desc_distance + 0.25 * mae + 0.10 * (1.0 - correlation)
    return {
        "D": float(score),
        "descriptor_distance": desc_distance,
        "descriptor_distance_terms": terms,
        "roi_mae": mae,
        "roi_normalized_correlation": correlation,
        "reflection_descriptor": ref_desc,
        "crack_descriptor": crack_desc,
    }


def render_all(scene, key_light, reflection_plate, crack_plates, bolt, reflection_materials, uniform_material, p0, p1):
    for path in (REFLECTION_DIR, CRACK_RENDER_DIR):
        path.mkdir(parents=True, exist_ok=True)
    bolt.hide_render = False
    for plate in crack_plates:
        plate.hide_render = True

    for name, parameters in REFLECTIONS.items():
        reflection_plate.hide_render = False
        reflection_plate.data.materials.clear()
        reflection_plate.data.materials.append(reflection_materials[name])
        for condition, position in (("default", DEFAULT_KEY_POS), ("side", SIDE_KEY_POS)):
            move_key_light(key_light, position)
            save_frame(REFLECTION_DIR / f"{name}_{condition}.png")
        reflection_plate.hide_render = True

    for name, plate in zip(CRACKS, crack_plates):
        plate.hide_render = False
        plate.data.materials.clear()
        plate.data.materials.append(uniform_material)
        for condition, position in (("default", DEFAULT_KEY_POS), ("side", SIDE_KEY_POS)):
            move_key_light(key_light, position)
            save_frame(CRACK_RENDER_DIR / f"{name}_{condition}.png")
        plate.hide_render = True


def make_contact_sheet():
    tile_w, tile_h = IMAGE_WIDTH, IMAGE_HEIGHT
    label_h = 32
    margin_x, margin_y = 20, 20
    label_w = 180
    image_x = margin_x + label_w
    sheet_w = image_x + tile_w * 2 + margin_x
    sheet_h = margin_y * 2 + label_h * 7 + tile_h * 6
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
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
    rows = [("Reflection R1", "R1", REFLECTION_DIR), ("Reflection R2", "R2", REFLECTION_DIR),
            ("Reflection R3", "R3", REFLECTION_DIR), ("Crack C1", "C1", CRACK_RENDER_DIR),
            ("Crack C2", "C2", CRACK_RENDER_DIR), ("Crack C3", "C3", CRACK_RENDER_DIR)]
    for label, name, directory in rows:
        draw.text((margin_x, y + tile_h // 2 - 12), label, fill=(0, 0, 0), font=small)
        # Labels are placed in a dedicated left margin; the image columns
        # remain exact render pixels and are not altered.
        for column, condition in enumerate(("default", "side")):
            img = Image.open(directory / f"{name}_{condition}.png").convert("RGB")
            sheet.paste(img, (image_x + column * tile_w, y))
        y += tile_h + label_h
    sheet.save(CONTACT_SHEET)
    print(f"Saved contact sheet: {CONTACT_SHEET}")


def score_pairs(p0, p1):
    results = []
    for reflection_name, reflection_parameters in REFLECTIONS.items():
        for crack_name, crack_parameters in CRACKS.items():
            default_ref = imageio.imread(str(REFLECTION_DIR / f"{reflection_name}_default.png"))
            default_crack = imageio.imread(str(CRACK_RENDER_DIR / f"{crack_name}_default.png"))
            side_ref = imageio.imread(str(REFLECTION_DIR / f"{reflection_name}_side.png"))
            side_crack = imageio.imread(str(CRACK_RENDER_DIR / f"{crack_name}_side.png"))
            default = pair_distance(default_ref, default_crack, p0, p1)
            side = pair_distance(side_ref, side_crack, p0, p1)
            results.append({
                "pair": f"{reflection_name}-{crack_name}",
                "D_default": default["D"],
                "D_side": side["D"],
                "G": side["D"] - default["D"],
                "reflection_parameters": {"center_mm": list(FEATURE_CENTER), "length_mm": FEATURE_LENGTH, "angle_degrees": FEATURE_ANGLE_DEG, **reflection_parameters},
                "crack_parameters": {"center_mm": list(FEATURE_CENTER), "length_mm": FEATURE_LENGTH, "angle_degrees": FEATURE_ANGLE_DEG, **crack_parameters, "subtractive_geometry": True},
                "default_details": default,
                "side_details": side,
            })
    results.sort(key=lambda item: (-item["G"], item["D_default"]))
    for index, item in enumerate(results, start=1):
        item["rank"] = index
    payload = {
        "metric": "weighted local ROI descriptor distance; G = D_side - D_default",
        "roi_xyxy_pixels": list(FIXED_ROI),
        "feature": {"center_mm": list(FEATURE_CENTER), "length_mm": FEATURE_LENGTH, "angle_degrees": FEATURE_ANGLE_DEG},
        "key_light": {"type": "AREA", "energy": KEY_ENERGY, "size_x_mm": KEY_SIZE_X, "size_y_mm": KEY_SIZE_Y, "color": KEY_COLOR.tolist(), "default_position": DEFAULT_KEY_POS.tolist(), "side_position": SIDE_KEY_POS.tolist(), "target": TARGET_CENTER.tolist()},
        "fill_light": {"position": FILL_POS.tolist(), "energy": FILL_ENERGY, "size_mm": FILL_SIZE, "color": FILL_COLOR.tolist()},
        "held_constant": ["plate.stl and bolt.stl geometry", "neutral-gray base material", "camera pose and intrinsics", "feature centre, length, and angle", "fill light and world/background", "Cycles settings", "AgX/exposure", "resolution 640x480"],
        "ranking": results,
    }
    PAIR_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "rank", "pair", "D_default", "D_side", "G",
        "reflection_width_mm", "reflection_polished_roughness", "reflection_metallic",
        "crack_top_width_mm", "crack_depth_mm", "crack_bottom_width_mm", "crack_bottom_offset_mm",
    ]
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rows = []
        for row in results:
            reflection = row["reflection_parameters"]
            crack = row["crack_parameters"]
            rows.append({
                "rank": row["rank"],
                "pair": row["pair"],
                "D_default": row["D_default"],
                "D_side": row["D_side"],
                "G": row["G"],
                "reflection_width_mm": reflection["width_mm"],
                "reflection_polished_roughness": reflection["polished_roughness"],
                "reflection_metallic": reflection["metallic"],
                "crack_top_width_mm": crack["top_width_mm"],
                "crack_depth_mm": crack["depth_mm"],
                "crack_bottom_width_mm": crack["bottom_width_mm"],
                "crack_bottom_offset_mm": crack["bottom_offset_mm"],
            })
        writer.writerows(rows)
    print(f"Saved pair scores: {PAIR_JSON}")
    print(f"Saved pair scores: {PAIR_CSV}")
    print("Ranking:")
    for item in results:
        print(f"  {item['rank']}: {item['pair']} D_default={item['D_default']:.6f} D_side={item['D_side']:.6f} G={item['G']:.6f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    gpu_enabled, device_summary = configure_cycles(scene)
    configure_color_management(scene)

    reflection_plate = import_stl(PLATE_PATH, "ReflectionPlate")
    bolt = import_stl(BOLT_PATH, "Bolt")
    crack_plates = [import_stl(CRACK_DIR / f"{name}.stl", f"CrackPlate_{name}") for name in CRACKS]
    apply_flat_shading(reflection_plate)
    for plate in crack_plates:
        apply_flat_shading(plate)
    apply_smooth_shading(bolt)
    bolt.data.materials.append(create_uniform_metal_material("BoltMetal"))
    uniform_material = create_uniform_metal_material("CrackPlateMetal")

    reflection_materials = {name: create_reflection_material(f"Reflection_{name}", **parameters) for name, parameters in REFLECTIONS.items()}
    reflection_plate.data.materials.append(reflection_materials["R1"])

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

    p0, p1 = projected_feature(scene, reflection_plate)
    print(f"Fixed ROI: {FIXED_ROI}; projected feature endpoints: {p0.tolist()} -> {p1.tolist()}")
    render_all(scene, key_light, reflection_plate, crack_plates, bolt, reflection_materials, uniform_material, p0, p1)
    make_contact_sheet()
    score_pairs(p0, p1)
    print(f"GPU enabled: {gpu_enabled}; devices: {device_summary}")
    print("Calibration complete: 12 final candidate renders.")


if __name__ == "__main__":
    main()

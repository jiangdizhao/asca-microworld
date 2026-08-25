import blenderproc as bproc

import json
import math
from pathlib import Path

import bpy
import imageio.v2 as imageio
import mathutils
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_PATH = REPO_ROOT / "outputs" / "plate_crack.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
OUTPUT_DIR = REPO_ROOT / "renders" / "crack"
DEFAULT_OUTPUT = OUTPUT_DIR / "default_light.png"
SIDE_OUTPUT = OUTPUT_DIR / "side_light.png"
METRICS_OUTPUT = OUTPUT_DIR / "roi_metrics.json"
REFLECTION_DEFAULT = REPO_ROOT / "renders" / "pbr_reflection" / "default_light.png"

DEFAULT_KEY_POS = np.array([-120.0, 156.0, 100.0])
SIDE_KEY_POS = np.array([140.0, -140.0, 100.0])
TARGET_CENTER = np.array([10.0, 8.0, 6.0])
KEY_ENERGY = 120000.0
KEY_SIZE_X = 25.0
KEY_SIZE_Y = 12.0

FILL_POS = np.array([-80.0, -60.0, 60.0])
FILL_ENERGY = 8000.0
FILL_SIZE = 80.0

CAM_LOCATION = np.array([140.0, -140.0, 100.0])
CAM_LOOK_AT = np.array([0.0, 0.0, 10.0])
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CRACK_CENTER = (10.0, 8.0)
CRACK_LENGTH = 36.0
CRACK_WIDTH = 1.5
CRACK_DEPTH = 0.25
CRACK_ANGLE_DEG = 35.0

BASE_COLOR = (0.75, 0.75, 0.75, 1.0)
BASE_METALLIC = 0.7
BASE_ROUGHNESS = 0.30


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
    rot_3x3 = bproc.camera.rotation_from_forward_vec(direction)
    mat_4x4 = np.eye(4)
    mat_4x4[:3, :3] = rot_3x3
    mat_4x4[:3, 3] = light_obj.location
    light_obj.matrix_world = mathutils.Matrix(mat_4x4.tolist())


def set_area_dimensions(light_obj, size_x, size_y):
    light_data = light_obj.data
    light_data.shape = "RECTANGLE"
    light_data.size = size_x
    light_data.size_y = size_y


def move_key_light(light, position):
    light.set_location(position)
    orient_area_toward(light.blender_obj, TARGET_CENTER)


def save_frame(output_path):
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(output_path), rgb)
    print(f"  Saved: {output_path}")


def project_point(scene, plate_obj, x, y, z):
    world = plate_obj.matrix_world @ mathutils.Vector((float(x), float(y), float(z)))
    co = world_to_camera_view(scene, scene.camera, world)
    return np.array([float(co.x * IMAGE_WIDTH), float((1.0 - co.y) * IMAGE_HEIGHT)])


def projected_crack_geometry(scene, plate_obj, padding=18):
    top_z = max(float(v[2]) for v in plate_obj.bound_box) + 0.02
    angle = math.radians(CRACK_ANGLE_DEG)
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-along[1], along[0]])
    center = np.array(CRACK_CENTER)
    corners = [
        center + sign_a * along * CRACK_LENGTH / 2 + sign_b * across * CRACK_WIDTH / 2
        for sign_a in (-1, 1)
        for sign_b in (-1, 1)
    ]
    projected_corners = [project_point(scene, plate_obj, p[0], p[1], top_z) for p in corners]
    p0 = project_point(
        scene,
        plate_obj,
        *(center - along * CRACK_LENGTH / 2),
        top_z,
    )
    p1 = project_point(
        scene,
        plate_obj,
        *(center + along * CRACK_LENGTH / 2),
        top_z,
    )
    xs = [p[0] for p in projected_corners]
    ys = [p[1] for p in projected_corners]
    roi = (
        max(0, int(math.floor(min(xs) - padding))),
        max(0, int(math.floor(min(ys) - padding))),
        min(IMAGE_WIDTH, int(math.ceil(max(xs) + padding + 1))),
        min(IMAGE_HEIGHT, int(math.ceil(max(ys) + padding + 1))),
    )
    return roi, p0, p1, projected_corners


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def line_masks(roi, p0, p1):
    x0, y0, x1, y1 = roi
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length_sq = dx * dx + dy * dy
    t = ((xx - p0[0]) * dx + (yy - p0[1]) * dy) / length_sq
    tc = np.clip(t, 0.0, 1.0)
    nearest_x = p0[0] + tc * dx
    nearest_y = p0[1] + tc * dy
    distance = np.hypot(xx - nearest_x, yy - nearest_y)
    active = (t >= 0.12) & (t <= 0.98)
    line = active & (distance <= 2.2)
    reference = active & (distance >= 8.0) & (distance <= 15.0)
    return xx, yy, t, tc, line, reference, nearest_x, nearest_y


def crack_image_metrics(image, roi, p0, p1):
    xx, yy, t, tc, line, reference, nearest_x, nearest_y = line_masks(roi, p0, p1)
    lum = luminance(image[..., :3].astype(np.float32))
    x0, y0, _, _ = roi
    local_lum = lum[y0:roi[3], x0:roi[2]]
    line_values = local_lum[line]
    reference_values = local_lum[reference]

    # Sample the two nearby normal offsets at each pixel, giving a local plate
    # reference even when the key light creates a broad luminance gradient.
    seg_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    nx = -(p1[1] - p0[1]) / seg_len
    ny = (p1[0] - p0[0]) / seg_len
    sample_a_x = np.clip(np.rint(nearest_x + nx * 10.0).astype(int), 0, IMAGE_WIDTH - 1)
    sample_a_y = np.clip(np.rint(nearest_y + ny * 10.0).astype(int), 0, IMAGE_HEIGHT - 1)
    sample_b_x = np.clip(np.rint(nearest_x - nx * 10.0).astype(int), 0, IMAGE_WIDTH - 1)
    sample_b_y = np.clip(np.rint(nearest_y - ny * 10.0).astype(int), 0, IMAGE_HEIGHT - 1)
    local_reference = (lum[sample_a_y, sample_a_x] + lum[sample_b_y, sample_b_x]) / 2.0
    line_reference = local_reference[line]
    contrast_values = np.abs(line_values - line_reference)
    distinguishable = contrast_values >= 10.0

    feature_pixels = int(distinguishable.sum())
    line_count = int(line.sum())
    return {
        "roi_xyxy_pixels": [int(v) for v in roi],
        "line_segment_pixels": line_count,
        "line_feature_extent_pixels": feature_pixels,
        "distinguishable_fraction": float(feature_pixels / line_count) if line_count else 0.0,
        "local_crack_contrast_abs_mean": float(contrast_values.mean()) if len(contrast_values) else 0.0,
        "local_crack_contrast_abs_p95": float(np.percentile(contrast_values, 95)) if len(contrast_values) else 0.0,
        "line_luminance_mean": float(line_values.mean()) if len(line_values) else 0.0,
        "nearby_plate_luminance_mean": float(reference_values.mean()) if len(reference_values) else 0.0,
        "line_luminance_std": float(line_values.std()) if len(line_values) else 0.0,
    }


def roi_summary(image, roi):
    x0, y0, x1, y1 = roi
    values = luminance(image[y0:y1, x0:x1, :3].astype(np.float32))
    return {
        "mean_luminance": float(values.mean()),
        "std_luminance": float(values.std()),
        "p95_minus_p50_luminance": float(np.percentile(values, 95) - np.percentile(values, 50)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    gpu_enabled, device_summary = configure_cycles(scene)
    configure_color_management(scene)

    try:
        bpy.ops.wm.stl_import(filepath=str(PLATE_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(PLATE_PATH))
    plate_obj = bpy.context.active_object
    plate_obj.name = "CrackedPlate"
    try:
        bpy.ops.wm.stl_import(filepath=str(BOLT_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(BOLT_PATH))
    bolt_obj = bpy.context.active_object
    bolt_obj.name = "Bolt"
    apply_flat_shading(plate_obj)
    apply_smooth_shading(bolt_obj)
    print(f"Cracked plate dimensions: {tuple(round(v, 3) for v in plate_obj.dimensions)}")
    print(f"Bolt dimensions: {tuple(round(v, 3) for v in bolt_obj.dimensions)}")

    plate_obj.data.materials.append(create_uniform_metal_material("CrackedPlateMetal"))
    bolt_obj.data.materials.append(create_uniform_metal_material("BoltMetal"))

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
    fill_light.set_color(np.array([0.9, 0.95, 1.0]))
    set_area_dimensions(fill_light.blender_obj, FILL_SIZE, FILL_SIZE)
    orient_area_toward(fill_light.blender_obj, TARGET_CENTER)

    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.08
    bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)

    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_energy(KEY_ENERGY)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))
    set_area_dimensions(key_light.blender_obj, KEY_SIZE_X, KEY_SIZE_Y)

    roi, p0, p1, projected_corners = projected_crack_geometry(scene, plate_obj)
    print("Projected crack endpoints: " + json.dumps([p0.tolist(), p1.tolist()]))
    print("Projected crack corners: " + json.dumps([p.tolist() for p in projected_corners]))
    print("Expanded crack ROI: " + str(roi))

    # One scene, one key object, and only its transform changes between A/B.
    print("Rendering crack condition A - default illumination ...")
    move_key_light(key_light, DEFAULT_KEY_POS)
    save_frame(DEFAULT_OUTPUT)

    print("Rendering crack condition B - side illumination ...")
    move_key_light(key_light, SIDE_KEY_POS)
    save_frame(SIDE_OUTPUT)

    img_default = imageio.imread(str(DEFAULT_OUTPUT))
    img_side = imageio.imread(str(SIDE_OUTPUT))
    default_metrics = crack_image_metrics(img_default, roi, p0, p1)
    side_metrics = crack_image_metrics(img_side, roi, p0, p1)
    comparison = {
        "crack_default_roi": roi_summary(img_default, roi),
        "reflection_default_roi": None,
        "crack_default_line_feature": default_metrics["line_feature_extent_pixels"],
        "reflection_default_line_feature": None,
    }
    if REFLECTION_DEFAULT.exists():
        reflection_default = imageio.imread(str(REFLECTION_DEFAULT))
        reflection_metrics = crack_image_metrics(reflection_default, roi, p0, p1)
        comparison["reflection_default_roi"] = roi_summary(reflection_default, roi)
        comparison["reflection_default_line_feature"] = reflection_metrics["line_feature_extent_pixels"]
        comparison["reflection_default_local_contrast"] = reflection_metrics["local_crack_contrast_abs_mean"]

    metrics = {
        "crack_geometry": {
            "construction": "original CadQuery plate minus one rotated shallow box cutter",
            "subtractive_geometry": True,
            "length_mm": CRACK_LENGTH,
            "width_mm": CRACK_WIDTH,
            "depth_mm": CRACK_DEPTH,
            "center_mm": list(CRACK_CENTER),
            "angle_degrees": CRACK_ANGLE_DEG,
        },
        "projected_crack": {
            "roi_xyxy_pixels": [int(v) for v in roi],
            "endpoint_a_xy_pixels": p0.tolist(),
            "endpoint_b_xy_pixels": p1.tolist(),
            "corners_xy_pixels": [p.tolist() for p in projected_corners],
        },
        "default_light": default_metrics,
        "side_light": side_metrics,
        "default_crack_vs_reflection": comparison,
        "key_light": {
            "type": "AREA",
            "energy": KEY_ENERGY,
            "size_x_mm": KEY_SIZE_X,
            "size_y_mm": KEY_SIZE_Y,
            "color": [1.0, 0.95, 0.9],
            "default_position": DEFAULT_KEY_POS.tolist(),
            "side_position": SIDE_KEY_POS.tolist(),
            "target": TARGET_CENTER.tolist(),
        },
        "held_constant": [
            "plate_crack.stl and bolt.stl geometry",
            "uniform neutral-gray material: metallic=0.7, roughness=0.30",
            "camera pose and intrinsics",
            "resolution 640x480",
            "fill AREA light transform, type, size, energy, and colour",
            "world background colour and strength",
            "Cycles settings",
            "AgX colour management, exposure, and gamma",
        ],
        "cycles": {
            "engine": scene.render.engine,
            "samples": scene.cycles.samples,
            "adaptive_sampling": scene.cycles.use_adaptive_sampling,
            "adaptive_threshold": scene.cycles.adaptive_threshold,
            "denoising": scene.cycles.use_denoising,
            "denoiser": scene.cycles.denoiser,
            "device_mode": scene.cycles.device,
            "gpu_enabled": gpu_enabled,
            "devices": device_summary,
        },
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== Geometric crack experiment complete ===")
    print(json.dumps(metrics, indent=2))
    print("Outputs:")
    print(f"  {DEFAULT_OUTPUT}")
    print(f"  {SIDE_OUTPUT}")
    print(f"  {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()

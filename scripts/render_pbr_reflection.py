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
PLATE_PATH = REPO_ROOT / "outputs" / "plate.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
OUTPUT_DIR = REPO_ROOT / "renders" / "pbr_reflection"
DEBUG_OUTPUT = OUTPUT_DIR / "debug_mask.png"
DEFAULT_OUTPUT = OUTPUT_DIR / "default_light.png"
SIDE_OUTPUT = OUTPUT_DIR / "side_light.png"
METRICS_OUTPUT = OUTPUT_DIR / "roi_metrics.json"

# One key AREA light is reused for the two final conditions.  Its type,
# energy, colour, and physical dimensions never change after creation.
DEFAULT_KEY_POS = np.array([-120.0, 156.0, 100.0])
SIDE_KEY_POS = np.array([140.0, -140.0, 100.0])
TARGET_CENTER = np.array([10.0, 8.0, 6.0])
KEY_ENERGY = 120000.0
KEY_SIZE_X = 25.0
KEY_SIZE_Y = 12.0

FILL_POS = np.array([-80.0, -60.0, 60.0])
FILL_ENERGY = 8000.0
FILL_SIZE = 80.0

STRIP_CENTER = (10.0, 8.0)
STRIP_LENGTH = 36.0
STRIP_WIDTH = 3.5
STRIP_ANGLE_DEG = 35.0

BASE_COLOR = (0.75, 0.75, 0.75, 1.0)
BASE_METALLIC = 0.7
BASE_ROUGHNESS = 0.30
POLISHED_METALLIC = 1.0
POLISHED_ROUGHNESS = 0.05

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def apply_flat_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = False


def apply_smooth_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = True


def create_plate_material_with_polished_strip(name):
    """Create the final material and its spatial BRDF mask.

    Blender MixRGB semantics are Color1 at Fac=0 and Color2 at Fac=1.
    Therefore Color1 is deliberately the outside/base value and Color2 is
    deliberately the inside/polished value.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = "PlateSurfaceOutput"
    output.location = (1100, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "FinalPlatePrincipledBSDF"
    bsdf.location = (850, 0)
    bsdf.inputs["Base Color"].default_value = BASE_COLOR

    geo = nodes.new("ShaderNodeNewGeometry")
    geo.name = "PlateLocalPosition"
    geo.location = (-700, 0)

    sub = nodes.new("ShaderNodeVectorMath")
    sub.name = "SubtractStripCenter"
    sub.location = (-520, 0)
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (STRIP_CENTER[0], STRIP_CENTER[1], 0.0)

    vrot = nodes.new("ShaderNodeVectorRotate")
    vrot.name = "RotateIntoStripFrame"
    vrot.location = (-320, 0)
    vrot.rotation_type = "Z_AXIS"
    vrot.inputs["Angle"].default_value = np.deg2rad(-STRIP_ANGLE_DEG)

    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.name = "StripFrameCoordinates"
    sep.location = (-100, 0)

    abs_len = nodes.new("ShaderNodeMath")
    abs_len.name = "AbsoluteStripLengthCoordinate"
    abs_len.location = (80, 160)
    abs_len.operation = "ABSOLUTE"
    cmp_len = nodes.new("ShaderNodeMath")
    cmp_len.name = "InsideStripLength"
    cmp_len.location = (250, 160)
    cmp_len.operation = "LESS_THAN"
    cmp_len.inputs[1].default_value = STRIP_LENGTH / 2.0

    abs_wid = nodes.new("ShaderNodeMath")
    abs_wid.name = "AbsoluteStripWidthCoordinate"
    abs_wid.location = (80, -40)
    abs_wid.operation = "ABSOLUTE"
    cmp_wid = nodes.new("ShaderNodeMath")
    cmp_wid.name = "InsideStripWidth"
    cmp_wid.location = (250, -40)
    cmp_wid.operation = "LESS_THAN"
    cmp_wid.inputs[1].default_value = STRIP_WIDTH / 2.0

    mask_and = nodes.new("ShaderNodeMath")
    mask_and.name = "StripMask"
    mask_and.label = "1 = INSIDE polished strip"
    mask_and.location = (420, 60)
    mask_and.operation = "MULTIPLY"

    # Critical ordering: Color1 is selected outside the mask and Color2 is
    # selected inside the mask (MixRGB: Fac=0 -> Color1, Fac=1 -> Color2).
    mix_r = nodes.new("ShaderNodeMixRGB")
    mix_r.name = "OutsideBase_InsidePolished_Roughness"
    mix_r.label = "outside .30 / inside .05"
    mix_r.location = (610, 170)
    mix_r.blend_type = "MIX"
    mix_r.inputs["Color1"].default_value = (BASE_ROUGHNESS,) * 3 + (1.0,)
    mix_r.inputs["Color2"].default_value = (POLISHED_ROUGHNESS,) * 3 + (1.0,)

    mix_m = nodes.new("ShaderNodeMixRGB")
    mix_m.name = "OutsideBase_InsidePolished_Metallic"
    mix_m.label = "outside .70 / inside 1.00"
    mix_m.location = (610, -10)
    mix_m.blend_type = "MIX"
    mix_m.inputs["Color1"].default_value = (BASE_METALLIC,) * 3 + (1.0,)
    mix_m.inputs["Color2"].default_value = (POLISHED_METALLIC,) * 3 + (1.0,)

    mix_color = nodes.new("ShaderNodeMixRGB")
    mix_color.name = "NeutralBaseColor"
    mix_color.label = "same neutral gray in both final regions"
    mix_color.location = (610, -190)
    mix_color.blend_type = "MIX"
    mix_color.inputs["Color1"].default_value = BASE_COLOR
    mix_color.inputs["Color2"].default_value = BASE_COLOR

    links.new(geo.outputs["Position"], sub.inputs[0])
    links.new(sub.outputs[0], vrot.inputs["Vector"])
    links.new(vrot.outputs[0], sep.inputs["Vector"])
    links.new(sep.outputs["X"], abs_len.inputs[0])
    links.new(sep.outputs["Y"], abs_wid.inputs[0])
    links.new(abs_len.outputs[0], cmp_len.inputs[0])
    links.new(abs_wid.outputs[0], cmp_wid.inputs[0])
    links.new(cmp_len.outputs[0], mask_and.inputs[0])
    links.new(cmp_wid.outputs[0], mask_and.inputs[1])
    links.new(mask_and.outputs[0], mix_r.inputs["Fac"])
    links.new(mask_and.outputs[0], mix_m.inputs["Fac"])
    links.new(mask_and.outputs[0], mix_color.inputs["Fac"])
    links.new(mix_r.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(mix_m.outputs["Color"], bsdf.inputs["Metallic"])
    links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])
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

    device_summary = []
    gpu_enabled = False
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # OptiX is preferred for the RTX 3070 Ti, with CUDA as the fallback.
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


def move_key_light(light, position, target):
    light.set_location(position)
    orient_area_toward(light.blender_obj, target)


def save_frame(output_path):
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(output_path), rgb)
    print(f"  Saved: {output_path}")


def mask_value(x, y):
    dx = x - STRIP_CENTER[0]
    dy = y - STRIP_CENTER[1]
    angle = math.radians(-STRIP_ANGLE_DEG)
    u = math.cos(angle) * dx - math.sin(angle) * dy
    v = math.sin(angle) * dx + math.cos(angle) * dy
    return int(abs(u) < STRIP_LENGTH / 2.0 and abs(v) < STRIP_WIDTH / 2.0)


def render_debug_mask(plate_mat):
    """Temporarily replace the final surface with a red/blue mask diagnostic."""
    nodes = plate_mat.node_tree.nodes
    links = plate_mat.node_tree.links
    output = nodes["PlateSurfaceOutput"]
    mix_color = nodes["NeutralBaseColor"]
    bsdf = nodes["FinalPlatePrincipledBSDF"]
    debug_emission = nodes.new("ShaderNodeEmission")
    debug_emission.name = "TEMPORARY_DEBUG_MASK_EMISSION"
    debug_emission.inputs["Strength"].default_value = 1.0
    mix_color.inputs["Color1"].default_value = (0.02, 0.08, 0.8, 1.0)  # outside
    mix_color.inputs["Color2"].default_value = (0.85, 0.02, 0.02, 1.0)  # inside
    links.new(mix_color.outputs["Color"], debug_emission.inputs["Color"])
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(debug_emission.outputs["Emission"], output.inputs["Surface"])
    save_frame(DEBUG_OUTPUT)

    # Restore the final, neutral-gray BRDF before either controlled condition.
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    mix_color.inputs["Color1"].default_value = BASE_COLOR
    mix_color.inputs["Color2"].default_value = BASE_COLOR
    nodes.remove(debug_emission)
    print(
        "Debug mask samples: "
        f"outside(0,-20)={mask_value(0, -20)}, "
        f"inside(center)={mask_value(*STRIP_CENTER)}, "
        f"outside(end+width)={mask_value(STRIP_CENTER[0] + STRIP_LENGTH / 2 + 1, STRIP_CENTER[1])}"
    )


def projected_strip_roi(scene, plate_obj, padding=18):
    """Return an expanded image ROI around the projected material-only strip."""
    top_z = max(float(v[2]) for v in plate_obj.bound_box) + 0.02
    center = np.array([STRIP_CENTER[0], STRIP_CENTER[1]])
    along = np.array([math.cos(math.radians(STRIP_ANGLE_DEG)), math.sin(math.radians(STRIP_ANGLE_DEG))])
    across = np.array([-along[1], along[0]])
    corners_2d = [
        center + sign_a * along * STRIP_LENGTH / 2 + sign_b * across * STRIP_WIDTH / 2
        for sign_a in (-1, 1)
        for sign_b in (-1, 1)
    ]
    camera = scene.camera
    projected = []
    for point in corners_2d:
        world = plate_obj.matrix_world @ mathutils.Vector((float(point[0]), float(point[1]), top_z))
        co = world_to_camera_view(scene, camera, world)
        projected.append((float(co.x * IMAGE_WIDTH), float((1.0 - co.y) * IMAGE_HEIGHT)))
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    x0 = max(0, int(math.floor(min(xs) - padding)))
    x1 = min(IMAGE_WIDTH, int(math.ceil(max(xs) + padding + 1)))
    y0 = max(0, int(math.floor(min(ys) - padding)))
    y1 = min(IMAGE_HEIGHT, int(math.ceil(max(ys) + padding + 1)))
    return (x0, y0, x1, y1), projected


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def roi_metrics(img_default, img_side, roi):
    x0, y0, x1, y1 = roi
    a = img_default[y0:y1, x0:x1, :3].astype(np.float32)
    b = img_side[y0:y1, x0:x1, :3].astype(np.float32)
    la = luminance(a)
    lb = luminance(b)
    diff = np.abs(a - b).mean(axis=2)
    ldiff = np.abs(la - lb)
    strong = ldiff >= 20.0

    def summary(values):
        return {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        }

    def bright_centroid(values):
        threshold = np.percentile(values, 98)
        ys, xs = np.where(values >= threshold)
        if len(xs) == 0:
            return None
        return {
            "x": float(x0 + xs.mean()),
            "y": float(y0 + ys.mean()),
            "threshold": float(threshold),
            "pixels": int(len(xs)),
        }

    centroid_a = bright_centroid(la)
    centroid_b = bright_centroid(lb)
    displacement = None
    if centroid_a and centroid_b:
        displacement = float(math.hypot(centroid_a["x"] - centroid_b["x"], centroid_a["y"] - centroid_b["y"]))

    return {
        "roi_xyxy_pixels": [int(x0), int(y0), int(x1), int(y1)],
        "roi_size_pixels": [int(x1 - x0), int(y1 - y0)],
        "default_local_intensity_contrast_p95_minus_p50": float(np.percentile(la, 95) - np.percentile(la, 50)),
        "side_local_intensity_contrast_p95_minus_p50": float(np.percentile(lb, 95) - np.percentile(lb, 50)),
        "default_luminance": summary(la),
        "side_luminance": summary(lb),
        "absolute_rgb_difference": summary(diff),
        "absolute_luminance_difference": summary(ldiff),
        "strong_change_threshold_luminance": 20.0,
        "strongly_changed_pixels": int(strong.sum()),
        "roi_pixels": int(strong.size),
        "strongly_changed_fraction": float(strong.mean()),
        "brightest_2_percent_centroid_default": centroid_a,
        "brightest_2_percent_centroid_side": centroid_b,
        "brightest_response_centroid_displacement_pixels": displacement,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    gpu_enabled, device_summary = configure_cycles(scene)
    configure_color_management(scene)

    # Geometry: the original STL objects remain separate and unchanged.
    try:
        bpy.ops.wm.stl_import(filepath=str(PLATE_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(PLATE_PATH))
    plate_obj = bpy.context.active_object
    plate_obj.name = "Plate"
    try:
        bpy.ops.wm.stl_import(filepath=str(BOLT_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(BOLT_PATH))
    bolt_obj = bpy.context.active_object
    bolt_obj.name = "Bolt"
    apply_flat_shading(plate_obj)
    apply_smooth_shading(bolt_obj)
    print(f"Plate dimensions: {tuple(round(v, 3) for v in plate_obj.dimensions)}")
    print(f"Plate location: {tuple(round(v, 3) for v in plate_obj.location)}")
    print(f"Bolt dimensions: {tuple(round(v, 3) for v in bolt_obj.dimensions)}")

    # Final material: one neutral base colour, with only roughness/metallic
    # changing spatially through the procedural mask.
    plate_mat = create_plate_material_with_polished_strip("PlatePBRMetal")
    plate_obj.data.materials.append(plate_mat)
    bolt_mat = bpy.data.materials.new(name="BoltPBRMetal")
    bolt_mat.use_nodes = True
    bolt_bsdf = bolt_mat.node_tree.nodes["Principled BSDF"]
    bolt_bsdf.inputs["Base Color"].default_value = BASE_COLOR
    bolt_bsdf.inputs["Metallic"].default_value = BASE_METALLIC
    bolt_bsdf.inputs["Roughness"].default_value = BASE_ROUGHNESS
    bolt_obj.data.materials.append(bolt_mat)

    # Fixed camera and explicit intrinsics.
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)
    scene.camera.data.lens = 50.0
    scene.camera.data.sensor_width = 36.0

    # Shared fill and world. Neither is touched after final material restore.
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

    # One key AREA light object is reused for both conditions.
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_energy(KEY_ENERGY)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))
    set_area_dimensions(key_light.blender_obj, KEY_SIZE_X, KEY_SIZE_Y)

    # Debug first, then restore the final material before A/B.
    print("Rendering temporary spatial-mask diagnostic ...")
    move_key_light(key_light, DEFAULT_KEY_POS, TARGET_CENTER)
    render_debug_mask(plate_mat)

    roi, projected_corners = projected_strip_roi(scene, plate_obj)
    print("Projected strip corners: " + json.dumps(projected_corners))
    print("Expanded strip ROI: " + str(roi))

    # Final condition A.
    print("Rendering A - default illumination ...")
    move_key_light(key_light, DEFAULT_KEY_POS, TARGET_CENTER)
    save_frame(DEFAULT_OUTPUT)

    # Final condition B. Between these two calls, ONLY this key light
    # transform changes: position and the corresponding orientation.
    print("Rendering B - side illumination ...")
    move_key_light(key_light, SIDE_KEY_POS, TARGET_CENTER)
    save_frame(SIDE_OUTPUT)

    img_default = imageio.imread(str(DEFAULT_OUTPUT))
    img_side = imageio.imread(str(SIDE_OUTPUT))
    metrics = roi_metrics(img_default, img_side, roi)
    metrics.update(
        {
            "strip": {
                "center_mm": list(STRIP_CENTER),
                "length_mm": STRIP_LENGTH,
                "width_mm": STRIP_WIDTH,
                "angle_degrees": STRIP_ANGLE_DEG,
                "outside": {"metallic": BASE_METALLIC, "roughness": BASE_ROUGHNESS},
                "inside": {"metallic": POLISHED_METALLIC, "roughness": POLISHED_ROUGHNESS},
            },
            "key_light": {
                "type": "AREA",
                "energy": KEY_ENERGY,
                "color": [1.0, 0.95, 0.9],
                "size_x_mm": KEY_SIZE_X,
                "size_y_mm": KEY_SIZE_Y,
                "default_position": DEFAULT_KEY_POS.tolist(),
                "side_position": SIDE_KEY_POS.tolist(),
                "target": TARGET_CENTER.tolist(),
            },
            "held_constant": [
                "plate.stl and bolt.stl geometry",
                "procedural strip mask and final material nodes",
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
    )
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("=== PBR reflection experiment complete ===")
    print(f"Final mask samples: outside={mask_value(0, -20)}, inside={mask_value(*STRIP_CENTER)}")
    print(f"Cycles: {scene.cycles.samples} samples, adaptive={scene.cycles.use_adaptive_sampling}, denoiser={scene.cycles.denoiser}, device={scene.cycles.device}")
    print(f"Strip BRDF: outside metallic={BASE_METALLIC}, roughness={BASE_ROUGHNESS}; inside metallic={POLISHED_METALLIC}, roughness={POLISHED_ROUGHNESS}")
    print(f"Key AREA: {KEY_SIZE_X}x{KEY_SIZE_Y} mm, energy={KEY_ENERGY}, A={DEFAULT_KEY_POS.tolist()}, B={SIDE_KEY_POS.tolist()}")
    print("ROI metrics: " + json.dumps(metrics, indent=2))
    print("Outputs:")
    print(f"  {DEBUG_OUTPUT}")
    print(f"  {DEFAULT_OUTPUT}")
    print(f"  {SIDE_OUTPUT}")
    print(f"  {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()

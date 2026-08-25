import blenderproc as bproc
from pathlib import Path

import bpy
import imageio.v2 as imageio
import mathutils
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_PATH = REPO_ROOT / "outputs" / "plate.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
OUTPUT_DIR = REPO_ROOT / "renders" / "pbr_reflection"
DEFAULT_OUTPUT = OUTPUT_DIR / "default_light.png"
SIDE_OUTPUT = OUTPUT_DIR / "side_light.png"

DEFAULT_KEY_POS = np.array([100.0, -100.0, 120.0])
SIDE_KEY_POS = np.array([160.0, 20.0, 70.0])
TARGET_CENTER = np.array([0.0, 0.0, 3.0])

STRIP_CENTER = (10.0, 8.0)
STRIP_LENGTH = 35.0
STRIP_WIDTH = 2.0
STRIP_ANGLE_DEG = 35.0

BASE_METALLIC = 0.7
BASE_ROUGHNESS = 0.30
POLISHED_METALLIC = 1.0
POLISHED_ROUGHNESS = 0.05


def apply_flat_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = False


def apply_smooth_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = True


def create_plate_material_with_polished_strip(name):
    """Procedural Principled BSDF with a low-roughness diagonal strip.
    No extra geometry. Same base colour. Lower roughness in the strip.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1200, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (900, 0)
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)

    geo = nodes.new("ShaderNodeNewGeometry")
    geo.location = (-600, 0)

    sub = nodes.new("ShaderNodeVectorMath")
    sub.location = (-400, 0)
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (STRIP_CENTER[0], STRIP_CENTER[1], 0.0)

    vrot = nodes.new("ShaderNodeVectorRotate")
    vrot.location = (-200, 0)
    vrot.rotation_type = "Z_AXIS"
    vrot.inputs["Angle"].default_value = np.deg2rad(-STRIP_ANGLE_DEG)

    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (0, 0)

    abs_len = nodes.new("ShaderNodeMath")
    abs_len.location = (150, 150)
    abs_len.operation = "ABSOLUTE"
    cmp_len = nodes.new("ShaderNodeMath")
    cmp_len.location = (300, 150)
    cmp_len.operation = "LESS_THAN"
    cmp_len.inputs[1].default_value = STRIP_LENGTH / 2.0

    abs_wid = nodes.new("ShaderNodeMath")
    abs_wid.location = (150, -50)
    abs_wid.operation = "ABSOLUTE"
    cmp_wid = nodes.new("ShaderNodeMath")
    cmp_wid.location = (300, -50)
    cmp_wid.operation = "LESS_THAN"
    cmp_wid.inputs[1].default_value = STRIP_WIDTH / 2.0

    mask_and = nodes.new("ShaderNodeMath")
    mask_and.location = (450, 50)
    mask_and.operation = "MULTIPLY"

    mix_r = nodes.new("ShaderNodeMixRGB")
    mix_r.location = (650, 150)
    mix_r.blend_type = "MIX"
    mix_r.inputs["Color1"].default_value = (POLISHED_ROUGHNESS,) * 3 + (1.0,)
    mix_r.inputs["Color2"].default_value = (BASE_ROUGHNESS,) * 3 + (1.0,)

    mix_m = nodes.new("ShaderNodeMixRGB")
    mix_m.location = (650, -50)
    mix_m.blend_type = "MIX"
    mix_m.inputs["Color1"].default_value = (POLISHED_METALLIC,) * 3 + (1.0,)
    mix_m.inputs["Color2"].default_value = (BASE_METALLIC,) * 3 + (1.0,)

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
    links.new(mix_r.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(mix_m.outputs["Color"], bsdf.inputs["Metallic"])
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
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for dev in prefs.devices:
            if dev.type in ("OPTIX", "CUDA"):
                dev.use = True
    except Exception:
        pass


def configure_color_management(scene):
    view = scene.view_settings
    view.view_transform = "AgX"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_mode = "RGB"


def orient_area_toward(light_obj, target):
    direction = target - np.array(light_obj.location)
    direction = direction / np.linalg.norm(direction)
    rot_3x3 = bproc.camera.rotation_from_forward_vec(direction)
    mat_4x4 = np.eye(4)
    mat_4x4[:3, :3] = rot_3x3
    mat_4x4[:3, 3] = light_obj.location
    light_obj.matrix_world = mathutils.Matrix(mat_4x4.tolist())


def move_key_light(light, position, target):
    light.set_location(position)
    orient_area_toward(light.blender_obj, target)


def save_frame(output_path):
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(output_path), rgb)
    print(f"  Saved: {output_path}")


def validate_roi(img_default, img_side):
    h, w = img_default.shape[:2]
    cx, cy = w // 2, h // 2
    roi_size = 60
    strip = img_default[cy-roi_size//2:cy+roi_size//2, cx-5:cx+5, :]
    ref = img_default[cy-roi_size//2:cy+roi_size//2, cx-roi_size//2-10:cx-roi_size//2-4, :]
    if strip.size == 0 or ref.size == 0:
        return 0.0, 0.0
    c_def = float(strip.mean()) - float(ref.mean())
    strip = img_side[cy-roi_size//2:cy+roi_size//2, cx-5:cx+5, :]
    ref = img_side[cy-roi_size//2:cy+roi_size//2, cx-roi_size//2-10:cx-roi_size//2-4, :]
    c_side = float(strip.mean()) - float(ref.mean())
    return c_def, c_side
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()
    scene = bpy.context.scene
    configure_cycles(scene)
    configure_color_management(scene)

    # --- Geometry ---
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

    # --- Shading ---
    apply_flat_shading(plate_obj)
    apply_smooth_shading(bolt_obj)

    # --- Materials ---
    plate_mat = create_plate_material_with_polished_strip("PlatePBRMetal")
    plate_obj.data.materials.append(plate_mat)
    bolt_mat = bpy.data.materials.new(name="BoltPBRMetal")
    bolt_mat.use_nodes = True
    bsd = bolt_mat.node_tree.nodes["Principled BSDF"]
    bsd.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    bsd.inputs["Metallic"].default_value = BASE_METALLIC
    bsd.inputs["Roughness"].default_value = BASE_ROUGHNESS
    bolt_obj.data.materials.append(bolt_mat)

    # --- Camera ---
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # --- Shared lighting ---
    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location([-80.0, -60.0, 60.0])
    fill_light.set_energy(15000)
    fill_light.set_color(np.array([0.9, 0.95, 1.0]))
    orient_area_toward(fill_light.blender_obj, TARGET_CENTER)
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.3
    bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_energy(50000)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))

    # --- Condition A ---
    print("Rendering A - default illumination ...")
    move_key_light(key_light, DEFAULT_KEY_POS, TARGET_CENTER)
    save_frame(DEFAULT_OUTPUT)

    # --- Condition B ---
    print("Rendering B - side illumination ...")
    move_key_light(key_light, SIDE_KEY_POS, TARGET_CENTER)
    save_frame(SIDE_OUTPUT)

    # --- ROI validation ---
    img_default = imageio.imread(str(DEFAULT_OUTPUT))
    img_side = imageio.imread(str(SIDE_OUTPUT))
    c_def, c_side = validate_roi(img_default, img_side)
    diff_img = np.abs(img_default.astype(float) - img_side.astype(float))
    gpu_used = any(d.use for d in bpy.context.preferences.addons[
        "cycles"].preferences.devices if d.type in ("OPTIX", "CUDA"))

    print()
    print("=== PBR reflection experiment complete ===")
    print(f"Cycles samples = {scene.cycles.samples}, denoiser = {scene.cycles.denoiser}")
    print(f"GPU used: {gpu_used}   View transform: {scene.view_settings.view_transform}")
    print()
    print("Held constant: geometry, camera, fill, world, material (same strip)")
    print(f"Strip: {STRIP_LENGTH}x{STRIP_WIDTH}mm at {STRIP_CENTER}, {STRIP_ANGLE_DEG}deg")
    print(f"Polished: metallic={POLISHED_METALLIC}, roughness={POLISHED_ROUGHNESS}")
    print(f"Base    : metallic={BASE_METALLIC}, roughness={BASE_ROUGHNESS}")
    print()
    print(f"Key: default={DEFAULT_KEY_POS.tolist()}  side={SIDE_KEY_POS.tolist()}")
    print(f"C_default = {c_def:.1f}   C_side = {c_side:.1f}")
    if abs(c_def) > 0.5:
        print(f"|C_side/C_default| = {abs(c_side/c_def):.2f}")
    print(f"Max pixel diff = {diff_img.max():.0f}/255")
    print(f"Mean pixel diff = {diff_img.mean():.1f}/255")
    print()
    print("Outputs:", DEFAULT_OUTPUT, SIDE_OUTPUT, sep="\n  ")


if __name__ == "__main__":
    main()
import blenderproc as bproc
from pathlib import Path

import bpy
import imageio.v2 as imageio
import mathutils
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_PATH = REPO_ROOT / "outputs" / "plate.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
INTERVENTION_DIR = REPO_ROOT / "renders" / "intervention"
DEFAULT_OUTPUT = INTERVENTION_DIR / "default_light.png"
SIDE_OUTPUT = INTERVENTION_DIR / "side_light.png"

DEFAULT_KEY_POS = np.array([100.0, -100.0, 120.0])
SIDE_KEY_POS = np.array([160.0, 20.0, 70.0])
TARGET_CENTER = np.array([0.0, 0.0, 3.0])


def apply_flat_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = False


def apply_smooth_shading(obj):
    for face in obj.data.polygons:
        face.use_smooth = True


def create_metal_material(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat


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


def main():
    INTERVENTION_DIR.mkdir(parents=True, exist_ok=True)
    bproc.init()

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

    # --- Material ---
    plate_obj.data.materials.append(create_metal_material("PlateMetal"))
    bolt_obj.data.materials.append(create_metal_material("BoltMetal"))

    # --- Camera ---
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # --- Shared fill light ---
    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location([-80.0, -60.0, 60.0])
    fill_light.set_energy(15000)
    fill_light.set_color(np.array([0.9, 0.95, 1.0]))
    orient_area_toward(fill_light.blender_obj, TARGET_CENTER)

    # --- Shared world background ---
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.3
    bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)

    # --- Key light (created once, repositioned between renders) ---
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_energy(50000)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))

    # --- Condition A: default illumination ---
    print("Rendering condition A \u2014 default illumination ...")
    move_key_light(key_light, DEFAULT_KEY_POS, TARGET_CENTER)
    save_frame(DEFAULT_OUTPUT)

    # --- Condition B: side illumination ---
    print("Rendering condition B \u2014 side illumination ...")
    move_key_light(key_light, SIDE_KEY_POS, TARGET_CENTER)
    save_frame(SIDE_OUTPUT)

    # --- Summary ---
    print()
    print("=== Illumination intervention complete ===")
    print("Held constant across both renders:")
    print("  Geometry           : outputs/plate.stl + outputs/bolt.stl")
    print("  Plate shading      : FLAT")
    print("  Bolt shading       : SMOOTH")
    print("  Material           : metallic=0.7, roughness=0.3, Base=(0.75, 0.75, 0.75)")
    print("  Camera             : pos=[140, -140, 100], look=[0, 0, 10]")
    print("  Resolution         : 640 x 480")
    print("  Fill light  (AREA) : pos=(-80, -60, 60), energy=15000, cool")
    print("  World background   : Strength=0.3, Color=(0.85, 0.85, 0.85)")
    print()
    print("Only the key-light position/direction changed:")
    print(f"  default_light key pos: {DEFAULT_KEY_POS.tolist()}")
    print(f"  side_light   key pos: {SIDE_KEY_POS.tolist()}")
    print()
    print("Outputs:")
    print(f"  {DEFAULT_OUTPUT}")
    print(f"  {SIDE_OUTPUT}")


if __name__ == "__main__":
    main()
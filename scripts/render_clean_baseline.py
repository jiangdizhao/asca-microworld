import blenderproc as bproc
from pathlib import Path

import bpy
import imageio.v2 as imageio
import mathutils
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATE_PATH = REPO_ROOT / "outputs" / "plate.stl"
BOLT_PATH = REPO_ROOT / "outputs" / "bolt.stl"
RENDER_DIR = REPO_ROOT / "renders"
OUTPUT_PATH = RENDER_DIR / "baseline_clean_rgb.png"


def apply_flat_shading(obj: bpy.types.Object) -> None:
    """Disable smooth shading on every face — each triangle gets a constant normal."""
    for face in obj.data.polygons:
        face.use_smooth = False


def apply_smooth_shading(obj: bpy.types.Object) -> None:
    """Enable smooth shading on every face — normals interpolate across triangles."""
    for face in obj.data.polygons:
        face.use_smooth = True


def create_metal_material(name: str) -> bpy.types.Material:
    """Create a reproducible light-gray metallic Principled BSDF material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat


def orient_area_toward(light_obj: bpy.types.Object, target: np.ndarray) -> None:
    """Rotate an AREA light so its -Z emission axis points toward *target*."""
    direction = target - np.array(light_obj.location)
    direction = direction / np.linalg.norm(direction)
    rot_3x3 = bproc.camera.rotation_from_forward_vec(direction)
    # Build full 4x4 world matrix preserving the object's location
    mat_4x4 = np.eye(4)
    mat_4x4[:3, :3] = rot_3x3
    mat_4x4[:3, 3] = light_obj.location
    light_obj.matrix_world = mathutils.Matrix(mat_4x4.tolist())


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    bproc.init()

    # ------------------------------------------------------------------
    # 1. Geometry — load plate and bolt as separate objects
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Shading control — critical fix for triangular artifacts
    # ------------------------------------------------------------------
    # Plate: flat shading — every triangle face gets a constant normal.
    # This eliminates the fan-shaped interpolation artifacts on the flat surface.
    apply_flat_shading(plate_obj)

    # Bolt: smooth shading — normals interpolate across faces so the hex
    # head and cylindrical shank read as smooth curved geometry.
    apply_smooth_shading(bolt_obj)

    # ------------------------------------------------------------------
    # 3. Material — physically plausible light-gray metal (same for both)
    # ------------------------------------------------------------------
    mat_plate = create_metal_material("PlateMetal")
    mat_bolt = create_metal_material("BoltMetal")
    plate_obj.data.materials.append(mat_plate)
    bolt_obj.data.materials.append(mat_bolt)

    # ------------------------------------------------------------------
    # 4. Camera — fixed three-quarter view (unchanged from baseline)
    # ------------------------------------------------------------------
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # ------------------------------------------------------------------
    # 5. Lighting — broad AREA lights instead of harsh point lights
    # ------------------------------------------------------------------
    target_center = np.array([0.0, 0.0, 3.0])  # centre of plate top surface

    # 5a. Key light: warm broad AREA from upper-right-front
    key_pos = np.array([100.0, -100.0, 120.0])
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_location(key_pos)
    key_light.set_energy(50000)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))
    orient_area_toward(key_light.blender_obj, target_center)

    # 5b. Fill light: cooler, weaker AREA from the opposite side
    fill_pos = np.array([-80.0, -60.0, 60.0])
    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location(fill_pos)
    fill_light.set_energy(15000)
    fill_light.set_color(np.array([0.9, 0.95, 1.0]))
    orient_area_toward(fill_light.blender_obj, target_center)

    # 5c. World illumination — mild neutral ambient
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.3
    bg.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)

    # ------------------------------------------------------------------
    # 6. Render
    # ------------------------------------------------------------------
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(OUTPUT_PATH), rgb)

    print("Saved clean baseline RGB render to: " + str(OUTPUT_PATH))
    print("Plate STL    : " + str(PLATE_PATH))
    print("Bolt STL     : " + str(BOLT_PATH))
    print("Resolution   : 640 x 480")
    print("Camera pos   : " + str(cam_location.tolist()))
    print("Camera look  : " + str(look_at.tolist()))
    print("Plate shading: FLAT (face.use_smooth = False)")
    print("Bolt shading : SMOOTH (face.use_smooth = True)")
    print("Material     : metallic=0.7, roughness=0.3, Base Color=(0.75, 0.75, 0.75)")
    print("Key light    : AREA, pos=(100, -100, 120), energy=50000, warm, oriented to target")
    print("Fill light   : AREA, pos=(-80, -60, 60), energy=15000, cool, oriented to target")
    print("World illum  : Strength=0.3, Color=(0.85, 0.85, 0.85)")


if __name__ == "__main__":
    main()
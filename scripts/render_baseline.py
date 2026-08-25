import blenderproc as bproc
from pathlib import Path

import bpy
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "outputs" / "plate_with_bolt.stl"
RENDER_DIR = REPO_ROOT / "renders"
OUTPUT_PATH = RENDER_DIR / "baseline_rgb.png"


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    bproc.init()

    # ------------------------------------------------------------------
    # 1. Geometry  (unchanged CadQuery export)
    # ------------------------------------------------------------------
    try:
        bpy.ops.wm.stl_import(filepath=str(MODEL_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(MODEL_PATH))
    obj = bpy.context.active_object

    # ------------------------------------------------------------------
    # 2. Material  — physically plausible light-gray metal
    # ------------------------------------------------------------------
    mat = bpy.data.materials.new(name="BaselineMetal")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.7
    bsdf.inputs["Roughness"].default_value = 0.3

    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # ------------------------------------------------------------------
    # 3. Camera  — three-quarter view
    # ------------------------------------------------------------------
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # ------------------------------------------------------------------
    # 4. Lighting  — explicitly controlled
    # ------------------------------------------------------------------
    # 4a. Key light: warm point light from upper-right-front
    key_light = bproc.types.Light()
    key_light.set_type("POINT")
    key_light.set_location([100.0, -100.0, 120.0])
    key_light.set_energy(50000)
    key_light.set_color(np.array([1.0, 0.95, 0.9]))

    # 4b. Fill light: cooler, weaker, from roughly opposite side
    fill_light = bproc.types.Light()
    fill_light.set_type("POINT")
    fill_light.set_location([-80.0, -60.0, 60.0])
    fill_light.set_energy(20000)
    fill_light.set_color(np.array([0.9, 0.95, 1.0]))

    # 4c. Back rim light to separate object from background
    rim_light = bproc.types.Light()
    rim_light.set_type("POINT")
    rim_light.set_location([0.0, 120.0, 50.0])
    rim_light.set_energy(10000)
    rim_light.set_color(np.array([1.0, 1.0, 1.0]))

    # 4d. World illumination (ambient) — mild neutral gray
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.3
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    # ------------------------------------------------------------------
    # 5. Render  — baseline RGB
    # ------------------------------------------------------------------
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(OUTPUT_PATH), rgb)

    print("Saved baseline RGB render to: " + str(OUTPUT_PATH))
    print("Geometry       : " + str(MODEL_PATH))
    print("Resolution     : 640 x 480")
    print("Camera position: " + str(cam_location.tolist()))
    print("Camera look-at : " + str(look_at.tolist()))
    print("Material       : metallic=0.7, roughness=0.3, Base Color=(0.75, 0.75, 0.75)")
    print("Key light      : POINT, pos=(100, -100, 120), energy=50000, warm")
    print("Fill light     : POINT, pos=(-80, -60, 60), energy=20000, cool")
    print("Rim light      : POINT, pos=(0, 120, 50), energy=10000, white")
    print("World illum    : Strength=0.3, Color=(0.8, 0.8, 0.8)")


if __name__ == "__main__":
    main()
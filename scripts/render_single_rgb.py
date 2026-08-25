import blenderproc as bproc
from pathlib import Path
import bpy
import numpy as np

import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "outputs" / "plate_with_bolt.stl"
RENDER_DIR = REPO_ROOT / "renders"
OUTPUT_PATH = RENDER_DIR / "plate_with_bolt_rgb.png"


def main():
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    bproc.init()

    # Import STL
    try:
        bpy.ops.wm.stl_import(filepath=str(MODEL_PATH))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(MODEL_PATH))

    # Camera
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])

    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # Light
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([120.0, -120.0, 180.0])
    light.set_energy(2000)

    # Render RGB
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(OUTPUT_PATH), rgb)

    print(f"Saved RGB image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
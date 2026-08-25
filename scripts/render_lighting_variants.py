import blenderproc as bproc
from pathlib import Path
import bpy
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "outputs" / "plate_with_bolt.stl"
RENDER_DIR = REPO_ROOT / "renders"
DEFAULT_OUTPUT = RENDER_DIR / "default_light.png"
SIDE_OUTPUT = RENDER_DIR / "side_light.png"


def import_stl(model_path: Path) -> None:
    """Import the combined plate-and-bolt STL into Blender."""
    try:
        bpy.ops.wm.stl_import(filepath=str(model_path))
    except Exception:
        bpy.ops.import_mesh.stl(filepath=str(model_path))


def save_render(output_path: Path) -> None:
    """Render the current scene and save the first RGB frame as PNG."""
    data = bproc.renderer.render()
    rgb = data["colors"][0]
    imageio.imwrite(str(output_path), rgb)


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    bproc.init()
    import_stl(MODEL_PATH)

    # Keep geometry and camera fixed for both renders.
    cam_location = np.array([140.0, -140.0, 100.0])
    look_at = np.array([0.0, 0.0, 10.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(look_at - cam_location)
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_resolution(640, 480)

    # Use one light and change only its position between the two renders.
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_energy(6000)

    # Default/front-ish illumination.
    light.set_location([120.0, -120.0, 180.0])
    save_render(DEFAULT_OUTPUT)

    # Side illumination: same object, same camera, only light position changes.
    light.set_location([180.0, 0.0, 90.0])
    save_render(SIDE_OUTPUT)

    print("Saved renders:")
    print(f"  - {DEFAULT_OUTPUT}")
    print(f"  - {SIDE_OUTPUT}")


if __name__ == "__main__":
    main()

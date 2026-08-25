"""Generate the three subtractive crack plates for the ambiguity calibration.

The validated plate and bolt outputs are not modified.  Each calibration
plate is made by the same CadQuery subtraction used by create_plate_crack.py;
only the top width and depth (and the corresponding small bottom taper) vary.
"""

from pathlib import Path

import cadquery as cq


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration_cracks"

PLATE_LENGTH = 100.0
PLATE_WIDTH = 70.0
PLATE_THICKNESS = 6.0
BOLT_CLEARANCE_DIAMETER = 9.0

CRACK_CENTER = (10.0, 8.0)
CRACK_LENGTH = 36.0
CRACK_ANGLE_DEG = 35.0
CUTTER_OVERTRAVEL = 0.20

CRACKS = {
    "C1": {"top_width_mm": 0.6, "depth_mm": 0.12, "bottom_width_mm": 0.05, "bottom_offset_mm": 0.10},
    "C2": {"top_width_mm": 0.8, "depth_mm": 0.18, "bottom_width_mm": 0.06, "bottom_offset_mm": 0.16},
    "C3": {"top_width_mm": 1.0, "depth_mm": 0.25, "bottom_width_mm": 0.08, "bottom_offset_mm": 0.22},
}


def build_original_plate() -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(PLATE_LENGTH, PLATE_WIDTH, PLATE_THICKNESS, centered=(True, True, False))
        .faces(">Z")
        .workplane()
        .hole(BOLT_CLEARANCE_DIAMETER)
    )


def build_crack_cutter(parameters: dict) -> cq.Workplane:
    width = parameters["top_width_mm"]
    depth = parameters["depth_mm"]
    bottom_width = parameters["bottom_width_mm"]
    bottom_offset = parameters["bottom_offset_mm"]
    cross_section = [
        (-width / 2.0, CUTTER_OVERTRAVEL),
        (width / 2.0, CUTTER_OVERTRAVEL),
        (bottom_offset + bottom_width / 2.0, -depth),
        (bottom_offset - bottom_width / 2.0, -depth),
    ]
    return (
        cq.Workplane("YZ")
        .polyline(cross_section)
        .close()
        .extrude(CRACK_LENGTH / 2.0, both=True)
        .rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), CRACK_ANGLE_DEG)
        .translate((CRACK_CENTER[0], CRACK_CENTER[1], PLATE_THICKNESS))
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "construction": "CadQuery original plate minus one rotated shallow asymmetric tapered groove cutter",
        "subtractive_geometry": True,
        "shared": {
            "plate_length_mm": PLATE_LENGTH,
            "plate_width_mm": PLATE_WIDTH,
            "plate_thickness_mm": PLATE_THICKNESS,
            "center_mm": list(CRACK_CENTER),
            "length_mm": CRACK_LENGTH,
            "angle_degrees": CRACK_ANGLE_DEG,
        },
        "candidates": {},
    }

    for name, parameters in CRACKS.items():
        cracked_plate = build_original_plate().cut(build_crack_cutter(parameters))
        solid_count = len(cracked_plate.solids().vals())
        if solid_count != 1 or not cracked_plate.val().isValid():
            raise RuntimeError(f"Invalid {name} result: solids={solid_count}, valid={cracked_plate.val().isValid()}")

        stl_path = OUTPUT_DIR / f"{name}.stl"
        step_path = OUTPUT_DIR / f"{name}.step"
        cq.exporters.export(cracked_plate, str(step_path))
        cq.exporters.export(cracked_plate, str(stl_path))
        manifest["candidates"][name] = {"stl": str(stl_path), "step": str(step_path), **parameters}
        print(f"Created {name}: {stl_path}")
        print(f"  top width/depth: {parameters['top_width_mm']} / {parameters['depth_mm']} mm")
        print(f"  bottom width/offset: {parameters['bottom_width_mm']} / {parameters['bottom_offset_mm']} mm")
        print("  CadQuery validity: valid=True, solids=1")

    manifest_path = OUTPUT_DIR / "geometry_manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download YCB meshes and convert them into CoppeliaSim .ttm models.

This creates a unified object set under:
  external/RLBench/rlbench/assets/

Example:
  /home/ngocbach@ad.asu.edu/anaconda3/envs/rlbench_gui/bin/python \
    scripts/fetch_ycb_to_ttm.py --objects 011_banana 013_apple 025_mug
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List

from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.shape import Shape


YCB_BUCKET = "https://ycb-benchmarks.s3.amazonaws.com"
DEFAULT_OBJECTS = [
    "011_banana",
    "012_strawberry",
    "013_apple",
    "017_orange",
    "024_bowl",
    "025_mug",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _raw_root(root: Path) -> Path:
    return root / "external" / "RLBench" / "rlbench" / "assets" / "ycb_raw"


def _assets_root(root: Path) -> Path:
    return root / "external" / "RLBench" / "rlbench" / "assets"


def _task_design_ttt(root: Path) -> Path:
    return root / "external" / "RLBench" / "rlbench" / "task_design.ttt"


def _download_tgz(object_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz = out_dir / f"{object_id}_google_16k.tgz"
    if tgz.exists():
        return tgz
    url = f"{YCB_BUCKET}/data/google/{object_id}_google_16k.tgz"
    print(f"[download] {object_id} <- {url}")
    urllib.request.urlretrieve(url, tgz)
    return tgz


def _extract_tgz(tgz: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(path=out_dir)
    obj_name = tgz.name.replace("_google_16k.tgz", "")
    return out_dir / obj_name / "google_16k"


def _mesh_path(extracted_dir: Path) -> Path:
    textured_dae = extracted_dir / "textured.dae"
    textured = extracted_dir / "textured.obj"
    nontextured = extracted_dir / "nontextured.stl"
    if textured_dae.exists():
        return textured_dae
    if textured.exists():
        return textured
    if nontextured.exists():
        return nontextured
    raise FileNotFoundError(f"No usable mesh in {extracted_dir}")


def _convert_to_ttm(
    pyrep: PyRep,
    object_id: str,
    mesh: Path,
    out_dir: Path,
    scale: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"ycb_{object_id}"
    out_path = out_dir / f"{model_name}.ttm"

    print(f"[convert] {object_id} -> {out_path.name}")
    # simImportShape preserves OBJ/DAE material + texture data when available.
    # Shape.import_mesh uses simImportMesh, which drops those visual assets.
    # bit4=16 asks CoppeliaSim to preserve imported texture information.
    handle = sim.simImportShape(0, str(mesh), 16, 0.0, scale)
    shape = Shape(handle)
    shape.set_name(model_name)
    shape.set_model(True)
    shape.save_model(str(out_path))
    shape.remove()
    return out_path


def _write_manifest(manifest_path: Path, records: List[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2) + "\n")


def build_objects(
    object_ids: Iterable[str],
    scale: float,
    skip_convert: bool,
) -> None:
    root = _repo_root()
    raw_root = _raw_root(root)
    assets_root = _assets_root(root)
    manifest_path = assets_root / "ycb_manifest.json"

    records = []
    extracted_dirs = {}

    for object_id in object_ids:
        tgz = _download_tgz(object_id, raw_root)
        extracted = _extract_tgz(tgz, raw_root)
        mesh = _mesh_path(extracted)
        extracted_dirs[object_id] = mesh
        records.append(
            {
                "object_id": object_id,
                "source_tgz": str(tgz),
                "mesh": str(mesh),
                "ttm": str(assets_root / f"ycb_{object_id}.ttm"),
            }
        )

    if not skip_convert:
        ttt = _task_design_ttt(root)
        if not ttt.exists():
            raise FileNotFoundError(f"Missing scene: {ttt}")
        pyrep = PyRep()
        pyrep.launch(str(ttt), headless=True)
        try:
            for object_id in object_ids:
                _convert_to_ttm(
                    pyrep=pyrep,
                    object_id=object_id,
                    mesh=extracted_dirs[object_id],
                    out_dir=assets_root,
                    scale=scale,
                )
        finally:
            pyrep.shutdown()

    _write_manifest(manifest_path, records)
    print(f"[done] manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects",
        nargs="+",
        default=DEFAULT_OBJECTS,
        help="YCB object IDs, e.g. 011_banana 013_apple 025_mug",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Mesh import scaling factor for Shape.import_mesh.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only download/extract and write manifest, do not create .ttm.",
    )
    args = parser.parse_args()

    build_objects(
        object_ids=args.objects,
        scale=args.scale,
        skip_convert=args.skip_convert,
    )


if __name__ == "__main__":
    main()

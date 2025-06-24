#!/usr/bin/env python3
"""
Visualise all scans that belong to ONE 3RScan environment in a single 3-D view,
giving every scan a different colour.

Usage
-----
python environment_visualization.py --json ./3RScan.json \
                               --dataset /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
                               --scene 10b1792a-3938-2467-8b4e-a93da27a0985

* If --scene is an integer, it is interpreted as the 0-based entry index
  inside 3RScan.json.
* If --scene is omitted, the script shows entry 0 (first environment).
"""

import argparse
import json
import os
import random
import sys
from typing import List, Sequence

import numpy as np
import open3d as o3d
import glob

"""
Fuse all scans of one 3RScan environment and show them in Open3D.
Works with the v2 download (mesh.refined.v2.obj).

Usage
-----
python environment_visualization.py --json  /…/3RScan.json \
                                    --dataset /…/3RScan/    \
                                    --scene  <ref-scan-id | index>
"""
import argparse
import glob
import json
import os
import random
import sys
from typing import List, Sequence, Dict

import numpy as np
import open3d as o3d


# ────────────────────────────────────────────────────────────── helpers ──
def discover_mesh(scan_dir: str) -> str:
    """Return the most complete mesh file inside one scan folder."""
    preferred = [
        "mesh.refined.v2.obj",          # new name (2023+)
        "mesh.refined.ply",             # legacy refined meshes
        "mesh.refined.obj",
        "mesh.obj",                     # coarse
        "scan_alignment_mesh.ply",
    ]
    for name in preferred:
        p = os.path.join(scan_dir, name)
        if os.path.isfile(p):
            return p

    # last-ditch fallback: first *.obj / *.ply we can find
    for ext in ("*.obj", "*.ply"):
        cand = glob.glob(os.path.join(scan_dir, ext))
        if cand:
            return sorted(cand)[0]

    raise FileNotFoundError(f"No mesh (.obj / .ply) found in {scan_dir}")


def colourise(mesh: o3d.geometry.TriangleMesh, rgb: Sequence[float]) -> None:
    """Paint all vertices with one RGB colour."""
    v = np.asarray(mesh.vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(v), 1)))


def collect_scan_ids(scene_entry: dict) -> List[str]:
    """[reference, rescan1, …]"""
    ids = [scene_entry["reference"]]
    ids.extend(s["reference"] for s in scene_entry["scans"])
    return ids


def build_xform_table(scene_entry: dict) -> Dict[str, np.ndarray]:
    """
    Returns {scan_id: 4×4 row-major}.
    * Reference scan → identity.
    * Each re-scan  → (transform).T   (transpose fixes column-major storage).
    """
    x = {scene_entry["reference"]: np.eye(4)}
    for rs in scene_entry["scans"]:
        M = np.array(rs["transform"], dtype=np.float32).reshape(4, 4).T
        x[rs["reference"]] = M
    return x


def load_and_prepare(scan_ids, dataset_dir, xforms, seed=42):
    random.seed(seed)
    geoms = []

    for sid in scan_ids:
        mesh_path = discover_mesh(os.path.join(dataset_dir, sid))
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.transform(xforms[sid])          # align into reference frame
        colourise(mesh, np.random.rand(3))   # unique colour
        geoms.append(mesh)
        print(f"✓ {sid}  ({os.path.relpath(mesh_path, dataset_dir)})")

    return geoms


# ──────────────────────────────────────────────────────────────── main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to 3RScan.json")
    ap.add_argument("--dataset", required=True,
                    help="Folder that contains all <scan_id> sub-folders")
    ap.add_argument("--scene", default="0",
                    help="Reference-scan ID *or* 0-based index (default: 0)")
    args = ap.parse_args()

    scenes = json.load(open(args.json, "r"))

    # select environment -------------------------------------------------
    try:                                # numeric index?
        scene_entry = scenes[int(args.scene)]
    except ValueError:                  # must be a reference hash
        matches = [s for s in scenes if s["reference"] == args.scene]
        if not matches:
            sys.exit(f"Scene '{args.scene}' not found.")
        scene_entry = matches[0]

    scan_ids = collect_scan_ids(scene_entry)
    xforms   = build_xform_table(scene_entry)

    print(f"\nEnvironment has {len(scan_ids)} scans:")
    for s in scan_ids:
        print("  •", s)

    # visualise -----------------------------------------------------------
    geoms = load_and_prepare(scan_ids, args.dataset, xforms)
    print("\nClose the Open3D window to quit.")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()

# ───────────────────── visualization_3rscan.py ─────────────────────
"""
Visualise a single 3RScan reconstruction with every object in a
different colour.  No captions, no matching – just a quick helper.

Usage:
    python visualization_3rscan.py --root  /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
                                   --scan-id <UUID>

"""

from pathlib import Path
import argparse, json, sys, numpy as np, open3d as o3d

# ─────────────────────────────── helpers ────────────────────────────
def load_scene(scan_dir: Path):
    """
    Returns
    -------
    mesh        : legacy-CPU open3d.geometry.TriangleMesh  
                  (vertex colours will be overwritten later)
    obj2faces   : dict { object-id : np.ndarray[face indices]}
    """
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)

    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    # make sure we have the legacy CPU mesh (triangle colours supported)
    try:
        mesh = mesh.cpu()          # Open3D ≥0.15, returns legacy mesh
    except AttributeError:
        pass                       # already legacy

    # ── map per-vertex RGB → 24-bit ints (for fast matching) ─────────
    vcol = np.asarray(mesh.vertex_colors)       # Nx3 floats 0–1
    v8   = (vcol * 255 + 0.5).astype(np.uint32)
    vhex = (v8[:,0] << 16) | (v8[:,1] << 8) | v8[:,2]

    # ── load objects.json for this scan ──────────────────────────────
    obj_json = scan_dir.parent / "objects.json"
    with open(obj_json) as fp:
        scans = {s["scan"]: s for s in json.load(fp)["scans"]}

    meta = scans.get(scan_dir.name)
    if meta is None:
        raise KeyError(f"{scan_dir.name} not found in {obj_json}")

    colour2oid = {int(o["ply_color"].lstrip("#"), 16): int(o["id"])
                  for o in meta["objects"]}

    # ── vertex → object-id array ─────────────────────────────────────
    v_oid = np.array([colour2oid.get(int(h), 0) for h in vhex],
                     dtype=np.int64)

    # ── facewise majority vote → object-id per triangle ──────────────
    tris   = np.asarray(mesh.triangles)
    f_oid  = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                      dtype=np.int64)

    # ── invert: object-id → list of face indices ─────────────────────
    obj2faces = {}
    for fid, oid in enumerate(f_oid):
        if oid == 0:            # 0 → “background”
            continue
        obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.asarray(v, dtype=np.int64)
                 for k, v in obj2faces.items()}

    if not obj2faces:
        raise RuntimeError(f"No labelled faces in {scan_dir.name}")

    return mesh, obj2faces


def colour_every_object(mesh, obj2faces, seed=42):
    """Paint each object in `obj2faces` with its own random colour."""
    rng    = np.random.default_rng(seed)
    n_vert = len(mesh.vertices)
    vcols  = np.zeros((n_vert, 3), dtype=np.float32)

    tris = np.asarray(mesh.triangles)
    for idx, (oid, faces) in enumerate(obj2faces.items()):
        c = rng.random(3)                       # bright random RGB
        for fid in faces:                       # each face (triangle)
            for vid in tris[fid]:               # three vertices
                vcols[int(vid)] = c

    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return mesh
# ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True,
                    help="folder that contains 3RScan/<scan-id>/")
    ap.add_argument("--scan-id", required=True,
                    help="UUID of a 3RScan reconstruction")
    args = ap.parse_args()

    scan_dir = Path(args.root) / args.scan_id
    print("Loading", scan_dir)

    mesh, obj2faces = load_scene(scan_dir)
    mesh            = colour_every_object(mesh, obj2faces)

    print(f"Scene {args.scan_id}: {len(obj2faces)} labelled objects")

    # ── simple Open3D viewer with quit / next callbacks ──────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=args.scan_id, width=1280, height=768)
    vis.add_geometry(mesh)

    keys_close = {32, 257}          # SPACE, ENTER
    keys_quit  = {81, 113, 256}     # q, Q, ESC
    for k in keys_close:
        vis.register_key_callback(k, lambda v: v.close())
    for k in keys_quit:
        vis.register_key_callback(k, lambda _: sys.exit(0))

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
# ────────────────────────────────────────────────────────────────────

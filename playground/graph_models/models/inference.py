#!/usr/bin/env python3
"""
inference.py
============
Batch text-to-scene retrieval:

    ScanScribe caption graph  →  top-k matching 3D-SSG scenes.

The scoring function is *identical* to visualization_graph-object.py:
0.5 × matching-probability  +  0.5 × cosine-similarity  (both ∈ [0,1]).
"""

from __future__ import annotations
import argparse, json, sys, time, torch, numpy as np, torch.nn.functional as F
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Local repo imports                                                         #
# --------------------------------------------------------------------------- #
sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')

from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph
from model_graph2graph import BigGNN            # same name as in the repo


# --------------------------------------------------------------------------- #
#  Scoring helper function
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def compute_match_score(model: BigGNN | None,
                        qg: SceneGraph,
                        sg: SceneGraph,
                        device: str = "cpu") -> float:
    """
    Returns the blended score in [0,1].  
    If *model* is None we return pure cosine similarity.
    """
    q_sub, s_sub = get_matching_subgraph(qg, sg)
    # fall back if either is degenerate
    def bad(g): return (g is None or len(g.nodes) <= 1
                        or (hasattr(g, "edge_idx") and len(g.edge_idx[0]) < 1))
    if bad(q_sub) or bad(s_sub):
        q_sub, s_sub = qg, sg

    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (torch.tensor(np.array(n), dtype=torch.float32, device=device),
                torch.tensor(np.array(e[0:2]), dtype=torch.int64,   device=device),
                torch.tensor(np.array(f),      dtype=torch.float32, device=device))

    q_n, q_e, q_f = prep(q_sub)
    s_n, s_e, s_f = prep(s_sub)

    # No GNN → cosine only (mapped to [0,1])
    if model is None:
        cos = F.cosine_similarity(q_n.mean(0, keepdim=True),
                                  s_n.mean(0, keepdim=True), dim=1).item()
        return (cos + 1) / 2

    q_emb, s_emb, m_p = model(q_n, s_n, q_e, s_e, q_f, s_f)
    cos = (F.cosine_similarity(q_emb, s_emb, dim=0).item() + 1) / 2
    return 0.5 * m_p.item() + 0.5 * cos


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", required=True, type=Path,
                   help="Folder that contains the processed_data/{3dssg,scanscribe}/ sub-folders")
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Trained BigGNN checkpoint (*.pt)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--jsonl_out", type=Path,
                   help="Write one ranked-list per query to this JSONL file")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    t0 = time.perf_counter()

    # 1) -------------------------------------------------------------------- #
    #     Load graphs (same paths/format as the visualiser)                   #
    # ----------------------------------------------------------------------- #
    g3d_raw = torch.load(args.graphs / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt",
                         map_location="cpu")
    scans_raw = torch.load(args.graphs / "scanscribe" / "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                           map_location="cpu")

    database_3dssg = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=1.0, embedding_type="word2vec",
                        use_attributes=True)
        for sid, g in g3d_raw.items()
    }
    queries = [
        SceneGraph(k.split("_")[0], txt_id=None,
                   graph=g, graph_type="scanscribe",
                   embedding_type="word2vec", use_attributes=True)
        for k, g in scans_raw.items()
    ]
    print(f"Loaded {len(queries)} ScanScribe captions, "
          f"{len(database_3dssg)} 3D-SSG scenes.")

    # 2) -------------------------------------------------------------------- #
    #     Model                                                               #
    # ----------------------------------------------------------------------- #
    device = args.device
    model = BigGNN(N=1, heads=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # 3) -------------------------------------------------------------------- #
    #     For each caption → rank all scenes                                  #
    # ----------------------------------------------------------------------- #
    jsonl = None
    if args.jsonl_out:
        jsonl = open(args.jsonl_out, "w")

    for qi, qg in enumerate(queries, 1):
        scores = {
            sid: compute_match_score(model, qg, sg, device)
            for sid, sg in database_3dssg.items()
        }
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.top_k]

        # console report
        print(f"\nQuery {qi:>4}/{len(queries)}  (scene_id={qg.scene_id})")
        for rank, (sid, sc) in enumerate(best, 1):
            gt_tag = "  *GT*" if sid == qg.scene_id else ""
            print(f"  {rank:>2}. {sid:<18}  score={sc:5.3f}{gt_tag}")

        # optional JSONL
        if jsonl:
            jsonl.write(json.dumps({
                "query_scene_id": qg.scene_id,
                "top_k": best
            }) + "\n")

    if jsonl:
        jsonl.close()
        print(f"\nWrote ranked lists to {args.jsonl_out}")

    print(f"\nFinished in {(time.perf_counter()-t0):.1f}s.")


if __name__ == "__main__":
    main()

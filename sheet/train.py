"""SHEET training entry point (algorithm box in the paper).

Usage:
    python -m sheet.train --data_dir data --out_dir output [--test_type ER]

Phase 1  build the CHTG from CSV tables
Phase 2  schema-guided heterogeneous propagation (encoder forward)
Phase 3  feature-preserving self-supervised optimisation
Phase 4  (optional) frozen-embedding downstream evaluation

Downstream labels are NEVER used to update the encoder: matching files
are only read at evaluation time.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .chtg import build_chtg, encode_raw_features
from .config import SHEETConfig
from .data import load_matches, load_tables, row_key
from .downstream.entity_resolution import evaluate_er
from .features import load_plm_cache
from .losses import (UncertaintyWeighting, degree_loss, edge_loss, feature_loss,
                     ipr_loss)
from .model import SHEET


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_torch(graph, device: torch.device):
    x_dict = {
        t: torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32,
                           device=device)
        for t, g in graph.x_dict.items()
    }
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    return x_dict, ei_dict


def degree_targets(graph, ei_dict, device):
    """log1p(in-degree) per node type, summed over all incoming relations."""
    targets = {}
    for ntype, x in graph.x_dict.items():
        n = 0 if x is None else x.shape[0]
        d = torch.zeros(max(n, 0), device=device)
        for (s, r, t), ei in ei_dict.items():
            if t == ntype and ei.numel():
                ones = torch.ones(ei.size(1), device=device)
                d.index_add_(0, ei[1], ones)
        targets[ntype] = torch.log1p(d)
    return targets


def train(cfg: SHEETConfig, device: torch.device):
    # ---------- Phase 1: CHTG construction (label-free) ----------
    tables = load_tables(cfg.data_dir)
    plm_cache = None
    cache_path = os.path.join(cfg.data_dir, "plm", "row_features.vec")
    if os.path.exists(cache_path):
        plm_cache = load_plm_cache(cache_path)
        print(f"[PLM] loaded {len(plm_cache)} cached features from {cache_path}")
    else:
        print("[PLM] no cache found -> using TF-IDF fallback encoder")

    graph = build_chtg(tables, cfg.graph, plm_cache=plm_cache)
    encode_raw_features(graph, tables, plm_cache=plm_cache)

    n_nodes = {t: (0 if x is None else x.shape[0]) for t, x in graph.x_dict.items()}
    n_edges = {f"{s}-{r}-{t}": e.size(1)
               for (s, r, t), e in graph.edge_index_dict.items()}
    print(f"[CHTG] nodes={n_nodes}  edge_families={len(n_edges)}  "
          f"edges={sum(n_edges.values()) // 2}")

    x_dict, ei_dict = to_torch(graph, device)
    input_dims = {t: x.shape[1] for t, x in x_dict.items()}
    deg_tgt = degree_targets(graph, ei_dict, device)

    # ---------- Phases 2-3: encoder + self-supervised optimisation ----------
    model = SHEET(input_dims, cfg.model).to(device)
    uw = UncertaintyWeighting().to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(uw.parameters()),
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        opt.zero_grad()

        z_dict = model(x_dict, ei_dict)
        l_feat = feature_loss(model.fdec(z_dict), x_dict)
        l_edge = edge_loss(model.edec(z_dict, ei_dict, num_neg=cfg.train.num_neg))
        l_deg = degree_loss(model.ddec.predict(z_dict), deg_tgt)
        l_ipr = ipr_loss(model.encoder.z_hidden, model.encoder.anchor,
                         cfg.train.lambda_ipr)
        loss = uw(l_feat, l_edge, l_deg) + l_ipr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch == 1 or epoch % max(1, cfg.train.num_epochs // 10) == 0:
            print(f"[epoch {epoch:4d}] loss={loss.item():.4f} "
                  f"feat={l_feat.item():.4f} edge={l_edge.item():.4f} "
                  f"deg={l_deg.item():.4f} ipr={l_ipr.item():.4f}")

    # ---------- export frozen embeddings ----------
    model.eval()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with torch.no_grad():
        z_dict = model(x_dict, ei_dict)
    export = {f"z_{t}": z.detach().cpu().numpy() for t, z in z_dict.items()}
    export["row_keys"] = np.array(graph.row_keys)
    export["row_tables"] = np.array(graph.row_tables)
    export["col_keys"] = np.array(graph.col_keys)
    emb_path = os.path.join(cfg.out_dir, "embeddings.npz")
    np.savez_compressed(emb_path, **export)
    print(f"[export] embeddings -> {emb_path}")

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_best.pt"))

    # ---------- Phase 4: frozen-embedding evaluation ----------
    if cfg.eval is not None:
        match_files = [f[:-4] for f in sorted(
            os.listdir(os.path.join(cfg.data_dir, "matches")))
            if f.endswith(".txt")] if os.path.isdir(
            os.path.join(cfg.data_dir, "matches")) else []
        for mf in match_files:
            gt = load_matches(cfg.data_dir, mf)
            res = evaluate_er(export, gt, cfg.eval)
            print(f"[ER:{mf}] precision={res['precision']:.4f} "
                  f"recall={res['recall']:.4f} f1={res['f1']:.4f} "
                  f"({res['num_pos']} gold pairs)")
            with open(os.path.join(cfg.out_dir, f"er_{mf}.json"), "w") as f:
                json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="SHEET training (reference impl.)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--output_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--candidate_k", type=int, default=10)
    ap.add_argument("--match_thresh", type=float, default=0.70)
    args = ap.parse_args()

    cfg = SHEETConfig()
    cfg.data_dir, cfg.out_dir = args.data_dir, args.out_dir
    cfg.train.num_epochs, cfg.train.lr, cfg.train.seed = args.num_epochs, args.lr, args.seed
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.output_dim = args.output_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.num_heads = args.num_heads
    cfg.eval.candidate_k, cfg.eval.match_thresh = args.candidate_k, args.match_thresh

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    train(cfg, device)


if __name__ == "__main__":
    main()

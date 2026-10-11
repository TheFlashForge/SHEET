"""CHTG construction (paper Sec. "Contextual Heterogeneous Tabular Graph").

Turns a collection of relational tables into a typed graph:

  node types   : row | column | text_value | numeric_value
  edge families: (1) structural containment   row/column <-> value
                 (2) semantic text-value edges  cos >= tau_sem, top-K
                 (3) numeric proximity edges    rel. distance <= delta_num
                 (4) unsupervised PK-FK candidates between columns
                 (5) candidate row-match edges  value-overlap based

Value nodes are deduplicated globally so that identical cell values share
one vertex (this is what makes cross-table edge families 2/3/5 useful).

For clarity this reference implementation computes neighbour lists with
dense similarity; the production system uses approximate nearest
neighbour search with identical semantics but better scalability.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .config import GraphConfig, NODE_TYPES, full_edge_types
from .data import Table, row_key
from .features import build_text_features, try_parse_float


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


@dataclass
class HeteroGraph:
    # per-type raw features ([N, D]); row/column are filled after build
    x_dict: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = field(
        default_factory=dict
    )
    # bookkeeping used by evaluation
    row_keys: List[str] = field(default_factory=list)    # row idx -> row key
    row_tables: List[str] = field(default_factory=list)  # row idx -> table
    col_keys: List[str] = field(default_factory=list)    # column idx -> table.col
    text_strings: List[str] = field(default_factory=list)
    num_values: List[float] = field(default_factory=list)

    @property
    def metadata(self):
        return list(NODE_TYPES), full_edge_types()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _topk_pairs(sim: np.ndarray, k: int, thresh: float) -> List[Tuple[int, int]]:
    """Undirected pairs (i, j), i < j, with j in TopK(i) and sim >= thresh."""
    pairs: List[Tuple[int, int]] = []
    k = min(k, sim.shape[1] - 1)
    if k < 1:
        return pairs
    for i in range(sim.shape[0]):
        row = sim[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, k)[:k]
        for j in idx:
            if row[j] >= thresh and i < j:
                pairs.append((i, int(j)))
    return pairs


def _token_set(s: str) -> Set[str]:
    return set(normalize(s).replace("_", " ").split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _undirected(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    arr = torch.tensor(pairs, dtype=torch.long)
    return torch.cat([arr, arr.flip(1)], dim=0).t().contiguous()  # [2, 2E]


def _typed_edges(edges: Set[Tuple[int, str, int]], ntype: str) -> torch.Tensor:
    """Directed [2, E] tensor of (src, dst); the reverse relation is added
    separately in the assemble step (HGT metadata convention)."""
    pairs = [(e[0], e[2]) for e in edges if e[1] == ntype]
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _numeric_features(values: List[float]) -> np.ndarray:
    """Numeric-value features: min-max normalised scalar + bias channel."""
    if not values:
        return np.zeros((0, 2), dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    lo, hi = float(v.min()), float(v.max())
    vn = (v - lo) / (hi - lo + 1e-9)
    return np.concatenate([vn, np.ones_like(vn)], axis=1)


# ----------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------

def build_chtg(
    tables: List[Table],
    cfg: GraphConfig,
    plm_cache: Optional[Dict[str, np.ndarray]] = None,
) -> HeteroGraph:
    g = HeteroGraph()

    # ---------------- nodes + containment edges ----------------
    text_vid: Dict[str, int] = {}
    num_vid: Dict[float, int] = {}

    row_val_edges: Set[Tuple[int, str, int]] = set()
    col_val_edges: Set[Tuple[int, str, int]] = set()

    for t in tables:
        col_start = len(g.col_keys)
        for c in t.columns:
            g.col_keys.append(f"{t.name}.{normalize(c)}")
        for row in t.rows:
            g.row_keys.append(row_key(t.name, row))
            g.row_tables.append(t.name)
            r = len(g.row_keys) - 1
            for c_idx, cell in enumerate(row):
                v = try_parse_float(cell)
                if v is not None:
                    if v not in num_vid:
                        num_vid[v] = len(g.num_values)
                        g.num_values.append(v)
                    vid, ntype = num_vid[v], "numeric_value"
                else:
                    key = normalize(cell)
                    if not key:
                        continue
                    if key not in text_vid:
                        text_vid[key] = len(g.text_strings)
                        g.text_strings.append(key)
                    vid, ntype = text_vid[key], "text_value"
                row_val_edges.add((r, ntype, vid))
                col_val_edges.add((col_start + c_idx, ntype, vid))

    # ---------------- (2) semantic text-value edges ----------------
    tv_x = (
        build_text_features(g.text_strings, plm_cache=plm_cache)
        if g.text_strings
        else np.zeros((0, 1), dtype=np.float32)
    )
    sem_pairs: List[Tuple[int, int]] = []
    if len(g.text_strings) > 2:
        norms = np.linalg.norm(tv_x, axis=1, keepdims=True) + 1e-9
        sem_pairs = _topk_pairs((tv_x / norms) @ (tv_x / norms).T,
                                cfg.k_sem, cfg.tau_sem)

    # ---------------- (3) numeric proximity edges ----------------
    num_pairs: List[Tuple[int, int]] = []
    if len(g.num_values) > 2:
        vals = np.asarray(g.num_values, dtype=np.float64)
        diff = np.abs(vals[:, None] - vals[None, :])
        denom = np.maximum(np.abs(vals[:, None]), np.abs(vals[None, :]))
        denom = np.maximum(denom, cfg.eps_num)
        d = diff / denom
        k = min(cfg.k_num, len(vals) - 1)
        for i in range(len(vals)):
            ci = np.flatnonzero((d[i] <= cfg.delta_num) & (np.arange(len(vals)) > i))
            if ci.size > k:  # keep the k closest
                ci = ci[np.argsort(d[i, ci])[:k]]
            num_pairs.extend((i, int(j)) for j in ci)

    # ---------------- (4) PK-FK candidate column edges ----------------
    col_value_sets: List[Set[str]] = [set() for _ in g.col_keys]
    for (c, ntype, v) in col_val_edges:
        label = g.num_values[v] if ntype == "numeric_value" else g.text_strings[v]
        col_value_sets[c].add(str(label))
    col_names = [k.split(".")[-1] for k in g.col_keys]

    pkfk_pairs: List[Tuple[int, int]] = []
    for i in range(len(g.col_keys)):
        for j in range(i + 1, len(g.col_keys)):
            if g.col_keys[i].split(".")[0] == g.col_keys[j].split(".")[0]:
                continue  # cross-table candidates only
            score = (cfg.pkfk_alpha * _jaccard(_token_set(col_names[i]),
                                               _token_set(col_names[j]))
                     + (1 - cfg.pkfk_alpha)
                     * _jaccard(col_value_sets[i], col_value_sets[j]))
            if score >= cfg.pkfk_thresh:
                pkfk_pairs.append((i, j))

    # ---------------- (5) candidate row-match edges ----------------
    row_values: List[Set[str]] = [set() for _ in g.row_keys]
    for (r, ntype, v) in row_val_edges:
        label = g.num_values[v] if ntype == "numeric_value" else g.text_strings[v]
        row_values[r].add(str(label))

    inverted: Dict[str, List[int]] = defaultdict(list)
    for r, vs in enumerate(row_values):
        for s in vs:
            inverted[s].append(r)

    overlap_cnt: Dict[Tuple[int, int], int] = defaultdict(int)
    for postings in inverted.values():
        postings = sorted(set(postings))
        if len(postings) < 2 or len(postings) > 2000:  # skip hub values
            continue
        for a_i in range(len(postings)):
            for b_i in range(a_i + 1, len(postings)):
                a, b = postings[a_i], postings[b_i]
                if g.row_tables[a] != g.row_tables[b]:  # cross-table only
                    overlap_cnt[(a, b)] += 1

    match_pairs = [(i, j) for (i, j), c in overlap_cnt.items()
                   if c >= cfg.m_cand]
    per_row: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (i, j) in match_pairs:
        per_row[i].append((j, overlap_cnt[(i, j)]))
        per_row[j].append((i, overlap_cnt[(i, j)]))
    kept: Set[Tuple[int, int]] = set()
    for r, nbrs in per_row.items():
        nbrs.sort(key=lambda x: -x[1])
        for (j, _) in nbrs[: cfg.k_cand]:
            kept.add((min(r, j), max(r, j)))

    # ---------------- assemble ----------------
    ei = g.edge_index_dict
    ei[("row", "has_value", "text_value")] = _typed_edges(row_val_edges, "text_value")
    ei[("row", "has_value", "numeric_value")] = _typed_edges(row_val_edges, "numeric_value")
    ei[("column", "has_column_value", "text_value")] = _typed_edges(col_val_edges, "text_value")
    ei[("column", "has_column_value", "numeric_value")] = _typed_edges(col_val_edges, "numeric_value")
    ei[("text_value", "sem_sim", "text_value")] = _undirected(sem_pairs)
    ei[("numeric_value", "num_close", "numeric_value")] = _undirected(num_pairs)
    ei[("column", "pkfk_candidate", "column")] = _undirected(pkfk_pairs)
    ei[("row", "potential_match", "row")] = _undirected(sorted(kept))

    # reverse relations with swapped endpoints (HGT metadata convention)
    for (s, r, t), ei_t in list(ei.items()):
        ei[(t, r + "_rev", s)] = ei_t.flip(0).clone()

    g.x_dict = {
        "row": None,        # filled by encode_raw_features()
        "column": None,     # filled by encode_raw_features()
        "text_value": tv_x,
        "numeric_value": _numeric_features(g.num_values),
    }
    return g


def encode_raw_features(
    g: HeteroGraph,
    tables: List[Table],
    plm_cache: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Fill row / column raw features (PLM anchors for the FPGR residual).

    Row text  = concatenation of its cell values;
    column text = column name + its most frequent (head) values.
    """
    row_texts = [" | ".join(normalize(c) for c in row)
                 for t in tables for row in t.rows]
    g.x_dict["row"] = build_text_features(row_texts, plm_cache=plm_cache)

    col_texts = []
    for t in tables:
        for c_idx, c in enumerate(t.columns):
            vals = [normalize(r[c_idx]) for r in t.rows[:20] if c_idx < len(r)]
            col_texts.append(normalize(c) + " " + " ".join(vals))
    g.x_dict["column"] = build_text_features(col_texts, plm_cache=plm_cache)

    # align dimensions across node types (the TF-IDF fallback may yield
    # different dims per corpus)
    dims = {x.shape[1] for x in g.x_dict.values() if x is not None and x.size}
    d = max(dims) if dims else 1
    if len(dims) > 1:
        for k, x in g.x_dict.items():
            if x is not None and x.size and x.shape[1] < d:
                g.x_dict[k] = np.pad(x, ((0, 0), (0, d - x.shape[1])))
    for k, x in g.x_dict.items():
        if x is None:
            g.x_dict[k] = np.zeros((0, d), dtype=np.float32)

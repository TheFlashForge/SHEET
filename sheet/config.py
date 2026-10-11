"""Configuration for SHEET.

SHEET: Schema-aware Heterogeneous graph Embedding for Table representations.

This is the reference / demonstration implementation accompanying the
manuscript.  It implements the full pipeline described in the paper
(CHTG construction -> schema-guided HGT propagation with FPGR -> self-
supervised typed reconstruction -> frozen-embedding evaluation) in a
compact form.  Hyper-parameter grids and large-scale experiment
automation are not part of this release.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Type vocabulary (paper Sec. "Contextual Heterogeneous Tabular Graph")
# ----------------------------------------------------------------------

NODE_TYPES: List[str] = ["row", "column", "text_value", "numeric_value"]

# Typed relations.  Each relation is stored together with its reverse so
# that message passing is bidirectional (as in HGT).
#
#   containment      : row <-> value, column <-> value      (edge family 1)
#   sem_sim          : text_value <-> text_value            (edge family 2)
#   num_close        : numeric_value <-> numeric_value      (edge family 3)
#   pkfk_candidate   : column <-> column                    (edge family 4)
#   potential_match  : row <-> row                          (edge family 5)
RELATIONS: List[Tuple[str, str, str]] = [
    ("row", "has_value", "text_value"),
    ("row", "has_value", "numeric_value"),
    ("column", "has_column_value", "text_value"),
    ("column", "has_column_value", "numeric_value"),
    ("text_value", "sem_sim", "text_value"),
    ("numeric_value", "num_close", "numeric_value"),
    ("column", "pkfk_candidate", "column"),
    ("row", "potential_match", "row"),
]

REVERSE_SUFFIX = "_rev"


def full_edge_types() -> List[Tuple[str, str, str]]:
    """Return relations plus their reverse counterparts (HGT metadata)."""
    etypes = []
    for s, r, t in RELATIONS:
        etypes.append((s, r, t))
        etypes.append((t, r + REVERSE_SUFFIX, s))
    return etypes


@dataclass
class GraphConfig:
    """CHTG construction hyper-parameters (paper Sec. CHTG)."""

    # semantic text-value edges: cos(x_i, x_j) >= tau_sem AND j in TopK
    tau_sem: float = 0.85
    k_sem: int = 10

    # numeric proximity edges: |vi-vj| / max(|vi|,|vj|,eps) <= delta_num
    delta_num: float = 0.05
    eps_num: float = 1e-6
    k_num: int = 10

    # PK-FK candidates: alpha * name_sim + (1-alpha) * value_overlap >= pkfk_thresh
    pkfk_alpha: float = 0.5
    pkfk_thresh: float = 0.6

    # candidate row-match edges: overlap >= m_cand AND j in TopK
    m_cand: int = 2
    k_cand: int = 20


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    output_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2

    # FPGR gate initialisation: gamma_0 = -2.0 -> alpha ~ 0.12
    fpgr_init: float = -2.0
    enable_fpgr: bool = True
    enable_schema_guidance: bool = True


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    # negative samples per positive edge, per relation type
    num_neg: int = 1
    # IPR weight (lambda_ipr)
    lambda_ipr: float = 0.1
    seed: int = 11


@dataclass
class EvalConfig:
    # number of nearest neighbours retrieved per query row in ER
    candidate_k: int = 10
    # cosine-similarity rejection threshold for candidate pairs
    match_thresh: float = 0.70


@dataclass
class SHEETConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # directory containing table CSVs / PLM caches
    data_dir: str = "data"
    out_dir: str = "output"

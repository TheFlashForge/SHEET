"""SHEET model (paper Sec. "The SHEET Framework").

Components, following the manuscript notation:

  * SchemaGuidedHGTLayer : relation-specific (s, rel, t) K/Q/V projections,
      schema-guided key modulation for value->row messages
      (k~ = k * sigmoid(W_g c_s)), relation-weighted sparse aggregation
      (a_ts = pi_rel * <q,k~>/sqrt(d_h), scatter-add, no softmax), and
      type-specific LayerNorm + FFN updates.
  * SchemaGuidedHGTEncoder : per-type input projections, stacked layers,
      Feature-Preserving Gated Residual (FPGR) with gamma_0 = -2.0,
      optional hidden->output expansion.
  * HeteroFeatureDecoder / HeteroEdgeDecoder / HeteroDegreeDecoder :
      typed feature, micro-structure (edge) and macro-structure (degree)
      reconstruction heads used only during self-supervised training.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NODE_TYPES, ModelConfig, full_edge_types

# relations whose source is a value node and destination is a row node:
# these receive the schema-guided key modulation (paper Sec. "Schema-Guided
# HGT Encoder", value-to-row message passing with the column-context gate).
SCHEMA_GUIDED_RELATIONS = {
    ("text_value", "has_value_rev", "row"),
    ("numeric_value", "has_value_rev", "row"),
}


def _key(s: str, r: str, t: str) -> str:
    return f"{s}__{r}__{t}"


class SchemaGuidedHGTLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, edge_types: List[Tuple[str, str, str]]):
        super().__init__()
        D, H = cfg.hidden_dim, cfg.num_heads
        self.cfg, self.edge_types = cfg, edge_types

        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict()
        self.pri = nn.ParameterDict()
        for (s, r, t) in edge_types:
            k = _key(s, r, t)
            self.k_lin[k] = nn.Linear(D, D)
            self.q_lin[k] = nn.Linear(D, D)
            self.v_lin[k] = nn.Linear(D, D)
            self.a_lin[k] = nn.Linear(D, D)
            self.pri[k] = nn.Parameter(torch.ones(H) * 0.1)

        # schema-context gate g(c_s) = sigmoid(W_g c_s)
        self.schema_gate = nn.Sequential(nn.Linear(D, D), nn.Sigmoid())

        self.ln1 = nn.ModuleDict({t: nn.LayerNorm(D) for t in NODE_TYPES})
        self.ln2 = nn.ModuleDict({t: nn.LayerNorm(D) for t in NODE_TYPES})
        self.ffn = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(D, 2 * D), nn.GELU(), nn.Dropout(cfg.dropout),
                    nn.Linear(2 * D, D), nn.Dropout(cfg.dropout),
                )
                for t in NODE_TYPES
            }
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        value_to_column: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        H, Dh = cfg.num_heads, cfg.hidden_dim // cfg.num_heads
        agg: Dict[str, torch.Tensor] = {}

        for etype, ei in edge_index_dict.items():
            s, r, t = etype
            key = _key(s, r, t)
            if key not in self.k_lin or ei.numel() == 0:
                continue
            src_x, dst_x = x_dict[s], x_dict[t]
            si, ti = ei[0], ei[1]

            k = self.k_lin[key](src_x[si]).view(-1, H, Dh)
            q = self.q_lin[key](dst_x[ti]).view(-1, H, Dh)
            v = self.v_lin[key](src_x[si]).view(-1, H, Dh)

            # --- schema-guided key modulation (value -> row messages) ---
            if cfg.enable_schema_guidance and etype in SCHEMA_GUIDED_RELATIONS:
                ctx = value_to_column.get(s) if value_to_column else None
                if ctx is not None:
                    col_ctx = ctx[si]                      # [E, D] column state
                    gate = self.schema_gate(col_ctx).view(-1, H, Dh)
                    k = k * gate

            # --- relation-weighted sparse attention (no softmax) ---
            attn = (q * k).sum(-1, keepdim=True) / math.sqrt(Dh)
            attn = attn * self.pri[key].view(1, H, 1)      # [E, H, 1]
            msg = self.a_lin[key]((v * attn).view(-1, cfg.hidden_dim))

            out = agg.get(t)
            if out is None:
                out = torch.zeros(dst_x.size(0), cfg.hidden_dim,
                                  device=msg.device, dtype=msg.dtype)
                agg[t] = out
            out.scatter_add_(0, ti.unsqueeze(1).expand_as(msg), msg)

        new_x = {}
        for ntype, x in x_dict.items():
            h = self.ln1[ntype](x + agg[ntype]) if ntype in agg else self.ln1[ntype](x)
            new_x[ntype] = self.ln2[ntype](h + self.ffn[ntype](h))
        return new_x


class SchemaGuidedHGTEncoder(nn.Module):
    def __init__(self, input_dims: Dict[str, int], cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        edge_types = full_edge_types()

        self.input_proj = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.Linear(input_dims[t], cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim), nn.GELU(),
                    nn.Dropout(cfg.dropout),
                )
                for t in NODE_TYPES
            }
        )
        self.layers = nn.ModuleList(
            SchemaGuidedHGTLayer(cfg, edge_types) for _ in range(cfg.num_layers)
        )

        # FPGR: per-type scalar gate, initialised PLM-preserving
        if cfg.enable_fpgr:
            self.fpgr_logit = nn.ParameterDict(
                {t: nn.Parameter(torch.tensor([cfg.fpgr_init])) for t in NODE_TYPES}
            )
            self.fpgr_norm = nn.ModuleDict(
                {t: nn.LayerNorm(cfg.hidden_dim) for t in NODE_TYPES}
            )

        self.expand = None
        if cfg.output_dim != cfg.hidden_dim:
            self.expand = nn.ModuleDict(
                {
                    t: nn.Sequential(
                        nn.Linear(cfg.hidden_dim, cfg.output_dim),
                        nn.LayerNorm(cfg.output_dim),
                    )
                    for t in NODE_TYPES
                }
            )

    @staticmethod
    def _column_context(
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Per value type: [N_v, D] column state gathered along containment."""
        ctx: Dict[str, torch.Tensor] = {}
        if "column" not in x_dict:
            return ctx
        col = x_dict["column"]
        for vtype in ("text_value", "numeric_value"):
            ei = edge_index_dict.get(("column", "has_column_value", vtype))
            if ei is None or ei.numel() == 0 or vtype not in x_dict:
                continue
            c = torch.zeros_like(x_dict[vtype])
            c.index_add_(0, ei[1], col[ei[0]])
            ctx[vtype] = c
        return ctx

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        proj = {t: self.input_proj[t](x) for t, x in x_dict.items()}
        h = dict(proj)
        for layer in self.layers:
            ctx = self._column_context(h, edge_index_dict)
            h = layer(h, edge_index_dict, value_to_column=ctx)

        # FPGR: h = LN(alpha * z_gnn + (1 - alpha) * p)
        if self.cfg.enable_fpgr:
            for t in h:
                a = torch.sigmoid(self.fpgr_logit[t])
                h[t] = self.fpgr_norm[t](a * h[t] + (1 - a) * proj[t])

        self.anchor = dict(proj)   # projected PLM anchors p_i (for IPR)
        self.z_hidden = dict(h)    # hidden states before expansion (for IPR)
        if self.expand is not None:
            h = {t: self.expand[t](z) for t, z in h.items()}
        return h


# ======================================================================
# Self-supervised decoding heads (training only)
# ======================================================================

class HeteroFeatureDecoder(nn.Module):
    """Type-specific reconstruction of the raw PLM anchor (cosine loss)."""

    def __init__(self, output_dim: int, input_dims: Dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {t: nn.Linear(output_dim, input_dims[t]) for t in NODE_TYPES}
        )

    def forward(self, z_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {t: self.heads[t](z) for t, z in z_dict.items()}


class HeteroEdgeDecoder(nn.Module):
    """Micro-structure: per-relation MLP over [z_s ; z_t] link logits."""

    def __init__(self, output_dim: int, edge_types: List[Tuple[str, str, str]]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                _key(*et): nn.Sequential(
                    nn.Linear(2 * output_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, 1),
                )
                for et in edge_types
            }
        )

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        num_neg: int = 1,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return {rel_key: (pos_logits, neg_logits)} with uniform negatives."""
        out = {}
        for et in self.heads:
            s, r, t = et.split("__")
            ei = edge_index_dict.get((s, r, t))
            if ei is None or ei.numel() == 0:
                continue
            si, ti = ei[0], ei[1]
            zs, zt = z_dict[s], z_dict[t]
            pos = self.heads[et](torch.cat([zs[si], zt[ti]], -1)).squeeze(-1)
            neg_t = ti[torch.randint(0, ti.numel(), (num_neg * ti.numel(),),
                                     device=ti.device)]
            neg = self.heads[et](torch.cat([zs[si.repeat(num_neg)], zt[neg_t]], -1)
                                 ).squeeze(-1)
            out[et] = (pos, neg)
        return out


class HeteroDegreeDecoder(nn.Module):
    """Macro-structure: predict per-type log-degree from the node embedding."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict(
            {t: nn.Linear(output_dim, 1) for t in NODE_TYPES}
        )

    def predict(
        self, z_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {t: self.heads[t](z).squeeze(-1) for t, z in z_dict.items()}


class SHEET(nn.Module):
    def __init__(self, input_dims: Dict[str, int], cfg: ModelConfig):
        super().__init__()
        self.encoder = SchemaGuidedHGTEncoder(input_dims, cfg)
        self.fdec = HeteroFeatureDecoder(cfg.output_dim, input_dims)
        self.edec = HeteroEdgeDecoder(cfg.output_dim, full_edge_types())
        self.ddec = HeteroDegreeDecoder(cfg.output_dim)

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

"""Self-supervised objectives (paper Sec. "Self-Supervised Multi-Objective
Optimization").

  L_feat,tau : per node type, 1 - mean cos(x_hat, x)
  L_edge     : per relation type, BCEWithLogits over positive and
               uniformly-sampled negative links, averaged over relations
  L_deg      : MSE on log1p in-degree profiles, averaged over node types
  L_ipr      : auxiliary Information Preservation Regularizer,
               1 - cos(z_hid, stopgrad(p)) averaged over types/nodes
  L_main     : homoscedastic uncertainty weighting (Kendall et al.)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_loss(x_hat: Dict[str, torch.Tensor],
                 x: Dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for t, xh in x_hat.items():
        if t not in x or x[t].numel() == 0 or xh.numel() == 0:
            continue
        cos = F.cosine_similarity(xh, x[t], dim=-1)
        terms.append(1.0 - cos.mean())
    return torch.stack(terms).mean() if terms else torch.zeros((), device=next(
        iter(x_hat.values())).device)


def edge_loss(edge_out: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
              ) -> torch.Tensor:
    terms = []
    for _, (pos, neg) in edge_out.items():
        logits = torch.cat([pos, neg], 0)
        target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], 0)
        terms.append(F.binary_cross_entropy_with_logits(logits, target))
    return torch.stack(terms).mean() if terms else torch.zeros(
        (), device=pos.device if edge_out else "cpu")


def degree_loss(deg_pred: Dict[str, torch.Tensor],
                deg_target: Dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for t, p in deg_pred.items():
        if t in deg_target and deg_target[t].numel() > 0:
            terms.append(F.mse_loss(p, deg_target[t]))
    return torch.stack(terms).mean() if terms else torch.zeros(
        (), device=next(iter(deg_pred.values())).device)


def ipr_loss(z_hid: Dict[str, torch.Tensor],
             anchor: Dict[str, torch.Tensor],
             lambda_ipr: float) -> torch.Tensor:
    """Soft tether against drift from the projected PLM anchors."""
    terms = []
    for t, z in z_hid.items():
        if t in anchor and anchor[t].numel() > 0 and z.numel() > 0:
            a = anchor[t]
            if z.shape[-1] != a.shape[-1]:  # hidden space may differ in width
                continue
            terms.append((1.0 - F.cosine_similarity(
                z, a.detach(), dim=-1)).mean())
    if not terms:
        return torch.zeros((), device=next(iter(z_hid.values())).device)
    return lambda_ipr * torch.stack(terms).mean()


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting over {feat, edge, deg}."""

    def __init__(self):
        super().__init__()
        self.s = nn.ParameterDict({k: nn.Parameter(torch.zeros(1))
                                   for k in ("feat", "edge", "deg")})

    def __call__(self, l_feat, l_edge, l_deg):
        total = 0.0
        for name, loss in (("feat", l_feat), ("edge", l_edge), ("deg", l_deg)):
            s = self.s[name]
            total = total + torch.exp(-s) * loss + s
        return total.squeeze()

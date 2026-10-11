"""Entity resolution on frozen SHEET row embeddings.

Protocol (row-level matching, paper Sec. "Row-level Matching"):
  * the encoder is frozen -- task labels never update it;
  * for every row of table A we retrieve its top-`candidate_k` nearest
    rows of table B by cosine similarity and keep the greedy best match
    if its similarity exceeds `match_thresh`;
  * precision / recall / F1 are computed against the labelled `y` pairs.

Ground-truth join: a Magellan pair (l, r) refers to rows whose *first*
CSV column value is l (in table A) and r (in table B).
"""

from typing import Dict, List, Tuple

import numpy as np


def _row_first_cells(row_keys: np.ndarray) -> np.ndarray:
    return np.array([k.split("\x1f")[1] for k in row_keys])


def evaluate_er(
    export: Dict[str, np.ndarray],
    matches: List[Tuple[str, str, str]],
    eval_cfg,
) -> Dict[str, float]:
    z = export["z_row"]
    tables = export["row_tables"]
    firsts = _row_first_cells(export["row_keys"])

    if z.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_pos": 0}

    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    # the two entity sources = two largest row groups
    uniq, counts = np.unique(tables, return_counts=True)
    if len(uniq) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "num_pos": sum(1 for _, _, l in matches if l == "y")}
    a_tbl, b_tbl = uniq[np.argsort(-counts)[:2]]
    ia = np.flatnonzero(tables == a_tbl)
    ib = np.flatnonzero(tables == b_tbl)

    gold_y = {(l.strip().lower(), r.strip().lower())
              for l, r, lab in matches if lab == "y"}

    sim = zn[ia] @ zn[ib].T                      # [|A|, |B|]
    k = min(eval_cfg.candidate_k, len(ib))
    pred: set = set()
    for qi in range(len(ia)):
        top = np.argpartition(-sim[qi], k)[:k]
        best = top[np.argmax(sim[qi, top])]
        if sim[qi, best] >= eval_cfg.match_thresh:
            pred.add((firsts[ia[qi]], firsts[ib[best]]))

    tp = len(pred & gold_y)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold_y) if gold_y else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "num_pos": len(gold_y)}

# SHEET

**S**chema-aware **H**eterogeneous graph **E**mb**e**dding for **T**able representations.

This repository is the **reference implementation accompanying our
manuscript**.  It contains a compact, self-contained version of the SHEET
pipeline -- Contextual Heterogeneous Tabular Graph (CHTG) construction,
schema-guided heterogeneous graph propagation with a Feature-Preserving
Gated Residual (FPGR), self-supervised typed reconstruction training, and
frozen-embedding entity-resolution evaluation -- so that the method
described in the paper can be inspected and exercised end-to-end.

It is intentionally **not** the full experimental framework used in the
study: benchmark harnesses, baseline integrations, hyper-parameter grids,
multi-seed orchestration and re-ranking components are not part of this
release.

## Pipeline (paper Section "The SHEET Framework")

```
tables (CSV)
   │
   ▼  sheet/chtg.py      Phase 1: typed CHTG construction
 4 node types  : row | column | text_value | numeric_value
 5 edge families: containment, semantic similarity, numeric proximity,
                  unsupervised PK-FK candidates, candidate row matches
   │
   ▼  sheet/model.py     Phases 2-3: schema-guided HGT + FPGR
 per-relation K/Q/V projections, column-context key gating on
 value->row messages, relation-weighted sparse aggregation (no softmax),
 type-specific LN/FFN, per-type PLM-preserving gate (gamma_0 = -2)
   │
   ▼  sheet/losses.py    self-supervised objectives
 typed feature reconstruction + per-relation edge reconstruction +
 degree reconstruction, uncertainty-weighted; auxiliary IPR tether
   │
   ▼  sheet/downstream/  Phase 4: frozen embeddings
 entity resolution (cosine retrieval, greedy top-1) -- labels are never
 used to update the encoder
```

## Repository layout

```
sheet/
├── config.py                      type vocabulary + all hyper-parameters
├── data.py                        CSV table & Magellan-match loading
├── features.py                    PLM anchors (cache) / TF-IDF fallback
├── chtg.py                        CHTG builder (Sec. "CHTG")
├── model.py                       schema-guided HGT, FPGR, decoders
├── losses.py                      typed reconstruction + IPR + weighting
├── train.py                       entry point: `python -m sheet.train`
└── downstream/entity_resolution.py  frozen-embedding ER evaluation
data/                              toy demo tables + matches (see data/README.md)
scripts/run_er.sh                  one-command demo
```

## Installation

```bash
conda create -n sheet python=3.10 -y && conda activate sheet
pip install -r requirements.txt        # numpy, torch, scikit-learn
```

## Quick start (bundled toy demo, CPU-friendly)

```bash
bash scripts/run_er.sh
```

Expected output ends with an ER evaluation on the bundled toy pair, e.g.

```
[CHTG] nodes={'row': 30, 'column': 10, 'text_value': 114, 'numeric_value': 11} ...
[export] embeddings -> output/embeddings.npz
[ER:shop_a__shop_b] precision=1.0000 recall=0.7143 f1=0.8333 (14 gold pairs)
```

The demo runs on the TF-IDF feature fallback.  For paper-level features,
pre-compute contextualised RoBERTa embeddings for the node strings and
place them at `data/plm/row_features.vec` (word2vec text format, one node
string per line); SHEET will pick them up automatically (see
`sheet/features.py`).

## Using your own tables

1. Put one CSV per table under `data/tables/`.
2. (For ER evaluation) add a Magellan-style label file under
   `data/matches/` (`l_value,r_value,y|n|u`; pairs join rows via their
   first CSV column).  Labels are only read at evaluation time.
3. Run `bash scripts/run_er.sh <data_dir> <out_dir>`.

## Notes

* Downstream labels are never used to update the encoder (frozen-probe
  protocol); evaluation only reads the exported embeddings.
* This code is provided to document the method.  The quantitative results
  in the paper were produced with the full experimental framework
  (RoBERTa anchors, ANN retrieval, benchmark-specific splits), which is not
  part of this release.

## Citation

If you use this code, please cite the paper (BibTeX entry to be added
with the camera-ready version).

## License

See `LICENSE`.

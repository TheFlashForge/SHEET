"""Table loading and ground-truth parsing.

Input format (see data/README.md):
  * one CSV file per table:            <data_dir>/tables/<name>.csv
  * Magellan-style labelled pairs:     <data_dir>/matches/<name>.txt
      lines of `l_value,r_value,matched` with matched in {y, n, u}
  * optional pre-computed PLM row features (word2vec .txt / .emb)
      under <data_dir>/plm/
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Table:
    name: str
    columns: List[str]
    rows: List[List[str]]  # raw cell strings, aligned with columns


def load_tables(data_dir: str, names: Optional[List[str]] = None) -> List[Table]:
    """Load all CSV tables under <data_dir>/tables (or only `names`)."""
    tdir = os.path.join(data_dir, "tables")
    tables: List[Table] = []
    for fname in sorted(os.listdir(tdir)):
        if not fname.endswith(".csv"):
            continue
        name = fname[:-4]
        if names is not None and name not in names:
            continue
        with open(os.path.join(tdir, fname), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            columns = next(reader)
            rows = [list(r) for r in reader]
        tables.append(Table(name=name, columns=columns, rows=rows))
    if not tables:
        raise FileNotFoundError(f"no CSV tables found under {tdir}")
    return tables


def load_matches(data_dir: str, name: str) -> List[Tuple[str, str, str]]:
    """Load Magellan-style match file: l,r,y|n|u per line."""
    path = os.path.join(data_dir, "matches", f"{name}.txt")
    out: List[Tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            l, r, lab = parts[0], parts[1], parts[2].lower()[:1]
            if lab in ("y", "n", "u"):
                out.append((l, r, lab))
    return out


def row_key(table: str, row: List[str]) -> str:
    """Stable string identity of a row (used to join GT matches)."""
    return table + "\x1f" + "\x1f".join(c.strip().lower() for c in row)


@dataclass
class Dataset:
    tables: List[Table] = field(default_factory=list)
    # ground-truth pairs as (key_l, key_r, label), both sides are row keys
    matches: List[Tuple[str, str, str]] = field(default_factory=list)

    def tables_named(self) -> Dict[str, Table]:
        return {t.name: t for t in self.tables}

# Data

This directory holds the inputs consumed by `python -m sheet.train`.

```
data/
├── tables/          one CSV per table:  <name>.csv  (header row + data rows)
├── matches/         optional, Magellan-style labelled pairs:
│                    <name>.txt with lines  l_value,r_value,y|n|u
│                    A pair (l, r) refers to rows whose FIRST CSV column
│                    value is l / r.  Labels are used ONLY for evaluation;
│                    the encoder never sees them.
└── plm/             optional, pre-computed PLM features for node strings,
    row_features.vec   word2vec-style: "<node_string> f1 f2 ... fD" per line
```

## Included demo data

`tables/shop_a.csv` and `tables/shop_b.csv` are a small toy product-matching
pair (beer listings from two shops) with `matches/shop_a__shop_b.txt`, so the
pipeline can be exercised end-to-end without downloading any benchmark.

## Benchmarks used in the paper

The eight ER benchmarks (Magellan/WDC-style), SOTAB (CTA/CPA) and Valentine
(schema matching) are public datasets and are not redistributed here.
Download them from their original sources and convert each pair of tables to
the CSV format above; the labelled pairs map directly onto the Magellan
`.txt` match format.

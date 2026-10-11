"""SHEET: Schema-aware Heterogeneous graph Embedding for Table representations.

Reference implementation accompanying the manuscript.  Pipeline:

    sheet.chtg     Contextual Heterogeneous Tabular Graph (CHTG) builder
    sheet.model    schema-guided HGT encoder + FPGR + decoding heads
    sheet.losses   typed feature / edge / degree reconstruction + IPR
    sheet.train    end-to-end training and embedding export
    sheet.downstream  frozen-embedding evaluation (entity resolution)
"""

__version__ = "0.1.0"

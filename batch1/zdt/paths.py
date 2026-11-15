"""Output folders for the manuscript-ready tables (markdown) and 600 dpi figures.

When the repository sits next to the manuscript sources (../manuscript), the files are written there, as during the
preparation of the paper. In a stand-alone clone they are written to tables/manuscript/ and figures/manuscript/.
"""
import os

if os.path.isdir("../manuscript/src/tables"):
    MAN_TABLES, MAN_FIGS = "../manuscript/src/tables/", "../manuscript/figs/"
else:
    MAN_TABLES, MAN_FIGS = "tables/manuscript/", "figures/manuscript/"
os.makedirs(MAN_TABLES, exist_ok=True)
os.makedirs(MAN_FIGS, exist_ok=True)

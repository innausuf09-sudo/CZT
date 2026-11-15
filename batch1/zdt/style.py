import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
FAMILY_OF = {"LR": "Classical", "RF": "Classical", "XGBoost": "Classical", "LightGBM": "Classical",
             "SVM": "Classical", "DNN": "Deep", "ResSE-CNN1D": "Deep", "LSTM": "Deep", "GRU": "Deep",
             "FT-Transformer": "Deep", "FG-GNN": "Graph/meta", "B-ProtoMeta": "Graph/meta",
             "CZT-Hybrid": "Hybrid (proposed)"}
FAM_COLOR = {"Classical": "#0072B2", "Deep": "#E69F00", "Graph/meta": "#009E73", "Hybrid (proposed)": "#D55E00"}
MODEL_COLOR = {"LR": "#0072B2", "RF": "#56B4E9", "XGBoost": "#004B7A", "LightGBM": "#3A8FC7", "SVM": "#9CC8E6",
               "DNN": "#E69F00", "ResSE-CNN1D": "#B07700", "LSTM": "#F2C25C", "GRU": "#C98F00",
               "FT-Transformer": "#8A5A00", "FG-GNN": "#009E73", "B-ProtoMeta": "#5CC7A6",
               "CZT-Hybrid": "#D55E00"}
MARKERS = {"Classical": "o", "Deep": "s", "Graph/meta": "^", "Hybrid (proposed)": "D"}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333", "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": int(os.environ.get("CZT_DPI", 300)), "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "lines.linewidth": 1.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42})


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

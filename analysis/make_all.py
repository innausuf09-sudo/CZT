"""Regenerate every table and figure of the paper and its supplement.

Run from the repository root:
    python -m analysis.make_all

Part 1 rebuilds the tables and figures of the original analysis, part 2 those of the revision (refits, calibration under
the deployment prior, triage with tuned baselines and all robustness checks). Steps that need the raw datasets
(data/ or the NetFlow cache) or the retrained fold-0 models are skipped with a message when those inputs are missing."""
import os
import runpy
from analysis import analyze as A
from analysis import paper_figures as P

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


def optional(fn, *args, need=()):
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print(f"skipped {args[0]}: missing {', '.join(missing)}")
        return
    fn(*args)


# main paper figures (numbered as in the manuscript)
runpy.run_module("analysis.fig_workflow", run_name="__main__")             # Figure 1
mean, sd, _ = A.e1_tables("s_table_p1_full", "s02_iid_heatmap")           # Table 4 source, Figure S2
A.e2_table("s_table_p2_full")                                               # Table 5 source
A.e2_heatmap("fig02_family_holdout")                                        # Figure 2
A.gap_figure("fig03_optimism_gap", mean)                                    # Figure 3
A.cd_diagram("fig04_critical_difference", "s_table_wilcoxon")              # Figure 4
optional(A.importance_figure, "fig05_permutation_importance",
         need=("results/e2_fold0_models.pkl", "data/UNSW_NB15_training-set.csv"))  # Figure 5
A.e3("s_table_p3_full", "fig06_nsl_kdd_unseen")                             # Figure 6
A.e4("s_table_p4_full", "fig07_transfer")                                   # Figure 7
P.calib_row()                                                               # Figure 8
P.triage_row()                                                              # Figure 9
P.detector_link()                                                           # Figure 10
# supplementary figures
optional(A.fig_data, "s01_data_characterisation", need=("data/UNSW_NB15_training-set.csv", "data/KDDTrain+.txt"))
A.roc_pr_figure("s03_roc_pr")
A.e6("s_table_cost", "s04_latency")
A.calib_figure("s07_calibration_detail", "s_table_calibration")
A.e7("s_table_triage", "s_table_ablation", "s12_triage_load_detail", "s06_nsga3_fronts",
     "s05_staffing_halflife", "s13_ablation_detail")
runpy.run_module("analysis.supp_figures", run_name="__main__")              # s08 to s11
runpy.run_module("analysis.paper_tables", run_name="__main__")              # Tables 4 to 10


# ------------------------------------------------------------------ part 2: revision
def module(name, need=()):
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print(f"skipped {name}: missing {', '.join(missing)}")
        return
    print(f"running {name}", flush=True)
    runpy.run_module(name, run_name="__main__")


module("analysis.data_details", need=("data/UNSW_NB15_training-set.csv", "data/KDDTrain+.txt"))
module("analysis.stats_full")             # full Wilcoxon and Friedman output
module("analysis.detection_v2_report")    # refit, TTL-free, CLAD, P5; manuscript tables T_new, T_dep, T_dec
module("analysis.calibration_prior")      # calibration under the deployment prior and decision costs
module("analysis.calibration_extra")      # intercept, slope, class-wise means, EM prior estimate
module("analysis.nfv3_artefacts", need=("results/e13_nfv3_cache.npz",))
module("analysis.catalogue_tables")       # preliminary catalogue analysis with the earlier policy definitions
module("analysis.triage_v2_report")       # main triage results; manuscript tables T_tri, T_vs, T_abl; Figs. 10, 11 sources
module("analysis.supp_tables")            # protocol sizes, ablations, index against tuned Risk, anchor means
module("analysis.partitions_report")      # random family partitions
module("analysis.robust_report")          # E17 and E18 robustness checks
module("analysis.fresh_report")           # fresh evaluation streams
module("analysis.grid_report")            # grid-level estimates; manuscript table T_grid
module("analysis.matched_report")         # matched-load tuning in all settings
module("analysis.seeds_report")           # wider CLAD grid and detector training seeds
module("analysis.site_report")            # site-disjoint protocol P6
module("analysis.fig9_v3")                # triage gain against novelty informativeness
print("all figures and tables written to figures/ and tables/")

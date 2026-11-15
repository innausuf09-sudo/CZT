# Calibrated Zero-Day Attack Detection and Whittle-Index Alert Triage under Leakage-Averse Evaluation

Code, stored results and reproduction package for the revised manuscript *Calibrated Zero-Day Attack Detection and
Whittle-Index Alert Triage under Leakage-Averse Evaluation* (International Journal of Intelligent Engineering and
Systems, manuscript ID 20267392).

The release for this manuscript is the tag **`v2.0-revision`** of this repository. Every table and figure of the
manuscript and its supplement is rebuilt from the files in `results/` with one command (see Quick reproduction), and
every experiment can be rerun from the raw datasets with `run_all.sh`.

## Quick reproduction (about 10 minutes, no datasets needed)

Requires Python 3.11.16. The commands below were run in a fresh clone; the log of that run is
`results/reproduction_log.txt`.

```
git clone https://github.com/innausuf09-sudo/CZT.git
cd CZT
git checkout v2.0-revision
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m analysis.make_all
```

`analysis.make_all` writes the CSV and markdown tables to `tables/`, the figures to `figures/`, and the tables in the
layout of the manuscript to `tables/manuscript/` and `figures/manuscript/`. Three steps need the raw datasets and are
skipped with a message when `data/` is empty: the data-characterisation tables (Supplementary Tables S1 to S4), the
dataset figure and the permutation-importance figure.

## Full reproduction from the raw data

```
python scripts/download_data.py
bash run_all.sh
```

`scripts/download_data.py` downloads UNSW-NB15 and NSL-KDD into `data/` and checks their SHA-256 sums. The NetFlow
corpora used by protocols P5 and P6 are described in `data/nfv3/README.md`. `run_all.sh` runs every experiment in
order (the original analysis, then the revision experiments E8 to E23) and finally `analysis.make_all`. On CPU the
original experiments take about 3 to 4 hours on two cores and the revision experiments about 20 hours on ten cores.
All random seeds are fixed in the scripts and recorded in `repro/config.json` and in the result files.

## Environment

Python 3.11.16 with the exact package versions in `requirements-lock.txt` (scikit-learn 1.9.1, XGBoost 3.2.0,
LightGBM 4.7.0, PyTorch 2.15.0, pymoo 0.6.2, SciPy 1.17.1, NumPy 2.4.6). `requirements.txt` lists the direct
dependencies without pins.

## Repository layout

Built from `git ls-files`; long folders are shortened.

```
CZT/
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── analysis/     # scripts that turn stored results into tables and figures (25 files)
├── data/         # dataset folder (files are downloaded, not stored)
│   ├── README.md
│   └── nfv3/README.md
├── experiments/  # one script per experiment
│   ├── exp_unsw.py, exp_dedup_effect.py, exp_latency.py, exp_triage.py, exp_kdd.py, retrain_fold0.py
│   ├── exp_triage_catalogue.py, exp_p2_refit.py, exp_triage_v2.py, exp_nfv3_temporal.py
│   ├── exp_detection_robust.py, exp_triage_robust.py, exp_clad_wide.py, exp_triage_fresh.py
│   └── exp_detector_seeds.py, exp_triage_matched.py, exp_site_disjoint.py
├── repro/        # cleaning masks, split indices, simulator streams, configuration
│   ├── config.json
│   ├── masks_unsw.npz
│   ├── splits_p1.npz
│   ├── splits_p2.npz
│   ├── splits_p3.npz
│   └── streams_e9.npz
├── requirements-lock.txt
├── requirements.txt
├── results/      # stored experiment outputs (JSON, NPZ, logs), including results/e10_refit/ (per-fold scores)
├── run_all.sh
├── scripts/
│   ├── download_data.py
│   └── sample_nfv3.py
├── tables/       # CSV and markdown tables written by the analysis scripts
└── zdt/          # library: data, detectors, CLAD, metrics, catalogue model, SOC simulator, policy gradient
```

## Experiments

| Experiment | Command | Output |
|---|---|---|
| P1 i.i.d. cross-validation | `python -m experiments.exp_unsw iid` | `results/e1_iid.json` |
| P2 family hold-out, 13 detectors | `python -m experiments.exp_unsw fh` | `results/e2_fh.json`, `results/e2_scores.npz` |
| Effect of de-duplication | `python -m experiments.exp_dedup_effect` | `results/e1b_dedup_effect.json` |
| Latency and footprint | `python -m experiments.exp_latency` | `results/e6_latency.json` |
| P3 unseen NSL-KDD types | `python -m experiments.exp_kdd novel` | `results/e3_kdd_novel.json` |
| P4 cross-dataset transfer | `python -m experiments.exp_kdd transfer` | `results/e4_transfer.json` |
| Triage of the submitted version | `python -m experiments.exp_triage` | `results/e7_triage.json`, `results/e7_nsga3.json` |
| E8 catalogue settings, earlier policies | `python -m experiments.exp_triage_catalogue` | `results/e8_catalogue.json` |
| E10 refit, TTL-free P2, CLAD, novelty scores | `python -m experiments.exp_p2_refit` | `results/e10_refit.json`, `results/e10_refit/` |
| E9 triage with tuned baselines (37 catalogue entries) | `python -m experiments.exp_triage_v2` | `results/e9_triage.json` |
| E9 random family partitions | `python -m experiments.exp_triage_v2 --partition K` (K = 0 to 7) | `results/e9_triage_partK.json` |
| E13 temporal protocol P5 | `python -m experiments.exp_nfv3_temporal` | `results/e13_nfv3_temporal.json` |
| E17 triage robustness | `python -m experiments.exp_triage_robust` | `results/e17_robust.json` |
| E18 detection robustness | `python -m experiments.exp_detection_robust` | `results/e18_detection_robust.json` |
| E19 CLAD with a wider grid | `python -m experiments.exp_clad_wide` | `results/e19_clad_wide.json` |
| E20 fresh evaluation streams | `python -m experiments.exp_triage_fresh` | `results/e20_fresh.json` |
| E21 detector training seeds | `python -m experiments.exp_detector_seeds` | `results/e21_detector_seeds.json` |
| E22 matched-load tuning | `python -m experiments.exp_triage_matched` | `results/e22_matched.json` |
| E23 site-disjoint protocol P6 | `python -m experiments.exp_site_disjoint` | `results/e23_site_disjoint.json` |

## Manuscript tables and figures

All items below are written by `python -m analysis.make_all`; the column "Script" names the module it calls.

| Manuscript item | Script | Output file |
|---|---|---|
| Table 1 (data and cleaning) | `analysis.data_details` | `tables/s_table_cleaning_per_family.csv` |
| Table 2 (simulator parameters) | fixed values of `zdt/triage.py` and `experiments/exp_triage_v2.py` | `repro/config.json` |
| Table 3 (P1) | `analysis.paper_tables` | `tables/T4_p1.md` |
| Table 4 (P2) | `analysis.paper_tables` | `tables/T5_p2.md` |
| Table 5 (refit, TTL-free, CLAD, P5) | `analysis.detection_v2_report` | `tables/manuscript/T_new.md` |
| Table 6 (deployment calibration) | `analysis.detection_v2_report` | `tables/manuscript/T_dep.md` |
| Table 7 (escalation cost) | `analysis.detection_v2_report` | `tables/manuscript/T_dec.md` |
| Table 8 (triage, three settings) | `analysis.triage_v2_report` | `tables/manuscript/T_tri.md` |
| Table 9 (index against tuned policies) | `analysis.triage_v2_report` | `tables/manuscript/T_vs.md` |
| Table 10 (ablation) | `analysis.triage_v2_report` | `tables/manuscript/T_abl.md` |
| Table 11 (grid-level comparisons) | `analysis.grid_report` | `tables/manuscript/T_grid.md` |
| Fig. 1 (workflow) | `analysis.fig_workflow` | `figures/manuscript/fig1.png` |
| Fig. 2 (change from P1 to P2) | `analysis.analyze` | `figures/fig03_optimism_gap.png` |
| Fig. 3 (gain against informativeness) | `analysis.fig9_v3` | `figures/manuscript/fig9.png` |
| Fig. 4 (candidate fronts) | `analysis.triage_v2_report` | `figures/manuscript/fig10.png` |

## Supplementary tables and figures

| Supplementary item | Script | Output file |
|---|---|---|
| Tables S1 to S4 (cleaning, feature lists) | `analysis.data_details` | `tables/s_table_cleaning_per_family.csv`, `tables/s_table_features_*.csv` |
| Tables S5 to S7 (sizes, P2 splits and overlap checks, validation priors) | `analysis.supp_tables`, `analysis.calibration_prior` | `tables/s_table_sizes.csv`, `results/e10_refit.json` (key `split_checks`), `tables/s_table_calibration_prior_folds.csv` |
| Table S8 (training details) | fixed settings of `zdt/models.py` | `repro/config.json` |
| Tables S9 to S12 (per-fold values, Wilcoxon) | `analysis.stats_full` | `tables/s_table_p2_folds_*.csv`, `tables/s_table_wilcoxon_full.csv`, `tables/s_table_friedman_full.json` |
| Tables S13 to S15 (refit, CLAD search, novelty AUROC) | `analysis.detection_v2_report` | `tables/s_table_e10_refit_folds.csv`, `tables/s_table_e10_novelty_auroc.csv` |
| Tables S16 to S18 (comparators, filters, P5 filter) | `analysis.robust_report` | `tables/s_table_e18_*.csv` |
| Tables S19 and S20 (wider CLAD grid, detector seeds) | `analysis.seeds_report` | `tables/s_table_e19_clad_wide.csv`, `tables/s_table_e21_detector_seeds.csv` |
| Tables S21 to S23 (calibration, decisions, diagnostics) | `analysis.calibration_prior`, `analysis.calibration_extra` | `tables/s_table_calibration_prior.csv`, `tables/s_table_calibration_decision.csv`, `tables/s_table_calibration_extra.csv` |
| Tables S24 to S27 (P5 per fold) | `analysis.detection_v2_report` | `tables/s_table_e13_nfv3_folds.csv` |
| Tables S28 and S29 (site-disjoint P6) | `analysis.site_report` | `tables/s_table_e23_site.csv`, `tables/s_table_e23_perclass.csv` |
| Tables S30 to S38 (triage in all settings, groups, ablation, tuned vectors, simulator sensitivity, partitions) | `analysis.triage_v2_report`, `analysis.supp_tables`, `analysis.partitions_report` | `tables/s_table_e9_*.csv` |
| Tables S39 to S47 (triage robustness) | `analysis.robust_report` | `tables/s_table_e17_*.csv` |
| Tables S48 and S49 (fresh streams) | `analysis.fresh_report` | `tables/s_table_e20_setting.csv`, `tables/s_table_e20_counts.csv` |
| Table S50 (matched-load tuning) | `analysis.matched_report` | `tables/s_table_e22_matched.csv` |
| Table S51 (grid-level estimates) | `analysis.grid_report` | `tables/s_table_grid.csv` |
| Tables S52 to S56 (prior work, detector settings, P3, cost, P4) | `analysis.paper_tables` | `tables/T6_p3.md`, `tables/T7_p4.md`, `tables/T8_calib_cost.md` |
| Table S57 (reproducibility checklist) | this repository | `repro/`, `requirements-lock.txt`, `results/` |
| Supplementary figures (per-family results, rank diagram, importance, reliability, simulator sensitivity, P3, P4, data, ROC, latency, training curves) | `analysis.analyze`, `analysis.supp_figures`, `analysis.triage_v2_report` | `figures/fig0*.png`, `figures/s*.png`, `figures/manuscript/fig11.png` |

## Reproducibility package (Supplementary Table S57)

| Item | Location |
|---|---|
| Duplicate and conflict masks | `repro/masks_unsw.npz` |
| Split indices | `repro/splits_p1.npz`, `repro/splits_p2.npz`, `repro/splits_p3.npz`; P4 and P5 seeds in `repro/config.json` |
| Detector and calibration configurations | `repro/config.json`; `zdt/models.py`, `zdt/clad.py`, `zdt/anomaly.py` |
| Per-fold scores and labels | `results/e2_scores.npz`, `results/e10_refit/*.npz` |
| Optimiser seeds and selected weights | `repro/config.json`, `tables/s_table_e9_tuned.csv`, `results/e9_triage.json` |
| Simulator streams | `repro/streams_e9.npz` (main runs); all other runs are regenerated from the seeds in their result files |
| Statistical output | `tables/s_table_wilcoxon_full.csv`, `tables/s_table_friedman_full.json`, `tables/s_table_e9_paired.csv`, `tables/s_table_grid.csv` |
| Environment | `requirements-lock.txt` (Python 3.11.16) |

## Data

The datasets are not stored here. UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset. NSL-KDD:
https://www.unb.ca/cic/datasets/nsl.html. NF-UNSW-NB15-v3 and NF-CICIDS2018-v3:
https://staff.itee.uq.edu.au/marius/NIDS_datasets/. Please cite the dataset authors when you use them.

## Licence and citation

MIT licence (see `LICENSE`). Citation metadata are in `CITATION.cff`.

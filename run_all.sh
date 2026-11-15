#!/usr/bin/env bash
# Full reproduction of every experiment, then all tables and figures. Run from the repository root.
# CPU only. The original runs take about 3 to 4 hours on two cores; the revision runs take about 20 hours on ten cores
# (E9 about 4 h, E17 about 4.8 h, E22 about 2.1 h, the partitions about 3 h).
# The NetFlow steps (E13, E18, E23) need the files described in data/nfv3/README.md.
set -e
mkdir -p results

# data
python scripts/download_data.py                                       # UNSW-NB15 and NSL-KDD, SHA-256 checked

# original analysis
python -m experiments.exp_unsw fh        > results/e2_fh.log 2>&1   # P2 family hold-out, 13 detectors
python -m experiments.exp_unsw iid       > results/e1_iid.log 2>&1  # P1 i.i.d. cross-validation
python -m experiments.exp_dedup_effect   > results/e1b.log 2>&1     # effect of de-duplication
python -m experiments.exp_latency        > results/e6.log 2>&1      # latency and footprint
python -m experiments.exp_triage         > results/e7.log 2>&1      # triage of the submitted version
python -m experiments.exp_kdd novel      > results/e3.log 2>&1      # P3 unseen NSL-KDD attack types
python -m experiments.exp_kdd transfer   > results/e4.log 2>&1      # P4 cross-dataset transfer
python -m experiments.retrain_fold0      > results/e2_fold0.log 2>&1 # fold-0 models for the permutation importance

# revision
python -m experiments.exp_triage_catalogue > results/e8_catalogue.log 2>&1   # E8 catalogue settings, earlier policies
python -m experiments.exp_p2_refit       > results/e10_refit.log 2>&1        # E10 refit, TTL-free P2, CLAD, novelty scores
python -m experiments.exp_triage_v2      > results/e9_triage.log 2>&1        # E9 triage with tuned baselines, 37 entries
for k in 0 1 2 3 4 5 6 7; do                                                  # E9 random family partitions
  python -m experiments.exp_triage_v2 --partition $k > results/e9_triage_part$k.log 2>&1
done
python -m experiments.exp_nfv3_temporal  > results/e13_nfv3_temporal.log 2>&1 # E13 temporal protocol P5
python -m experiments.exp_detection_robust > results/e18_detection_robust.log 2>&1 # E18 anomaly baselines, filters
python -m experiments.exp_triage_robust  > results/e17_robust.log 2>&1        # E17 triage robustness
python -m experiments.exp_clad_wide      > results/e19_clad_wide.log 2>&1     # E19 CLAD with a wider grid
python -m experiments.exp_triage_fresh   > results/e20_fresh.log 2>&1         # E20 fresh evaluation streams
python -m experiments.exp_detector_seeds > results/e21_detector_seeds.log 2>&1 # E21 detector training seeds
python -m experiments.exp_triage_matched > results/e22_matched.log 2>&1       # E22 matched-load tuning
python scripts/sample_nfv3.py data/nfv3/NF-CICIDS2018-v3.csv                  # seeded sample for P6
python -m experiments.exp_site_disjoint  > results/e23_site_disjoint.log 2>&1 # E23 site-disjoint protocol P6

# tables and figures
python -m analysis.make_all

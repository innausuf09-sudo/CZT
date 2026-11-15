# E8 findings (catalogue observation model, tuned baselines, CZT-T ablations)

Run: `python -m experiments.exp_triage_catalogue` (3651 s, 8 processes), tables via `python -m analysis.catalogue_tables`.
Detector CZT-Hybrid, rho = 1.2, c = 4, 2 groups x 20 paired replications (common random numbers), 37 catalogue scenarios.
Every tunable policy re-tuned per scenario with the same NSGA-III budget (84 x 60, seed 7, SBX eta 30 / PM eta 20).
Intervals are 95% paired t-intervals over 40 replications. Tables: `tables/s_table_e8_{means,paired,ablation,tuned}.*`.

1. Original catalogue model (m_k 1, m_z 0, m_b 0.5) reproduces the paper: Risk ZCV 0.586, CZT-T 0.675, CZT-TE 0.637.
   CZT-T - Risk = +0.088 [0.073, 0.103].
2. Tuned Risk (p^gamma s^beta with the same s_u range and budget) reaches 0.663. CZT-T - Risk-T = +0.012 [0.006, 0.017];
   CZT-T - Linear-T = +0.006 [-0.003, 0.015] (no detected difference). CZT-TE is below Risk-TE (-0.019) and Linear-TE (-0.019).
3. CZT-T ablations (original model): no severity -0.132, s_u = 0.5 -0.110, no score -0.003, no decay +0.003,
   no deadline +0.007 (but critical SLA misses 1.3% -> 5.9%). The gain comes from imputing high severity for unmatched alerts.
4. Sensitivity: the advantage of any s_u-imputing tuned rule over plain Risk shrinks as zero-days match the catalogue
   (m_z 0.4, m_b 0.2: about 0) and is absent when matching is uninformative (all rates 0.5).
   Over 37 scenarios CZT-T vs Risk-T: better 7, worse 19, no detected difference 11. CZT-T vs Linear-T: better 6, worse 18.
5. Known-attack harm: CZT-T captures less known-attack value than Risk in 33 of 37 scenarios (original: -0.063).
6. Group heterogeneity (central m_k 0.8, m_z 0.2, m_b 0.5): CZT-T - Risk is 0.000 [-0.016, 0.016] for G1 and +0.050 for G2.
7. Caveat: the knee rule over four objectives does not target ZCV. In the central scenario CZT tuned there (0.598) is below
   the parameters tuned in the original scenario (0.633), so tuned-policy comparisons partly reflect where the knee lands.

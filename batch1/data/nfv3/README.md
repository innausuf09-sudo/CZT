# NetFlow v3 data (not stored in the repository)

Place the University of Queensland NetFlow v3 files here (https://staff.itee.uq.edu.au/marius/NIDS_datasets/):

| File | Use | SHA-1 |
|---|---|---|
| `NF-UNSW-NB15-v3.csv` | temporal protocol P5 and site-disjoint protocol P6 | `2cba4879ead17c51e3268a01c06af41186cd90e4` |
| `NF-CICIDS2018-v3.csv` | site-disjoint protocol P6 | `ddcad6d826b73a128a447a0189137efd6c8f70c9` |

NF-CICIDS2018-v3 is reduced before use with the seeded stratified sampler:

```
python scripts/sample_nfv3.py data/nfv3/NF-CICIDS2018-v3.csv
```

which writes `data/nfv3/NF-CICIDS2018-v3_sample.csv` (seed 20261109; every attack class up to 60,000 flows and
1,000,000 benign flows) and a report of the class counts.

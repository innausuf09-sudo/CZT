# CZT[README.md](https://github.com/user-attachments/files/34739144/README.md)
# Optimized Hybrid Models for Zero-Day Cyber Forensic Triage

Code and results accompanying the manuscript *"Optimized Hybrid Models for Zero-Day Cyber Forensic Triage"* (submitted to the International Journal of Intelligent Engineering and Systems, IJIES).

The repository contains the pipeline used in the study: hybrid detection models for previously unseen (zero-day) attacks, probability calibration of model outputs, and a triage layer that ranks Security Operations Centre (SOC) alerts using a Whittle-index policy with NSGA-III multi-objective tuning.

## Repository structure

```
zero-day-triage/
├── data/              # dataset files (not tracked; see "Data" below)
├── src/               # source code
│   ├── preprocess.py  # cleaning, feature scaling, zero-day split
│   ├── models.py      # baseline and hybrid detectors
│   ├── calibrate.py   # calibration and reliability metrics
│   ├── triage.py      # Whittle-index alert prioritisation, NSGA-III tuning
│   └── evaluate.py    # metrics, tables and figures
├── notebooks/         # exploratory and plotting notebooks
├── results/
│   ├── figures/       # figures reported in the manuscript
│   └── tables/        # result tables (CSV)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Rename or remove the script entries above so they match the files you upload.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/USERNAME/zero-day-triage.git
cd zero-day-triage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The experiments use publicly available intrusion-detection datasets. Because of their size and licence terms, they are not redistributed here. Download them from the original sources listed below and place the files in `data/`.

| Dataset | Source |
|---|---|
| DATASET_NAME_1 | URL |
| DATASET_NAME_2 | URL |

To simulate zero-day conditions, selected attack families are withheld from training and appear only at test time; the held-out families are listed in the manuscript and in `src/preprocess.py`.

## Reproducing the results

```bash
python src/preprocess.py --data data/ --out data/processed/
python src/models.py     --data data/processed/ --out results/
python src/calibrate.py  --pred results/predictions.csv --out results/
python src/triage.py     --pred results/calibrated.csv --out results/
python src/evaluate.py   --results results/
```

Random seeds are fixed in each script so that reported values can be reproduced. Minor numerical differences may still occur across hardware and library versions.

## Citation

If you use this code, please cite the manuscript (details will be updated on publication). A machine-readable citation is provided in `CITATION.cff`.

## Licence

Released under the MIT Licence; see `LICENSE`.

## Contact

For questions about the code, please open an issue in this repository.

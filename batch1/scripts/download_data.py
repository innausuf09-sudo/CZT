"""Download the two public datasets into data/ and verify their checksums.

Run from the repository root:
    python scripts/download_data.py

UNSW-NB15 is fetched from a public GitHub mirror of the official partitioned
files; the official source is https://research.unsw.edu.au/projects/unsw-nb15-dataset.
NSL-KDD is fetched from a public GitHub mirror; the official source is
https://www.unb.ca/cic/datasets/nsl.html. If a mirror is unavailable, download
the same four files manually and place them in data/."""
import hashlib
import os
import urllib.request

FILES = {
    "KDDTrain+.txt": ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                      "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95"),
    "KDDTest+.txt": ("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                     "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84"),
    "UNSW_NB15_training-set.csv": ("https://media.githubusercontent.com/media/jamshaid120/UNSW_NB15-Complete-dataset/main/UNSW_NB15_training-set.csv",
                                   "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"),
    "UNSW_NB15_testing-set.csv": ("https://media.githubusercontent.com/media/jamshaid120/UNSW_NB15-Complete-dataset/main/UNSW_NB15_testing-set.csv",
                                  "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs("data", exist_ok=True)
    for name, (url, digest) in FILES.items():
        path = os.path.join("data", name)
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        ok = sha256(path) == digest
        print(f"{name}: {'checksum ok' if ok else 'CHECKSUM MISMATCH (file differs from the one used in the paper)'}")


if __name__ == "__main__":
    main()

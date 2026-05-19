"""Generate the vendored 30-row German Credit slice for notebooks/01_quickstart.ipynb.

Run once and commit ``notebooks/data/german_credit_sample.csv``. Deterministic
at seed 42. Preserves the raw 20-feature schema: categorical columns stay as
strings (one-hot encoding happens in the notebook), numeric columns stay as
integers, and the ``class`` target is mapped from ``"good"``/``"bad"`` to
``0``/``1`` for sklearn compatibility.

Usage:
    python scripts/generate_german_credit_sample.py

Requires: scikit-learn, pandas (both in the [notebook] extra).
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RANDOM_STATE = 42
N_ROWS = 30
OUTPUT_PATH = Path(__file__).parent.parent / "notebooks" / "data" / "german_credit_sample.csv"


def main() -> None:
    bundle = fetch_openml(name="credit-g", version=1, as_frame=True)
    frame: pd.DataFrame = bundle.frame  # 20 features + 'class' target

    # Map target labels ("good"/"bad") to integers (0/1).
    frame = frame.copy()
    frame["class"] = frame["class"].map({"good": 0, "bad": 1}).astype("int64")

    # Deterministic 30-row sample.
    sample = frame.sample(n=N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({sample.shape[0]} rows x {sample.shape[1]} cols)")
    print(sample.head())


if __name__ == "__main__":
    main()

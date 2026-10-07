"""Train a custom gesture classifier from data/gestures.csv.

Usage:
    python train_gestures.py [--epochs 300] [--test-size 0.2]

Saves the trained model to models/custom_gesture.npz and web/custom_gesture.json.
"""
import argparse
from collections import Counter

import numpy as np

from gesture_common import CLASSIFIER_PATH, DATA_PATH, WEB_MODEL_PATH, GestureClassifier


def load_data():
    if not DATA_PATH.exists():
        raise ValueError(f"No data at {DATA_PATH}. Collect some samples first.")
    raw = np.genfromtxt(DATA_PATH, delimiter=",", dtype=str, skip_header=1, ndmin=2,
                        encoding="utf-8")
    return raw[:, 1:].astype(np.float32), raw[:, 0]


def stratified_split(y, test_size, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def report(y_true, y_pred, classes, log=print):
    log(f"\n{'label':<15}{'precision':>10}{'recall':>10}{'support':>10}")
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        precision = tp / max(np.sum(y_pred == c), 1)
        recall = tp / max(np.sum(y_true == c), 1)
        log(f"{c:<15}{precision:>10.3f}{recall:>10.3f}{np.sum(y_true == c):>10}")
    log(f"\naccuracy: {np.mean(y_true == y_pred):.3f}")

    log("\nConfusion matrix (rows=true, cols=pred):")
    width = max(len(c) for c in classes) + 2
    log(" " * width + "".join(f"{c:>{width}}" for c in classes))
    for t in classes:
        row = [np.sum((y_true == t) & (y_pred == p)) for p in classes]
        log(f"{t:<{width}}" + "".join(f"{n:>{width}}" for n in row))


def train(epochs=300, test_size=0.2, log=print):
    """Train on data/gestures.csv, save to models/custom_gesture.npz, return the model."""
    X, y = load_data()
    counts = Counter(y)
    log("Samples per label:")
    for label, n in sorted(counts.items()):
        log(f"  {label:<15} {n}")
    if len(counts) < 2:
        raise ValueError("Need at least 2 different labels to train.")
    if min(counts.values()) < 5:
        raise ValueError("Each label needs at least 5 samples.")

    train_idx, test_idx = stratified_split(y, test_size)
    model = GestureClassifier.fit(X[train_idx], y[train_idx], epochs=epochs,
                                  val_X=X[test_idx], val_y=y[test_idx], log=log)
    report(y[test_idx], model.predict(X[test_idx]), model.classes_, log)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(CLASSIFIER_PATH)
    model.export_json(WEB_MODEL_PATH)
    log(f"\nSaved model to {CLASSIFIER_PATH} (web: {WEB_MODEL_PATH})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    try:
        train(args.epochs, args.test_size)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()

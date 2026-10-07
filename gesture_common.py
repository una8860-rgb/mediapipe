"""Shared helpers for the custom gesture pipeline (collect / train / infer)."""
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).parent
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "custom_gesture.npz"
DATA_PATH = ROOT / "data" / "gestures.csv"
WEB_MODEL_PATH = ROOT / "web" / "custom_gesture.json"
NUM_FEATURES = 21 * 3
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def create_hand_landmarker(num_hands=1):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(landmarks, handedness_name):
    """Convert 21 landmarks to a position/scale-invariant 63-dim vector.

    - Translate so the wrist is the origin.
    - Scale so the farthest point is at distance 1.
    - Mirror left hands so one model works for both hands.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    if handedness_name == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return points


class SampleStore:
    """Append-only CSV of (label, 63 features) rows with per-label counts."""

    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_rows([])
        with self.path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        self.counts = Counter(row[0] for row in rows if row)

    def _header(self):
        return ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]

    def _write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._header())
            writer.writerows(rows)

    def add(self, label, features):
        self.add_many(label, [features])

    def add_many(self, label, rows):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([label] + [f"{v:.6f}" for v in features] for features in rows)
        self.counts[label] += len(rows)

    def delete(self, label):
        with self.path.open(newline="", encoding="utf-8") as f:
            rows = [row for row in list(csv.reader(f))[1:] if row and row[0] != label]
        self._write_rows(rows)
        del self.counts[label]


class GestureClassifier:
    """Small MLP (numpy only) with standardized inputs and softmax output."""

    def __init__(self, classes, mean, std, weights):
        self.classes_ = np.asarray(classes)
        self.mean, self.std = mean, std
        self.weights = weights  # [(W, b), ...]

    def predict_proba(self, X):
        h = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.weights):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)
        h -= h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @classmethod
    def fit(cls, X, y, hidden=(128, 64), epochs=300, lr=1e-3, batch_size=64,
            weight_decay=1e-4, val_X=None, val_y=None, patience=30, seed=42, log=print):
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        y_idx = np.searchsorted(classes, y)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
        Xn = (X - mean) / std

        sizes = [X.shape[1], *hidden, len(classes)]
        weights = [(rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32),
                    np.zeros(b, dtype=np.float32)) for a, b in zip(sizes, sizes[1:])]
        model = cls(classes, mean, std, weights)
        m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in weights]
        v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in weights]
        b1, b2, step = 0.9, 0.999, 0

        best_acc, best_weights, wait = -1.0, None, 0
        for epoch in range(epochs):
            order = rng.permutation(len(Xn))
            for start in range(0, len(Xn), batch_size):
                idx = order[start:start + batch_size]
                # forward
                acts = [Xn[idx]]
                for i, (W, b) in enumerate(weights):
                    z = acts[-1] @ W + b
                    acts.append(np.maximum(z, 0) if i < len(weights) - 1 else z)
                logits = acts[-1] - acts[-1].max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                # backward
                grad = probs
                grad[np.arange(len(idx)), y_idx[idx]] -= 1
                grad /= len(idx)
                step += 1
                for i in reversed(range(len(weights))):
                    W, b = weights[i]
                    gW = acts[i].T @ grad + weight_decay * W
                    gb = grad.sum(axis=0)
                    if i > 0:
                        grad = (grad @ W.T) * (acts[i] > 0)
                    for j, g in enumerate((gW, gb)):
                        m[i][j][...] = b1 * m[i][j] + (1 - b1) * g
                        v[i][j][...] = b2 * v[i][j] + (1 - b2) * g * g
                        mh = m[i][j] / (1 - b1 ** step)
                        vh = v[i][j] / (1 - b2 ** step)
                        weights[i][j][...] -= lr * mh / (np.sqrt(vh) + 1e-8)

            if val_X is not None:
                acc = float(np.mean(model.predict(val_X) == val_y))
                if (epoch + 1) % 10 == 0:
                    log(f"epoch {epoch + 1}: val acc {acc:.3f}")
                if acc > best_acc:
                    best_acc, wait = acc, 0
                    best_weights = [(W.copy(), b.copy()) for W, b in weights]
                else:
                    wait += 1
                    if wait >= patience:
                        log(f"Early stopping at epoch {epoch + 1}")
                        break
        if best_weights is not None:
            model.weights = best_weights
        return model

    def save(self, path):
        arrays = {"classes": self.classes_, "mean": self.mean, "std": self.std}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez(path, **arrays)

    def export_json(self, path):
        """Write the model as JSON for the browser demo (web/index.html)."""
        import json
        data = {
            "classes": [str(c) for c in self.classes_],
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in self.weights],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        n = sum(1 for k in data.files if k.startswith("W"))
        weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        return cls(data["classes"], data["mean"], data["std"], weights)

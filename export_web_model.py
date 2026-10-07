"""Export models/custom_gesture.npz to web/custom_gesture.json for the browser demo.

train_gestures.py / gesture_app.py already do this after every training run;
use this script only to export an existing model.
"""
from gesture_common import CLASSIFIER_PATH, WEB_MODEL_PATH, GestureClassifier

if __name__ == "__main__":
    model = GestureClassifier.load(CLASSIFIER_PATH)
    model.export_json(WEB_MODEL_PATH)
    print(f"Exported {list(model.classes_)} -> {WEB_MODEL_PATH}")

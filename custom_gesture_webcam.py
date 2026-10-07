"""Run the custom-trained gesture classifier on the webcam.

Usage:
    python custom_gesture_webcam.py [--camera 0] [--num-hands 2] [--threshold 0.7]

Press 'q' or ESC to quit.
"""
import argparse
import time

import cv2
import mediapipe as mp

from gesture_common import (CLASSIFIER_PATH, GestureClassifier, create_hand_landmarker,
                            draw_hand, landmarks_to_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="below this probability the gesture is shown as Unknown")
    args = parser.parse_args()

    if not CLASSIFIER_PATH.exists():
        raise SystemExit(f"No model at {CLASSIFIER_PATH}. Run train_gestures.py first.")
    classifier = GestureClassifier.load(CLASSIFIER_PATH)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    start = time.monotonic()
    prev = start
    with create_hand_landmarker(num_hands=args.num_hands) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                mp_image, int((time.monotonic() - start) * 1000))

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_name = handedness[0].category_name
                features = landmarks_to_features(landmarks, hand_name)
                probs = classifier.predict_proba([features])[0]
                best = probs.argmax()
                label = classifier.classes_[best] if probs[best] >= args.threshold else "Unknown"

                points = draw_hand(frame, landmarks)
                x_min = min(p[0] for p in points)
                y_min = min(p[1] for p in points)
                cv2.putText(frame, f"{hand_name}: {label} ({probs[best]:.2f})",
                            (x_min, max(y_min - 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

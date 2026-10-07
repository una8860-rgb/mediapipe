"""Collect hand-landmark samples for a custom gesture from the webcam.

Usage:
    python collect_gestures.py --label rock
    python collect_gestures.py --label none   # "no particular gesture" class

Keys:
    SPACE   start / pause recording
    q, ESC  quit (samples are saved as they are recorded)

Samples are appended to data/gestures.csv (label + 63 features per row).
"""
import argparse
import csv
import time

import cv2
import mediapipe as mp

from gesture_common import DATA_PATH, NUM_FEATURES, create_hand_landmarker, draw_hand, landmarks_to_features


def count_existing(label):
    if not DATA_PATH.exists():
        return 0
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.reader(f) if row and row[0] == label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="gesture name to record")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.05,
                        help="min seconds between saved samples")
    args = parser.parse_args()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DATA_PATH.exists()
    out = DATA_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    count = count_existing(args.label)
    recording = False
    last_saved = 0.0
    start = time.monotonic()

    with create_hand_landmarker(num_hands=1) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now = time.monotonic()
            result = landmarker.detect_for_video(mp_image, int((now - start) * 1000))

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks, (0, 0, 255) if recording else (0, 255, 0))
                if recording and now - last_saved >= args.interval:
                    features = landmarks_to_features(
                        landmarks, result.handedness[0][0].category_name)
                    writer.writerow([args.label] + [f"{v:.6f}" for v in features])
                    out.flush()
                    count += 1
                    last_saved = now

            status = "REC" if recording else "PAUSED (SPACE to record)"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{args.label}] samples: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Collect gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key in (ord("q"), 27):
                break

    out.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved. '{args.label}' now has {count} samples in {DATA_PATH}")


if __name__ == "__main__":
    main()

# MediaPipe Gesture Studio

MediaPipe Hand Landmarker로 손 제스처를 직접 수집·학습하고 웹캠으로 인식하는 프로젝트입니다.

**웹 데모 (인식 전용):** https://una8860-rgb.github.io/mediapipe/

- **0** → 🤩 두근두근 기대하는 표정
- **5** → 🎆 폭죽 + 🥳 신나는 파티 (0으로 기대감을 채운 뒤 5를 펼치면 더 큰 피날레)

## 설치

```
pip install mediapipe opencv-python numpy pillow
```

## 실행

| 명령 | 설명 |
|---|---|
| `python web_server.py` | 웹 Gesture Studio (수집·학습·인식) — http://localhost:8000 |
| `python gesture_app.py` | 데스크톱 Gesture Studio (tkinter) |
| `python hand_landmarker_webcam.py` | 손 특징점 21개 표시 |
| `python gesture_recognizer_webcam.py` | MediaPipe 기본 제스처 8종 인식 |

CLI 버전: `collect_gestures.py --label <이름>` → `train_gestures.py` → `custom_gesture_webcam.py`

## 구조

- `gesture_common.py` — 특징점 정규화, 데이터 저장, numpy MLP 분류기
- `models/` — `hand_landmarker.task`, `gesture_recognizer.task`, 학습된 `custom_gesture.npz`
- `web/studio.html` — 웹 Studio (`custom_gesture.json` 모델 사용)
- `data/gestures.csv` — 수집 데이터 (git에는 포함하지 않음)

GitHub Pages에는 서버가 없어서 웹 데모는 인식만 됩니다. 수집과 학습은 로컬에서 `python web_server.py`로 하세요.
학습하면 `web/custom_gesture.json`이 갱신되므로, 커밋·푸시하면 웹 데모에도 반영됩니다.

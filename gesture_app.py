"""Custom Gesture Studio - collect, train and test custom gestures in one window.

Usage:
    python gesture_app.py

Tabs:
    1. 수집  pick/enter a label, press [녹화] (or SPACE) and show the gesture
    2. 학습  train the classifier on everything collected so far
    3. 인식  run the trained classifier live on the webcam
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont, ImageTk

import train_gestures
from gesture_common import (CLASSIFIER_PATH, DATA_PATH, GestureClassifier, SampleStore,
                            create_hand_landmarker, draw_hand, landmarks_to_features)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
COLLECT_INTERVAL = 0.05  # seconds between saved samples (~20 samples/s)
VIDEO_SIZE = (800, 600)
TAB_COLLECT, TAB_TRAIN, TAB_RECOGNIZE = range(3)


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Custom Gesture Studio")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.store = SampleStore(DATA_PATH)
        self.classifier = GestureClassifier.load(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None
        self.landmarker = create_hand_landmarker(num_hands=2)
        self.cap = None
        self.start_time = time.monotonic()
        self.last_ts = -1
        self.last_saved = 0.0
        self.recording = False
        self.session_count = 0
        self.train_queue = queue.Queue()
        self.training = False
        self.font_small = load_font(22)
        self.font_big = load_font(30)

        self.build_ui()
        self.open_camera(0)
        self.refresh_label_list()
        self.refresh_prob_bars()
        self.root.bind_all("<space>", self.on_space)
        self.loop()

    # ---------- UI ----------
    def build_ui(self):
        style = ttk.Style()
        style.configure("Big.TLabel", font=("Malgun Gothic", 22, "bold"))
        style.configure("Rec.TButton", font=("Malgun Gothic", 12, "bold"))

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = ttk.Label(left)
        self.video.pack()
        self.status = ttk.Label(left, text="", anchor="w")
        self.status.pack(fill="x", pady=(4, 0))

        right = ttk.Frame(main, width=360, padding=(10, 0, 0, 0))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        cam_row = ttk.Frame(right)
        cam_row.pack(fill="x", pady=(0, 8))
        ttk.Label(cam_row, text="카메라").pack(side="left")
        self.camera_var = tk.StringVar(value="0")
        cam_box = ttk.Combobox(cam_row, textvariable=self.camera_var, values=[str(i) for i in range(5)],
                               width=4, state="readonly")
        cam_box.pack(side="left", padx=6)
        cam_box.bind("<<ComboboxSelected>>", lambda e: self.open_camera(int(self.camera_var.get())))

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.set_recording(False))
        self.build_collect_tab()
        self.build_train_tab()
        self.build_recognize_tab()

    def build_collect_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="  1. 수집  ")

        ttk.Label(tab, text="제스처 이름 (새 이름을 입력하거나 목록에서 선택)").pack(anchor="w")
        self.label_var = tk.StringVar()
        self.label_box = ttk.Combobox(tab, textvariable=self.label_var, font=("Malgun Gothic", 12))
        self.label_box.pack(fill="x", pady=(2, 8))

        target_row = ttk.Frame(tab)
        target_row.pack(fill="x", pady=(0, 8))
        ttk.Label(target_row, text="이번 녹화 목표 개수 (0 = 무제한)").pack(side="left")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(target_row, from_=0, to=5000, increment=50, textvariable=self.target_var,
                    width=7).pack(side="right")

        self.rec_button = ttk.Button(tab, text="● 녹화 시작  (SPACE)", style="Rec.TButton",
                                     command=lambda: self.set_recording(not self.recording))
        self.rec_button.pack(fill="x", ipady=8)
        self.rec_info = ttk.Label(tab, text="", foreground="#c00")
        self.rec_info.pack(anchor="w", pady=(4, 10))

        ttk.Label(tab, text="수집된 데이터").pack(anchor="w")
        self.label_tree = ttk.Treeview(tab, columns=("label", "count"), show="headings", height=8)
        self.label_tree.heading("label", text="제스처")
        self.label_tree.heading("count", text="샘플 수")
        self.label_tree.column("label", width=180)
        self.label_tree.column("count", width=80, anchor="e")
        self.label_tree.pack(fill="both", expand=True, pady=(2, 6))
        self.label_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        ttk.Button(tab, text="선택한 제스처 데이터 삭제", command=self.delete_selected_label).pack(fill="x")

        ttk.Label(tab, foreground="#666", wraplength=320, justify="left", text=(
            "팁: 제스처당 300개 이상, 손 각도/거리를 바꿔가며 녹화하세요.\n"
            "아무 동작도 아닌 손 모양을 'none' 같은 이름으로 함께 수집하면 오인식이 줄어듭니다."
        )).pack(anchor="w", pady=(10, 0))

    def build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="  2. 학습  ")

        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="최대 에포크").pack(side="left")
        self.epochs_var = tk.IntVar(value=300)
        ttk.Spinbox(row, from_=10, to=2000, increment=50, textvariable=self.epochs_var,
                    width=7).pack(side="right")

        self.train_button = ttk.Button(tab, text="학습 시작", style="Rec.TButton", command=self.start_training)
        self.train_button.pack(fill="x", ipady=8)
        self.train_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.train_progress.pack(fill="x", pady=6)

        self.train_log = tk.Text(tab, height=20, font=("Consolas", 9), wrap="none")
        self.train_log.pack(fill="both", expand=True)

    def build_recognize_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="  3. 인식  ")

        self.result_label = ttk.Label(tab, text="-", style="Big.TLabel", anchor="center")
        self.result_label.pack(fill="x", pady=(0, 10))

        ttk.Label(tab, text="최소 확신도 (이보다 낮으면 Unknown)").pack(anchor="w")
        self.threshold_var = tk.DoubleVar(value=0.7)
        thr_row = ttk.Frame(tab)
        thr_row.pack(fill="x", pady=(2, 10))
        ttk.Scale(thr_row, from_=0.3, to=0.99, variable=self.threshold_var).pack(side="left", fill="x", expand=True)
        self.threshold_text = ttk.Label(thr_row, width=5)
        self.threshold_text.pack(side="right")

        ttk.Label(tab, text="제스처별 확률 (첫 번째 손)").pack(anchor="w")
        self.prob_frame = ttk.Frame(tab)
        self.prob_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.prob_bars = {}

    # ---------- state helpers ----------
    def open_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("카메라 오류", f"카메라 {index}번을 열 수 없습니다.")

    def refresh_label_list(self):
        self.label_tree.delete(*self.label_tree.get_children())
        for label, n in sorted(self.store.counts.items()):
            self.label_tree.insert("", "end", iid=label, values=(label, n))
        self.label_box["values"] = sorted(self.store.counts)

    def update_label_count(self, label):
        if self.label_tree.exists(label):
            self.label_tree.item(label, values=(label, self.store.counts[label]))
        else:
            self.refresh_label_list()

    def refresh_prob_bars(self):
        for child in self.prob_frame.winfo_children():
            child.destroy()
        self.prob_bars = {}
        if self.classifier is None:
            ttk.Label(self.prob_frame, text="학습된 모델이 없습니다.\n'2. 학습' 탭에서 먼저 학습하세요.",
                      foreground="#666").pack(anchor="w")
            return
        for label in self.classifier.classes_:
            row = ttk.Frame(self.prob_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=str(label), width=14).pack(side="left")
            bar = ttk.Progressbar(row, maximum=1.0)
            bar.pack(side="left", fill="x", expand=True)
            self.prob_bars[str(label)] = bar

    def set_recording(self, on):
        if on:
            label = self.label_var.get().strip()
            if not label:
                messagebox.showwarning("제스처 이름 필요", "녹화할 제스처 이름을 입력하세요.")
                return
            if "," in label:
                messagebox.showwarning("이름 오류", "제스처 이름에 쉼표(,)는 쓸 수 없습니다.")
                return
            if self.tabs.index("current") != TAB_COLLECT:
                return
            self.label_var.set(label)
            self.session_count = 0
            self.root.focus_set()  # so SPACE toggles instead of typing into the entry
        self.recording = on
        self.rec_button.config(text="■ 녹화 정지  (SPACE)" if on else "● 녹화 시작  (SPACE)")
        if not on:
            self.rec_info.config(text="")

    def on_space(self, event):
        if isinstance(event.widget, (ttk.Entry, tk.Entry, tk.Text)):
            return
        if self.tabs.index("current") == TAB_COLLECT:
            self.set_recording(not self.recording)
            return "break"

    def on_tree_select(self, _event):
        selection = self.label_tree.selection()
        if selection:
            self.label_var.set(selection[0])

    def delete_selected_label(self):
        selection = self.label_tree.selection()
        if not selection:
            return
        label = selection[0]
        if messagebox.askyesno("삭제 확인", f"'{label}' 데이터 {self.store.counts[label]}개를 삭제할까요?"):
            self.set_recording(False)
            self.store.delete(label)
            self.refresh_label_list()

    # ---------- training ----------
    def start_training(self):
        if self.training:
            return
        self.set_recording(False)
        self.training = True
        self.train_button.config(state="disabled")
        self.train_progress.start(10)
        self.train_log.delete("1.0", "end")
        epochs = self.epochs_var.get()

        def worker():
            try:
                model = train_gestures.train(epochs=epochs, log=self.train_queue.put)
                self.train_queue.put(("done", model))
            except Exception as e:  # report any failure in the log
                self.train_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_training(self):
        while not self.train_queue.empty():
            msg = self.train_queue.get()
            if isinstance(msg, tuple):
                kind, payload = msg
                self.training = False
                self.train_button.config(state="normal")
                self.train_progress.stop()
                if kind == "done":
                    self.classifier = payload
                    self.refresh_prob_bars()
                    self.train_log.insert("end", "\n학습 완료! '3. 인식' 탭에서 확인하세요.\n")
                else:
                    self.train_log.insert("end", f"\n오류: {payload}\n")
            else:
                self.train_log.insert("end", msg + "\n")
            self.train_log.see("end")

    # ---------- main loop ----------
    def loop(self):
        self.poll_training()
        self.threshold_text.config(text=f"{self.threshold_var.get():.2f}")

        ok, frame = (self.cap.read() if self.cap is not None and self.cap.isOpened() else (False, None))
        if ok:
            self.process_frame(cv2.flip(frame, 1))
        self.root.after(5, self.loop)

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - self.start_time) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        tab = self.tabs.index("current")
        texts = []  # (xy, text, color, font) drawn with PIL so Korean labels render
        now = time.monotonic()

        for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            hand_name = handedness[0].category_name
            color = (0, 255, 0)

            if tab == TAB_COLLECT and self.recording and i == 0:
                color = (0, 0, 255)
                if now - self.last_saved >= COLLECT_INTERVAL:
                    label = self.label_var.get().strip()
                    self.store.add(label, landmarks_to_features(landmarks, hand_name))
                    self.session_count += 1
                    self.last_saved = now
                    target = self.target_var.get()
                    if target and self.session_count >= target:
                        self.set_recording(False)
                    self.update_label_count(label)

            points = draw_hand(frame, landmarks, color)
            anchor = (min(p[0] for p in points), max(min(p[1] for p in points) - 40, 0))

            if tab == TAB_RECOGNIZE and self.classifier is not None:
                probs = self.classifier.predict_proba([landmarks_to_features(landmarks, hand_name)])[0]
                best = probs.argmax()
                label = str(self.classifier.classes_[best]) if probs[best] >= self.threshold_var.get() else "Unknown"
                texts.append((anchor, f"{hand_name}: {label} ({probs[best]:.2f})", (255, 255, 0), self.font_small))
                if i == 0:
                    self.result_label.config(text=label)
                    for cls, p in zip(self.classifier.classes_, probs):
                        self.prob_bars[str(cls)]["value"] = float(p)

        if tab == TAB_RECOGNIZE and not result.hand_landmarks:
            self.result_label.config(text="-")
            for bar in self.prob_bars.values():
                bar["value"] = 0

        if tab == TAB_COLLECT and self.recording:
            label = self.label_var.get().strip()
            self.rec_info.config(text=f"녹화 중: '{label}' 이번 {self.session_count}개 / 전체 {self.store.counts[label]}개")
            if not result.hand_landmarks:
                texts.append(((10, 50), "손이 보이지 않습니다", (255, 80, 80), self.font_small))
            texts.append(((10, 10), f"● REC  {label}", (255, 60, 60), self.font_big))

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(VIDEO_SIZE)
        scale_x, scale_y = VIDEO_SIZE[0] / frame.shape[1], VIDEO_SIZE[1] / frame.shape[0]
        draw = ImageDraw.Draw(image)
        for (x, y), text, fill, font in texts:
            draw.text((x * scale_x, y * scale_y), text, font=font, fill=fill,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        self.photo = ImageTk.PhotoImage(image)
        self.video.config(image=self.photo)

        hands = len(result.hand_landmarks)
        model_info = f"모델: {', '.join(map(str, self.classifier.classes_))}" if self.classifier is not None else "모델: 없음"
        self.status.config(text=f"감지된 손: {hands}    |    {model_info}")

    def on_close(self):
        self.recording = False
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GestureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

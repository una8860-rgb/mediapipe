"""Local server for the web Gesture Studio (web/studio.html).

Serves the web page plus a small API so the browser can store samples in
data/gestures.csv and train with train_gestures.py - the same data and model
files the desktop gesture_app.py uses.

Usage:
    python web_server.py [--port 8000]
    then open http://localhost:8000/
"""
import argparse
import json
import math
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import train_gestures
from gesture_common import DATA_PATH, NUM_FEATURES, ROOT, SampleStore

STATIC_PREFIXES = ("/web/", "/models/")
store = SampleStore(DATA_PATH)
store_lock = threading.Lock()
train_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/") or "error" in fmt.lower():
            super().log_message(fmt, *args)

    # ---------- helpers ----------
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def counts(self):
        return dict(sorted(store.counts.items()))

    def end_headers(self):
        if self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---------- routes ----------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/web/studio.html")
            self.end_headers()
        elif path == "/api/labels":
            self.send_json({"counts": self.counts()})
        elif path.startswith(STATIC_PREFIXES):
            super().do_GET()
        else:
            self.send_error(404)

    def do_HEAD(self):
        if urlparse(self.path).path.startswith(STATIC_PREFIXES):
            super().do_HEAD()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_json()
        except json.JSONDecodeError:
            return self.send_json({"error": "invalid JSON"}, 400)

        if path == "/api/samples":
            label = str(body.get("label", "")).strip()
            rows = body.get("rows", [])
            if not label or "," in label:
                return self.send_json({"error": "제스처 이름이 비었거나 쉼표가 들어 있습니다."}, 400)
            if not all(isinstance(r, list) and len(r) == NUM_FEATURES
                       and all(isinstance(v, (int, float)) and math.isfinite(v) for v in r) for r in rows):
                return self.send_json({"error": f"each row must be {NUM_FEATURES} numbers"}, 400)
            with store_lock:
                store.add_many(label, rows)
                return self.send_json({"counts": self.counts()})

        if path == "/api/train":
            if not train_lock.acquire(blocking=False):
                return self.send_json({"ok": False, "error": "이미 학습 중입니다.", "log": ""}, 409)
            log = []
            try:
                epochs = int(body.get("epochs", 300))
                with store_lock:
                    model = train_gestures.train(epochs=epochs, log=log.append)
                self.send_json({"ok": True, "log": "\n".join(log),
                                "classes": [str(c) for c in model.classes_]})
            except Exception as e:  # surface any training failure to the page
                self.send_json({"ok": False, "error": str(e), "log": "\n".join(log)})
            finally:
                train_lock.release()
            return

        self.send_error(404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/labels/"):
            label = unquote(path[len("/api/labels/"):])
            with store_lock:
                if label in store.counts:
                    store.delete(label)
                return self.send_json({"counts": self.counts()})
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"Gesture Studio running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

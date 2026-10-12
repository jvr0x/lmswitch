"""HTTP front for a decision model. Speaks POST /v1/systemone.

One process, one model, started by the systemone runtime. The model is
loaded before the socket opens, so a 200 from GET /v1/models means the
weights are actually resident.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None
_KIND = ""
_NAME = ""


def _b64_bytes(value: str) -> bytes:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _pil_images(raw) -> list:
    from PIL import Image

    images = []
    for item in raw or []:
        if not isinstance(item, str):
            raise ValueError("images must be base64 or data URLs")
        images.append(Image.open(io.BytesIO(_b64_bytes(item))).convert("RGB"))
    return images


def _video_frames(raw: str, fps: float = 2.0, max_frames: int = 16):
    import numpy as np

    blob = _b64_bytes(raw)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clip.mp4"
        src.write_bytes(blob)
        out = Path(tmp) / "f%03d.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), "-vf", f"fps={fps},scale=480:-2",
             "-frames:v", str(max_frames), str(out)],
            check=True,
        )
        frames = []
        for path in sorted(Path(tmp).glob("f*.jpg")):
            from PIL import Image
            frames.append(np.asarray(Image.open(path).convert("RGB")))
        if len(frames) < 2:
            raise ValueError("video produced fewer than 2 frames")
        return np.stack(frames)


def _load() -> None:
    global _MODEL, _PROCESSOR, _KIND, _NAME
    import torch

    path = os.environ["SYSTEMONE_MODEL"]
    _KIND = os.environ.get("SYSTEMONE_KIND", "d1")
    _NAME = os.environ.get("SYSTEMONE_NAME", Path(path).name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if _KIND in ("d1", "d3"):
        from transformers import AutoModel
        _MODEL = AutoModel.from_pretrained(
            path, trust_remote_code=True, dtype=dtype).to(device).eval()
        return
    if _KIND == "clef":
        import sys
        sys.path.insert(0, path)
        from joint_schema_model import load_release_model
        _MODEL, _PROCESSOR = load_release_model(path, device=device, dtype=dtype)
        return
    raise SystemExit(f"unknown SYSTEMONE_KIND {_KIND}")


def _decide(body: dict) -> dict:
    started = time.perf_counter()
    images = _pil_images(body.get("images"))
    video = body.get("video")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    state = body.get("state", "")
    import torch
    with _LOCK, torch.inference_mode():
        if _KIND in ("d1", "d3"):
            if video and _KIND == "d1":
                raise ValueError("d1-3B does not take video")
            kwargs = {}
            if images:
                kwargs["images"] = images
            if video and _KIND == "d3":
                kwargs["videos"] = [video]
            if _KIND == "d3":
                # d3's system_one takes keyword-only arguments.
                result = _MODEL.system_one(state=state, questions=questions, **kwargs)
            else:
                result = _MODEL.system_one(state, questions, **kwargs)
            result["model"] = _NAME
        else:
            from joint_schema_model import systemone
            request = {
                "model": _NAME,
                "state": state if state is not None else "",
                "questions": questions,
            }
            if images:
                request["images"] = images
            if video:
                request["videos"] = [_video_frames(video)]
            result = systemone(_MODEL, _PROCESSOR, request)
    result.setdefault("usage", {})
    result["usage"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in ("/health", "/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": _NAME, "object": "model"}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, _decide(body))
        except Exception as exc:
            self._send(400, {"error": str(exc)})


def main() -> None:
    port = int(os.environ.get("SYSTEMONE_PORT", "8170"))
    print(f"loading {_KIND or os.environ.get('SYSTEMONE_KIND')} from {os.environ.get('SYSTEMONE_MODEL')}", flush=True)
    _load()
    print(f"serving /v1/systemone on {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), _Handler).serve_forever()


if __name__ == "__main__":
    main()

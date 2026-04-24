"""
WebSocket server that streams live pipeline data to the React dashboard.

Usage:
    python server.py                           # webcam 0, default settings
    python server.py --source 0                # webcam
    python server.py --source rtsp://...       # IP camera
    python server.py --port 8765               # custom port
    python server.py --no-window               # headless (no cv2 window)

The server runs the full PosePipeline and streams JSON frames + JPEG images
over WebSocket to any connected dashboard client.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import threading
import time
from typing import Optional, Union

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import Settings, CombinedAnalyzerConfig
from app.pipeline import PosePipeline
from utils.schemas import ObjectDetection

_DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "dist")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)-40s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Baby Guardian API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built React dashboard from /
if os.path.isdir(_DASHBOARD_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DASHBOARD_DIR, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        # API routes take priority — this only catches unmatched paths
        index = os.path.join(_DASHBOARD_DIR, "index.html")
        return FileResponse(index)
else:
    logger.warning("Dashboard build not found at %s — open dashboard separately with npm run dev", _DASHBOARD_DIR)

# ── Shared state between pipeline thread and WebSocket ──────────────────────

class PipelineState:
    """Thread-safe container for the latest frame data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_jpg: Optional[bytes] = None
        self._analysis_json: Optional[str] = None
        self._fps: float = 0.0
        self._running = False
        self._frame_index = 0
        self._source: Optional[Union[int, str]] = None
        self._last_error: Optional[str] = None

    def update(self, frame_jpg: bytes, analysis_json: str, fps: float, frame_index: int):
        with self._lock:
            self._frame_jpg = frame_jpg
            self._analysis_json = analysis_json
            self._fps = fps
            self._frame_index = frame_index

    def clear_stream(self):
        with self._lock:
            self._frame_jpg = None
            self._analysis_json = None
            self._fps = 0.0
            self._frame_index = 0

    @property
    def source(self):
        with self._lock:
            return self._source

    @source.setter
    def source(self, val):
        with self._lock:
            self._source = val

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    @last_error.setter
    def last_error(self, val):
        with self._lock:
            self._last_error = val

    def get(self) -> tuple[Optional[bytes], Optional[str], float, int]:
        with self._lock:
            return self._frame_jpg, self._analysis_json, self._fps, self._frame_index

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, val):
        with self._lock:
            self._running = val


state = PipelineState()


class StreamSourceRequest(BaseModel):
    source: str


# ── Frame grabber (separate thread to always hold latest frame) ─────────────

class FrameGrabber:
    """
    Continuously reads frames from the camera in a dedicated thread.
    Always returns the LATEST frame, discarding stale buffered ones.
    This eliminates the 5-10 second lag from OpenCV's internal buffer on IP cams.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        """Return the latest frame (never stale)."""
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=3.0)


# ── Pipeline runner (background thread) ─────────────────────────────────────

def run_pipeline(source: Union[int, str], settings: Settings, show_window: bool):
    combined_cfg = settings.combined
    """Run the detection pipeline in a background thread, pushing results to state."""
    state.source = source
    state.last_error = None
    state.clear_stream()
    pipeline = PosePipeline(settings)

    # Open camera with minimal buffer
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        message = f"Could not open video source: {source}"
        logger.error(message)
        state.last_error = message
        state.running = False
        return

    # Minimize OpenCV internal buffer (key for IP cam latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    is_ip_cam = isinstance(source, str)

    # Use frame grabber thread for IP cams to always get the latest frame
    grabber = FrameGrabber(cap) if is_ip_cam else None

    logger.info("Pipeline started on source: %s (ip_cam=%s)", source, is_ip_cam)
    state.running = True

    frame_idx = 0
    fps_display = 0.0

    WIN_NAME = "Baby Monitor — Pose Analysis"
    if show_window:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 540)

    try:
        while state.running:
            # Get frame: use grabber for IP cam, direct read for webcam
            if grabber:
                ret, frame = grabber.read()
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                if is_ip_cam:
                    time.sleep(0.05)
                    continue
                break

            t0 = time.time()
            timestamp_ms = t0 * 1000

            # Run detection
            poses = pipeline.detector.detect(frame)
            detections = pipeline._run_object_detection(frame)
            analysis, pairs = pipeline._build_analysis(
                poses, frame_idx, timestamp_ms, detections=detections,
            )

            # Annotate frame
            annotated = pipeline._annotate(frame, pairs, detections)

            # FPS
            elapsed = time.time() - t0
            instant_fps = 1.0 / max(elapsed, 1e-6)
            fps_display = 0.7 * fps_display + 0.3 * instant_fps

            # Alerts
            for i, person in enumerate(analysis.persons):
                risk = person["risk"]
                pipeline.alert_mgr.check(
                    person_index=i,
                    label=risk["label"],
                    score=risk["score"],
                    reasons=risk["reasons"],
                    frame=annotated,
                )

            # Encode annotated frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, jpg_buf = cv2.imencode('.jpg', annotated, encode_params)
            frame_jpg = jpg_buf.tobytes()

            # Build JSON payload
            payload = {
                "frame_index": frame_idx,
                "timestamp_ms": timestamp_ms,
                "fps": round(fps_display, 1),
                "persons": analysis.persons,
                "detections": [_det_to_dict(d, combined_cfg) for d in analysis.detections],
                "status": _compute_overall_status(analysis.persons),
            }
            analysis_json = json.dumps(payload, default=str)

            # Push to shared state
            state.update(frame_jpg, analysis_json, fps_display, frame_idx)

            # Optional cv2 window
            if show_window:
                cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(WIN_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1

    finally:
        state.running = False
        if grabber:
            grabber.stop()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        pipeline._cleanup()
        logger.info("Pipeline stopped.")


class PipelineController:
    """Owns the single active pipeline thread and supports safe restarts."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._settings: Optional[Settings] = None
        self._show_window = True

    def configure(self, settings: Settings, show_window: bool):
        with self._lock:
            self._settings = copy.deepcopy(settings)
            self._show_window = show_window

    def start(self, source: Union[int, str]):
        with self._lock:
            if self._settings is None:
                raise RuntimeError("Pipeline controller is not configured")

            self._stop_locked()
            state.last_error = None
            state.source = source
            state.clear_stream()

            thread = threading.Thread(
                target=run_pipeline,
                args=(source, copy.deepcopy(self._settings), self._show_window),
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self):
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        state.running = False
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        state.clear_stream()


controller = PipelineController()


def _classify_detection_risk(class_name: str, cfg: CombinedAnalyzerConfig) -> str:
    """
    Classify a detected object as 'danger', 'hazard', or 'safe'.

    danger  — object that can cause suffocation if over baby's face
    hazard  — object that is dangerous near a baby (sharp, chokable, etc.)
    safe    — person or neutral object
    """
    name = class_name.lower()
    if name in {c.lower() for c in cfg.face_danger_classes}:
        return "danger"
    if name in {c.lower() for c in cfg.hazard_classes}:
        return "hazard"
    return "safe"


def _det_to_dict(det, cfg: CombinedAnalyzerConfig) -> dict:
    """Convert ObjectDetection to a JSON-safe dict with risk_level."""
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": list(det.bbox),
        "track_id": getattr(det, 'track_id', None),
        "risk_level": _classify_detection_risk(det.class_name, cfg),
    }


def _compute_overall_status(persons: list[dict]) -> str:
    """Determine worst-case status across all persons."""
    if not persons:
        return "safe"
    labels = [p["risk"]["label"] for p in persons]
    if "dangerous" in labels:
        return "danger"
    if "uncertain" in labels:
        return "warning"
    return "safe"


# ── WebSocket endpoints ─────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Stream live pipeline data to the dashboard.

    Sends JSON messages with analysis data + base64-encoded JPEG frame.
    """
    await websocket.accept()
    logger.info("Dashboard client connected")

    last_frame_index = -1

    try:
        while True:
            frame_jpg, analysis_json, fps, frame_index = state.get()

            # Only send new frames
            if frame_index > last_frame_index and analysis_json is not None:
                last_frame_index = frame_index

                # Send analysis data
                await websocket.send_text(analysis_json)

                # Send frame as binary (dashboard decodes as JPEG blob)
                if frame_jpg is not None:
                    await websocket.send_bytes(frame_jpg)

            await asyncio.sleep(0.03)  # ~30 updates/sec max to dashboard

    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)


@app.get("/api/status")
async def get_status():
    """Health check + current pipeline status."""
    _, analysis_json, fps, frame_index = state.get()
    return {
        "running": state.running,
        "fps": fps,
        "frame_index": frame_index,
        "source": state.source,
        "last_error": state.last_error,
    }


@app.post("/api/source")
async def set_source(payload: StreamSourceRequest):
    """Start or restart the pipeline with a new camera source."""
    raw_source = payload.source.strip()
    if not raw_source:
        return {"ok": False, "error": "Please enter a camera index or live stream URL."}

    source: Union[int, str] = raw_source
    try:
        source = int(raw_source)
    except ValueError:
        pass

    controller.start(source)

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if state.running and state.source == source:
            return {"ok": True, "source": str(source), "running": True}
        if state.last_error:
            return {"ok": False, "error": state.last_error}
        await asyncio.sleep(0.1)

    return {
        "ok": state.running and state.source == source,
        "source": str(source),
        "running": state.running,
        "error": state.last_error,
    }


@app.get("/api/snapshot")
async def get_snapshot():
    """
    Serve the latest alert snapshot image.

    This endpoint is used by Twilio to fetch snapshot images when sending
    WhatsApp alerts with media attachments. When using ngrok or a public
    server, set WHATSAPP_SNAPSHOT_URL to the public URL of this endpoint
    (e.g. https://abc123.ngrok.io/api/snapshot).
    """
    snapshot_path = Settings().whatsapp.snapshot_path
    if os.path.isfile(snapshot_path):
        return FileResponse(snapshot_path, media_type="image/jpeg")
    return {"error": "No snapshot available"}


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Baby Guardian — WebSocket Server",
        epilog="""
Examples:
  python server.py --no-window                          # webcam, headless
  python server.py --source "http://192.168.1.5:8080/video" --no-window   # IP Webcam app (Android)
  python server.py --source "http://192.168.1.5:4747/video" --no-window   # DroidCam
  python server.py --source "rtsp://user:pass@192.168.1.10:554/stream"    # RTSP camera
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", "-s", default="0",
                   help="Camera index (0,1,..) or IP camera URL "
                        "(e.g. http://192.168.1.5:8080/video for IP Webcam app)")
    p.add_argument("--port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    p.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    p.add_argument("--no-window", "--nowindow", action="store_true",
                   help="Run headless — no cv2 window (use dashboard only)")
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--obj-conf", type=float, default=0.35)
    p.add_argument("--model-path", default="yolov8m.pt")
    p.add_argument("--device", default="auto")

    # ── WhatsApp alerts ──
    p.add_argument("--whatsapp", action="store_true",
                   help="Enable WhatsApp alerts via Twilio")
    p.add_argument("--wa-sid", default="",
                   help="Twilio Account SID (or set TWILIO_ACCOUNT_SID env var)")
    p.add_argument("--wa-token", default="",
                   help="Twilio Auth Token (or set TWILIO_AUTH_TOKEN env var)")
    p.add_argument("--wa-from", default="whatsapp:+14155238886",
                   help="Twilio WhatsApp sender (default: sandbox number)")
    p.add_argument("--wa-to", default="",
                   help="Your WhatsApp number e.g. whatsapp:+923001234567")
    p.add_argument("--wa-cooldown", type=float, default=30.0,
                   help="Seconds between WhatsApp messages (default: 30)")
    p.add_argument("--wa-all-levels", action="store_true",
                   help="Send WhatsApp for UNCERTAIN too, not just DANGEROUS")
    p.add_argument("--wa-snapshot-url", default="",
                   help="Public URL for snapshot images (e.g. https://abc.ngrok.io/api/snapshot)")
    return p


def main():
    args = build_parser().parse_args()

    # Parse source
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    # Build settings
    settings = Settings()
    settings.pose_model.model_complexity = args.complexity
    settings.object_detection.confidence_threshold = args.obj_conf
    settings.object_detection.model_path = args.model_path
    settings.object_detection.device = args.device

    # WhatsApp settings — auto-enable if .env has credentials
    has_env_creds = bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("WHATSAPP_TO"))
    settings.whatsapp.enabled = args.whatsapp or has_env_creds
    settings.whatsapp.account_sid = args.wa_sid
    settings.whatsapp.auth_token = args.wa_token
    settings.whatsapp.from_number = args.wa_from
    settings.whatsapp.to_number = args.wa_to
    settings.whatsapp.whatsapp_cooldown_seconds = args.wa_cooldown
    settings.whatsapp.only_on_dangerous = not args.wa_all_levels
    settings.whatsapp.snapshot_url = args.wa_snapshot_url

    show_window = not args.no_window
    controller.configure(settings, show_window)
    controller.start(source)

    logger.info("Starting server on http://%s:%d", args.host, args.port)
    if os.path.isdir(_DASHBOARD_DIR):
        logger.info("Dashboard → http://localhost:%d", args.port)
    else:
        logger.warning("Dashboard not built — run: cd dashboard && npm run build")

    # Run FastAPI server (blocks main thread)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

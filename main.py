"""
CLI entrypoint for the baby monitor pipeline.

Usage:
    python main.py image  --source path/to/image.jpg
    python main.py video  --source path/to/video.mp4
    python main.py live   [--source 0]
    python main.py live   --source "rtsp://user:pass@192.168.1.10:554/stream"
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv
load_dotenv()

from config.settings import Settings
from utils.log_setup import setup_logging
from app.pipeline import PosePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baby Monitor — Pose Estimation + Object Detection Pipeline",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        # ── Pose detection ──
        sp.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy")
        sp.add_argument("--num-poses", type=int, default=3,
                        help="Max number of poses (people) to detect per frame (default: 3)")
        sp.add_argument("--output", default="output", help="Output directory")
        sp.add_argument("--json", action="store_true", dest="print_json",
                        help="Print structured JSON output")

        # ── Object detection ──
        sp.add_argument("--no-object-detection", action="store_true",
                        help="Disable YOLO object detection (pose-only mode)")
        sp.add_argument("--obj-conf", type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
        sp.add_argument("--model-path", default="yolov8m.pt",
                        help="Path to YOLO model weights (default: yolov8m.pt, auto-downloads)")
        # To use custom trained model instead:
        # sp.add_argument("--model-path", default="best.pt")
        sp.add_argument("--obj-imgsz", type=int, default=320,
                        help="YOLO input size in px (320=fast, 640=accurate, default: 320)")
        sp.add_argument("--obj-skip", type=int, default=3,
                        help="Run YOLO every N frames (default: 3)")
        sp.add_argument("--no-async", action="store_true",
                        help="Disable threaded YOLO inference")
        sp.add_argument("--no-tracking", action="store_true",
                        help="Disable YOLO BoT-SORT tracking")

        # ── Device ──
        sp.add_argument("--device", default="auto",
                        help="Compute device: auto, cpu, cuda, cuda:0 (default: auto)")
        sp.add_argument("--no-half", action="store_true",
                        help="Disable FP16 half-precision on GPU")

        # ── Alert system ──
        sp.add_argument("--alert-cooldown", type=float, default=5.0,
                        help="Seconds between repeated alerts for same person (default: 5.0)")

        # ── WhatsApp alerts (Twilio) ──
        sp.add_argument("--whatsapp", action="store_true",
                        help="Enable WhatsApp alerts via Twilio")
        sp.add_argument("--wa-sid", default="",
                        help="Twilio Account SID (or set TWILIO_ACCOUNT_SID env var)")
        sp.add_argument("--wa-token", default="",
                        help="Twilio Auth Token (or set TWILIO_AUTH_TOKEN env var)")
        sp.add_argument("--wa-from", default="whatsapp:+14155238886",
                        help="Twilio WhatsApp sender number (default: sandbox number)")
        sp.add_argument("--wa-to", default="",
                        help="Your WhatsApp number e.g. whatsapp:+923001234567")
        sp.add_argument("--wa-cooldown", type=float, default=30.0,
                        help="Seconds between WhatsApp messages (default: 30)")
        sp.add_argument("--wa-all-levels", action="store_true",
                        help="Send WhatsApp for UNCERTAIN too, not just DANGEROUS")
        sp.add_argument("--wa-snapshot-url", default="",
                        help="Public URL for snapshot images (e.g. https://abc.ngrok.io/api/snapshot)")

        # ── Logging ──
        sp.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
        sp.add_argument("--log-file", default=None,
                        help="Optional log file path")

    # Image
    img_parser = subparsers.add_parser("image", help="Analyze a single image")
    img_parser.add_argument("--source", required=True, help="Path to image file")
    add_common_args(img_parser)

    # Video
    vid_parser = subparsers.add_parser("video", help="Analyze a video file")
    vid_parser.add_argument("--source", required=True, help="Path to video file")
    vid_parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    add_common_args(vid_parser)

    # Live
    live_parser = subparsers.add_parser("live", help="Real-time camera analysis")
    live_parser.add_argument("--source", default="0",
                             help="Camera index (e.g. 0) or IP camera / RTSP URL")
    add_common_args(live_parser)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Setup logging first
    setup_logging(level=args.log_level, log_file=args.log_file)

    settings = Settings()

    # Pose
    settings.pose_model.model_complexity = args.complexity
    settings.pose_model.num_poses = args.num_poses
    settings.output_dir = args.output

    # Object detection
    settings.object_detection.enabled = not args.no_object_detection
    settings.object_detection.confidence_threshold = args.obj_conf
    settings.object_detection.model_path = args.model_path
    settings.object_detection.imgsz = args.obj_imgsz
    settings.object_detection.skip_frames = args.obj_skip
    settings.object_detection.async_inference = not args.no_async
    settings.object_detection.enable_tracking = not args.no_tracking
    settings.object_detection.device = args.device
    settings.object_detection.half_precision = not args.no_half

    # Alert
    settings.alert.cooldown_seconds = args.alert_cooldown

    # WhatsApp — auto-enable if .env has credentials
    import os
    has_env_creds = bool(os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("WHATSAPP_TO"))
    settings.whatsapp.enabled = args.whatsapp or has_env_creds
    settings.whatsapp.account_sid = args.wa_sid
    settings.whatsapp.auth_token = args.wa_token
    settings.whatsapp.from_number = args.wa_from
    settings.whatsapp.to_number = args.wa_to
    settings.whatsapp.whatsapp_cooldown_seconds = args.wa_cooldown
    settings.whatsapp.only_on_dangerous = not args.wa_all_levels
    settings.whatsapp.snapshot_url = args.wa_snapshot_url

    pipeline = PosePipeline(settings)

    if args.mode == "image":
        result = pipeline.analyze_image(args.source)
        _print_summary(result, print_json=args.print_json)

    elif args.mode == "video":
        results = pipeline.analyze_video(args.source, max_frames=args.max_frames)
        print(f"\nProcessed {len(results)} frames.")
        alerts = [r for r in results if any(p["risk"]["label"] != "safe" for p in r.persons)]
        if alerts:
            print(f"Alerts on {len(alerts)} frames:")
            for r in alerts[:10]:
                _print_summary(r, print_json=args.print_json)
        else:
            print("All frames assessed as safe.")
        if args.print_json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))

    elif args.mode == "live":
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass
        pipeline.run_live(source=source)


def _print_summary(analysis, print_json: bool = False) -> None:
    print(f"\n--- Frame {analysis.frame_index} (t={analysis.timestamp_ms:.0f}ms) ---")

    # Object detections
    if analysis.detections:
        print(f"  Objects detected: {len(analysis.detections)}")
        for det in analysis.detections:
            track_info = f" [track {det.track_id}]" if det.track_id is not None else ""
            print(f"    - {det.class_name}{track_info} (conf={det.confidence:.2f})")

    if not analysis.persons:
        print("  No person detected.")
        return
    for i, person in enumerate(analysis.persons):
        risk = person["risk"]
        print(f"  Person {i}: {risk['label'].upper()} (score={risk['score']:.3f})")
        for reason in risk["reasons"]:
            print(f"    - {reason}")
    if analysis.annotated_frame_path:
        print(f"  Annotated output: {analysis.annotated_frame_path}")
    if print_json:
        print(analysis.to_json())


if __name__ == "__main__":
    main()

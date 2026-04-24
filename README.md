# Baby Monitor — Pose Detection Module

Standalone pose detection and risk analysis module for a baby monitoring system.
Uses **MediaPipe Pose** (33 BlazePose landmarks) to detect body keypoints, then
applies heuristic rules to classify each pose as **safe**, **dangerous**, or **uncertain**.

## Project Structure

```
pro/
├── app/
│   ├── __init__.py
│   └── pipeline.py          # Orchestrates detection → analysis → output
├── config/
│   ├── __init__.py
│   └── settings.py           # All tunable parameters
├── models/
│   └── __init__.py
├── services/
│   ├── __init__.py
│   ├── pose_detector.py       # MediaPipe Pose wrapper
│   └── pose_analyzer.py       # Heuristic risk classifier (6 rules)
├── utils/
│   ├── __init__.py
│   ├── schemas.py             # Data classes (PoseResult, RiskAssessment, etc.)
│   └── drawing.py             # Skeleton + risk label annotation
├── output/                    # Annotated images/videos
├── best.pt                    # Object detection model (not used by this module)
├── main.py                    # CLI entrypoint
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

No model download needed — MediaPipe downloads its model automatically on first run.

## Usage

### Single image

```bash
python main.py image --source path/to/baby_photo.jpg
```

### Video file

```bash
python main.py video --source path/to/baby_video.mp4
```

### Webcam (local camera)

```bash
python main.py live --source 0
```

### IP camera / RTSP stream

```bash
python main.py live --source "rtsp://user:pass@192.168.1.10:554/stream"
python main.py live --source "http://192.168.1.10:8080/video"
```

Press `q` to quit the live view.

### Options

| Flag            | Default | Description                                       |
|-----------------|---------|---------------------------------------------------|
| `--complexity`  | `1`     | MediaPipe model: 0=lite, 1=full, 2=heavy          |
| `--output`      | `output`| Directory for annotated files                      |
| `--json`        | off     | Print structured JSON to stdout                    |
| `--max-frames`  | `0`     | Max frames to process (video mode, 0=all)          |

### Higher accuracy

```bash
python main.py image --source baby.jpg --complexity 2
```

## Risk Classification Rules

The `PoseAnalyzer` runs 6 heuristic checks on each detected pose:

| Rule                  | What it detects                                        |
|-----------------------|--------------------------------------------------------|
| **Face-down**         | Nose below shoulders with flat torso layout (prone)    |
| **Neck angle**        | Head tilted >60 degrees from vertical                  |
| **Collapsed posture** | Landmarks clustered in small area of bounding box      |
| **Body inversion**    | Hips above shoulders (upside-down)                     |
| **Limb crossing**     | Both arms crossed behind body (tangling risk)          |
| **Low visibility**    | Too few landmarks visible — pose unreliable            |

Scores are averaged and compared to thresholds (configurable in `config/settings.py`):

- **safe**: score < 0.30
- **uncertain**: 0.30 <= score < 0.55
- **dangerous**: score >= 0.55

## Output Format

```json
{
  "frame_index": 0,
  "timestamp_ms": 0.0,
  "persons": [
    {
      "pose": {
        "keypoints": [
          {"name": "nose", "x": 320.5, "y": 180.2, "z": -12.3, "confidence": 0.95},
          ...
        ],
        "person_confidence": 0.87,
        "bbox": [100.0, 50.0, 400.0, 350.0]
      },
      "risk": {
        "label": "safe",
        "score": 0.1,
        "reasons": ["No risk signals detected."]
      }
    }
  ],
  "annotated_frame_path": "output/annotated_baby.jpg"
}
```

## Merging with Object Detection (best.pt)

This module is designed for easy integration with your YOLO object detection model.

### Integration strategy

```
   Frame ──► ObjectDetector (best.pt)  ──► person bounding boxes
                                                │
                                                ▼ crop each person
   Frame ──► PoseDetector (MediaPipe)  ──► PoseResult per person
                                                │
                                                ▼
                                        DecisionEngine ──► FinalVerdict
```

### Steps to merge

1. **Create an `ObjectDetector` service** wrapping `best.pt` (same pattern as `PoseDetector`).
2. **Use object detection crops as pose input**: run best.pt first to get person bounding boxes, crop each person, then feed crops to `PoseDetector.detect()`.
3. **Build a `DecisionEngine`** that combines `PoseResult` + `ObjectDetection` for unified risk (e.g. "blanket detected over face" = object + pose).
4. **Add `CombinedPipeline`** in `app/` that orchestrates both detectors.
5. **Update `config/settings.py`** with object detection parameters.

The schemas are deliberately simple dataclasses — extend with `ObjectDetection` and `CombinedAnalysis`.

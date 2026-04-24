"""
Central configuration for the baby monitor pipeline.

All thresholds, model paths, and tunable parameters live here.
Adjust these values to calibrate sensitivity for your monitoring environment.
"""

from dataclasses import dataclass, field


@dataclass
class PoseModelConfig:
    """Configuration for MediaPipe Pose."""

    # Model complexity: 0 = lite, 1 = full, 2 = heavy.
    # Higher = more accurate but slower.
    model_complexity: int = 1

    # Maximum number of poses (people) to detect per frame.
    # Set to 1 for single-baby monitoring, 2-3 if siblings/caregivers may appear.
    num_poses: int = 3

    # Minimum confidence for the person detection stage
    min_detection_confidence: float = 0.5

    # Minimum confidence for landmark tracking between frames
    min_tracking_confidence: float = 0.5

    # Enable segmentation mask output (not needed for pose-only, but available)
    enable_segmentation: bool = False

    # Minimum visibility score to consider a landmark "visible"
    landmark_visibility_threshold: float = 0.5


@dataclass
class RiskThresholds:
    """
    Tunable thresholds for the heuristic risk classifier.

    Lower values = more sensitive (more alerts). Higher = more lenient.
    """

    # --- Face-down / prone detection ---
    prone_nose_below_shoulder_margin: float = 0.02  # fraction of frame height

    # --- Keypoint visibility ---
    min_keypoint_confidence: float = 0.50
    min_visible_keypoint_ratio: float = 0.30  # ~10 of 33

    # --- Face occlusion ---
    # Minimum number of face landmarks (nose, eyes, mouth, ears) that must be
    # visible to consider the face NOT occluded.  Face has 11 landmarks (0-10).
    # If fewer than this many pass min_keypoint_confidence, _check_face_occluded fires.
    min_visible_face_landmarks: int = 4

    # --- Head turn detection ---
    # If one ear is visible and the other is not, and nose confidence is low,
    # this indicates the head is turned into a surface.
    head_turn_nose_conf_threshold: float = 0.4

    # --- Z-depth face-down detection ---
    # When nose z is significantly more positive (further from camera) than
    # shoulder midpoint z, it suggests the face is pointing away / pressed down.
    # This value is a multiplier of shoulder width used as the threshold.
    z_depth_face_away_ratio: float = 0.3

    # --- Neck angle ---
    max_neck_angle_deg: float = 60.0

    # --- Body compactness (collapsed posture) ---
    min_keypoint_spread_ratio: float = 0.15

    # --- Overall risk score ---
    dangerous_score_threshold: float = 0.55
    uncertain_score_threshold: float = 0.30

    # --- Corroboration bonus (weighted-max scoring) ---
    corroboration_bonus_per_rule: float = 0.04
    max_corroboration_bonus: float = 0.12

    # --- Temporal smoothing (EMA) ---
    ema_alpha: float = 0.4
    smoother_reset_after_frames: int = 15


@dataclass
class ObjectDetectionConfig:
    """Configuration for YOLO object detection model."""

    # Path to the YOLO model weights.
    # "yolov8m.pt" uses the pretrained COCO model (auto-downloads on first run).
    # To use a custom model, set to e.g. "best.pt"
    # model_path: str = "best.pt"  # custom trained model
    model_path: str = "yolov8m.pt"

    # Minimum confidence threshold for detections.
    # Lower this if objects aren't being detected (0.35 is good for mAP~0.55 models)
    confidence_threshold: float = 0.35

    # IoU threshold for NMS
    iou_threshold: float = 0.45

    # Enable object detection in the pipeline
    enabled: bool = True

    # YOLO input image size (smaller = faster, less accurate)
    # 320 is good for real-time; 640 for accuracy
    imgsz: int = 320

    # Run YOLO every N frames and reuse cached results in between.
    # 1 = every frame (slow), 3 = every 3rd frame (recommended), 5 = fast
    skip_frames: int = 3

    # Run YOLO in a background thread so it doesn't block the main loop
    async_inference: bool = True

    # Use GPU if available, otherwise fall back to CPU
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    # Use half-precision (FP16) on GPU for ~2x speedup. Ignored on CPU.
    half_precision: bool = True

    # Enable YOLO built-in tracking (BoT-SORT) for consistent object IDs
    enable_tracking: bool = True


@dataclass
class WhatsAppConfig:
    """
    Twilio WhatsApp notification settings.

    Setup (one-time):
      1. Sign up at https://twilio.com (free trial includes WhatsApp sandbox)
      2. Go to Messaging → Try it out → Send a WhatsApp message
      3. Join the sandbox by sending the join code from your phone
      4. Fill in account_sid, auth_token, from_number, to_number below
         OR pass them via CLI flags / environment variables.

    Environment variables (recommended over hardcoding):
      TWILIO_ACCOUNT_SID
      TWILIO_AUTH_TOKEN
      WHATSAPP_FROM       e.g. whatsapp:+14155238886  (Twilio sandbox number)
      WHATSAPP_TO         e.g. whatsapp:+923001234567 (your phone)
    """

    enabled: bool = False

    # Twilio credentials — leave empty and use env vars instead
    account_sid: str = ""
    auth_token: str = ""

    # Twilio WhatsApp sandbox number (prefix with "whatsapp:")
    from_number: str = "whatsapp:+14155238886"

    # Your phone number (prefix with "whatsapp:")
    to_number: str = ""

    # Only send WhatsApp alerts for DANGEROUS (not uncertain)
    only_on_dangerous: bool = True

    # Minimum seconds between WhatsApp messages (avoid spam)
    # Independent of the console alert cooldown
    whatsapp_cooldown_seconds: float = 30.0

    # Attach a snapshot image to the WhatsApp message
    send_snapshot: bool = True

    # Local path where snapshots are saved before sending
    snapshot_path: str = "output/alert_snapshot.jpg"

    # Public URL for Twilio to fetch snapshots. Twilio cannot access local files,
    # so this must be an internet-accessible URL pointing to /api/snapshot.
    # Set via WHATSAPP_SNAPSHOT_URL env var or --wa-snapshot-url flag.
    # Example: "https://abc123.ngrok.io/api/snapshot"
    # Leave empty to send text-only alerts.
    snapshot_url: str = ""


@dataclass
class AlertConfig:
    """Configuration for the alert cooldown / debounce system."""

    # Minimum seconds between alerts for the same person
    cooldown_seconds: float = 5.0

    # Re-alert if severity changes (e.g. uncertain → dangerous)
    realert_on_severity_change: bool = True

    # Enable sound alert (console bell) on dangerous detections
    enable_sound: bool = False


@dataclass
class CombinedAnalyzerConfig:
    """
    Configuration for cross-model reasoning (pose + object detection).

    These rules fire when specific object classes are detected near
    specific body regions (e.g. blanket overlapping with face).
    """

    # Object class names (from your YOLO model) considered dangerous near the face.
    # Using COCO class names for yolov8m.pt pretrained model.
    # (COCO doesn't have "pillow" or "blanket" — add those if using a custom model)
    face_danger_classes: list[str] = field(default_factory=lambda: [
        "teddy bear", "book", "bed", "couch",
        # "pillow", "blanket",  # uncomment when using custom best.pt
    ])

    # IoU threshold: how much an object bbox must overlap with the face region
    # for a cross-model rule to fire.
    face_object_overlap_iou: float = 0.15

    # Risk score when a dangerous object overlaps the face region
    face_object_overlap_score: float = 0.90

    # Risk score when YOLO detects a person but MediaPipe finds no pose
    # (baby is fully occluded / covered)
    person_no_pose_score: float = 0.75

    # YOLO class names that count as "person" for the above rule
    person_classes: list[str] = field(default_factory=lambda: [
        "person",
    ])

    # Object classes that are dangerous if detected NEAR a baby (not necessarily on face).
    # These trigger alerts when within proximity of the baby's body bbox.
    # Using COCO class names for yolov8m.pt.
    hazard_classes: list[str] = field(default_factory=lambda: [
        "knife", "scissors", "bottle", "fork",
        "cell phone", "remote",
        # "power plugs and sockets", "balloon", "nail", "screwdriver",  # custom model only
    ])

    # How much to expand the baby's body bbox when checking hazard proximity.
    # 0.3 = 30% expansion in each direction (catches nearby objects).
    hazard_proximity_expansion: float = 0.3

    # Risk score when a hazard object is near (overlapping expanded body bbox)
    hazard_proximity_score: float = 0.55


@dataclass
class Settings:
    """Top-level settings container."""

    pose_model: PoseModelConfig = field(default_factory=PoseModelConfig)
    risk: RiskThresholds = field(default_factory=RiskThresholds)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    combined: CombinedAnalyzerConfig = field(default_factory=CombinedAnalyzerConfig)
    whatsapp: WhatsAppConfig = field(default_factory=WhatsAppConfig)

    # Output directory for annotated images / videos
    output_dir: str = "output"

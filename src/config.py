"""
Central Configuration Module.
Defines project filesystem paths, keypoint schemas, and model hyperparameters.
"""

from pathlib import Path

# Base Filesystem Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = DATA_DIR / "labels.csv"

# Pose Estimation Keypoint Schema (COCO-17 standard)
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Model Parameters and Inference Constants
POSE_MODEL_NAME = "yolov8n-pose.pt"
MIN_KEYPOINT_CONFIDENCE = 0.25
DEFAULT_TEMPORAL_FRACTIONS = [0.5, 0.75, 1.0]
SMOOTHING_WINDOW_LENGTH = 5


def ensure_directories():
    """Ensures all necessary project directories exist."""
    for directory in [DATA_DIR, CLIPS_DIR, PROCESSED_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

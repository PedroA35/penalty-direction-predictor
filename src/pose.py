"""
Pose Detection and Landmark Extraction Module.
Uses YOLO-Pose to locate the penalty taker and extract 17 keypoints.
Includes smart caching: automatically skips clips that have already been processed.
"""

from pathlib import Path
import subprocess
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"

# COCO 17 keypoint names
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


def extract_pose_from_clip(clip_path: Path, output_video: bool = True, force: bool = False) -> pd.DataFrame:
    """
    Processes a video clip and extracts pose landmarks.
    If the pose CSV already exists and force=False, skips processing.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = clip_path.stem
    csv_out = PROCESSED_DIR / f"{clip_id}_pose.csv"
    annotated_path = CLIPS_DIR / f"{clip_id}_annotated.mp4"

    # Verificar se já foi processado anteriormente (Cache inteligente)
    if not force and csv_out.exists() and (not output_video or annotated_path.exists()):
        print(f"[SKIP] {clip_id} already processed. Skipping.")
        return pd.read_csv(csv_out)

    print(f"\n[EXTRACTING] Processing {clip_path.name}...")

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {clip_path}")
        return pd.DataFrame()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    raw_annotated_path = CLIPS_DIR / f"{clip_id}_raw_annotated.mp4"
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(raw_annotated_path), fourcc, fps, (width, height))

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        best_person_idx = None
        max_area = 0.0

        if results.boxes is not None and len(results.boxes) > 0:
            for idx, box in enumerate(results.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    best_person_idx = idx

        frame_data = {"clip_id": clip_id, "frame": frame_idx}

        if best_person_idx is not None and results.keypoints is not None:
            kpts = results.keypoints.data[best_person_idx].cpu().numpy()
            for i, name in enumerate(KEYPOINT_NAMES):
                frame_data[f"{name}_x"] = float(kpts[i][0])
                frame_data[f"{name}_y"] = float(kpts[i][1])
                frame_data[f"{name}_conf"] = float(kpts[i][2]) if len(kpts[i]) > 2 else 1.0

            if video_writer is not None:
                annotated_frame = results[best_person_idx].plot()
                video_writer.write(annotated_frame)
        else:
            for name in KEYPOINT_NAMES:
                frame_data[f"{name}_x"] = np.nan
                frame_data[f"{name}_y"] = np.nan
                frame_data[f"{name}_conf"] = 0.0

            if video_writer is not None:
                video_writer.write(frame)

        records.append(frame_data)
        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        # Convert to H.264 for universal playback
        conv_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_annotated_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(annotated_path)
        ]
        try:
            subprocess.run(conv_cmd, check=True)
            if raw_annotated_path.exists():
                raw_annotated_path.unlink()
        except Exception:
            if raw_annotated_path.exists():
                raw_annotated_path.rename(annotated_path)

        print(f"[OK] Annotated video saved: {annotated_path.name}")

    df_pose = pd.DataFrame(records)
    df_pose.to_csv(csv_out, index=False)
    print(f"[OK] Landmarks saved: {csv_out.name} ({len(df_pose)} frames)")

    return df_pose


def process_all_clips(force: bool = False):
    """Processes all raw MP4 clips, automatically skipping already cached ones."""
    clips = [
        c for c in CLIPS_DIR.glob("*.mp4")
        if not c.stem.endswith("_annotated") and not c.stem.endswith("_raw_annotated")
    ]
    if not clips:
        print("[INFO] No clips found in data/clips/.")
        return

    # Contar quantos faltam processar
    pending = [c for c in clips if not (PROCESSED_DIR / f"{c.stem}_pose.csv").exists() or force]
    print(f"Total clips: {len(clips)} | Pending to extract: {len(pending)}")

    for clip in clips:
        extract_pose_from_clip(clip, output_video=True, force=force)


if __name__ == "__main__":
    process_all_clips()

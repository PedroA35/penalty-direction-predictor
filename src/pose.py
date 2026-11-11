"""
Módulo de deteção e extração de pose do marcador de penálti.
Utiliza YOLO-Pose para localizar o marcador (selecionando a pessoa em maior plano/foco)
e extrai os 17 keypoints corporais essenciais.
Gera um vídeo de validação com o esqueleto sobreposto.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"

# Nomes dos 17 keypoints segundo o formato padrão COCO / YOLO-Pose
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


def extract_pose_from_clip(clip_path: Path, output_video: bool = True) -> pd.DataFrame:
    """
    Processa um ficheiro de vídeo, extrai as poses do marcador e opcionalmente
    gera um vídeo anotado com o esqueleto para validação visual.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = clip_path.stem

    # Carrega modelo leve YOLOv8 nano pose (corre em CPU a alta velocidade)
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {clip_path}")
        return pd.DataFrame()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if output_video:
        annotated_path = CLIPS_DIR / f"{clip_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # Selecionar o marcador: se houver várias pessoas, escolher a maior bounding box
        # Na câmara de transmissão padrão, o marcador está muito mais próximo que o guarda-redes
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
            # Coordenadas (17, 2 ou 3) -> [x, y, conf]
            kpts = results.keypoints.data[best_person_idx].cpu().numpy()

            for i, name in enumerate(KEYPOINT_NAMES):
                frame_data[f"{name}_x"] = float(kpts[i][0])
                frame_data[f"{name}_y"] = float(kpts[i][1])
                frame_data[f"{name}_conf"] = float(kpts[i][2]) if len(kpts[i]) > 2 else 1.0

            if video_writer is not None:
                # Desenha o esqueleto e caixa delimitadora no frame
                annotated_frame = results[best_person_idx].plot()
                video_writer.write(annotated_frame)
        else:
            # Caso não haja deteção no frame
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
        # O OpenCV grava em mpeg4 por defeito, que o Windows Media Player não abre sem codecs.
        # Convertemos rapidamente para H.264 (padrão universal do Windows e Web)
        import subprocess
        temp_path = annotated_path.with_suffix(".tmp.mp4")
        conv_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(annotated_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(temp_path)
        ]
        try:
            subprocess.run(conv_cmd, check=True)
            temp_path.replace(annotated_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
        print(f"[VALIDAÇÃO VISUAL] Vídeo anotado compatível guardado em: {annotated_path.name}")

    df_pose = pd.DataFrame(records)
    csv_out = PROCESSED_DIR / f"{clip_id}_pose.csv"
    df_pose.to_csv(csv_out, index=False)
    print(f"[DADOS] Pose guardada em: {csv_out.name} ({len(df_pose)} frames processados)")

    return df_pose


def process_all_clips():
    """Processa todos os clips MP4 que não sejam os de anotação."""
    clips = [c for c in CLIPS_DIR.glob("*.mp4") if not c.stem.endswith("_annotated")]
    if not clips:
        print("[INFO] Nenhum clip encontrado em data/clips/. Corre primeiro o download.py.")
        return

    print(f"A processar {len(clips)} clip(s)...")
    for clip in clips:
        print(f"\nA extrair pose de {clip.name}...")
        extract_pose_from_clip(clip, output_video=True)


if __name__ == "__main__":
    process_all_clips()


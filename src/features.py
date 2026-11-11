"""
Módulo de Engenharia de Atributos Biomecânicos.
Transforma as coordenadas brutas de pose (x, y) em ângulos articulares
e distâncias normalizadas invariantes à escala do vídeo e à distância da câmara.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = DATA_DIR / "labels.csv"


def calculate_angle_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """
    Calcula o ângulo em graus do vetor p1 -> p2 em relação ao eixo horizontal horizontal.
    Retorna um valor entre -180 e 180 graus.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


def calculate_joint_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """
    Calcula o ângulo articular em graus no vértice b formado pelos segmentos ba e bc.
    (ex: anca -> joelho -> tornozelo para flexão do joelho).
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


def extract_frame_features(row: pd.Series) -> dict[str, float]:
    """
    Extrai ângulos e proporções biomecânicas a partir de uma linha de landmarks de um frame.
    """
    # Pontos anatómicos principais
    l_shoulder = (row.get("left_shoulder_x", np.nan), row.get("left_shoulder_y", np.nan))
    r_shoulder = (row.get("right_shoulder_x", np.nan), row.get("right_shoulder_y", np.nan))
    l_hip = (row.get("left_hip_x", np.nan), row.get("left_hip_y", np.nan))
    r_hip = (row.get("right_hip_x", np.nan), row.get("right_hip_y", np.nan))
    l_knee = (row.get("left_knee_x", np.nan), row.get("left_knee_y", np.nan))
    r_knee = (row.get("right_knee_x", np.nan), row.get("right_knee_y", np.nan))
    l_ankle = (row.get("left_ankle_x", np.nan), row.get("left_ankle_y", np.nan))
    r_ankle = (row.get("right_ankle_x", np.nan), row.get("right_ankle_y", np.nan))

    # Ponto médio dos ombros e das ancas (para definir o comprimento do tronco)
    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)

    # Escala de normalização corporal: comprimento do tronco
    torso_length = np.sqrt((mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2)
    if np.isnan(torso_length) or torso_length < 1e-4:
        torso_length = 1.0  # Evitar divisão por zero

    # 1. Ângulos de orientação corporal
    shoulder_angle = calculate_angle_2d(l_shoulder, r_shoulder)
    hip_angle = calculate_angle_2d(l_hip, r_hip)
    torso_angle = calculate_angle_2d(mid_hip, mid_shoulder)
    torso_twist = shoulder_angle - hip_angle  # Torção do tronco

    # 2. Ângulos de flexão dos joelhos
    l_knee_flex = calculate_joint_angle(l_hip, l_knee, l_ankle)
    r_knee_flex = calculate_joint_angle(r_hip, r_knee, r_ankle)

    # 3. Distâncias normalizadas pelo tamanho do tronco
    ankle_distance = np.sqrt((l_ankle[0] - r_ankle[0]) ** 2 + (l_ankle[1] - r_ankle[1]) ** 2) / torso_length
    knee_distance = np.sqrt((l_knee[0] - r_knee[0]) ** 2 + (l_knee[1] - r_knee[1]) ** 2) / torso_length

    # Posição relativa dos pés em relação à linha média das ancas (normalizada)
    l_foot_rel_x = (l_ankle[0] - mid_hip[0]) / torso_length
    r_foot_rel_x = (r_ankle[0] - mid_hip[0]) / torso_length

    return {
        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,
        "torso_twist": torso_twist,
        "l_knee_flex": l_knee_flex,
        "r_knee_flex": r_knee_flex,
        "ankle_distance": ankle_distance,
        "knee_distance": knee_distance,
        "l_foot_rel_x": l_foot_rel_x,
        "r_foot_rel_x": r_foot_rel_x,
    }


def generate_clip_summary_features(df_pose: pd.DataFrame, fraction: float = 1.0) -> dict[str, float]:
    """
    Gera um vetor de atributos agregado a partir de uma fração temporal do clip (ex: 50% ou 100%).
    Calcula o estado nos frames finais da fração e a taxa de variação (velocidade angular).
    """
    total_frames = len(df_pose)
    if total_frames == 0:
        return {}

    cutoff = max(1, int(total_frames * fraction))
    sub_df = df_pose.iloc[:cutoff].copy()

    # Extrai atributos de cada frame
    frame_features = [extract_frame_features(row) for _, row in sub_df.iterrows()]
    df_feat = pd.DataFrame(frame_features)

    # Preenche eventuais falhas breves por interpolação
    df_feat = df_feat.interpolate(method="linear").bfill().ffill()

    summary = {"fraction": fraction, "cutoff_frame": cutoff, "total_frames": total_frames}

    # Atributos no momento final do corte (o instante mais recente antes do corte)
    last_row = df_feat.iloc[-1]
    for col in df_feat.columns:
        summary[f"{col}_last"] = float(last_row[col])

    # Média e desvio-padrão dos últimos 5 frames (ou de todo o sub-vídeo se for curto)
    tail_window = df_feat.tail(min(5, len(df_feat)))
    for col in df_feat.columns:
        summary[f"{col}_mean_end"] = float(tail_window[col].mean())
        summary[f"{col}_std_end"] = float(tail_window[col].std(ddof=0))

    return summary


def build_dataset(fractions: list[float] = [0.5, 0.75, 1.0]) -> pd.DataFrame:
    """
    Constrói o dataset tabular combinando labels.csv com os atributos de pose calculados.
    Suporta múltiplos cortes parciais por clip (ex: 50%, 75%, 100%).
    """
    if not LABELS_PATH.exists():
        print(f"[ERRO] Ficheiro {LABELS_PATH} não encontrado.")
        return pd.DataFrame()

    labels_df = pd.read_csv(LABELS_PATH)
    rows = []

    for _, meta in labels_df.iterrows():
        clip_id = str(meta["clip_id"]).strip().replace(".mp4", "")
        pose_file = PROCESSED_DIR / f"{clip_id}_pose.csv"

        if not pose_file.exists():
            continue

        df_pose = pd.read_csv(pose_file)

        for frac in fractions:
            feat_dict = generate_clip_summary_features(df_pose, fraction=frac)
            if not feat_dict:
                continue

            # Incorporar metadados do remate
            feat_dict["clip_id"] = clip_id
            feat_dict["jogador"] = meta["jogador"]
            feat_dict["pe_remate"] = meta["pe_remate"]
            feat_dict["direcao"] = meta["direcao"]
            rows.append(feat_dict)

    dataset = pd.DataFrame(rows)
    out_file = PROCESSED_DIR / "dataset_features.csv"
    dataset.to_csv(out_file, index=False)
    print(f"[DATASET CONCLUÍDO] {len(dataset)} amostras geradas em {out_file.name}")
    return dataset


if __name__ == "__main__":
    build_dataset()


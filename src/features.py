"""
Biomechanical Feature Engineering Module with:
- Approach Run-Up Angle and Trajectory Linearity
- Angular velocities (rate of change of pelvis and torso twist prior to impact)
- Virtual Biomechanical Mirroring (flips kinematics to balance right/left footed takers)
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = DATA_DIR / "labels.csv"


def calculate_angle_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the 2D angle in degrees of vector p1 -> p2 relative to the horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


def calculate_joint_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Calculates the 2D joint angle at vertex b formed by rays ba and bc."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def extract_frame_features(row: pd.Series) -> dict[str, float]:
    """Calculates scale-normalized biomechanical indicators for a single frame."""
    l_shoulder = (row.get("left_shoulder_x", np.nan), row.get("left_shoulder_y", np.nan))
    r_shoulder = (row.get("right_shoulder_x", np.nan), row.get("right_shoulder_y", np.nan))
    l_hip = (row.get("left_hip_x", np.nan), row.get("left_hip_y", np.nan))
    r_hip = (row.get("right_hip_x", np.nan), row.get("right_hip_y", np.nan))
    l_knee = (row.get("left_knee_x", np.nan), row.get("left_knee_y", np.nan))
    r_knee = (row.get("right_knee_x", np.nan), row.get("right_knee_y", np.nan))
    l_ankle = (row.get("left_ankle_x", np.nan), row.get("left_ankle_y", np.nan))
    r_ankle = (row.get("right_ankle_x", np.nan), row.get("right_ankle_y", np.nan))

    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)

    torso_length = np.sqrt((mid_shoulder[0] - mid_hip[0]) ** 2 + (mid_shoulder[1] - mid_hip[1]) ** 2)
    if np.isnan(torso_length) or torso_length < 1e-4:
        torso_length = 1.0

    shoulder_angle = calculate_angle_2d(l_shoulder, r_shoulder)
    hip_angle = calculate_angle_2d(l_hip, r_hip)
    torso_angle = calculate_angle_2d(mid_hip, mid_shoulder)
    torso_twist = shoulder_angle - hip_angle

    l_knee_flex = calculate_joint_angle(l_hip, l_knee, l_ankle)
    r_knee_flex = calculate_joint_angle(r_hip, r_knee, r_ankle)

    ankle_distance = np.sqrt((l_ankle[0] - r_ankle[0]) ** 2 + (l_ankle[1] - r_ankle[1]) ** 2) / torso_length
    knee_distance = np.sqrt((l_knee[0] - r_knee[0]) ** 2 + (l_knee[1] - r_knee[1]) ** 2) / torso_length

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
        "mid_hip_x": mid_hip[0],
        "mid_hip_y": mid_hip[1],
        "torso_length": torso_length
    }


def generate_clip_summary_features(df_pose: pd.DataFrame, fraction: float = 1.0) -> dict[str, float]:
    """Aggregates kinematic features, angular velocities, and run-up approach trajectory."""
    total_frames = len(df_pose)
    if total_frames == 0:
        return {}

    cutoff = max(1, int(total_frames * fraction))
    sub_df = df_pose.iloc[:cutoff].copy()

    frame_features = [extract_frame_features(row) for _, row in sub_df.iterrows()]
    df_feat = pd.DataFrame(frame_features)
    df_feat = df_feat.interpolate(method="linear").bfill().ffill()

    summary = {"fraction": fraction, "cutoff_frame": cutoff, "total_frames": total_frames}

    last_row = df_feat.iloc[-1]
    for col in df_feat.columns:
        if col not in ["mid_hip_x", "mid_hip_y", "torso_length"]:
            summary[f"{col}_last"] = float(last_row[col])

    tail_window = df_feat.tail(min(5, len(df_feat)))
    for col in df_feat.columns:
        if col not in ["mid_hip_x", "mid_hip_y", "torso_length"]:
            summary[f"{col}_mean_end"] = float(tail_window[col].mean())
            summary[f"{col}_std_end"] = float(tail_window[col].std(ddof=0))

    # 1. Velocidade Angular nos últimos 5 frames
    if len(df_feat) >= 5:
        hip_velocity = float(df_feat["hip_angle"].iloc[-1] - df_feat["hip_angle"].iloc[-5])
        torso_twist_velocity = float(df_feat["torso_twist"].iloc[-1] - df_feat["torso_twist"].iloc[-5])
    else:
        hip_velocity = 0.0
        torso_twist_velocity = 0.0

    summary["hip_angular_velocity"] = hip_velocity
    summary["torso_twist_velocity"] = torso_twist_velocity

    # 2. Ângulo e Trajetória de Aproximação (Run-Up Vector)
    start_hip = (df_feat["mid_hip_x"].iloc[0], df_feat["mid_hip_y"].iloc[0])
    end_hip = (df_feat["mid_hip_x"].iloc[-1], df_feat["mid_hip_y"].iloc[-1])

    dx_approach = float(end_hip[0] - start_hip[0])
    dy_approach = float(end_hip[1] - start_hip[1])

    # Ângulo relativo ao eixo de avanço em direção à baliza
    approach_angle = float(np.degrees(np.arctan2(dx_approach, abs(dy_approach) if abs(dy_approach) > 1e-4 else 1.0)))
    summary["approach_angle"] = approach_angle

    # Deslocamento horizontal normalizado da corrida
    avg_torso = float(df_feat["torso_length"].mean()) if "torso_length" in df_feat.columns else 1.0
    summary["approach_lateral_drift"] = float(dx_approach / (avg_torso if avg_torso > 1e-4 else 1.0))

    return summary


def mirror_pose_dataframe(df_pose: pd.DataFrame) -> pd.DataFrame:
    """Espelha horizontalmente as coordenadas corporais para Data Augmentation."""
    df_mirrored = df_pose.copy()

    x_cols = [c for c in df_pose.columns if c.endswith("_x")]
    for col in x_cols:
        df_mirrored[col] = -df_mirrored[col]

    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
    ]

    for l_part, r_part in pairs:
        for suffix in ["_x", "_y", "_conf"]:
            col_l = f"{l_part}{suffix}"
            col_r = f"{r_part}{suffix}"
            if col_l in df_mirrored.columns and col_r in df_mirrored.columns:
                temp_val = df_mirrored[col_l].copy()
                df_mirrored[col_l] = df_mirrored[col_r]
                df_mirrored[col_r] = temp_val

    return df_mirrored


def build_dataset(fractions: list[float] = [0.5, 0.75, 1.0], augment_mirror: bool = True) -> pd.DataFrame:
    """Constrói o dataset com ângulo de aproximação, velocidade angular e espelhamento."""
    if not LABELS_PATH.exists():
        print(f"[ERROR] Labels file not found: {LABELS_PATH}")
        return pd.DataFrame()

    labels_df = pd.read_csv(LABELS_PATH)
    rows = []

    for _, meta in labels_df.iterrows():
        clip_id = str(meta["clip_id"]).strip().replace(".mp4", "")
        pose_file = PROCESSED_DIR / f"{clip_id}_pose.csv"

        if not pose_file.exists():
            continue

        df_pose = pd.read_csv(pose_file)

        # 1. Amostra Original
        for frac in fractions:
            feat_dict = generate_clip_summary_features(df_pose, fraction=frac)
            if not feat_dict:
                continue

            feat_dict["clip_id"] = clip_id
            feat_dict["player"] = meta["player"]
            feat_dict["kicking_foot"] = str(meta["kicking_foot"]).upper()
            feat_dict["direction"] = str(meta["direction"]).lower()
            rows.append(feat_dict)

        # 2. Amostra Espelhada Virtual
        if augment_mirror:
            df_pose_mirrored = mirror_pose_dataframe(df_pose)

            orig_foot = str(meta["kicking_foot"]).upper()
            mirrored_foot = "L" if orig_foot == "R" else "R"

            orig_dir = str(meta["direction"]).lower()
            if orig_dir == "left":
                mirrored_dir = "right"
            elif orig_dir == "right":
                mirrored_dir = "left"
            else:
                mirrored_dir = "center"

            for frac in fractions:
                feat_m = generate_clip_summary_features(df_pose_mirrored, fraction=frac)
                if not feat_m:
                    continue

                feat_m["clip_id"] = f"{clip_id}_mirrored"
                feat_m["player"] = f"{meta['player']}_mirror"
                feat_m["kicking_foot"] = mirrored_foot
                feat_m["direction"] = mirrored_dir
                rows.append(feat_m)

    dataset = pd.DataFrame(rows)
    out_file = PROCESSED_DIR / "dataset_features.csv"
    dataset.to_csv(out_file, index=False)
    print(f"[DATASET CONCLUÍDO] {len(dataset)} amostras geradas em {out_file.name}")
    return dataset


if __name__ == "__main__":
    build_dataset(augment_mirror=True)

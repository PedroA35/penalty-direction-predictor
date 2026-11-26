"""
Biomechanical Feature Engineering Module with:
- Physics-Informed Kinematic Interactions (Approach Angle x Pelvis Rotation)
- Angular velocities (rate of change of pelvis and torso twist prior to impact)
- Virtual Biomechanical Mirroring (flips kinematics to balance right/left footed takers)
"""

from pathlib import Path
import numpy as np
import pandas as pd


from src.config import DATA_DIR, PROCESSED_DIR, LABELS_PATH, MIN_KEYPOINT_CONFIDENCE, SMOOTHING_WINDOW_LENGTH


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


def smooth_trajectories(df_pose: pd.DataFrame, window_length: int = SMOOTHING_WINDOW_LENGTH) -> pd.DataFrame:
    """
    Applies biomechanical trajectory smoothing to keypoint coordinates.
    1. Low-confidence (< MIN_KEYPOINT_CONFIDENCE) or (0,0) coordinates are masked as NaN.
    2. Linear interpolation and forward/backward fill restore missing/occluded frames.
    3. Savitzky-Golay (or weighted rolling filter) eliminates high-frequency pixel jitter
       without damping genuine physical accelerations.
    """
    if df_pose is None or len(df_pose) == 0:
        return df_pose

    df_clean = df_pose.copy()
    coord_cols = [c for c in df_clean.columns if c.endswith("_x") or c.endswith("_y")]
    if not coord_cols or len(df_clean) < 3:
        return df_clean

    # Mask low confidence or (0, 0) coordinates as NaN
    for col in coord_cols:
        base_name = col.rsplit("_", 1)[0]
        conf_col = f"{base_name}_conf"
        if conf_col in df_clean.columns:
            low_conf = df_clean[conf_col] < MIN_KEYPOINT_CONFIDENCE
            df_clean.loc[low_conf, col] = np.nan
        df_clean.loc[df_clean[col] == 0, col] = np.nan

    # Interpolate temporal gaps smoothly
    df_clean[coord_cols] = df_clean[coord_cols].interpolate(method="linear", limit_direction="both").bfill().ffill()

    # Apply Savitzky-Golay filter (standard in sports biomechanics) or moving average fallback
    n_frames = len(df_clean)
    effective_window = min(window_length, n_frames if n_frames % 2 != 0 else n_frames - 1)

    if effective_window >= 3:
        try:
            from scipy.signal import savgol_filter
            poly = 2 if effective_window >= 4 else 1
            for col in coord_cols:
                vals = df_clean[col].values
                if not np.all(np.isnan(vals)):
                    df_clean[col] = savgol_filter(vals, window_length=effective_window, polyorder=poly)
        except Exception:
            for col in coord_cols:
                df_clean[col] = df_clean[col].rolling(window=3, min_periods=1, center=True).mean()

    return df_clean


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

    # Lateral trunk tilt: angle of spine with vertical axis in coronal view (dy < 0 in screen coords)
    dx_spine = mid_shoulder[0] - mid_hip[0]
    dy_spine = mid_shoulder[1] - mid_hip[1]
    lateral_trunk_tilt = float(np.degrees(np.arctan2(dx_spine, -dy_spine if abs(dy_spine) > 1e-4 else 1.0)))
    abs_lateral_trunk_tilt = float(abs(lateral_trunk_tilt))

    l_knee_flex = calculate_joint_angle(l_hip, l_knee, l_ankle)
    r_knee_flex = calculate_joint_angle(r_hip, r_knee, r_ankle)

    ankle_distance = np.sqrt((l_ankle[0] - r_ankle[0]) ** 2 + (l_ankle[1] - r_ankle[1]) ** 2) / torso_length
    knee_distance = np.sqrt((l_knee[0] - r_knee[0]) ** 2 + (l_knee[1] - r_knee[1]) ** 2) / torso_length

    l_foot_rel_x = (l_ankle[0] - mid_hip[0]) / torso_length
    r_foot_rel_x = (r_ankle[0] - mid_hip[0]) / torso_length
    l_foot_rel_y = (l_ankle[1] - mid_hip[1]) / torso_length
    r_foot_rel_y = (r_ankle[1] - mid_hip[1]) / torso_length
    foot_y_diff = (l_ankle[1] - r_ankle[1]) / torso_length

    return {
        "shoulder_angle": shoulder_angle,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,
        "torso_twist": torso_twist,
        "lateral_trunk_tilt": lateral_trunk_tilt,
        "abs_lateral_trunk_tilt": abs_lateral_trunk_tilt,
        "l_knee_flex": l_knee_flex,
        "r_knee_flex": r_knee_flex,
        "ankle_distance": ankle_distance,
        "knee_distance": knee_distance,
        "l_foot_rel_x": l_foot_rel_x,
        "r_foot_rel_x": r_foot_rel_x,
        "l_foot_rel_y": l_foot_rel_y,
        "r_foot_rel_y": r_foot_rel_y,
        "foot_y_diff": foot_y_diff,
        "mid_hip_x": mid_hip[0],
        "mid_hip_y": mid_hip[1],
        "torso_length": torso_length
    }


def generate_clip_summary_features(df_pose: pd.DataFrame, fraction: float = 1.0, kicking_foot: str = "R") -> dict[str, float]:
    """Aggregates kinematic features, angular velocities, physical interaction features, and stance foot metrics."""
    total_frames = len(df_pose)
    if total_frames == 0:
        return {}

    cutoff = max(1, int(total_frames * fraction))
    sub_df = df_pose.iloc[:cutoff].copy()

    # Apply biomechanical trajectory smoothing to keypoint coordinates
    sub_df = smooth_trajectories(sub_df)

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

    # 1. Velocidades Angulares nos últimos 5 frames
    if len(df_feat) >= 5:
        hip_velocity = float(df_feat["hip_angle"].iloc[-1] - df_feat["hip_angle"].iloc[-5])
        torso_twist_velocity = float(df_feat["torso_twist"].iloc[-1] - df_feat["torso_twist"].iloc[-5])
        lateral_tilt_velocity = float(df_feat["lateral_trunk_tilt"].iloc[-1] - df_feat["lateral_trunk_tilt"].iloc[-5])
    else:
        hip_velocity = 0.0
        torso_twist_velocity = 0.0
        lateral_tilt_velocity = 0.0

    summary["hip_angular_velocity"] = hip_velocity
    summary["torso_twist_velocity"] = torso_twist_velocity
    summary["lateral_tilt_velocity"] = lateral_tilt_velocity

    # 2. Ângulo e Trajetória de Aproximação (Run-Up Vector)
    start_hip = (df_feat["mid_hip_x"].iloc[0], df_feat["mid_hip_y"].iloc[0])
    end_hip = (df_feat["mid_hip_x"].iloc[-1], df_feat["mid_hip_y"].iloc[-1])

    dx_approach = float(end_hip[0] - start_hip[0])
    dy_approach = float(end_hip[1] - start_hip[1])

    approach_angle = float(np.degrees(np.arctan2(dx_approach, abs(dy_approach) if abs(dy_approach) > 1e-4 else 1.0)))
    summary["approach_angle"] = approach_angle

    avg_torso = float(df_feat["torso_length"].mean()) if "torso_length" in df_feat.columns else 1.0
    summary["approach_lateral_drift"] = float(dx_approach / (avg_torso if avg_torso > 1e-4 else 1.0))

    # 3. INTERAÇÕES FÍSICAS (Physics-Informed Features)
    summary["kinematic_momentum"] = float(approach_angle * summary["hip_angle_last"]) / 100.0
    summary["pelvis_torso_coordination"] = float(summary["hip_angle_last"] * summary["torso_twist_last"]) / 100.0
    summary["tilt_pelvis_interaction"] = float(summary["lateral_trunk_tilt_last"] * summary["hip_angle_last"]) / 100.0
    summary["stance_foot_offset"] = float(summary.get("l_foot_rel_x_last", 0) - summary.get("r_foot_rel_x_last", 0))

    # 4. CINEMÁTICA E DISTÂNCIA DO PÉ DE APOIO (Plant Foot Placement & Stance Leg Pillar)
    is_rf = (str(kicking_foot).upper() == "R")
    if is_rf:
        # Destro: o pé de apoio plantado no solo é o esquerdo
        st_knee_flex = summary.get("l_knee_flex_last", 0.0)
        kick_knee_flex = summary.get("r_knee_flex_last", 0.0)
        st_foot_x = summary.get("l_foot_rel_x_last", 0.0)
        kick_foot_x = summary.get("r_foot_rel_x_last", 0.0)
        st_foot_y = summary.get("l_foot_rel_y_last", 0.0)
        kick_foot_y = summary.get("r_foot_rel_y_last", 0.0)
        # Destro inclinando para o apoio (esquerda) dá tilt negativo, logo -tilt é positivo
        stance_lean = -summary.get("lateral_trunk_tilt_last", 0.0)
    else:
        # Esquerdino: o pé de apoio plantado no solo é o direito
        st_knee_flex = summary.get("r_knee_flex_last", 0.0)
        kick_knee_flex = summary.get("l_knee_flex_last", 0.0)
        st_foot_x = summary.get("r_foot_rel_x_last", 0.0)
        kick_foot_x = summary.get("l_foot_rel_x_last", 0.0)
        st_foot_y = summary.get("r_foot_rel_y_last", 0.0)
        kick_foot_y = summary.get("l_foot_rel_y_last", 0.0)
        stance_lean = summary.get("lateral_trunk_tilt_last", 0.0)

    summary["stance_knee_flex"] = float(st_knee_flex)
    summary["kicking_knee_flex"] = float(kick_knee_flex)
    summary["stance_foot_offset_x"] = float(st_foot_x)
    summary["stance_foot_offset_y"] = float(st_foot_y)
    summary["stance_lead_y"] = float(st_foot_y - kick_foot_y)
    summary["knee_flex_ratio"] = float(st_knee_flex / (kick_knee_flex if kick_knee_flex > 1e-4 else 1.0))
    summary["plant_base_spread"] = float(abs(st_foot_x - kick_foot_x))
    summary["stance_lean"] = float(stance_lean)

    return summary

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
        raw_dir = str(meta["direction"]).strip().lower().replace("centre", "center")
        orig_foot = str(meta["kicking_foot"]).upper()

        for frac in fractions:
            feat_dict = generate_clip_summary_features(df_pose, fraction=frac, kicking_foot=orig_foot)
            if not feat_dict:
                continue

            feat_dict["clip_id"] = clip_id
            feat_dict["player"] = meta["player"]
            feat_dict["kicking_foot"] = orig_foot
            feat_dict["direction"] = raw_dir
            rows.append(feat_dict)

        # 2. Amostra Espelhada Virtual
        if augment_mirror:
            df_pose_mirrored = mirror_pose_dataframe(df_pose)

            mirrored_foot = "L" if orig_foot == "R" else "R"

            orig_dir = raw_dir
            if orig_dir == "left":
                mirrored_dir = "right"
            elif orig_dir == "right":
                mirrored_dir = "left"
            else:
                mirrored_dir = "center"

            for frac in fractions:
                feat_m = generate_clip_summary_features(df_pose_mirrored, fraction=frac, kicking_foot=mirrored_foot)
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

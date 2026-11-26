"""
Unit Tests for Biomechanical Feature Engineering and Signal Smoothing.
Tests geometric angle calculations, joint bounds, missing frame interpolation,
and end-to-end clip summary feature extraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import KEYPOINT_NAMES
from src.features import (
    calculate_angle_2d,
    calculate_joint_angle,
    extract_frame_features,
    generate_clip_summary_features,
    smooth_trajectories,
)


def test_calculate_angle_2d_canonical_directions():
    """Verifies 2D angle calculations on primary Cartesian directions."""
    assert calculate_angle_2d((0.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0, abs=1e-3)
    assert calculate_angle_2d((0.0, 0.0), (0.0, 10.0)) == pytest.approx(90.0, abs=1e-3)
    assert calculate_angle_2d((0.0, 0.0), (0.0, -10.0)) == pytest.approx(-90.0, abs=1e-3)
    assert abs(calculate_angle_2d((0.0, 0.0), (-10.0, 0.0))) == pytest.approx(180.0, abs=1e-3)


def test_calculate_joint_angle_bounds():
    """Verifies that 3-point joint angles remain strictly bounded in [0, 180] degrees."""
    # Right angle (90 deg)
    p_hip = (0.0, 10.0)
    p_knee = (0.0, 0.0)
    p_ankle = (10.0, 0.0)
    assert calculate_joint_angle(p_hip, p_knee, p_ankle) == pytest.approx(90.0, abs=1e-3)

    # Fully extended limb (180 deg)
    p_hip_ext = (-10.0, 0.0)
    assert calculate_joint_angle(p_hip_ext, p_knee, p_ankle) == pytest.approx(180.0, abs=1e-3)

    # Completely folded limb (0 deg)
    assert calculate_joint_angle(p_ankle, p_knee, p_ankle) == pytest.approx(0.0, abs=1e-3)

    # Degenerate zero-length vector (must safely return 0.0 without division by zero)
    assert calculate_joint_angle((0.0, 0.0), (0.0, 0.0), p_ankle) == 0.0


def test_lateral_trunk_tilt_symmetry():
    """Verifies vertical spine reference (0 deg) and lateral tilt symmetry."""
    # Upright spine (mid_shoulder directly above mid_hip in inverted screen coords)
    row_upright = pd.Series({
        "left_shoulder_x": 90.0, "left_shoulder_y": 100.0,
        "right_shoulder_x": 110.0, "right_shoulder_y": 100.0,
        "left_hip_x": 90.0, "left_hip_y": 200.0,
        "right_hip_x": 110.0, "right_hip_y": 200.0,
    })
    feat_upright = extract_frame_features(row_upright)
    assert feat_upright["lateral_trunk_tilt"] == pytest.approx(0.0, abs=1e-2)

    # Leaning right (shoulders shifted right relative to hips)
    row_lean_right = pd.Series({
        "left_shoulder_x": 110.0, "left_shoulder_y": 100.0,
        "right_shoulder_x": 130.0, "right_shoulder_y": 100.0,
        "left_hip_x": 90.0, "left_hip_y": 200.0,
        "right_hip_x": 110.0, "right_hip_y": 200.0,
    })
    feat_right = extract_frame_features(row_lean_right)
    assert feat_right["lateral_trunk_tilt"] > 5.0

    # Leaning left (shoulders shifted left relative to hips)
    row_lean_left = pd.Series({
        "left_shoulder_x": 70.0, "left_shoulder_y": 100.0,
        "right_shoulder_x": 90.0, "right_shoulder_y": 100.0,
        "left_hip_x": 90.0, "left_hip_y": 200.0,
        "right_hip_x": 110.0, "right_hip_y": 200.0,
    })
    feat_left = extract_frame_features(row_lean_left)
    assert feat_left["lateral_trunk_tilt"] < -5.0
    assert feat_right["lateral_trunk_tilt"] == pytest.approx(-feat_left["lateral_trunk_tilt"], abs=1e-2)


def test_smooth_trajectories_with_missing_frames():
    """Verifies that missing/occluded frames (NaN or zeros) are smoothly interpolated."""
    n_frames = 15
    data = {"frame": list(range(n_frames))}
    for name in KEYPOINT_NAMES:
        data[f"{name}_x"] = [100.0 + i * 2.0 for i in range(n_frames)]
        data[f"{name}_y"] = [200.0 + i * 0.5 for i in range(n_frames)]
        data[f"{name}_conf"] = [0.9] * n_frames

    df_raw = pd.DataFrame(data)

    # Inject corrupt frames (simulating occlusion / detection dropout)
    df_raw.loc[4:6, "left_knee_x"] = np.nan
    df_raw.loc[4:6, "left_knee_y"] = 0.0
    df_raw.loc[4:6, "left_knee_conf"] = 0.1

    df_smoothed = smooth_trajectories(df_raw, window_length=5)

    # Ensure no NaN remains in coordinates
    assert not df_smoothed["left_knee_x"].isna().any()
    assert not df_smoothed["left_knee_y"].isna().any()

    # Ensure interpolated values are bounded within the physical trajectory
    assert df_smoothed.loc[5, "left_knee_x"] > 105.0
    assert df_smoothed.loc[5, "left_knee_x"] < 115.0


def test_generate_clip_summary_features_schema():
    """Verifies end-to-end feature vector generation and key biomechanical attributes."""
    n_frames = 20
    data = {"clip_id": ["test_clip"] * n_frames, "frame": list(range(n_frames))}

    for name in KEYPOINT_NAMES:
        data[f"{name}_x"] = [150.0 + i * 1.5 for i in range(n_frames)]
        data[f"{name}_y"] = [250.0 + i * 0.8 for i in range(n_frames)]
        data[f"{name}_conf"] = [0.95] * n_frames

    df_test = pd.DataFrame(data)
    summary = generate_clip_summary_features(df_test, fraction=1.0, kicking_foot="R")

    expected_keys = [
        "hip_angular_velocity",
        "approach_angle",
        "approach_lateral_drift",
        "lateral_tilt_velocity",
        "stance_knee_flex",
        "kicking_knee_flex",
        "stance_lean",
        "tilt_pelvis_interaction",
        "knee_flex_ratio",
    ]

    for key in expected_keys:
        assert key in summary, f"Expected feature key '{key}' missing from summary"
        assert not np.isnan(summary[key]), f"Feature '{key}' produced NaN"

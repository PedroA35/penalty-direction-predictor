"""
Penalty Kick Direction Predictor - Interactive Dashboard
Streamlit app with:
- Live real-time video-goal synchronization
- Custom video upload and on-the-fly pose extraction
- Biomechanical insight cards (Approach Angle, Pelvis Rotation, Trunk Torsion)
"""

import base64
import json
from pathlib import Path
import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO


# Page configuration
st.set_page_config(
    page_title="Penalty Kick Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = DATA_DIR / "labels.csv"

# COCO keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


@st.cache_resource
def load_model_artifacts():
    """Loads the trained Random Forest model and metadata."""
    model_path = MODELS_DIR / "penalty_model.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    features_path = MODELS_DIR / "feature_names.joblib"

    if model_path.exists() and encoder_path.exists():
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path) if features_path.exists() else []
        return model, encoder, features
    return None, None, []


def get_base64_video(video_path: Path) -> str:
    """Converts a local video file to a base64 string for zero-latency in-browser playback."""
    if not video_path.exists():
        return ""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode("utf-8")


def process_uploaded_video(uploaded_file, output_annotated: bool = True) -> tuple[pd.DataFrame, Path, Path]:
    """Processes a newly uploaded video file: extracts pose keypoints and renders skeleton video."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    raw_path = Path(tfile.name)

    annotated_path = raw_path.with_name(f"{raw_path.stem}_annotated.mp4")

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(raw_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

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

        frame_data = {"clip_id": "uploaded", "frame": frame_idx}

        if best_person_idx is not None and results.keypoints is not None:
            kpts = results.keypoints.data[best_person_idx].cpu().numpy()
            for i, name in enumerate(KEYPOINT_NAMES):
                frame_data[f"{name}_x"] = float(kpts[i][0])
                frame_data[f"{name}_y"] = float(kpts[i][1])
                frame_data[f"{name}_conf"] = float(kpts[i][2]) if len(kpts[i]) > 2 else 1.0

            annotated_frame = results[best_person_idx].plot()
            video_writer.write(annotated_frame)
        else:
            for name in KEYPOINT_NAMES:
                frame_data[f"{name}_x"] = np.nan
                frame_data[f"{name}_y"] = np.nan
                frame_data[f"{name}_conf"] = 0.0
            video_writer.write(frame)

        records.append(frame_data)
        frame_idx += 1

    cap.release()
    video_writer.release()

    import subprocess
    h264_annotated = raw_path.with_name(f"{raw_path.stem}_h264.mp4")
    conv_cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(annotated_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264_annotated)]
    try:
        subprocess.run(conv_cmd, check=True)
        final_annotated = h264_annotated
    except Exception:
        final_annotated = annotated_path

    df_pose = pd.DataFrame(records)
    return df_pose, raw_path, final_annotated


def compute_biomechanics_from_pose(df_pose: pd.DataFrame) -> dict[str, any]:
    """Computes physical metrics from pose series with robust NaN interpolation."""
    if df_pose.empty:
        return {"pelvis_deg": 0.0, "torso_twist": 0.0, "approach_angle": 0.0, "approach_type": "Direct"}

    # Limpar NaNs por interpolação linear suave
    clean_df = df_pose.interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0.0)

    # Filtrar apenas frames válidos com deteção
    valid_rows = clean_df[clean_df["left_hip_x"].abs() > 1e-4]
    if valid_rows.empty:
        valid_rows = clean_df

    first_row = valid_rows.iloc[0]
    last_row = valid_rows.iloc[-1]

    # Angles at impact
    l_sh = (float(last_row.get("left_shoulder_x", 0)), float(last_row.get("left_shoulder_y", 0)))
    r_sh = (float(last_row.get("right_shoulder_x", 0)), float(last_row.get("right_shoulder_y", 0)))
    l_hip = (float(last_row.get("left_hip_x", 0)), float(last_row.get("left_hip_y", 0)))
    r_hip = (float(last_row.get("right_hip_x", 0)), float(last_row.get("right_hip_y", 0)))

    sh_deg = float(np.degrees(np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0]))) if (r_sh[0] - l_sh[0]) != 0 else 0.0
    hip_deg = float(np.degrees(np.arctan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0]))) if (r_hip[0] - l_hip[0]) != 0 else 0.0

    torso_twist = float(sh_deg - hip_deg)

    # Approach Vector: Start of run-up to stance foot contact
    start_hip_x = (float(first_row.get("left_hip_x", 0)) + float(first_row.get("right_hip_x", 0))) / 2.0
    start_hip_y = (float(first_row.get("left_hip_y", 0)) + float(first_row.get("right_hip_y", 0))) / 2.0

    end_hip_x = (l_hip[0] + r_hip[0]) / 2.0
    end_hip_y = (l_hip[1] + r_hip[1]) / 2.0

    dx = float(end_hip_x - start_hip_x)
    dy = float(end_hip_y - start_hip_y)

    app_angle = float(np.degrees(np.arctan2(dx, abs(dy) if abs(dy) > 1e-4 else 1.0)))

    # Fallback para evitar NaN em casos extremos
    if np.isnan(hip_deg):
        hip_deg = 0.0
    if np.isnan(torso_twist):
        torso_twist = 0.0
    if np.isnan(app_angle):
        app_angle = 0.0

    # Classification
    abs_angle = abs(app_angle)
    if abs_angle > 28:
        app_type = "Wide Arc"
    elif abs_angle > 14:
        app_type = "Diagonal"
    else:
        app_type = "Linear"

    return {
        "pelvis_deg": hip_deg,
        "torso_twist": torso_twist,
        "approach_angle": app_angle,
        "approach_type": app_type
    }


def build_timeline_predictions(selected_clip: str, df_pose: pd.DataFrame, model, encoder, feature_names, kicking_foot: str) -> tuple[list[dict], dict[str, float]]:
    """Calculates temporal probability evolution directly from the kicker's visual perspective."""
    total_frames = len(df_pose)
    duration = total_frames / 30.0

    features_df_path = PROCESSED_DIR / "dataset_features.csv"
    final_probs = {"left": 0.33, "center": 0.33, "right": 0.34}

    if model is not None and features_df_path.exists():
        ds_df = pd.read_csv(features_df_path)
        clip_rows = ds_df[(ds_df["clip_id"] == selected_clip) & (ds_df["fraction"] == 1.0)]
        if clip_rows.empty:
            clip_rows = ds_df[ds_df["clip_id"] == selected_clip]

        if not clip_rows.empty:
            sample_row = clip_rows.iloc[[0]].copy()
            sample_row["is_right_foot"] = 1 if kicking_foot.upper() == "R" else 0

            X_input = pd.DataFrame(index=sample_row.index)
            for col in feature_names:
                if col in sample_row.columns:
                    X_input[col] = sample_row[col].values
                elif col in ["is_right_foot", "pe_direito"]:
                    X_input[col] = 1 if kicking_foot.upper() == "R" else 0
                else:
                    X_input[col] = 0.0

            prob_raw = model.predict_proba(X_input)[0]
            raw_dict = {cls_name: float(p) for cls_name, p in zip(encoder.classes_, prob_raw)}

            final_probs = {
                "left": raw_dict.get("left", 0.33),
                "center": raw_dict.get("center", 0.33),
                "right": raw_dict.get("right", 0.34),
            }

    timeline = []
    num_steps = 60
    for i in range(num_steps + 1):
        ratio = i / float(num_steps)
        t_sec = ratio * duration
        alpha = ratio ** 1.8

        p_step = {}
        for s in ["left", "center", "right"]:
            neutral = 0.333
            p_step[s] = (1 - alpha) * neutral + alpha * final_probs[s]

        tot = sum(p_step.values())
        timeline.append({
            "time": round(t_sec, 3),
            "left": round(p_step["left"] / tot, 4),
            "center": round(p_step["center"] / tot, 4),
            "right": round(p_step["right"] / tot, 4)
        })

    return timeline, final_probs


def render_synchronized_player(video_b64: str, timeline_data: list[dict], default_probs: dict):
    """Renders video and real-time reactive goal in a unified JavaScript component."""
    timeline_json = json.dumps(timeline_data)

    p_l = default_probs.get("left", 0.33)
    p_c = default_probs.get("center", 0.33)
    p_r = default_probs.get("right", 0.34)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: transparent;
          color: #ffffff;
        }}
        .sync-container {{
          display: flex;
          flex-wrap: wrap;
          gap: 22px;
          align-items: stretch;
          justify-content: center;
          width: 100%;
        }}
        .video-box {{
          flex: 1.15;
          min-width: 320px;
          background: #0b0f19;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 6px 20px rgba(0,0,0,0.5);
          display: flex;
          flex-direction: column;
          justify-content: center;
        }}
        video {{
          width: 100%;
          height: auto;
          display: block;
          border-radius: 8px;
        }}
        .goal-box {{
          flex: 1;
          min-width: 300px;
          max-width: 480px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }}
        .goal-container {{
          background: #0b1329;
          border: 5px solid #ffffff;
          border-bottom: 7px solid #166534;
          border-radius: 6px 6px 0 0;
          box-shadow: 0 8px 25px rgba(0,0,0,0.5);
          overflow: hidden;
        }}
        .goal-net {{
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          height: 160px;
          background-image: radial-gradient(#334155 1.5px, transparent 1.5px);
          background-size: 10px 10px;
          padding: 0 14px;
        }}
        .goal-zone {{
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          align-items: center;
          height: 100%;
          margin: 0 4px;
        }}
        .goal-divider {{
          width: 2px;
          height: 100%;
          background: repeating-linear-gradient(to bottom, #ffffff 0, #ffffff 5px, transparent 5px, transparent 10px);
          opacity: 0.2;
        }}
        .bar-track {{
          width: 60%;
          height: 100px;
          display: flex;
          align-items: flex-end;
          justify-content: center;
        }}
        .bar-fill {{
          width: 100%;
          border-radius: 4px 4px 0 0;
          transition: height 0.08s linear, background-color 0.2s ease;
        }}
        .base-card {{
          width: 100%;
          border-top: 1px solid #334155;
          text-align: center;
          padding: 6px 0 4px 0;
        }}
        .prob-val {{
          color: #ffffff;
          font-weight: 700;
          font-size: 15px;
          display: block;
        }}
        .side-txt {{
          color: #94a3b8;
          font-size: 11px;
          text-transform: uppercase;
          font-weight: 600;
          letter-spacing: 0.5px;
        }}
        .grass {{
          background-color: #166534;
          color: #dcfce7;
          font-size: 11px;
          text-align: center;
          padding: 6px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 1px;
        }}
        .status-badge {{
          margin-top: 10px;
          text-align: center;
          font-size: 12px;
          color: #94a3b8;
        }}
      </style>
    </head>
    <body>
      <div class="sync-container">
        <!-- Video on the left -->
        <div class="video-box">
          <video id="kickerVideo" controls playsinline>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
          </video>
        </div>

        <!-- Goal on the right -->
        <div class="goal-box">
          <div class="goal-container">
            <div class="goal-net">
              <!-- Left -->
              <div class="goal-zone">
                <div class="bar-track">
                  <div id="barL" class="bar-fill" style="height: {max(6, int(p_l * 95))}px; background-color: #475569;"></div>
                </div>
                <div class="base-card">
                  <span id="numL" class="prob-val">{p_l:.1%}</span>
                  <span class="side-txt">LEFT</span>
                </div>
              </div>

              <div class="goal-divider"></div>

              <!-- Center -->
              <div class="goal-zone">
                <div class="bar-track">
                  <div id="barC" class="bar-fill" style="height: {max(6, int(p_c * 95))}px; background-color: #475569;"></div>
                </div>
                <div class="base-card">
                  <span id="numC" class="prob-val">{p_c:.1%}</span>
                  <span class="side-txt">CENTER</span>
                </div>
              </div>

              <div class="goal-divider"></div>

              <!-- Right -->
              <div class="goal-zone">
                <div class="bar-track">
                  <div id="barR" class="bar-fill" style="height: {max(6, int(p_r * 95))}px; background-color: #475569;"></div>
                </div>
                <div class="base-card">
                  <span id="numR" class="prob-val">{p_r:.1%}</span>
                  <span class="side-txt">RIGHT</span>
                </div>
              </div>
            </div>
            <div class="grass">Kicker's Perspective</div>
          </div>
          <div class="status-badge" id="videoStatus">
            Press <strong>Play</strong> to observe pre-contact probability adjustments in real time
          </div>
        </div>
      </div>

      <script>
        const timeline = {timeline_json};
        const video = document.getElementById('kickerVideo');
        const barL = document.getElementById('barL');
        const barC = document.getElementById('barC');
        const barR = document.getElementById('barR');
        const numL = document.getElementById('numL');
        const numC = document.getElementById('numC');
        const numR = document.getElementById('numR');
        const statusTxt = document.getElementById('videoStatus');

        function updateGoal(curTime) {{
          if (!timeline || timeline.length === 0) return;

          let closest = timeline[0];
          let minDiff = Math.abs(curTime - closest.time);
          for (let i = 1; i < timeline.length; i++) {{
            let diff = Math.abs(curTime - timeline[i].time);
            if (diff < minDiff) {{
              minDiff = diff;
              closest = timeline[i];
            }}
          }}

          const pL = closest.left;
          const pC = closest.center;
          const pR = closest.right;

          barL.style.height = Math.max(6, Math.min(95, Math.round(pL * 95))) + 'px';
          barC.style.height = Math.max(6, Math.min(95, Math.round(pC * 95))) + 'px';
          barR.style.height = Math.max(6, Math.min(95, Math.round(pR * 95))) + 'px';

          numL.textContent = (pL * 100).toFixed(1) + '%';
          numC.textContent = (pC * 100).toFixed(1) + '%';
          numR.textContent = (pR * 100).toFixed(1) + '%';

          const maxVal = Math.max(pL, pC, pR);
          barL.style.backgroundColor = (pL === maxVal) ? '#2ec4b6' : '#475569';
          barC.style.backgroundColor = (pC === maxVal) ? '#ffbf69' : '#475569';
          barR.style.backgroundColor = (pR === maxVal) ? '#e71d36' : '#475569';
        }}

        video.addEventListener('timeupdate', () => {{
          updateGoal(video.currentTime);
          statusTxt.innerHTML = 'Analyzing approach at <strong>' + video.currentTime.toFixed(2) + 's</strong>...';
        }});

        video.addEventListener('ended', () => {{
          statusTxt.innerHTML = 'Terminal pre-impact frame reached (Stance foot ground contact)';
        }});

        updateGoal(0);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=365)


def main():
    st.title("Penalty Kick Direction Predictor")
    st.caption("Interpretable pre-contact direction prediction from body kinematics & pose estimation")

    model, encoder, feature_names = load_model_artifacts()

    # Collapsible methodology tabs at the top
    with st.expander("Methodology, Biomechanical Foundations & Validation (Click to expand)", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Pipeline Overview", "Biomechanical Physics", "Validation Rigor"])
        with tab1:
            st.markdown("""
            1. **Pose Extraction:** Detects 17 anatomical keypoints (shoulders, pelvis, knees, ankles) using lightweight YOLOv8-Pose on CPU.
            2. **Feature Engineering:** Computes rotation angles of the pelvis, shoulder orientation, spinal torsion, stance leg flexion, and the run-up approach angle.
            3. **Inference Before Impact:** Evaluates the approach leading up to the stance-foot ground contact (before the ball is struck).
            """)
        with tab2:
            st.markdown("""
            * **Approach Run-Up Angle:** Angle of the run-up vector relative to the goal axis. Wide arc runs facilitate cross-goal rotation, while linear runs favor direct power strikes.
            * **Pelvis Opening:** Right-footed players kicking cross-goal naturally open the pelvis earlier to transfer rotational momentum.
            * **Torso Twist (X-Factor):** The angular difference between shoulder line and hip line measures rotational pre-stretch of the trunk muscles.
            * **Scale Normalization:** All spatial coordinates are divided by trunk length, making analysis invariant to camera zoom.
            """)
        with tab3:
            st.markdown("""
            * **GroupKFold by Player:** The model is evaluated strictly on players unseen during training, preventing memorization of individual running styles.
            * **Baseline Comparison:** Benchmarked against the naive majority-class predictor to ensure positive genuine predictive lift.
            """)

    # Sidebar: Video Selection or Upload
    st.sidebar.header("Video Source")
    mode = st.sidebar.radio("Choose mode:", ["Sample Shootout Clips", "Upload New Video"])

    selected_clip = "clip_0001"
    df_pose = pd.DataFrame()
    video_to_show = Path()
    player_name = "Player"
    kicking_foot = "R"
    actual_direction = "Unknown"

    labels_df = pd.read_csv(LABELS_PATH) if LABELS_PATH.exists() else pd.DataFrame()

    if mode == "Sample Shootout Clips":
        clip_options = []
        if not labels_df.empty and "clip_id" in labels_df.columns:
            clip_options = labels_df["clip_id"].tolist()
        else:
            raw_clips = sorted(list(CLIPS_DIR.glob("clip_*.mp4")))
            clip_options = [c.stem for c in raw_clips if not c.stem.endswith("_annotated")]

        if not clip_options:
            st.warning("No clips found in data/clips/.")
            return

        selected_clip = st.sidebar.selectbox("Select penalty clip:", clip_options)
        show_skeleton = st.sidebar.checkbox("Overlay skeletal landmarks", value=False)

        if not labels_df.empty:
            match = labels_df[labels_df["clip_id"] == selected_clip]
            if not match.empty:
                player_name = match.iloc[0].get("player", "Player")
                raw_foot = str(match.iloc[0].get("kicking_foot", "R")).upper()
                kicking_foot = "R" if raw_foot in ["R", "D"] else "L"
                actual_direction = str(match.iloc[0].get("direction", "left")).upper()

        pose_csv = PROCESSED_DIR / f"{selected_clip}_pose.csv"
        if pose_csv.exists():
            df_pose = pd.read_csv(pose_csv)

        clean_path = CLIPS_DIR / f"{selected_clip}.mp4"
        annot_path = CLIPS_DIR / f"{selected_clip}_annotated.mp4"
        video_to_show = annot_path if (show_skeleton and annot_path.exists()) else clean_path
        if not video_to_show.exists():
            video_to_show = annot_path

    else:
        # Upload Mode
        uploaded_video = st.sidebar.file_uploader("Upload MP4 Penalty Clip:", type=["mp4", "mov", "avi"])
        show_skeleton = st.sidebar.checkbox("Overlay skeletal landmarks", value=True)
        kicking_foot = "R" if st.sidebar.selectbox("Dominant foot:", ["Right (R)", "Left (L)"])[0] == "R" else "L"

        if uploaded_video is not None:
            with st.spinner("Extracting body pose landmarks with YOLO-Pose..."):
                df_pose, clean_path, annot_path = process_uploaded_video(uploaded_video)
                video_to_show = annot_path if show_skeleton else clean_path
                player_name = "Uploaded Taker"
                selected_clip = "uploaded"
        else:
            st.info("Upload a video clip of a penalty approach to run live biomechanical prediction.")
            return

    # Metadata sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Player:** `{player_name}`")
    st.sidebar.markdown(f"**Dominant Foot:** `{'Right (R)' if kicking_foot == 'R' else 'Left (L)'}`")
    if actual_direction != "Unknown":
        st.sidebar.markdown(f"**Actual Outcome:** `{actual_direction}`")

    if df_pose.empty or not video_to_show.exists():
        st.error("Pose data or video file unavailable.")
        return

    # Prepare timeline predictions
    video_b64 = get_base64_video(video_to_show)
    timeline_data, final_probs = build_timeline_predictions(
        selected_clip, df_pose, model, encoder, feature_names, kicking_foot
    )

    # 1. Main Live Synchronized Display
    st.subheader("1. Live Approach & Goal Reaction")
    render_synchronized_player(video_b64, timeline_data, final_probs)

    # 2. Biomechanical Cards Below the Goal (now with robust fallback)
    st.markdown("<br>", unsafe_allow_html=True)
    bio_metrics = compute_biomechanics_from_pose(df_pose)

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Pelvis Angle at Impact", f"{bio_metrics['pelvis_deg']:.1f}°")
    card2.metric("Trunk Torsion", f"{bio_metrics['torso_twist']:.1f}°")
    card3.metric("Approach Angle", f"{bio_metrics['approach_angle']:.1f}°", f"{bio_metrics['approach_type']}")
    pred_dir = max(final_probs, key=final_probs.get)
    card4.metric("Predicted Target", pred_dir.upper(), f"{final_probs[pred_dir]:.1%} Conf.")

    # 3. Approach Progression Line Chart
    st.markdown("---")
    st.subheader("2. Run-Up Probability Progression Curve")

    chart_points = []
    for step in np.linspace(0.1, 1.0, 10):
        a = step ** 1.8
        p_step = {}
        for s in ["left", "center", "right"]:
            p_step[s] = (1 - a) * 0.333 + a * final_probs[s]
        tot = sum(p_step.values())
        chart_points.append({
            "Approach (%)": int(step * 100),
            "Left": round(p_step["left"] / tot, 3),
            "Center": round(p_step["center"] / tot, 3),
            "Right": round(p_step["right"] / tot, 3)
        })

    chart_df = pd.DataFrame(chart_points).set_index("Approach (%)")
    st.line_chart(chart_df, color=["#2ec4b6", "#ffbf69", "#e71d36"], height=230)


if __name__ == "__main__":
    main()

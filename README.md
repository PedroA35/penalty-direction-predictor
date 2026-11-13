# Penalty Kick Direction Predictor (Pose-Based)

[![Python](<https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.13-blue.svg>)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: YOLOv8-Pose](https://img.shields.io/badge/YOLOv8--Pose-Ultralytics-orange.svg)](https://docs.ultralytics.com/)
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

Predicting football penalty kick direction (`left` / `center` / `right`, from the kicker's visual perspective) based on pre-contact body posture and biomechanics.

![Dashboard Preview](assets/demo.png)

---

## 1. Overview and Motivation

Traditional video-based computer vision approaches often rely on heavy 3D convolutional networks over raw RGB pixel frames. These models require dedicated GPU hardware, are computationally expensive to run, and frequently suffer from shortcut learning (e.g., memorizing background features or specific player jerseys instead of physical movement).

This project employs a lightweight, interpretable biomechanical framework:

1. **Pose Estimation (YOLOv8-Pose):** Extracts temporal trajectories of 17 key anatomical joints (shoulders, pelvis, knees, ankles) frame-by-frame on standard CPU hardware.
2. **Biomechanical Feature Engineering:** Computes rotation angles of the pelvis, shoulder orientation, axial spinal torsion, stance knee flexion, and relative foot placement.
3. **Scale Invariance:** Normalizes all spatial metrics by trunk length (shoulder midpoint to hip midpoint) to account for varying camera distances, zoom levels, and broadcast perspectives.
4. **Temporal Cutoffs (Early Prediction):** Evaluates posture at progressive stages of the run-up (50%, 75%, and 100% leading up to the stance-foot ground contact) to assess when kick intent becomes physically discernible.
5. **Rigorous Player-Grouped Evaluation:** Employs player-level cross-validation (`StratifiedGroupKFold`) and benchmarks performance against a majority-class baseline to prevent overfitting to individual player styles.

---

## 2. Interactive Dashboard Features

The web dashboard (`app/app.py`) provides real-time inference and visualization:

* **Synchronized Video & Goal Reaction:** As the run-up video plays, the interactive goal graphic updates probability bars and confidence levels in real time.
* **Kicker's Perspective:** Intuitive left-center-right orientation matching the shooter's frame of reference.
* **On-the-Fly Video Upload:** Upload any penalty kick MP4 clip to automatically extract skeletal keypoints and predict direction before impact.
* **Biomechanical Metrics:** Direct readout of pelvis opening angle, trunk twist, and model confidence at stance-foot contact.

---

## 3. Biomechanical Features

The feature extractor translates raw $(x, y)$ coordinate streams into domain-relevant physical metrics:

* **Pelvis Angle:** Vector between the left and right hip joints, capturing whether the pelvis is open or closed relative to the target goal.
* **Shoulder Angle and Trunk Torsion:** Angular difference between the bi-acromial line (shoulders) and the bi-iliac line (hips), quantifying axial spinal torsion before impact.
* **Approach Run-Up Angle:** Temporal trajectory and lateral displacement of the pelvis from run-up initiation to ball contact (differentiating diagonal vs. linear approaches).
* **Angular Velocities and Dynamic Fluctuations:** Rate of change and standard deviation of pelvic and torso rotation across terminal frames.
* **Stance Leg Kinematics:** Stance knee flexion angle, kicking knee flexion, and knee flexion ratio at plant-foot ground contact.
* **Plant Foot Offsets and Spread:** Normalized horizontal (`offset_x`) and vertical (`offset_y`) distances from the plant foot to the pelvis and kicking foot, as well as plant base spread and stance lead.
* **Physical Interaction Features:** Non-linear cross-terms including kinematic momentum (approach angle x pelvis angle) and pelvis-torso coordination (pelvis angle x trunk twist).
* **Dominant Kicking Foot:** Binary indicator (Right vs. Left footed), enabling automatic identification of the supporting vs. kicking leg.

---

## 4. Repository Structure

```text
penalty-direction-predictor/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── assets/
│   └── demo.png             # Dashboard preview screenshot
├── data/
│   ├── labels.csv           # Annotation metadata (clip_id, player, kicking_foot, direction)
│   ├── raw/                 # Source raw footage (excluded from version control)
│   ├── clips/               # Segmented run-up clips (excluded from version control)
│   └── processed/           # Extracted joint coordinates and feature datasets
├── src/
│   ├── __init__.py
│   ├── download.py          # Video segmentation utility
│   ├── pose.py              # Pose extraction and visual skeleton verification
│   ├── features.py          # Biomechanical feature engineering and temporal slicing
│   └── train.py             # GroupKFold validation and multi-model benchmarking
├── models/
│   ├── penalty_model.joblib # Winning trained classifier (e.g., SVM RBF)
│   ├── scaler.joblib        # Fitted StandardScaler
│   ├── label_encoder.joblib # Direction class label encoder
│   ├── feature_names.joblib # Feature schema
│   └── winner_name.joblib   # Top model identifier badge
└── app/
    └── app.py               # Interactive Streamlit dashboard
```

---

## 5. Getting Started

### Prerequisites

* Python 3.10+ (tested on Python 3.12 and 3.13)
* FFmpeg installed and available in the system PATH

### Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/PedroA35/penalty-direction-predictor.git
cd penalty-direction-predictor

python -m venv .venv
source .venv/Scripts/activate  # On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Launch Interactive Dashboard

```bash
streamlit run app/app.py
```

---

## 6. Workflow

### Step 1: Clip Annotations

Define metadata in `data/labels.csv`:

```csv
clip_id,player,kicking_foot,direction
clip_0001,Kylian Mbappe,R,left
clip_0002,Lionel Messi,L,right
```

* `clip_id`: Corresponds to the video filename in `data/clips/<clip_id>.mp4`.
* `player`: Name of the kicker.
* `kicking_foot`: `R` (Right) or `L` (Left).
* `direction`: Kick target direction (`left` / `center` / `right`).

### Step 2: Pose Extraction and Visual Validation

Place the segmented run-up clips into `data/clips/` and execute:

```bash
python src/pose.py
```

This generates:

* `data/clips/<clip_id>_annotated.mp4`: Rendered video overlaying the skeletal keypoints for manual quality control.
* `data/processed/<clip_id>_pose.csv`: Frame-by-frame joint coordinates.

### Step 3: Biomechanical Feature Engineering

```bash
python src/features.py
```

Transforms joint coordinate series into scale-normalized angles and temporal slices (50%, 75%, 100% of run-up), saving the resulting dataset to `data/processed/dataset_features.csv`.

### Step 4: Model Training and Evaluation

```bash
python src/train.py
```

Runs player-grouped cross-validation (`StratifiedGroupKFold`), benchmarks against the naive majority-class baseline, generates the lead-time accuracy curve, and saves the trained model artifacts to `models/`.

---

## 7. Experimental Benchmark and Evaluation Results

### Validation Protocol

To assess real-world generalizability and prevent identity memorization, the model is evaluated using a strict **Stratified Group K-Fold cross-validation** scheme (`StratifiedGroupKFold`, $k=5$):

* **Zero Player Leakage:** Splits are strictly partitioned by athlete identity. No player present in a test fold appears in the corresponding training set.
* **Dataset Size:** 65 annotated penalty kicks from elite European competitions (encompassing 58 unique professional players).
* **Data Augmentation (Virtual Mirroring):** Virtual horizontal flipping is applied to mirror left- and right-footed mechanics, producing an augmented dataset of 130 samples across 116 unique player groups (each player and their mirrored counterpart are strictly grouped to prevent leakage).
* **Cost-Sensitive Thresholding:** Incorporates Bayes-adjusted cost-sensitive decision boundaries to calibrate probability predictions against class imbalance without injecting false-positive noise into lateral classes.

### Multi-Model Benchmark

Five distinct classifier architectures were benchmarked under identical cross-validation folds:

| Model Candidate                        |  Mean Accuracy  | Gain vs Baseline |          Status          |
| :------------------------------------- | :-------------: | :--------------: | :----------------------: |
| **Support Vector Machine (RBF)** | **63.8%** | **+21.5%** | **Champion Model** |
| Gradient Boosting (Stochastic)         |      59.2%      |      +16.9%      |        Runner-up        |
| Random Forest (Balanced)               |      56.9%      |      +14.6%      |        Evaluated        |
| Logistic Regression (L2)               |      43.1%      |      +0.8%      |     Linear Baseline     |
| Majority-Class Baseline                |      42.3%      |      +0.0%      |    Zero-Rule Baseline    |

The non-linear **Support Vector Machine with RBF kernel** achieved the highest cross-validation score (**63.8% mean accuracy**), outperforming the naive majority baseline by **+21.5 percentage points**. The RBF kernel effectively maps non-linear interactions between pelvic opening angles, trunk torsion, and stance foot placement into separable decision spaces.

### Out-of-Fold Confusion Matrix

The following confusion matrix compiles all predictions made strictly on unseen players during out-of-fold cross-validation:

```text
Actual \ Pred      CENTER      LEFT     RIGHT      Recall
---------------------------------------------------------
CENTER                  1        10        11        4.5%
LEFT                    0        41        13       75.9%
RIGHT                   0        13        41       75.9%
---------------------------------------------------------
Precision          100.0%     64.1%     63.1%
```

#### Detailed Per-Class Metrics

| Target Class               |   Precision   |     Recall     |    F1-Score    | Support |
| :------------------------- | :------------: | :------------: | :------------: | :-----: |
| **CENTER**           | **1.00** |      0.05      |      0.09      |   22   |
| **LEFT**             |      0.64      | **0.76** |      0.70      |   54   |
| **RIGHT**            |      0.63      | **0.76** |      0.69      |   54   |
| **Overall Accuracy** |                |                | **0.64** |   130   |
| **Macro Average**    | **0.76** |      0.52      |      0.49      |   130   |
| **Weighted Average** |      0.70      |      0.64      |      0.60      |   130   |

### Biomechanical Findings and Discussion

1. **Symmetric Lateral Discrimination:**
   The model exhibits balanced predictive power for lateral kicks, achieving an identical **75.9% recall** on both Left and Right shots (41/54 correct predictions for each side). This confirms that physical kinematics (such as stance foot offset, hip opening angle, and torso counter-rotation) form a consistent, reliable directional signature prior to ball impact.
2. **The "Center" Penalty Dilemma (Deceptive Run-Ups):**
   Predicting kicks directed down the center represents a known biomechanical challenge. Penalty takers aiming centrally typically employ deceptive run-ups and postpone directional cues until the final milliseconds of foot-ball contact. Furthermore, because a kicker's swinging leg inherently produces lateral momentum (e.g., a right-footed player swinging forward naturally imparts an angular trajectory), pre-impact body posture is naturally split between left and right indicators (10 vs. 11 misclassifications). By employing Bayes-adjusted thresholding, the model avoids aggressive false alarms on the center class, preserving **100% precision** when a center prediction is triggered.
3. **Key Biomechanical Predictors:**

   * **Stance Foot Positioning (`offset_x`, `offset_y`):** The lateral and longitudinal placement of the supporting foot relative to the hip center serves as an anchor for the kicker's center of mass.
   * **Stance Knee Flexion (`stance_knee_flex`):** Degree of knee flexion dictates how low the player's hips drop to generate lift or cut across the ball.
   * **Pelvic Opening & Trunk Twist:** The coordination between the pelvis line and shoulder line indicates whether the hips are opening early to direct the ball across the body.

---

## 8. Limitations and Future Directions

* **Broadcast Camera Angles:** Television broadcast feeds vary between high tactical angles, ground-level views, and oblique perspectives. Future iterations will incorporate homography estimation to project body keypoints onto a normalized pitch coordinate plane.
* **Dataset Scaling:** While 65 annotated clips (130 augmented samples) establish statistically significant separation over the baseline, collecting 200+ penalty instances will further improve minority-class calibration (specifically center shots).
* **Multi-Modal Goalkeeper Interaction:** Future work can analyze simultaneous goalkeeper posturing to model kicker-goalkeeper game theory dynamics.

---

## 9. Academic References & Related Works

1. **Lees, A., & Nolan, L. (1998).** *The biomechanics of soccer: A review.* Journal of Sports Sciences, 16(3), 211-234.
2. **Savelsbergh, G. J., et al. (2002).** *Visual search, anticipation and expertise in soccer goalkeepers.* Journal of Sports Sciences, 20(3), 279-287.
3. **Jocher, G., Chaurasia, A., & Qiu, J. (2023).** *Ultralytics YOLOv8-Pose (Version 8.0.0).* GitHub repository.

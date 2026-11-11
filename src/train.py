"""
Model Training and Rigorous Player-Grouped Evaluation.
Evaluates using StratifiedGroupKFold on unseen players and benchmarks against
the majority-class baseline. Saves the trained model to models/.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_and_preprocess_data(dataset_path: Path = PROCESSED_DIR / "dataset_features.csv"):
    """Loads feature dataset and sets up target labels and player groups."""
    if not dataset_path.exists():
        print(f"[ERROR] Features dataset not found: {dataset_path}")
        print("Please run: python src/features.py")
        return None, None, None, None, None

    df = pd.read_csv(dataset_path)

    # Encode target direction in English: left, center, right
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["direction"])

    # Dominant foot: Right (R) = 1, Left (L) = 0
    df["is_right_foot"] = (df["kicking_foot"].str.upper() == "R").astype(int)

    # Group identifier for cross-validation
    groups = df["player"].values

    meta_cols = ["clip_id", "player", "kicking_foot", "direction", "fraction", "cutoff_frame", "total_frames"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    return df, X, y, groups, target_encoder


def evaluate_with_group_kfold(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, target_encoder: LabelEncoder):
    """Rigorous player-level cross-validation benchmarked against the majority baseline."""
    unique_players = len(np.unique(groups))
    n_splits = min(5, unique_players)

    if n_splits < 2:
        print("[WARNING] Insufficient distinct players for grouped cross-validation.")
        return

    cv = StratifiedGroupKFold(n_splits=n_splits)

    rf_accuracies = []
    baseline_accuracies = []
    y_true_all = []
    y_pred_all = []

    print(f"\n--- PLAYER-GROUPED CROSS-VALIDATION ({n_splits} Folds, {unique_players} Players) ---")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_players = np.unique(groups[val_idx])

        # Baseline: Always predicts the most common class in the training split
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        base_acc = dummy.score(X_val, y_val)
        baseline_accuracies.append(base_acc)

        # Biomechanical Random Forest model
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)
        rf_acc = accuracy_score(y_val, val_preds)
        rf_accuracies.append(rf_acc)

        y_true_all.extend(y_val)
        y_pred_all.extend(val_preds)

        print(f"Fold {fold} (Test Players: {', '.join(val_players)}):")
        print(f"  -> Majority Baseline: {base_acc:.1%}")
        print(f"  -> Random Forest:     {rf_acc:.1%}")

    mean_rf = np.mean(rf_accuracies)
    mean_base = np.mean(baseline_accuracies)

    print("\n=== FINAL VALIDATION SUMMARY ===")
    print(f"Mean Baseline Accuracy:       {mean_base:.1%}")
    print(f"Mean Biomechanical Accuracy:  {mean_rf:.1%}")
    print(f"Genuine Predictive Lift:      {mean_rf - mean_base:+.1%}")


def train_and_save_final_model():
    """Trains final model on all data and exports artifacts for the dashboard."""
    df, X, y, groups, target_encoder = load_and_preprocess_data()
    if X is None or len(X) == 0:
        return

    df_full = df[df["fraction"] == 1.0].copy()
    meta_cols = ["clip_id", "player", "kicking_foot", "direction", "fraction", "cutoff_frame", "total_frames"]
    feat_cols = [c for c in df_full.columns if c not in meta_cols]

    X_full = df_full[feat_cols].fillna(0)
    y_full = target_encoder.transform(df_full["direction"])

    evaluate_with_group_kfold(df_full, X_full, y_full, df_full["player"].values, target_encoder)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    final_clf.fit(X_full, y_full)

    joblib.dump(final_clf, MODELS_DIR / "penalty_model.joblib")
    joblib.dump(target_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feat_cols, MODELS_DIR / "feature_names.joblib")
    print(f"\n[MODEL EXPORTED] Successfully saved to {MODELS_DIR.name}/")


if __name__ == "__main__":
    train_and_save_final_model()

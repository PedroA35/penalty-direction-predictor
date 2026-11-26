"""
Model Benchmark and Rigorous Player-Grouped Evaluation.
Compares multiple algorithms using StratifiedGroupKFold on unseen players:
- Majority Baseline (Control)
- Logistic Regression (Regularized L2)
- Balanced Random Forest
- Gradient Boosting (Tuned Stochastic Boosting)
- Support Vector Machine (SVC with Probability Calibration)

Prints confusion matrix, per-class metrics, and automatically exports the top-performing model to models/.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


from src.config import DATA_DIR, PROCESSED_DIR, MODELS_DIR


def load_and_preprocess_data(dataset_path: Path = PROCESSED_DIR / "dataset_features.csv"):
    """Loads feature dataset and sets up target labels and player groups."""
    if not dataset_path.exists():
        print(f"[ERROR] Features dataset not found: {dataset_path}")
        print("Please run: python src/features.py")
        return None, None, None, None, None

    df = pd.read_csv(dataset_path)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["direction"])

    df["is_right_foot"] = (df["kicking_foot"].str.upper() == "R").astype(int)
    groups = df["player"].values

    meta_cols = ["clip_id", "player", "kicking_foot", "direction", "fraction", "cutoff_frame", "total_frames"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    return df, X, y, groups, target_encoder


def get_model_candidates():
    """Returns candidate classification algorithms for benchmarking."""
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        "Support Vector Machine (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True, C=1.0, random_state=42)
    }


def benchmark_models(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Evaluates all candidate algorithms using player-grouped cross-validation."""
    unique_players = len(np.unique(groups))
    n_splits = min(5, unique_players)

    cv = StratifiedGroupKFold(n_splits=n_splits)
    models = get_model_candidates()

    results = {name: [] for name in models}
    oof_preds = {name: np.zeros_like(y) for name in models}

    # Normalize features for linear models / SVM
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        y_train, y_val = y[train_idx], y[val_idx]
        sample_weights_tr = compute_sample_weight("balanced", y_train)

        for name, base_clf in models.items():
            clf = clone(base_clf)
            # Use scaled data for Logistic Regression and SVM, raw for trees
            X_tr = X_scaled.iloc[train_idx] if ("Logistic" in name or "SVM" in name) else X.iloc[train_idx]
            X_vl = X_scaled.iloc[val_idx] if ("Logistic" in name or "SVM" in name) else X.iloc[val_idx]

            try:
                clf.fit(X_tr, y_train, sample_weight=sample_weights_tr)
            except TypeError:
                clf.fit(X_tr, y_train)

            if hasattr(clf, "predict_proba") and name != "Majority Baseline":
                probs = clf.predict_proba(X_vl)
                priors = np.bincount(y_train, minlength=len(np.unique(y))) / len(y_train)
                # Ponderação Bayesiana: atenua a penalização a priori sobre classes minoritárias (Center)
                adj_scores = probs / np.sqrt(np.maximum(priors, 1e-4))
                preds = np.argmax(adj_scores, axis=1)
            else:
                preds = clf.predict(X_vl)

            oof_preds[name][val_idx] = preds

            score = accuracy_score(y_val, preds)
            results[name].append(score)

    return results, oof_preds, scaler


def run_benchmark_and_export():
    """Runs the full model benchmark, prints summary table, confusion matrix, and exports the winner."""
    df, X, y, groups, target_encoder = load_and_preprocess_data()
    if X is None or len(X) == 0:
        return

    # Evaluate exclusively on full approach samples (fraction = 1.0)
    df_full = df[df["fraction"] == 1.0].copy()
    meta_cols = ["clip_id", "player", "kicking_foot", "direction", "fraction", "cutoff_frame", "total_frames"]
    feat_cols = [c for c in df_full.columns if c not in meta_cols]

    X_full = df_full[feat_cols].fillna(0)
    y_full = target_encoder.transform(df_full["direction"])
    groups_full = df_full["player"].values

    print("\n" + "="*65)
    print("   PENALTY DIRECTION PREDICTOR - MULTI-MODEL BENCHMARK")
    print(f"   Validation: StratifiedGroupKFold on {len(np.unique(groups_full))} players")
    print("="*65)

    results, oof_preds, scaler = benchmark_models(X_full, y_full, groups_full)

    baseline_score = np.mean(results["Majority Baseline"])
    ranking = []

    for name, scores in results.items():
        mean_score = np.mean(scores)
        lift = mean_score - baseline_score
        ranking.append((name, mean_score, lift))

    ranking.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Model Candidate':<32} | {'Mean Accuracy':<15} | {'Gain vs Baseline'}")
    print("-" * 65)
    for name, score, lift in ranking:
        print(f"{name:<32} | {score:<15.1%} | {lift:+.1%}")
    print("-" * 65)

    winner_name = ranking[0][0]
    print(f"\n[WINNER SELECTED] Top performing model: {winner_name} ({ranking[0][1]:.1%})")

    # Confusion Matrix and Per-Class Breakdown on Out-Of-Fold Predictions
    winner_preds = oof_preds[winner_name]
    classes = target_encoder.classes_
    cm = confusion_matrix(y_full, winner_preds)

    print("\n" + "="*65)
    print(f"   CONFUSION MATRIX - {winner_name.upper()}")
    print("   (Evaluated strictly on unseen players via GroupKFold)")
    print("="*65)

    header = f"{'Actual \\ Pred':<16}" + "".join([f"{c.upper():>14}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, actual_class in enumerate(classes):
        row_str = f"{actual_class.upper():<16}"
        for j in range(len(classes)):
            row_str += f"{cm[i, j]:>14}"
        print(row_str)

    print("\n" + "-"*65)
    print("PER-CLASS CLASSIFICATION REPORT:")
    print(classification_report(y_full, winner_preds, target_names=[c.upper() for c in classes], digits=1, zero_division=0))

    # Fit final winning model on all data with balanced sample weights
    models = get_model_candidates()
    best_clf = models[winner_name]

    if "Logistic" in winner_name or "SVM" in winner_name:
        X_final = scaler.fit_transform(X_full)
    else:
        X_final = X_full

    sample_weights_full = compute_sample_weight("balanced", y_full)
    try:
        best_clf.fit(X_final, y_full, sample_weight=sample_weights_full)
    except TypeError:
        best_clf.fit(X_final, y_full)

    # Feature importances (if tree-based)
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("=== TOP BIOMECHANICAL FEATURES (Physics-Informed) ===")
        for rank, idx in enumerate(sorted_idx[:5], 1):
            print(f"  {rank}. {feat_cols[idx]}: {importances[idx]:.1%}")

    # Export artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODELS_DIR / "penalty_model.joblib")
    joblib.dump(target_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feat_cols, MODELS_DIR / "feature_names.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(winner_name, MODELS_DIR / "winner_name.joblib")
    print(f"\n[MODEL EXPORTED] Best model and metadata successfully saved to {MODELS_DIR.name}/")


if __name__ == "__main__":
    run_benchmark_and_export()

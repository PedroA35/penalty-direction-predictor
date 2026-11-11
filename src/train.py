"""
Módulo de treino, avaliação rigorosa por jogador e curvas de antecedência.
Garante que a validação é feita por GroupKFold (nunca dividindo o mesmo jogador
entre treino e teste) e compara sempre com o baseline ingénuo da classe maioritária.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import joblib


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_and_preprocess_data(dataset_path: Path = PROCESSED_DIR / "dataset_features.csv"):
    """Carrega o dataset e prepara as variáveis preditoras (X) e a variável alvo (y)."""
    if not dataset_path.exists():
        print(f"[ERRO] Dataset de atributos não encontrado: {dataset_path}")
        print("Executa primeiro: python src/features.py")
        return None, None, None, None, None

    df = pd.read_csv(dataset_path)

    # Codificar a direção (alvo): esquerda, centro, direita
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["direcao"])

    # Codificar o pé de remate (D=1, E=0)
    df["pe_direito"] = (df["pe_remate"].str.upper() == "D").astype(int)

    # Identificadores de agrupamento
    groups = df["jogador"].values

    # Remover colunas de metadados das variáveis preditoras (features)
    meta_cols = ["clip_id", "jogador", "pe_remate", "direcao", "fraction", "cutoff_frame", "total_frames"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    return df, X, y, groups, target_encoder


def evaluate_with_group_kfold(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, target_encoder: LabelEncoder):
    """
    Avaliação rigorosa com validação cruzada agrupada por jogador.
    Compara o Random Forest com o Baseline de classe mais frequente.
    """
    unique_players = len(np.unique(groups))
    n_splits = min(5, unique_players)

    if n_splits < 2:
        print("[AVISO] Número insuficiente de jogadores distintos para validação agrupada.")
        print("Precisas de pelo menos 2 a 3 jogadores diferentes no labels.csv.")
        return

    cv = StratifiedGroupKFold(n_splits=n_splits)

    rf_accuracies = []
    baseline_accuracies = []
    y_true_all = []
    y_pred_all = []

    print(f"\n--- VALIDAÇÃO CRUZADA POR JOGADOR ({n_splits} Folds, {unique_players} Jogadores Únicos) ---")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_players = np.unique(groups[val_idx])

        # 1. Baseline: prevê sempre o lado mais frequente no conjunto de treino
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        base_acc = dummy.score(X_val, y_val)
        baseline_accuracies.append(base_acc)

        # 2. Modelo Random Forest
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)
        rf_acc = accuracy_score(y_val, val_preds)
        rf_accuracies.append(rf_acc)

        y_true_all.extend(y_val)
        y_pred_all.extend(val_preds)

        print(f"Fold {fold} (Jogadores de teste: {', '.join(val_players)}):")
        print(f"  -> Baseline (mais frequente): {base_acc:.1%}")
        print(f"  -> Random Forest:            {rf_acc:.1%}")

    mean_rf = np.mean(rf_accuracies)
    mean_base = np.mean(baseline_accuracies)

    print("\n=== RESULTADO FINAL DE VALIDAÇÃO ===")
    print(f"Acurácia Média Baseline (chute no lado mais comum): {mean_base:.1%}")
    print(f"Acurácia Média Modelo Biomecânico (Random Forest):   {mean_rf:.1%}")
    print(f"Ganho real sobre o Baseline:                       {mean_rf - mean_base:+.1%}")

    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_true_all, y_pred_all, target_names=target_encoder.classes_))

    # Matriz de Confusão
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
    print("Matriz de Confusão (Linhas: Real | Colunas: Previsto):")
    print(cm_df)


def analyze_lead_time_curve(df: pd.DataFrame, target_encoder: LabelEncoder):
    """
    Analisa a curva de antecedência: como varia o acerto do modelo
    conforme a fração de corrida disponível (ex: 50% vs 75% vs 100%).
    """
    fractions = sorted(df["fraction"].unique())
    print("\n=== CURVA DE ACERTO VS. ANTECEDÊNCIA TEMPORAL ===")

    for frac in fractions:
        sub_df = df[df["fraction"] == frac].copy()
        if len(sub_df) < 5:
            continue

        meta_cols = ["clip_id", "jogador", "pe_remate", "direcao", "fraction", "cutoff_frame", "total_frames"]
        feature_cols = [c for c in sub_df.columns if c not in meta_cols]

        X_f = sub_df[feature_cols].fillna(0)
        y_f = target_encoder.transform(sub_df["direcao"])
        groups_f = sub_df["jogador"].values

        unique_p = len(np.unique(groups_f))
        if unique_p < 2:
            continue

        cv = StratifiedGroupKFold(n_splits=min(3, unique_p))
        scores = []
        for train_i, test_i in cv.split(X_f, y_f, groups=groups_f):
            clf = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42)
            clf.fit(X_f.iloc[train_i], y_f[train_i])
            scores.append(accuracy_score(y_f[test_i], clf.predict(X_f.iloc[test_i])))

        print(f"Fração de corrida disponível: {int(frac * 100)}% -> Acurácia média: {np.mean(scores):.1%}")


def train_and_save_final_model():
    """Treina o modelo em todos os dados e guarda em ficheiro para a demo em Streamlit."""
    df, X, y, groups, target_encoder = load_and_preprocess_data()
    if X is None or len(X) == 0:
        return

    # Usar apenas os atributos de 100% do clip para o modelo de inferência final
    df_full = df[df["fraction"] == 1.0].copy()
    meta_cols = ["clip_id", "jogador", "pe_remate", "direcao", "fraction", "cutoff_frame", "total_frames"]
    feat_cols = [c for c in df_full.columns if c not in meta_cols]

    X_full = df_full[feat_cols].fillna(0)
    y_full = target_encoder.transform(df_full["direcao"])

    evaluate_with_group_kfold(df_full, X_full, y_full, df_full["jogador"].values, target_encoder)
    analyze_lead_time_curve(df, target_encoder)

    # Treina o modelo final com todos os dados
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    final_clf.fit(X_full, y_full)

    joblib.dump(final_clf, MODELS_DIR / "penalty_model.joblib")
    joblib.dump(target_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feat_cols, MODELS_DIR / "feature_names.joblib")
    print(f"\n[MODELO GUARDADO] Modelo e codificadores exportados para a pasta {MODELS_DIR.name}/")


if __name__ == "__main__":
    train_and_save_final_model()


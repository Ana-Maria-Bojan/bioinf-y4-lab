
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Config
# --------------------------
# TODO: 
HANDLE = "Ana-Maria-Bojan"


DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
TOPK_FEATURES = 20 

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONFUSION = OUT_DIR / f"confusion_rf_{HANDLE}.png"
OUT_REPORT = OUT_DIR / f"classification_report_{HANDLE}.txt"
OUT_FEATIMP = OUT_DIR / f"feature_importance_{HANDLE}.csv"
OUT_CLUSTER_CROSSTAB = OUT_DIR / f"cluster_crosstab_{HANDLE}.csv"


def ensure_exists(path: Path) -> None:
   
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul: {path}")
    print(f"[INFO] Fișier găsit: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    
    df = pd.read_csv(path)
    
    # X = toate coloanele mai puțin ultima
    X = df.iloc[:, :-1]
    # y = ultima coloană
    y = df.iloc[:, -1]
    
    print(f"[INFO] Dataset încărcat. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Folosește LabelEncoder pentru a converti etichetele string în valori numerice.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"[INFO] Etichete encodate: {list(le.classes_)}")
    return y_encoded, le


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> RandomForestClassifier:
    """
    Inițializează și antrenează RandomForestClassifier.
    """
    print(f"[INFO] Antrenare Random Forest cu {n_estimators} estimatori...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Folosește toate procesoarele
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_png: Path,
    out_txt: Path,
) -> None:
    """
    Calculează predicțiile, classification report și matricea de confuzie.
    """
    print("[INFO] Evaluare model...")
    y_pred = model.predict(X_test)
    
    target_names = [str(cls) for cls in label_encoder.classes_]
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("--- Classification Report ---")
    print(report)
    
    # Salvare raport în fișier text
    out_txt.write_text(report, encoding="utf-8")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Matricea de confuzie salvată la: {out_png}")


def compute_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Extrage importanța trăsăturilor, sortează și salvează.
    """
    importances = model.feature_importances_
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    
    # Sortare descrescătoare
    df_imp = df_imp.sort_values("Importance", ascending=False)
    
    # Salvare
    df_imp.to_csv(out_csv, index=False)
    print(f"[INFO] Feature importance salvat la: {out_csv}")
    
    # Afișăm top 5 în consolă
    print(f"--- Top 5 Gene Importante ---\n{df_imp.head(5)}")
    
    return df_imp


def run_kmeans_and_crosstab(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    n_clusters: int,
    out_csv: Path,
) -> None:
    """
    (Opțional) Rulare KMeans și comparare clustere vs etichete reale.
    """
    print(f"[INFO] Rulare KMeans cu {n_clusters} clustere...")
    
    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=RANDOM_STATE, 
        n_init="auto"
    )
    clusters = kmeans.fit_predict(X) # X are deja doar coloane numerice
    
    # Creăm un DataFrame pentru comparație
    # decodificăm y înapoi în string-uri (ex: "Cancer_A") pentru claritate
    y_str = label_encoder.inverse_transform(y)
    
    df_compare = pd.DataFrame({
        "True_Label": y_str,
        "KMeans_Cluster": clusters
    })
    
    # Crosstab (tabel de contingență)
    ctab = pd.crosstab(df_compare["True_Label"], df_compare["KMeans_Cluster"])
    
    ctab.to_csv(out_csv)
    print("--- Crosstab: Etichete Reale vs Clustere ---")
    print(ctab)
    print(f"[INFO] Crosstab salvat la: {out_csv}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # 1. Verificăm fișierul
    ensure_exists(DATA_CSV)

    # 2. Încărcăm datele
    X, y = load_dataset(DATA_CSV)

    # 3. Encodăm etichetele și împărțim în train/test
    y_enc, le = encode_labels(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc, # Păstrează proporția claselor
    )
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Antrenăm modelul RF și evaluăm
    rf_model = train_random_forest(X_train, y_train, N_ESTIMATORS, RANDOM_STATE)
    evaluate_model(rf_model, X_test, y_test, le, OUT_CONFUSION, OUT_REPORT)

    # 5. Calculăm importanța trăsăturilor
    feat_imp_df = compute_feature_importance(rf_model, X.columns, OUT_FEATIMP)

    # 6. (Opțional) KMeans
    n_classes = len(le.classes_)
    run_kmeans_and_crosstab(X, y_enc, le, n_clusters=n_classes, out_csv=OUT_CLUSTER_CROSSTAB)

    print("\n[DONE] Pipeline finalizat cu succes!")
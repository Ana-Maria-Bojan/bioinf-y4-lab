
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Config
# --------------------------
# TODO: Modifică aici dacă folosești alt handle
HANDLE = "Ana-Maria-Bojan"

DATA_CSV = Path(f"data/work/{HANDLE}/lab08/expression_matrix_{HANDLE}.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_ITER_LOGREG = 2000  # Crescut pentru convergență

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    """
    Verifică dacă fișierul de input există.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul: {path}")
    print(f"[INFO] Fișier găsit: {path}")


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Citește CSV-ul cu pandas.
    """
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Label
    print(f"[INFO] Dataset încărcat: {X.shape} samples.")
    return X, y


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Folosește LabelEncoder pentru a obține y_enc.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, LogisticRegression, StandardScaler]:
    """
    Antrenează Random Forest și Logistic Regression.
    IMPORTANT: Logistic Regression necesită date scalate (StandardScaler).
    """
    print("[INFO] Scalare date (StandardScaler)...")
    scaler = StandardScaler()
    # Scalăm doar pe train, apoi aplicăm transformarea pe test mai târziu
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Random Forest (nu necesită neapărat scalare, dar merge și cu ea)
    print(f"[INFO] Antrenare Random Forest ({N_ESTIMATORS} trees)...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 2. Logistic Regression (Multinomial)
    print("[INFO] Antrenare Logistic Regression...")
    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs", # bun pentru multiclass
        max_iter=MAX_ITER_LOGREG,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    logreg.fit(X_train_scaled, y_train)

    return rf, logreg, scaler


def compare_models(
    rf: RandomForestClassifier,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_txt: Path,
) -> None:
   
    print("[INFO] Evaluare modele...")
    
    # Trebuie să scalăm X_test folosind ACELAȘI scaler antrenat pe train
    X_test_scaled = scaler.transform(X_test)

    # Predicții
    y_pred_rf = rf.predict(X_test)
    y_pred_logreg = logreg.predict(X_test_scaled)

    target_names = [str(c) for c in label_encoder.classes_]

    # Generare rapoarte
    report_rf = classification_report(y_test, y_pred_rf, target_names=target_names)
    report_logreg = classification_report(y_test, y_pred_logreg, target_names=target_names)

    # Afișare în consolă
    print("\n" + "="*30)
    print("RANDOM FOREST REPORT")
    print("="*30)
    print(report_rf)

    print("\n" + "="*30)
    print("LOGISTIC REGRESSION REPORT")
    print("="*30)
    print(report_logreg)

    # Salvare în fișier
    combined_report = (
        f"COMPARISON REPORT: {HANDLE}\n"
        f"{'='*30}\n"
        f"RANDOM FOREST\n"
        f"{'='*30}\n{report_rf}\n\n"
        f"{'='*30}\n"
        f"LOGISTIC REGRESSION (Scaled)\n"
        f"{'='*30}\n{report_logreg}\n"
    )
    
    out_txt.write_text(combined_report, encoding="utf-8")
    print(f"[INFO] Raport salvat la: {out_txt}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # 1. Verificare fișier
    ensure_exists(DATA_CSV)

    # 2. Încărcare date
    X, y = load_dataset(DATA_CSV)

    # 3. Encodare și Split
    y_enc, le = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # 4. Antrenare
    rf_model, logreg_model, scaler_obj = train_models(X_train, y_train)

    # 5. Comparare
    compare_models(rf_model, logreg_model, scaler_obj, X_test, y_test, le, OUT_REPORT_TXT)

    print("\n[DONE] Comparatie finalizata.")
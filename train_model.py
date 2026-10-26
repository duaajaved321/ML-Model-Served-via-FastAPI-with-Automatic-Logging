"""Train a reproducible demonstration model on UCI dataset ID 529."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from schema import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'diabetes_pipeline.joblib'

def get_uci_data():
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=529)
        X = data.data.features.copy()
        y = data.data.targets.squeeze().copy()
        return X, y
    except Exception as exc:
        raise RuntimeError('Could not retrieve UCI dataset 529. Check your internet connection and retry.') from exc

def train():
    X, y = get_uci_data()
    X.columns = [c.strip().lower().replace(' ', '_') for c in X.columns]
    aliases = {'sudden_weight_loss':'sudden_weight_loss', 'muscle_stiffness':'muscle_stiffness'}
    X = X.rename(columns=aliases)
    missing = set(FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f'Dataset missing features: {sorted(missing)}; got {list(X.columns)}')
    X = X[FEATURES]
    X['age'] = pd.to_numeric(X['age'], errors='raise')
    for col in FEATURES[1:]:
        X[col] = X[col].astype(str).str.strip().str.title()
    y = y.astype(str).str.strip().str.title()
    valid = X.notna().all(axis=1) & y.isin(['Positive', 'Negative'])
    X, y = X.loc[valid], y.loc[valid]
    # Exact duplicates may occur naturally in questionnaire records; removed as in our previous exercise.
    combined = X.copy()
    combined['class'] = y
    combined = combined.drop_duplicates().reset_index(drop=True)
    X, y = combined[FEATURES], combined['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer(
        transformers=[('category', OneHotEncoder(handle_unknown='ignore'), FEATURES[1:])],
        remainder='passthrough',
    )
    pipeline = Pipeline([
        ('preprocessing', prep),
        ('classifier', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    report = {
        'dataset': 'UCI Early Stage Diabetes Risk Prediction (ID 529)',
        'retained_records': int(len(X)),
        'training_records': int(len(X_train)),
        'test_records': int(len(X_test)),
        'accuracy': float(accuracy_score(y_test, predictions)),
        'classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0),
        'note': 'Demonstration model, not validated for medical use.'
    }
    (ROOT/'models'/'training_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Model saved: {MODEL_PATH}')
    print(f'Test accuracy (random holdout; not clinical validation): {report["accuracy"]:.3f}')
    print(f'Rows used after de-duplication: {len(X)}')

if __name__ == '__main__':
    train()

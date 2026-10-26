"""Tests use generated records only; no synthetic outputs should be treated as UCI results."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import FEATURES
from app import app

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Synthetic model tests infrastructure independently from UCI training."""
    base = json.loads((ROOT/'examples'/'valid_request.json').read_text())
    rows, labels = [], []
    for i in range(32):
        row = base.copy()
        row['age'] = 20+i
        row['gender'] = 'Male' if i%2 else 'Female'
        row['polyuria'] = 'Yes' if i%2 else 'No'
        row['polydipsia'] = 'Yes' if i%2 else 'No'
        rows.append(row)
        labels.append('Positive' if i%2 else 'Negative')
    transformer = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES[1:])], remainder='passthrough')
    model = Pipeline([('preprocessing', transformer), ('classifier', RandomForestClassifier(n_estimators=15, random_state=42))])
    model.fit(pd.DataFrame(rows)[FEATURES], labels)
    model_path = tmp_path/'test_model.joblib'
    joblib.dump(model, model_path)
    monkeypatch.setenv('MODEL_PATH', str(model_path))
    monkeypatch.setenv('AUDIT_DB_PATH', str(tmp_path/'test_audit.db'))
    with TestClient(app) as c:
        yield c, tmp_path/'test_audit.db'

def audit_rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute('SELECT http_status, outcome, error_type FROM prediction_audit ORDER BY id').fetchall()

def test_valid_predict_and_audit(client):
    c, db = client
    payload = json.loads((ROOT/'examples'/'valid_request.json').read_text())
    r = c.post('/predict', json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['prediction'] in {'Positive','Negative'}
    assert abs(sum(body['probabilities'].values())-1) < 0.00001
    assert body['request_id']
    assert audit_rows(db) == [(200,'success',None)]

def test_validation_fail_audit(client):
    c, db = client
    r = c.post('/predict', json={'age':-5, 'gender':'Male', 'polyuria':'Yes'})
    assert r.status_code == 422
    assert 'request_id' in r.json()
    assert audit_rows(db) == [(422,'failed','RequestValidationError')]

def test_extra_fields_rejected(client):
    c, db = client
    payload = json.loads((ROOT/'examples'/'valid_request.json').read_text())
    payload['full_name'] = 'should not be accepted'
    r = c.post('/predict', json=payload)
    assert r.status_code == 422
    assert audit_rows(db)[0][1] == 'failed'

def test_health(client):
    c, _ = client
    r = c.get('/health')
    assert r.status_code == 200 and r.json()['model_loaded'] is True

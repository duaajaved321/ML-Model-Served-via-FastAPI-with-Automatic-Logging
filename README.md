# Diabetes Risk Prediction API — FastAPI + SQLite

A complete **educational ML-serving assignment** built with FastAPI, a scikit-learn Random Forest pipeline, Pydantic validation, and automatic SQLite audit logging. **Not a medical device, diagnostic tool, or medical advice.**

## Features

- `POST /predict` accepts 16 fields (age, gender, 14 binary symptom responses) and returns the predicted class plus `Positive`/`Negative` class probabilities.
- Pydantic rejects invalid types, unknown keys, missing fields, and age outside 1–120; validation errors use HTTP 422.
- Each successful prediction is logged as HTTP 200; invalid request is logged as HTTP 422; inference errors are logged as HTTP 500.
- SQLite records UTC timestamp, request UUID, outcome, HTTP status, predicted class or error category **without saving raw patient answers**.
- `GET /health` confirms server and model are ready.
- Training code uses the **actual UCI dataset ID 529** when executed with internet access; test suite uses **synthetic data only**, independent of UCI availability.

## Project files

```text
app.py                     API, validation handler and prediction logic
database.py                SQLite table and inserts
schema.py                  Strict request body fields
train_model.py             Fetch UCI ID 529, train Random Forest, save model
models/                    Trained model and metrics appear here after training
audit_report.py            Print the 20 latest audit rows and totals
examples/valid_request.json
examples/invalid_request.json
examples/audit_demo.txt    Evidence from real test requests, synthetic TEST MODEL
tests/test_app.py           4 FastAPI / SQLite tests
requirements.txt
.gitignore
README.md
```

## Setup (Windows PowerShell / VS Code)

From the extracted project directory:

```powershell
conda create -n diabetes_api python=3.11 -y
conda activate diabetes_api
python -m pip install -r requirements.txt
```

Or without Conda:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 1. Train and save the model

Requires internet access **once**, to fetch UCI dataset 529:

```powershell
python train_model.py
```

This creates `models/diabetes_pipeline.joblib` and `models/training_metrics.json` locally. The saved pipeline includes **one-hot preprocessing and the model**, so the API accepts human-readable `Yes`/`No` values rather than manually encoded columns.

Training: clean column names, remove exact duplicate questionnaires including target, stratified 80/20 split (`random_state=42`), OneHotEncoder, RandomForestClassifier. `training_metrics.json` contains test accuracy and classification metrics. **A single random split on this small dataset cannot establish real-world/clinical performance**.

> The ZIP deliberately does not contain a model falsely described as trained on the UCI data. The creation environment could not access the UCI download. Run `python train_model.py` on an internet-connected PC to obtain your actual trained model. The included FastAPI tests have already passed using a clearly identified *synthetic test model*.

### 2. Start the API

```powershell
uvicorn app:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

### 3. Make requests

**Valid JSON body** (`POST http://127.0.0.1:8000/predict`):

```json
{
  "age": 45,
  "gender": "Male",
  "polyuria": "Yes",
  "polydipsia": "Yes",
  "sudden_weight_loss": "No",
  "weakness": "Yes",
  "polyphagia": "No",
  "genital_thrush": "No",
  "visual_blurring": "Yes",
  "itching": "No",
  "irritability": "No",
  "delayed_healing": "No",
  "partial_paresis": "No",
  "muscle_stiffness": "No",
  "alopecia": "No",
  "obesity": "No"
}
```

In PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -InFile examples/valid_request.json
```

Response **shape** (illustrative values; actual results depend on the trained model):

```json
{
  "request_id": "<generated-uuid>",
  "prediction": "Positive",
  "probabilities": {"Negative": 0.12, "Positive": 0.88},
  "disclaimer": "Educational demonstration only; not a medical diagnosis or screening tool."
}
```

Invalid request (age negative and many required fields absent):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -InFile examples/invalid_request.json
```

FastAPI returns **422 Unprocessable Entity**, with `request_id` and a list of validation issues. PowerShell may display an HTTP error for a non-2xx status; that is expected.

Or send both requests via the interactive `/docs` page.

### 4. Show audit trail

```powershell
python audit_report.py
```

A separate SQLite `audit.db` will be created in the project root. Each prediction attempt creates one row:

| HTTP | Outcome | Logged data |
| --- | --- | --- |
| 200 | success | request ID, timestamp, predicted class |
| 422 | failed | request ID, timestamp, `RequestValidationError` |
| 500 | failed | request ID, timestamp, `PredictionError` |

See [`examples/audit_demo.txt`](examples/audit_demo.txt) for an audit demonstration from integration tests executed against **a synthetic training fixture**. Your UCI-trained run will produce new timestamps and UUIDs.

The endpoint **does not expose audit records publicly**; use `audit_report.py` on the server for inspection. SQL queries use placeholders rather than direct string concatenation.

### 5. Automated tests

```powershell
python -m pytest -q
```

Tests cover successful prediction and probability sum, validation failures logged with 422, extra-field rejection, and health check. **Four tests passed locally** on the provided project. Tests train an artificial fixture model, not the UCI model, so these tests verify **API behavior**, not predictive accuracy.

## Limitations and ethics

- Medical demonstration only. A questionnaire prediction is **not diabetes diagnosis** and should not be used to guide treatment or health decisions.
- UCI dataset was collected in one hospital setting and is small; risk of sample bias and weak out-of-sample generalization.
- The returned model probabilities are **not calibrated clinical risk estimates**.
- One stratified holdout score is not clinical validation. Near-identical records, selection bias, feature reliability and dataset shifts can inflate apparent performance.
- Validation bounds (`age` 1–120, `Male/Female`, `Yes/No`) reflect the API contract rather than all human possibilities; the original dataset has its own demographics and limitations.
- SQLite is reasonable for small, single-instance projects, not high-throughput distributed traffic.
- Audit records exclude questionnaire answers for privacy, but production deployment needs authentication, TLS, retention controls, logging monitoring, and explicit data policies.
- There is no live public hosting; this is a locally served deployable API.

## Data attribution

Early Stage Diabetes Risk Prediction Dataset, UCI Machine Learning Repository (2020), DOI: https://doi.org/10.24432/C5VG8H ; licensed CC BY 4.0. Dataset originally recorded from questionnaires at Sylhet Diabetes Hospital, Bangladesh. The training script fetches this dataset from the authoritative repository; no personal medical records are included in this ZIP.

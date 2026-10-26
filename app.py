"""FastAPI model-serving endpoint with automatic success/failure SQLite logging."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from database import initialize, log_request
from schema import PatientInput, FEATURES

ROOT = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    model_path = Path(os.getenv('MODEL_PATH', str(ROOT / 'models' / 'diabetes_pipeline.joblib')))
    if not model_path.is_file():
        raise RuntimeError(f'Model not found at {model_path}. Run: python train_model.py')
    # joblib files must be trusted; do not load arbitrary user uploads.
    app.state.model = joblib.load(model_path)
    yield
    app.state.model = None

app = FastAPI(title='Educational Diabetes Risk Prediction API', version='1.0', lifespan=lifespan)

class PredictResponse(BaseModel):
    request_id: str
    prediction: str
    probabilities: dict[str, float]
    disclaimer: str

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': app.state.model is not None}

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    if request.url.path != '/predict':
        return JSONResponse(status_code=422, content={'detail': exc.errors()})
    request_id = str(uuid4())
    # Only error type stored; never store patient answers in SQLite.
    log_request(request_id, 422, 'failed', error_type='RequestValidationError')
    errors = [
        {'field': '.'.join(map(str, e['loc'])), 'message': e['msg'], 'type': e['type']}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={'request_id': request_id, 'detail': errors})

@app.post('/predict', response_model=PredictResponse)
def predict(patient: PatientInput):
    request_id = str(uuid4())
    try:
        inputs = pd.DataFrame([patient.model_dump()], columns=FEATURES)
        model = app.state.model
        prediction = str(model.predict(inputs)[0])
        probabilities = model.predict_proba(inputs)[0]
        classes = list(model.classes_)
        result = {str(k): round(float(v), 6) for k, v in zip(classes, probabilities)}
    except Exception:
        log_request(request_id, 500, 'failed', error_type='PredictionError')
        return JSONResponse(status_code=500, content={
            'request_id': request_id, 'detail': 'Prediction failed; see server logs.'
        })
    log_request(request_id, 200, 'success', predicted_class=prediction)
    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        probabilities=result,
        disclaimer='Educational demonstration only; not a medical diagnosis or screening tool.'
    )

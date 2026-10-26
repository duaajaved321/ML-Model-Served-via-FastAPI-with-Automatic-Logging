"""Single source of truth for API input and model columns."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal['Yes', 'No']
SYMPTOMS = [
    'polyuria', 'polydipsia', 'sudden_weight_loss', 'weakness',
    'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
    'irritability', 'delayed_healing', 'partial_paresis',
    'muscle_stiffness', 'alopecia', 'obesity'
]
FEATURES = ['age', 'gender', *SYMPTOMS]

class PatientInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    age: int = Field(ge=1, le=120, strict=True, description='Age in whole years')
    gender: Literal['Male', 'Female']
    polyuria: YesNo
    polydipsia: YesNo
    sudden_weight_loss: YesNo
    weakness: YesNo
    polyphagia: YesNo
    genital_thrush: YesNo
    visual_blurring: YesNo
    itching: YesNo
    irritability: YesNo
    delayed_healing: YesNo
    partial_paresis: YesNo
    muscle_stiffness: YesNo
    alopecia: YesNo
    obesity: YesNo

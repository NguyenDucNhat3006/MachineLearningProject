"""Run: python -m uvicorn api:app --host 127.0.0.1 --port 8000"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Any
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AwareDatetime, ConfigDict

from airpulse_v4 import recommend, load_bundle, InputError, SCHEMA_VERSION, VERSION

ROOT=Path(__file__).resolve().parent
MODEL_PATH=Path(os.getenv("AIRPULSE_MODEL",str(ROOT/"trained/airpulse_v4_model.joblib")))
if not MODEL_PATH.is_absolute():MODEL_PATH=ROOT/MODEL_PATH


@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.is_file():raise RuntimeError(f"Missing model: {MODEL_PATH}; run python train.py")
    app.state.bundle=load_bundle(MODEL_PATH)
    yield
    app.state.bundle=None


app=FastAPI(title="AirPulse V4 Research API",version=VERSION,lifespan=lifespan)
origins=[x.strip() for x in os.getenv("AIRPULSE_CORS_ORIGINS",
    "https://airpulse-demo.vercel.app,http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,
                   allow_methods=["GET","POST","OPTIONS"],allow_headers=["Content-Type"])


class PriceQuote(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    itinerary_id: str=Field(min_length=1,max_length=200)
    observed_at: AwareDatetime
    departure_at: AwareDatetime
    origin: str=Field(pattern=r"^[A-Za-z]{3}$")
    destination: str=Field(pattern=r"^[A-Za-z]{3}$")
    airline: str=Field(min_length=1,max_length=100)
    price_vnd: float=Field(gt=0,le=1_000_000_000)
    currency: Literal["VND"]="VND"
    fare_family: str="UNSPECIFIED"
    seller: str="UNSPECIFIED"
    cabin: str="UNSPECIFIED"
    baggage: str="UNSPECIFIED"


class ForecastRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mode: Literal["replay","live"]="replay"
    as_of: AwareDatetime | None=None
    history: list[PriceQuote]=Field(min_length=4,max_length=500)


class InsightResponse(BaseModel):
    schema_version: str
    model_version: str
    model_id: str
    mode: str
    status: str
    training_data_kind: str
    recommendation: dict[str,Any]
    forecast_slots: list[dict[str,Any]]
    probability_event: dict[str,Any]
    warnings: list[str]
    policy_validation: dict[str,Any] | None=None
    valid_until: str | None=None


@app.get("/health")
def health():
    bundle=getattr(app.state,"bundle",None)
    return {"status":"ok" if bundle is not None else "unavailable",
            "model_loaded":bundle is not None,"schema_version":SCHEMA_VERSION,
            "live_validated":bool(bundle and bundle.get("validated_for_live")),
            "model_id":bundle.get("model_id") if bundle else None}


@app.post("/predict",response_model=InsightResponse)
@app.post("/insight",response_model=InsightResponse)
def insight(body:ForecastRequest):
    bundle=getattr(app.state,"bundle",None)
    if bundle is None:raise HTTPException(status_code=503,detail={"code":"MODEL_UNAVAILABLE"})
    history=pd.DataFrame([q.model_dump(mode="json") for q in body.history])
    try:
        return recommend(bundle,history,mode=body.mode,as_of=body.as_of)
    except InputError as e:
        raise HTTPException(status_code=422,detail={"code":e.code,"message":str(e)}) from e


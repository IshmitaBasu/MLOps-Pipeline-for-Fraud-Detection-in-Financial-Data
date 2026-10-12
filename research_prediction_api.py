"""Local-only research API over a frozen model and read-only history state."""

# %% Imports and request validation
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator

from research_serving_utils import ResearchPredictor, ServingError, load_deployment

Number = StrictInt | StrictFloat
NonNegative = Annotated[Number, Field(ge=0, allow_inf_nan=False)]
Finite = Annotated[Number, Field(allow_inf_nan=False)]


# %% Validated transaction request schema
class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: StrictStr
    event_timestamp: StrictStr
    sender_account: StrictStr
    transaction_type: StrictStr | None
    merchant_category: StrictStr | None
    location: StrictStr
    device_used: StrictStr | None
    payment_channel: StrictStr | None
    amount: NonNegative
    time_since_last_transaction: NonNegative | None
    spending_deviation_score: Finite | None
    velocity_score: StrictInt | None
    geo_anomaly_score: Finite | None

    @field_validator(
        "transaction_id",
        "sender_account",
        "transaction_type",
        "merchant_category",
        "location",
        "device_used",
        "payment_channel",
    )
    @classmethod
    def valid_string(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("Non-empty strings without surrounding whitespace are required.")
        return value

    @field_validator("event_timestamp")
    @classmethod
    def valid_timestamp(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", value):
            raise ValueError("Use ISO-8601 whole-second precision with an explicit timezone.")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("An ISO-8601 timestamp is required.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond:
            raise ValueError("Timestamp must have a timezone and whole-second precision.")
        return parsed.astimezone(timezone.utc).isoformat()


# %% Application factory; startup loads artifacts, never fits a model
def create_app(manifest_path: Path | None = None, predictor: ResearchPredictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_predictor = predictor is None
        if owns_predictor:
            if manifest_path is None:
                raise ValueError("A prepared local manifest is required.")
            app.state.predictor = load_deployment(manifest_path, manifest_path.parent / "predictions.sqlite")
        else:
            app.state.predictor = predictor
            predictor.check_ready()
        try:
            yield
        finally:
            if owns_predictor:
                app.state.predictor.close()

    app = FastAPI(
        title="Financial fraud research prototype",
        version="1.0",
        lifespan=lifespan,
        description="Uncalibrated research scores; no production approval. Frozen training history only.",
    )

    @app.exception_handler(ServingError)
    async def serving_failure(request: Request, exc: ServingError):
        try:
            request.app.state.predictor.event("failure", str(exc))
        except Exception:
            pass  # The client still receives a failure; a logging outage never turns it into a prediction.
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        errors = [{"location": list(error["loc"]), "message": error["msg"]} for error in exc.errors()]
        try:
            request.app.state.predictor.event("validation_failure", "Invalid request; raw payload not logged.")
        except Exception:
            pass
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(request: Request) -> dict[str, Any]:
        service = request.app.state.predictor
        service.check_ready()
        return {
            "status": "ready",
            "model_run_id": service.manifest["model_run_id"],
            "state_cutoff": service.manifest["state_cutoff"],
            "production_approved": False,
        }

    @app.post("/predict")
    def predict(transaction: TransactionRequest, request: Request) -> dict[str, Any]:
        return request.app.state.predictor.predict(transaction.model_dump())

    return app

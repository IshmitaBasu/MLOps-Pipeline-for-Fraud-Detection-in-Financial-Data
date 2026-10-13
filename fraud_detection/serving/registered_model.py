"""Registry packaging for the unchanged evaluated pipeline; no fitting or feature retrieval."""

# %% Portable imports (this file is included in the MLflow model)
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd


# %% Verify original model bytes before loading the trusted local artifact
class FrozenResearchModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.contract = json.loads(Path(context.artifacts["contract"]).read_text(encoding="utf-8"))
        path = Path(context.artifacts["evaluated_pipeline"])
        self.source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.source_sha256 != self.contract["source_model_sha256"]:
            raise ValueError("Registered pipeline checksum does not match its evaluated source.")
        if self.contract["production_approved"] is not False:
            raise ValueError("This model is restricted to research use.")
        self.pipeline = joblib.load(path)
        if list(self.pipeline.feature_names_in_) != self.contract["model_features"]:
            raise ValueError("Registered input features do not match the fitted pipeline.")
        if not np.array_equal(self.pipeline.classes_, [0, 1]):
            raise ValueError("Registered pipeline must have classes [0, 1].")

    # %% Prediction accepts already-computed model features, not raw API transactions
    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Registered model inputs must be a feature DataFrame.")
        if not model_input.columns.is_unique or set(model_input.columns) != set(self.contract["model_features"]):
            raise ValueError("All and only the registered model features are required.")
        scores = self.pipeline.predict_proba(model_input[self.contract["model_features"]])[:, 1]
        return pd.DataFrame(
            {"score": scores, "decision": np.where(scores >= self.contract["threshold"], "flagged", "not_flagged")},
            index=model_input.index,
        )

"""Registry packaging safeguards without accessing canonical model or registry state."""

# %% Imports and synthetic portable model
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd

from fraud_detection.common.io import sha256_file
from fraud_detection.serving.registered_model import FrozenResearchModel


class SyntheticPipeline:
    feature_names_in_ = np.array(["amount"])
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        score = frame["amount"].to_numpy() / 100
        return np.column_stack([1 - score, score])


# %% Exact original bytes and fixed rule
class PackageTests(unittest.TestCase):
    def package(self, directory):
        model = directory / "pipeline.joblib"
        joblib.dump(SyntheticPipeline(), model)
        contract = {
            "source_model_sha256": sha256_file(model),
            "production_approved": False,
            "model_features": ["amount"],
            "threshold": 0.5,
        }
        contract_path = directory / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return SimpleNamespace(artifacts={"evaluated_pipeline": str(model), "contract": str(contract_path)})

    def test_loading_does_not_predict_and_fixed_rule_applies(self):
        with TemporaryDirectory() as temporary:
            context = self.package(Path(temporary))
            model = FrozenResearchModel()
            with patch.object(SyntheticPipeline, "predict_proba", side_effect=AssertionError("No scoring on load")):
                model.load_context(context)
            result = model.predict(None, pd.DataFrame({"amount": [20.0, 50.0, 80.0]}))
            self.assertEqual(result.decision.tolist(), ["not_flagged", "flagged", "flagged"])
            np.testing.assert_allclose(result.score, [0.2, 0.5, 0.8])
            with self.assertRaises(ValueError):
                model.predict(None, pd.DataFrame({"amount": [20.0], "is_fraud": [0]}))

    def test_tampered_bytes_fail_before_deserialisation(self):
        with TemporaryDirectory() as temporary:
            context = self.package(Path(temporary))
            Path(context.artifacts["evaluated_pipeline"]).write_bytes(b"altered")
            with patch("fraud_detection.serving.registered_model.joblib.load") as loader, self.assertRaises(ValueError):
                FrozenResearchModel().load_context(context)
            loader.assert_not_called()

    def test_production_approval_rejected(self):
        with TemporaryDirectory() as temporary:
            context = self.package(Path(temporary))
            path = Path(context.artifacts["contract"])
            contract = json.loads(path.read_text(encoding="utf-8"))
            contract["production_approved"] = True
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaises(ValueError):
                FrozenResearchModel().load_context(context)


# %% Idempotent registry version lookup and no automatic aliases
class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "stages/16_register_research_model.py"
        spec = importlib.util.spec_from_file_location("registration_runner_test", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def entry(self, version="1"):
        return SimpleNamespace(
            version=version,
            tags={
                "evaluated_run_id": self.runner.FINAL_RUN_ID,
                "source_model_sha256": "hash",
                "production_approved": "false",
            },
        )

    def page(self, entries, token=None):
        class Page(list):
            pass

        page = Page(entries)
        page.token = token
        return page

    def test_pagination_and_duplicate_identity_are_checked(self):
        client = MagicMock()
        entry = self.entry()
        client.search_model_versions.side_effect = [self.page([], "next"), self.page([entry])]
        self.assertEqual(self.runner.matching_versions(client, {"source_model_sha256": "hash"}), [entry])
        client.search_model_versions.side_effect = None
        client.search_model_versions.return_value = self.page([entry, self.entry("2")])
        with self.assertRaises(ValueError):
            self.runner.matching_versions(client, {"source_model_sha256": "hash"})

    def test_rerun_verifies_existing_version_without_registry_writes(self):
        client = MagicMock()
        client.search_model_versions.return_value = self.page([self.entry()])
        with (
            patch.object(self.runner, "verify_version", return_value={"version": "1"}) as verify,
            patch.object(self.runner.mlflow, "register_model") as register,
        ):
            result = self.runner.register_model(client, Path("unused"), {"source_model_sha256": "hash"})
        self.assertEqual(result["version"], "1")
        verify.assert_called_once()
        register.assert_not_called()
        client.set_registered_model_alias.assert_not_called()


# %% Direct test entry point
if __name__ == "__main__":
    unittest.main()

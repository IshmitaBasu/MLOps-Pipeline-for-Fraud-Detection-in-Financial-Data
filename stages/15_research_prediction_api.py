"""Prepare, verify or serve the frozen research model; no retraining or test scoring."""

# %% Imports and command-line options
from __future__ import annotations

import argparse
from pathlib import Path

# %% Repository bootstrap for direct script and VS Code cell execution
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fraud_detection.serving.api import create_app
from fraud_detection.serving.service import load_deployment, prepare_deployment
from fraud_detection.paths import REPOSITORY_ROOT

PROJECT_DIR = REPOSITORY_ROOT
DEFAULT_MANIFEST = PROJECT_DIR / "serving_artifacts" / "research_v1" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify", "serve"), default="verify")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


# %% Preparation and read-only readiness verification
def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    if args.mode == "prepare":
        if manifest.name != "manifest.json":
            raise ValueError("Preparation requires an output path ending in manifest.json.")
        path = prepare_deployment(PROJECT_DIR, manifest.parent)
        print(f"Prepared isolated serving bundle: {path}")
        print("No model was trained, registered or promoted; no test rows were scored.")
    elif args.mode == "verify":
        service = load_deployment(manifest, manifest.parent / "predictions.sqlite")
        try:
            service.check_ready()
            print(f"Ready: evaluated research model {service.manifest['model_run_id']}")
            print(f"Online queries must be after {service.manifest['state_cutoff']}.")
            print("Readiness verified without making predictions or evaluating the test set.")
        finally:
            service.close()

    # %% Single-process, loopback-only development server
    else:
        import uvicorn

        uvicorn.run(create_app(manifest_path=manifest), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()

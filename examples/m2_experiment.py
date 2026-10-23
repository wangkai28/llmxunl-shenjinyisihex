import json
import math
import os
from datetime import datetime

from bioelectric.experiment import run_experiment
from bioelectric.llm_provider import GatewayLLMProvider
from bioelectric.spiking import SpikingNetwork


def _sanitize_non_finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(item) for item in value]
    return value


def main():
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "qwen3.8-flash")
    if not base_url or not api_key:
        raise SystemExit("OPENAI_BASE_URL / OPENAI_API_KEY not configured")

    provider = GatewayLLMProvider(base_url, api_key, model, n_bins=8)
    cfg = dict(steps=60, observe_every=10, n_bins=8)

    rw = run_experiment(
        SpikingNetwork(n_units=64, seed=42), provider,
        mode="readwrite", **cfg,
    )
    ro = run_experiment(
        SpikingNetwork(n_units=64, seed=42), provider,
        mode="readonly", **cfg,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "config": cfg,
        "readwrite": {"errors": rw.errors, "convergence": rw.convergence},
        "readonly": {"errors": ro.errors, "convergence": ro.convergence},
    }
    os.makedirs("data", exist_ok=True)
    path = f"data/m2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_non_finite(report), f,
            ensure_ascii=False, indent=2, allow_nan=False,
        )

    print("== M2 behavioral-closure experiment ==")
    print(f"model={model} steps={cfg['steps']} observe_every={cfg['observe_every']}")
    print(f"readwrite: converged={rw.convergence['converged']} slope={rw.convergence['slope']:.4f} errors={[round(e,3) if e is not None else None for e in rw.errors]}")
    print(f"readonly : converged={ro.convergence['converged']} slope={ro.convergence['slope']:.4f} errors={[round(e,3) if e is not None else None for e in ro.errors]}")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()

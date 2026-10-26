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
    cfg = dict(steps=300, observe_every=10, n_bins=8, writeback_gain=2.5)
    net_kwargs = dict(
        n_units=64, seed=42,
        background_rate=0.05, background_strength=0.6,
    )

    def _make_network():
        return SpikingNetwork(**net_kwargs)

    # 方案 A：LLM 主动注入（llm 策略）
    rw_llm = run_experiment(
        _make_network(), provider, mode="readwrite",
        writeback_policy="llm", **cfg,
    )
    # 方案 B：误差驱动写回
    rw_err = run_experiment(
        _make_network(), provider, mode="readwrite",
        writeback_policy="error-driven", error_threshold=0.15, **cfg,
    )
    # 对照组：不写回
    ro = run_experiment(
        _make_network(), provider, mode="readonly", **cfg,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "config": cfg,
        "network": net_kwargs,
        "readwrite_llm": {
            "errors": rw_llm.errors,
            "self_descriptions": rw_llm.self_descriptions,
            "convergence": rw_llm.convergence,
            "actions": rw_llm.actions,
        },
        "readwrite_error_driven": {
            "errors": rw_err.errors,
            "self_descriptions": rw_err.self_descriptions,
            "convergence": rw_err.convergence,
            "actions": rw_err.actions,
        },
        "readonly": {
            "errors": ro.errors,
            "self_descriptions": ro.self_descriptions,
            "convergence": ro.convergence,
            "actions": ro.actions,
        },
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
    print(f"llm       : converged={rw_llm.convergence['converged']} slope={rw_llm.convergence['slope']:.4f}")
    print(f"err-driven: converged={rw_err.convergence['converged']} slope={rw_err.convergence['slope']:.4f}")
    print(f"readonly : converged={ro.convergence['converged']} slope={ro.convergence['slope']:.4f} errors={[round(e,3) if e is not None else None for e in ro.errors]}")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()

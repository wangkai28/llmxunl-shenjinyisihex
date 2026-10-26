import json
import math
import os
from datetime import datetime

from bioelectric.balanced_network import BalancedNetwork
from bioelectric.experiment import run_experiment
from bioelectric.llm_provider import GatewayLLMProvider, load_aigw_config


def _sanitize_non_finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_non_finite(item) for item in value]
    return value


def main():
    config_toml = os.environ.get(
        "CODEX_CONFIG_TOML", os.path.expanduser("~/.codex/config.toml")
    )
    aigw = load_aigw_config(config_toml)
    model = os.environ.get("AIGW_MODEL", "deepseek-v4-flash")

    provider = GatewayLLMProvider(aigw["base_url"], aigw["token"], model, n_bins=8)
    cfg = dict(steps=200, observe_every=10, n_bins=8, writeback_gain=2.0)
    net_kwargs = dict(
        n_units=1000, k_exc=10, j_inh=-4.0,
        external_mean=0.2, external_noise=0.1, delay_max=4,
        seed=42,
    )

    def _make_network():
        return BalancedNetwork(**net_kwargs)

    rw = run_experiment(
        _make_network(), provider, mode="readwrite",
        writeback_policy="llm", **cfg,
    )
    ro = run_experiment(
        _make_network(), provider, mode="readonly", **cfg,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "network": "balanced",
        "config": cfg,
        "net_kwargs": net_kwargs,
        "readwrite_llm": {
            "errors": rw.errors,
            "self_descriptions": rw.self_descriptions,
            "convergence": rw.convergence,
            "identity_continuities": rw.identity_continuities,
            "narrative_coherence": rw.narrative_coherence,
            "actions": rw.actions,
        },
        "readonly": {
            "errors": ro.errors,
            "self_descriptions": ro.self_descriptions,
            "convergence": ro.convergence,
            "identity_continuities": ro.identity_continuities,
            "narrative_coherence": ro.narrative_coherence,
            "actions": ro.actions,
        },
    }
    os.makedirs("data", exist_ok=True)
    path = f"data/m2_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_non_finite(report), f,
            ensure_ascii=False, indent=2, allow_nan=False,
        )

    print("== M2 balanced-network experiment ==")
    print(f"model={model} network=balanced n_units=1000")
    print(f"readwrite: coherent={rw.narrative_coherence.get('coherent')} mean={rw.narrative_coherence.get('mean'):.3f} std={rw.narrative_coherence.get('std'):.3f}")
    print(f"readonly : coherent={ro.narrative_coherence.get('coherent')} mean={ro.narrative_coherence.get('mean'):.3f} std={ro.narrative_coherence.get('std'):.3f}")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()

import json
import math
import os
from datetime import datetime

from bioelectric.experiment import run_experiment
from bioelectric.llm_provider import GatewayLLMProvider, load_aigw_config
from bioelectric.metrics import paired_sign_test
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
    # 优先使用 Codex 的 AIGW 网关（公司网关，token 不落地）。
    # 可通过环境变量 AIGW_MODEL 覆盖模型名。
    config_toml = os.environ.get(
        "CODEX_CONFIG_TOML",
        os.path.expanduser("~/.codex/config.toml"),
    )
    aigw = load_aigw_config(config_toml)
    model = os.environ.get("AIGW_MODEL", "deepseek-v4-flash")
    base_url = aigw["base_url"]
    api_key = aigw["token"]

    n_rounds = int(os.environ.get("N_ROUNDS", "3"))
    steps = int(os.environ.get("STEPS", "150"))
    seed_start = int(os.environ.get("SEED_START", "100"))
    observe_every = 10
    n_bins = 8

    provider = GatewayLLMProvider(base_url, api_key, model, n_bins=n_bins)

    a_means = []
    b_means = []
    per_round = []

    for r in range(n_rounds):
        seed = seed_start + r
        net_kwargs = dict(
            n_units=64, seed=seed,
            background_rate=0.05, background_strength=0.6,
        )
        a = run_experiment(
            SpikingNetwork(**net_kwargs), provider,
            mode="readwrite", writeback_policy="llm", writeback_gain=2.5,
            steps=steps, observe_every=observe_every, n_bins=n_bins,
        )
        b = run_experiment(
            SpikingNetwork(**net_kwargs), provider,
            mode="readonly",
            steps=steps, observe_every=observe_every, n_bins=n_bins,
        )
        a_m = a.narrative_coherence.get("mean")
        b_m = b.narrative_coherence.get("mean")
        a_means.append(a_m)
        b_means.append(b_m)
        per_round.append({
            "seed": seed,
            "a_identity_mean": a_m,
            "b_identity_mean": b_m,
            "a_coherent": a.narrative_coherence.get("coherent"),
            "b_coherent": b.narrative_coherence.get("coherent"),
        })
        print(f"[round {r+1}/{n_rounds}] seed={seed}  A={a_m:.3f}  control={b_m:.3f}")

    test = paired_sign_test(a_means, b_means)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "n_rounds": n_rounds,
        "steps": steps,
        "observe_every": observe_every,
        "per_round": per_round,
        "paired_sign_test": _sanitize_non_finite(test),
    }
    os.makedirs("data", exist_ok=True)
    path = f"data/replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_non_finite(report), f, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n== replication result ==")
    print(f"A mean identity: {[round(m, 3) if m is not None else None for m in a_means]}")
    print(f"control mean  : {[round(m, 3) if m is not None else None for m in b_means]}")
    print(f"mean_diff={test['mean_diff']:.4f}  wins={test['wins']}/{test['n']}  p={test['p_value']:.4f}  significant={test['significant']}")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()

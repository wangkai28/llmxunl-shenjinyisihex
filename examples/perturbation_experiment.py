import csv
import os
from datetime import datetime

from bioelectric.perturbation import factor_effects, perturbation_sweep


DIMENSION_LABELS = {
    "core_activity_rate": "核心活动率（自我强度）",
    "periphery_activity_rate": "外围活动率",
    "core_focus": "核心聚焦度（自我边界）",
    "sync_index": "同步指数（异步稳定性）",
}

FACTOR_LABELS = {
    "noise_scale": "噪声增强",
    "prune_fraction": "突触剪除",
    "lesion_fraction": "单元失活",
}


def main():
    n_units = int(os.environ.get("N_UNITS", "1000"))
    core_size = int(os.environ.get("CORE_SIZE", "80"))
    seed = int(os.environ.get("SEED", "0"))

    dataset = perturbation_sweep(
        n_units=n_units, core_size=core_size,
        noise_levels=[1.0, 2.0, 4.0],
        prune_fractions=[0.0, 0.2, 0.4],
        lesion_fractions=[0.0, 0.2, 0.4],
        seed=seed, measure_steps=100,
    )

    os.makedirs("data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/perturbation_{ts}.csv"
    fieldnames = list(dataset[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    effects = factor_effects(dataset)

    print("== 干扰测试结果 ==")
    print(f"组合数={len(dataset)}  CSV -> {csv_path}")
    print()
    print("各环境因素对认知维度的边际效应（正值=增强，负值=削弱）：")
    print()
    for factor, dims in effects.items():
        print(f"  {FACTOR_LABELS.get(factor, factor)}：")
        for dim, eff in dims.items():
            print(f"    {DIMENSION_LABELS.get(dim, dim)}: {eff:+.4f}")
        print()


if __name__ == "__main__":
    main()

import json
import os
import shutil
import tempfile

from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


SELF_LABELS = {"id": "本我", "superego": "超我", "tao": "道家"}


def main():
    state_dir = tempfile.mkdtemp(prefix="identity_demo_")
    core = IdentityCore(persona="tao", state_dir=state_dir, seed=0)

    print("== 初始状态 ==")
    print(f"  性格参数: {core._persona_cfg}")
    print()

    # 一连串情境事件
    timeline = [
        Event(type="calm", intensity=0.3, content="安静地待着"),
        Event(type="danger", intensity=0.9, content="突然遇到危险"),
        Event(type="danger", intensity=0.9, content="危险还在持续"),
        Event(type="danger", intensity=0.9, content="危险加剧"),
        Event(type="calm", intensity=0.4, content="危险过去，平静下来"),
        Event(type="moral", intensity=0.8, content="面对道德困境"),
        Event(type="moral", intensity=0.8, content="道德抉择"),
    ]

    print("== 事件流 → 主导自我切换 ==")
    print(f"{'事件':<24} {'强度':>4} {'主导自我':>8}")
    for ev in timeline:
        state = core.process_event(ev)
        print(f"{ev.content:<24} {ev.intensity:>4} {SELF_LABELS[state['dominant']]:>8}")

    print()
    print("== 性格演化（经历重大事件后）==")
    print(f"  初始   : {core._persona_cfg}")

    # 再对比：一个"一直平静"的核心 vs 一个"一直危险"的核心
    print()
    print("== 对比：不同经历塑造不同性格 ==")
    calm_core = IdentityCore(persona="tao", state_dir=tempfile.mkdtemp(), seed=0)
    danger_core = IdentityCore(persona="tao", state_dir=tempfile.mkdtemp(), seed=0)
    for _ in range(10):
        calm_core.process_event(Event(type="calm", intensity=0.9))
        danger_core.process_event(Event(type="danger", intensity=0.9))
    print(f"  平静经历 10 次后: {calm_core._persona_cfg}")
    print(f"  危险经历 10 次后: {danger_core._persona_cfg}")

    # 持久化验证
    print()
    print("== 持久化：重启后性格是否保持 ==")
    core.save()
    core2 = IdentityCore(persona="tao", state_dir=state_dir, seed=0)
    core2.load()
    print(f"  保存前 j_inh: {core._persona_cfg['j_inh']:.4f}")
    print(f"  加载后 j_inh: {core2._persona_cfg['j_inh']:.4f}")
    print(f"  网络 j_inh 生效: {core2._network.j_inh:.4f}")

    shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

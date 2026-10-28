from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


def _core(tmp_path):
    return IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)


def test_reflect_and_integrate_records_reflection(tmp_path):
    core = _core(tmp_path)
    before = len(core.history())
    core.reflect_and_integrate()
    after = len(core.history())
    # 反思应被记为一条经历（自传）
    assert after > before


def test_reflect_and_integrate_returns_text(tmp_path):
    core = _core(tmp_path)
    text = core.reflect_and_integrate()
    assert isinstance(text, str)
    assert "自我审视" in text


def test_reflection_reduces_dominant_imbalance(tmp_path):
    core = _core(tmp_path)
    # 制造一个强烈的主导（危险）
    core.process_event(Event(type="danger", intensity=1.0))
    before = core.query()["id"]
    core.reflect_and_integrate()
    after = core.query()["id"]
    # 反思后，主导自我的活动应有所回落（自我平衡）
    assert after < before


def test_reflection_triggers_slow_evolution(tmp_path):
    core = _core(tmp_path)
    n_before = core._ocean["N"]
    # 多次反思应触发极缓慢的 OCEAN 演化
    for _ in range(5):
        core.reflect_and_integrate()
    assert core._ocean["N"] != n_before

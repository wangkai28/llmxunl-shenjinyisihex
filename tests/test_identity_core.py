from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


def test_save_persists_activities(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    core.save()
    import json, os
    snap = json.loads(open(os.path.join(str(tmp_path / "id"), "snapshot.json"), encoding="utf-8").read())
    assert "activities" in snap
    assert "tao" in snap["activities"]


def test_history_returns_entries(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="danger", intensity=1.0))
    core.process_event(Event(type="calm", intensity=1.0))
    hist = core.history()
    assert len(hist) == 2
    assert hist[0]["event"] == "danger"


def test_significant_event_marked(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0,
                        significant_threshold=0.8)
    core.process_event(Event(type="danger", intensity=0.9))
    core.process_event(Event(type="calm", intensity=0.3))
    hist = core.history()
    assert hist[0]["significant"] is True
    assert hist[1]["significant"] is False


def test_persona_param_changes_network(tmp_path):
    # 不同 persona 应产生不同的网络性格参数
    id_core = IdentityCore(persona="id", state_dir=str(tmp_path / "a"), seed=0)
    super_core = IdentityCore(persona="superego", state_dir=str(tmp_path / "b"), seed=0)
    # 本我应比超我有更强的外部驱动（冲动）
    assert id_core._network.external_mean > super_core._network.external_mean


def test_process_event_changes_dominant_self(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    # 危险事件 → 本我主导
    state = core.process_event(Event(type="danger", intensity=1.0))
    assert state["dominant"] == "id"
    # 平静事件 → 道家主导
    state = core.process_event(Event(type="calm", intensity=1.0))
    assert state["dominant"] == "tao"


def test_save_and_load_preserves_identity(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="moral", intensity=1.0))
    core.save()
    # 新建一个核心，加载后应恢复主导自我
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2.query()["dominant"] == "superego"


def test_query_returns_activity(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    state = core.query()
    assert "dominant" in state
    assert "id" in state and "superego" in state and "tao" in state


def test_query_dominant_agrees_with_activity(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    state = core.query()
    names = ["id", "superego", "tao"]
    highest = max(names, key=lambda n: state[n])
    assert state["dominant"] == highest


def test_query_is_idempotent(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    first = core.query()
    second = core.query()
    assert first["dominant"] == second["dominant"]


def test_save_load_roundtrip_dominant(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="moral", intensity=1.0))
    core.save()
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2.query()["dominant"] == "superego"


def test_significant_event_evolves_persona(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    before = core._persona_cfg["external_mean"]
    # 多次危险重大事件 → 本我倾向更强（external_mean 上升）
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    assert core._persona_cfg["external_mean"] > before


def test_persona_evolution_persists(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    core.save()
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2._persona_cfg["external_mean"] == core._persona_cfg["external_mean"]


def test_evolution_applies_to_live_network(tmp_path):
    # 演化后的参数应真正作用到网络（j_inh 通过 set_j_inh 生效）
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    before_j_inh = core._network.j_inh
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    # danger 演化使 j_inh 弱化（朝 0 靠近），网络 j_inh 应随之变化
    assert core._network.j_inh != before_j_inh
    assert core._network.j_inh == core._persona_cfg["j_inh"]


def test_load_reapplies_persona_to_network(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    for _ in range(5):
        core.process_event(Event(type="danger", intensity=1.0))
    core.save()
    core2 = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core2.load()
    assert core2._network.j_inh == core2._persona_cfg["j_inh"]


def test_ocean_derives_initial_persona_params(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    assert hasattr(core, "_ocean")
    assert set(core._ocean.keys()) == {"O", "E", "C", "A", "N"}


def test_query_includes_ocean(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    core.process_event(Event(type="calm", intensity=1.0))
    state = core.query()
    assert "ocean" in state
    assert "O" in state["ocean"]


def test_ocean_evolves_slowly(tmp_path):
    core = IdentityCore(persona="tao", state_dir=str(tmp_path / "id"), seed=0)
    before_n = core._ocean["N"]
    for _ in range(20):
        core.process_event(Event(type="danger", intensity=1.0))
    assert core._ocean["N"] > before_n  # 神经质缓慢上升


def test_persona_ocean_preset_mapping(tmp_path):
    superego = IdentityCore(persona="superego", state_dir=str(tmp_path / "a"), seed=0)
    id_core = IdentityCore(persona="id", state_dir=str(tmp_path / "b"), seed=0)
    assert superego._ocean["C"] > id_core._ocean["C"]  # 超我比本我更尽责

from bioelectric.event import Event
from bioelectric.identity_core import IdentityCore


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

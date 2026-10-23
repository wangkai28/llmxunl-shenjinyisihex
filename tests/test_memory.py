from bioelectric.memory import ReflectionMemory


def test_add_and_recent():
    m = ReflectionMemory(capacity=5)
    m.add("one")
    m.add("two")
    assert m.recent(10) == ["one", "two"]


def test_capacity_evicts_oldest():
    m = ReflectionMemory(capacity=2)
    m.add("a")
    m.add("b")
    m.add("c")
    assert m.recent(10) == ["b", "c"]


def test_save_and_load(tmp_path):
    m = ReflectionMemory()
    m.add("x")
    path = tmp_path / "mem.json"
    m.save(str(path))
    m2 = ReflectionMemory.load(str(path))
    assert m2.recent(10) == ["x"]

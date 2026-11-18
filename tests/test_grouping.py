import numpy as np
import pytest

import photo_cleanup as pc


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def item(uuid, ts, emb=(1, 0), **raw):
    full = {f: None for f in pc.FEATURES}
    full.update(raw)
    return pc.Item(uuid=uuid, ts=ts, path=f"/x/{uuid}", filename=uuid, emb=unit(*emb), raw=full)


def uuids(groups):
    return sorted(sorted(i.uuid for i in g) for g in groups)


# ---------------------------------------------------------------- group_items

def test_empty_and_single():
    assert pc.group_items([], 10, 0.9) == []
    assert pc.group_items([item("a", 0)], 10, 0.9) == []


def test_similar_close_in_time_are_grouped():
    items = [item("a", 0), item("b", 5)]
    assert uuids(pc.group_items(items, 10, 0.9)) == [["a", "b"]]


def test_time_gap_splits_windows():
    items = [item("a", 0), item("b", 5), item("c", 100), item("d", 103)]
    assert uuids(pc.group_items(items, 10, 0.9)) == [["a", "b"], ["c", "d"]]


def test_time_gap_boundary_is_inclusive():
    assert len(pc.group_items([item("a", 0), item("b", 10)], 10, 0.9)) == 1
    assert pc.group_items([item("a", 0), item("b", 10.01)], 10, 0.9) == []


def test_dissimilar_photos_not_grouped():
    items = [item("a", 0, (1, 0)), item("b", 1, (0, 1))]
    assert pc.group_items(items, 10, 0.9) == []


def test_similarity_threshold_boundary():
    # cos(45°) ≈ 0.7071
    items = [item("a", 0, (1, 0)), item("b", 1, (1, 1))]
    assert len(pc.group_items(items, 10, 0.70)) == 1
    assert pc.group_items(items, 10, 0.72) == []


def test_transitive_linking():
    # a~b and b~c but a !~ c: union-find should still join all three
    items = [item("a", 0, (1, 0)), item("b", 1, (1, 1)), item("c", 2, (0, 1))]
    assert uuids(pc.group_items(items, 10, 0.70)) == [["a", "b", "c"]]


def test_similar_pairs_separated_within_one_window():
    items = [item("a", 0, (1, 0)), item("b", 1, (0, 1)), item("c", 2, (1, 0)), item("d", 3, (0, 1))]
    assert uuids(pc.group_items(items, 10, 0.9)) == [["a", "c"], ["b", "d"]]


def test_max_window_caps_window_size():
    items = [item(f"p{i}", i) for i in range(6)]
    groups = pc.group_items(items, 100, 0.9, max_window=3)
    assert sorted(len(g) for g in groups) == [3, 3]


# ---------------------------------------------------------------- rank_group

W = dict(pc.DEFAULT_WEIGHTS)


def test_best_feature_scores_highest():
    lo = item("lo", 0, topiq=0.2, laion=4.0)
    hi = item("hi", 1, topiq=0.8, laion=6.0)
    assert [i.uuid for i in pc.rank_group([lo, hi], W)] == ["hi", "lo"]


def test_identical_features_normalize_to_half():
    a, b = item("a", 0, topiq=0.5), item("b", 1, topiq=0.5)
    pc.rank_group([a, b], W)
    assert a.norm["topiq"] == b.norm["topiq"] == 0.5
    assert a.score == pytest.approx(b.score)


def test_missing_feature_uses_median_and_is_not_penalized_unfairly():
    a = item("a", 0, topiq=0.2)
    b = item("b", 1, topiq=0.6)
    c = item("c", 2, topiq=None)  # imputed with median (0.4) -> norm 0.5
    pc.rank_group([a, b, c], W)
    assert c.norm["topiq"] == pytest.approx(0.5)


def test_feature_missing_for_whole_group_is_ignored():
    a = item("a", 0, topiq=0.2)
    b = item("b", 1, topiq=0.9)
    pc.rank_group([a, b], W)
    # only topiq contributes, so score is just its normalized value
    assert (a.score, b.score) == (pytest.approx(0.0), pytest.approx(1.0))


def test_zero_weight_feature_ignored():
    a = item("a", 0, topiq=0.1, laion=9)
    b = item("b", 1, topiq=0.9, laion=1)
    ranked = pc.rank_group([a, b], {**W, "laion": 0})
    assert ranked[0].uuid == "b"


def test_blink_penalty_demotes_closed_eyes():
    # identical quality, so only the blink penalty separates them
    open_ = item("open", 0, topiq=0.5, eyes=1.0)
    blink = item("blink", 1, topiq=0.5, eyes=0.1)
    ranked = pc.rank_group([blink, open_], {**W, "eyes": 0.0})  # eyes weight 0: penalty still applies
    assert [i.uuid for i in ranked] == ["open", "blink"]
    assert open_.score - blink.score == pytest.approx(pc.BLINK_PENALTY)


def test_all_weights_zero_does_not_divide_by_zero():
    a, b = item("a", 0, topiq=0.1), item("b", 1, topiq=0.9)
    pc.rank_group([a, b], {f: 0 for f in pc.FEATURES})
    assert a.score == b.score == 0


# ---------------------------------------------------------------- helpers

def test_eye_ratio():
    assert pc._eye_ratio([(0, 0), (1, 0), (1, 0.3), (0, 0.3)]) == pytest.approx(0.3)
    assert pc._eye_ratio([(0, 0), (1, 1)]) is None          # too few points
    assert pc._eye_ratio([(0, 0), (0, 1), (0, 2), (0, 3)]) is None  # zero width


def test_batched():
    assert list(pc.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_load_weights_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "WEIGHTS_PATH", tmp_path / "w.json")
    assert pc.load_weights() == pc.DEFAULT_WEIGHTS
    (tmp_path / "w.json").write_text('{"topiq": 0.9, "bogus": 1}')
    w = pc.load_weights()
    assert w["topiq"] == 0.9 and w["laion"] == pc.DEFAULT_WEIGHTS["laion"] and "bogus" not in w


def test_load_items_filters_and_sorts(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "HOME", tmp_path)
    monkeypatch.setattr(pc, "DB_PATH", tmp_path / "c.sqlite")
    con = pc.open_db()
    emb = np.array([1, 0], dtype=np.float32).tobytes()
    rows = [("b", 20.0, emb), ("a", 10.0, emb), ("c", 5.0, emb), ("noemb", 1.0, None)]
    for u, ts, e in rows:
        con.execute("INSERT INTO photos(uuid, ts, path, filename, emb, topiq, is_utility) VALUES(?,?,?,?,?,?,?)",
                    (u, ts, f"/p/{u}", None, e, 0.5, 0))
    items = pc.load_items(con, {"a", "b", "noemb"})
    assert [i.uuid for i in items] == ["a", "b"]  # filtered by uuids, emb NOT NULL, sorted by ts
    assert items[0].raw["topiq"] == 0.5 and items[0].filename == ""

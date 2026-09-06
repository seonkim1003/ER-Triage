import numpy as np
import pytest

from ertriage.evaluate import holm, subgroup_contrasts


def test_holm_matches_the_step_down_definition_and_stays_monotone():
    np.testing.assert_allclose(holm([.01, .02, .03, .04]), [.04, .06, .06, .06])
    np.testing.assert_allclose(holm([.001, .5]), [.002, .5])
    np.testing.assert_allclose(holm([.04, .01]), [.04, .02])  # order preserved, not sorted
    assert holm([.9, .9, .9]).max() == 1.  # clipped at one
    single = holm([.02])
    assert single == [.02] and holm([0.])[0] == 0.


@pytest.mark.parametrize("bad", [[], [.5, np.nan], [-.1], [1.5]])
def test_holm_rejects_values_that_are_not_p_values(bad):
    with pytest.raises(ValueError, match="Holm"):
        holm(bad)


def cohort(seed=0, patients=240, hours=25, weak_level=None):
    """Patients in two recorded levels; `weak_level` gets scores unrelated to its labels."""
    rng = np.random.default_rng(seed)
    y, p, ids, levels = [], [], [], {}
    for i in range(patients):
        pid, level = f"p{i:04d}", "a" if i % 2 else "b"
        levels[pid] = level
        labels = rng.integers(0, 2, hours)
        signal = rng.normal(size=hours) if level == weak_level else labels + rng.normal(0, .45, hours)
        y.append(labels)
        p.append(1 / (1 + np.exp(-signal)))
        ids.append(np.repeat(pid, hours))
    return (np.concatenate(y), np.concatenate(p), np.concatenate(ids),
            {"unit": levels, "site": {k: "only" for k in levels}})


def test_contrasts_find_a_planted_difference():
    y, p, ids, families = cohort(seed=1, weak_level="a")
    result = subgroup_contrasts(y, p, ids, families, seed=42, draws=300)
    assert result["correction"] == "holm_bonferroni"
    by_level = {c["level"]: c for c in result["contrasts"]}
    # 'unit' is binary so it asks one question; the single-level 'site' is dropped entirely.
    assert set(by_level) == {"a"} and result["family_size"] == 1
    weak = by_level["a"]
    assert weak["auroc_difference"] < 0 and weak["significant_at_05_after_holm"]
    assert weak["low"] < weak["auroc_difference"] < weak["high"] and weak["high"] < 0
    assert weak["p_value"] <= weak["p_value_holm"]


def test_a_binary_descriptor_is_one_contrast_but_three_levels_are_three():
    """Level-versus-rest on a binary descriptor is one comparison, not two mirrored ones.

    Counting both halves would double that descriptor's share of the family and
    make Holm more conservative than the question warrants.
    """
    y, p, ids, families = cohort(seed=2)
    assert subgroup_contrasts(y, p, ids, {"unit": families["unit"]}, draws=100)["family_size"] == 1
    thirds = {pid: "xyz"[i % 3] for i, pid in enumerate(sorted(set(ids)))}
    wider = subgroup_contrasts(y, p, ids, {"band": thirds}, draws=100)
    assert wider["family_size"] == 3
    assert {c["level"] for c in wider["contrasts"]} == {"x", "y", "z"}


def test_exchangeable_levels_do_not_survive_correction():
    y, p, ids, families = cohort(seed=7, weak_level=None)
    result = subgroup_contrasts(y, p, ids, families, seed=42, draws=300)
    assert not any(c["significant_at_05_after_holm"] for c in result["contrasts"])
    for contrast in result["contrasts"]:
        assert contrast["low"] < 0 < contrast["high"]  # intervals straddle zero
        assert contrast["p_value_holm"] >= contrast["p_value"]


def test_p_values_are_floored_at_one_draw_rather_than_reported_as_zero():
    y, p, ids, families = cohort(seed=3, weak_level="a")
    result = subgroup_contrasts(y, p, ids, families, seed=42, draws=200)
    assert all(c["p_value"] >= 1 / c["draws"] for c in result["contrasts"])


def test_a_degenerate_family_reports_nothing_rather_than_failing():
    y, p, ids, _ = cohort(seed=5)
    assert subgroup_contrasts(y, p, ids, {}, draws=10) == []
    only = {"site": {k: "only" for k in np.unique(ids)}}
    assert subgroup_contrasts(y, p, ids, only, draws=10) == []

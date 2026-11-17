import pytest

from peporacle.results import paired, verdict


@pytest.mark.parametrize(
    ("diffs", "expected"),
    [
        ([0.001, -0.001, 0.0], "equivalent"),
        ([0.03, 0.031, 0.029], "A higher"),
        ([-0.03, -0.031, -0.029], "B higher"),
        ([0.012, 0.016, 0.008], "inconclusive"),  # mean beyond the margin, interval is not
        ([0.05, -0.05, 0.0], "inconclusive"),
    ],
)
def test_verdict(diffs, expected):
    assert verdict(diffs, "A", "B")["verdict"] == expected


def runs(model, offset):
    return [
        {"model": model, "dataset": d, "seed": s, "test_r2": 0.9 + offset + 0.001 * s}
        for d in ("x", "y")
        for s in range(3)
    ]


def test_paired_against_one_reference_and_by_operator():
    reference = runs("atomic attention", 0.0)
    a = runs("GCNConv", 0.05) + runs("NNConv", 0.0)
    rows = paired(a, reference, a_name="op", b_name="ref")
    assert [(r["operator"], r["verdict"]) for r in rows] == [
        ("GCNConv", "op higher"),
        ("GCNConv", "op higher"),
        ("NNConv", "equivalent"),
        ("NNConv", "equivalent"),
    ]
    with pytest.raises(ValueError, match="several models"):
        paired(reference, a, a_name="ref", b_name="op")
    by_name = paired(
        a, runs("GCNConv", 0.0) + runs("NNConv", 0.05), a_name="op", b_name="alone", same_model=True
    )
    assert [r["verdict"] for r in by_name] == ["op higher"] * 2 + ["alone higher"] * 2


def test_partial_runs_are_not_assessed():
    a = [r for r in runs("GCNConv", 0.05) if r["seed"] != 2]
    rows = paired(a, runs("atomic attention", 0.0), a_name="op", b_name="ref")
    assert {r["verdict"] for r in rows} == {"not assessed (2 of 3 seeds)"}
    assert rows[0]["diff_seed2"] is None and rows[0]["diff_mean"] is None

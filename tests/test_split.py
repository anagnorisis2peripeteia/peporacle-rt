import numpy as np

from peporacle.data import LabelScaler, split_indices


def test_partitions_are_disjoint_and_complete():
    seqs = [f"PEPTIDE{i}" for i in range(1000)]
    s = split_indices(seqs, seed=0)
    parts = [set(s.train), set(s.val), set(s.test)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set.union(*parts) == set(range(1000))
    assert (len(s.train), len(s.val), len(s.test)) == (810, 90, 100)


def test_repeated_sequence_never_crosses_a_split():
    seqs = [f"P{i % 300}" for i in range(900)]
    s = split_indices(seqs, seed=3)
    side = {}
    for name, idx in (("train", s.train), ("val", s.val), ("test", s.test)):
        for i in idx:
            assert side.setdefault(seqs[i], name) == name


def test_seed_fixes_the_split():
    seqs = [f"P{i}" for i in range(500)]
    a, b, c = split_indices(seqs, 1), split_indices(seqs, 1), split_indices(seqs, 2)
    assert np.array_equal(a.test, b.test) and not np.array_equal(a.test, c.test)


def test_label_scaler_round_trip():
    y = np.array([3.0, 10.5, 42.0, 7.25])
    scaler = LabelScaler.fit(y)
    assert np.allclose(scaler.inverse(scaler.transform(y)), y, atol=1e-5)

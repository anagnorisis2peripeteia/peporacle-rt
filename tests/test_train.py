from dataclasses import replace

import pytest
import torch

import peporacle.train as train_mod
from peporacle.cache import GraphCache, write_graphs
from peporacle.config import ModelConfig, TrainConfig
from peporacle.data import make_loaders, split_indices
from peporacle.model import AtomicAttention

MODEL = ModelConfig(hidden_dim=32, n_blocks=2, n_heads=2, dropout=0.1)
TRAIN = TrainConfig(
    lr=1e-3, weight_decay=1e-4, batch_size=8, grad_accum=2, grad_clip=5.0, warmup_epochs=1, epochs=3
)


@pytest.fixture
def cache(tmp_path, graphs):
    many = graphs * 8
    labels = [float(g.x[:, 0].mean()) for g in many]
    write_graphs(tmp_path / "c", many, labels, [f"S{i}" for i in range(len(many))])
    return GraphCache(tmp_path / "c")


def run(cache, checkpoint=None, cfg=TRAIN):
    split = split_indices(cache.sequences, seed=0)
    tr, va, _, scaler = make_loaders(cache, split, TRAIN.batch_size)
    torch.manual_seed(0)
    model = AtomicAttention(cache.node_dim, MODEL)
    result = train_mod.fit(
        model,
        train_loader=tr,
        val_loader=va,
        val_labels=cache.labels[split.val],
        scaler=scaler,
        cfg=cfg,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
    )
    return model, result


def test_learns(cache):
    _, result = run(cache, cfg=replace(TRAIN, epochs=15))
    assert result.best_val_r2 > 0.5


def test_resumed_run_matches_uninterrupted_run(cache, tmp_path, monkeypatch):
    reference, _ = run(cache)

    real_epoch, calls = train_mod.train_epoch, []

    def crash_on_second_epoch(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_epoch(*args)

    ckpt = tmp_path / "run.pt"
    monkeypatch.setattr(train_mod, "train_epoch", crash_on_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        run(cache, ckpt)
    monkeypatch.setattr(train_mod, "train_epoch", real_epoch)
    resumed, result = run(cache, ckpt)

    assert result.resumed_from_epoch == 1
    for (k, a), b in zip(
        reference.state_dict().items(), resumed.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b), k

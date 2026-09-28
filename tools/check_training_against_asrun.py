"""Train with the as-run loop and the release loop from the same state; compare bit for bit.

    python tools/check_training_against_asrun.py AS_RUN_DIR CACHE_DIR [EPOCHS]

Both runs start from the same weights and random state on the CPU and see the same
batches, so identical code paths must give identical weights and validation scores.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from peporacle.cache import GraphCache
from peporacle.config import load_frozen
from peporacle.data import make_loaders, split_indices
from peporacle.model import AtomicAttention
from peporacle.train import predict, regression_metrics, train_epoch, warmup_cosine

asrun_dir, cache_dir = Path(sys.argv[1]), Path(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # noqa: PLR2004
sys.path.insert(0, str(asrun_dir))
from pep_core.dataset import _attach_labels  # noqa: E402
from pep_core.models import GATWithGTModel  # noqa: E402
from pep_core.train import _cosine_warmup_scheduler, _eval_epoch, _train_epoch  # noqa: E402

torch.set_num_threads(1)  # single-threaded kernels, so both runs reduce in the same order
m_cfg, t_cfg = load_frozen(Path(__file__).resolve().parents[1] / "configs" / "frozen.toml")
cache = GraphCache(cache_dir)
split = split_indices(cache.sequences, seed=0)
cpu = torch.device("cpu")

# as-run: pep_core.dataset.random_split builds loaders over a list of labelled graphs
graphs = [cache[i] for i in range(len(cache))]
labels = cache.labels
train_y = labels[split.train]
_attach_labels(graphs, list(labels), "regression", train_y.mean(), train_y.std())
old_train = DataLoader([graphs[i] for i in split.train], batch_size=t_cfg.batch_size, shuffle=True)
old_val = DataLoader([graphs[i] for i in split.val], batch_size=t_cfg.batch_size)
new_train, new_val, _, scaler = make_loaders(cache, split, t_cfg.batch_size)

torch.manual_seed(0)
old = GATWithGTModel(
    cache.node_dim,
    12,
    hidden_dim=m_cfg.hidden_dim,
    n_local=0,
    n_glob=m_cfg.n_blocks,
    n_heads=m_cfg.n_heads,
    dropout=m_cfg.dropout,
)
rng = torch.get_rng_state()
new = AtomicAttention(cache.node_dim, m_cfg)
new.load_state_dict(
    {
        k.replace("global_layers.", "blocks.").replace("head.mlp.", "head."): v
        for k, v in old.state_dict().items()
    }
)

cfg_old = SimpleNamespace(task="regression", grad_accum=t_cfg.grad_accum, grad_clip=t_cfg.grad_clip)
runs = {}
arms = (("as-run", old, old_train, old_val), ("release", new, new_train, new_val))
for name, model, loader, val in arms:
    torch.set_rng_state(rng)
    opt = AdamW(model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    if name == "as-run":
        sched = _cosine_warmup_scheduler(opt, t_cfg.warmup_epochs, t_cfg.epochs)
    else:
        sched = warmup_cosine(opt, t_cfg.warmup_epochs, t_cfg.epochs)
    scores = []
    for _ in range(epochs):
        if name == "as-run":
            loss = _train_epoch(model, loader, opt, torch.nn.MSELoss(), cfg_old, cpu)
            sched.step()
            scores.append((loss, _eval_epoch(model, val, "regression", cpu)["r2"]))
        else:
            loss = train_epoch(model, loader, opt, t_cfg, cpu)
            sched.step()
            y = labels[split.val]
            scores.append((loss, regression_metrics(y, predict(model, val, scaler, cpu))["r2"]))
    runs[name] = (model, scores)

(old_m, old_s), (new_m, new_s) = runs["as-run"], runs["release"]
renamed = {
    k.replace("global_layers.", "blocks.").replace("head.mlp.", "head."): v
    for k, v in old_m.state_dict().items()
}
new_sd = new_m.state_dict()
assert renamed.keys() == new_sd.keys()
differ = [k for k in new_sd if not torch.equal(renamed[k], new_sd[k])]
same = not differ
if differ:
    print(
        "differing tensors:",
        differ[:5],
        "max abs diff",
        max(float((renamed[k] - new_sd[k]).abs().max()) for k in differ),
    )
for e, ((lo, ro), (ln, rn)) in enumerate(zip(old_s, new_s, strict=True)):
    print(f"epoch {e}: loss as-run {lo:.10f} release {ln:.10f} | val r2 {ro:.6f} vs {rn:.6f}")
print(f"weights after {epochs} epochs bit-identical: {same}")

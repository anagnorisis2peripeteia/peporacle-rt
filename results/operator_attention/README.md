# Operator + attention results

The five operators from [`operator_ablation/`](../operator_ablation/), each followed by atomic
attention, on the same nine datasets and seeds 0-2: 135 runs on one NVIDIA RTX 5070.

## Models

Each model is the operator model from `operator_ablation/` with 5 atomic-attention blocks
inserted between the last operator layer and the pooling: the atom embedding (width 128),
15 layers of one operator, 5 attention blocks over all atoms of the peptide, then the same
mean and max pooling and prediction head. The attention blocks are those of atomic attention
(4 heads), which itself has 15 attention blocks and no operator layers.

| operator | parameters, operator + attention |
|---|---:|
| `TransformerConv` | 2.04 M |
| `GCNConv` | 1.27 M |
| `SAGEConv` | 1.52 M |
| `ARMAConv` | 11.39 M |
| `NNConv` | 4.47 M |

## Protocol

As for `operator_ablation/`: the frozen settings in `configs/frozen.toml`, the same splits
(seed *k* has the same test peptides for every model, so each run pairs by seed with atomic
attention and with the operator alone), 100 epochs, and the test score from the epoch with
the best validation R². Nothing was tuned for these models.

All runs were capped at 65% of the card's 12 GB. Under that cap NNConv + attention used
micro-batches of 8 (4 on Misc) with 16 (32) accumulation steps; every model updates once
per 128 peptides. The `batch_size` and `grad_accum` columns record this per run.

The models are `peporacle.operators.OperatorModel` with 5 attention blocks, and
`python scripts/sweep_operators.py operator_attention` reruns them. The published runs came
from the original research scripts; `tools/check_operators_against_asrun.py` loads each
original model's weights into `OperatorModel` and checks that the predictions, and the
weights after a training epoch, are bit-identical.

## Files

- `runs.csv`: one row per run, with the same columns as `operator_ablation/runs.csv`.
  `run_seconds` is training time summed over any restarts from a checkpoint; the GPU was at
  times shared with other jobs, so these are not clean timings.
- `summary.csv`: mean and sample SD of test R², mean MAE and mean run time per operator
  and dataset.
- `paired_vs_atomic_attention.csv`: operator + attention minus atomic attention test R², per
  seed and as mean, SD and 90% t interval (2 degrees of freedom), with a paired t-test
  p-value. `verdict` compares the interval with the ±0.01 R² margin used in
  `operator_ablation/`: `equivalent` if the interval lies inside it, `atomic attention higher`
  or `operator + attention higher` if it lies entirely beyond it, otherwise `inconclusive`.
- `paired_vs_operator_only.csv`: the same comparison against the operator alone, with
  verdicts `equivalent`, `operator only higher`, `operator + attention higher` or
  `inconclusive`.

Against atomic attention, 37 of the 45 comparisons are equivalent, 0 favour
atomic attention, 1 favours operator + attention (ARMAConv on Misc) and 7 are
inconclusive. Against the operator alone, 23 are equivalent, 13 favour operator +
attention, 0 favour the operator alone and 9 are inconclusive.

## Test R² (mean ± SD, 3 seeds)

Operator + attention; the operator-only results are in `operator_ablation/`.

| dataset | atomic attention | TransformerConv | GCNConv | SAGEConv | ARMAConv | NNConv |
|---|---:|---:|---:|---:|---:|---:|
| HeLa | 0.9618 ± 0.0087 | 0.9542 ± 0.0134 | 0.9576 ± 0.0119 | 0.9633 ± 0.0055 | 0.9460 ± 0.0261 | 0.9515 ± 0.0067 |
| HeLa, modified | 0.9585 ± 0.0069 | 0.9527 ± 0.0056 | 0.9583 ± 0.0062 | 0.9581 ± 0.0057 | 0.9407 ± 0.0131 | 0.9515 ± 0.0038 |
| Yeast | 0.9644 ± 0.0020 | 0.9679 ± 0.0032 | 0.9662 ± 0.0023 | 0.9640 ± 0.0037 | 0.9716 ± 0.0011 | 0.9691 ± 0.0032 |
| SCX | 0.9947 ± 0.0008 | 0.9947 ± 0.0004 | 0.9945 ± 0.0006 | 0.9953 ± 0.0003 | 0.9950 ± 0.0003 | 0.9937 ± 0.0006 |
| Luna HILIC | 0.9826 ± <0.0001 | 0.9873 ± 0.0012 | 0.9854 ± 0.0006 | 0.9835 ± 0.0005 | 0.9885 ± 0.0002 | 0.9859 ± 0.0006 |
| Luna Silica | 0.9824 ± 0.0005 | 0.9855 ± 0.0006 | 0.9849 ± 0.0007 | 0.9829 ± 0.0004 | 0.9867 ± 0.0005 | 0.9854 ± 0.0006 |
| Atlantis Silica | 0.9803 ± 0.0004 | 0.9850 ± 0.0009 | 0.9829 ± 0.0023 | 0.9809 ± 0.0008 | 0.9861 ± 0.0006 | 0.9836 ± 0.0009 |
| XBridge Amide | 0.9862 ± 0.0003 | 0.9914 ± 0.0004 | 0.9898 ± 0.0008 | 0.9868 ± 0.0005 | 0.9927 ± 0.0003 | 0.9905 ± 0.0001 |
| Misc | 0.9780 ± 0.0006 | 0.9894 ± 0.0007 | 0.9826 ± 0.0010 | 0.9777 ± 0.0006 | 0.9920 ± 0.0008 | 0.9842 ± 0.0015 |

# Operator ablation results

Five message-passing operators from PyTorch Geometric, each trained in place of atomic
attention on the same nine datasets and seeds 0-2: 135 runs on one NVIDIA RTX 5070.

## Models

Each model has the same linear atom embedding (width 128), mean and max pooling and
prediction head as atomic attention, with 15 layers of one operator in place of the 15
attention blocks and no attention. Each layer adds its output, after an activation and
dropout, to its input and then applies layer normalisation.

| operator | settings | reads bond features | parameters |
|---|---|---|---:|
| `TransformerConv` | 4 heads x 32 channels, concatenated | yes | 1.05 M |
| `GCNConv` | symmetric normalisation | no | 0.28 M |
| `SAGEConv` | mean aggregation, normalised output | no | 0.53 M |
| `ARMAConv` | 3 stacks x 7 layers | no | 10.4 M |
| `NNConv` | linear map from the 12 bond features to a 128 x 128 weight, mean aggregation | yes | 3.48 M |

Atomic attention has 3.0 M parameters.

## Protocol

Everything except the layers is as for atomic attention: the frozen settings in
`configs/frozen.toml`, the same splits (`peporacle.data.split_indices`, so seed *k* has
the same test peptides for every model), 100 epochs, and the test score from the epoch with
the best validation R². The operators were not tuned separately.

All runs were capped at 65% of the card's 12 GB (`--gpu-memory-fraction 0.65`). Under that
cap NNConv does not fit a batch of 32, so it used micro-batches of 8 (4 on Misc) with 16
(32) accumulation steps; every model updates once per 128 peptides. The
`batch_size` and `grad_accum` columns record this per run.

These runs used the original research scripts; the operator models are not yet part of
`peporacle`.

## Files

- `runs.csv`: one row per run. `test_mae` and `test_rmse` are in each dataset's
  retention-time units. `run_seconds` is training time summed over any restarts from a
  checkpoint; the GPU was at times shared with other jobs, so these are not clean timings.
- `summary.csv`: mean and sample SD of test R², mean MAE and mean run time per operator
  and dataset.
- `paired_vs_atomic_attention.csv`: operator minus atomic attention test R², per seed and
  as mean, SD and 90% t interval (2 degrees of freedom), with a paired t-test p-value.
  `verdict` compares the interval with a margin of ±0.01 R², fixed before these runs:
  `equivalent` if the interval lies inside the margin, `atomic attention higher` or
  `operator higher` if it lies entirely beyond it, otherwise `inconclusive`.

Of the 45 operator-dataset comparisons, 23 are equivalent,
15 favour atomic attention, 1 favours the operator (ARMAConv on Misc) and
6 are inconclusive.

## Test R² (mean ± SD, 3 seeds)

| dataset | atomic attention | TransformerConv | GCNConv | SAGEConv | ARMAConv | NNConv |
|---|---:|---:|---:|---:|---:|---:|
| HeLa | 0.9618 ± 0.0087 | 0.8864 ± 0.0267 | 0.9114 ± 0.0072 | 0.9277 ± 0.0153 | 0.9173 ± 0.0230 | 0.9176 ± 0.0080 |
| HeLa, modified | 0.9585 ± 0.0069 | 0.8654 ± 0.0363 | 0.8909 ± 0.0243 | 0.9359 ± 0.0058 | 0.9233 ± 0.0131 | 0.8994 ± 0.0139 |
| Yeast | 0.9644 ± 0.0020 | 0.9501 ± 0.0037 | 0.9495 ± 0.0064 | 0.9536 ± 0.0048 | 0.9660 ± 0.0026 | 0.9537 ± 0.0076 |
| SCX | 0.9947 ± 0.0008 | 0.9553 ± 0.0045 | 0.9170 ± 0.0101 | 0.9921 ± 0.0007 | 0.9751 ± 0.0019 | 0.9090 ± 0.0102 |
| Luna HILIC | 0.9826 ± <0.0001 | 0.9850 ± 0.0007 | 0.9837 ± 0.0008 | 0.9801 ± 0.0007 | 0.9871 ± 0.0006 | 0.9835 ± 0.0013 |
| Luna Silica | 0.9824 ± 0.0005 | 0.9815 ± 0.0009 | 0.9806 ± 0.0013 | 0.9780 ± 0.0003 | 0.9832 ± 0.0014 | 0.9762 ± 0.0019 |
| Atlantis Silica | 0.9803 ± 0.0004 | 0.9769 ± 0.0011 | 0.9779 ± 0.0028 | 0.9762 ± 0.0008 | 0.9828 ± 0.0011 | 0.9732 ± 0.0025 |
| XBridge Amide | 0.9862 ± 0.0003 | 0.9885 ± 0.0015 | 0.9878 ± 0.0004 | 0.9836 ± 0.0010 | 0.9912 ± 0.0005 | 0.9894 ± 0.0002 |
| Misc | 0.9780 ± 0.0006 | 0.9869 ± 0.0009 | 0.9742 ± 0.0006 | 0.9637 ± 0.0005 | 0.9905 ± 0.0005 | 0.9841 ± 0.0009 |

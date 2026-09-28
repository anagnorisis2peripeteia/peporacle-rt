# Atomic attention results

Test-set results of the frozen configuration (`configs/frozen.toml`) on each dataset,
seeds 0-2, trained on one NVIDIA RTX 5070.

- `runs.csv`: one row per run. `test_mae` and `test_rmse` are in each dataset's
  retention-time units.
- `summary.csv`: mean and sample SD of test R² per dataset; mean MAE where all three
  runs recorded it.

For 11 runs (`record = r2_and_time_only`: HeLa, HeLa-modified and Yeast seeds 0-2, SCX
seeds 0-1) the result files were lost with the machine they ran on. Their test R² and
run times were recovered from the output printed during training; MAE, RMSE and split
sizes were not. The split sizes are fixed by the seed and can be regenerated with
`peporacle.data.split_indices`.

Runs from `scripts/sweep.py` write their own records to `results/sweep/`.

# Tuning record

The atomic-attention configuration was chosen by a 20-trial Optuna TPE search
(`peporacle tune`) on a fixed 10,000-peptide random subset of `misc_dia`
(`numpy.random.default_rng(0xC0FFEE)`, sampled without replacement), scoring the
validation split carved from the subset's training set. Test peptides were never
scored during tuning.

The Optuna database was lost when the training machine was reinstalled. What
survives comes from the run logs:

- `trials.csv`: the 15 completed trials whose results were logged individually,
  and the two trials (1, 4) that failed after machine crashes. `dropout` and `lr`
  were logged for some trials only; blanks mean not recorded, not zero.
- Trials 17-22: five completed and one failed after a crash; their individual
  values were not logged. Trial 23 started after the target of 20 completed
  trials was reached and was stopped.
- Summary over all 20 completed trials: best 0.95399 (trial 11), median 0.95065,
  worst 0.94443, standard deviation 0.00276. The top five lie within 0.00146 of
  each other, less than the seed-to-seed spread, so the objective is flat near
  the optimum.

Trial 11 is the frozen configuration in `configs/frozen.toml`. Its nine searched
parameters were logged at full precision; the batch size (32) was fixed before
tuning and not searched.

Each trial trained a fixed 100 epochs and was scored on the validation set with
its final-epoch weights. The sweep instead keeps the best-validation-epoch
weights; `peporacle tune` and `peporacle sweep` reproduce each rule as it was run.

Two details of the original run:

- The subset graphs were featurised on the x86 training machine. Two
  conformer-derived bond-distance columns differ from the Apple-silicon build of
  the `misc_dia` cache in the last few bits. `peporacle tune` takes the same
  10,000 peptides from the published cache.
- Optuna's sampler was seeded from the clock, so rerunning the search will not
  reproduce this trial sequence.

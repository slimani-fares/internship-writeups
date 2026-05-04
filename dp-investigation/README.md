# DP-SGD Investigation

py-spy profiling of a DP-SGD FedAvg run in declearn with the RDP
accountant, looking at how much overhead the per-step privacy-budget
check adds.  

---

## 1. Discovery

A single py-spy observation pass on declearn's DP-SGD path.

**Setup:** MNIST quickrun (channels-first reshape), 2 rounds × 1
epoch, batch 48, 2 simulated clients. DP enabled with
`budget=[5.0, 1e-5]`, `sclip_norm=1.0`, `accountant="rdp"`. Profiled
with `py-spy record --subprocesses --format speedscope` at 100 Hz.

**Run:** wall time 198.54 s.

In the speedscope flamegraph, opacus's RDP epsilon computation
dominates total runtime:

| function                              | self-time | % of wall |
|---------------------------------------|-----------|-----------|
| `opacus.accountants.rdp.get_privacy_spent` (subtree) | 138.09 s  | 75.26%    |
| `_compute_log_a_for_frac_alpha`       | 39.7 s    | 21.4%     |
| `_compute_log_a_for_int_alpha`        | 17.6 s    | 9.5%      |
| `_log_erfc`                           | 15.8 s    | 8.5%      |

All four hits live inside the RDP accountant. The entry point is
`DPTrainingManager._prevent_budget_overspending`
(`declearn/training/dp/_manager.py:221`), which calls
`self.get_privacy_spent()` once per training step.

**Artifacts** in [data/profiles/](data/profiles/):
`pyspy_speedscope_baseline.json`.

---

## 2. Investigation

`get_privacy_spent()` calls opacus's `accountant.get_epsilon(delta)`,
which walks the full accountant history every time. With one call
per training step, the cumulative cost is O(N²) in step count.

For this run: ~250 SGD steps × 2 rounds × 2 clients = ~1000
`get_epsilon` calls.

---

## 3. The Patch

```python
# DPTrainingManager._prevent_budget_overspending
if self.accountant is not None and self._dp_states is not None:
    noise, srate = self._dp_states
    self.accountant.step(noise_multiplier=noise, sample_rate=srate)
    # this part will be removed
    if self.get_privacy_spent()[0] > self._dp_budget[0]:
        # Remove the step from the history as it will not be taken.
        last = self.accountant.history.pop(-1)
        if last[-1] > 1:
            last = (last[0], last[1], last[2] - 1)
            self.accountant.history.append(last)
        raise StopIteration(
            "Local DP budget would be exceeded by taking the next "
            "training step."
        )
```

A first try: drop the per-step `get_epsilon` check and run the same
workload to compare against the before. The per-round check in
`_training_round` is already there, so the budget is still observed
at every round boundary.

Patch and patched source:
[reproducing/h1_defer_get_epsilon.diff](reproducing/h1_defer_get_epsilon.diff),
[reproducing/_manager.py](reproducing/_manager.py).

---

## 4. Experiment & Results

Controlled before/after at 2 clients, 1 round, 3 seeds (42, 43, 44).
Baseline (canonical 2.8.0 fork) vs patched (variant H1), same TOML,
same data split, same hardware.

|                | Baseline | Patched  | Speedup |
|----------------|----------|----------|---------|
| Wall time      | 89.42 s  | 30.99 s  | 2.89×   |
| Final ε        | 4.996405 | 4.996405 | Δε = 0  |
| Mean accuracy  | 0.1058   | 0.1273   | +0.0215 |

The patch yields a 2.89× speedup on the same workload. Final epsilon
and accuracy stay the same across both arms.

---

## 5. Preview

To inspect the profile in detail (full call trees, per-function
self-time), open the speedscope JSON in
[speedscope.app](https://www.speedscope.app/) (drag and drop into
the browser):

- [data/profiles/pyspy_speedscope_baseline.json](data/profiles/pyspy_speedscope_baseline.json) baseline 2c × 2r

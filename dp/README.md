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

**Run:** wall time 190.66 s, sampled CPU 179.8 s.

In the speedscope flamegraph, opacus's RDP epsilon computation
dominates total runtime:

| function                              | time      | % of sampled CPU |
|---------------------------------------|-----------|------------------|
| `opacus.accountants.rdp.get_privacy_spent` (subtree) | 133.0 s | 74.0%   |
| `_compute_log_a_for_frac_alpha` (self)| 75.7 s    | 42.1%            |
| `_compute_log_a_for_int_alpha` (self) | 22.3 s    | 12.4%            |
| `_log_erfc` (self)                    | 14.0 s    | 7.8%             |

All four hits live inside the RDP accountant. The entry point is
`DPTrainingManager._prevent_budget_overspending`
(`declearn/training/dp/_manager.py:221`), which calls
`self.get_privacy_spent()` once per training step.

Canonical profile:
[data/profiles/canonical.json](data/profiles/canonical.json).

---

## 2. Investigation

`get_privacy_spent()` calls opacus's `accountant.get_epsilon(delta)`,
which walks the full accountant history every time. With one call
per training step, the cumulative cost is O(N²) in step count.

For this run: ~250 SGD steps × 2 rounds × 2 clients = ~1000
`get_epsilon` calls.

---

## 3. Strategies

Three candidate fixes. All preserve the per-step
`accountant.step()` call (cheap, just appends to history), so the
accountant's view of "what training did" is byte-identical to
canonical. Only the timing of `get_epsilon()` changes.

**H1 (defer).** Drop the per-step `get_epsilon` check entirely. The
round-end logging line in `_training_round` is the only remaining
budget probe.

```python
# DPTrainingManager._prevent_budget_overspending
if self.accountant is not None and self._dp_states is not None:
    noise, srate = self._dp_states
    self.accountant.step(noise_multiplier=noise, sample_rate=srate)
    # this part will be removed
    if self.get_privacy_spent()[0] > self._dp_budget[0]:
        ...
        raise StopIteration(...)
```

Trade-off: mid-round budget detection is lost. If a configuration
over-trains beyond what σ was calibrated for, exhaustion is caught
one round late, not one step late.

Full patched source:
[reproducing/_manager_h1.py](reproducing/_manager_h1.py).

**H2 (periodic K=10).** Same as canonical, but call `get_epsilon`
every 10th step instead of every step.

```python
self._step_counter += 1
if self._step_counter % self._budget_check_period != 0:
    return
# fall through to the canonical get_epsilon check
```

Trade-off: detection is delayed by up to K-1 steps. K is a tunable
parameter.

Full patched source:
[reproducing/_manager_h2.py](reproducing/_manager_h2.py).

**H3 (precompute N_max).** opacus's `RDPAccountant` stores history
as `(σ, q, count)` tuples, and `get_epsilon` walks tuples, not
individual steps. Asking "what would ε be after k more steps with
(σ, q)?" costs the same whether k is 1 or 100 000.

At round start, binary-search the largest step count that keeps
total ε ≤ budget. Per-step is then a single integer compare. The
binary search uses
`accountant.history = snapshot + [(noise, srate, k)]` so each probe
is one cheap `get_epsilon` on a compact history.

```python
def _compute_max_steps_for_round(self, noise, srate, max_probe=100_000):
    history_snapshot = list(self.accountant.history)
    try:
        lo, hi = 0, max_probe
        while lo < hi:
            mid = (lo + hi + 1) // 2
            self.accountant.history = history_snapshot + [(noise, srate, mid)]
            if self.accountant.get_epsilon(delta=self._dp_budget[1]) <= self._dp_budget[0]:
                lo = mid
            else:
                hi = mid - 1
        return lo
    finally:
        self.accountant.history = history_snapshot
```

`_prevent_budget_overspending` becomes:

```python
self.accountant.step(noise_multiplier=noise, sample_rate=srate)
self._step_counter_this_round += 1
if self._step_counter_this_round > self._max_steps_this_round:
    # canonical-style rollback, then raise StopIteration
    ...
```

Trade-off: ~17 binary-search probes per round, ~5 s of total
overhead across this 2-round run (h3 wall 52.68 s vs h1 wall
47.38 s). Mid-round detection is preserved exactly.

Full patched source:
[reproducing/_manager_h3.py](reproducing/_manager_h3.py).

---

## 4. Experiment & Results

Controlled A/B at 2 clients, 2 rounds, 3 seeds (42, 43, 44), no
py-spy. Same conditions. Per-run data:
[data/ab_results.json](data/ab_results.json).

| arm           | wall (s) | speedup | mid-round detection |
|---------------|----------|---------|---------------------|
| canonical     | 156.52   | 1.00×   | yes                 |
| h1_deferred   | 47.38    | 3.30×   | lost                |
| h2_periodic   | 58.57    | 2.67×   | within K-1 steps    |
| **h3_precompute** | **52.68** | **2.97×** | **preserved**   |

All four arms produce byte-identical `eps_per_round` for any given
seed:
`[4.5187, 4.5187, 4.9928, 4.9928]` (end-of-round-1 client A and B,
end-of-round-2 client A and B). The accountant history is the same
across arms by construction; only the timing of `get_epsilon`
differs.

---

## 5. Pick: H3

Of the three, H3 is the one to upstream:

1. 2.97× speedup vs canonical (52.68 s vs 156.52 s).
2. Mid-round detection preserved.
3. Bounded overhead: O(log N_max) probes per round, ~5 s total on
   this 2-round configuration; scales linearly in rounds.

---

## 6. Preview

Same arms re-run once with py-spy at 100 Hz (single seed, sampling
overhead +12 to +22% relative to §4):

| canonical | h1_deferred | h2_periodic | h3_precompute |
|-----------|-------------|-------------|---------------|
| 190.66 s  | 53.25 s     | 66.68 s     | 59.65 s       |

Speedscope JSONs for a detailed view (drop them into
[speedscope.app](https://www.speedscope.app/)):

- [data/profiles/canonical.json](data/profiles/canonical.json) canonical
- [data/profiles/h1_deferred.json](data/profiles/h1_deferred.json) H1 defer
- [data/profiles/h2_periodic.json](data/profiles/h2_periodic.json) H2 periodic K=10
- [data/profiles/h3_precompute.json](data/profiles/h3_precompute.json) H3 precompute N_max

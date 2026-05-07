# Vector Dispatch

A writeup of a performance investigation into declearn's `Vector`
abstraction, looking at the per-tensor Python dispatch tax that shows
up as a leaf hotspot in every algorithm we profiled.

## 1. Discovery

Across prior profiling experiments on declearn (regularizers, DP-SGD,
SCAFFOLD, SecAgg), the same leaf frame kept landing at the top of the
self-time table regardless of the algorithm under test:
`declearn.model.api._vector.Vector._apply_operation`.

**Setup:** MNIST quickrun (small CNN, 6 parameter tensors), 2 rounds
× 1 epoch, batch 48, 2 simulated clients. Profiled with
`py-spy record --subprocesses --format speedscope` at 100 Hz.

`_apply_operation` self-time across the experiments:

| algorithm      | `_apply_operation` self-% |
|----------------|---------------------------|
| vanilla FedAvg | 19.4%                     |
| regularizers   | 19.1-21.2%                |
| SCAFFOLD       | 20.1%                     |
| fairness       | 16.2-18.2%                |

---

## 2. Investigation

`Vector._apply_operation` (`declearn/model/api/_vector.py:354`)
walks the parameter dict and calls the underlying torch op once per
tensor in Python:

```python
coefs = {
    key: func(self.coefs[key], other.coefs[key])
    for key in self.coefs
}
```

With N tensors and M training steps per round, that's N × M
Python-level dispatches per client. The torch ops themselves are
essentially free at this size, the cost is the Python iteration.

torch ships batched counterparts (`torch._foreach_add`,
`_foreach_mul`, `_foreach_sign`, ...) that do the same op as a
single C-level call over a list of tensors. Substituting them in
`TorchVector` collapses the per-tensor loop into one batched call
per Vector op.

---

## 3. The Patch

**Before** : one torch call per parameter tensor (parent dispatch
in `model/api/_vector.py`):

```python
coefs = {
    key: func(self.coefs[key], other.coefs[key])
    for key in self.coefs
}
```

**After** : one batched torch call per Vector op
(`TorchVector` override in `model/torch/_vector.py`):

```python
foreach_op = self._FOREACH_BINARY.get(func)
if foreach_op is not None:
    keys = list(self.coefs.keys())
    self_tensors = [self.coefs[k] for k in keys]
    other_arg = [other.coefs[k] for k in keys]
    result = foreach_op(self_tensors, other_arg)
    return type(self)(dict(zip(keys, result, strict=True)))
return super()._apply_operation(other, func)
```

The change lives entirely in `TorchVector`: a small lookup table
maps the framework's known torch ops (add, mul, sub, div, pow,
minimum, maximum, sign, abs, neg, reciprocal) to their
`_foreach_*` counterparts. Any op not in the table falls through
to the canonical per-tensor loop, so unsupported cases keep their
original behavior.

Patched file:
[reproducing/patched-source/_vector.py](reproducing/patched-source/_vector.py).
Diff:
[reproducing/torchvector_foreach.diff](reproducing/torchvector_foreach.diff).

---

## 4. Experiment & Results

Controlled A/B at 2 clients, 2 rounds, 3 seeds (42, 43, 44), no
py-spy. Three algorithms exercised: vanilla FedAvg, lasso, SCAFFOLD.
The dispatch saving scales with the number of parameter tensors,
so we ran the same A/B on two model sizes.

| model            | algo     | master (s) | variant (s) | speedup |
|------------------|----------|------------|-------------|---------|
| small (6 tensors)| vanilla  | 29.01      | 28.03       | 1.03×   |
| small            | lasso    | 29.49      | 29.72       | 0.99×   |
| small            | SCAFFOLD | 28.78      | 28.06       | 1.03×   |
| big (86 tensors) | vanilla  | 177.85     | 165.97      | 1.07×   |
| big              | lasso    | 230.92     | 219.86      | 1.05×   |
| big              | SCAFFOLD | 182.21     | 175.88      | 1.04×   |

At small CNN scale the speedup is within run-to-run noise (master
std 0.3-1.1 s on a 28-30 s mean), inconclusive. At big CNN scale
the improvement clears noise on all three algorithms. Accuracy stays
within 0.10 of master across all seeds.

Per-run data:
[data/ab_results_small.json](data/ab_results_small.json),
[data/ab_results_big.json](data/ab_results_big.json).

---

## 5. Conclusion

The patch removes the per-tensor Python dispatch from `TorchVector`
without changing semantics. The win scales with parameter-tensor
count: negligible on small models like the 6-tensor mnist quickrun
CNN, but noticeable on larger ones (a clean 4-7% on the 86-tensor
CNN, likely larger on ResNet-class or transformer models with
hundreds of tensors).

The per-run gain is small, but the patch sits in core Vector
arithmetic rather than behind an optional flag or specific
algorithm setting. Every torch FedAvg run picks it up by default,
across regularizers, SCAFFOLD, fairness, and any future algorithm
built on the same Vector primitives. Worth upstreaming.

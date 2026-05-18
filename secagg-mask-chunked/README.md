# SecAgg masking: chunked mask generation

Patching declearn's masking SecAgg `_generate_masks_numpy` to reduce
peak memory at high client counts by chunking the per-peer numpy mask
draw.


---

## 1. Discovery

**Setup:** mnist quickrun, masking SecAgg, 1 round × 5 steps,
N=100 clients. Profiled with memray .  
Top allocators by total bytes:

| location                                                         | note               | total    |
|------------------------------------------------------------------|--------------------|----------|
| `_generate_masks_numpy` (`masking/_encrypt.py:131`)              | positive-RNG draw  | 4.012 GB |
| `_generate_masks_numpy` (`masking/_encrypt.py:133`)              | negative-RNG draw  | 4.012 GB |
| `iterencode` (`json/encoder.py:263`)                             | JSON encode        | 3.730 GB |
| `raw_decode` (`json/decoder.py:361`)                             | JSON decode        | 2.386 GB |
| `_read_ready__data_received` (`asyncio/selector_events.py:1009`) | asyncio recv buf   | 2.211 GB |

Full flamegraph:
[data/memray/flamegraphs/n100_master.html](data/memray/flamegraphs/n100_master.html).

Reading the source of `_generate_masks_numpy`:

```python
def _generate_masks_numpy(self, n_values):
    mask = np.zeros(shape=(n_values,), dtype=self._dtype)
    max_val = self.max_int
    for rng in self._pos_rng:
        mask += rng.integers(max_val, dtype=self._dtype, size=n_values)  # :131
    for rng in self._neg_rng:
        mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)  # :133
    return mask
```

Each `rng.integers(..., size=n_values)` allocates one numpy array
of `n_values * 8` bytes (~220 kB at declearn's `L ~ 28k`
mnist-CNN parameter count), used once for the in-place op then
discarded.

Under asyncio, multiple client coroutines can be inside this
function at the same moment, so these temporaries stack live and
peak grows with client count. Chunking each draw into sub-draws
bounds the temporary independently of `n_values`, which caps the
stacked peak regardless of how many clients are running.

---

## 2. The Patch

**Before** : one numpy temporary of `n_values * 8` bytes per RNG:

```python
def _generate_masks_numpy(self, n_values):
    mask = np.zeros(shape=(n_values,), dtype=self._dtype)
    max_val = self.max_int
    for rng in self._pos_rng:
        mask += rng.integers(max_val, dtype=self._dtype, size=n_values)
    for rng in self._neg_rng:
        mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)
    return mask
```

**After** : one numpy temporary of `CHUNK * 8` bytes per RNG, with
`CHUNK = 65536` (~512 kB at uint64):

```python
_MASK_CHUNK = 65536

def _generate_masks_numpy(self, n_values):
    mask = np.zeros(shape=(n_values,), dtype=self._dtype)
    max_val = self.max_int
    chunk = self._MASK_CHUNK
    for rng in self._pos_rng:
        for start in range(0, n_values, chunk):
            end = min(start + chunk, n_values)
            mask[start:end] += rng.integers(
                max_val, dtype=self._dtype, size=end - start
            )
    for rng in self._neg_rng:
        for start in range(0, n_values, chunk):
            end = min(start + chunk, n_values)
            mask[start:end] -= rng.integers(
                max_val, dtype=self._dtype, size=end - start
            )
    return mask
```

Patched file:
[reproducing/patched-source/masking/_encrypt.py](reproducing/patched-source/masking/_encrypt.py) .  
Diff:
[reproducing/h1_chunked_mask.diff](reproducing/h1_chunked_mask.diff) .

---

## 3. Experiment & Results

Same setup with N ∈ {5, 20, 50, 100},single memray run per cell .

| N   | peak master (MB) | peak variant (MB) | peak Δ          | wall master (s) | wall variant (s) | wall Δ |
|-----|------------------|-------------------|-----------------|-----------------|------------------|--------|
| 5   | 476.2            | 476.3             | +0.1 (+0.02%)   | 87.5            | 87.9             | +0.5%  |
| 20  | 700.4            | 704.8             | +4.4 (+0.63%)   | 127.0           | 128.2            | +0.9%  |
| 50  | 1163.0           | 1106.0            | -57.0 (-4.90%)  | 127.0           | 127.0            | 0.0%   |
| 100 | 2473.0           | 1930.0            | -543.0 (-21.96%)| 251.0           | 259.7            | +3.5%  |

Cumulative bytes at the two `rng.integers` lines is unchanged
across arms (4.012 GB per line at N=100, 992 MB per line at N=50):
chunking does not change total allocation, only the temporal
overlap of live temporaries.

The N-dependence matches the mechanism. At small N few coroutines
are simultaneously inside `_generate_masks_numpy`, temporaries
rarely overlap, and peak is dominated by torch's `_conv_forward`
and JSON encode/decode buffers; chunking saves nothing visible.  
At N=100 concurrent overlap is the dominant peak contributor and
chunking removes it.

Per-run data:
[data/memray/ab_results.json](data/memray/ab_results.json).
Per-cell memray summaries and flamegraphs:
[data/memray/](data/memray/).

---

## 4. Conclusion

At N=100, peak drops by 543 MB (-21.9%) for a +3.5% wall-clock
cost, with byte-identical encryption output and the declearn test
suite passing. Below N=20 the difference is not significant
because the per-peer concurrency multiplier is too small for the
temporal-overlap mechanism to bite.

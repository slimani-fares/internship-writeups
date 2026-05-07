# SecAgg Investigation

A writeup of a performance investigation into declearn's masking-variant
SecAgg, from the initial profiling pass through the controlled
experiment that validated the patch.

## Layout

```
secagg/
├── data/
│   └── profiles/             ← py-spy speedscope JSONs
└── reproducing/
    └── patched-source/       ← modified declearn files
```

---

## 1. Discovery

Exploratory py-spy profiling of declearn's masking SecAgg, varying
only N (number of simulated clients). Three single runs at N = 3, 5,
10.

**Setup:** MNIST quickrun, 3 rounds × 10 local steps,
batch 48. Masking SecAgg with `bitsize=64`, `clipval=1e8`, per-client
Ed25519 keys. Profiled with
`py-spy record --subprocesses --format speedscope` at 100 Hz.

**Runs:**

| N         | 3      | 5      | 10      |
|-----------|--------|--------|---------|
| Wall time | 32.3 s | 79.3 s | 323.4 s |

Comparing the speedscope flamegraphs across N values,
`_generate_masks_numpy` (`declearn/secagg/masking/_encrypt.py`,
lines 131+133) stood out and its share of total runtime grew with
N:

| N                          | 3     | 5     | 10    |
|----------------------------|-------|-------|-------|
| Wall (s)                   | 32.3  | 79.3  | 323.4 |
| `encrypt_uint` subtree (s) | 22.5  | 66.8  | 306.1 |
| % of wall                  | 69.7% | 84.2% | 94.6% |

**Artifacts** in [data/profiles/](data/profiles/):
`pyspy_speedscope_n{3,5,10}.json`.

---

## 2. Investigation

Encryption was done unit by unit: for every scalar in the parameter
vector, the encrypter asked the PRNG for one fresh mask number,
added it modulo `max_int`, and moved on. With many parameters and
many peers, that's a lot of tiny numpy calls: Python/numpy dispatch
overhead, not crypto work.

---

## 3. The Patch

**Before** : one mask draw per element:

```python
# Encrypter.encrypt_vector
enc_val = [self.encrypt_uint(val) for val in int_val]

# encrypt_uint internally calls self._generate_masks(1) every time
```

**After** : one mask draw per vector:

```python
# Encrypter.encrypt_vector
enc_val = self.encrypt_uint_vector(int_val)

# new MaskingEncrypter override:
def encrypt_uint_vector(self, values):
    if not values:
        return []
    masks = self._generate_masks(len(values))
    return [(v + int(m)) % self.max_int for v, m in zip(values, masks)]
```

Three additive edits land this:

- `declearn/secagg/api/_encrypt.py:96–107` new
  `encrypt_uint_vector` on the `Encrypter` ABC, default impl is the
  per-element fallback (Joye-Libert inherits unchanged).
- `declearn/secagg/masking/_encrypt.py:143–154`  the
  `MaskingEncrypter` override above.
- `declearn/secagg/api/_encrypt.py:189` the one-line swap inside
  `encrypt_vector`.

The modified files can be found here:
[reproducing/patched-source/api/_encrypt.py](reproducing/patched-source/api/_encrypt.py)
and
[reproducing/patched-source/masking/_encrypt.py](reproducing/patched-source/masking/_encrypt.py).


---

## 4. Experiment & Results

Controlled before/after at N=10: baseline (unpatched fork) vs
patched (`declearn-for-secagg-batched`), same workload, same seeds,
same hardware.

|                                | Baseline         | Patched       | Speedup |
|--------------------------------|------------------|---------------|---------|
| Wall time                      | 323.43 s         | 19.85 s       | 16.3×   |
| `encrypt_vector` (entry point) | 306.65 s (94.8%) | 0.93 s (4.7%) | 330×    |

---

## 5. Preview

To inspect the profiles in detail, full call trees, per-function
self-time, side-by-side run comparison, open the speedscope JSONs
in [speedscope.app](https://www.speedscope.app/) (drag and drop into
the browser):

- [data/profiles/pyspy_speedscope_n3.json](data/profiles/pyspy_speedscope_n3.json)  baseline N=3 
- [data/profiles/pyspy_speedscope_n5.json](data/profiles/pyspy_speedscope_n5.json)  baseline N=5 
- [data/profiles/pyspy_speedscope_n10_baseline.json](data/profiles/pyspy_speedscope_n10_baseline.json)  baseline N=10 
- [data/profiles/pyspy_speedscope_n10_patched.json](data/profiles/pyspy_speedscope_n10_patched.json)  patched N=10 

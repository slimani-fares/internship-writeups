# Benchmarks preparation

Integrating a benchmarking system into declearn means setting up a
fixed set of benchmarks that get launched on each new release.

declearn already ships examples (quickrun, the heart-uci and mnist
demos), but these are user-facing demonstrations of the API, not
benchmarks. The quickrun launcher in particular has three hardcoded
constraints that block most of the variations a benchmark suite needs
to cover:

- `secagg=None` is hardcoded in `quickrun/_run.py`, so SecAgg cannot
  be reached from a quickrun config.
- `InMemoryDataset` is wired in where `FairnessInMemoryDataset` is
  needed for fairness experiments.
- The number of clients is tied to the number of subdirectories in
  the data folder, so changing N requires regenerating data manually.

A dedicated benchmarking example was therefore built on the side, one
that bypasses quickrun and drives `FederatedServer` + N
`FederatedClient`s directly through declearn's main API.

## Layout

Three layers:

- **Builder**
  ([`reproducing/workload/build.py`](reproducing/workload/build.py)):
  one function `build_benchmark(...)` turns a small set of toggles
  (`backend`, `n_clients`, `dp`, `scaffold`, `secagg`, `regularizer`,
  `rounds`, `batch_size`) into a
  [`BenchmarkSpec`](reproducing/workload/spec.py) (server `Model`,
  `FLOptimConfig`, `FLRunConfig`, per-client `Dataset`s, optional
  SecAgg config, network config).
- **Runner**
  ([`reproducing/workload/runner.py`](reproducing/workload/runner.py)):
  spawns the server + clients on asyncio, awaits `asyncio.gather` to
  completion. The quickrun replacement.
- **ASV classes**
  ([`reproducing/__init__.py`](reproducing/__init__.py)): thin
  wrappers declaring which slice of the parameter space each benchmark
  exercises. Each class has `params`, `param_names`, a `setup(...)`
  that triggers data prep, and a `time_run(...)` that calls
  `build_benchmark + run_benchmark`.

Supporting modules under
[`reproducing/workload/`](reproducing/workload/):
[`baseline.py`](reproducing/workload/baseline.py) (fixed baseline
values), [`data.py`](reproducing/workload/data.py) (MNIST split + per-
layout caching), and
[`models/`](reproducing/workload/models/) (one file per backend).

## Hardware

Cluster node Magnet 8 at Inria Lille (`fslimani@magnet8`, used for
now), NVIDIA A10 GPU (23 GB, compute capability 8.6), CUDA driver
570.169.

## The matrix

8 cells per commit. Fixed baseline (torch + averaging + adam,
`rounds=2`, `batch_size=48`, `dataset_fraction=0.1`), each category
varies one axis from it.

| Class                 | Cells | Axis varied                              | Notes                                          |
|-----------------------|-------|------------------------------------------|------------------------------------------------|
| BackendsBenchmark     | 2     | backend in {torch, tensorflow}           | sklearn dropped, haiku dropped                 |
| RegularizersBenchmark | 3     | regularizer in {lasso, ridge, fedprox}   | torch FedAvg                                   |
| DPBenchmark           | 1     | DP-SGD on torch                          | budget=[5.0, 1e-5], rdp accountant             |
| ScaffoldBenchmark     | 1     | SCAFFOLD on torch                        |                                                |
| SecAggBenchmark       | 1     | secagg in {masking}                      | joye-libert dropped (timeout), pinned to n=5   |

`n_clients` is fixed at 5 globally (`N_CLIENTS_AXIS = [5]`) to keep
wall-time bounded. `dataset_fraction = 0.1` trains each client on 10 %
of its MNIST shard; the full FL pipeline (registration, aggregation,
eval, encryption, DP accounting) is exercised identically, only the
inner training loop sees less data, since the goal is to benchmark
declearn plumbing, not MNIST training.

On hold for now:

- **sklearn**: ~78 min/version with full data. The linear model used
  here runs on CPU and does not pick up GPU acceleration the way torch
  and tensorflow do, and the inner training loop iterates per step
  over a larger effective batch count. Could come back if a
  sklearn-specific perf question appears.
- **haiku**: `models/haiku_cnn.py` is a stub that raises
  `NotImplementedError`. Waiting on a working model implementation.
- **n_clients=20**: amplified overhead disproportionately at full
  shard sizes. Re-enable once re-validated against the patched fork.
- **joye-libert**: modular exponentiation × CNN parameter count
  exceeds practical timeouts. Reproducible with a smaller bitsize or
  smaller model.


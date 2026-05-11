# MessagePack serialization: memory comparison

Upstream declearn MR !82 (commit `d402885b`) replaces JSON with MessagePack
across the messaging layer. This page records the memory footprint of a
plain federated workload before and after that change.

## Setup

| Field        | Value                                              |
|--------------|----------------------------------------------------|
| Workload     | MNIST quickrun, small CNN, no SecAgg               |
| Aggregator   | plain averaging                                    |
| Per run      | 5 rounds, 5 SGD steps, batch 48                    |
| Client count | N ∈ {2, 5, 10}                                     |
| Repeats      | 2 per cell, 12 runs total                          |
| Profiler     | memray, Python-only mode                           |
| Baseline     | declearn `develop` @ `9668576`                     |
| Treatment    | declearn MR !82 `msgpack_serial` snapshot          |

## Results

| N  | Peak baseline (MB) | Peak msgpack (MB) | Peak Δ           | Total baseline (GB) | Total msgpack (GB) | Total Δ            | Wall Δ |
|----|--------------------|--------------------|-------------------|----------------------|---------------------|--------------------|--------|
| 2  | 408.0              | 406.3              | -1.7 (-0.4%)      | 3.3                  | 3.1                 | -221 MB (-6.7%)    | flat   |
| 5  | 428.9              | 417.2              | -11.7 (-2.7%)     | 5.1                  | 4.5                 | -599 MB (-11.7%)   | flat   |
| 10 | 459.9              | 440.0              | -19.9 (-4.3%)     | 8.2                  | 7.0                 | -1.18 GB (-14.3%)  | flat   |



## Data

Result flamegraphs are available here:
[baseline N=10](data/flamegraphs/baseline_n10.html),
[msgpack N=10](data/flamegraphs/msgpack_n10.html).

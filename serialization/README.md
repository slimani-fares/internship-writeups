# MessagePack serialization: comparison

Upstream declearn MR !82 (commit `d402885b`) replaces JSON with MessagePack
across the messaging layer. This page records measurements of a plain
federated workload before and after that change.

## Setup

| Field        | Value                                              |
|--------------|----------------------------------------------------|
| Workload     | MNIST quickrun, small CNN, no SecAgg               |
| Aggregator   | plain averaging                                    |
| Per run      | 5 rounds, 5 SGD steps, batch 48                    |
| Client count | N ∈ {2, 5, 10, 20, 50, 100}                        |
| Repeats      | 2 per cell, 24 runs total                          |
| Profiler     | memray, Python-only mode                           |
| Baseline     | declearn `develop` @ `9668576`                     |
| Treatment    | declearn MR !82 `msgpack_serial` snapshot          |

## Results

| N   | Peak baseline (MB) | Peak msgpack (MB) | Peak Δ            | Total baseline (GB) | Total msgpack (GB) | Total Δ            | Wall time Δ |
|-----|--------------------|--------------------|--------------------|----------------------|---------------------|--------------------|--------|
| 2   | 408.0              | 406.3              | -1.7 (-0.4%)       | 3.3                  | 3.1                 | -221 MB (-6.7%)    | -0.3%  |
| 5   | 428.9              | 417.2              | -11.7 (-2.7%)      | 5.1                  | 4.5                 | -599 MB (-11.7%)   | -4.4%  |
| 10  | 459.9              | 440.0              | -19.9 (-4.3%)      | 8.2                  | 7.0                 | -1.18 GB (-14.3%)  | -2.7%  |
| 20  | 527.5              | 483.7              | -43.8 (-8.3%)      | 14.5                 | 12.1                | -2.36 GB (-16.3%)  | -6.8%  |
| 50  | 720.6              | 615.5              | -105.1 (-14.6%)    | 33.2                 | 27.3                | -5.91 GB (-17.8%)  | -9.5%  |
| 100 | 1047.5             | 835.9              | -211.6 (-20.2%)    | 65.1                 | 53.0                | -12.16 GB (-18.7%) | -12.1% |



## Data

Result flamegraphs (one per arm per N) are available in
[data/flamegraphs/](data/flamegraphs/).

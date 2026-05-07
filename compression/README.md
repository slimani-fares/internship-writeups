# Compression

py-spy profiling of a torch FedAvg run in declearn over the
WebSockets transport, looking at how much overhead
the per-message WebSocket compression adds. If you are not familiar
with how WebSocket compression works, see
[this explainer](https://sachin9996.github.io/websocket-compression-explained/)
first.

## 1. Discovery

Cross-N py-spy profiling of a torch FedAvg run, varying only N
(number of simulated clients). Profiled at N = 5 and N = 100.

**Setup:** `config_fedavg_torch.toml`, 2 rounds, batch 48, 1 epoch,
IID split (seed 42). Profiled with
`py-spy record --subprocesses --format speedscope` at 100 Hz.

Comparing the speedscope flamegraphs across N values,
`permessage_deflate.encode`
(`websockets/extensions/permessage_deflate.py`, line 173) appeared at
the top of the self-time table at N=100 but had been negligible at
lower N:

| N                      | 5      | 100     |
|------------------------|--------|---------|
| Wall (s)               | 21.49  | 53.97   |
| `encode` self-time (s) | 1.03   | 20.78   |
| % of wall              | 4.79%  | 38.50%  |

**Artifacts** in [data/profiles/](data/profiles/):
`pyspy_speedscope_n5_baseline.json`,
`pyspy_speedscope_n100_baseline.json`.

---

## 2. Investigation

`permessage_deflate.encode` is the per-message DEFLATE compression
hook from the `websockets` library: every WebSocket message is
compressed before going on the wire. The library turns it on by
default, and declearn never passes a `compression=` argument to
`ws.serve()` or `ws.connect()`, so there is no way to turn it off
from declearn's side.

To measure the actual cost and check whether removing it hurts
anything, we need to expose `compression` as a parameter, then run
the same workload with and without it.

---

## 3. The Patch

Two edits.

In `declearn/communication/websockets/_server.py`, in `start()`, add
`compression=None` to the `ws.serve(...)` call at line 134:

```python
server = ws.serve(
    self._handle_connection,
    host=self.host,
    port=self.port,
    logger=self.logger,
    ssl=self._ssl,
    ping_timeout=None,
    compression=None,
)
```

In `declearn/communication/websockets/_client.py`, in `start()`, add
`"compression": None` to the kwargs dict at line 122:

```python
kwargs = {
    "uri": self.server_uri,
    "logger": self.logger,
    "ssl": self._ssl,
    "extra_headers": (...),
    "ping_timeout": None,
    "compression": None,
}
```

The modified files (originals at
`declearn/communication/websockets/_server.py` and
`declearn/communication/websockets/_client.py`) are kept here:
[reproducing/_server.py](reproducing/_server.py) and
[reproducing/_client.py](reproducing/_client.py).

---

## 4. Experiment & Results

Controlled before/after at N=100: stock declearn (compression default
is `"deflate"`) vs modified declearn (compression=None on both server
and client), same workload, same data split, same hardware, same
py-spy invocation.

Wall-clock and compression self-time (from the speedscope profiles):

|                  | Baseline         | Patched      | Δ        |
|------------------|------------------|--------------|----------|
| Wall time        | 53.97 s          | 32.32 s      | -21.65 s |
| `encode` (server)| 20.78 s (38.50%) | 0.00 s       | -20.78 s |
| `decode` (client)| 2.43 s (4.50%)   | 0.00 s       | -2.43 s  |

Whole-run speedup: **40.1% shorter wall time**.

---

## 5. Preview

To inspect the profiles in detail (full call trees, per-function
self-time, side-by-side run comparison), open the speedscope JSONs
in [speedscope.app](https://www.speedscope.app/) (drag and drop into
the browser):

- [data/profiles/pyspy_speedscope_n5_baseline.json](data/profiles/pyspy_speedscope_n5_baseline.json) baseline N=5
- [data/profiles/pyspy_speedscope_n100_baseline.json](data/profiles/pyspy_speedscope_n100_baseline.json) baseline N=100
- [data/profiles/pyspy_speedscope_n100_no_compression.json](data/profiles/pyspy_speedscope_n100_no_compression.json) patched N=100

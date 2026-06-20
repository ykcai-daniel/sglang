# PD Disaggregation: Configurable Parameters and Tuning Guide

Parameters are grouped by which side they affect and what failure mode they address.

---

## Decode Side

### `--num-reserved-decode-tokens` (default: 512)

**File:** `server_args.py:835`, used in `decode.py:376,876,1086,1151`

The number of KV slots reserved per active decode request when computing whether a new request can be pre-allocated. Each request in the running batch, retracted queue, and last batch occupies `num_reserved_decode_tokens` slots of headroom.

```
allocatable_tokens = available + evictable
                     − num_reserved_decode_tokens × n_active_reqs
                     − Σ(retracted_req costs)
```

**Increase when:** retraction storms are frequent (decode OOM, retracted_queue blocks all new admits). A higher value gives the running batch more headroom but admits fewer new requests.

**Decrease when:** decode GPU is underutilized because the pre-alloc budget runs out before the pool is actually full. A lower value is more aggressive but risks more frequent OOM retractions.

**Relationship:** analogous to `new_token_ratio` on the normal scheduler, but fixed rather than adaptive.

---

### `--disaggregation-decode-polling-interval` (default: 1)

**File:** `server_args.py:837`, used in `decode.py:1891`

`process_decode_queue` (the function that drains completed RDMA transfers into `waiting_queue`) runs only every N scheduler steps, where N = this value. At the default of 1, it runs every step.

**Increase when:** the overhead of polling `disagg_decode_transfer_queue` each step is measurable (high request rate with many in-flight transfers). Setting to 2–4 roughly halves polling overhead.

**Trade-off:** higher values add latency jitter — a completed KV transfer may sit in the transfer queue for up to N steps before the request is admitted to `waiting_queue`.

---

### `SGLANG_DISAGGREGATION_QUEUE_SIZE` (default: 4)

**File:** `environ.py:295`, used in `mooncake/conn.py:189`, `nixl/conn.py:306`

The size of the in-flight RDMA transfer queue per connection in the Mooncake/NIXL backend. Limits how many concurrent KV transfers can be tracked simultaneously per decode worker.

**Increase when:** decode workers have high parallelism (many TP ranks) and many requests arrive in bursts, causing transfer queue saturation. Must satisfy `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE >= SGLANG_DISAGGREGATION_QUEUE_SIZE`.

---

### `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` (default: derived from TP size)

**File:** `environ.py:294`, used in `mooncake/conn.py:185`

Thread pool size for the Mooncake RDMA transfer backend. The default is automatically derived from `tp_size`. Override when you need more concurrent transfer threads (e.g., high-bandwidth IB with many in-flight transfers).

---

### `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (default: 300 seconds)

**File:** `environ.py:299`, used in `conn.py:187`

Timeout in seconds for a request waiting in `KVPoll.WaitingForInput` state (KV pre-allocated on decode side, waiting for prefill to push the KV). After this timeout, the request is aborted with HTTP 500.

**Increase when:** network is slow or variable (high-latency IB fabric, congested links) causing KV transfers to take longer than expected. The error message at `conn.py:1058` suggests `600` for slow networks.

---

### `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS` (default: 0)

**File:** `environ.py:307`, used in `model_runner_kv_cache_mixin.py:310`

Number of requests to pre-allocate KV buffers for at model runner startup. Non-zero values eagerly reserve GPU memory at launch. Generally leave at 0 unless startup latency for the first batch of requests is critical.

---

### `--disaggregation-decode-enable-kvcache` (flag, default: off)

**File:** `server_args.py:834`

Enables CPU offloading of KV cache on the decode side (requires `--hicache-storage-backend`). When enabled, retracted requests can offload their KV to CPU rather than releasing it entirely, allowing faster re-admission without re-transfer from prefill. Incompatible with HiSparse.

---

## Prefill Side

### `--optimistic-prefill-retries` (default: 0)

**File:** `server_args.py:838`, used in `prefill.py:353,995`

Number of times a request can skip the bootstrap wait and attempt prefill speculatively while still in `KVPoll.Bootstrapping` state. A request with `prefill_retry_count < optimistic_prefill_retries` is treated as if it had already bootstrapped and moved to `waiting_queue` early.

**Use case:** reduces TTFT when bootstrap round-trips have significant latency overhead. The optimistic path bets that the decode side will have KV allocated by the time prefill finishes; if not, `optimistic_release_and_requeue` (prefill.py:993) releases KV and re-queues the request for another retry.

**Risk:** if prefill finishes before decode bootstraps, the KV is released and recomputed — wasted GPU work. Effective only when prefill latency >> bootstrap latency.

**After exhausting retries:** falls back to `disagg_prefill_bootstrap_queue` (prefill.py:1012), the normal bootstrap wait path.

---

### `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` (default: 300 seconds)

**File:** `environ.py:296`, used in `conn.py:164`

Timeout in seconds for a request waiting in `KVPoll.Bootstrapping` state (waiting for decode side to pre-allocate KV and signal readiness). After this timeout, `KVPoll.Failed` is returned to `pop_bootstrapped`, which calls `handle_bootstrap_failure` and aborts with HTTP 500.

**This is the only explicit bound on how long a request can be stuck waiting for decode-side headroom.** When the decode side is overloaded and cannot pre-allocate, all waiting requests will eventually hit this timeout.

**Increase when:** decode side is genuinely slow to pre-alloc (high load, large model) and the extra wait is acceptable. The error message at `conn.py:830` suggests `600` for high-load scenarios.

**Decrease when:** faster failure detection is preferred over waiting (low-latency SLO environments).

---

### `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` (default: 5.0 seconds)

**File:** `environ.py:297`, used in `conn.py:178`

How frequently the bootstrap connection sends heartbeats to check if the remote side (decode) is still alive. Shorter intervals detect decode-side failures faster at the cost of more network overhead.

---

### `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` (default: 2)

**File:** `environ.py:298`, used in `conn.py:182`

Number of consecutive heartbeat failures before the bootstrap connection is declared dead and the request is failed. `interval × max_failure` is the effective dead-decode detection window (default: 10 seconds).

---

### `SGLANG_DISAGG_STAGING_BUFFER` / `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` / `SGLANG_DISAGG_STAGING_POOL_SIZE_MB`

**File:** `environ.py:371-373`, used in `prefill.py:418`, `decode.py:324`, `mooncake/conn.py:175`, `nixl/conn.py:292`

- `SGLANG_DISAGG_STAGING_BUFFER=1`: enables a CPU staging buffer between GPU KV and RDMA send. Designed for non-MLA models; incompatible with MLA backends.
- `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` (default: 64): size of each staging buffer chunk.
- `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` (default: 4096): total staging pool size. The error at `staging_handler.py:734` ("increase this") fires when the pool is exhausted under burst traffic.

**Use when:** GPU-to-RDMA direct transfer has poor performance (PCIe bandwidth issues); staging through CPU pinned memory can improve throughput on some hardware configurations.

---

## Both Sides (Scheduler-Level)

### `--schedule-conservativeness` (default: 1.0)

**File:** `server_args.py:439`, used in `new_token_ratio_tracker.py:23`

Multiplier applied to `SGLANG_INIT_NEW_TOKEN_RATIO` at startup. Only relevant on the **prefill side** (the decode side uses `num_reserved_decode_tokens` instead). In PD mode, the prefill side has no running decode batch, so `new_token_ratio` rarely engages — but it still gates prefill admission for chunked-prefill scenarios.

**Relevant on prefill side when:** chunked prefill is enabled and the prefill worker also accumulates partial decode state.

---

### `SGLANG_INIT_NEW_TOKEN_RATIO` (default: 0.7)

**File:** `environ.py:279`, `new_token_ratio_tracker.py:23`

Starting value of the decode reservation factor on the **normal** (non-PD) scheduler. In PD prefill mode, effective headroom reservation is effectively 0 (no decode batch). In PD decode mode, `num_reserved_decode_tokens` is the active mechanism instead.

---

### `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL` (default: 120 seconds)

**File:** `environ.py:328`

How frequently stale/orphaned bootstrap entries are cleaned up from the bootstrap registry. Increase on long-lived sessions where many requests time out; decrease if memory from orphaned entries is a concern.

---

## Summary Table

| Parameter | Side | Default | Tuning Signal |
|---|---|---|---|
| `--num-reserved-decode-tokens` | Decode | 512 | Frequent retractions → increase; idle decode GPU → decrease |
| `--disaggregation-decode-polling-interval` | Decode | 1 | High polling CPU overhead → increase (adds latency jitter) |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Decode | 4 | Transfer queue saturation under burst → increase |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | Decode | auto | Saturated RDMA threads → increase |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | Decode | 300s | Slow network / high-load decode → increase |
| `--optimistic-prefill-retries` | Prefill | 0 | High bootstrap latency, low retry cost → increase |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | Prefill | 300s | Slow decode pre-alloc → increase; fast failure detection → decrease |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | Prefill | 5.0s | Faster dead-decode detection → decrease |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | Prefill | 2 | More tolerance for transient failures → increase |
| `SGLANG_DISAGG_STAGING_BUFFER` | Both | off | Poor GPU-to-RDMA bandwidth → enable |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | Both | 4096 | Pool exhaustion errors under burst → increase |
| `--schedule-conservativeness` | Prefill | 1.0 | Frequent retraction on prefill (chunked mode) → increase |

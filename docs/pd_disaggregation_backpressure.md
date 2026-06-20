# PD Disaggregation: Backpressure and Failure Modes

## Overview

The prefill and decode fleets are fully decoupled — there is no shared memory and no direct signaling channel between prefill and decode schedulers beyond the RDMA transfer and the bootstrap handshake. All backpressure is local-side admission control. Failure modes when the P:D ratio is wrong are qualitatively different depending on which side is the bottleneck.

---

## Decode Side: `num_reserved_decode_tokens` — Primary Admission Control

**File:** `python/sglang/srt/disaggregation/decode.py:780`

Before adding a new incoming request to the transfer queue, `pop_preallocated` checks:

```
allocatable_tokens = available_kv_slots + evictable_kv_slots
                     − num_reserved_decode_tokens × n_active_reqs
                     − prealloc_cost_of_retracted_reqs
```

`num_reserved_decode_tokens` (configured via `--num-reserved-decode-tokens`) is the decode analogue of `new_token_ratio` on the normal scheduler. It reserves N KV slots per already-running decode request to ensure the existing batch can continue decoding for N more steps. A new incoming request cannot be preallocated if admitting it would leave the running batch unable to decode.

This reservation covers requests across multiple stages:
- `running_batch.reqs` (currently decoding)
- `retracted_queue` (retracted, waiting for headroom)
- `scheduler.last_batch.reqs` (just finished prebuilt extend, not yet in running_batch)

---

## Decode Side: `retracted_queue` — Reactive OOM Recovery

**File:** `python/sglang/srt/disaggregation/decode.py:463, 568, 1885`

When the decode batch OOM fires (`update_running_batch` → `check_decode_mem` fails → `retract_decode`), retracted requests go to `disagg_decode_prealloc_queue.retracted_queue`, not `waiting_queue`:

```python
def add(self, req, is_retracted=False):
    if is_retracted:
        self.retracted_queue.append(req)  # separate from the prealloc queue
```

**Critical gate in `process_decode_queue`:** if `retracted_queue` is non-empty, new prealloc is completely blocked:

```python
if len(disagg_decode_prealloc_queue.retracted_queue) > 0:
    return   # no new requests admitted while retracted reqs are pending
```

`resume_retracted_reqs` (decode.py:568) re-admits retracted requests only when there is sufficient headroom — verified by `_prealloc_required_tokens` against `_allocatable_token_budgets(count_retracted=False)`. Retracted requests release their GPU KV (CPU copy retained via `load_kv_cache`), and `resume_retracted_reqs` re-allocates GPU KV and restores from CPU when space exists.

The `_allocatable_token_budgets` formula ensures retracted requests hold their full headroom reservation, preventing thundering-herd re-admission:

```
budget = available − max(reserved_active, need_for_single_req) − Σ(retracted_req costs)
```

---

## Prefill Side: Bootstrap Queue as the Only Cross-Fleet Feedback Signal

**File:** `python/sglang/srt/disaggregation/prefill.py:307`

The prefill scheduler has no direct awareness of the decode side's KV pool pressure. Its only backpressure is through the bootstrap handshake itself.

When a request arrives at the prefill scheduler, it enters `PrefillBootstrapQueue`. To make progress, it must reach `KVPoll.WaitingForInput` — meaning the decode side has pre-allocated its KV buffer and registered an RDMA receive slot. `pop_bootstrapped` polls every step and only moves requests that reach `WaitingForInput` into `waiting_queue`.

**This is the only feedback channel between the two fleets.** If the decode side cannot pre-allocate (its `pop_preallocated` stops because `allocatable_tokens` is exhausted), requests stay stuck in `KVPoll.Bootstrapping` on the prefill side. The prefill scheduler sees a growing `disagg_prefill_bootstrap_queue.queue` and an empty `waiting_queue`. Since `get_new_batch_prefill` requires a non-empty `waiting_queue`, the prefill scheduler **goes idle** — it stops computing. This is backpressure working correctly, but entirely emergently — it is not a designed flow-control protocol.

---

## Failure Modes by Imbalanced P:D Ratio

### Case 1: Too Many Prefill Workers (decode is the bottleneck)

**Condition:** prefill throughput > decode throughput; KV arrives faster than decode can drain.

**Chain of events:**
1. Decode KV pool fills → `pop_preallocated` stops admitting
2. New requests stuck in `KVPoll.Bootstrapping` on decode side
3. `PrefillBootstrapQueue.queue` grows on prefill side
4. `waiting_queue` on prefill empties → `get_new_batch_prefill` returns None
5. Prefill GPU goes idle

**Visible symptoms:**
- Prefill GPU utilization drops toward 0
- Decode GPU utilization at 100%
- `disagg_prefill_bootstrap_queue` grows unboundedly (no size cap)
- TTFT increases as requests wait for bootstrap completion
- No client-facing error until bootstrap timeout or handshake failure

**Silent cap:** `disagg_prefill_bootstrap_queue` is not explicitly bounded, but each bootstrapping request holds a `MetadataBuffer` slot (pool size = `max_running_requests × 2`). Once all buffer slots are consumed, `ensure_metadata_buffer` (prefill.py:356) returns False and the request stays in `Bootstrapping` — silently capping the queue without any error.

### Case 2: Too Few Prefill Workers (prefill is the bottleneck)

**Condition:** decode throughput > prefill throughput; decode workers idle-wait for KV.

**Chain of events:**
1. `DisaggDecodeTransferQueue` has no completed transfers
2. `waiting_queue` on decode empties
3. `get_new_prebuilt_batch` returns None
4. Decode GPU goes idle, `running_batch` drains as requests finish

**Visible symptoms:**
- Decode GPU underutilized
- `num_reserved_decode_tokens` reservation is irrelevant (pool is mostly empty)
- No backpressure mechanism engages — decode simply waits

### Case 3: Transient Burst (temporarily mismatched ratio)

**Condition:** Burst of completions or arrivals temporarily overloads the decode side.

**Recovery path:**
1. `update_running_batch` → `check_decode_mem` fails → `retract_decode`
2. Retracted requests moved to `retracted_queue` (CPU KV copy retained)
3. `process_decode_queue` blocks all new prealloc until `retracted_queue` is drained
4. `resume_retracted_reqs` re-admits one-by-one as `_allocatable_token_budgets` allows
5. GPU KV restored from CPU; request re-enters running batch

This is analogous to `retract_decode` in normal mode but happens at the prealloc stage — before the request even starts its first decode step.

---

## Summary: Backpressure Channels and Gaps

| Condition | Mechanism | Location | Gap |
|---|---|---|---|
| Decode KV pool full | `pop_preallocated` gates on `allocatable_tokens` | decode.py:780 | bootstrap queue grows unboundedly |
| Decode retracted reqs present | new prealloc blocked entirely | decode.py:1885 | — |
| Prefill bootstrap stuck | `waiting_queue` empty → prefill goes idle | prefill.py:424 | no timeout, no metric, silent |
| Decode OOM during decode step | `retract_decode` → `retracted_queue` | normal path | retracted reqs block all new admits |
| Metadata buffer exhausted | `ensure_metadata_buffer` → req stays Bootstrapping | prefill.py:356 | silent cap, no client error |

---

## Key Architectural Observation

There is no explicit backpressure signal from decode back to prefill. The only coupling is through the RDMA bootstrap handshake: if decode cannot pre-allocate, prefill naturally stalls. This is entirely emergent — it is not a designed flow-control mechanism. There are no counters, metrics, or alerts specifically for `disagg_prefill_bootstrap_queue` depth or bootstrap wait latency in the scheduler itself.

The practical implication is that an imbalanced P:D ratio will manifest as **silent latency increases** rather than explicit errors until a bootstrap handshake fails (`KVPoll.Failed`, decode.py:643), at which point the prefill side raises HTTP 500 for that request.

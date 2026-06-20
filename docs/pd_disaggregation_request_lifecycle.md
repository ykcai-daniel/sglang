# PD Disaggregation: Full Request Lifecycle

Both fleets run independent schedulers. A request touches code on both sides, linked only by `bootstrap_room` (a unique integer ID assigned by the router).

---

## Prefill Side Lifecycle

### Stage 1 — Arrival and Bootstrap Queue Entry

**`scheduler.py:1935–1991`**, **`scheduler.py:2185–2191`**

The request arrives over ZMQ from the tokenizer manager with three PD-specific fields:
- `bootstrap_host` — IP of the decode worker
- `bootstrap_port` — decode worker's bootstrap port (default 8998)
- `bootstrap_room` — unique integer correlating this request across both fleets

`process_input_requests` → `_add_request_to_queue` routes it to `disagg_prefill_bootstrap_queue.add(req)`.

Inside `add`:
- `create_sender(req)` instantiates a `KVSender` (Mooncake or NIXL) and immediately begins the bootstrap handshake with the decode side at `{bootstrap_host}:{bootstrap_port}` using `bootstrap_room` as the channel ID.
- `_process_req(req)` sets `req.sampling_params.max_new_tokens = 1`. This makes `PrefillAdder`'s memory estimate treat this request as producing only 1 output token (correct, because the decode side generates all real output tokens).
- `req.time_stats.set_prefill_bootstrap_queue_entry_time()` stamps the entry time for observability.

The request is appended to `disagg_prefill_bootstrap_queue.queue`.

**HiCache interaction: none at this stage.**

---

### Stage 2 — Bootstrap Queue Polling

**`prefill.py:307–374`**, called from `event_loop_normal_disagg_prefill:424`

Each scheduler step, before `get_next_batch_to_run`, calls:
```python
self.waiting_queue.extend(
    self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
)
```

`pop_bootstrapped` polls all pending senders via `poll_and_all_reduce_attn_cp_tp_group` (required for TP consensus):

| `KVPoll` state | Meaning | Action |
|---|---|---|
| `Bootstrapping` | Decode hasn't pre-allocated yet | Stay in queue; or promote optimistically if `optimistic_prefill_retries > 0` |
| `WaitingForInput` | Decode has pre-allocated KV and is ready | `finalize_bootstrap(req)` → move to `waiting_queue` |
| `Failed` | Handshake timed out or network error | `handle_bootstrap_failure` → abort HTTP 500 |

`finalize_bootstrap` (prefill.py:262):
1. Allocates a `MetadataBuffer` slot (pool size = `max_running_requests × 2`). If exhausted, stays stuck silently.
2. Reads `decode_prefix_len` from the sender — the decode side may have a radix or HiCache hit and tells prefill "start sending from token N". Sets `req.start_send_idx = decode_prefix_len`.
3. Calls `sender.init(num_pages, metadata_buffer_index)` to arm the RDMA send path.

**HiCache interaction: none on prefill side during bootstrap. The `decode_prefix_len` value comes from the decode side's HiCache query (see decode Stage 2).**

---

### Stage 3 — Prefill Admission

**`schedule_policy.py:170`, `schedule_policy.py:853`**

The request is now in `waiting_queue`, indistinguishable from a normal request. `get_new_batch_prefill` runs the standard admission pipeline:

1. `check_hicache_events()` — polls L1→L2 write completions; unlocks nodes whose async copy to CPU finished. (**HiCache L1→L2**)
2. `policy.calc_priority()` → `match_prefix_for_req()` — LPM lookup into the **prefill-side radix tree**. Populates `prefix_indices` (L1 GPU hits), `host_hit_length` (L2 CPU hits), and triggers `init_load_back` for L2 hits. (**HiCache L1 and L2 read**)
3. `PrefillAdder.add_one_req()` — checks three budgets: `rem_total_tokens` (KV pool minus decode reservation), `rem_input_tokens` (step input cap), `rem_chunk_tokens` (chunked prefill cap). No `new_token_ratio` reservation needed here because `max_new_tokens = 1`.
4. `ScheduleBatch.init_new()` → `prepare_for_extend()` — allocates KV slots for the extend portion; may call `evict_from_tree_cache` to free LRU prefix cache nodes. (**HiCache eviction**)

---

### Stage 4 — Prefill Forward Pass

**`scheduler.py:2996`**, **`prefill.py:438`**

`run_batch(batch)` dispatches to `model_worker.forward_batch_generation(batch)`. The GPU computes attention over the full input sequence. KV for the new tokens is materialized in GPU memory at the slots allocated in Stage 3.

**HiCache interaction: none during the forward pass itself.**

---

### Stage 5 — Prefill Result Processing and KV Transfer

**`prefill.py:497–668`** — `process_batch_result_disagg_prefill`

For each request in the batch:

1. **`maybe_cache_unfinished_req(req, self.tree_cache)`** — inserts the completed prefix into the prefill-side radix tree and triggers `write_backup` to asynchronously copy the node to L2 (CPU). (**HiCache L1→L2 write initiation**)

2. **`self.send_kv_chunk(req, last_chunk=True)`** — initiates RDMA push of KV pages from prefill GPU to decode GPU. Pages are identified by `req.prefix_indices[start_send_idx:]` — starting from `decode_prefix_len` (the decode side's cache hit boundary). Only pages the decode side doesn't already have are sent.

3. The request is appended to `disagg_prefill_inflight_queue`. KV slots remain locked (ref-counted) until transfer completes.

For **chunked prefill** (multi-chunk requests): `send_kv_chunk(req, last_chunk=False)` is called after each intermediate chunk. `maybe_cache_unfinished_req` is called after each chunk. The request stays in the batch across multiple steps until the last chunk.

---

### Stage 6 — Inflight Queue: Waiting for RDMA Completion

**`prefill.py:670–778`** — `process_disagg_prefill_inflight_queue`

Called at the end of each event loop iteration. Polls all in-flight senders via `poll_and_all_reduce_attn_cp_tp_group`:

| `KVPoll` state | Action |
|---|---|
| `WaitingForInput` / `Transferring` | Stay in inflight queue |
| `Success` | `release_kv_cache(req, self.tree_cache)` → unlocks radix tree node; sets `req.finished_reason = FINISH_LENGTH(0)`; streams response to client |
| `Failed` | `release_kv_cache` + `prepare_abort` + HTTP 500 |

**HiCache interaction at `release_kv_cache`:** calls `tree_cache.cache_finished_req(req)` which runs `dec_lock_ref` on the node, making it evictable by future LRU eviction. The actual KV data remains in L1 (GPU) until evicted. (**HiCache L1 unlock**)

**Prefill side is done.** The prefill GPU's KV memory may now be evicted. The response sent to the client is a minimal "prefill complete" signal; actual output tokens come from the decode side.

---

## Decode Side Lifecycle

### Stage 1 — Arrival and Prealloc Queue Entry

**`scheduler.py:2185–2191`**, **`decode.py:524–540`**

The same request arrives at the decode scheduler simultaneously (sent by the router/proxy). It carries the same `bootstrap_host`, `bootstrap_port`, `bootstrap_room`.

`_add_request_to_queue` routes it to `disagg_decode_prealloc_queue.add(req)`.

Inside `add` → `_create_receiver_and_enqueue`:
- Creates a `KVReceiver` connected to `{bootstrap_host}:{bootstrap_port}` using `bootstrap_room`. The receiver registers a pre-allocated GPU memory region that the prefill side will RDMA-write into.
- The `DecodeRequest` wrapper (holding `req` + `kv_receiver`) is appended to `disagg_decode_prealloc_queue.queue`.

---

### Stage 2 — Prealloc Queue: Prefix Matching and KV Slot Reservation

**`decode.py:747–937`** — `pop_preallocated`, called from `process_decode_queue`

This is the decode-side admission gate, called each step. For each queued request:

1. **`_match_prefix_and_lock(req)`** — LPM lookup into the **decode-side radix tree** (only if `--disaggregation-decode-enable-radix-cache`). Calls `match_prefix_for_req` which returns `DecodePrefixMatch` with:
   - `l1_prefix_len` — tokens already on decode GPU (L1 hit)
   - `l2_host_hit_length` — tokens on decode CPU (L2 hit)
   - `l3_storage_hit_length` — tokens in remote storage (L3 hit, only if `enable_decode_hicache`)

   The **combined `decode_prefix_len = l1 + l2 + l3`** is reported back to the prefill side as "start sending from here." (**HiCache L1, L2, L3 read on decode side**)

2. **Budget check** — verifies `required_alloc_tokens + num_reserved_decode_tokens <= full_allocatable_tokens`. If not, breaks the loop. If `retracted_queue` is non-empty, the entire function returns early (no new admissions).

3. **`_pre_alloc(req, prefix_indices, prefix_len, total_prefix_len)`** — allocates GPU KV slots for `origin_input_len - prefix_len` tokens (the non-cached portion). Writes slot indices to `req_to_token_pool`.

4. **`_start_hicache_prefetch(req, prefix_match)`** — if `enable_decode_hicache` and there are L3 hits: fires an async prefetch from L3 storage → L2 CPU. (**HiCache L3→L2 prefetch**)

5. The `DecodeRequest` is moved to `disagg_decode_transfer_queue`.

---

### Stage 3 — Transfer Queue: Waiting for RDMA + HiCache Restore

**`decode.py:1580–1688`** — `pop_transferred`, called from `process_decode_queue`

Each step, `pop_transferred` drives two parallel state machines per in-flight request:

**RDMA poll** (`_poll_with_metadata_gate`): wraps each `kv_receiver` in `HiCacheRestoreGatedKVReceiver`. This gate intercepts `KVPoll.Success` and returns `KVPoll.Transferring` instead if the HiCache restore is still `PENDING`. Ensures RDMA and HiCache restore both complete before the request graduates.

**HiCache local restore** (`_process_hicache_local_restores`): for requests with L2/L3 prefix hits, drives the `load_back` state machine:
1. Waits for L3 prefetch to drain to L2 (`check_prefetch_progress`).
2. Re-runs `match_prefix_for_req` to get the current device state.
3. Calls `tree_cache.init_load_back(...)` to start a DMA from CPU (L2) → GPU (L1) for the missing prefix portion. (**HiCache L2→L1 loadback**)
4. Sets `hicache_restore_status = READY` when DMA completes (or `FAILED` on error).

Only when both RDMA poll = `Success` AND HiCache restore = `READY` does the request graduate:
- `_commit_transfer_to_req(decode_req)` — wires up the received KV indices into the request's `prefix_indices`, sets `extend_input_len`, calls `maybe_cache_unfinished_req`. (**HiCache: insert received KV into decode-side radix tree**)
- The raw `Req` is moved to `waiting_queue`.

On failure: `release_kv_cache(req, tree_cache, is_insert=False)` — releases the pre-allocated slots without inserting into the radix tree.

---

### Stage 4 — Decode Admission

**`decode.py:1806`** — `get_new_prebuilt_batch`

Requests in `waiting_queue` are admitted via `get_new_prebuilt_batch` (not `get_new_batch_prefill`):
- No `PrefillAdder` — KV is already present in GPU memory.
- Only constraint: `max_running_requests - current_running_bs`.
- Creates a `ScheduleBatch` in DECODE mode directly.

**`process_prebuilt`** (`decode_schedule_batch_mixin.py:112`):
- Calls `maybe_cache_unfinished_req(req, self.tree_cache)` — inserts the full prefix KV into the decode-side radix tree, enabling future requests with the same prompt to get a decode-side L1 cache hit. (**HiCache: L1 cache population on decode side**)
- Sets up `input_ids`, `seq_lens`, `out_cache_loc` tensors for the first decode step.

The prebuilt batch is then merged into `running_batch` via `merge_batch`.

---

### Stage 5 — Decode Loop

**`scheduler.py:2850`** — `update_running_batch` (identical to non-PD mode)

Each decode step:
1. `filter_batch()` — removes finished requests from `running_batch`.
2. `flush_write_through_acks()` — releases L2 node locks for KV that has been successfully written to CPU. (**HiCache L2 write completion**)
3. `check_decode_mem()` — verifies `available_kv_slots >= len(reqs)`. Calls `evict_from_tree_cache` if needed to free LRU prefix cache nodes. (**HiCache eviction**)
4. `[if OOM] retract_decode()` — evicts decode requests back to `disagg_decode_prealloc_queue.retracted_queue` (not `waiting_queue`). The request's KV may be offloaded to CPU.
5. `prepare_for_decode()` → `alloc_for_decode()` — allocates 1 new KV slot per request for the next generated token.
6. `run_batch()` — GPU forward pass.
7. `process_batch_result_decode()` — appends generated token; on finish: `release_kv_cache` → `cache_finished_req` → inserts completed KV into decode-side radix tree. (**HiCache: L1 cache population**)

Generated tokens are streamed back to the client via `output_streamer.stream_output`.

---

## HiCache Interaction Map

| Stage | Side | HiCache Tier | Operation | Code |
|---|---|---|---|---|
| Prefill admission | Prefill | L1→L2 | `check_hicache_events`: poll async write completions | `scheduler.py` via `_get_new_batch_prefill_raw` |
| Prefill admission | Prefill | L1, L2 | `match_prefix_for_req`: LPM lookup; `init_load_back` for L2 hits | `schedule_policy.py:80` |
| Prefill admission | Prefill | L1 | `evict_from_tree_cache` in `prepare_for_extend` | `schedule_batch.py:1839` |
| Prefill result | Prefill | L1→L2 | `maybe_cache_unfinished_req` → `write_backup`: async copy to CPU | `prefill.py:582` |
| Prefill inflight done | Prefill | L1 | `release_kv_cache` → `dec_lock_ref`: make node evictable | `prefill.py:715` |
| Decode prealloc | Decode | L1, L2, L3 | `_match_prefix_and_lock`: full 3-tier lookup; `l3_storage_hit_length` query | `decode_hicache_mixin.py:61` |
| Decode prealloc | Decode | L3→L2 | `_start_hicache_prefetch`: fire L3 storage prefetch | `decode_hicache_mixin.py:99` |
| Decode transfer | Decode | L2→L1 | `_try_hicache_queue_load_back` → `init_load_back`: DMA L2→L1 | `decode_hicache_mixin.py:181` |
| Decode transfer done | Decode | L1 | `_commit_transfer_to_req` → `maybe_cache_unfinished_req`: insert received KV | `decode.py:1643` |
| Decode prebuilt | Decode | L1 | `process_prebuilt` → `maybe_cache_unfinished_req`: insert full prefix | `decode_schedule_batch_mixin.py:121` |
| Decode step | Decode | L1→L2 | `flush_write_through_acks`: release L2 write locks | `scheduler.py:2850` |
| Decode step | Decode | L1 | `evict_from_tree_cache` in `check_decode_mem` | `schedule_batch.py:2289` |
| Decode finish | Decode | L1 | `release_kv_cache` → `cache_finished_req`: insert completed KV, dec_lock_ref | `batch_result_processor.py:848` |

---

## State Machine Summary

```
CLIENT
  │
  ▼ (prefill side receives)                   (decode side receives simultaneously)
  │                                                         │
  ▼                                                         ▼
[PrefillBootstrapQueue]                       [DecodePreallocQueue.queue]
  │                                                         │
  │  KVPoll.WaitingForInput                   match_prefix (HiCache L1/L2/L3 lookup)
  │  (decode signaled ready)                  pre-alloc KV slots
  │                                           _start_hicache_prefetch (L3→L2)
  ▼                                                         │
[waiting_queue] ←── get admitted                            ▼
  │                                           [DecodeTransferQueue.queue]
  ▼                                                         │
[prefill batch]                                RDMA transferring KV
  │                                           _process_hicache_local_restores (L2→L1)
  │  GPU forward (attention over input)                     │
  │                                            KVPoll.Success + HiCache READY
  │  maybe_cache_unfinished_req (HiCache L1→L2)             │
  │  send_kv_chunk (RDMA push to decode)                    ▼
  │                                           [waiting_queue] (decode side)
  ▼                                                         │
[inflight_queue]                              get_new_prebuilt_batch
  │                                           process_prebuilt (HiCache insert)
  │  KVPoll.Success                                         │
  │  release_kv_cache (HiCache dec_lock_ref)                ▼
  │  stream_output → CLIENT                 [running_batch] (decode loop)
  │                                                         │
  ▼                                           alloc 1 slot/step, GPU forward
DONE (prefill side)                           stream tokens → CLIENT
                                                            │
                                              release_kv_cache (HiCache insert)
                                                            │
                                                          DONE
```

# SGLang Scheduler: Core Functions, Roles, and Relationships

## Architecture in One Sentence

The scheduler is a single-threaded event loop that alternates between two decisions per step: **admit requests from `waiting_queue` into a prefill batch**, or **advance the persistent decode batch** — never both (unless mixed-chunk mode). All state lives in `running_batch` (decode pool) and `waiting_queue` (admission queue).

---

## Layer 1: Event Loop (the outer shell)

**Files:** `scheduler.py:1432`, `scheduler.py:1459`

```
event_loop_normal / event_loop_overlap
```

Both loops do the same four things per iteration:

1. `recv_requests()` → `process_input_requests()`
2. `get_next_batch_to_run()` — decides **what** to run
3. `run_batch(batch)` — dispatches to GPU
4. `process_batch_result(batch, result)` — handles outputs, updates cache

`event_loop_overlap` pipelines step 3 of iteration N with step 4 of iteration N-1: the GPU runs N while the CPU processes N-1's results, hiding CPU overhead.

---

## Layer 2: Batch Decision

**File:** `scheduler.py:2425`

```
get_next_batch_to_run
```

**Role:** The top-level scheduler oracle. Decides whether to run prefill or decode this step.

**Internal sequence:**

```
1. merge last prefill batch → running_batch   (graduate finished-prefill reqs into decode)
2. new_batch = get_new_batch_prefill()
3. if new_batch:  return new_batch                              ← PREFILL wins
   else:          return update_running_batch(running_batch)   ← DECODE
```

Nothing else in the scheduler has any say in the prefill-vs-decode decision. This is the single point of authority.

---

## Layer 3A: Prefill Path

```
get_new_batch_prefill                          scheduler.py:2553
  └── _get_new_batch_prefill_raw               scheduler.py:2573
        ├── check_hicache_events()             poll L1→L2 write completions
        ├── policy.calc_priority()             sort waiting_queue by prefix hit length
        │     └── match_prefix_for_req()       per-req LPM lookup into radix tree
        ├── PrefillAdder loop
        │     └── adder.add_one_req()          per-req budget check + L2 load trigger
        │           └── init_load_back()       allocate L1 slots for L2-hit prefix
        ├── waiting_queue[:] = ...             dequeue admitted reqs
        └── ScheduleBatch.init_new()           create the batch object
              └── prepare_for_extend()         allocate KV slots, build input tensors
```

### When `get_new_batch_prefill` returns `None`

The function returns `None` — falling through to the decode path — in five cases:

| Cause | Code location |
|---|---|
| `waiting_queue` is empty | `scheduler.py:2590` |
| `running_batch.batch_is_full` sticky flag is set | `scheduler.py:2590` |
| Max concurrent request slot count reached | `scheduler.py:2601` |
| First req hits `NO_TOKEN` (KV pool exhausted) | `schedule_policy.py:894` |
| All reqs blocked on L3 prefetch (HiCache) | `scheduler.py:2689` |

`batch_is_full` is a sticky flag — once set, it persists across steps until a running request finishes (`filter_batch` shrinks the batch) or `retract_decode` frees memory.

### Key sub-functions

**`policy.calc_priority`** (`schedule_policy.py:170`): calls `match_prefix_for_req` for every waiting request to discover L1/L2 prefix hits, then sorts by LPM (longest prefix match) or DFS-weight. This is the most expensive CPU operation per step — O(queue length) radix tree lookups.

**`PrefillAdder.add_one_req`** (`schedule_policy.py:853`): the per-request admission gate. Checks three budgets in order:

| Budget | Variable | Result if exceeded |
|---|---|---|
| Total KV slots (input + max_new + page) | `rem_total_tokens` | `NO_TOKEN` — stop loop |
| Total input tokens this prefill step | `rem_input_tokens` | `OTHER` — skip req, try next |
| Per-request chunk cap | `rem_chunk_tokens` | split into chunked prefill |

`rem_total_tokens` = `available_kv + evictable_kv − Σ(remaining_decode_tokens × new_token_ratio)`. The `new_token_ratio` term is what reserves headroom for decode.

**`prepare_for_extend`** (`schedule_batch.py:1839`): materializes the batch into GPU tensors. Computes `input_ids` (tokens after prefix), allocates new KV slots for the extend portion via `alloc_for_extend → evict_from_tree_cache`, builds `seq_lens`, `prefix_lens`, `out_cache_loc`.

---

## Layer 3B: Decode Path

```
update_running_batch                           scheduler.py:2850
  ├── filter_batch()                           remove finished requests
  ├── flush_write_through_acks()               [HiCache] release L2-written node locks
  ├── check_decode_mem()                       can we fit 1 new token per req?
  │     └── evict_from_tree_cache()            evict LRU prefix cache nodes if needed
  ├── [if OOM] retract_decode()               kick requests back to waiting_queue
  └── prepare_for_decode()                     alloc 1 KV slot per req, set DECODE mode
        └── alloc_for_decode()                 common.py:438
```

**`check_decode_mem`** (`schedule_batch.py:2289`): checks `available_kv_slots >= len(running_batch.reqs)`. Always tries to evict unlocked prefix cache nodes before declaring OOM.

**`retract_decode`** (`schedule_batch.py:2302`): OOM recovery. Sorts running requests by `(most output_ids, shortest input_ids)` — evicting the cheapest to re-prefill first — and pops them back to `waiting_queue` until memory is sufficient. Always keeps at least 1 request. Updates `new_token_ratio` upward after eviction to be more conservative in future prefill admissions.

**`prepare_for_decode`** (`schedule_batch.py:2443`): allocates exactly 1 new KV slot per request (the slot for the next generated token). This is the only GPU memory allocation in the entire decode path.

---

## Layer 4: Forward Pass

**File:** `scheduler.py:2996`

```
run_batch
  ├── [non-overlap] model_worker.forward_batch_generation(batch)
  └── [overlap]     schedule_stream sync → forward_stream
                    model_worker.forward_batch_generation(batch)
                    future_map.publish(...)   ← notify schedule stream of new seq_lens
```

**Role:** Pure dispatch. Sends the batch to the model worker (which manages TP/PP workers). Returns a `GenerationBatchResult` containing `next_token_ids` and `logits_output`. No scheduling logic lives here.

---

## Layer 5: Result Processing

**File:** `batch_result_processor.py:178, 590`

```
process_batch_result                           scheduler.py:3201
  ├── [EXTEND] process_batch_result_prefill    batchproc.py:178
  │     ├── for each req: append next_token_id
  │     ├── req.update_finish_state()
  │     ├── if finished:   release_kv_cache()   → cache_finished_req → radix insert
  │     └── if unfinished: maybe_cache_unfinished_req() → insert prefix → write_backup
  │
  └── [DECODE] process_batch_result_decode     batchproc.py:590
        ├── free_group_begin()                 defer KV free until end
        ├── for each req: append next_token_id
        ├── if finished:   release_kv_cache()  → radix insert
        └── free_group_end()                   atomically release KV slots
```

**`release_kv_cache`** (`common.py:481`): calls `tree_cache.cache_finished_req(req)`, which inserts the completed request's tokens into the radix tree and `dec_lock_ref` to make the prefix evictable. This is the only point where completed KV data enters the prefix cache.

**`maybe_cache_unfinished_req`** (`common.py:46`): called after each chunked prefill chunk. Inserts the partial prefix into the radix tree and triggers `write_backup` to start promoting the node to L2 (HiCache).

---

## Layer 6: Request Ingestion

**File:** `scheduler.py:1550`

```
process_input_requests
  └── _request_dispatcher(recv_req)            route by request type
        └── _add_request_to_queue(req)         scheduler.py:2177
              ├── [normal mode]   waiting_queue.append(req)
              ├── [PD prefill]    disagg_prefill_bootstrap_queue.add(req)
              └── [PD decode]     disagg_decode_prealloc_queue.add(req)
```

Requests arrive from the tokenizer manager over ZMQ. `process_input_requests` runs at the top of every loop iteration before `get_next_batch_to_run`. A request always sits in `waiting_queue` for at least one full iteration before it can be scheduled.

---

## Decode Batch Heuristics

There is no explicit "form decode batch" step. `running_batch` is persistent — requests are merged in after prefill and filtered out when finished. The heuristics govern **how many new requests can enter** and **which ones get evicted**.

### Heuristic 1: `new_token_ratio` — Pessimistic Decode Reservation

**File:** `new_token_ratio_tracker.py`, `schedule_policy.py:509`

When admitting a prefill request, the scheduler subtracts a reservation for each currently decoding request:

```
reservation per decode req = (max_new_tokens − already_generated) × new_token_ratio
rem_total_tokens = available_kv + evictable_kv − Σ(reservations)
```

**Adaptive behavior:**
- Starts at `SGLANG_INIT_NEW_TOKEN_RATIO × schedule_conservativeness`
- **Decays each decode step** when no OOM occurs (more aggressive over time)
- **Resets upward** after `retract_decode` fires, recalibrated to actual decode progress

### Heuristic 2: Retraction Sort — "Cheapest to Re-prefill First"

**File:** `schedule_batch.py:2314`

```python
sorted_indices.sort(
    key=lambda i: (len(output_ids), -len(origin_input_ids)),
    reverse=True,
)
```

Evict requests with **most output tokens** and **shortest input** first. Rationale: they've consumed a decode slot the longest but are cheap to re-prefill (short prefix to recompute).

### Heuristic 3: `batch_is_full` Sticky Flag — Hysteresis

When `NO_TOKEN` is returned from `add_one_req`, `batch_is_full = True` is set. This persists across steps, causing the scheduler to skip the entire `calc_priority + PrefillAdder` loop until the decode batch shrinks. Prevents wasted CPU on repeated failed admission attempts.

---

## Relationship Diagram

```
                     ┌──────────────────────────────────────────┐
                     │         event_loop (per iteration)        │
                     └──────────────────────────────────────────┘
                            │                       │
                 process_input_requests       get_next_batch_to_run
                            │                       │
                   _add_request_to_queue      ┌─────┴──────┐
                            │                 │            │
                    waiting_queue    get_new_batch    update_running_batch
                                      _prefill()           │
                                           │          ├─ filter_batch
                                  ┌────────┴──┐       ├─ flush_write_through_acks
                            calc_priority  PrefillAdder    ├─ check_decode_mem
                                   │        loop       │     └── evict_from_tree_cache
                            match_prefix  add_one_req  ├─ retract_decode
                                            │          └─ prepare_for_decode
                                       init_load_back        └── alloc_for_decode
                                            │
                                    ScheduleBatch.init_new
                                            │
                                    prepare_for_extend
                                            │
                                       ┌────┴────┐
                                    run_batch     │
                                    (GPU fwd)     │
                                       │          │
                               process_batch_result
                               ├─ [EXTEND] release_kv_cache
                               │           maybe_cache_unfinished_req
                               └─ [DECODE] release_kv_cache
                                              │
                               cache_finished_req → radix tree insert
                               write_backup       → L1→L2 async copy (HiCache)
```

---

## Three Critical Invariants

1. **One batch type per step.** `get_next_batch_to_run` returns either a prefill batch OR advances the decode batch — never both (except mixed-chunk mode).

2. **The decode batch is never constructed.** `running_batch` is persistent. Requests enter via `merge_batch` after prefill completes; they leave via `filter_batch` when finished.

3. **KV memory is allocated in exactly two places.** `prepare_for_extend` (new tokens during prefill) and `alloc_for_decode` (one slot per request during decode). Eviction is triggered lazily inside both via `evict_from_tree_cache`.

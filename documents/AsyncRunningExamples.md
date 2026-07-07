# Async / Parallel Running Examples — Selection & Feasibility Map

Research asset for **[#460](https://github.com/Lizn-zn/VeriTile/issues/460)** (parent map
**[#458](https://github.com/Lizn-zn/VeriTile/issues/458)**, executes the first "Next step" of
**[#409](https://github.com/Lizn-zn/VeriTile/issues/409)**). It picks the minimal
async/parallel kernels VeriTile adopts as the **running examples for property 3
(parallel/asynchronous correctness)** and annotates, construct by construct, what is
*representable* vs *blocked-by-gap* against VeriTile today.

This **blocks [#459](https://github.com/Lizn-zn/VeriTile/issues/459)** (async modeling
substrate: Lean state-machine vs TLA): the constructs the chosen examples require are the
input to that decision. It also grounds the async-implementation fog in #458's
"Not yet specified".

Terminology follows [`CONTEXT.md`](../CONTEXT.md) (world layer, neutral core, region,
memory-space tag). Current-state facts are cross-checked against
[`ArchitectureHandoff.md`](./ArchitectureHandoff.md) and
[`ConcurrencySemantics.md`](./ConcurrencySemantics.md).

---

## 1. Selection criterion

#409's brief: *pick a minimal overlap kernel that exercises compute–communication overlap
without dragging in the whole collective zoo.* Concretely a good running example should:

1. exercise **at least one comm primitive** (put/get/signal/wait) against **peer/remote
   memory** — the multi-rank core that single-GPU VeriTile cannot state today;
2. exercise **compute–communication overlap** (the property-3 payload) — a `dot`/GEMM whose
   tiles are consumed as remote data arrives;
3. stay **small and self-contained** — one collective, no MoE routing, no task-graph
   runtime, no full attention — so the line-by-line feasibility map is tractable and the
   #459 modeling decision is made against the smallest sufficient construct set.

## 2. Survey of the candidate sources

| Source | What it is | Verdict for a v1 running example |
| --- | --- | --- |
| **Triton-distributed — `mega_triton_kernel/kernels/`** ([link](https://github.com/ByteDance-Seed/Triton-distributed/tree/main/python/triton_dist/mega_triton_kernel/kernels)) | A **task-graph megakernel** runtime: `task_context.py`, `prefetch.py`, `barrier.py`, `allreduce.py`, `flash_attn.py`, `linear.py`, `mlp_fc1.py`, `norm.py`, … A persistent kernel schedules sub-tasks (linear, norm, allreduce, attention) from an in-GPU task queue. | **Rejected as the minimal case.** The overlap here is *emergent from a scheduler*, not from one legible kernel: correctness would drag in the task-queue runtime, persistent-block scheduling, and a barrier framework — the opposite of minimal. Keep as a **later stress target** once the substrate exists; it is the "does this scale past a toy" check, not the running example. |
| **TileScale — `examples/distributed/`** ([link](https://github.com/tile-ai/tilescale/tree/main/examples/distributed)) | A graded ladder of TileLang distributed examples: `example_simple_shift.py`, `example_allgather.py`, `example_allgather_gemm.py`, **`example_allgather_gemm_overlapped.py`**, `example_gemm_rs_overlapped.py`, `example_all_to_all.py`, Cannon/SUMMA, `primitives/`, `deepseek_deepep/`. | **Selected.** This is where the *minimal* rungs live. TileLang exposes the comm primitives as first-class intrinsics (`T.putmem_nbi_block`, `T.putmem_signal_nbi_block`, `T.signal_wait_until`, `T.get_pe`), so each construct maps cleanly onto a candidate neutral-core node — exactly the granularity #409's gap table wants. |
| **DeepEP-in-TileScale** ([commit `eccbd61`](https://github.com/tile-ai/tilescale/commit/eccbd61c37671beccb5208e65088a49987496dc2), `deepseek_deepep/`) | MoE expert-parallel **dispatch/combine all-to-all** (DeepEP), with token routing, variable per-expert counts, and IBGDA/low-latency NVSHMEM paths. | **Rejected as the minimal case.** This *is* the collective zoo: all-to-all + data-dependent routing + ragged layouts. Every gap the minimal cases expose is present here too, plus routing/layout complexity that is orthogonal to property 3's core. Keep as the **eventual async/distributed validation target** named in #458, not the running example. |

**Decision.** Adopt **two** TileScale examples, chosen as the *floor* and the *canonical
overlap*:

- **Example A — `simple_shift`** (ring `put`). The absolute minimum: one NVSHMEM `put` to a
  peer, no compute, no signal. Isolates the multi-rank + peer-addressing + put gaps with
  nothing else attached. This is the smallest kernel that VeriTile *cannot state at all
  today*, so it is the cleanest unit test for "did the world layer land".
- **Example B — `allgather_gemm_overlapped`** (AllGather→GEMM overlap). The canonical
  compute–communication overlap #409 names. Adds `put+signal` / `wait`, tile-wise GEMM
  consuming remote tiles as they arrive, and double-buffered pipelining — i.e. every
  property-3 construct, still in one readable kernel.

Rungs in between (`example_allgather`, non-overlapped `example_allgather_gemm`) are noted as
**intermediate milestones**: same gaps as B minus the overlap/pipelining, useful for staging
the implementation but not needed as separate running examples.

---

## 3. Feasibility legend

Annotation against the **#409 gap checklist** (multi-rank · NVSHMEM put/get ·
signal/fence/wait · double-buffering):

- ✅ **Representable** — a neutral-core node / semantics exists today (grep-able identifier
  cited).
- 🟡 **Partial** — syntactically expressible but the semantics that makes it *correct* is
  missing (typically the memory-space tag or scope discipline).
- ❌ **Blocked-by-gap** — no node, no semantics; requires a new foundation layer. The
  "Gap" column names which of #409's four gaps it belongs to.

Current-state anchors (from `ArchitectureHandoff.md` §3/§4/§9, `ConcurrencySemantics.md`):
the neutral core is single-rank; `BlockState.mem : RegionName → Nat → MemCell` is one flat
local region namespace (no peer/remote); `Op` has `load` but **no `put`/`get`/`signal`/
`wait`**; `exec` steps one program's statements **sequentially** with no scheduler;
`Concurrency/Trace.lean` is a **placeholder vocabulary** (`MemoryEvent`, `HappensBefore`,
`PermissionModel`) not wired into `exec`; the only concurrency theorem surface is
`ConcurrentTrace.RefinesSequential` (equal final memory, explicitly weaker than
bisimulation); atomics are local, single-cell, sequentially linearized only.

---

## 4. Example A — `simple_shift` (ring put)

TileLang kernel body (verbatim, `examples/distributed/example_simple_shift.py`):

```python
def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        mype = T.alloc_local([1], "int32")
        npes = T.alloc_local([1], "int32")
        peer = T.alloc_local([1], "int32")
        mype[0] = T.get_pe()                 # NVSHMEM my_pe
        npes[0] = T.get_pe_num()             # NVSHMEM n_pes
        peer[0] = (mype[0] + 1) % npes[0]    # ring neighbour
        T.putmem_nbi_block(T.address_of(B[0, 0]),
                           T.address_of(A[0, 0]),
                           block_M * block_N * 2,   # nbytes (fp16)
                           peer[0])                 # write A → peer's B
```

Semantics: every PE copies its local `A` tile into the `B` region of its ring successor
`(pe+1) % npes`. A pure one-shot `put`; no compute, no synchronization.

### Feasibility annotation

| Construct | Status | Gap (#409) | Rationale |
| --- | --- | --- | --- |
| `T.Kernel(grid) as (bx, by)`, `ceildiv` grid | ✅ | — | `Launch/Grid.lean`: `Grid`, `GridIndex`, `program_id`; `cdiv` helper. ND grid already modeled. |
| `T.alloc_local` scalar temps, `%`, `+` | ✅ | — | Local registers + integer `Op` arithmetic (`add/mod`) in the neutral core. |
| `T.get_pe()` | ❌ | **Multi-rank** | No rank identity. Neutral core is single-rank; there is no `WORLD_SIZE`/`my_pe`. Needs the **world layer** (rank → per-rank memory) above `BlockState`. |
| `T.get_pe_num()` | ❌ | **Multi-rank** | Same; no world-size notion. (Cf. `num_programs`, itself a known grid gap — but rank count ≠ grid size.) |
| `peer = (mype+1) % npes` as an **address selector** | ❌ | **Multi-rank** | The arithmetic is fine; using the result to *name a remote region* is not — there is no peer-addressing map. |
| `T.address_of(A[0,0])` / `B[0,0]` | 🟡 | Multi-rank | A *local* address is `Op.ref`/`ptrAdd`; a *symmetric-heap* address that resolves per-rank does not exist. |
| `T.putmem_nbi_block(dst, src, nbytes, peer)` | ❌ | **NVSHMEM put/get** | The core has `load`/`store` on **local** regions only. No `put` node; no remote write; no `nbytes`-typed bulk transfer to a peer. |
| `_nbi_` (non-blocking, completes at next fence/quiet) | ❌ | signal/fence/wait + overlap | No async token, no fence/quiet, no in-flight-op state. `exec` is sequential; there is nothing for "non-blocking" to mean. |

**Roll-up:** `simple_shift` is **entirely blocked** on the multi-rank world layer plus a `put`
node. It needs *none* of signal/wait, GEMM, or pipelining — which is exactly why it is the
minimal first target: landing the world layer + `put` + a peer-addressing map makes it
*stateable* (the property is "after the shift, each rank's `B` equals its predecessor's `A`"),
and it can be checked with **safety only** (equal final memory) — no liveness, no
happens-before. This is the smallest evidence for #459 that a **plain state-machine** may
suffice for the floor.

---

## 5. Example B — `allgather_gemm_overlapped` (AllGather → GEMM overlap)

The canonical compute–communication overlap kernel #409 names. Structure (distilled from
`example_allgather_gemm_overlapped.py`): each rank owns a shard of `A`; an all-gather
publishes every shard into a symmetric `A_ag` buffer while a tiled GEMM consumes each
`A_ag` tile the moment its arrival `signal` fires, so communication of tile `k+1` overlaps
computation on tile `k`.

Representative kernel-body constructs (verbatim intrinsics from the example):

```python
mype[0] = T.get_pe(); npes[0] = T.get_pe_num(); peer[0] = ...
# producer: publish my shard to peer's symmetric A_ag + raise arrival signal
T.copy(A[by*block_M, bx*block_K], A_shared)
T.copy(A_shared, A_ag[mype[0]*M, bx*block_K])
T.putmem_signal_nbi_block(T.address_of(A_ag[mype[0]*M, 0]),
                          T.address_of(A[0, 0]), block_M*block_K*2,
                          T.address_of(signal[k]), k+1, 9,  # sig_op = NVSHMEM_SIGNAL_SET
                          peer[0])
# consumer: wait for tile k, then GEMM it
T.signal_wait_until(T.address_of(signal[k]), 0, k+1)   # cmp EQ, value k+1
T.copy(A_ag[bk*M, k*block_K], A_shared)                # double-buffered stage
T.copy(B[k*block_K, bx*block_N], B_shared)
T.gemm(A_shared, B_shared, C_local)                    # accumulate
T.copy(C_local, C[bk*M, bx*block_N])
T.clear(C_local)
```

(Some TileScale variants drive the all-gather from the host with a copy-engine
`cp_engine_producer_all_gather_full_mesh_pull` + `cuStreamWriteValue32` signals and a
`T.wait_eq(signal_buffer[...], 1)` in-kernel wait; the in-kernel `putmem_signal` form above
is the self-contained one and the better modeling target.)

### Feasibility annotation

| Construct | Status | Gap (#409) | Rationale |
| --- | --- | --- | --- |
| `T.gemm(A_shared, B_shared, C_local)` accumulate | ✅ | — | `Op.dot`; tile-level GEMM with an accumulator is core-modeled (§3). |
| Tiled loop over `k`, `T.clear`, index arithmetic | ✅ | — | `forRange`/`forRangeDyn`, register clear, integer `Op`s. |
| `C[bk*M, bx*block_N] ← C_local` (result store) | ✅ | — | `store` to a local region; framing via `WriteFootprint`. |
| `T.copy(global ↔ shared)` (`A_shared`, `B_shared`) | 🟡 | double-buffering | Expressible as load/store today, **but** shared memory has no **memory-space tag** and no scope semantics. Correctness of reuse across pipeline stages depends on that tag + a barrier discipline that are Phase-2/absent. |
| `T.get_pe` / `T.get_pe_num` / `peer` | ❌ | **Multi-rank** | Same as Example A — world layer absent. |
| `A_ag` symmetric (all-gather) buffer, `A_ag[bk*M, …]` peer slice | ❌ | **Multi-rank** | Symmetric heap indexed by rank does not exist; the flat local namespace cannot name "rank `bk`'s slice". |
| `T.putmem_signal_nbi_block(dst, src, nbytes, sig, val, op, peer)` | ❌ | **NVSHMEM put/get** *and* **signal** | Fused remote-write **+** signal-set. Two missing primitives in one node: no remote `put`, and no cross-rank `signal` write (local atomics are single-cell, sequentially linearized — not a cross-rank release). |
| `sig_op = 9` (`NVSHMEM_SIGNAL_SET`) as a **release** | ❌ | signal/fence/wait | The signal is the *release* edge that makes the published tile visible-before-wait. No happens-before / release-acquire relation exists (`HappensBefore` in `Trace.lean` is an un-wired placeholder). |
| `T.signal_wait_until(sig, EQ, k+1)` / `T.wait_eq(...)` | ❌ | **signal/fence/wait** | No blocking/wait primitive and no cross-program synchronization; `exec` never yields. The *acquire* edge pairing with the release above. |
| `_nbi_` non-blocking **overlap** of put(k+1) with gemm(k) | ❌ | double-buffering + interleaving | The whole point of the kernel. Needs an **executable interleaving scheduler** — `exec` is strictly sequential per program; `Concurrency/Trace.lean` is a placeholder, not wired in. This is #409's "step 3 = research-grade". |
| Double-buffered `A_shared` (stage `k` computes while stage `k+1` loads) | ❌ | **double-buffering** | No async token, no pipeline/stage state, no shared-memory model. Even the safety property ("gemm reads the correct version of the tile") is unstatable without visibility + the acquire/release pair. |

**Roll-up:** `allgather_gemm_overlapped` exercises **all four** #409 gaps. Its compute half
(`gemm`, tiling, result store, index math) is ✅ **already modeled today**; everything that
makes it *distributed and overlapped* — world layer, symmetric heap, `put`, `signal`,
`wait`, interleaving, double-buffering — is ❌. Critically, unlike Example A, its correctness
predicate **references visibility**: "the GEMM on tile `k` reads the version of the remote
tile that the matching `signal` released." That is a happens-before / release-acquire claim,
not merely equal-final-memory — so it is the example that will tell #459 whether a plain
state-machine suffices or a temporal/ordering layer is genuinely forced.

---

## 6. Consolidated gap map (both examples vs #409's four gaps)

| #409 gap | `simple_shift` | `allgather_gemm_overlapped` | Foundation layer required |
| --- | --- | --- | --- |
| **Multi-rank / WORLD_SIZE / symmetric memory** | ❌ (get_pe, peer, put target) | ❌ (get_pe, `A_ag` symmetric slice) | **World layer** over `BlockState` (rank → per-rank memory) + peer-addressing map. |
| **NVSHMEM put/get** | ❌ (`putmem_nbi_block`) | ❌ (`putmem_signal_nbi_block`) | New neutral-core comm node(s): `put`/`get` on peer regions. Union policy: one closed GADT. |
| **signal / fence / wait** | — (not used) | ❌ (`putmem_signal`, `signal_wait_until`) | `signal`/`wait`/`fence` nodes **+** a happens-before / release-acquire relation (wire up `Trace.lean`). |
| **Double-buffering / async / interleaving** | 🟡 (`_nbi_` only, no consumer) | ❌ (overlap is the point) | Executable interleaving scheduler + memory-consistency model + async-token/pipeline state + memory-space tags. #409's research-grade "step 3". |

Also surfaced, orthogonal to the four: **memory-space tags** (`shared`) — 🟡 today,
Phase-2 in #465 — gate `T.copy`-to-shared and double-buffer correctness even before the
comm layer.

## 7. What this feeds

- **Blocks #459 (modeling substrate).** The two examples bracket the decision:
  - `simple_shift` needs **safety only** (equal final memory), which `RefinesSequential`
    already gestures at → evidence a **plain Lean state-machine** covers the floor.
  - `allgather_gemm_overlapped`'s correctness is a **visibility/ordering** claim
    (signal establishes happens-before; overlap must read the released version) → this is
    the concrete case #459 must weigh for whether a **TLA-style / temporal-ordering layer**
    is needed, or a small-step interleaving semantics + a release-acquire relation suffices.
    v1 likely needs **safety over interleaved traces**, not liveness/fairness — neither
    example requires eventual-delivery/fairness to state its equal-result property.
- **Grounds the async-implementation stack** (#458 "Not yet specified"): the four layers
  #409 names map onto the gap-map rows above, and the two examples give the substrate its
  first regression targets (floor + canonical overlap).
- **Sequencing hint for the impl tickets:** world layer + `put` (unlocks `simple_shift`,
  safety-only) → `signal`/`wait` + happens-before (unlocks non-overlapped
  `allgather_gemm`) → interleaving scheduler + double-buffering (unlocks the overlapped
  variant). Each rung is an independently-checkable milestone.

## 8. Deferred (kept as later validation targets, not v1 running examples)

- **Triton-distributed `mega_triton_kernel`** — task-graph megakernel; the "does it scale
  past a toy" stress target once the substrate exists.
- **DeepEP-in-TileScale (`deepseek_deepep`)** — MoE dispatch/combine all-to-all; the
  eventual async/distributed *validation* case named in #458, gated on routing + ragged
  layout support well beyond property 3's core.
- **`example_gemm_rs_overlapped` (GEMM→ReduceScatter)** — the natural second overlap shape
  (reduce on the consumer side); a good third example once AllGather-GEMM is proven.

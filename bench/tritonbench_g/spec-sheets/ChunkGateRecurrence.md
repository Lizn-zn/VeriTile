# Spec sheet — `bench/tritonbench_g/chunk_gate_recurrence/ChunkGateRecurrence.lean`

**Python source:** `bench/tritonbench_g/chunk_gate_recurrence/chunk_gate_recurrence.py`

## Public theorem: `chunk_gate_recurrence_forward_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Genuine dimension-general forward output summary** for
`chunk_gate_recurrence.py`'s `_fwd_recurrence`.

Both public forward surfaces (`last_kv` present/absent) lower to the algorithm
layer for arbitrary symbolic dimensions, and every observable forward writeback
realizes the genuine closed form `fwdClosed` (`seed · ∏ d + Σ S · ∏ d`) over the
*input* regions `S`, `D`, `last_kv` — never a read-back of the kernel's own
output `O`:

* the initial store (`last_kv` branch) realizes `fwdClosed(0) = last_kv`;
* the initial store (zero branch) realizes `fwdClosed(0) = 0`;
* one loop body realizes `fwdClosed(t_rel+1)` from the materialized previous
  state `AccPrev = fwdClosed(t_rel)`.

Side conditions are honest: per-store output-offset injectivity
(`hOutInj0`, `hOutInjStep`) and the carry invariant `hAcc*`. The cross-chunk
fold over `range(NUM_BLOCK-1)` is the trusted boundary. -/
```
</details>

**Statement:**
```lean
theorem chunk_gate_recurrence_forward_output_summary_general
    (AccPrev S D O LastKv : RegionName) (HAS_LAST_KV : Bool) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    (hOutInjStep : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx) :
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.true NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.false NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K
          D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (t_rel + 1) idx))
```

**Assumptions / layout contracts:**
- `hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)`
- `fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx`
- `hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx`

**Closed-form spec defs (transitive):** `outOffset`, `forwardStepTileOffset`, `accOffset`, `fwdClosed`, `chunk_gate_recurrence_fwd_surface`, `chunk_gate_recurrence_initial_last_kv_store_slice`, `chunk_gate_recurrence_initial_zero_store_slice`, `chunk_gate_recurrence_forward_step_store_slice`, `kIndex`, `vIndex`, `fwdSeed`, `fwdGate`, `dOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx
```
</details>

<details><summary><code>forwardStepTileOffset</code></summary>

```lean
def forwardStepTileOffset
    (s : BlockState)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    t_rel * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState) (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    kIndex idx * D_MODEL_V + s.pids 2 * BLOCK_MODEL_V + vIndex idx
```
</details>

<details><summary><code>fwdClosed</code></summary>

```
/-- Genuine closed form for forward output chunk `m`:
`seed · ∏_{j<m} d_j + Σ_{t<m} S_t · ∏_{t<j<m} d_j`. -/
```
```lean
noncomputable def fwdClosed
    (s : BlockState) (S D LastKv : RegionName) (HAS_LAST_KV : Bool)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V m : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  fwdSeed s LastKv HAS_LAST_KV D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx *
      (∏ j ∈ Finset.range m, fwdGate s D NUM_BLOCK j) +
    ∑ t ∈ Finset.range m,
      s.readMem S
          (forwardStepTileOffset s t NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V idx) *
        (∏ j ∈ Finset.Ico (t + 1) m, fwdGate s D NUM_BLOCK j)
```
</details>

<details><summary><code>chunk_gate_recurrence_fwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gate_recurrence.py`'s `_fwd_recurrence`.

The optional `last_kv` argument is represented by `HAS_LAST_KV`. The backward
kernel walks the chunk dimension in reverse with pointer decrements, so it is
kept separate from this forward surface. -/
```
```lean
def chunk_gate_recurrence_fwd_surface
    (S D O last_kv : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (HAS_LAST_KV : Bool) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)

  S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
    offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
  O = O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
    offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
  if HAS_LAST_KV {
    last_kv = last_kv + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      tl.arange(0, $(BLOCK_MODEL_K))[:, None] * $(D_MODEL_V) +
      offset_s * $(BLOCK_MODEL_V) + tl.arange(0, $(BLOCK_MODEL_V))[None, :]
    acc = tl.load(last_kv).to(tl.float32)
  } else {
    acc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  }
  tl.store(O, (acc).to(O.dtype.element_ty))
  O += $(D_MODEL_K) * $(D_MODEL_V)
  D = D + offset_bh * $(NUM_BLOCK)
  for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
    d_i = tl.load(D)
    S_i = tl.load(S)
    acc = acc * d_i + S_i
    tl.store(O, (acc).to(O.dtype.element_ty))
    D += $(1)
    S += $(D_MODEL_K) * $(D_MODEL_V)
    O += $(D_MODEL_K) * $(D_MODEL_V)
  }
}
```
</details>

<details><summary><code>chunk_gate_recurrence_initial_last_kv_store_slice</code></summary>

```
/-- Initial forward output store for the `last_kv is not None` Python branch. -/
```
```lean
def chunk_gate_recurrence_initial_last_kv_store_slice
    (LastKv O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(LastKv + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}
```
</details>

<details><summary><code>chunk_gate_recurrence_initial_zero_store_slice</code></summary>

```
/-- Initial forward output store for the `last_kv is None` Python branch.

The source kernel initializes `acc` with a zero tile and immediately stores that
tile into the first output chunk before entering the recurrence loop. -/
```
```lean
def chunk_gate_recurrence_initial_zero_store_slice
    (O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}
```
</details>

<details><summary><code>chunk_gate_recurrence_forward_step_store_slice</code></summary>

```
/-- One forward recurrence step:
`acc = acc * d_i + S_i`, then store the updated accumulator into the next output
chunk. This isolates the Python loop body arithmetic from the full loop
induction. -/
```
```lean
def chunk_gate_recurrence_forward_step_store_slice
    (AccPrev S D O : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  prev = tl.load(AccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  s_i = tl.load(S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  acc = prev * d_i + s_i
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], acc)
}
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.1.val
```
</details>

<details><summary><code>vIndex</code></summary>

```lean
def vIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>fwdSeed</code></summary>

```lean
noncomputable def fwdSeed
    (s : BlockState) (LastKv : RegionName) (HAS_LAST_KV : Bool)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  if HAS_LAST_KV then
    s.readMem LastKv
      (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
  else 0
```
</details>

<details><summary><code>fwdGate</code></summary>

```
/-! ## Genuine forward closed form (the gated-recurrence fold)

`chunk_gate_recurrence.py`'s `_fwd_recurrence` seeds `acc` from `last_kv` (or
zero), stores it into output chunk `0`, and then for each chunk `t` runs the
gated update `acc = acc * d_t + S_t` and stores the new `acc` into output chunk
`t+1`. The scalar gate `d_t = D[offset_bh·NUM_BLOCK + t]` broadcasts over the
whole `[BLOCK_MODEL_K, BLOCK_MODEL_V]` state tile.

Unrolling the recurrence gives the **genuine closed form** for output chunk `m`
at tile element `idx`:

```
O[m][idx] = seed[idx] · ∏_{j<m} d_j  +  Σ_{t<m} S_t[idx] · ∏_{t<j<m} d_j
```

where `seed[idx] = last_kv[idx]` if `HAS_LAST_KV` else `0`. This is a standalone
specification over the *input* regions `S`, `D`, `last_kv` — never a read-back of
the kernel's own output `O`.

`fwdGate s D NUM_BLOCK j := D[offset_bh·NUM_BLOCK + j]` is the scalar gate at
chunk `j`; `fwdSeed` is the seeded initial state; `fwdClosed` is the closed form
above. -/
```
```lean
noncomputable def fwdGate
    (s : BlockState) (D : RegionName) (NUM_BLOCK j : Nat) : ℝ :=
  s.readMem D (dOffset s j NUM_BLOCK)
```
</details>

<details><summary><code>dOffset</code></summary>

```lean
def dOffset (s : BlockState) (t_rel NUM_BLOCK : Nat) : Nat :=
  s.pids 0 * NUM_BLOCK + t_rel
```
</details>

## Public theorem: `chunk_gate_recurrence_backward_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Genuine dimension-general backward output summary** for
`chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The reverse-loop backward surface lowers to the algorithm layer for arbitrary
symbolic dimensions, and each backward step face realizes a genuine arithmetic
closed form over the *input* regions: `DI` realizes `Dacc·d_i + DS_i`
(`bwdDaccStepSpec`), `DG` realizes the reduction `Σ Dacc·S_i` (`bwdDGStepSpec`),
and the boundary `DL` realizes the post-loop accumulator (`bwdDLStoreSpec`), all
under the materialized previous-state buffers `DaccPrev`/`DaccPre`. None is a
read-back of the kernel's own outputs.

Side conditions are honest: per-store output-offset injectivity
(`hDIInj` for `DI`, `hDLInj` for `DL`; the scalar `DG` store needs none). The
cross-chunk reverse fold over `range(NUM_BLOCK-1)` is the trusted boundary. -/
```
</details>

**Statement:**
```lean
theorem chunk_gate_recurrence_backward_output_summary_general
    (DaccPrev DaccPre DS S D DI DG DL : RegionName) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState)
    (hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    (hDLInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)) :
    ((chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V).toAlgorithm? =
        Except.ok
          (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
            NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V).toAlgKernel) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGStepSpec s DaccPrev DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccPre DL
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDLStoreSpec s DaccPre D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
```

**Assumptions / layout contracts:**
- `hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)`
- `hDLInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)`

**Closed-form spec defs (transitive):** `timeTileOffset`, `bwdDLOffset`, `chunk_gate_recurrence_bwd_surface`, `chunk_gate_recurrence_bwd_dacc_step_DI_store_slice`, `bwdDaccStepSpec`, `chunk_gate_recurrence_bwd_dg_step_store_slice`, `bwdDGOffset`, `bwdDGStepSpec`, `chunk_gate_recurrence_bwd_DL_store_slice`, `bwdDLStoreSpec`, `kIndex`, `vIndex`, `dOffset`

<details><summary><code>timeTileOffset</code></summary>

```lean
def timeTileOffset
    (s : BlockState)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * NUM_BLOCK * D_MODEL_K * D_MODEL_V +
    t_rel * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    s.pids 2 * BLOCK_MODEL_V + kIndex idx * D_MODEL_V + vIndex idx
```
</details>

<details><summary><code>bwdDLOffset</code></summary>

```lean
def bwdDLOffset
    (s : BlockState)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  s.pids 0 * D_MODEL_K * D_MODEL_V +
    s.pids 1 * D_MODEL_V * BLOCK_MODEL_K +
    s.pids 2 * BLOCK_MODEL_V +
    idx.1.val * D_MODEL_V + idx.2.1.val
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_surface</code></summary>

```
/-- Faithful transcription of `chunk_gate_recurrence.py`'s `_bwd_recurrence`.

The Python backward kernel starts from the last/penultimate chunk positions,
walks the chunk axis backward by decrementing pointers inside a forward
`range(NUM_BLOCK - 1)` loop, writes `DG`/`DI` for intermediate chunks, and
finally writes `DL` from the accumulated state. -/
```
```lean
def chunk_gate_recurrence_bwd_surface
    (S D DI DG DL DS : RegionName)
    (_NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  NUM_K = $(D_MODEL_K) // $(BLOCK_MODEL_K)
  NUM_V = $(D_MODEL_V) // $(BLOCK_MODEL_V)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  S = S + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
  DI = DI + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(2)) * $(D_MODEL_K) * $(D_MODEL_V)
  DS = DS + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :] + ($(NUM_BLOCK) - $(1)) * $(D_MODEL_K) * $(D_MODEL_V)
  DG = DG + offset_bh * $(NUM_BLOCK) * NUM_K * NUM_V +
    offset_d * NUM_V + offset_s + ($(NUM_BLOCK) - $(2)) * NUM_K * NUM_V
  D = D + offset_bh * $(NUM_BLOCK) + ($(NUM_BLOCK) - $(1))
  Dacc = tl.zeros([$(BLOCK_MODEL_K), $(BLOCK_MODEL_V)], dtype=tl.float32)
  for _i in range($(0), $(NUM_BLOCK) - $(1), $(1)) {
    S_i = tl.load(S)
    DS_i = tl.load(DS)
    d_i = tl.load(D)
    Dacc = Dacc * d_i + DS_i
    DG_i = tl.sum(Dacc * (S_i).to(tl.float32))
    tl.store(DG, (DG_i).to(DG.dtype.element_ty))
    tl.store(DI, (Dacc).to(DI.dtype.element_ty))
    S -= $(D_MODEL_K) * $(D_MODEL_V)
    DI -= $(D_MODEL_K) * $(D_MODEL_V)
    DS -= $(D_MODEL_K) * $(D_MODEL_V)
    DG -= NUM_K * NUM_V
    D -= $(1)
  }
  DL = DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    offs_v[None, :]
  DS_i = tl.load(DS)
  d_i = tl.load(D)
  Dacc = Dacc * d_i + DS_i
  tl.store(DL, (Dacc).to(DL.dtype.element_ty))
}
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_dacc_step_DI_store_slice</code></summary>

```
/-- One backward recurrence step for the tile accumulator:
`Dacc = Dacc * d_i + DS_i`, then store the updated accumulator into `DI` at the
current reverse-loop chunk. This isolates the reverse loop body's accumulator
arithmetic from the full loop induction. -/
```
```lean
def chunk_gate_recurrence_bwd_dacc_step_DI_store_slice
    (DaccPrev DS D DI : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  ds_i = tl.load(DS + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  dacc = prev * d_i + ds_i
  tl.store(DI + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :],
    (dacc).to(DI.dtype.element_ty))
}
```
</details>

<details><summary><code>bwdDaccStepSpec</code></summary>

```lean
noncomputable def bwdDaccStepSpec
    (s : BlockState) (DaccPrev DS D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPrev
      (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx) *
    s.readMem D (dOffset s t_rel NUM_BLOCK) +
  s.readMem DS
      (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V idx)
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_dg_step_store_slice</code></summary>

```
/-- One backward recurrence step for the compact `DG` scalar:
compute `Dacc = Dacc * d_i + DS_i`, then `DG_i = tl.sum(Dacc * S_i)` and store
that scalar into the `[B*H, NUM_BLOCK, NUM_K, NUM_V]` gradient layout. -/
```
```lean
def chunk_gate_recurrence_bwd_dg_step_store_slice
    (DaccPrev DS S D DG : RegionName)
    (t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat) : ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  ds_i = tl.load(DS + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  s_i = tl.load(S + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  dacc = prev * d_i + ds_i
  dg_i = tl.sum(dacc * s_i)
  tl.store(DG + offset_bh * $(NUM_BLOCK) * $(NUM_K) * $(NUM_V) +
    $(t_rel) * $(NUM_K) * $(NUM_V) + offset_d * $(NUM_V) + offset_s,
    (dg_i).to(DG.dtype.element_ty))
}
```
</details>

<details><summary><code>bwdDGOffset</code></summary>

```lean
def bwdDGOffset (s : BlockState) (t_rel NUM_BLOCK NUM_K NUM_V : Nat) : Nat :=
  s.pids 0 * NUM_BLOCK * NUM_K * NUM_V +
    t_rel * NUM_K * NUM_V + s.pids 1 * NUM_V + s.pids 2
```
</details>

<details><summary><code>bwdDGStepSpec</code></summary>

```lean
noncomputable def bwdDGStepSpec
    (s : BlockState) (DaccPrev DS S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ℝ :=
  ∑ i : Fin BLOCK_MODEL_K, ∑ j : Fin BLOCK_MODEL_V,
    let idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] :=
      TileShape.insertAxisIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] 1
        (TileShape.insertAxisIndex [BLOCK_MODEL_K] 0 PUnit.unit i) j
    bwdDaccStepSpec s DaccPrev DS D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
        BLOCK_MODEL_K BLOCK_MODEL_V idx *
      s.readMem S
        (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_DL_store_slice</code></summary>

```
/-- Proof-oriented DL final-state store slice of
`chunk_gate_recurrence.py`'s `_bwd_recurrence`. Takes a precomputed `DaccPre`
[BLOCK_MODEL_K, BLOCK_MODEL_V] tile (the post-loop accumulator) and proves
the writeback into `DL` at the canonical `(offset_bh, offset_d, offset_s)`
layout. -/
```
```lean
def chunk_gate_recurrence_bwd_DL_store_slice
    (DaccPre DL : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  base = offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  dacc = tl.load(DaccPre + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  tl.store(DL + base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :], dacc)
}
```
</details>

<details><summary><code>bwdDLStoreSpec</code></summary>

```lean
noncomputable def bwdDLStoreSpec
    (s : BlockState) (DaccPre : RegionName)
    (D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  s.readMem DaccPre
    (bwdDLOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.1.val
```
</details>

<details><summary><code>vIndex</code></summary>

```lean
def vIndex (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>dOffset</code></summary>

```lean
def dOffset (s : BlockState) (t_rel NUM_BLOCK : Nat) : Nat :=
  s.pids 0 * NUM_BLOCK + t_rel
```
</details>

## Also present (pinned special-case summaries)
- `chunk_gate_recurrence_forward_store_slice_compute_correct`
- `chunk_gate_recurrence_initial_last_kv_store_slice_compute_correct`
- `chunk_gate_recurrence_initial_zero_store_slice_compute_correct`
- `chunk_gate_recurrence_forward_step_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_DI_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_DG_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_DL_store_slice_compute_correct`
- `chunk_gate_recurrence_forward_store_python_test_shape_compute_correct`
- `chunk_gate_recurrence_initial_last_kv_python_test_shape_compute_correct`
- `chunk_gate_recurrence_initial_zero_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_dacc_step_DI_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_dg_step_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_DL_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_DI_surface_compute_correct`
- `chunk_gate_recurrence_bwd_DG_surface_compute_correct`
- `chunk_gate_recurrence_bwd_DL_surface_compute_correct`
- `chunk_gate_recurrence_forward_step_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_DI_python_test_shape_compute_correct`
- `chunk_gate_recurrence_bwd_DG_python_test_shape_compute_correct`
- `chunk_gate_recurrence_forward_python_test_shape_all_outputs_compute_correct`
- `chunk_gate_recurrence_backward_python_test_shape_all_outputs_compute_correct`
- `chunk_gate_recurrence_forward_python_test_shape_summary`
- `chunk_gate_recurrence_backward_python_test_shape_summary`

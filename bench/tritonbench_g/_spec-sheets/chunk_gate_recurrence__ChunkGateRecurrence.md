# Spec sheet — `bench/tritonbench_g/chunk_gate_recurrence/ChunkGateRecurrence.lean`

**Python source:** `bench/tritonbench_g/chunk_gate_recurrence/chunk_gate_recurrence.py`

## Public theorem: `chunk_gate_recurrence_output_summary_general`

<details><summary>docstring</summary>

```
/-- **SCOPE — this is a claim about six hand-cut store/step slices, not about the
launched kernels.** The launched surfaces appear only in the three lowering
clauses. Neither `range(NUM_BLOCK-1)` fold is modeled: the carry pins `hAcc`,
`hDacc`, `hDaccTail` are assumptions, and no clause chains one step's output into
the next step's carry buffer.

**Genuine shape-general output summary** for both Python kernels of
`chunk_gate_recurrence.py` — `_fwd_recurrence` and `_bwd_recurrence` — bundled
into one headline.

Every `expected` below is a standalone closed form over the *input* regions
(`S`, `D`, `last_kv` forward; `DS` = the incoming `DO`, `D`, `S` backward) — never
a read-back of the kernel's own `O`/`DI`/`DG`/`DL`:

* both forward surfaces (`last_kv` present/absent) and the backward surface lower
  to the algorithm layer at arbitrary symbolic dimensions;
* the forward initial store realizes `fwdClosed(0) = last_kv` (`last_kv` branch)
  and `fwdClosed(0) = 0` (zero branch);
* one forward loop body realizes `fwdClosed(t_rel+1)` from
  `AccPrev = fwdClosed(t_rel)`;
* one reverse loop body realizes `bwdClosed(t_rel+1)` into `DI` chunk `t_rel` and
  `Σ bwdClosed(t_rel+1) · S_{t_rel}` into `DG` chunk `t_rel`, from
  `DaccPrev = bwdClosed(t_rel+2)` — with `DS` and the gate `d` read one chunk
  **ahead** of `S`/`DI`/`DG`, as Python's split pointer initialisation dictates;
* the post-loop step realizes `bwdClosed(0)` into `DL`, restoring the
  `Dacc = Dacc * d_i + DS_i` that precedes that store.

Side conditions are honest: per-store output-offset injectivity, the single chunk
in-range bound, and the three carry pins. The cross-chunk folds over
`range(NUM_BLOCK-1)` (forward and reverse) are the trusted boundary, as is the
fact that `NUM_BLOCK - 1` decrements leave the backward pointers at chunk `0`. -/
```
</details>

**Statement:**
```lean
specification chunk_gate_recurrence_output_summary_general
    (AccPrev S D O LastKv : RegionName) (HAS_LAST_KV : Bool)
    (DaccPrev DaccTail DS DI DG DL : RegionName) (t_rel : Nat)
    (NUM_HEAD NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V : Nat)
    (s : BlockState)
    -- forced by `BlockState.scatter_readback_nd`: the forward initial tile store
    -- must not have two lanes colliding on one `O` cell.
    (hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    -- same, for the forward loop body's store into `O` chunk `t_rel + 1`.
    (hOutInjStep : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
    -- same, for the reverse loop body's store into `DI` chunk `t_rel`.
    (hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
    -- same, for the post-loop store into the chunk-free `DL` tile.
    (hDLInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
    -- forced by `bwdClosed_step`: the reverse loop body's gate/`DS` index is
    -- `t_rel + 1`, and peeling it off the fold needs it to be a real chunk.
    -- Python's loop runs `t_rel = NUM_BLOCK-2 … 0`, so `t_rel + 1 ≤ NUM_BLOCK-1`.
    -- (`bwdClosed_step` at the post-loop index `0` needs `0 < NUM_BLOCK`; that is
    -- implied by `hBwdIdx`, so it is not stated separately.)
    (hBwdIdx : t_rel + 1 < NUM_BLOCK)
    -- the forward carry: `acc` entering loop iteration `t_rel` is the `t_rel`-step
    -- fold. Assumed — the `range(NUM_BLOCK-1)` fold is the trusted boundary.
    (hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx)
    -- the reverse carry: `Dacc` entering the iteration that writes chunk `t_rel`
    -- is the reverse fold at gate index `t_rel + 2`. Assumed for the same reason;
    -- at the loop's first iteration (`t_rel = NUM_BLOCK - 2`) it is discharged by
    -- `bwdClosed_top`, matching Python's `Dacc = tl.zeros(...)` seed.
    (hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx)
    -- the reverse carry as the loop *exits* (gate index `1`), feeding the
    -- post-loop `DL` step. A separate region from `DaccPrev` on purpose: pinning
    -- one buffer to two different fold stages would silently constrain `t_rel`.
    (hDaccTail : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccTail
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V 1 idx) :
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.true).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (chunk_gate_recurrence_fwd_surface S D O LastKv
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      Bool.false).toAlgorithm? = Except.ok alg) ∧
    ((chunk_gate_recurrence_bwd_surface S D DI DG DL DS
      NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
      BLOCK_MODEL_V).toAlgorithm? =
        Except.ok
          (chunk_gate_recurrence_bwd_surface S D DI DG DL DS
            NUM_HEAD NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V).toAlgKernel) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_last_kv_store_slice LastKv O
        NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.true NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_initial_zero_store_slice O NUM_BLOCK
        D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv Bool.false NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V 0 idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_forward_step_store_slice AccPrev S D O
        t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (O, forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K
          D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V (t_rel + 1) idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dacc_step_DI_store_slice DaccPrev
        DS D DI t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DI, timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V (t_rel + 1) idx)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_dg_step_store_slice DaccPrev DS
        S D DG t_rel NUM_BLOCK NUM_K NUM_V D_MODEL_K D_MODEL_V BLOCK_MODEL_K
        BLOCK_MODEL_V)
      (initialState := s)
      (write := fun _ : PUnit =>
        some (DG, bwdDGOffset s t_rel NUM_BLOCK NUM_K NUM_V))
      (expected := fun _ =>
        bwdDGClosed s DS S D t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gate_recurrence_bwd_DL_store_slice DaccTail DS D DL
        0 NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        some (DL, accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx))
      (expected := fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V 0 idx))
```

**Assumptions / layout contracts:**
- `hOutInj0 : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        outOffset s NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)`
- `fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        forwardStepTileOffset s (t_rel + 1) NUM_BLOCK D_MODEL_K D_MODEL_V
          BLOCK_MODEL_K BLOCK_MODEL_V idx`
- `hDIInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)`
- `hDLInj : Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)`
- `hBwdIdx : t_rel + 1 < NUM_BLOCK`
- `hAcc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem AccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = fwdClosed s S D LastKv HAS_LAST_KV NUM_BLOCK D_MODEL_K D_MODEL_V
            BLOCK_MODEL_K BLOCK_MODEL_V t_rel idx`
- `hDacc : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccPrev
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V (t_rel + 2) idx`
- `hDaccTail : ∀ idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V],
      s.readMem DaccTail
          (accOffset s D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V idx)
        = bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
            BLOCK_MODEL_V 1 idx`

**Closed-form spec defs (transitive):** `outOffset`, `forwardStepTileOffset`, `timeTileOffset`, `accOffset`, `fwdClosed`, `bwdClosed`, `chunk_gate_recurrence_fwd_surface`, `chunk_gate_recurrence_bwd_surface`, `chunk_gate_recurrence_initial_last_kv_store_slice`, `chunk_gate_recurrence_initial_zero_store_slice`, `chunk_gate_recurrence_forward_step_store_slice`, `chunk_gate_recurrence_bwd_dacc_step_DI_store_slice`, `chunk_gate_recurrence_bwd_dg_step_store_slice`, `bwdDGOffset`, `bwdDGClosed`, `chunk_gate_recurrence_bwd_DL_store_slice`, `kIndex`, `vIndex`, `fwdSeed`, `fwdGate`, `dOffset`

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

<details><summary><code>bwdClosed</code></summary>

```
/-- Genuine closed form of the reverse carry with gate/`DS` index `a`:
`Σ_{a ≤ u < NUM_BLOCK} DS_u · ∏_{a ≤ w < u} d_w`. The gate factor reuses
`fwdGate` because the backward kernel reads the very same `cross_decay` row
(`d + offset_bh * NUM_BLOCK + ·`) as the forward kernel. -/
```
```lean
noncomputable def bwdClosed
    (s : BlockState) (DS D : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V a : Nat)
    (idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V]) : ℝ :=
  ∑ u ∈ Finset.Ico a NUM_BLOCK,
    s.readMem DS
        (timeTileOffset s u NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx) *
      ∏ w ∈ Finset.Ico a u, fwdGate s D NUM_BLOCK w
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

<details><summary><code>chunk_gate_recurrence_bwd_dacc_step_DI_store_slice</code></summary>

```
/-- One backward recurrence step for the tile accumulator:
`Dacc = Dacc * d_i + DS_i`, then store the updated accumulator into `DI` at the
current reverse-loop chunk. This isolates the reverse loop body's accumulator
arithmetic from the full loop induction.

Faithful addressing (see the table above): `DS` and `D` are read at index
`t_rel + 1`, while `DI` is written at chunk `t_rel`. -/
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
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  out_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel + 1))
  dacc = prev * d_i + ds_i
  tl.store(DI + out_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :],
    (dacc).to(DI.dtype.element_ty))
}
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_dg_step_store_slice</code></summary>

```
/-- One backward recurrence step for the compact `DG` scalar:
compute `Dacc = Dacc * d_i + DS_i`, then `DG_i = tl.sum(Dacc * S_i)` and store
that scalar into the `[B*H, NUM_BLOCK, NUM_K, NUM_V]` gradient layout.

Faithful addressing: `DS` and `D` are read at index `t_rel + 1` while `S` is read
and `DG` written at chunk `t_rel`. -/
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
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel + 1) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  s_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccPrev + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  s_i = tl.load(S + s_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel + 1))
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

<details><summary><code>bwdDGClosed</code></summary>

```
/-- Genuine closed form of the `DG` scalar written at chunk `t_rel`:
`Σ_{k,v} bwdClosed(t_rel+1)[k,v] · S_{t_rel}[k,v]` — the reverse carry contracted
against the `S` chunk *one behind* the carry's own index, exactly as the Python
loop pairs them. -/
```
```lean
noncomputable def bwdDGClosed
    (s : BlockState) (DS S D : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : ℝ :=
  ∑ i : Fin BLOCK_MODEL_K, ∑ j : Fin BLOCK_MODEL_V,
    let idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] :=
      TileShape.insertAxisIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] 1
        (TileShape.insertAxisIndex [BLOCK_MODEL_K] 0 PUnit.unit i) j
    bwdClosed s DS D NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
        (t_rel + 1) idx *
      s.readMem S
        (timeTileOffset s t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K
          BLOCK_MODEL_V idx)
```
</details>

<details><summary><code>chunk_gate_recurrence_bwd_DL_store_slice</code></summary>

```
/-- The post-loop `DL` boundary-gradient store (`chunk_gate_recurrence.py` lines
90-94). After the reverse loop's `NUM_BLOCK - 1` decrements the `DS` and `d`
pointers sit at chunk `0`, and Python performs **one more** gated multiply-add
`Dacc = Dacc * d_i + DS_i` before storing into `DL`. `DL` is a single
`[B, H, D_k, D_v]` tile with no chunk axis, hence the `accOffset` layout on the
store.

The tail index is a parameter `t_rel`; the headline instantiates it at `0`, which
is where the loop's `NUM_BLOCK - 1` decrements leave the pointers (the arrival
position is part of the trusted fold boundary). -/
```
```lean
def chunk_gate_recurrence_bwd_DL_store_slice
    (DaccTail DS D DL : RegionName)
    (t_rel NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  k_off = tl.arange(0, $(BLOCK_MODEL_K))
  v_off = tl.arange(0, $(BLOCK_MODEL_V))
  ds_base = offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
    $(t_rel) * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    offset_s * $(BLOCK_MODEL_V)
  prev = tl.load(DaccTail + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :])
  ds_i = tl.load(DS + ds_base +
    k_off[:, None] * $(D_MODEL_V) + v_off[None, :])
  d_i = tl.load(D + offset_bh * $(NUM_BLOCK) + $(t_rel))
  dacc = prev * d_i + ds_i
  tl.store(DL + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
    offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
    k_off[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
    v_off[None, :], (dacc).to(DL.dtype.element_ty))
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
spec over the *input* regions `S`, `D`, `last_kv` — never a read-back of the
kernel's own output `O`. (No docstring line may begin at column 0 with a
declaration keyword such as `theorem` or `specification`: `scripts/spec_sheet.py`
matches those at column 0 and would report a phantom headline.)

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

## Public theorem: `chunk_gate_recurrence_forward_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `chunk_gate_recurrence.py`'s forward
output chunk store: for every disjoint flat placement of `Acc` / `O`, every program
coordinate whose lanes are in bounds, and every launch state whose `Acc` chunk holds
`xs`, the translated pointer kernel terminates, every lane of the `O` chunk holds
`xs idx`, and every other memory cell is unchanged.

Both windows are built from all three program axes with different leading factors on
the two buffers. Dimension-general in `NUM_BLOCK`, `D_MODEL_K`, `D_MODEL_V` and the
two block sizes. Honest side-condition: output-address injectivity at every program
coordinate, the same hypothesis the per-write-map summary takes. -/
```
</details>

**Statement:**
```lean
specification chunk_gate_recurrence_forward_store_io_correctness
    (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
          + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val)) :
    fwdStoreIO Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V ⊨ fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
          + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val)`

**Closed-form spec defs (transitive):** `fwdStoreIO`, `chunk_gate_recurrence_forward_store_slice`

<details><summary><code>fwdStoreIO</code></summary>

```
/-- IO signature of the forward output store on the three-axis tile surface. -/
```
```lean
def fwdStoreIO (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : Masked3DTileKernelIO₁ where
  kernel := chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
  inp := Acc
  out := O
  shape := [BLOCK_MODEL_K, BLOCK_MODEL_V]
  read := fun p₀ p₁ p₂ idx =>
    p₀ * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  write := fun p₀ p₁ p₂ idx =>
    p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  mask := fun _p₀ _p₁ _p₂ _ => True
```
</details>

<details><summary><code>chunk_gate_recurrence_forward_store_slice</code></summary>

```
/-- Proof-oriented forward recurrence tile-store slice of
`chunk_gate_recurrence.py`'s `_fwd_recurrence`.

The full kernel repeatedly advances `O` by one KV block and stores the running
`acc`. This slice models one such tile store from a precomputed `Acc` tile into
`O`, preserving the source offset decomposition over `(offset_bh, offset_d,
offset_s)`. -/
```
```lean
def chunk_gate_recurrence_forward_store_slice
    (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(Acc + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], (acc).to(O.dtype.element_ty))
}
```
</details>

## Public theorem: `chunk_gate_recurrence_forward_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `chunk_gate_recurrence.py`'s forward output chunk
store: for **every** rounding model `R`, the same Hoare triple as
`chunk_gate_recurrence_forward_store_io_correctness`, but run under `execR R` and
read back as `.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy — no arithmetic, and the `.to(O.dtype.element_ty)`
erases to `.real` — so the slice is cast-free and the exact run transports
verbatim. The content of the rounding face here is exactly that: *this kernel
introduces no rounding event of its own*, at any `R`. -/
```
</details>

**Statement:**
```lean
specification chunk_gate_recurrence_forward_store_io_correctnessR
    (R : RoundingModel) (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
          + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val)) :
    fwdStoreIO Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx
```

**Assumptions / layout contracts:**
- `hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_MODEL_K, BLOCK_MODEL_V] =>
        p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
          + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val)`

**Closed-form spec defs (transitive):** `fwdStoreIO`, `chunk_gate_recurrence_forward_store_slice`

<details><summary><code>fwdStoreIO</code></summary>

```
/-- IO signature of the forward output store on the three-axis tile surface. -/
```
```lean
def fwdStoreIO (Acc O : RegionName) (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) : Masked3DTileKernelIO₁ where
  kernel := chunk_gate_recurrence_forward_store_slice Acc O NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V
  inp := Acc
  out := O
  shape := [BLOCK_MODEL_K, BLOCK_MODEL_V]
  read := fun p₀ p₁ p₂ idx =>
    p₀ * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  write := fun p₀ p₁ p₂ idx =>
    p₀ * NUM_BLOCK * D_MODEL_K * D_MODEL_V + p₁ * D_MODEL_V * BLOCK_MODEL_K
      + idx.1.val * D_MODEL_V + p₂ * BLOCK_MODEL_V + idx.2.1.val
  mask := fun _p₀ _p₁ _p₂ _ => True
```
</details>

<details><summary><code>chunk_gate_recurrence_forward_store_slice</code></summary>

```
/-- Proof-oriented forward recurrence tile-store slice of
`chunk_gate_recurrence.py`'s `_fwd_recurrence`.

The full kernel repeatedly advances `O` by one KV block and stores the running
`acc`. This slice models one such tile store from a precomputed `Acc` tile into
`O`, preserving the source offset decomposition over `(offset_bh, offset_d,
offset_s)`. -/
```
```lean
def chunk_gate_recurrence_forward_store_slice
    (Acc O : RegionName)
    (NUM_BLOCK D_MODEL_K D_MODEL_V BLOCK_MODEL_K BLOCK_MODEL_V : Nat) :
    ComputeKernel := triton {
  offset_bh = tl.program_id(0)
  offset_d = tl.program_id(1)
  offset_s = tl.program_id(2)
  offs_k = tl.arange(0, $(BLOCK_MODEL_K))
  offs_v = tl.arange(0, $(BLOCK_MODEL_V))
  acc = tl.load(Acc + offset_bh * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :])
  tl.store(O + offset_bh * $(NUM_BLOCK) * $(D_MODEL_K) * $(D_MODEL_V) +
      offset_d * $(D_MODEL_V) * $(BLOCK_MODEL_K) +
      offs_k[:, None] * $(D_MODEL_V) + offset_s * $(BLOCK_MODEL_V) +
      offs_v[None, :], (acc).to(O.dtype.element_ty))
}
```
</details>

## Also present (pinned special-case summaries)
- `chunk_gate_recurrence_forward_store_slice_compute_correct`
- `chunk_gate_recurrence_initial_last_kv_store_slice_compute_correct`
- `chunk_gate_recurrence_initial_zero_store_slice_compute_correct`
- `chunk_gate_recurrence_forward_step_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_dacc_step_DI_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_dg_step_store_slice_compute_correct`
- `chunk_gate_recurrence_bwd_DL_store_slice_compute_correct`

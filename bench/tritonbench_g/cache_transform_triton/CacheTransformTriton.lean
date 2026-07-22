import VeriTile.Triton

/-!
# `cache_transform_triton` — strict per-kernel correctness

`cache_transform_triton.py` has two kernels that gather rows of a precomputed
cos/sin rotary cache into per-token output buffers:
* `prefill_cache_kernel` — program `(idx0, idx1)` computes a flat token `idx`,
  derives the original sequence row `ori_seq_idx = idx - max(cumsum_lens ≤ idx)`
  via a `tl.where` / `tl.max` reduction over `cumsum_lengths`, then copies the
  `HIDDEN_DIM` cos/sin cache row into `cos_output` / `sin_output`, masked by
  `idx < total_length`.
* `decoding_cache_kernel` — program `pid` takes a `BLOCK_SIZE × HIDDEN_DIM`
  tile, reads the source row `lengths[idx]` per token, and copies the cos/sin
  cache rows into the outputs, masked by `idx < NUM_SEQS`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (grid sizing, `cumsum`/`lengths` inputs,
and cross-program composition into the output buffers) is the *trusted
boundary*, not a proof obligation here. Because the program ids are universally
quantified, each per-program statement covers every program of its grid.

## Proof architecture

Both kernels have `⊨` headlines on the shared **gather** genre
`GatherMasked2DKernelIO₂ₓ₂` (two float inputs read at one gather window driven
by one pinned `.nat` index vector, two outputs with per-output `WriteInj`
antecedents):

```
prefill_cache_correctness                          ← TOP THEOREM (prefill, ⊨)
  ├─ prefill_cache_kernel_flattenOk                 flat-memory bridge covers it
  ├─ prefill_cache_kernel_traceSafe                 per-execution safety walk
  └─ prefill_cache_kernel_region_run                region-model masked triple
       └─ prefill_cache_kernel_correct              algorithm-layer readback per cell
decoding_cache_correctness                         ← TOP THEOREM (decode slice, ⊨)
  ├─ decoding_cache_one_seq_block_flattenOk
  ├─ decoding_cache_one_seq_block_traceSafe
  └─ decoding_cache_one_seq_block_region_run        region-model masked triple
```

The `⊨` headline for prefill is over the full `prefill_cache_kernel` (one
HIDDEN_DIM-wide token row per program `(idx0, idx1)`, index vector =
`cumsum_lengths`, arity `N_ELEMENTS`). The `⊨` headline for decode is over the
one-sequence slice `decoding_cache_one_seq_block` (one HIDDEN_DIM block per
program `(seq, hid_block)`, single index cell `lengths[seq]`). The remaining
algorithm-layer readback / `ComputeCorrect` lemmas (`decoding_cache_kernel_*`,
`prefill_cache_{cos,sin}_store_slice_*`) are the retained value-level layer.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased. The source row index is data-dependent: prefill derives it from a
`max(where(cumsum_lens ≤ idx, cumsum_lens, 0))` reduction (modeled as
`reduceMaxNatDrop`, total via a `0 < N_ELEMENTS` hypothesis — the sole extra
side condition on the prefill headline), decode reads it from `lengths`. Both
headlines require `cos_output ≠ sin_output` (distinct output buffers, so neither
store clobbers the other); the per-output no-aliasing `WriteInj` obligations are
carried by the skin as antecedents on each value leg. `@triton.autotune` is not
modeled.
-/

namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `cache_transform_triton.py`'s
`prefill_cache_kernel`. -/
def prefill_cache_kernel
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx0 = tl.program_id(axis=0)
  idx1 = tl.program_id(axis=1)
  idx = idx0 * $(BLOCK_SIZE) + idx1
  cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, $(N_ELEMENTS)))
  ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, $(0)))
  cos_cache_part = tl.load(
    cos_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  sin_cache_part = tl.load(
    sin_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  tl.store(
    cos_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    cos_cache_part,
    mask=idx < $(total_length))
  tl.store(
    sin_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    sin_cache_part,
    mask=idx < $(total_length))
}

/-- Faithful transcription of `cache_transform_triton.py`'s
`decoding_cache_kernel`.

Allowed mechanical Lean-syntax-only changes:
- `lengths` is a typed Lean Nat region so its `tl.load` call does not need an
  extra `dtype=` kwarg. -/
def decoding_cache_kernel
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  ori_seq_idx = tl.load(lengths + idx,
    mask=idx < $(NUM_SEQS), other=None)
  cos_cache_part = tl.load(cos_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  sin_cache_part = tl.load(sin_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  tl.store(cos_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    cos_cache_part, mask=idx[:, None] < $(NUM_SEQS))
  tl.store(sin_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    sin_cache_part, mask=idx[:, None] < $(NUM_SEQS))
}

/-- Proof-oriented one-sequence, one-hidden-block slice of
`cache_transform_triton.py`'s `decoding_cache_kernel`.

This models the decoding branch: load the source cache row from `lengths[seq]`
and copy the same hidden block from cos/sin caches into cos/sin outputs. -/
def decoding_cache_one_seq_block
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    ComputeKernel := triton {
  seq = tl.program_id(0)
  hid_block = tl.program_id(1)
  offs = tl.arange(0, $(BLOCK_H))
  hid = hid_block * $(BLOCK_H) + offs
  ori_seq_idx = tl.load(lengths + seq, mask=seq < $(NUM_SEQS),
    other=$(0))
  cos_part = tl.load(cos_cache + ori_seq_idx * $(cache_stride) +
      hid * $(hidden_stride),
    mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)), other=0.0)
  sin_part = tl.load(sin_cache + ori_seq_idx * $(cache_stride) +
      hid * $(hidden_stride),
    mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)), other=0.0)
  tl.store(cos_output + seq * $(cache_stride) + hid * $(hidden_stride),
    cos_part, mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)))
  tl.store(sin_output + seq * $(cache_stride) + hid * $(hidden_stride),
    sin_part, mask=(seq < $(NUM_SEQS)) and (hid < $(HIDDEN_DIM)))
}

def hidIndex (s : BlockState) (BLOCK_H : Nat) (i : Fin BLOCK_H) : Nat :=
  s.pids 1 * BLOCK_H + i.val

def seqIndex (s : BlockState) : Nat :=
  s.pids 0

def sourceSeq (s : BlockState) (lengths : RegionName) : Nat :=
  s.readMemValue .nat lengths (seqIndex s)

def active (s : BlockState) (HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (i : Fin BLOCK_H) : Prop :=
  seqIndex s < NUM_SEQS ∧ hidIndex s BLOCK_H i < HIDDEN_DIM

instance activeDecidable
    (s : BlockState) (HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) (i : Fin BLOCK_H) :
    Decidable (active s HIDDEN_DIM NUM_SEQS BLOCK_H i) := by
  unfold active
  infer_instance

def outOffset (s : BlockState) (cache_stride hidden_stride BLOCK_H : Nat)
    (i : Fin BLOCK_H) : Nat :=
  seqIndex s * cache_stride + hidIndex s BLOCK_H i * hidden_stride

def cacheOffset
    (s : BlockState) (lengths : RegionName)
    (cache_stride hidden_stride BLOCK_H : Nat) (i : Fin BLOCK_H) : Nat :=
  sourceSeq s lengths * cache_stride + hidIndex s BLOCK_H i * hidden_stride

def rowIndex (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * BLOCK_SIZE + i.val

def decodeOutOffset
    (s : BlockState) (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Nat :=
  rowIndex s BLOCK_SIZE idx.1 * cache_stride + idx.2.1.val * hidden_stride

def decodeCacheOffset
    (s : BlockState) (lengths : RegionName)
    (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Nat :=
  s.readMemValue .nat lengths (rowIndex s BLOCK_SIZE idx.1) * cache_stride +
    idx.2.1.val * hidden_stride

def decodeActive (s : BlockState) (NUM_SEQS BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) : Prop :=
  rowIndex s BLOCK_SIZE idx.1 < NUM_SEQS

instance decodeActiveDecidable
    (s : BlockState) (NUM_SEQS BLOCK_SIZE : Nat)
    (idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM]) :
    Decidable (decodeActive s NUM_SEQS BLOCK_SIZE idx) := by
  unfold decodeActive
  infer_instance

/-- Algorithm-layer correctness for the full decoding cache-copy surface. -/
theorem decoding_cache_kernel_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx))
    (hExec : exec (decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE)
        s = some s') :
    (∀ idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM],
      s'.readMem cos_output
          (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx) =
        if decodeActive s NUM_SEQS BLOCK_SIZE idx then
          s.readMem cos_cache
            (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        else
          s.readMem cos_output
            (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) ∧
    (∀ idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM],
      s'.readMem sin_output
          (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx) =
        if decodeActive s NUM_SEQS BLOCK_SIZE idx then
          s.readMem sin_cache
            (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        else
          s.readMem sin_output
            (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) := by
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        (s.pids 0 * BLOCK_SIZE + idx.1.val) * cache_stride +
          idx.2.1.val * hidden_stride) := by
    intro a b h
    exact hOutInj (by
      simpa [decodeOutOffset, rowIndex] using h)
  by_cases hB : 0 < BLOCK_SIZE
  · by_cases hH : 0 < HIDDEN_DIM
    · simp [exec, decoding_cache_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
            Tile.expandDim, Tile.uop, NumericDType.add, NumericDType.mul,
            ComparableDType.lt, TileShape.dropInsertedIndex,
            Tile.remap, BlockState.readMemValue, hB, hH] at hExec
      subst s'
      constructor
      · intro idx
        simp only [decodeOutOffset, rowIndex]
        rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := sin_output) (R := cos_output) (h_ne := hRegion)
          (P := fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
            s.pids 0 * BLOCK_SIZE + idx.1.val < NUM_SEQS)
          (off := (s.pids 0 * BLOCK_SIZE + idx.1.val) * cache_stride +
            idx.2.1.val * hidden_stride)
          (l := TileShape.allIndices [BLOCK_SIZE, HIDDEN_DIM])]
        rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
        by_cases hActive : s.pids 0 * BLOCK_SIZE + idx.1.val < NUM_SEQS
        · simp [decodeActive, decodeCacheOffset, decodeOutOffset, rowIndex,
                BlockState.readMemValue, hActive]
        · simp [decodeActive, rowIndex, hActive]
      · intro idx
        simp only [decodeOutOffset, rowIndex]
        rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
        by_cases hActive : s.pids 0 * BLOCK_SIZE + idx.1.val < NUM_SEQS
        · simp [decodeActive, decodeCacheOffset, decodeOutOffset, rowIndex,
                BlockState.readMemValue, hActive]
        · rw [BlockState.scatter_prop_masked_preserves_other_region
            (region := cos_output) (R := sin_output) (h_ne := Ne.symm hRegion)
            (P := fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
              s.pids 0 * BLOCK_SIZE + idx.1.val < NUM_SEQS)
            (off := (s.pids 0 * BLOCK_SIZE + idx.1.val) * cache_stride +
              idx.2.1.val * hidden_stride)
            (l := TileShape.allIndices [BLOCK_SIZE, HIDDEN_DIM])]
          simp [decodeActive, rowIndex, hActive]
    · constructor
      · intro idx
        exact False.elim (hH (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt))
      · intro idx
        exact False.elim (hH (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt))
  · constructor
    · intro idx
      exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))
    · intro idx
      exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the full decoding cache-copy surface. -/
theorem decoding_cache_kernel_compute_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (TileIndex [BLOCK_SIZE, HIDDEN_DIM])
          (TileIndex [BLOCK_SIZE, HIDDEN_DIM]) =>
        match i with
        | .inl idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (cos_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if decodeActive s NUM_SEQS BLOCK_SIZE idx then
              some (sin_output, decodeOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none)
      (expected := fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        | .inr idx =>
            s.readMem sin_cache
              (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_cache_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := decoding_cache_kernel_correct cos_cache sin_cache lengths cos_output
    sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE s s'
    hRegion hOutInj hExec
  cases i with
  | inl idx =>
      by_cases hActive : decodeActive s NUM_SEQS BLOCK_SIZE idx
      · have hi := h.1 idx
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]
  | inr idx =>
      by_cases hActive : decodeActive s NUM_SEQS BLOCK_SIZE idx
      · have hi := h.2 idx
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]

/-- Algorithm-layer correctness for the sin output of the full decoding
cache-copy surface. The sin store is the outer (last) foldl in the kernel,
so the active-branch readback reduces via `scatter_readback_prop_masked_nd`
directly. The inactive-branch readback strips the inner cos foldl via
`foldl_writeMem_const_region_prop_masked_readMem_other`, which needs
`cos_output ≠ sin_output`. -/
theorem decoding_cache_kernel_sin_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx))
    (hExec : exec (decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE)
        s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM],
      s'.readMem sin_output
          (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx) =
        if decodeActive s NUM_SEQS BLOCK_SIZE idx then
          s.readMem sin_cache
            (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)
        else
          s.readMem sin_output
            (decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx) :=
  (decoding_cache_kernel_correct cos_cache sin_cache lengths cos_output
    sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE s s'
    hRegion hOutInj hExec).2

/-- Compute-facing correctness for the sin output of the full decoding
cache-copy surface. -/
theorem decoding_cache_kernel_sin_compute_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
          decodeActive s NUM_SEQS BLOCK_SIZE idx)
        (fun idx => (sin_output,
          decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)))
      (expected := fun idx =>
        s.readMem sin_cache
          (decodeCacheOffset s lengths cache_stride hidden_stride BLOCK_SIZE idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_cache_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := decoding_cache_kernel_sin_correct cos_cache sin_cache lengths
    cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
    BLOCK_SIZE s s' hRegion hOutInj hExec idx
  simpa [hActive] using h

/-- Algorithm-layer correctness for the one-sequence decoding cache copy. -/
theorem decoding_cache_one_seq_block_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (s s' : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hCosInj : Function.Injective
      (fun i : Fin BLOCK_H => outOffset s cache_stride hidden_stride BLOCK_H i))
    (hSinInj : Function.Injective
      (fun i : Fin BLOCK_H => outOffset s cache_stride hidden_stride BLOCK_H i))
    (hExec : exec (decoding_cache_one_seq_block cos_cache sin_cache lengths
        cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
        BLOCK_H) s = some s') :
    (∀ i : Fin BLOCK_H,
      s'.readMem cos_output (outOffset s cache_stride hidden_stride BLOCK_H i) =
        if active s HIDDEN_DIM NUM_SEQS BLOCK_H i then
          s.readMem cos_cache
            (cacheOffset s lengths cache_stride hidden_stride BLOCK_H i)
        else
          s.readMem cos_output
            (outOffset s cache_stride hidden_stride BLOCK_H i)) ∧
    (∀ i : Fin BLOCK_H,
      s'.readMem sin_output (outOffset s cache_stride hidden_stride BLOCK_H i) =
        if active s HIDDEN_DIM NUM_SEQS BLOCK_H i then
          s.readMem sin_cache
            (cacheOffset s lengths cache_stride hidden_stride BLOCK_H i)
        else
          s.readMem sin_output
            (outOffset s cache_stride hidden_stride BLOCK_H i)) := by
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_H] =>
        s.pids 0 * cache_stride + (s.pids 1 * BLOCK_H + idx.1.val) *
          hidden_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hCosInj
      simpa [outOffset, seqIndex, hidIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  have hRawInjSin : Function.Injective
      (fun idx : TileIndex [BLOCK_H] =>
        s.pids 0 * cache_stride + (s.pids 1 * BLOCK_H + idx.1.val) *
          hidden_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hSinInj
      simpa [outOffset, seqIndex, hidIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_H
  · simp [exec, decoding_cache_one_seq_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBH] at hExec
    subst s'
    constructor
    · intro i
      simp only [outOffset, seqIndex, hidIndex]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := sin_output) (R := cos_output) (h_ne := hRegion)
        (P := fun idx : TileIndex [BLOCK_H] =>
          s.pids 0 < NUM_SEQS ∧ s.pids 1 * BLOCK_H + idx.1.val < HIDDEN_DIM)
        (off := s.pids 0 * cache_stride +
          (s.pids 1 * BLOCK_H + i.val) * hidden_stride)
        (l := TileShape.allIndices [BLOCK_H])]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
        (i, PUnit.unit)]
      by_cases hSeq : s.pids 0 < NUM_SEQS
      · by_cases hHid : s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM
        · simp [active, cacheOffset, sourceSeq, outOffset, seqIndex, hidIndex,
                BlockState.readMemValue, hSeq, hHid]
        · simp [active, seqIndex, hidIndex, hSeq, hHid]
      · simp [active, seqIndex, hSeq]
    · intro i
      simp only [outOffset, seqIndex, hidIndex]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInjSin
        (i, PUnit.unit)]
      by_cases hSeq : s.pids 0 < NUM_SEQS
      · by_cases hHid : s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM
        · simp [active, cacheOffset, sourceSeq, outOffset, seqIndex, hidIndex,
                BlockState.readMemValue, hSeq, hHid]
        · rw [BlockState.scatter_prop_masked_preserves_other_region
            (region := cos_output) (R := sin_output) (h_ne := Ne.symm hRegion)
            (P := fun idx : TileIndex [BLOCK_H] =>
              s.pids 0 < NUM_SEQS ∧
                s.pids 1 * BLOCK_H + idx.1.val < HIDDEN_DIM)
            (off := s.pids 0 * cache_stride +
              (s.pids 1 * BLOCK_H + i.val) * hidden_stride)
            (l := TileShape.allIndices [BLOCK_H])]
          simp [active, seqIndex, hidIndex, hSeq, hHid]
      · rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := cos_output) (R := sin_output) (h_ne := Ne.symm hRegion)
          (P := fun idx : TileIndex [BLOCK_H] =>
            s.pids 0 < NUM_SEQS ∧
              s.pids 1 * BLOCK_H + idx.1.val < HIDDEN_DIM)
          (off := s.pids 0 * cache_stride +
            (s.pids 1 * BLOCK_H + i.val) * hidden_stride)
          (l := TileShape.allIndices [BLOCK_H])]
        simp [active, seqIndex, hSeq]
  · constructor
    · intro i
      exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))
    · intro i
      exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the cos/sin decoding cache copy. -/
theorem decoding_cache_one_seq_block_compute_correct
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hCosInj : Function.Injective
      (fun i : Fin BLOCK_H => outOffset s cache_stride hidden_stride BLOCK_H i))
    (hSinInj : Function.Injective
      (fun i : Fin BLOCK_H => outOffset s cache_stride hidden_stride BLOCK_H i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := decoding_cache_one_seq_block cos_cache sin_cache lengths
        cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
        BLOCK_H)
      (initialState := s)
      (write := fun i : Sum (Fin BLOCK_H) (Fin BLOCK_H) =>
        match i with
        | .inl lane =>
            if active s HIDDEN_DIM NUM_SEQS BLOCK_H lane then
              some (cos_output, outOffset s cache_stride hidden_stride BLOCK_H lane)
            else none
        | .inr lane =>
            if active s HIDDEN_DIM NUM_SEQS BLOCK_H lane then
              some (sin_output, outOffset s cache_stride hidden_stride BLOCK_H lane)
            else none)
      (expected := fun i =>
        match i with
        | .inl lane =>
            s.readMem cos_cache
              (cacheOffset s lengths cache_stride hidden_stride BLOCK_H lane)
        | .inr lane =>
            s.readMem sin_cache
              (cacheOffset s lengths cache_stride hidden_stride BLOCK_H lane)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_cache_one_seq_block]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := decoding_cache_one_seq_block_correct cos_cache sin_cache lengths
    cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
    BLOCK_H s s' hRegion hCosInj hSinInj hExec
  cases i with
  | inl lane =>
      by_cases hActive : active s HIDDEN_DIM NUM_SEQS BLOCK_H lane
      · have hi := h.1 lane
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]
  | inr lane =>
      by_cases hActive : active s HIDDEN_DIM NUM_SEQS BLOCK_H lane
      · have hi := h.2 lane
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]

/-- Proof-oriented cos-output store slice of `cache_transform_triton.py`'s
`prefill_cache_kernel`. Takes a precomputed `CosPre` tile and proves the
masked tile writeback into `cos_output` at the per-(idx0, idx1) row. -/
def prefill_cache_cos_store_slice
    (CosPre cos_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx0 = tl.program_id(axis=0)
  idx1 = tl.program_id(axis=1)
  idx = idx0 * $(BLOCK_SIZE) + idx1
  cos_part = tl.load(CosPre + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  tl.store(cos_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    cos_part, mask=idx < $(total_length))
}

def prefillIdx (s : BlockState) (BLOCK_SIZE : Nat) : Nat :=
  s.pids 0 * BLOCK_SIZE + s.pids 1

def prefillOutOffset
    (s : BlockState) (cache_stride hidden_stride BLOCK_SIZE : Nat)
    (i : Fin HIDDEN_DIM) : Nat :=
  prefillIdx s BLOCK_SIZE * cache_stride + i.val * hidden_stride

def prefillActive (s : BlockState) (total_length BLOCK_SIZE : Nat) : Prop :=
  prefillIdx s BLOCK_SIZE < total_length

instance prefillActiveDecidable (s : BlockState) (total_length BLOCK_SIZE : Nat) :
    Decidable (prefillActive s total_length BLOCK_SIZE) := by
  unfold prefillActive
  infer_instance

theorem prefill_cache_cos_store_slice_correct
    (CosPre cos_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i))
    (hExec : exec (prefill_cache_cos_store_slice CosPre cos_output
        cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE) s = some s') :
    ∀ i : Fin HIDDEN_DIM,
      s'.readMem cos_output
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) =
        if prefillActive s total_length BLOCK_SIZE then
          s.readMem CosPre (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)
        else
          s.readMem cos_output
            (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HIDDEN_DIM] =>
        (s.pids 0 * BLOCK_SIZE + s.pids 1) * cache_stride + idx.1.val * hidden_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [prefillOutOffset, prefillIdx] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, prefill_cache_cos_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [prefillOutOffset, prefillIdx]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hAct : s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length
  · simp [prefillActive, prefillIdx, hAct]
  · simp [prefillActive, prefillIdx, hAct]

theorem prefill_cache_cos_store_slice_compute_correct
    (CosPre cos_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := prefill_cache_cos_store_slice CosPre cos_output
        cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HIDDEN_DIM => prefillActive s total_length BLOCK_SIZE)
        (fun i => (cos_output,
          prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)))
      (expected := fun i =>
        s.readMem CosPre
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prefill_cache_cos_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prefill_cache_cos_store_slice_correct CosPre cos_output
    cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented sin-output store slice of `cache_transform_triton.py`'s
`prefill_cache_kernel`. Sin-side analog of the cos store slice. -/
def prefill_cache_sin_store_slice
    (SinPre sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  idx0 = tl.program_id(axis=0)
  idx1 = tl.program_id(axis=1)
  idx = idx0 * $(BLOCK_SIZE) + idx1
  sin_part = tl.load(SinPre + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  tl.store(sin_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    sin_part, mask=idx < $(total_length))
}

theorem prefill_cache_sin_store_slice_correct
    (SinPre sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i))
    (hExec : exec (prefill_cache_sin_store_slice SinPre sin_output
        cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE) s = some s') :
    ∀ i : Fin HIDDEN_DIM,
      s'.readMem sin_output
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) =
        if prefillActive s total_length BLOCK_SIZE then
          s.readMem SinPre (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)
        else
          s.readMem sin_output
            (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HIDDEN_DIM] =>
        (s.pids 0 * BLOCK_SIZE + s.pids 1) * cache_stride + idx.1.val * hidden_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [prefillOutOffset, prefillIdx] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, prefill_cache_sin_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [prefillOutOffset, prefillIdx]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
  by_cases hAct : s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length
  · simp [prefillActive, prefillIdx, hAct]
  · simp [prefillActive, prefillIdx, hAct]

theorem prefill_cache_sin_store_slice_compute_correct
    (SinPre sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := prefill_cache_sin_store_slice SinPre sin_output
        cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HIDDEN_DIM => prefillActive s total_length BLOCK_SIZE)
        (fun i => (sin_output,
          prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)))
      (expected := fun i =>
        s.readMem SinPre
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prefill_cache_sin_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prefill_cache_sin_store_slice_correct SinPre sin_output
    cache_stride hidden_stride total_length HIDDEN_DIM BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-- Source row index `ori_seq_idx` computed by the full prefill kernel:
`idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, 0))`. The reduction
is `reduceMaxNatDrop` over the `[N_ELEMENTS]` tile loaded from `cumsum_lengths`.
When `N_ELEMENTS = 0` the reduction returns `none`; we treat that case as `0`
so the spec is total. -/
noncomputable def prefillOriSeqIdx
    (s : BlockState) (cumsum_lengths : Region .nat) (BLOCK_SIZE N_ELEMENTS : Nat) : Nat :=
  let idx := prefillIdx s BLOCK_SIZE
  let tile : Tile .nat [N_ELEMENTS] :=
    ⟨fun i => s.readMemValue .nat (Region.cast cumsum_lengths) i.1.val⟩
  let cond : Tile .bool [N_ELEMENTS] :=
    ⟨fun i => decide (tile.data i ≤ idx)⟩
  let zero : Tile .nat [N_ELEMENTS] := ⟨fun _ => 0⟩
  let masked : Tile .nat [N_ELEMENTS] := Tile.select cond tile zero
  let reduced : Option (Tile .nat (TileShape.eraseAxis [N_ELEMENTS] ⟨0, by simp⟩)) :=
    Tile.reduceMaxNatDrop (shape := [N_ELEMENTS]) ⟨0, by simp⟩ masked
  let maxv : Nat := match reduced with
    | some t => t.data PUnit.unit
    | none => 0
  idx - maxv

/-- Cache offset for the full prefill kernel using the data-dependent source row
`ori_seq_idx`. -/
noncomputable def prefillCacheOffset
    (s : BlockState) (cumsum_lengths : Region .nat)
    (cache_stride hidden_stride BLOCK_SIZE N_ELEMENTS : Nat) (i : Fin HIDDEN_DIM) : Nat :=
  prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS * cache_stride +
    i.val * hidden_stride

/-- Algorithm-layer correctness for the full prefill cache-copy surface. -/
theorem prefill_cache_kernel_correct
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hN : 0 < N_ELEMENTS)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i))
    (hExec : exec (prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE) s = some s') :
    (∀ i : Fin HIDDEN_DIM,
      s'.readMem cos_output
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) =
        if prefillActive s total_length BLOCK_SIZE then
          s.readMem cos_cache
            (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
              BLOCK_SIZE N_ELEMENTS i)
        else
          s.readMem cos_output
            (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) ∧
    (∀ i : Fin HIDDEN_DIM,
      s'.readMem sin_output
          (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i) =
        if prefillActive s total_length BLOCK_SIZE then
          s.readMem sin_cache
            (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
              BLOCK_SIZE N_ELEMENTS i)
        else
          s.readMem sin_output
            (prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) := by
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HIDDEN_DIM] =>
        (s.pids 0 * BLOCK_SIZE + s.pids 1) * cache_stride + idx.1.val * hidden_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [prefillOutOffset, prefillIdx] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  by_cases hH : 0 < HIDDEN_DIM
  · simp [exec, prefill_cache_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.le,
          Tile.reduceMaxNat, Tile.reduceMaxNatDrop, TileShape.axisDim,
          hN, hH] at hExec
    -- Note: `prefillOriSeqIdx` unfolds to the same `Finset.sup'` expression
    -- that the symbolic execution produces.
    have hOri : prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS =
        s.pids 0 * BLOCK_SIZE + s.pids 1 -
          Finset.univ.sup' (Finset.univ_nonempty_iff.mpr ⟨⟨0, hN⟩⟩)
            (fun x : Fin N_ELEMENTS =>
              if s.readMemValue TileDType.nat cumsum_lengths.cast x.val ≤
                  s.pids 0 * BLOCK_SIZE + s.pids 1 then
                s.readMemValue TileDType.nat cumsum_lengths.cast x.val
              else 0) := by
      simp [prefillOriSeqIdx, prefillIdx, Tile.reduceMaxNatDrop,
            TileShape.axisDim, hN]
      rfl
    subst s'
    refine ⟨?_, ?_⟩
    · intro i
      simp only [prefillOutOffset, prefillIdx, prefillCacheOffset,
                 prefillActive, hOri]
      rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
            sin_output _ _ _ _ _ cos_output _ hRegion]
      rw [BlockState.scatter_readback_prop_masked_nd
            (region := cos_output) _ _ _ _ hRawInj (i, PUnit.unit)]
      by_cases hAct : s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length
      · simp [hAct]
        rfl
      · simp [hAct]
    · intro i
      simp only [prefillOutOffset, prefillIdx, prefillCacheOffset,
                 prefillActive, hOri]
      rw [BlockState.scatter_readback_prop_masked_nd
            (region := sin_output) _ _ _ _ hRawInj (i, PUnit.unit)]
      by_cases hAct : s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length
      · simp [hAct]
        rfl
      · rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
              cos_output _ _ _ _ _ sin_output _ (Ne.symm hRegion)]
        simp [hAct]
  · constructor
    · intro i
      exact False.elim (hH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))
    · intro i
      exact False.elim (hH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the full prefill cache-copy surface. -/
theorem prefill_cache_kernel_compute_correct
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hN : 0 < N_ELEMENTS)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Sum (Fin HIDDEN_DIM) (Fin HIDDEN_DIM) =>
        match i with
        | .inl idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (cos_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none
        | .inr idx =>
            if prefillActive s total_length BLOCK_SIZE then
              some (sin_output, prefillOutOffset s cache_stride hidden_stride
                BLOCK_SIZE idx)
            else none)
      (expected := fun i =>
        match i with
        | .inl idx =>
            s.readMem cos_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx)
        | .inr idx =>
            s.readMem sin_cache
              (prefillCacheOffset s cumsum_lengths cache_stride hidden_stride
                BLOCK_SIZE N_ELEMENTS idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prefill_cache_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := prefill_cache_kernel_correct cos_cache sin_cache cumsum_lengths
    cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
    N_ELEMENTS BLOCK_SIZE s s' hRegion hN hOutInj hExec
  cases i with
  | inl idx =>
      by_cases hActive : prefillActive s total_length BLOCK_SIZE
      · have hi := h.1 idx
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]
  | inr idx =>
      by_cases hActive : prefillActive s total_length BLOCK_SIZE
      · have hi := h.2 idx
        simp [hActive] at hi ⊢
        exact hi
      · simp [hActive]

/-! ## `⊨` migration -/

private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

theorem decoding_cache_one_seq_block_flattenOk
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    ((decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
      sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [decoding_cache_one_seq_block, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem decoding_cache_one_seq_block_traceSafe
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (bounds : RegionBounds) (s : BlockState) (src : Nat)
    (hsrc : s.pids 0 < NUM_SEQS → s.readMemValue .nat lengths (s.pids 0) = src)
    (hbx : s.pids 0 < NUM_SEQS → s.pids 0 < bounds lengths)
    (hbr1 : ∀ i : Fin BLOCK_H, s.pids 0 < NUM_SEQS →
      s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      src * cache_stride + (s.pids 1 * BLOCK_H + i.val) * hidden_stride
        < bounds cos_cache)
    (hbr2 : ∀ i : Fin BLOCK_H, s.pids 0 < NUM_SEQS →
      s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      src * cache_stride + (s.pids 1 * BLOCK_H + i.val) * hidden_stride
        < bounds sin_cache)
    (hbw1 : ∀ i : Fin BLOCK_H, s.pids 0 < NUM_SEQS →
      s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      s.pids 0 * cache_stride + (s.pids 1 * BLOCK_H + i.val) * hidden_stride
        < bounds cos_output)
    (hbw2 : ∀ i : Fin BLOCK_H, s.pids 0 < NUM_SEQS →
      s.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      s.pids 0 * cache_stride + (s.pids 1 * BLOCK_H + i.val) * hidden_stride
        < bounds sin_output) :
    Kernel.TraceSafe bounds
      ((decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [decoding_cache_one_seq_block, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, BlockState.readMemValue]
  refine ⟨hbx, ?_, ?_, hbw1, hbw2⟩
  · intro a hSeq hHid
    rw [if_pos hSeq]
    have h := hbr1 a hSeq hHid
    rw [← hsrc hSeq] at h
    exact h
  · intro a hSeq hHid
    rw [if_pos hSeq]
    have h := hbr2 a hSeq hHid
    rw [← hsrc hSeq] at h
    exact h

theorem decoding_cache_one_seq_block_region_run
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (hRegion : cos_output ≠ sin_output)
    (s₀ : BlockState) (src : Nat) (xs ys : Fin BLOCK_H → ℝ)
    (hsrc : s₀.pids 0 < NUM_SEQS → s₀.readMemValue .nat lengths (s₀.pids 0) = src)
    (hx : ∀ i : Fin BLOCK_H, s₀.pids 0 < NUM_SEQS →
      s₀.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      s₀.readMem cos_cache
        (src * cache_stride + (s₀.pids 1 * BLOCK_H + i.val) * hidden_stride) = xs i)
    (hy : ∀ i : Fin BLOCK_H, s₀.pids 0 < NUM_SEQS →
      s₀.pids 1 * BLOCK_H + i.val < HIDDEN_DIM →
      s₀.readMem sin_cache
        (src * cache_stride + (s₀.pids 1 * BLOCK_H + i.val) * hidden_stride) = ys i) :
    ∃ s1, exec ((decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H).toAlgKernel) s₀
        = some s1
      ∧ ((∀ j k : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + k.val < HIDDEN_DIM) →
            s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride
              = s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + k.val) * hidden_stride →
            j = k) →
          ∀ j : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
            s1.readMem cos_output
              (s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride)
              = xs j)
      ∧ ((∀ j k : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + k.val < HIDDEN_DIM) →
            s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride
              = s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + k.val) * hidden_stride →
            j = k) →
          ∀ j : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
            s1.readMem sin_output
              (s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride)
              = ys j)
      ∧ (∀ r o,
          (r ≠ cos_output ∨ ∀ j : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
              o ≠ s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride) →
          (r ≠ sin_output ∨ ∀ j : Fin BLOCK_H,
            (s₀.pids 0 < NUM_SEQS ∧ s₀.pids 1 * BLOCK_H + j.val < HIDDEN_DIM) →
              o ≠ s₀.pids 0 * cache_stride + (s₀.pids 1 * BLOCK_H + j.val) * hidden_stride) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ : ∃ s1,
      exec ((decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H).toAlgKernel) s₀
        = some s1 := by
    simp [exec, decoding_cache_one_seq_block, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, BlockState.readMemValue]
  have hs1' := hs1
  simp [exec, decoding_cache_one_seq_block, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, BlockState.readMemValue] at hs1'
  refine ⟨s1, hs1, ?_, ?_, ?_⟩
  · -- cos value leg (`WriteInj₁`-gated readback).
    intro hCosInj j hj
    obtain ⟨hSeq, hHid⟩ := hj
    rw [← hs1']
    refine ((BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      sin_output _ _ _ _ _ cos_output _ hRegion).trans ?_)
    refine ((BlockState.scatter_readback_prop_masked_nd_of_true
      (region := cos_output) (shape := [BLOCK_H]) _ _ _ _
      (j, PUnit.unit) ?_ ?_).trans ?_)
    · exact ⟨hSeq, hHid⟩
    · rintro ⟨a, u⟩ ha heq
      cases u
      have hval : a = j := hCosInj a j (by simpa using ha) ⟨hSeq, hHid⟩ (by simpa using heq)
      subst hval
      rfl
    · have hb := hx j hSeq hHid
      rw [← hsrc hSeq] at hb
      simpa [BlockState.readMemValue, hHid, hSeq] using hb
  · -- sin value leg (`WriteInj₂`-gated readback).
    intro hSinInj j hj
    obtain ⟨hSeq, hHid⟩ := hj
    rw [← hs1']
    refine ((BlockState.scatter_readback_prop_masked_nd_of_true
      (region := sin_output) (shape := [BLOCK_H]) _ _ _ _
      (j, PUnit.unit) ?_ ?_).trans ?_)
    · exact ⟨hSeq, hHid⟩
    · rintro ⟨a, u⟩ ha heq
      cases u
      have hval : a = j := hSinInj a j (by simpa using ha) ⟨hSeq, hHid⟩ (by simpa using heq)
      subst hval
      rfl
    · have hb := hy j hSeq hHid
      rw [← hsrc hSeq] at hb
      simpa [BlockState.readMemValue, hHid, hSeq] using hb
  · -- frame off the two active store windows.
    intro r o hc1 hc2
    rw [← hs1']
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
    · rintro k - hPk ⟨hreg, hoff⟩
      rcases hc2 with hne | hno
      · exact hne hreg.symm
      · exact (hno k.1 (by simpa using hPk)) (by simpa using hoff.symm)
    · rintro k - hPk ⟨hreg, hoff⟩
      rcases hc1 with hne | hno
      · exact hne hreg.symm
      · exact (hno k.1 (by simpa using hPk)) (by simpa using hoff.symm)

def decodingCacheIO
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    GatherMasked2DKernelIO₂ₓ₂ where
  kernel := decoding_cache_one_seq_block cos_cache sin_cache lengths cos_output
    sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
  in1 := cos_cache
  in2 := sin_cache
  idxbuf := lengths
  out1 := cos_output
  out2 := sin_output
  B := BLOCK_H
  N := 1
  readx := fun pid₀ _ _ => pid₀
  read := fun _ pid₁ ids i =>
    ids 0 * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  write1 := fun pid₀ pid₁ _ i =>
    pid₀ * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  write2 := fun pid₀ pid₁ _ i =>
    pid₀ * cache_stride + (pid₁ * BLOCK_H + i.val) * hidden_stride
  mask := fun pid₀ _ _ => pid₀ < NUM_SEQS
  readMask := fun pid₀ pid₁ _ i =>
    pid₀ < NUM_SEQS ∧ pid₁ * BLOCK_H + i.val < HIDDEN_DIM

open scoped VeriTile.Triton.GatherMasked2DKernelIO₂ₓ₂ in
specification decoding_cache_correctness
    (cos_cache sin_cache : RegionName) (lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat)
    (hRegion : cos_output ≠ sin_output) :
    decodingCacheIO cos_cache sin_cache lengths cos_output sin_output
        cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
      ⊨ fun _ _ _ xs ys => (xs, ys) := by
  refine GatherMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact decoding_cache_one_seq_block_flattenOk cos_cache sin_cache lengths
      cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
  · intro bounds s ids hpin hbx hbr1 hbr2 hbw1 hbw2
    exact decoding_cache_one_seq_block_traceSafe cos_cache sin_cache lengths
      cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
      bounds s (ids ⟨0, Nat.one_pos⟩)
      (fun h => hpin ⟨0, Nat.one_pos⟩ h)
      (fun h => hbx ⟨0, Nat.one_pos⟩ h)
      (fun i h1 h2 => hbr1 i ⟨h1, h2⟩)
      (fun i h1 h2 => hbr2 i ⟨h1, h2⟩)
      (fun i h1 h2 => hbw1 i ⟨h1, h2⟩)
      (fun i h1 h2 => hbw2 i ⟨h1, h2⟩)
  · intro s₀ ids xs ys hpin hx hy
    exact decoding_cache_one_seq_block_region_run cos_cache sin_cache lengths
      cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H
      hRegion s₀ (ids ⟨0, Nat.one_pos⟩) xs ys
      (fun h => hpin ⟨0, Nat.one_pos⟩ h)
      (fun i h1 h2 => hx i ⟨h1, h2⟩)
      (fun i h1 h2 => hy i ⟨h1, h2⟩)

/-- Pure source-row index for the prefill kernel, phrased over the loaded index
vector `ids` (rather than reading it back from memory as `prefillOriSeqIdx`).
Mirrors `prefillOriSeqIdx`'s `reduceMaxNatDrop` reduction so the `⊨` read window
stays total. -/
noncomputable def prefillOriSeqIdxOfIds
    (idx N_ELEMENTS : Nat) (ids : Fin N_ELEMENTS → Nat) : Nat :=
  let tile : Tile .nat [N_ELEMENTS] := ⟨fun i => ids i.1⟩
  let cond : Tile .bool [N_ELEMENTS] :=
    ⟨fun i => decide (tile.data i ≤ idx)⟩
  let zero : Tile .nat [N_ELEMENTS] := ⟨fun _ => 0⟩
  let masked : Tile .nat [N_ELEMENTS] := Tile.select cond tile zero
  let reduced : Option (Tile .nat (TileShape.eraseAxis [N_ELEMENTS] ⟨0, by simp⟩)) :=
    Tile.reduceMaxNatDrop (shape := [N_ELEMENTS]) ⟨0, by simp⟩ masked
  let maxv : Nat := match reduced with
    | some t => t.data PUnit.unit
    | none => 0
  idx - maxv

/-- Bridge: the pinned index vector `ids` makes the pure `prefillOriSeqIdxOfIds`
coincide with the memory-reading `prefillOriSeqIdx`. -/
theorem prefillOriSeqIdxOfIds_eq
    (s : BlockState) (cumsum_lengths : Region .nat) (BLOCK_SIZE N_ELEMENTS : Nat)
    (ids : Fin N_ELEMENTS → Nat)
    (hpin : ∀ x : Fin N_ELEMENTS,
      ids x = s.readMemValue .nat cumsum_lengths x.val) :
    prefillOriSeqIdxOfIds (s.pids 0 * BLOCK_SIZE + s.pids 1) N_ELEMENTS ids
      = prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS := by
  unfold prefillOriSeqIdxOfIds prefillOriSeqIdx
  simp only [prefillIdx, hpin]
  rfl

theorem prefill_cache_kernel_flattenOk
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) :
    ((prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output sin_output
      cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
      BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [prefill_cache_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem prefill_cache_kernel_traceSafe
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState) (hN : 0 < N_ELEMENTS)
    (hbx : ∀ x : Fin N_ELEMENTS, x.val < bounds cumsum_lengths)
    (hbr1 : ∀ i : Fin HIDDEN_DIM,
      s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length →
      prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS
          * cache_stride + i.val * hidden_stride < bounds cos_cache)
    (hbr2 : ∀ i : Fin HIDDEN_DIM,
      s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length →
      prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS
          * cache_stride + i.val * hidden_stride < bounds sin_cache)
    (hbw1 : ∀ i : Fin HIDDEN_DIM,
      s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length →
      (s.pids 0 * BLOCK_SIZE + s.pids 1) * cache_stride + i.val * hidden_stride
        < bounds cos_output)
    (hbw2 : ∀ i : Fin HIDDEN_DIM,
      s.pids 0 * BLOCK_SIZE + s.pids 1 < total_length →
      (s.pids 0 * BLOCK_SIZE + s.pids 1) * cache_stride + i.val * hidden_stride
        < bounds sin_output) :
    Kernel.TraceSafe bounds
      ((prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output sin_output
        cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [prefill_cache_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
    Tile.reduceMaxNat, Tile.reduceMaxNatDrop, TileShape.axisDim,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, ComparableDType.le, BlockState.readMemValue, hN]
  have hOri : prefillOriSeqIdx s cumsum_lengths BLOCK_SIZE N_ELEMENTS =
      s.pids 0 * BLOCK_SIZE + s.pids 1 -
        Finset.univ.sup' (Finset.univ_nonempty_iff.mpr ⟨⟨0, hN⟩⟩)
          (fun x : Fin N_ELEMENTS =>
            if s.readMemValue TileDType.nat cumsum_lengths.cast x.val ≤
                s.pids 0 * BLOCK_SIZE + s.pids 1 then
              s.readMemValue TileDType.nat cumsum_lengths.cast x.val
            else 0) := by
    simp [prefillOriSeqIdx, prefillIdx, Tile.reduceMaxNatDrop, TileShape.axisDim, hN]
    rfl
  refine ⟨hbx, ?_, ?_, hbw1, hbw2⟩
  · intro a hActive
    have h := hbr1 a hActive
    rw [hOri] at h
    exact h
  · intro a hActive
    have h := hbr2 a hActive
    rw [hOri] at h
    exact h

theorem prefill_cache_kernel_region_run
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (hRegion : cos_output ≠ sin_output) (hN : 0 < N_ELEMENTS)
    (s₀ : BlockState) (ids : Fin N_ELEMENTS → Nat) (xs ys : Fin HIDDEN_DIM → ℝ)
    (hpin : ∀ x : Fin N_ELEMENTS,
      s₀.readMemValue .nat cumsum_lengths x.val = ids x)
    (hx : ∀ i : Fin HIDDEN_DIM, s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
      s₀.readMem cos_cache
        (prefillOriSeqIdxOfIds (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) N_ELEMENTS ids
          * cache_stride + i.val * hidden_stride) = xs i)
    (hy : ∀ i : Fin HIDDEN_DIM, s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
      s₀.readMem sin_cache
        (prefillOriSeqIdxOfIds (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) N_ELEMENTS ids
          * cache_stride + i.val * hidden_stride) = ys i) :
    ∃ s1, exec ((prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin HIDDEN_DIM,
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length) →
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length) →
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride
              = (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + k.val * hidden_stride →
            j = k) →
          ∀ j : Fin HIDDEN_DIM,
            s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
            s1.readMem cos_output
              ((s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride)
              = xs j)
      ∧ ((∀ j k : Fin HIDDEN_DIM,
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length) →
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length) →
            (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride
              = (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + k.val * hidden_stride →
            j = k) →
          ∀ j : Fin HIDDEN_DIM,
            s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
            s1.readMem sin_output
              ((s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride)
              = ys j)
      ∧ (∀ r o,
          (r ≠ cos_output ∨ ∀ j : Fin HIDDEN_DIM,
            s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
              o ≠ (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride) →
          (r ≠ sin_output ∨ ∀ j : Fin HIDDEN_DIM,
            s₀.pids 0 * BLOCK_SIZE + s₀.pids 1 < total_length →
              o ≠ (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) * cache_stride + j.val * hidden_stride) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ : ∃ s1,
      exec ((prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE).toAlgKernel) s₀ = some s1 := by
    simp [exec, prefill_cache_kernel, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
      Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
      Tile.reduceMaxNat, Tile.reduceMaxNatDrop, TileShape.axisDim,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.le, BlockState.readMemValue, hN]
  have hs1' := hs1
  simp [exec, prefill_cache_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.select,
    Tile.reduceMaxNat, Tile.reduceMaxNatDrop, TileShape.axisDim,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt, ComparableDType.le, BlockState.readMemValue, hN] at hs1'
  have hbridge : prefillOriSeqIdx s₀ cumsum_lengths BLOCK_SIZE N_ELEMENTS
      = prefillOriSeqIdxOfIds (s₀.pids 0 * BLOCK_SIZE + s₀.pids 1) N_ELEMENTS ids :=
    (prefillOriSeqIdxOfIds_eq s₀ cumsum_lengths BLOCK_SIZE N_ELEMENTS ids
      (fun x => (hpin x).symm)).symm
  refine ⟨s1, hs1, ?_, ?_, ?_⟩
  · -- cos value leg
    intro hCosInj j hActive
    have hOutInj : Function.Injective
        (fun i : Fin HIDDEN_DIM =>
          prefillOutOffset s₀ cache_stride hidden_stride BLOCK_SIZE i) := by
      intro a b h
      exact hCosInj a b hActive hActive (by simpa [prefillOutOffset, prefillIdx] using h)
    have hc := (prefill_cache_kernel_correct cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE s₀ s1 hRegion hN hOutInj hs1).1 j
    simp only [prefillOutOffset, prefillIdx, prefillCacheOffset, prefillActive] at hc
    rw [if_pos hActive] at hc
    rw [hc, hbridge]
    exact hx j hActive
  · -- sin value leg
    intro hSinInj j hActive
    have hOutInj : Function.Injective
        (fun i : Fin HIDDEN_DIM =>
          prefillOutOffset s₀ cache_stride hidden_stride BLOCK_SIZE i) := by
      intro a b h
      exact hSinInj a b hActive hActive (by simpa [prefillOutOffset, prefillIdx] using h)
    have hc := (prefill_cache_kernel_correct cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE s₀ s1 hRegion hN hOutInj hs1).2 j
    simp only [prefillOutOffset, prefillIdx, prefillCacheOffset, prefillActive] at hc
    rw [if_pos hActive] at hc
    rw [hc, hbridge]
    exact hy j hActive
  · -- frame off the two active store windows.
    intro r o hc1 hc2
    rw [← hs1']
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
    · rintro k - hPk ⟨hreg, hoff⟩
      rcases hc2 with hne | hno
      · exact hne hreg.symm
      · exact (hno k.1 (by simpa using hPk)) (by simpa using hoff.symm)
    · rintro k - hPk ⟨hreg, hoff⟩
      rcases hc1 with hne | hno
      · exact hne hreg.symm
      · exact (hno k.1 (by simpa using hPk)) (by simpa using hoff.symm)

noncomputable def prefillCacheIO
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) :
    GatherMasked2DKernelIO₂ₓ₂ where
  kernel := prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
    sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE
  in1 := cos_cache
  in2 := sin_cache
  idxbuf := cumsum_lengths
  out1 := cos_output
  out2 := sin_output
  B := HIDDEN_DIM
  N := N_ELEMENTS
  readx := fun _ _ j => j.val
  read := fun pid₀ pid₁ ids i =>
    prefillOriSeqIdxOfIds (pid₀ * BLOCK_SIZE + pid₁) N_ELEMENTS ids * cache_stride
      + i.val * hidden_stride
  write1 := fun pid₀ pid₁ _ i =>
    (pid₀ * BLOCK_SIZE + pid₁) * cache_stride + i.val * hidden_stride
  write2 := fun pid₀ pid₁ _ i =>
    (pid₀ * BLOCK_SIZE + pid₁) * cache_stride + i.val * hidden_stride
  mask := fun _ _ _ => True
  readMask := fun pid₀ pid₁ _ _ => pid₀ * BLOCK_SIZE + pid₁ < total_length

open scoped VeriTile.Triton.GatherMasked2DKernelIO₂ₓ₂ in
specification prefill_cache_correctness
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (hRegion : cos_output ≠ sin_output) (hN : 0 < N_ELEMENTS) :
    prefillCacheIO cos_cache sin_cache cumsum_lengths cos_output sin_output
        cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE
      ⊨ fun _ _ _ xs ys => (xs, ys) := by
  refine GatherMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact prefill_cache_kernel_flattenOk cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE
  · intro bounds s ids hpin hbx hbr1 hbr2 hbw1 hbw2
    have hbridge := prefillOriSeqIdxOfIds_eq s cumsum_lengths BLOCK_SIZE N_ELEMENTS
      ids (fun x => (hpin x trivial).symm)
    refine prefill_cache_kernel_traceSafe cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE bounds s hN
      (fun x => hbx x trivial) ?_ ?_
      (fun i h => hbw1 i h)
      (fun i h => hbw2 i h)
    · intro i h; rw [← hbridge]; exact hbr1 i h
    · intro i h; rw [← hbridge]; exact hbr2 i h
  · intro s₀ ids xs ys hpin hx hy
    exact prefill_cache_kernel_region_run cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE hRegion hN s₀ ids xs ys
      (fun x => hpin x trivial)
      (fun i h => hx i h)
      (fun i h => hy i h)

end VeriTile.Bench.TritonBenchG.CacheTransformTriton

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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

```
prefill_cache_kernel_output_summary               ← TOP THEOREM (prefill)
  ├─ (toAlgorithm? = Except.ok _)                  surface lowers (incl. where/max reduction)
  └─ prefill_cache_kernel_compute_correct          ← ComputeCorrect over cos+sin scatter
       └─ prefill_cache_kernel_correct             algorithm-layer readback per cell
decoding_cache_kernel_output_summary              ← TOP THEOREM (decode)
  ├─ (toAlgorithm? = Except.ok _)                  surface lowers
  └─ decoding_cache_kernel_compute_correct         ← ComputeCorrect over cos+sin scatter
       └─ decoding_cache_kernel_correct            algorithm-layer readback per cell
```
Single-store slices (`prefill_cache_{cos,sin}_store_slice_*`,
`decoding_cache_one_seq_block_*`, `decoding_cache_kernel_sin_*`) decompose the
two-output proofs.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased. The source row index is data-dependent: prefill derives it from a
`max(where(cumsum_lens ≤ idx, cumsum_lens, 0))` reduction (modeled as
`reduceMaxNatDrop`, total via a `0 < N_ELEMENTS` hypothesis), decode reads it
from `lengths`. Both top theorems require `cos_output ≠ sin_output` (the two
output buffers are distinct, so neither store clobbers the other) and an
`hOutInj` injectivity side condition (no two cells of one program alias).
`@triton.autotune` is not modeled.
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

/-- Per-kernel output summary for `prefill_cache_kernel`: the DSL surface lowers
to the algorithm layer (including the `where`/`max` source-row reduction), and
both masked cos/sin scatters are compute-correct — every active cell holds the
matching cos/sin cache cell at the derived source row, inactive cells are
preserved. -/
theorem prefill_cache_kernel_output_summary
    (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .nat)
    (cos_output sin_output : RegionName)
    (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hN : 0 < N_ELEMENTS)
    (hOutInj : Function.Injective
      (fun i : Fin HIDDEN_DIM =>
        prefillOutOffset s cache_stride hidden_stride BLOCK_SIZE i)) :
    (∃ alg, (prefill_cache_kernel cos_cache sin_cache cumsum_lengths cos_output
        sin_output cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS
        BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
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
  refine ⟨?_, ?_⟩
  · simp [prefill_cache_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact prefill_cache_kernel_compute_correct cos_cache sin_cache cumsum_lengths
      cos_output sin_output cache_stride hidden_stride total_length HIDDEN_DIM
      N_ELEMENTS BLOCK_SIZE s hRegion hN hOutInj

/-- Per-kernel output summary for `decoding_cache_kernel`: the DSL surface lowers
to the algorithm layer, and both masked cos/sin scatters are compute-correct —
every active cell holds the matching cos/sin cache cell at the `lengths`-selected
source row, inactive cells are preserved. -/
theorem decoding_cache_kernel_output_summary
    (cos_cache sin_cache : RegionName) (lengths : Region .nat) (cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegion : cos_output ≠ sin_output)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE, HIDDEN_DIM] =>
        decodeOutOffset s cache_stride hidden_stride BLOCK_SIZE idx)) :
    (∃ alg, (decoding_cache_kernel cos_cache sin_cache lengths cos_output
        sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
        BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
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
  refine ⟨?_, ?_⟩
  · simp [decoding_cache_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact decoding_cache_kernel_compute_correct cos_cache sin_cache lengths
      cos_output sin_output cache_stride hidden_stride HIDDEN_DIM NUM_SEQS
      BLOCK_SIZE s hRegion hOutInj

end VeriTile.Bench.TritonBenchG.CacheTransformTriton

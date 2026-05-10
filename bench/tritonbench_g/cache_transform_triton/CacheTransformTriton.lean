import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-sequence, one-hidden-block slice of
`cache_transform_triton.py`'s `decoding_cache_kernel`.

This models the decoding branch: load the source cache row from `lengths[seq]`
and copy the same hidden block from cos/sin caches into cos/sin outputs. -/
def decoding_cache_one_seq_block
    (cos_cache sin_cache lengths cos_output sin_output : RegionName)
    (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_H : Nat) :
    ComputeKernel := triton {
  seq = tl.program_id(0)
  hid_block = tl.program_id(1)
  offs = tl.arange(0, $(BLOCK_H))
  hid = hid_block * $(BLOCK_H) + offs
  ori_seq_idx = tl.load(lengths + seq, mask=seq < $(NUM_SEQS),
    other=$(0), dtype=tl.uint64)
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

/-- Algorithm-layer correctness for the one-sequence decoding cache copy. -/
theorem decoding_cache_one_seq_block_correct
    (cos_cache sin_cache lengths cos_output sin_output : RegionName)
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
  · simp [exec, decoding_cache_one_seq_block, stepStmts, stepStmt, evalOp,
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
    (cos_cache sin_cache lengths cos_output sin_output : RegionName)
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

end VeriTile.Bench.TritonBenchG.CacheTransformTriton

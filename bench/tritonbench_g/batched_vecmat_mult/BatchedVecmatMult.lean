import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

The Python wrapper asserts that `M`, `N`, and `K` are divisible by their block
sizes, so this surface keeps the same unmasked block loads and stores. The
Python body vectorizes the `block_m` rows and writes the reduction as
`tl.broadcast(a, b)` followed by `tl.trans(tl.sum(..., axis=2))`.

Allowed mechanical Lean-syntax-only changes apply. -/
def batched_vecmat_surface
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}

/-- The full Python-shaped batched vecmat surface lowers to the algorithm
layer, including the K-block loop, broadcast, reduction, transpose, and output
store. -/
theorem batched_vecmat_surface_toAlgorithm_supported
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M
      BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg := by
  simp [batched_vecmat_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-`m`, one-`n`-block slice of
`batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

This captures a single row of A and one N block of the corresponding B batch:
load `A[m, k]`, load `B[m, n, k]`, reduce over K, and store `output[m, n]`. -/
def batched_vecmat_one_row_block
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = tl.arange(0, $(BLOCK_K))
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  acc = tl.sum(a[None, :] * b, axis=1)
  tl.store(output + m_index * $(dim_n) + offset_n, acc,
    mask=offset_n < $(N))
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val

def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)

def aOffset (s : BlockState) (dim_k : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + j.val

def bOffset (s : BlockState) (dim_n dim_k BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k + j.val

noncomputable def vecmatProdTile
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kj.val < K then
          some (s.readMem A (aOffset s dim_k kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kj.val < K then
          some (s.readMem B (bOffset s dim_n dim_k BLOCK_N ni kj))
        else some (0.0 : ℝ)) }

noncomputable def vecmatSpec
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTile s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-row batched vecmat slice. -/
theorem batched_vecmat_one_row_block_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i))
    (hExec : exec (batched_vecmat_one_row_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem output (outOffset s dim_n BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          vecmatSpec s A B dim_n dim_k N K BLOCK_N BLOCK_K i
        else s.readMem output (outOffset s dim_n BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, batched_vecmat_one_row_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [outOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 1 * BLOCK_N + i.val < N
    · simp [hi, vecmatSpec, vecmatProdTile, aOffset, bOffset, outOffset, nIndex,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex]
      congr 1
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < K then
            some (s.readMem A (s.pids 0 * dim_k + x.val))
          else some 0.0)
          (if x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k + x.val))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < K then
            some (s.readMem A (s.pids 0 * dim_k + x.val))
          else some 0.0)
          (if s.pids 1 * BLOCK_N + i.val < N ∧ x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k + x.val))
          else some 0.0)
      by_cases hxK : x.val < K
      · simp [hxK, hi]
      · simp [hxK]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-row batched vecmat slice. -/
theorem batched_vecmat_one_row_block_compute_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (output, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatSpec s A B dim_n dim_k N K BLOCK_N BLOCK_K i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_one_row_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := batched_vecmat_one_row_block_correct A B output dim_n dim_k N K
    BLOCK_N BLOCK_K s s' hOutInj hExec i
  simpa [hActive] using h

/-- One-row, one-`k_index` loop slice of `batched_vecmat_kernel`.

Unlike `batched_vecmat_one_row_block`, this slice includes the Python loop's
`k_index * block_k` offset in both A and B loads, so it represents an arbitrary
iteration of the K-block loop. -/
def batched_vecmat_one_row_k_block
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = tl.program_id(2)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  acc = tl.sum(a[None, :] * b, axis=1)
  tl.store(output + m_index * $(dim_n) + offset_n, acc,
    mask=offset_n < $(N))
}

def kIndex (s : BlockState) (BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 2 * BLOCK_K + j.val

def aOffsetK (s : BlockState) (dim_k BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + kIndex s BLOCK_K j

def bOffsetK (s : BlockState) (dim_n dim_k BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k +
    kIndex s BLOCK_K j

noncomputable def vecmatProdTileK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kIndex s BLOCK_K kj < K then
          some (s.readMem A (aOffsetK s dim_k BLOCK_K kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kIndex s BLOCK_K kj < K then
          some (s.readMem B (bOffsetK s dim_n dim_k BLOCK_N BLOCK_K ni kj))
        else some (0.0 : ℝ)) }

noncomputable def vecmatSpecK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))

theorem batched_vecmat_one_row_k_block_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i))
    (hExec : exec (batched_vecmat_one_row_k_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem output (outOffset s dim_n BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          vecmatSpecK s A B dim_n dim_k N K BLOCK_N BLOCK_K i
        else s.readMem output (outOffset s dim_n BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, batched_vecmat_one_row_k_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [outOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 1 * BLOCK_N + i.val < N
    · simp [hi, vecmatSpecK, vecmatProdTileK, aOffsetK, bOffsetK, outOffset,
            nIndex, kIndex, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
      congr 1
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
          (if s.pids 1 * BLOCK_N + i.val < N ∧
              s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
      by_cases hxK : s.pids 2 * BLOCK_K + x.val < K
      · simp [hxK, hi]
      · simp [hxK]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem batched_vecmat_one_row_k_block_compute_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_k_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (output, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatSpecK s A B dim_n dim_k N K BLOCK_N BLOCK_K i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_one_row_k_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := batched_vecmat_one_row_k_block_correct A B output dim_n dim_k N K
    BLOCK_N BLOCK_K s s' hOutInj hExec i
  simpa [hActive] using h

/-- Materialized accumulator update for one K-block loop iteration.

The Python kernel keeps `vecmat` in registers and performs `vecmat += ...`.
This proof slice materializes the incoming accumulator in `AccPre`, adds the
current `k_index` contribution, and stores the updated accumulator to `Out`. -/
def batched_vecmat_one_row_k_accum_slice
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = tl.program_id(2)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  mask_n = offset_n < $(N)
  old = tl.load(AccPre + m_index * $(dim_n) + offset_n,
    mask=mask_n, other=0.0)
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  delta = tl.sum(a[None, :] * b, axis=1)
  acc = old + delta
  tl.store(Out + m_index * $(dim_n) + offset_n, acc, mask=mask_n)
}

noncomputable def vecmatAccumSpecK
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun old delta => old + delta)
      (some (s.readMem AccPre (outOffset s dim_n BLOCK_N i)))
      ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
        (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
          (i, PUnit.unit)))

theorem batched_vecmat_one_row_k_accum_slice_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i))
    (hExec : exec (batched_vecmat_one_row_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Out (outOffset s dim_n BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          vecmatAccumSpecK s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K i
        else s.readMem Out (outOffset s dim_n BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, batched_vecmat_one_row_k_accum_slice, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.expandDim, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [outOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 1 * BLOCK_N + i.val < N
    · simp [hi, vecmatAccumSpecK, vecmatSpecK, vecmatProdTileK, aOffsetK,
            bOffsetK, outOffset, nIndex, kIndex, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex]
      congr 1
      congr 1
      change @Finset.sum (Fin BLOCK_K) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map₂ (fun x1 x2 => x1 * x2)
            (if s.pids 2 * BLOCK_K + x.val < K then
              some (s.readMem A
                (s.pids 0 * dim_k + (s.pids 2 * BLOCK_K + x.val)))
            else some 0.0)
            (if s.pids 2 * BLOCK_K + x.val < K then
              some (s.readMem B
                (s.pids 0 * dim_n * dim_k +
                  (s.pids 1 * BLOCK_N + i.val) * dim_k +
                  (s.pids 2 * BLOCK_K + x.val)))
            else some 0.0)) = _
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
          (if s.pids 1 * BLOCK_N + i.val < N ∧
              s.pids 2 * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                (s.pids 2 * BLOCK_K + x.val)))
          else some 0.0)
      by_cases hxK : s.pids 2 * BLOCK_K + x.val < K
      · simp [hxK, hi]
      · simp [hxK]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem batched_vecmat_one_row_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatAccumSpecK s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_one_row_k_accum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := batched_vecmat_one_row_k_accum_slice_correct AccPre A B Out
    dim_n dim_k N K BLOCK_N BLOCK_K s s' hOutInj hExec i
  simpa [hActive] using h

/-- A proof-oriented loop-iteration slice with `k_index` fixed by the caller.

The Python kernel's `for k_index in range(k_blocks)` counter is not a program
ID.  This slice keeps the same accumulator update as
`batched_vecmat_one_row_k_accum_slice`, but binds `k_index` to a literal loop
iteration so wrappers can name the exact `k=0`, `k=1`, ... Python iterations. -/
def batched_vecmat_one_row_const_k_accum_slice
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = $(kIdx)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  mask_n = offset_n < $(N)
  old = tl.load(AccPre + m_index * $(dim_n) + offset_n,
    mask=mask_n, other=0.0)
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  delta = tl.sum(a[None, :] * b, axis=1)
  acc = old + delta
  tl.store(Out + m_index * $(dim_n) + offset_n, acc, mask=mask_n)
}

def withKIndex (s : BlockState) (kIdx : Nat) : BlockState :=
  { s with pids := fun ax => if ax = 2 then kIdx else s.pids ax }

@[simp] theorem withKIndex_pids_zero (s : BlockState) (kIdx : Nat) :
    (withKIndex s kIdx).pids 0 = s.pids 0 := by
  simp [withKIndex]

@[simp] theorem withKIndex_pids_one (s : BlockState) (kIdx : Nat) :
    (withKIndex s kIdx).pids 1 = s.pids 1 := by
  simp [withKIndex]

@[simp] theorem withKIndex_pids_two (s : BlockState) (kIdx : Nat) :
    (withKIndex s kIdx).pids 2 = kIdx := by
  simp [withKIndex]

noncomputable def vecmatConstKAccumSpec
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  vecmatAccumSpecK (withKIndex s kIdx) AccPre A B dim_n dim_k N K
    BLOCK_N BLOCK_K i

theorem batched_vecmat_one_row_const_k_accum_slice_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i))
    (hExec : exec (batched_vecmat_one_row_const_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K kIdx) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem Out (outOffset s dim_n BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          vecmatConstKAccumSpec s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K
            kIdx i
        else s.readMem Out (outOffset s dim_n BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, batched_vecmat_one_row_const_k_accum_slice, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
          Tile.ptrAdd, Tile.expandDim, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [outOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 1 * BLOCK_N + i.val < N
    · simp [hi, vecmatConstKAccumSpec, vecmatAccumSpecK, vecmatSpecK,
            vecmatProdTileK, aOffsetK, bOffsetK, outOffset, nIndex, kIndex,
            withKIndex, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex]
      congr 1
      congr 1
      change @Finset.sum (Fin BLOCK_K) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map₂ (fun x1 x2 => x1 * x2)
            (if kIdx * BLOCK_K + x.val < K then
              some (s.readMem A
                (s.pids 0 * dim_k + (kIdx * BLOCK_K + x.val)))
            else some 0.0)
            (if kIdx * BLOCK_K + x.val < K then
              some (s.readMem B
                (s.pids 0 * dim_n * dim_k +
                  (s.pids 1 * BLOCK_N + i.val) * dim_k +
                    (kIdx * BLOCK_K + x.val)))
            else some 0.0)) = _
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if kIdx * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (kIdx * BLOCK_K + x.val)))
          else some 0.0)
          (if kIdx * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                  (kIdx * BLOCK_K + x.val)))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if kIdx * BLOCK_K + x.val < K then
            some (s.readMem A
              (s.pids 0 * dim_k + (kIdx * BLOCK_K + x.val)))
          else some 0.0)
          (if s.pids 1 * BLOCK_N + i.val < N ∧
              kIdx * BLOCK_K + x.val < K then
            some (s.readMem B
              (s.pids 0 * dim_n * dim_k +
                (s.pids 1 * BLOCK_N + i.val) * dim_k +
                  (kIdx * BLOCK_K + x.val)))
          else some 0.0)
      by_cases hxK : kIdx * BLOCK_K + x.val < K
      · simp [hxK, hi]
      · simp [hxK]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem batched_vecmat_one_row_const_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_const_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K kIdx)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K
          kIdx i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_one_row_const_k_accum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := batched_vecmat_one_row_const_k_accum_slice_correct AccPre A B Out
    dim_n dim_k N K BLOCK_N BLOCK_K kIdx s s' hOutInj hExec i
  simpa [hActive] using h

/-- Python `test_vecmat` uses `K = 128` and `block_k = 64`, hence exactly two
loop iterations.  This names the first accumulator update (`k_index = 0`). -/
theorem batched_vecmat_test_first_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_const_k_accum_slice AccPre A B Out
        dim_n dim_k N 128 BLOCK_N 64 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s AccPre A B dim_n dim_k N 128 BLOCK_N 64
          0 i) :=
  batched_vecmat_one_row_const_k_accum_slice_compute_correct AccPre A B Out
    dim_n dim_k N 128 BLOCK_N 64 0 s hOutInj

/-- Python `test_vecmat`'s second and final accumulator update
(`k_index = 1`) for `K = 128`, `block_k = 64`. -/
theorem batched_vecmat_test_second_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_const_k_accum_slice AccPre A B Out
        dim_n dim_k N 128 BLOCK_N 64 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s AccPre A B dim_n dim_k N 128 BLOCK_N 64
          1 i) :=
  batched_vecmat_one_row_const_k_accum_slice_compute_correct AccPre A B Out
    dim_n dim_k N 128 BLOCK_N 64 1 s hOutInj

/-- Final vectorized `BLOCK_M × BLOCK_N` output store from the Python surface.

The Python wrapper asserts divisibility, so the final store is unmasked:
`tl.store(output + output_tile, vecmat)`. This slice materializes `vecmat` in
`VecmatPre` and proves the full two-dimensional writeback shape. -/
def batched_vecmat_block_output_store_slice
    (VecmatPre output : RegionName) (dim_n BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  offset_m = m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  out_tile = offset_m[:, None] * $(dim_n) + offset_n[None, :]
  vecmat = tl.load(VecmatPre + out_tile)
  tl.store(output + out_tile, vecmat)
}

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def blockOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  mIndex s BLOCK_M idx.1 * dim_n + nIndex s BLOCK_N idx.2.1

noncomputable def blockOutputStoreSpec
    (s : BlockState) (VecmatPre : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  s.readMem VecmatPre (blockOutOffset s dim_n BLOCK_M BLOCK_N idx)

theorem batched_vecmat_block_output_store_slice_correct
    (VecmatPre output : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        blockOutOffset s dim_n BLOCK_M BLOCK_N idx))
    (hExec : exec (batched_vecmat_block_output_store_slice VecmatPre output
        dim_n BLOCK_M BLOCK_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s'.readMem output (blockOutOffset s dim_n BLOCK_M BLOCK_N idx) =
        blockOutputStoreSpec s VecmatPre dim_n BLOCK_M BLOCK_N idx := by
  intro idx
  simp [exec, batched_vecmat_block_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        Tile.expandDim, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_N] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * dim_n +
      (s.pids 1 * BLOCK_N + idx.2.1.val)
  let valueFn : TileIndex [BLOCK_M, BLOCK_N] → ℝ :=
    fun idx => s.readMem VecmatPre (offsetFn idx)
  let P : TileIndex [BLOCK_M, BLOCK_N] → Prop := fun _ => True
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, blockOutOffset, mIndex, nIndex] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := output)
    (s := ((((((s.setReg "m_index" TileDType.nat [] (Tile.scalar (s.pids 0))).setReg
      "n_index" TileDType.nat [] (Tile.scalar (s.pids 1))).setReg
      "offset_m" TileDType.nat [BLOCK_M]
      { data := fun i => s.pids 0 * BLOCK_M + i.1.val }).setReg
      "offset_n" TileDType.nat [BLOCK_N]
      { data := fun i => s.pids 1 * BLOCK_N + i.1.val }).setReg
      "out_tile" TileDType.nat [BLOCK_M, BLOCK_N]
      { data := fun i =>
          (s.pids 0 * BLOCK_M + i.1.val) * dim_n +
            (s.pids 1 * BLOCK_N + i.2.1.val) }).setReg
      "vecmat" TileDType.real [BLOCK_M, BLOCK_N]
      { data := fun i =>
          some (s.readMem VecmatPre
            ((s.pids 0 * BLOCK_M + i.1.val) * dim_n +
              (s.pids 1 * BLOCK_N + i.2.1.val))) }))
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P) hOffsetInj idx
  simpa [offsetFn, valueFn, P, blockOutOffset, blockOutputStoreSpec, mIndex,
    nIndex, TileShape.dropInsertedIndex] using hscatter

theorem batched_vecmat_block_output_store_slice_compute_correct
    (VecmatPre output : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        blockOutOffset s dim_n BLOCK_M BLOCK_N idx)) :
    ComputeCorrect.Realizes
      (kernel := batched_vecmat_block_output_store_slice VecmatPre output
        dim_n BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (output, blockOutOffset s dim_n BLOCK_M BLOCK_N idx))
      (expected := fun idx =>
        blockOutputStoreSpec s VecmatPre dim_n BLOCK_M BLOCK_N idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_block_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact batched_vecmat_block_output_store_slice_correct VecmatPre output
    dim_n BLOCK_M BLOCK_N s s' hOutInj hExec idx

/-! ## Python test-shape wrappers

`test_vecmat` runs the same mathematical shape twice, varying only launch
metadata (`num_warps`, `num_stages`).  The checked shape is `M = N = K = 128`
with `block_m = 16`, `block_n = 32`, and `block_k = 64`, so the Python loop has
exactly two K-block iterations before the final block output store. -/

theorem batched_vecmat_python_row_output_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 32 => outOffset s 128 32 i) := by
  intro a b h
  apply Fin.ext
  simp [outOffset, nIndex] at h
  omega

theorem batched_vecmat_python_block_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 32] => blockOutOffset s 128 16 32 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩
    ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ h
  simp [blockOutOffset, mIndex, nIndex] at h
  have hm : ma = mb := by omega
  have hn : na = nb := by omega
  subst mb
  subst nb
  rfl

/-- Python `test_vecmat` all-output proof-slice coverage: the two concrete
K-block accumulator updates and the final `16 × 32` output block writeback all
realize the checked `(128, 128, 128)` shape. -/
theorem batched_vecmat_python_test_shape_all_outputs_compute_correct
    (Acc0 Acc1 A B VecmatPre output : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_const_k_accum_slice Acc0 A B Acc1
        128 128 128 128 32 64 0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => nIndex s 32 i < 128)
        (fun i => (Acc1, outOffset s 128 32 i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s Acc0 A B 128 128 128 128 32 64 0 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := batched_vecmat_one_row_const_k_accum_slice Acc1 A B VecmatPre
        128 128 128 128 32 64 1)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => nIndex s 32 i < 128)
        (fun i => (VecmatPre, outOffset s 128 32 i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s Acc1 A B 128 128 128 128 32 64 1 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := batched_vecmat_block_output_store_slice VecmatPre output
        128 16 32)
      (initialState := s)
      (write := fun idx : TileIndex [16, 32] =>
        some (output, blockOutOffset s 128 16 32 idx))
      (expected := fun idx =>
        blockOutputStoreSpec s VecmatPre 128 16 32 idx)) := by
  constructor
  · exact batched_vecmat_test_first_k_accum_slice_compute_correct
      Acc0 A B Acc1 128 128 128 32 s
      (batched_vecmat_python_row_output_offset_injective s)
  constructor
  · exact batched_vecmat_test_second_k_accum_slice_compute_correct
      Acc1 A B VecmatPre 128 128 128 32 s
      (batched_vecmat_python_row_output_offset_injective s)
  · exact batched_vecmat_block_output_store_slice_compute_correct VecmatPre
      output 128 16 32 s
      (batched_vecmat_python_block_output_offset_injective s)

end VeriTile.Bench.TritonBenchG.BatchedVecmatMult

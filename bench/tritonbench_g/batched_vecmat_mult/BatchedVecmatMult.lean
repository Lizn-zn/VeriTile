import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of `batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

The Python wrapper asserts that `M`, `N`, and `K` are divisible by their block
sizes, so this surface keeps the same unmasked block loads and stores. The
Python body vectorizes the `block_m` rows and writes the reduction as
`tl.broadcast(a, b)` followed by `tl.trans(tl.sum(..., axis=2))`; this spells
out the same computation as an explicit `block_m` loop because the current DSL
only has rank-1-to-rank-2 slice insertion syntax. -/
def batched_vecmat_surface
    (A B output : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(axis=0)
  n_index = tl.program_id(axis=1)
  offsets_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offsets_k = tl.arange(0, $(BLOCK_K))
  for m_inner in range($(0), $(BLOCK_M), $(1)) {
    m_offset = m_index * $(BLOCK_M) + m_inner
    vecmat = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
    for k_index in range($(0), $(dim_k / BLOCK_K), $(1)) {
      k_offsets = k_index * $(BLOCK_K) + offsets_k
      a = tl.load(A + m_offset * $(dim_k) + k_offsets)
      b = tl.load(B + m_offset * $(dim_n) * $(dim_k) +
        offsets_n[:, None] * $(dim_k) + k_offsets[None, :])
      vecmat += tl.sum(a[None, :] * b, axis=1)
    }
    tl.store(output + m_offset * $(dim_n) + offsets_n, vecmat)
  }
}

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
  · simp [exec, batched_vecmat_one_row_block, stepStmts, stepStmt, evalOp,
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
      rfl
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

end VeriTile.Bench.TritonBenchG.BatchedVecmatMult

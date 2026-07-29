import VeriTile.Triton

/-!
# `batched_vecmat_mult` — strict per-kernel correctness

`batched_vecmat_kernel` is a batched vector-matrix multiply: for batch row tile
`m_index` and column tile `n_index`, program `(m,n)` loops over K blocks,
loading an `A` tile `[block_m, block_k]` and a `B` tile
`[block_n, block_m, block_k]`, broadcasts `A`, reduces `tl.sum(expanded_a * b,
axis=2)`, transposes, accumulates into `vecmat`, and stores the
`[block_m, block_n]` tile to `output`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`batched_vecmat_kernel[grid](...)`, the grid
`(M//block_m, N//block_n)`, scheduling, and how the runtime composes per-program
writes into one buffer) is the *trusted boundary*, not a proof obligation here.
Program ids are universally quantified, so the per-program statements cover
every program of the grid.

## Proof architecture

The headline result is a **genuine closed-form GEMV** over the full dynamic
K-loop: every output cell of the computed `BLOCK_M × BLOCK_N` tile equals the
independent batched vector-matrix reference
`out[m,n] = Σ_{kk < dim_k} A[m,kk] · B[m,n,kk]` (over `ℝ`), NOT the kernel's own
emitted value.

```
batched_vecmat_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
  └─ vecmat_exec_closed_form                       ← exec-side closed form (every cell = ∑ A·B)
       ├─ vecmat_preLoop      (P 0: vecmat = 0, scalar regs seeded)
       ├─ vecmat_step         (one K-block: vecmat += trans(reduceSum(a·b)) advances the partial sum)
       ├─ vecmat_postLoop     (final unmasked store = the closed form)
       └─ forRangeDyn_inv     (dynamic-loop invariant principle, drives the K-loop)
```

`forRangeDyn_inv` is the master invariant principle for the *dynamic*
`Stmt.forRangeDyn` that `for k_index in range(k_blocks)` lowers to (the K-block
count `k_blocks = dim_k // block_k` is a runtime register, not a literal). It
resolves the runtime bounds via `evalOp` and then drives the loop body by the
caller's invariant `P`, built here on top of the existing `forRangeAux_inv`.

The supporting one-row reduction slices (`vecmatSpec`, `vecmatSpecK`,
`vecmatAccumSpecK`, the block output store) remain as verified lemmas.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `num_warps`/`num_stages`
are not modeled. The host launch (grid `(M//block_m, N//block_n)`, scheduling,
and how the runtime composes per-program writes into one buffer) is the trusted
boundary; the per-program statement is universally quantified over the initial
state `s`, so it covers every program of the grid. The layout contract is the
kernel's own row-major pointer arithmetic: `A[m,kk]` at `A + m·dim_k + kk`,
`B[m,n,kk]` at `B + m·dim_n·dim_k + n·dim_k + kk`, `out[m,n]` at
`output + m·dim_n + n`. The contracted dimension is `dim_k = BLOCK_K · numKBlocks`
(the kernel asserts `K % block_k == 0`); the loop runs `numKBlocks` iterations.
Preconditions: `0 < BLOCK_K`, output-offset injectivity (distinct lanes hit
distinct addresses), and a clean initial `undef`.
-/

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `batched_vecmat_closed_form_correct` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_block_output_store_slice VecmatPre output
        dim_n BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (output, blockOutOffset s dim_n BLOCK_M BLOCK_N idx))
      (expected := fun idx =>
        blockOutputStoreSpec s VecmatPre dim_n BLOCK_M BLOCK_N idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_block_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact batched_vecmat_block_output_store_slice_correct VecmatPre output
    dim_n BLOCK_M BLOCK_N s s' hOutInj hExec idx


/-! ## Genuine GEMV closed-form correctness over the full dynamic K-loop

`for k_index in range(k_blocks)` (with `k_blocks = dim_k // block_k` a runtime
register) lowers to `Stmt.forRangeDyn`. The lemmas below build a closed-form
characterization of the kernel's `output` against the independent batched
vector-matrix reference `out[m,n] = Σ_{kk < dim_k} A[m,kk] · B[m,n,kk]`. -/

/-- Master invariant principle for the *dynamic* `forRangeDyn` loop. Resolves
the runtime bounds via `evalOp`, then drives the body by the caller's invariant
`P`, exposing the loop-exit counter `final` (with `stop ≤ final`). -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simp only [Option.bind]
  exact h_aux

/-! ### Genuine batched vector-matrix (GEMV) spec -/

/-- Global output row of tile lane `i`: `m_index · BLOCK_M + i`. -/
def gRow (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

/-- Global output column of tile lane `j`: `n_index · BLOCK_N + j`. -/
def gCol (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val

/-- `A[m, kk] = readMem A (gRow i · dim_k + kk)` (the kernel's row-major A layout). -/
noncomputable def aElem (s : BlockState) (A : RegionName) (dim_k BLOCK_M : Nat)
    (i : Fin BLOCK_M) (kk : Nat) : ℝ :=
  s.readMem A (gRow s BLOCK_M i * dim_k + kk)

/-- `B[m, n, kk] = readMem B (gRow i · dim_n · dim_k + gCol j · dim_k + kk)`
(the kernel's row-major B layout). -/
noncomputable def bElem (s : BlockState) (B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (kk : Nat) : ℝ :=
  s.readMem B (gRow s BLOCK_M i * dim_n * dim_k + gCol s BLOCK_N j * dim_k + kk)

/-- **Genuine GEMV spec**: `out[i,j] = Σ_{kk < BLOCK_K·numKBlocks} A[i,kk] · B[i,j,kk]`. -/
noncomputable def gemvSpec (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun kk => aElem s A dim_k BLOCK_M i kk
      * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j kk)

/-- Partial GEMV accumulator after `c` K-blocks: `Σ_{kk < c·BLOCK_K} A·B`. -/
noncomputable def gemvPartial (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun kk => aElem s A dim_k BLOCK_M i kk
      * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j kk)

/-- One-block step of the partial accumulator: the new block contributes the
dot over the `BLOCK_K` keys `c·BLOCK_K + e`. -/
theorem gemvPartial_succ (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) :
    gemvPartial s A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K i j (c + 1)
      = gemvPartial s A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s A dim_k BLOCK_M i (c * BLOCK_K + e.val)
              * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + e.val)) := by
  unfold gemvPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => aElem s A dim_k BLOCK_M i (c * BLOCK_K + e)
        * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + e)]

/-! ### Body decomposition (prefix ++ for-loop ++ store) -/

/-- The 6-statement K-loop body, transcribed from the elaborated surface. -/
def vecmatLoopBody (A B : RegionName) (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    List Stmt :=
  [ Stmt.assign TileDType.nat [BLOCK_M, BLOCK_K] "a_tile"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))),
    Stmt.assign TileDType.real [BLOCK_M, BLOCK_K] "a"
      (Op.load TileDType.real (MemAccess.region A (Op.ref TileDType.nat [BLOCK_M, BLOCK_K] "a_tile")) MaskOpt.none),
    Stmt.assign TileDType.nat [BLOCK_N, BLOCK_M, BLOCK_K] "b_tile"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))),
    Stmt.assign TileDType.real [BLOCK_N, BLOCK_M, BLOCK_K] "b"
      (Op.load TileDType.real (MemAccess.region B (Op.ref TileDType.nat [BLOCK_N, BLOCK_M, BLOCK_K] "b_tile")) MaskOpt.none),
    Stmt.assign TileDType.real [1, BLOCK_M, BLOCK_K] "expanded_a"
      (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [BLOCK_M, BLOCK_K] "a")),
    Stmt.assign TileDType.real [BLOCK_M, BLOCK_N] "vecmat"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame
        (Op.ref TileDType.real [BLOCK_M, BLOCK_N] "vecmat")
        (Op.transpose (batch := []) (M := BLOCK_N) (N := BLOCK_M)
          (Op.reduceSum ⟨2, by simp⟩ Bool.false
            (Op.mul NumericDType.real Broadcast.nil.consSame.consSame.consL
              (Op.ref TileDType.real [1, BLOCK_M, BLOCK_K] "expanded_a")
              (Op.ref TileDType.real [BLOCK_N, BLOCK_M, BLOCK_K] "b"))))) ]

/-- Post-loop unmasked store of `vecmat` to `output_tile`. -/
def vecmatStoreStmt (output : RegionName) (BLOCK_M BLOCK_N : Nat) : Stmt :=
  Stmt.store TileDType.real [BLOCK_M, BLOCK_N]
    (MemAccess.region output (Op.ref TileDType.nat [BLOCK_M, BLOCK_N] "output_tile"))
    (Op.ref TileDType.real [BLOCK_M, BLOCK_N] "vecmat") MaskOpt.none

/-- Body decomposition: prefix (5) ++ [forRangeDyn loop, store]. By `rfl`. -/
theorem vecmat_body_split (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 5
        ++ (Stmt.forRangeDyn "k_index" (Op.constNat 0) (Op.ref TileDType.nat [] "k_blocks") (Op.constNat 1)
              (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)
            :: vecmatStoreStmt output BLOCK_M BLOCK_N :: []) := by
  rfl

/-! ### Loop invariant -/

/-- **Loop invariant** (counter `c` = number of K-blocks done so far).

After `c` K-blocks: program ids and `mem`/`undef` fixed; the `m_index` /
`n_index` / `output_tile` registers seeded; and `vecmat` equals the partial
GEMV accumulator `gemvPartial … c`. -/
noncomputable def vecmatInvariant
    (A B output : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧
  (s.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [BLOCK_M, BLOCK_N] "output_tile" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### evalOp helpers for the loop-body statements -/

/-- `a_tile` eval: cell `(i,k) = (gRow i)·dim_k + (c·BLOCK_K + k)`. -/
theorem vecmat_atile_eval (s : BlockState) (dim_k BLOCK_M BLOCK_K mi ki : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hk : s.regs .nat [] "k_index" = some (Tile.scalar ki)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
          (mi * BLOCK_M + idx.1.val) * dim_k + (ki * BLOCK_K + idx.2.1.val)⟩ : Tile .nat [BLOCK_M, BLOCK_K]) := by
  simp only [evalOp.eq_def, hm, hk, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]

/-- `b_tile` eval: cell `(j,i,k) = (gRow i)·dim_n·dim_k + (gCol j)·dim_k + (c·BLOCK_K + k)`. -/
theorem vecmat_btile_eval (s : BlockState) (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K mi ni ki : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hn : s.regs .nat [] "n_index" = some (Tile.scalar ni))
    (hk : s.regs .nat [] "k_index" = some (Tile.scalar ki)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_M, BLOCK_K] =>
          (mi * BLOCK_M + idx.2.1.val) * dim_n * dim_k
            + (ni * BLOCK_N + idx.1.val) * dim_k
            + (ki * BLOCK_K + idx.2.2.1.val)⟩ : Tile .nat [BLOCK_N, BLOCK_M, BLOCK_K]) := by
  simp only [evalOp.eq_def, hm, hn, hk, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, TileShape.insertAxis,
    NumericDType.add, NumericDType.mul]

/-- No-mask region load eval: reads `readMem` at each offset. -/
theorem vecmat_load_eval {shape : TileShape} (s : BlockState) (R : RegionName) (name : RegName)
    (offs : Tile .nat shape) (h : s.regs .nat shape name = some offs) :
    evalOp (Op.load TileDType.real (MemAccess.region R (Op.ref TileDType.nat shape name)) MaskOpt.none) s
      = some (⟨fun i => some (s.readMem R (offs.data i))⟩ : Tile .real shape) := by
  simp only [evalOp_load_region_none, evalOp_ref, h, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-- **`vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))` eval.**

Given the loaded `expanded_a`/`b` tiles whose lanes are the genuine `A`/`B`
elements at K-offset `c·BLOCK_K + k`, and `vecmat` holding the partial sum after
`c` blocks, the update yields the partial sum after `c + 1` blocks. -/
theorem vecmat_update_eval (s s0 : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c : Nat)
    (aT : Tile .real [BLOCK_M, BLOCK_K]) (bT : Tile .real [BLOCK_N, BLOCK_M, BLOCK_K])
    (hvm : s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some
        ⟨fun idx => some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩)
    (hea : s.regs .real [1, BLOCK_M, BLOCK_K] "expanded_a" = some (Tile.expandDim ⟨0, by simp⟩ aT))
    (hb : s.regs .real [BLOCK_N, BLOCK_M, BLOCK_K] "b" = some bT)
    (haval : ∀ (i : Fin BLOCK_M) (k : Fin BLOCK_K),
        aT.data (i, k, PUnit.unit) = some (aElem s0 A dim_k BLOCK_M i (c * BLOCK_K + k.val)))
    (hbval : ∀ (j : Fin BLOCK_N) (i : Fin BLOCK_M) (k : Fin BLOCK_K),
        bT.data (j, i, k, PUnit.unit)
          = some (bElem s0 B dim_n dim_k BLOCK_M BLOCK_N i j (c * BLOCK_K + k.val))) :
    evalOp
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame
        (Op.ref TileDType.real [BLOCK_M, BLOCK_N] "vecmat")
        (Op.transpose (batch := []) (M := BLOCK_N) (N := BLOCK_M)
          (Op.reduceSum ⟨2, by simp⟩ Bool.false
            (Op.mul NumericDType.real Broadcast.nil.consSame.consSame.consL
              (Op.ref TileDType.real [1, BLOCK_M, BLOCK_K] "expanded_a")
              (Op.ref TileDType.real [BLOCK_N, BLOCK_M, BLOCK_K] "b"))))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 (c + 1))⟩
            : Tile .real [BLOCK_M, BLOCK_N]) := by
  simp only [evalOp.eq_def, hvm, hea, hb, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  -- reduce the per-lane value to vecmat[i,j] + Σ_k a[i,k]·b[j,i,k]
  simp only [Tile.bop, Tile.transpose, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, WithBot.realAdd]
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  simp only [TileShape.insertAxisIndex, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.dropInsertedIndex, haval, hbval, WithBot.realMul, Option.map₂,
    Option.bind, Option.map, WithBot.some_eq_coe]
  -- the block sum is now `∑ ↑(aElem · bElem)`; collapse to `↑(∑ …)` and fold
  rw [← WithBot.coe_sum, gemvPartial_succ]
  rfl


set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one
block. The `vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))` update adds the
`c`-th block's dot to the partial GEMV accumulator. -/
theorem vecmat_step (A B output : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (c : Nat) (s : BlockState)
    (hinv : vecmatInvariant A B output s0 dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c s) :
    ∃ s', stepStmts (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)
        (s.setReg "k_index" .nat [] (Tile.scalar c)) = some s'
      ∧ vecmatInvariant A B output s0 dim_n dim_k BLOCK_M BLOCK_N BLOCK_K (c + 1) s' := by
  obtain ⟨hpids, hm, hn, hot, hvm, hundef, hmem⟩ := hinv
  set sk := s.setReg "k_index" .nat [] (Tile.scalar c) with hsk
  have hmk : sk.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0)) := by simp [hsk, hm]
  have hnk : sk.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1)) := by simp [hsk, hn]
  have hkk : sk.regs .nat [] "k_index" = some (Tile.scalar c) := by simp [hsk]
  have hvmk : sk.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (gemvPartial s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩ := by
    simp [hsk, hvm]
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  -- a_tile / b_tile offset tiles
  set atT : Tile .nat [BLOCK_M, BLOCK_K] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.1.val) * dim_k + (c * BLOCK_K + idx.2.1.val)⟩ with hatT
  set btT : Tile .nat [BLOCK_N, BLOCK_M, BLOCK_K] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.2.1.val) * dim_n * dim_k
        + (s0.pids 1 * BLOCK_N + idx.1.val) * dim_k + (c * BLOCK_K + idx.2.2.1.val)⟩ with hbtT
  set aT : Tile .real [BLOCK_M, BLOCK_K] :=
    ⟨fun i => some (sk.readMem A (atT.data i))⟩ with haT
  set bT : Tile .real [BLOCK_N, BLOCK_M, BLOCK_K] :=
    ⟨fun i => some (sk.readMem B (btT.data i))⟩ with hbT
  -- chain the six assigns
  simp only [vecmatLoopBody]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_atile_eval sk dim_k BLOCK_M BLOCK_K (s0.pids 0) c hmk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_load_eval _ A "a_tile" atT (by simp [haT, hatT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_btile_eval _ dim_n dim_k BLOCK_M BLOCK_N BLOCK_K (s0.pids 0) (s0.pids 1) c
          (by simp [hmk]) (by simp [hnk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_load_eval _ B "b_tile" btT (by simp [hbtT])))]
  -- expanded_a = expandDim 0 a
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (shape := [1, BLOCK_M, BLOCK_K]) (name := "expanded_a")
        (e := Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [BLOCK_M, BLOCK_K] "a"))
        (v := Tile.expandDim ⟨0, by simp⟩ aT)
        (by
          refine evalOp_expandDim_ref_of_regs _ _ _ _ _ aT ?_
          simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
            String.reduceEq, not_false_eq_true, haT, BlockState.setReg_readMem]))]
  -- vecmat += trans(reduceSum(expanded_a * b, axis 2))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_update_eval _ s0 A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c aT bT
          (by simp [hvmk])
          (by simp only [BlockState.setReg_same]; rfl)
          (by simp [hbT, BlockState.setReg_readMem])
          (by
            intro i k
            simp only [haT, hatT, aElem, gRow, hrmem])
          (by
            intro j i k
            simp only [hbT, hbtT, bElem, gRow, gCol, hrmem])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp [hsk, hpids]
  · -- m_index
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hm]
  · -- n_index
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hn]
  · -- output_tile
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hot]
  · -- vecmat = gemvPartial (c+1)
    simp only [BlockState.setReg_same]
  · -- undef
    intro rg o; simp [hsk, hundef]
  · -- mem
    funext region offset
    simp only [BlockState.setReg_mem]
    rw [show sk.mem region offset = s.mem region offset from by simp [hsk], hmem]

/-! ### preLoop and postLoop -/

/-- `output_tile` eval: cell `(i,j) = (gRow i)·dim_n + (gCol j)`. -/
theorem vecmat_otile_eval (s : BlockState) (dim_n BLOCK_M BLOCK_N mi ni : Nat)
    (hm : s.regs .nat [] "m_index" = some (Tile.scalar mi))
    (hn : s.regs .nat [] "n_index" = some (Tile.scalar ni)) :
    evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_n))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
            (Op.arange BLOCK_N)))) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          (mi * BLOCK_M + idx.1.val) * dim_n + (ni * BLOCK_N + idx.2.1.val)⟩ : Tile .nat [BLOCK_M, BLOCK_N]) := by
  simp only [evalOp.eq_def, hm, hn, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.expandDim, Tile.scalar, Tile.vec, Broadcast.leftIndex,
    Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 2000000 in
/-- **preLoop** (statements 0–4): from a clean input state (`undef = 0`), the
prologue (`m_index`, `n_index`, `output_tile`, `vecmat = 0`, `k_blocks`) steps to
a state satisfying `vecmatInvariant … 0` — the base case (`vecmat = 0`). -/
theorem vecmat_preLoop (A B output : RegionName) (s : BlockState)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M
        BLOCK_N BLOCK_K).toAlgKernel.body.take 5) s = some s'
      ∧ vecmatInvariant A B output s dim_n dim_k BLOCK_M BLOCK_N BLOCK_K 0 s'
      ∧ s'.regs .nat [] "k_blocks" = some (Tile.scalar (dim_k / BLOCK_K)) := by
  rw [show (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 5
        = [Stmt.assign .nat [] "m_index" (Op.programId 0),
           Stmt.assign .nat [] "n_index" (Op.programId 1),
           Stmt.assign .nat [BLOCK_M, BLOCK_N] "output_tile"
             (Op.add NumericDType.nat Broadcast.nil.consL.consR
               (Op.mul NumericDType.nat Broadcast.scalarR
                 (Op.expandDim ⟨1, by simp⟩
                   (Op.add NumericDType.nat Broadcast.scalarL
                     (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                     (Op.arange BLOCK_M)))
                 (Op.constNat dim_n))
               (Op.expandDim ⟨0, by simp⟩
                 (Op.add NumericDType.nat Broadcast.scalarL
                   (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                   (Op.arange BLOCK_N)))),
           Stmt.assign .real [BLOCK_M, BLOCK_N] "vecmat" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)),
           Stmt.assign .nat [] "k_blocks"
             (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat dim_k) (Op.constNat BLOCK_K))]
      from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.programId 0)
        (v := Tile.scalar (s.pids 0)) (by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.programId 1)
        (v := Tile.scalar (s.pids 1)) (by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_otile_eval _ dim_n BLOCK_M BLOCK_N (s.pids 0) (s.pids 1)
          (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.full [BLOCK_M, BLOCK_N] (Op.const 0))
        (v := (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]))
        (by simp [evalOp_full, evalOp_const, Option.bind]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (e := Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat dim_k) (Op.constNat BLOCK_K))
        (v := Tile.scalar (dim_k / BLOCK_K))
        (by simp [evalOp, Tile.bop, IntegralDType.floorDiv, NumericDType.div]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · simp
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · -- output_tile seeded
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
  · -- vecmat = 0 = gemvPartial 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some ?_
    ext idx
    simp only [gemvPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · intro rg o
    simp [hundef]
  · funext region offset
    simp only [BlockState.setReg_mem]
  · -- k_blocks readback
    simp only [BlockState.setReg_same]

/-- The output store address for tile lane `(i,j)`: `gRow i · dim_n + gCol j`. -/
def vecmatOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  gRow s BLOCK_M idx.1 * dim_n + gCol s BLOCK_N idx.2.1

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the unmasked store
of `vecmat` to `output_tile` writes the genuine GEMV value `gemvSpec` at every
output lane (given the output-offset map is injective). -/
theorem vecmat_postLoop (A B output : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N))
    (st : BlockState)
    (hinv : vecmatInvariant A B output s0 dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        numKBlocks st) :
    ∃ sfin, stepStmts (vecmatStoreStmt output BLOCK_M BLOCK_N :: []) st = some sfin
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          sfin.readMem output (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N idx)
            = gemvSpec s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
                idx.1 idx.2.1 := by
  obtain ⟨hpids, hm, hn, hot, hvm, hundef, hmem⟩ := hinv
  set vmT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => some (gemvPartial s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        idx.1 idx.2.1 numKBlocks)⟩ with hvmT
  set otT : Tile .nat [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩
    with hotT
  -- the store evaluates to the scatter foldl
  have hstore : stepStmt (vecmatStoreStmt output BLOCK_M BLOCK_N) st
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_N]).foldl
          (fun acc i =>
            acc.writeMem output (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N i)
              (gemvPartial s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
                i.1 i.2.1 numKBlocks)) st) := by
    simp only [vecmatStoreStmt, stepStmt]
    rw [show evalOp (Op.ref .real [BLOCK_M, BLOCK_N] "vecmat") st = some vmT from by
          rw [evalOp_ref, hvm]]
    rw [show evalOp (Op.ref .nat [BLOCK_M, BLOCK_N] "output_tile") st = some otT from by
          rw [evalOp_ref, hot]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ st (fun acc i _ => ?_))
    simp only [if_true, otT, hotT, vecmatOutOffset, gRow, gCol, vmT, hvmT,
      BlockState.writeMemTyped_real, Region.cast_id, FloatDType.real_storeValue,
      WithBot.unbotD_some]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [BlockState.scatter_readback_nd (region := output) (s := st)
      (offsetFn := vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N)
      (valueFn := fun i => gemvPartial s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        i.1 i.2.1 numKBlocks)
      hInj idx]
  simp only [gemvSpec, gemvPartial, Nat.mul_comm numKBlocks BLOCK_K]

/-! ### Composition: full exec closed form -/

set_option maxHeartbeats 4000000 in
/-- **Top exec reduction**: composes `vecmat_preLoop` + `vecmat_step` (driven by
`forRangeDyn_inv`) + `vecmat_postLoop` into the full `exec` result. Every output
lane equals the genuine GEMV value `gemvSpec`. -/
theorem vecmat_exec_closed_form (A B output : RegionName) (s : BlockState)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (vecmatOutOffset s dim_n BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    (match exec (batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
        BLOCK_M BLOCK_N BLOCK_K) s with
      | some s' => s'.readMem output (vecmatOutOffset s dim_n BLOCK_M BLOCK_N idx)
      | none => (0.0 : ℝ)) =
      gemvSpec s A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
        idx.1 idx.2.1 := by
  set dim_k := BLOCK_K * numKBlocks with hdimk
  -- preLoop establishes the invariant at c = 0 and the k_blocks readback
  obtain ⟨s0, hpre_eq, hP0, hkb⟩ :=
    vecmat_preLoop A B output s _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K hundef
  -- k_blocks resolves to numKBlocks
  have hnum : dim_k / BLOCK_K = numKBlocks := by
    rw [hdimk, Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  -- drive the dynamic K-loop, carrying `c ≤ numKBlocks` so the exit counter pins down
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRangeDyn_inv (idx := "k_index")
      (start := 0) (stop := numKBlocks) (step := 1)
      (startOp := Op.constNat 0) (stopOp := Op.ref .nat [] "k_blocks") (stepOp := Op.constNat 1)
      (s_init := s0)
      (P := fun c st =>
        vecmatInvariant A B output s dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c st ∧ c ≤ numKBlocks)
      (by simp [evalOp])
      (by rw [evalOp_ref, hkb, hnum])
      (by simp [evalOp])
      (by omega) ⟨hP0, by omega⟩
      (fun c st hlt ⟨hinv, _⟩ => by
        obtain ⟨st', hstep, hinv'⟩ :=
          vecmat_step A B output s dim_n dim_k BLOCK_M BLOCK_N BLOCK_K c st hinv
        exact ⟨st', by simpa using hstep, hinv', by omega⟩)
  -- loop exit: final = numKBlocks
  have hfinalEq : final = numKBlocks := le_antisymm hPLoop.2 hfinal
  subst hfinalEq
  -- postLoop reads off the closed form
  obtain ⟨sfin, hTail, hpost⟩ :=
    vecmat_postLoop A B output s dim_n dim_k BLOCK_M BLOCK_N BLOCK_K final
      hInj sLoop hPLoop.1
  have hexec : exec (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) s
      = some sfin := by
    rw [exec, vecmat_body_split A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Closed-form GEMV correctness for `batched_vecmat_mult` (general statement).**

For arbitrary `dim_n`, tile dims `BLOCK_M`/`BLOCK_N`, K-block size `BLOCK_K`, and
K-block count `numKBlocks` (so the contracted dim is `dim_k = BLOCK_K · numKBlocks`),
every output cell of the computed `BLOCK_M × BLOCK_N` tile equals the genuine
batched vector-matrix value `Σ_{kk < dim_k} A[m,kk] · B[m,n,kk]` (over ℝ) — NOT
the kernel's own executed value.

Layout: `A[m,kk]` at `A + m·dim_k + kk`, `B[m,n,kk]` at
`B + m·dim_n·dim_k + n·dim_k + kk`, `out[m,n]` at `output + m·dim_n + n` (the
kernel's row-major pointer arithmetic). Preconditions: `0 < BLOCK_M`,
`0 < BLOCK_N`, `0 < BLOCK_K`, output-offset injectivity, clean initial `undef`. -/
specification batched_vecmat_closed_form_correct
    (A B output : RegionName) (s : BlockState)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (vecmatOutOffset s dim_n BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
        BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (output, vecmatOutOffset s dim_n BLOCK_M BLOCK_N idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        gemvSpec s A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
          idx.1 idx.2.1) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [batched_vecmat_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := vecmat_exec_closed_form A B output s0 _dim_m dim_n BLOCK_M BLOCK_N BLOCK_K
    numKBlocks hBM hBN hBK hInj hundef idx
  have hExec2 : exec (batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
      BLOCK_M BLOCK_N BLOCK_K) s0 = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real] using hmain

/-! ## Python test-shape summary

`test_vecmat` runs `M = N = K = 128`, `block_m = 16`, `block_n = 32`,
`block_k = 64`, so `numKBlocks = 128 / 64 = 2`. -/

end Correct_without_Rounding

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre)

Everything below is purely additive; the exact surface above is untouched.
Structure of the `execR R` story: this kernel has **zero rounding events** —
the two unmasked region loads, the `expanded_a * b` product, the axis-2
`tl.sum`, the `tl.trans` and the accumulator add are all at `.real`, and the
terminal store is an unmasked `.real` region store, which `stepStmtR`
delegates to the exact `writeMemTyped` (a `.real` write is not a
quantization). The 5-statement prologue and the whole `forRangeDyn` K-loop
are therefore cast-free and collapse verbatim onto the exact stepper
(`stepForRangeAuxR_castFree`), reusing the proven `vecmat_preLoop` /
`vecmat_step` / `forRangeDyn_inv` invariant stack unchanged; only the single
store statement is re-proved on the `R` side (`vecmat_postLoopR`), where the
`.real` write coincides with the `writeMemAsR R .real` rounded write because
`R.storeValue .real` **is** the exact store demotion
(`RoundingModel.storeValue_real`). The skin's readback contract then carries
`R.round .real`, which is the identity by the model's defining `round_real`. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₂

/-! ### Body decomposition names and cast-free collapses -/

/-- The 5-statement prologue (`m_index`, `n_index`, `output_tile`,
`vecmat = 0`, `k_blocks`) as an explicit list. -/
private def vecmatPrologue (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [Stmt.assign .nat [] "m_index" (Op.programId 0),
   Stmt.assign .nat [] "n_index" (Op.programId 1),
   Stmt.assign .nat [BLOCK_M, BLOCK_N] "output_tile"
     (Op.add NumericDType.nat Broadcast.nil.consL.consR
       (Op.mul NumericDType.nat Broadcast.scalarR
         (Op.expandDim ⟨1, by simp⟩
           (Op.add NumericDType.nat Broadcast.scalarL
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
             (Op.arange BLOCK_M)))
         (Op.constNat dim_n))
       (Op.expandDim ⟨0, by simp⟩
         (Op.add NumericDType.nat Broadcast.scalarL
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
           (Op.arange BLOCK_N)))),
   Stmt.assign .real [BLOCK_M, BLOCK_N] "vecmat" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)),
   Stmt.assign .nat [] "k_blocks"
     (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat dim_k) (Op.constNat BLOCK_K))]

/-- The prologue is exactly `body.take 5`. By `rfl`. -/
private theorem vecmat_take5_eq (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N
        BLOCK_K).toAlgKernel.body.take 5
      = vecmatPrologue dim_n dim_k BLOCK_M BLOCK_N BLOCK_K := rfl

/-- `vecmat_body_split` with the prologue named. -/
private theorem vecmat_body_split' (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    (batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = vecmatPrologue dim_n dim_k BLOCK_M BLOCK_N BLOCK_K
        ++ (Stmt.forRangeDyn "k_index" (Op.constNat 0) (Op.ref TileDType.nat [] "k_blocks")
              (Op.constNat 1) (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)
            :: vecmatStoreStmt output BLOCK_M BLOCK_N :: []) := by
  rw [vecmat_body_split A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K,
    vecmat_take5_eq A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K]

/-- The prologue is cast-free: the two `program_id`s, the `.nat` index tile,
the zero accumulator and the `.nat` floor division are all exact ops. -/
private theorem vecmatPrologue_castFree (R : RoundingModel)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (vecmatPrologue dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) t
      = stepStmts (vecmatPrologue dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) t := by
  simp only [vecmatPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The K-loop body is cast-free: the two `.nat` address tiles, the two
unmasked `.real` region loads, the `tl.broadcast` expandDim and the
`trans ∘ sum ∘ (*)` accumulator update carry no `castFloat` and no store, so
the body steps identically under `stepStmtsR R`. -/
private theorem vecmatBody_castFree (R : RoundingModel) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) t
      = stepStmts (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) t := by
  simp only [vecmatLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The `a_tile` address op is cast-free. -/
private theorem evalR_atile (R : RoundingModel) (dim_k BLOCK_M BLOCK_K : Nat) (s : BlockState) :
    evalOpR R
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s
      = evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
              (Op.arange BLOCK_M)))
          (Op.constNat dim_k))
        (Op.expandDim ⟨0, by simp⟩
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
            (Op.arange BLOCK_K)))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- The `b_tile` address op is cast-free. -/
private theorem evalR_btile (R : RoundingModel)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) (s : BlockState) :
    evalOpR R
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s
      = evalOp
      (Op.add NumericDType.nat Broadcast.nil.consL.consR.consR
        (Op.add NumericDType.nat Broadcast.nil.consSame.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨2, by simp⟩
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "m_index") (Op.constNat BLOCK_M))
                    (Op.arange BLOCK_M))))
              (Op.constNat dim_n))
            (Op.constNat dim_k))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨2, by simp⟩
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "n_index") (Op.constNat BLOCK_N))
                  (Op.arange BLOCK_N))))
            (Op.constNat dim_k)))
        (Op.expandDim ⟨0, by simp⟩
          (Op.expandDim ⟨0, by simp⟩
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "k_index") (Op.constNat BLOCK_K))
              (Op.arange BLOCK_K))))) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- `stepStmtR R` on the dynamic K-loop resolves its three cast-free bound
ops (`0`, the `k_blocks` register, `1`) and becomes the `R` strided fold. -/
private theorem stepStmtR_kLoopV (R : RoundingModel) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K kb : Nat) (s : BlockState)
    (hkb : s.regs .nat [] "k_blocks" = some (Tile.scalar kb)) :
    stepStmtR R (Stmt.forRangeDyn "k_index" (Op.constNat 0) (Op.ref TileDType.nat [] "k_blocks")
        (Op.constNat 1) (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)) s
      = stepForRangeAuxR R "k_index" 0 kb 1
          (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) s := by
  simp only [stepStmtR, evalOpR_ref, hkb,
    show evalOpR R (Op.constNat (0 : Nat)) s = some (Tile.scalar 0) from by
      simp only [evalOpR.eq_def, evalOp_constNat],
    show evalOpR R (Op.constNat (1 : Nat)) s = some (Tile.scalar 1) from by
      simp only [evalOpR.eq_def, evalOp_constNat],
    Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The whole dynamic K-loop statement is cast-free (bound ops + body). -/
private theorem vecmatKLoop_castFree (R : RoundingModel) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K kb : Nat) (s : BlockState)
    (hkb : s.regs .nat [] "k_blocks" = some (Tile.scalar kb)) :
    stepStmtR R (Stmt.forRangeDyn "k_index" (Op.constNat 0) (Op.ref TileDType.nat [] "k_blocks")
        (Op.constNat 1) (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)) s
      = stepStmt (Stmt.forRangeDyn "k_index" (Op.constNat 0) (Op.ref TileDType.nat [] "k_blocks")
        (Op.constNat 1) (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)) s := by
  rw [stepStmtR_kLoopV R A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K kb s hkb,
    stepForRangeAuxR_castFree R _
      (vecmatBody_castFree R A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) "k_index",
    stepForRangeAux.forRangeDyn_unfold, evalOp_constNat, evalOp_ref, hkb, evalOp_constNat]
  rfl

/-! ### The tail under `execR R` -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write:
`writeMemTypedR` delegates `.real` to the exact `writeMemTyped`, and
`RoundingModel.storeValue_real` says `R.storeValue .real` is exactly that
store demotion. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

set_option maxHeartbeats 4000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` terminal store terminates and writes, at every output lane, the
exact-ℝ cell `MemCell.of .real (real.ofReal (gemvSpec …))` — the `.real`
store has no rounding event — with a per-cell frame outside the output
window. -/
private theorem vecmat_postLoopR (R : RoundingModel) (A B output : RegionName) (s0 : BlockState)
    (dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hInj : Function.Injective (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N))
    (st : BlockState)
    (hinv : vecmatInvariant A B output s0 dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        numKBlocks st) :
    ∃ sfin, stepStmtsR R (vecmatStoreStmt output BLOCK_M BLOCK_N :: []) st = some sfin
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          sfin.mem output (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N idx)
            = MemCell.of FloatDType.real.toTileDType
                (FloatDType.real.ofReal
                  (gemvSpec s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
                    idx.1 idx.2.1)))
      ∧ (∀ r o, (r ≠ output ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
            o ≠ vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N idx) →
          sfin.mem r o = st.mem r o) := by
  obtain ⟨hpids, hm, hn, hot, hvm, hundef, hmem⟩ := hinv
  set vmT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => some (gemvPartial s0 A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        idx.1 idx.2.1 numKBlocks)⟩ with hvmT
  set otT : Tile .nat [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩
    with hotT
  have hstore : stepStmtR R (vecmatStoreStmt output BLOCK_M BLOCK_N) st
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_N]).foldl
          (fun acc i =>
            acc.writeMemAsR R .real output (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N i)
              (vmT.data i)) st) := by
    simp only [vecmatStoreStmt, stepStmtR]
    rw [show evalOpR R (Op.ref .real [BLOCK_M, BLOCK_N] "vecmat") st = some vmT from by
      rw [evalOpR_ref, hvm]]
    rw [show evalOpR R (Op.ref .nat [BLOCK_M, BLOCK_N] "output_tile") st = some otT from by
      rw [evalOpR_ref, hot]]
    simp only [bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ st (fun acc i _ => ?_))
    simp only [if_true, otT, hotT, vecmatOutOffset, gRow, gCol, Region.cast_id,
      writeMemTypedR_real_eq]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx
    rw [BlockState.scatter_memcell_R_nd R .real st (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N)
        (fun i => vmT.data i) hInj idx]
    simp only [hvmT, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, gemvSpec, gemvPartial, Nat.mul_comm numKBlocks BLOCK_K]
  · intro r o hcond
    have hnot : ∀ k ∈ TileShape.allIndices [BLOCK_M, BLOCK_N],
        ¬(output = r ∧ vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N k = o) := by
      intro k _ hk
      obtain ⟨hor, hko⟩ := hk
      rcases hcond with hne | hall
      · exact hne hor.symm
      · exact hall k hko.symm
    rw [BlockState.foldl_writeMemAsR_preserve_cell R .real
      (vecmatOutOffset s0 dim_n BLOCK_M BLOCK_N) (fun i => vmT.data i) r o
      (TileShape.allIndices [BLOCK_M, BLOCK_N]) hnot st]

/-! ### Safety-walk invariant (weak shape half of `vecmatInvariant`) -/

/-- Safety-walk loop invariant: the *shape* half of `vecmatInvariant`
(`m_index` / `n_index` / `output_tile` seeded, *some* accumulator tile) with
no `undef` / `mem` / value pins, and no dependence on the loop counter — the
body recomputes both address tiles from `m_index` / `n_index` / `k_index`
every iteration. Needed because the `⊨[R]` skin's `hts` obligation quantifies
over arbitrary launch states, so the safety walk cannot assume the
clean-`undef` precondition that `vecmat_preLoop`'s full invariant needs. -/
private def vecmatSafeInv (s0 : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (s : BlockState) : Prop :=
  (s.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0))) ∧
  (s.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1))) ∧
  (s.regs .nat [BLOCK_M, BLOCK_N] "output_tile" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩) ∧
  (∃ vT : Tile .real [BLOCK_M, BLOCK_N], s.regs .real [BLOCK_M, BLOCK_N] "vecmat" = some vT)

set_option maxHeartbeats 2000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `vecmatSafeInv` (plus the `k_blocks` readback the dynamic loop's
`stop` op resolves through). -/
private theorem vecmat_preLoopW (s : BlockState) (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ s', stepStmts (vecmatPrologue dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) s = some s'
      ∧ vecmatSafeInv s dim_n BLOCK_M BLOCK_N s'
      ∧ s'.regs .nat [] "k_blocks" = some (Tile.scalar (dim_k / BLOCK_K)) := by
  simp only [vecmatPrologue]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.programId 0)
        (v := Tile.scalar (s.pids 0)) (by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.programId 1)
        (v := Tile.scalar (s.pids 1)) (by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_otile_eval _ dim_n BLOCK_M BLOCK_N (s.pids 0) (s.pids 1)
          (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (e := Op.full [BLOCK_M, BLOCK_N] (Op.const 0))
        (v := (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_N]))
        (by simp [evalOp_full, evalOp_const, Option.bind]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (e := Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat dim_k) (Op.constNat BLOCK_K))
        (v := Tile.scalar (dim_k / BLOCK_K))
        (by simp [evalOp, Tile.bop, IntegralDType.floorDiv, NumericDType.div]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
  · refine ⟨⟨fun _ => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true]
  · simp only [BlockState.setReg_same]

set_option maxHeartbeats 4000000 in
/-- Weak step lemma: one body iteration from `vecmatSafeInv` steps
successfully (exact stepper; the body is cast-free) and re-establishes the
invariant — the shape half of `vecmat_step`, valid from arbitrary launch
states. -/
private theorem vecmat_stepW (A B : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) (c : Nat) (s : BlockState)
    (hinv : vecmatSafeInv s0 dim_n BLOCK_M BLOCK_N s) :
    ∃ s', stepStmts (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K)
        (s.setReg "k_index" .nat [] (Tile.scalar c)) = some s'
      ∧ vecmatSafeInv s0 dim_n BLOCK_M BLOCK_N s' := by
  obtain ⟨hm, hn, hot, ⟨vT, hvm⟩⟩ := hinv
  set sk := s.setReg "k_index" .nat [] (Tile.scalar c) with hsk
  have hmk : sk.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0)) := by simp [hsk, hm]
  have hnk : sk.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1)) := by simp [hsk, hn]
  have hkk : sk.regs .nat [] "k_index" = some (Tile.scalar c) := by simp [hsk]
  set atT : Tile .nat [BLOCK_M, BLOCK_K] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.1.val) * dim_k + (c * BLOCK_K + idx.2.1.val)⟩ with hatT
  set btT : Tile .nat [BLOCK_N, BLOCK_M, BLOCK_K] :=
    ⟨fun idx => (s0.pids 0 * BLOCK_M + idx.2.1.val) * dim_n * dim_k
        + (s0.pids 1 * BLOCK_N + idx.1.val) * dim_k + (c * BLOCK_K + idx.2.2.1.val)⟩ with hbtT
  set aT : Tile .real [BLOCK_M, BLOCK_K] :=
    ⟨fun i => some (sk.readMem A (atT.data i))⟩ with haT
  set bT : Tile .real [BLOCK_N, BLOCK_M, BLOCK_K] :=
    ⟨fun i => some (sk.readMem B (btT.data i))⟩ with hbT
  simp only [vecmatLoopBody]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_atile_eval sk dim_k BLOCK_M BLOCK_K (s0.pids 0) c hmk hkk))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_load_eval _ A "a_tile" atT (by simp [haT, hatT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_btile_eval _ dim_n dim_k BLOCK_M BLOCK_N BLOCK_K (s0.pids 0) (s0.pids 1) c
          (by simp [hmk]) (by simp [hnk]) (by simp [hkk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (vecmat_load_eval _ B "b_tile" btT (by simp [hbtT])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (shape := [1, BLOCK_M, BLOCK_K]) (name := "expanded_a")
        (e := Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [BLOCK_M, BLOCK_K] "a"))
        (v := Tile.expandDim ⟨0, by simp⟩ aT)
        (by
          refine evalOp_expandDim_ref_of_regs _ _ _ _ _ aT ?_
          simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
            String.reduceEq, not_false_eq_true, haT, BlockState.setReg_readMem]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (shape := [BLOCK_M, BLOCK_N]) (name := "vecmat")
        (e := Op.add NumericDType.real Broadcast.nil.consSame.consSame
          (Op.ref TileDType.real [BLOCK_M, BLOCK_N] "vecmat")
          (Op.transpose (batch := []) (M := BLOCK_N) (N := BLOCK_M)
            (Op.reduceSum ⟨2, by simp⟩ Bool.false
              (Op.mul NumericDType.real Broadcast.nil.consSame.consSame.consL
                (Op.ref TileDType.real [1, BLOCK_M, BLOCK_K] "expanded_a")
                (Op.ref TileDType.real [BLOCK_N, BLOCK_M, BLOCK_K] "b")))))
        (v := Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame vT
          (Tile.transpose []
            (Tile.reduceSum ⟨2, by simp⟩ Bool.false
              (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.consSame.consL
                (Tile.expandDim ⟨0, by simp⟩ aT) bT))))
        (by
          simp only [evalOp.eq_def, Option.bind, Option.map,
            BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
            not_false_eq_true, hsk, hvm]
          rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hm]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hn]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, hsk, hot]
  · refine ⟨Tile.bop NumericDType.real.add Broadcast.nil.consSame.consSame vT
        (Tile.transpose []
          (Tile.reduceSum ⟨2, by simp⟩ Bool.false
            (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.consSame.consL
              (Tile.expandDim ⟨0, by simp⟩ aT) bT))), ?_⟩
    simp only [BlockState.setReg_same]

/-! ### The lane bridges -/

/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `A`-tile. -/
def aLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_M * BLOCK_K) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `B`-stream lane feeding output lane `l` at inner key `e`: the
`[BLOCK_N, BLOCK_M, BLOCK_K]` cell `(j, i, e)` flattened row-major through
two nested `Lane2D` bridges — the `(j, i)` pair first (into
`Fin (BLOCK_N * BLOCK_M)`), then the key. -/
def bLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_M * BLOCK_K) :=
  Lane2D.encode
    (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)

/-! ### The safety walk -/

set_option maxHeartbeats 4000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two **unmasked**
region loads' offset tiles are the freshly computed `a_tile` / `b_tile`
shapes, in bounds by the skin's `read1` / `read2` windows (instantiated at
step `c`); the four remaining assigns are register-only and unconditionally
safe. -/
private theorem vecmat_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B : RegionName) (s0 : BlockState)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K T : Nat) (c : Nat) (hc : c < T)
    (sk : BlockState)
    (hmk : sk.regs .nat [] "m_index" = some (Tile.scalar (s0.pids 0)))
    (hnk : sk.regs .nat [] "n_index" = some (Tile.scalar (s0.pids 1)))
    (hkk : sk.regs .nat [] "k_index" = some (Tile.scalar c))
    (hbA : ∀ (t : Fin T) (l : Fin (BLOCK_M * BLOCK_K)),
      (s0.pids 0 * BLOCK_M + l.val / BLOCK_K) * dim_k + (t.val * BLOCK_K + l.val % BLOCK_K)
        < bounds A)
    (hbB : ∀ (t : Fin T) (l : Fin (BLOCK_N * BLOCK_M * BLOCK_K)),
      (s0.pids 0 * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n * dim_k
        + (s0.pids 1 * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * dim_k
        + (t.val * BLOCK_K + l.val % BLOCK_K) < bounds B) :
    Stmt.TraceSafeListR R bounds (vecmatLoopBody A B dim_n dim_k BLOCK_M BLOCK_N BLOCK_K) sk := by
  simp only [vecmatLoopBody]
  -- 1. a_tile
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s1 h1
  obtain ⟨v1, hv1, rfl⟩ := stepStmtR_assign_inv h1
  rw [evalR_atile, vecmat_atile_eval sk dim_k BLOCK_M BLOCK_K (s0.pids 0) c hmk hkk] at hv1
  obtain rfl := Option.some.inj hv1
  -- 2. a = load A[a_tile]
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro offsets hoffs i _
    rw [evalOpR_ref] at hoffs
    simp only [BlockState.setReg_same] at hoffs
    obtain rfl := Option.some.inj hoffs
    show (s0.pids 0 * BLOCK_M + i.1.val) * dim_k + (c * BLOCK_K + i.2.1.val)
        < bounds (Region.cast A)
    have h' := hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simpa only [Region.cast_id] using h'
  intro s2 h2
  obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
  -- 3. b_tile
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) ?_
  intro s3 h3
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv h3
  rw [evalR_btile, vecmat_btile_eval _ dim_n dim_k BLOCK_M BLOCK_N BLOCK_K
    (s0.pids 0) (s0.pids 1) c (by simp [hmk]) (by simp [hnk]) (by simp [hkk])] at hv3
  obtain rfl := Option.some.inj hv3
  -- 4. b = load B[b_tile]
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro offsets hoffs i _
    rw [evalOpR_ref] at hoffs
    simp only [BlockState.setReg_same] at hoffs
    obtain rfl := Option.some.inj hoffs
    show (s0.pids 0 * BLOCK_M + i.2.1.val) * dim_n * dim_k
        + (s0.pids 1 * BLOCK_N + i.1.val) * dim_k + (c * BLOCK_K + i.2.2.1.val)
        < bounds (Region.cast B)
    have h' := hbB ⟨c, hc⟩
      (Lane2D.encode (Lane2D.encode (i.1, i.2.1, PUnit.unit), i.2.2.1, PUnit.unit))
    rw [Lane2D.encode_div, Lane2D.encode_mod, Lane2D.encode_div, Lane2D.encode_mod] at h'
    simpa only [Region.cast_id] using h'
  intro s4 h4
  obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv h4
  -- 5, 6: register-only
  refine Stmt.TraceSafeListR.of_forall _ _ ?_
  intro stx hstx s'
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hstx
  rcases hstx with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 2000000 in
/-- `TraceSafeListR` for the terminal store: the single unmasked `.real`
store's offsets are the invariant's `output_tile` shape — the skin's `write`
window. -/
private theorem vecmat_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (output : RegionName) (s0 : BlockState) (dim_n BLOCK_M BLOCK_N : Nat) (st : BlockState)
    (hot : st.regs .nat [BLOCK_M, BLOCK_N] "output_tile" = some
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (s0.pids 0 * BLOCK_M + idx.1.val) * dim_n + (s0.pids 1 * BLOCK_N + idx.2.1.val)⟩)
    (hbO : ∀ l : Fin (BLOCK_M * BLOCK_N),
      (s0.pids 0 * BLOCK_M + l.val / BLOCK_N) * dim_n + (s0.pids 1 * BLOCK_N + l.val % BLOCK_N)
        < bounds output) :
    Stmt.TraceSafeListR R bounds (vecmatStoreStmt output BLOCK_M BLOCK_N :: []) st := by
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
  simp only [vecmatStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro offsets hoffs i _
  rw [evalOpR_ref, hot] at hoffs
  obtain rfl := Option.some.inj hoffs
  show (s0.pids 0 * BLOCK_M + i.1.val) * dim_n + (s0.pids 1 * BLOCK_N + i.2.1.val)
      < bounds (Region.cast output)
  have h' := hbO (Lane2D.encode (i.1, i.2.1, PUnit.unit))
  rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
  simpa only [Region.cast_id] using h'

set_option maxHeartbeats 4000000 in
/-- **The `TraceSafeR` walk for the whole kernel**, driven by
`Stmt.forRangeTraceSafeR_inv` over the weak `vecmatSafeInv` (reached through
the `forRangeDyn` arm of `TraceSafeR`, whose bound ops resolve to `0`, the
`k_blocks` register and `1`). The three bound groups are the skin's `read1` /
`read2` / `write` windows. -/
private theorem vecmat_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (A B output : RegionName)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (s : BlockState)
    (hbA : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_M * BLOCK_K)),
      (s.pids 0 * BLOCK_M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
        + (t.val * BLOCK_K + l.val % BLOCK_K) < bounds A)
    (hbB : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_N * BLOCK_M * BLOCK_K)),
      (s.pids 0 * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n * (BLOCK_K * numKBlocks)
        + (s.pids 1 * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * (BLOCK_K * numKBlocks)
        + (t.val * BLOCK_K + l.val % BLOCK_K) < bounds B)
    (hbO : ∀ l : Fin (BLOCK_M * BLOCK_N),
      (s.pids 0 * BLOCK_M + l.val / BLOCK_N) * dim_n + (s.pids 1 * BLOCK_N + l.val % BLOCK_N)
        < bounds output) :
    ((batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N
        BLOCK_K).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [vecmat_body_split' A B output _dim_m dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K]
  have hnum : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  have hstep : ∀ c s', c < numKBlocks → vecmatSafeInv s dim_n BLOCK_M BLOCK_N s' →
      Stmt.TraceSafeListR R bounds
        (vecmatLoopBody A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K)
        (s'.setReg "k_index" .nat [] (Tile.scalar c)) ∧
      ∃ s'', stepStmtsR R
          (vecmatLoopBody A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K)
          (s'.setReg "k_index" .nat [] (Tile.scalar c)) = some s'' ∧
        vecmatSafeInv s dim_n BLOCK_M BLOCK_N s'' := by
    intro c s' hcx hP
    obtain ⟨hmx, hnx, hotx, hvx⟩ := hP
    refine ⟨vecmat_bodySafeR R bounds A B s dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        numKBlocks c hcx _ (by simp [hmx]) (by simp [hnx]) (by simp) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ := vecmat_stepW A B s dim_n (BLOCK_K * numKBlocks)
      BLOCK_M BLOCK_N BLOCK_K c s' ⟨hmx, hnx, hotx, hvx⟩
    exact ⟨s'', by rw [vecmatBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro st hst s'
    simp only [vecmatPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre, hP0, hkb⟩ :=
      vecmat_preLoopW s dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
    rw [vecmatPrologue_castFree R dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K s,
      hpre] at hs1
    obtain rfl := Option.some.inj hs1
    rw [hnum] at hkb
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
        by simp [Op.SafeAtR.eq_def], ?_⟩
      rw [evalOpR_ref, hkb,
        show evalOpR R (Op.constNat (0 : Nat)) s1x = some (Tile.scalar 0) from by
          simp only [evalOpR.eq_def, evalOp_constNat],
        show evalOpR R (Op.constNat (1 : Nat)) s1x = some (Tile.scalar 1) from by
          simp only [evalOpR.eq_def, evalOp_constNat]]
      show Stmt.forRangeTraceSafeR R bounds "k_index" 0 numKBlocks 1
        (vecmatLoopBody A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K) s1x
      exact Stmt.forRangeTraceSafeR_inv R bounds "k_index" numKBlocks 1
        (vecmatLoopBody A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K)
        (fun _ st => vecmatSafeInv s dim_n BLOCK_M BLOCK_N st) hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRangeDyn_inv (idx := "k_index") (start := 0) (stop := numKBlocks) (step := 1)
          (startOp := Op.constNat 0) (stopOp := Op.ref TileDType.nat [] "k_blocks")
          (stepOp := Op.constNat 1)
          (P := fun _ st => vecmatSafeInv s dim_n BLOCK_M BLOCK_N st)
          (by simp [evalOp_constNat]) (by rw [evalOp_ref, hkb]) (by simp [evalOp_constNat])
          (by omega) hP0
          (fun c stx _ hinv => vecmat_stepW A B s dim_n (BLOCK_K * numKBlocks)
            BLOCK_M BLOCK_N BLOCK_K c stx hinv)
      rw [vecmatKLoop_castFree R A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        numKBlocks s1x hkb, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      obtain ⟨-, -, hotL, -⟩ := hPL
      exact vecmat_tailSafeR R bounds output s dim_n BLOCK_M BLOCK_N sLoop hotL hbO

/-- The full `batched_vecmat_mult` surface (unmasked region loads, the
broadcast/reduce/transpose accumulator update, the dynamic K-loop and the
unmasked region store) sits inside the flat-memory bridge's covered fragment
(`FlattenOk`). -/
theorem batched_vecmat_flattenOk (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ((batched_vecmat_surface A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N
        BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [vecmat_body_split' A B output _dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K]
  simp [vecmatPrologue, vecmatLoopBody, vecmatStoreStmt, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature and spec bridge -/

/-- **Streaming IO signature** of `batched_vecmat_mult` on the two-stream
fold skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BLOCK_M, BLOCK_K]` `A`-tile and the `[BLOCK_N, BLOCK_M, BLOCK_K]` `B`-tile;
after the loop one `[BLOCK_M, BLOCK_N]` output tile is stored at the
**`.real`** grid (`outDType` default — the kernel's
`tl.store(output + output_tile, vecmat)` is untyped, so the terminal store
has no quantization event). The kernel uses a genuine **2-D** pid grid, so
`pid₀ = m_index` and `pid₁ = n_index` both feed the windows:

* `read1` lane `l = (i, e)` (row-major over `[BLOCK_M, BLOCK_K]`), step `t`:
  `(pid₀·BLOCK_M + i)·dim_k + (t·BLOCK_K + e)` — the kernel's `a_tile`.
* `read2` lane `l = (j, i, e)` (row-major over
  `[BLOCK_N, BLOCK_M, BLOCK_K]`, i.e. `j = l/BLOCK_K/BLOCK_M`,
  `i = l/BLOCK_K % BLOCK_M`, `e = l % BLOCK_K`), step `t`:
  `(pid₀·BLOCK_M + i)·dim_n·dim_k + (pid₁·BLOCK_N + j)·dim_k + (t·BLOCK_K + e)`
  — the kernel's `b_tile`.
* `write` lane `l = (i, j)`:
  `(pid₀·BLOCK_M + i)·dim_n + (pid₁·BLOCK_N + j)` — the kernel's
  `output_tile` (= `vecmatOutOffset` in pid form).

The kernel has no masks (the Python wrapper asserts exact divisibility): all
windows are `True`. -/
def batchedVecmatIO (A B output : RegionName)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
    BLOCK_M BLOCK_N BLOCK_K
  inp1 := A
  inp2 := B
  out := output
  T := numKBlocks
  B1 := BLOCK_M * BLOCK_K
  B2 := BLOCK_N * BLOCK_M * BLOCK_K
  C := BLOCK_M * BLOCK_N
  read1 := fun p₀ _ t l =>
    (p₀ * BLOCK_M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  read2 := fun p₀ p₁ t l =>
    (p₀ * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n * (BLOCK_K * numKBlocks)
      + (p₁ * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  write := fun p₀ p₁ l =>
    (p₀ * BLOCK_M + l.val / BLOCK_N) * dim_n + (p₁ * BLOCK_N + l.val % BLOCK_N)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True

/-- Under the two stream pins, `gemvSpec` at the decoded output lane **is**
the skin-level double fold `∑ t, ∑ e, xs · ys` (`gemmSum_blocks` +
address-identity of the windows with the kernel's `a_tile` / `b_tile`
shapes). -/
private theorem gemvSpec_eq_streamSum (A B : RegionName) (s₀ : BlockState)
    (dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BLOCK_M * BLOCK_K) → ℝ)
    (ys : Fin numKBlocks → Fin (BLOCK_N * BLOCK_M * BLOCK_K) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_M * BLOCK_K)),
      s₀.readMem A ((s₀.pids 0 * BLOCK_M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
          + (t.val * BLOCK_K + l.val % BLOCK_K))
        = xs t l)
    (hy : ∀ (t : Fin numKBlocks) (l : Fin (BLOCK_N * BLOCK_M * BLOCK_K)),
      s₀.readMem B ((s₀.pids 0 * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n
            * (BLOCK_K * numKBlocks)
          + (s₀.pids 1 * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * (BLOCK_K * numKBlocks)
          + (t.val * BLOCK_K + l.val % BLOCK_K))
        = ys t l)
    (l : Fin (BLOCK_M * BLOCK_N)) :
    gemvSpec s₀ A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
  rw [show gemvSpec s₀ A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
        (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = gemmSum
          (fun kk => aElem s₀ A (BLOCK_K * numKBlocks) BLOCK_M (Lane2D.decode l).1 kk)
          (fun kk => bElem s₀ B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N
            (Lane2D.decode l).1 (Lane2D.decode l).2.1 kk)
          (numKBlocks * BLOCK_K) from by
    unfold gemvSpec gemmSum
    rw [Nat.mul_comm BLOCK_K numKBlocks]]
  rw [gemmSum_blocks]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ A (BLOCK_K * numKBlocks) BLOCK_M (Lane2D.decode l).1
      (t.val * BLOCK_K + e.val) = xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hx t (aLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N
      (Lane2D.decode l).1 (Lane2D.decode l).2.1 (t.val * BLOCK_K + e.val)
      = ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hy t (bLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `batched_vecmat_mult` surface implements, on
its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ batched GEMV fold**
over the streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](j,i,e)` — the spec `f` is exact real
arithmetic. The kernel has **no rounding events** (both region loads, the
`expanded_a * b` product, the axis-2 `tl.sum`, the `tl.trans` and the store
are all at `.real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` terminal store is exact under `execR R`
(`RoundingModel.storeValue_real`).

Layer map: the 5-statement prologue and the whole dynamic K-loop are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `vecmat_preLoop` / `vecmat_step` / `forRangeDyn_inv` stack
above is reused unchanged; only the single store statement is re-proved on
the `R` side (`vecmat_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the loop trip count is the runtime register
  `k_blocks = dim_k // block_k`, which is `numKBlocks` at
  `dim_k = BLOCK_K · numKBlocks` only for a nonzero block size; at
  `BLOCK_K = 0` the division is `0` and the loop never runs, so the fold
  would be empty while the spec sums nothing either — but the block index
  arithmetic `l / BLOCK_K` is meaningless. It holds for every real launch
  (`tl.arange(0, 0)` is not a tile, and the Python wrapper asserts
  `K % block_k == 0`).
* `hBN : BLOCK_N ≤ dim_n` — output-offset injectivity (`rowMajor2D_inj`):
  the output tile's column width must fit the row stride `dim_n`, otherwise
  distinct lanes `(i, j)` collide at `i·dim_n + j` and the per-lane readback
  would be last-writer-wins, making the statement false. It holds for every
  valid tiling (the Python wrapper asserts `N % block_n == 0` with the grid
  covering `N`).

Relation to the exact surface: the exact headline
`batched_vecmat_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same GEMV content on the
streaming skin, for every `R` at once. Both faces are kept per the
rounding-as-default doctrine. -/
specification batched_vecmat_io_correctness (R : RoundingModel)
    (A B output : RegionName)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K) (hBN : BLOCK_N ≤ dim_n) :
    batchedVecmatIO A B output _dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
  refine StreamMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact batched_vecmat_flattenOk A B output _dim_m dim_n (BLOCK_K * numKBlocks)
      BLOCK_M BLOCK_N BLOCK_K
  · -- safety walk
    intro bounds s xs ys _hx _hy hbr1 hbr2 hbw
    simp only [batchedVecmatIO] at hbr1 hbr2 hbw ⊢
    exact vecmat_traceSafeR R bounds A B output _dim_m dim_n BLOCK_M BLOCK_N BLOCK_K
      numKBlocks hBK s (fun t l => hbr1 t l trivial) (fun t l => hbr2 t l trivial)
      (fun l => hbw l trivial)
  · -- the rounded Hoare triple
    intro s₀ xs ys hundef hx hy
    simp only [batchedVecmatIO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hInj : Function.Injective (vecmatOutOffset s₀ dim_n BLOCK_M BLOCK_N) := by
      have heq : vecmatOutOffset s₀ dim_n BLOCK_M BLOCK_N
          = fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              (s₀.pids 0 * BLOCK_M * dim_n + s₀.pids 1 * BLOCK_N)
                + idx.1.val * dim_n + idx.2.1.val := by
        funext idx; simp only [vecmatOutOffset, gRow, gCol]; ring
      rw [heq]; exact rowMajor2D_inj _ dim_n hBN
    have hnum : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
      rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
    obtain ⟨s1, hpre, hP0, hkb⟩ :=
      vecmat_preLoop A B output s₀ _dim_m dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        hundef'
    rw [hnum] at hkb
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRangeDyn_inv (idx := "k_index") (start := 0) (stop := numKBlocks) (step := 1)
        (startOp := Op.constNat 0) (stopOp := Op.ref TileDType.nat [] "k_blocks")
        (stepOp := Op.constNat 1)
        (P := fun c st =>
          vecmatInvariant A B output s₀ dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K c st
            ∧ c ≤ numKBlocks)
        (by simp [evalOp_constNat]) (by rw [evalOp_ref, hkb]) (by simp [evalOp_constNat])
        (by omega) ⟨hP0, by omega⟩
        (fun c stx hlt hinv => by
          obtain ⟨st', hstep, hinv'⟩ :=
            vecmat_step A B output s₀ dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K c stx
              hinv.1
          exact ⟨st', hstep, hinv', by omega⟩)
    have hfinalEq : final = numKBlocks := le_antisymm hPLoop.2 hfinal
    rw [hfinalEq] at hPLoop
    have hmem0 : sLoop.mem = s₀.mem := hPLoop.1.2.2.2.2.2.2
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      vecmat_postLoopR R A B output s₀ dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks hInj sLoop
        hPLoop.1
    have hLoopR : stepStmtR R (Stmt.forRangeDyn "k_index" (Op.constNat 0)
        (Op.ref TileDType.nat [] "k_blocks") (Op.constNat 1)
        (vecmatLoopBody A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K)) s1
        = some sLoop := by
      rw [vecmatKLoop_castFree R A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K
        numKBlocks s1 hkb]
      exact hLoopStmt
    have hpre' : stepStmts (vecmatPrologue dim_n (BLOCK_K * numKBlocks)
        BLOCK_M BLOCK_N BLOCK_K) s₀ = some s1 := by
      rw [← vecmat_take5_eq A B output _dim_m dim_n (BLOCK_K * numKBlocks)
        BLOCK_M BLOCK_N BLOCK_K]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
          BLOCK_M BLOCK_N BLOCK_K).toAlgKernel s₀ = some sfin
      unfold execR
      rw [vecmat_body_split' A B output _dim_m dim_n (BLOCK_K * numKBlocks)
          BLOCK_M BLOCK_N BLOCK_K,
        stepStmtsR_append,
        vecmatPrologue_castFree R dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K s₀,
        hpre', Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro l _
      have hcell := hval (Lane2D.decode l)
      have haddr : (s₀.pids 0 * BLOCK_M + l.val / BLOCK_N) * dim_n
            + (s₀.pids 1 * BLOCK_N + l.val % BLOCK_N)
          = vecmatOutOffset s₀ dim_n BLOCK_M BLOCK_N (Lane2D.decode l) := rfl
      rw [haddr, readMemAs_real_of_cell hcell, R.round_real_apply,
        gemvSpec_eq_streamSum A B s₀ dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks xs ys
          (fun t j => hx t j trivial) (fun t j => hy t j trivial) l]
    · intro r o hcond
      have hcond' : r ≠ output ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          o ≠ vecmatOutOffset s₀ dim_n BLOCK_M BLOCK_N idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx => ?_
          have h := hno (Lane2D.encode idx) trivial
          simpa [Lane2D.encode_div, Lane2D.encode_mod, vecmatOutOffset, gRow, gCol] using h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.BatchedVecmatMult

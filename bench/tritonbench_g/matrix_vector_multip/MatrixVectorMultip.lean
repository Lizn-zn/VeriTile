import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `matrix_vector_multip` — strict per-kernel correctness

`mv_kernel` computes a matrix-vector product `C = A · B`: program `pid` owns rows
`[pid·BLOCK_N, …)` of `A`, loops over the `M` columns in `BLOCK_M`-wide chunks
accumulating `a * b`, reduces over the `M` axis (`tl.sum(acc, axis=1)`), and
stores the resulting length-`BLOCK_N` vector to `C`, masked by `offset_n < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`mv_kernel[grid](...)`, the grid size `cdiv(N, BLOCK_N)`,
the strides passed from the host, and how the runtime composes per-program
writes into `C`) is the *trusted boundary*, not a proof obligation here. The
program id enters only via `BlockState`, so the per-program statement covers
every program of the grid.

## Proof architecture

```
mv_kernel_python_test_shape_complete_summary    ← TOP THEOREM (both checked shapes)
  ├─ mv_kernel_python_case2_output_summary       (N=4, M=3)
  │    ├─ (toAlgorithm? = Except.ok _) via mv_kernel_python_case2_surface_toAlgorithm_supported
  │    └─ mv_kernel_python_case2_one_block_compute_correct
  └─ mv_kernel_python_case3_output_summary        (N=32, M=16)
       ├─ (toAlgorithm? = Except.ok _) via mv_kernel_python_case3_surface_toAlgorithm_supported
       └─ mv_kernel_python_case3_one_block_compute_correct
            └─ mv_kernel_one_block_compute_correct → mv_kernel_one_block_correct
                 └─ mvSpec / mvProdTile  (row-wise `Tile.reduceSum` of `a * b`)
```

`mv_kernel_surface_toAlgorithm_supported` separately establishes that the full
looping surface (not just the one-block slice) lowers to the algorithm layer.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled — its config grid all uses `BLOCK_M ≥ 32`, so for the bundled test
shapes (`M ∈ {3, 16}`) the `M` loop executes exactly one block; the
`mv_kernel_one_block` slice captures that single-block path, with the full
looping surface verified only to lower (`toAlgorithm?`). The `.to(tl.float32)`
casts reduce to the identity post-erasure. Masked loads use `other=0.0`. The
output scatter requires the per-row offset map to be injective
(`mv_kernel_python_stride1_output_offset_injective` for the contiguous Python
strides).
-/

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `mv_kernel_python_case2_output_summary`, `mv_kernel_python_case3_output_summary` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- Faithful transcription of `matrix_vector_multip.py`'s `mv_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_M: tl.constexpr` -> Lean `Nat` parameters. -/
def mv_kernel
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))[:, None]
  offset_m = tl.arange(0, $(BLOCK_M))[None, :]
  n_mask = offset_n < $(N)
  A_ptrs = A + offset_n * $(stride_an) + offset_m * $(stride_am)
  B_ptrs = B + offset_m * $(stride_bm)
  acc = tl.zeros([$(BLOCK_N), $(BLOCK_M)], dtype=tl.float32)
  for m in range(0, $(M), $(BLOCK_M)) {
    m_mask = (m + offset_m) < $(M)
    a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
    acc += a * b
    A_ptrs += $(BLOCK_M) * $(stride_am)
    B_ptrs += $(BLOCK_M) * $(stride_bm)
  }
  acc = tl.sum(acc, axis=1)
  C_ptrs = C + offset_n * $(stride_cn)
  tl.store(C_ptrs, acc[:, None], mask=n_mask)
}

/-- The full matrix-vector multiplication surface lowers to the algorithm
layer, including the `M` loop, masked loads, vector multiply, reduction, and
masked output store. -/
theorem mv_kernel_surface_toAlgorithm_supported
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ∃ alg, (mv_kernel A B C N M stride_an stride_am stride_bm stride_cn
      BLOCK_N BLOCK_M).toAlgorithm? = Except.ok alg := by
  simp [mv_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-`BLOCK_M` slice of `matrix_vector_multip.py`'s
`mv_kernel`.

The full surface loops over `M` in `BLOCK_M` chunks. This slice captures the
single-block path used by the bundled tests (`M = 3` and `M = 16`, while the
autotune choices have `BLOCK_M >= 32`): load one A tile and one B tile, reduce
over `BLOCK_M`, and write one C block. -/
def mv_kernel_one_block
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset_n = pid * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_m = tl.arange(0, $(BLOCK_M))
  a = tl.load(A + offset_n[:, None] * $(stride_an) + offset_m[None, :] * $(stride_am),
    mask=(offset_n[:, None] < $(N)) and (offset_m[None, :] < $(M)), other=0.0).to(tl.float32)
  b = tl.load(B + offset_m * $(stride_bm), mask=offset_m < $(M), other=0.0).to(tl.float32)
  acc = tl.sum(a * b[None, :], axis=1)
  tl.store(C + offset_n * $(stride_cn), acc, mask=offset_n < $(N))
}

def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * BLOCK_N + i.val

def cOffset (s : BlockState) (stride_cn BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  nIndex s BLOCK_N i * stride_cn

def aOffset
    (s : BlockState) (stride_an stride_am BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_M) : Nat :=
  nIndex s BLOCK_N i * stride_an + j.val * stride_am

def bOffset (stride_bm : Nat) (j : Fin BLOCK_M) : Nat :=
  j.val * stride_bm

noncomputable def mvProdTile
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat) :
    Tile .real [BLOCK_N, BLOCK_M] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let mj := (TileShape.dropInsertedIndex [BLOCK_M] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if s.pids 0 * BLOCK_N + ni.val < N ∧ mj.val < M then
          some (s.readMem A ((s.pids 0 * BLOCK_N + ni.val) * stride_an + mj.val * stride_am))
        else some (0.0 : ℝ))
        (if mj.val < M then
          some (s.readMem B (mj.val * stride_bm))
        else some (0.0 : ℝ)) }

noncomputable def mvSpec
    (s : BlockState) (A B : RegionName)
    (N M stride_an stride_am stride_bm BLOCK_N BLOCK_M : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_M]) ⟨1, by simp⟩ Bool.false
      (mvProdTile s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M)).data
        (i, PUnit.unit))

/-- Algorithm-layer correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_one_block_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i))
    (hExec : exec (mv_kernel_one_block A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M) s = some s') :
    ∀ i : Fin BLOCK_N,
      s'.readMem C (cOffset s stride_cn BLOCK_N i) =
        if nIndex s BLOCK_N i < N then
          mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i
        else s.readMem C (cOffset s stride_cn BLOCK_N i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        (s.pids 0 * BLOCK_N + idx.1.val) * stride_cn) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [cOffset, nIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBN : 0 < BLOCK_N
  · simp [exec, mv_kernel_one_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBN] at hExec
    rw [← hExec]
    simp only [cOffset, nIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_N + i.val < N
    · simp [hi, mvSpec, mvProdTile, aOffset, bOffset, cOffset, nIndex,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex]
      congr 1
      apply Finset.sum_congr rfl
      intro x _
      change Option.map₂ (fun x1 x2 => x1 * x2)
          (if x.val < M then
            some (s.readMem A
              ((s.pids 0 * BLOCK_N + i.val) * stride_an + x.val * stride_am))
          else some 0.0)
          (if x.val < M then
            some (s.readMem B (x.val * stride_bm))
          else some 0.0) =
        Option.map₂ (fun x1 x2 => x1 * x2)
          (if s.pids 0 * BLOCK_N + i.val < N ∧ x.val < M then
            some (s.readMem A
              ((s.pids 0 * BLOCK_N + i.val) * stride_an + x.val * stride_am))
          else some 0.0)
          (if x.val < M then
            some (s.readMem B (x.val * stride_bm))
          else some 0.0)
      by_cases hxM : x.val < M
      · simp [hxM, hi]
      · simp [hxM]
    · simp [hi]
  · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-block matrix-vector slice. -/
theorem mv_kernel_one_block_compute_correct
    (A B C : RegionName)
    (N M stride_an stride_am stride_bm stride_cn BLOCK_N BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => cOffset s stride_cn BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := mv_kernel_one_block A B C N M stride_an stride_am stride_bm
        stride_cn BLOCK_N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (C, cOffset s stride_cn BLOCK_N i)))
      (expected := fun i =>
        mvSpec s A B N M stride_an stride_am stride_bm BLOCK_N BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mv_kernel_one_block, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := mv_kernel_one_block_correct A B C N M stride_an stride_am stride_bm
    stride_cn BLOCK_N BLOCK_M s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Python test-case wrappers

The bundled Python test constructs contiguous tensors for shapes `(N, M) =
(4, 3)` and `(32, 16)`. Triton's autotune choices all have `BLOCK_M >= 32`, so
each checked path executes a single `M` block; the wrappers below leave
`BLOCK_N`/`BLOCK_M` explicit to cover the autotuned choices while pinning the
Python tensor strides. -/

theorem mv_kernel_python_stride1_output_offset_injective
    (s : BlockState) (BLOCK_N : Nat) :
    Function.Injective (fun i : Fin BLOCK_N => cOffset s 1 BLOCK_N i) := by
  intro a b h
  apply Fin.ext
  simpa [cOffset, nIndex] using h

theorem mv_kernel_python_case2_surface_toAlgorithm_supported
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) :
    ∃ alg, (mv_kernel A B C 4 3 3 1 1 1 BLOCK_N BLOCK_M).toAlgorithm? =
      Except.ok alg := by
  exact mv_kernel_surface_toAlgorithm_supported A B C 4 3 3 1 1 1
    BLOCK_N BLOCK_M

theorem mv_kernel_python_case3_surface_toAlgorithm_supported
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) :
    ∃ alg, (mv_kernel A B C 32 16 16 1 1 1 BLOCK_N BLOCK_M).toAlgorithm? =
      Except.ok alg := by
  exact mv_kernel_surface_toAlgorithm_supported A B C 32 16 16 1 1 1
    BLOCK_N BLOCK_M

theorem mv_kernel_python_case2_one_block_compute_correct
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mv_kernel_one_block A B C 4 3 3 1 1 1 BLOCK_N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < 4)
        (fun i => (C, cOffset s 1 BLOCK_N i)))
      (expected := fun i =>
        mvSpec s A B 4 3 3 1 1 BLOCK_N BLOCK_M i) := by
  exact mv_kernel_one_block_compute_correct A B C 4 3 3 1 1 1
    BLOCK_N BLOCK_M s (mv_kernel_python_stride1_output_offset_injective s BLOCK_N)

theorem mv_kernel_python_case3_one_block_compute_correct
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mv_kernel_one_block A B C 32 16 16 1 1 1 BLOCK_N BLOCK_M)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < 32)
        (fun i => (C, cOffset s 1 BLOCK_N i)))
      (expected := fun i =>
        mvSpec s A B 32 16 16 1 1 BLOCK_N BLOCK_M i) := by
  exact mv_kernel_one_block_compute_correct A B C 32 16 16 1 1 1
    BLOCK_N BLOCK_M s (mv_kernel_python_stride1_output_offset_injective s BLOCK_N)

/-- Factored Python case-2 theorem surface for `mv_kernel`.

The shape tuple is benchmark-local glue, while the reusable reduction/store
mechanics stay in the earlier one-block correctness theorem. -/
abbrev mv_kernel_python_case2_prop
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) : Prop :=
  (∃ alg, (mv_kernel A B C 4 3 3 1 1 1 BLOCK_N BLOCK_M).toAlgorithm? =
    Except.ok alg) ∧
  ComputeCorrect.Realizes
    (kernel := mv_kernel_one_block A B C 4 3 3 1 1 1 BLOCK_N BLOCK_M)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < 4)
      (fun i => (C, cOffset s 1 BLOCK_N i)))
    (expected := fun i =>
      mvSpec s A B 4 3 3 1 1 BLOCK_N BLOCK_M i)

/-- Factored Python case-3 theorem surface for `mv_kernel`. -/
abbrev mv_kernel_python_case3_prop
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) : Prop :=
  (∃ alg, (mv_kernel A B C 32 16 16 1 1 1 BLOCK_N BLOCK_M).toAlgorithm? =
    Except.ok alg) ∧
  ComputeCorrect.Realizes
    (kernel := mv_kernel_one_block A B C 32 16 16 1 1 1 BLOCK_N BLOCK_M)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < 32)
      (fun i => (C, cOffset s 1 BLOCK_N i)))
    (expected := fun i =>
      mvSpec s A B 32 16 16 1 1 BLOCK_N BLOCK_M i)


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Public Python case-2 summary for `mv_kernel`. -/
theorem mv_kernel_python_case2_output_summary
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) :
    mv_kernel_python_case2_prop A B C BLOCK_N BLOCK_M s := by
  constructor
  · exact mv_kernel_python_case2_surface_toAlgorithm_supported A B C
      BLOCK_N BLOCK_M
  · exact mv_kernel_python_case2_one_block_compute_correct A B C
      BLOCK_N BLOCK_M s


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- Public Python case-3 summary for `mv_kernel`. -/
theorem mv_kernel_python_case3_output_summary
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) :
    mv_kernel_python_case3_prop A B C BLOCK_N BLOCK_M s := by
  constructor
  · exact mv_kernel_python_case3_surface_toAlgorithm_supported A B C
      BLOCK_N BLOCK_M
  · exact mv_kernel_python_case3_one_block_compute_correct A B C
      BLOCK_N BLOCK_M s

end Correct

/-! # ══════════ TEST-SHAPE — concrete instances / pinned scaffolding ══════════ -/

section TestShape

/-- Combined checked-shape summary for `matrix_vector_multip.py`. -/
theorem mv_kernel_python_test_shape_complete_summary
    (A B C : RegionName) (BLOCK_N BLOCK_M : Nat) (s : BlockState) :
    mv_kernel_python_case2_prop A B C BLOCK_N BLOCK_M s ∧
    mv_kernel_python_case3_prop A B C BLOCK_N BLOCK_M s := by
  constructor
  · exact mv_kernel_python_case2_output_summary A B C BLOCK_N BLOCK_M s
  · exact mv_kernel_python_case3_output_summary A B C BLOCK_N BLOCK_M s
end TestShape

end VeriTile.Bench.TritonBenchG.MatrixVectorMultip

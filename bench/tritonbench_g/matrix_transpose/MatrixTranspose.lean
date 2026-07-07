import VeriTile.Triton

/-!
# `matrix_transpose` — strict per-kernel correctness

`kernel` transposes a `SIZE_M × D_HEAD` matrix into a `D_HEAD × SIZE_M` output:
it builds 2-D strided pointer tiles `matrix_ptr` (row `size_m`, col `d_head`) and
`out_ptr` (row `d_head`, col `size_m`), loads the whole matrix tile, and stores
`tl.trans(matrix)` to `out_ptr`. The whole matrix is processed by a single
program (grid `(1,)`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`kernel[(1,)](...)`, the supplied strides from
`matrix.stride()` / `out.stride()`, and tensor allocation) is the *trusted
boundary*, not a proof obligation here. Since the grid has a single program,
the per-program statement is the whole computation.

## Proof architecture

```
kernel_output_summary                         ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ kernel_compute_correct                   ← ComputeCorrect over the cellwise store
       └─ kernel_correct                      ← algorithm-layer readback per cell
```

The spec is the transpose readback: cell `idx` of the output equals the matrix
cell at the transposed address (`matrixAddr`/`outAddr` capture the strided
address arithmetic). No optimizer/reduction oracle applies.

## Modeling boundary

Arithmetic over addresses is exact `Nat`; element values are over `ℝ` (the
upstream `float16` dtype is erased — post-erasure all dtypes unify to `ℝ`, so the
transpose is value-preserving and not bit-accurate). The proof carries an explicit no-alias side condition `hOutInj`: the
output address map `outAddr` must be injective over output tile indices (the
standard non-overlapping-store assumption); the host's contiguous `out` buffer
satisfies it.
-/

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose

open VeriTile.Triton

/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
                + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
             + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}

/-- Source address read by `kernel` at a logical output tile index. -/
def matrixAddr (matrix_stridex matrix_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.2.1.val * matrix_stridex + idx.1.val * matrix_stridey

/-- Output address written by `kernel` at a logical output tile index. -/
def outAddr (out_stridex out_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.1.val * out_stridex + idx.2.1.val * out_stridey

/-- Algorithm-layer cellwise correctness for `kernel`.

The injectivity hypothesis is the standard no-alias condition for the output
tile: if two logical output indices map to the same memory cell, a cellwise
postcondition cannot distinguish them after the store fold. -/
theorem kernel_correct
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx))
    (s' : BlockState)
    (hExec : exec (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD) s = some s') :
    ∀ idx : TileIndex [D_HEAD, SIZE_M],
      s'.readMem Out (outAddr out_stridex out_stridey idx)
        = s.readMem M (matrixAddr matrix_stridex matrix_stridey idx) := by
  intro idx
  simp [exec, kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.ptrAdd, Tile.expandDim, Tile.transpose,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst s'
  have hRawInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        idx.1.val * out_stridex + idx.2.1.val * out_stridey) := by
    simpa [outAddr] using hOutInj
  simp [matrixAddr, outAddr]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]

/-- Compute-facing cellwise correctness for `kernel`. -/
theorem kernel_compute_correct
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx)) :
    ComputeCorrect.Realizes
      (kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [D_HEAD, SIZE_M] =>
          some (Out, outAddr out_stridex out_stridey idx))
      (expected := fun idx => s.readMem M (matrixAddr matrix_stridex matrix_stridey idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := kernel_correct M Out matrix_stridex matrix_stridey out_stridex
    out_stridey SIZE_M D_HEAD s hOutInj s' hExec idx
  exact h

/-- Per-kernel output summary for `kernel`: the DSL surface lowers to the
algorithm layer, and the cellwise store to `Out` is compute-correct — every
output cell `idx` holds the transposed matrix cell, under the no-alias side
condition `hOutInj`. -/
theorem kernel_output_summary
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx)) :
    (∃ alg, (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [D_HEAD, SIZE_M] =>
          some (Out, outAddr out_stridex out_stridey idx))
      (expected := fun idx => s.readMem M (matrixAddr matrix_stridex matrix_stridey idx)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact kernel_compute_correct M Out matrix_stridex matrix_stridey out_stridex
    out_stridey SIZE_M D_HEAD s hOutInj

end VeriTile.Bench.TritonBenchG.MatrixTranspose

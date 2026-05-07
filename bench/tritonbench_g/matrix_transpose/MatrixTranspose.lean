import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
  matrix_ptr = M + d_head_arange[None, :] * $(matrix_stridey)
                + size_m_arange[:, None] * $(matrix_stridex)
  out_ptr = Out + d_head_arange[None, :] * $(out_stridex)
             + size_m_arange[:, None] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, matrix)
}

/-- Source address read by `kernel` at a logical output tile index. -/
def matrixAddr (matrix_stridex matrix_stridey : Nat)
    (idx : TileIndex [SIZE_M, D_HEAD]) : Nat :=
  idx.2.1.val * matrix_stridey + idx.1.val * matrix_stridex

/-- Output address written by `kernel` at a logical output tile index. -/
def outAddr (out_stridex out_stridey : Nat)
    (idx : TileIndex [SIZE_M, D_HEAD]) : Nat :=
  idx.2.1.val * out_stridex + idx.1.val * out_stridey

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
      (fun idx : TileIndex [SIZE_M, D_HEAD] => outAddr out_stridex out_stridey idx))
    (s' : BlockState)
    (hExec : exec (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD) s = some s') :
    ∀ idx : TileIndex [SIZE_M, D_HEAD],
      s'.readMem Out (outAddr out_stridex out_stridey idx)
        = s.readMem M (matrixAddr matrix_stridex matrix_stridey idx) := by
  intro idx
  simp [exec, kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst s'
  have hRawInj : Function.Injective
      (fun idx : TileIndex [SIZE_M, D_HEAD] =>
        idx.2.1.val * out_stridex + idx.1.val * out_stridey) := by
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
      (fun idx : TileIndex [SIZE_M, D_HEAD] => outAddr out_stridex out_stridey idx)) :
    ComputeKernel.ComputeCorrect
      (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD)
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [SIZE_M, D_HEAD],
          s'.readMem Out (outAddr out_stridex out_stridey idx)
            = s.readMem M (matrixAddr matrix_stridex matrix_stridey idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := kernel_correct M Out matrix_stridex matrix_stridey out_stridex
    out_stridey SIZE_M D_HEAD s hOutInj s' hExec idx
  exact h

end VeriTile.Bench.TritonBenchG.MatrixTranspose

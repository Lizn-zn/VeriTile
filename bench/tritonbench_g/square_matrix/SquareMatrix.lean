import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.SquareMatrix

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `square_matrix.py`'s `square_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def square_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-inf)
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}

/-- Algorithm-layer correctness for `square_kernel`.

For the current row pid, active columns write `x^2`; inactive tail columns are
preserved. -/
theorem square_kernel_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * output_row_stride + i.val
      (exec (square_kernel output_ptr input_ptr input_row_stride output_row_stride
            n_cols BLOCK_SIZE) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < n_cols then xs i * xs i else s.readMem output_ptr outAddr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * output_row_stride + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, square_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.mul,
        ComparableDType.lt]
  unfold InputRowLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · simp [hi, h_x]
  · simp [hi]

/-- Compute-facing correctness for `square_kernel`. -/
theorem square_kernel_compute_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs) :
    ComputeKernel.ComputeCorrect
      (square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK_SIZE,
          let outAddr := s.pid * output_row_stride + i.val
          s'.readMem output_ptr outAddr
            = if i.val < n_cols then xs i * xs i else s.readMem output_ptr outAddr) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := square_kernel_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE hBlockSize s xs h_x i
  rw [hExec] at hi
  exact Option.some.inj hi

end VeriTile.Bench.TritonBenchG.SquareMatrix

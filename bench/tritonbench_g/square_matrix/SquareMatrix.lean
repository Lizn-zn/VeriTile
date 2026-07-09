import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `square_matrix` — strict per-kernel correctness

`square_kernel` squares a matrix row-wise: program `row_idx` loads one row of
`input_ptr` (a `BLOCK_SIZE`-wide tile starting at `row_idx·input_row_stride`,
masked by `col_offsets < n_cols` with out-of-bounds lanes defaulting to `-inf`),
computes `row * row`, and stores `x²` to the corresponding row of `output_ptr`,
masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`square_kernel[(n_rows,)](...)`, the one-program-per-row
grid, the host-side `BLOCK_SIZE = next_power_of_2(n_cols)` choice, `num_warps`,
and how the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `row_idx` is universally
quantified (as `s.pid`), the per-program statement covers every row of the grid.

## Proof architecture

```
square_kernel_output_summary                  ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ square_kernel_compute_correct            ← ComputeCorrect over the masked store
       └─ square_kernel_correct               ← algorithm-layer readback per column
```

The spec is plain elementwise `xs i * xs i` — no optimizer/reduction oracle
applies. The row is presented via `InputRowLoadedAt` (the values each column
lane loads from the strided row).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The `other=-float('inf')` masked-load default does not affect any
in-bounds (`col_offsets < n_cols`) lane, which is all the spec constrains;
out-of-bounds output columns are preserved verbatim. No output/input
disjointness is assumed: the row is read into registers before the scatter.
-/

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
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
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
  simp [exec, square_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.mul,
        ComparableDType.lt]
  unfold InputRowLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i => xs i * xs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := square_kernel_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa [hActive] using Option.some.inj hi

/-- Per-kernel output summary for `square_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active column holds `xs i * xs i`, out-of-bounds columns are preserved. -/
theorem square_kernel_output_summary
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs) :
    (∃ alg, (square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i => xs i * xs i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact square_kernel_compute_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.SquareMatrix

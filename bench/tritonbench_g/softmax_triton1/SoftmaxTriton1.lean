import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `softmax_triton1` — strict per-kernel correctness

`softmax_kernel` is a row-wise stable softmax: program `row_idx` loads one row
(block `[0, BLOCK_SIZE)`, masked by `col_offsets < n_cols`, with masked lanes
read as `-inf`), subtracts the row max, exponentiates, divides by the row sum,
and stores the normalized row back, masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel[(n_rows,)](...)`, the one-program-per-row
grid, the `BLOCK_SIZE = next_power_of_2(n_cols)` choice, `num_warps` autotuning,
and how the runtime composes per-row writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `row_idx`/`pid` is universally
quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
softmax_kernel_output_summary                 ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ softmax_kernel_compute_correct           ← ComputeCorrect over the masked store
       └─ softmax_kernel_correct              ← algorithm-layer readback per lane
```

The spec `softmaxSpec` is the exact stable softmax: `reduceMax` over the masked
row, lane-wise `exp(row - max)`, `reduceSum` for the denominator, then the
lane-wise quotient. In-bounds lanes hold `softmaxSpec`, out-of-bounds lanes are
preserved.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`; the manual `num_warps` heuristic is not
modeled. The reduction runs over the full `BLOCK_SIZE` block, but masked lanes
load `⊥` (matching `other=-float("inf")`), so the reduction-over-padded-block
matches the upstream `-inf` semantics. No output/input disjointness is assumed:
the row is read into registers before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton1

open VeriTile.Triton

/-- Faithful 1:1 transcription of `softmax_triton1.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Masked input row tile used by `softmax_kernel`. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (input_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * input_row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem input_ptr off) else none }

/-- Exact stable-softmax value computed by the kernel at lane `idx`. -/
noncomputable def softmaxSpec
    (s : BlockState) (input_ptr : RegionName)
    (input_row_stride n_cols BLOCK_SIZE : Nat) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile s input_ptr input_row_stride n_cols BLOCK_SIZE
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer cellwise correctness for `softmax_kernel`. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
          n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * output_row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec s input_ptr input_row_stride n_cols BLOCK_SIZE i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB] at hExec
    subst s'
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, softmaxSpec, softmaxInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, hB]
      congr
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing cellwise correctness for `softmax_kernel`. -/
theorem softmax_kernel_compute_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr input_row_stride n_cols BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `softmax_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
in-bounds lane holds `softmaxSpec`, out-of-bounds lanes are preserved. -/
theorem softmax_kernel_output_summary
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr input_row_stride n_cols BLOCK_SIZE i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact softmax_kernel_compute_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE s

end VeriTile.Bench.TritonBenchG.SoftmaxTriton1

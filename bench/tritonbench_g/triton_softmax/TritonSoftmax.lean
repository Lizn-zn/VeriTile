import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax

open VeriTile.Triton

/-- Faithful 1:1 transcription of `triton_softmax.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- Python `-float('inf')` → DSL literal `-inf`. -/
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  out_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  row = tl.load(row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols), other=-inf)
  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(out_row_start_ptr + tl.arange(0, $(BLOCK_SIZE)),
    softmax_output, mask=tl.arange(0, $(BLOCK_SIZE)) < $(n_cols))
}

/-- Masked input row tile used by `softmax_kernel`. Masked lanes are `⊥`,
matching `other=-inf`. -/
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
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * output_row_stride + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB] at hExec
    subst s'
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
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

end VeriTile.Bench.TritonBenchG.TritonSoftmax

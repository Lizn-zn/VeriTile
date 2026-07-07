import VeriTile.Triton

/-!
# `softmax_triton3` — strict per-kernel correctness

`softmax_kernel` is a row-wise stable softmax with an optional additive mask:
program `row_idx` loads one row (masked by `col_offsets < n_cols`, masked lanes
read as `-inf`), subtracts the row max, optionally (`mask_ptr is not None`, here
the `HAS_MASK` constexpr) adds a per-lane additive mask, exponentiates, divides
by the row sum, and stores back masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel[grid](...)`, the one-program-per-row grid
(or the large-input `BLOCK_M` tiled grid), the `BLOCK_SIZE = next_power_of_2`
choice, `num_warps` autotuning, and how the runtime composes per-row writes) is
the *trusted boundary*, not a proof obligation here. Because `row_idx`/`pid` is
universally quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
softmax_kernel_output_summary                 ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ softmax_kernel_compute_correct           ← ComputeCorrect over the masked store
       └─ softmax_kernel_correct              ← algorithm-layer readback per lane
                                                 (case-splits on HAS_MASK)
```

The spec `softmaxSpec` is the exact stable softmax: `reduceMax` over the masked
row, lane-wise `exp(row - max [+ mask])`, `reduceSum` for the denominator, then
the lane-wise quotient. `HAS_MASK` gates the additive-mask branch; in-bounds
lanes hold `softmaxSpec`, out-of-bounds lanes are preserved.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`; the manual `num_warps` heuristic is not
modeled. The `.to(tl.float32)` casts on the loaded row and mask reduce to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). The
reduction runs over the full `BLOCK_SIZE` block, but masked input lanes load `⊥`
(matching `other=-float("inf")`) and masked mask lanes load `0`, so the
reduction-over-padded-block matches the upstream semantics. No output/input
disjointness is assumed: the row is read into registers before the masked
scatter.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3

open VeriTile.Triton

/-- Faithful transcription of `softmax_triton3.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `mask_ptr is not None` -> Lean `HAS_MASK : Bool` constexpr gate.
- Python `.to(tl.float32)` casts are represented explicitly in the Compute
  layer; the algorithm-layer theorem observes their Real projection. -/
def softmax_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Masked input row tile used by `softmax_kernel`. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem input_ptr off) else none }

/-- Optional additive mask tile. Inactive lanes are `0`, matching
`tl.load(..., other=0)`, but the final store is still masked by `n_cols`. -/
noncomputable def softmaxMaskTile
    (s : BlockState) (mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * row_stride + idx.1.val
      if idx.1.val < n_cols then some (s.readMem mask_ptr off) else some 0 }

/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`,
including the optional additive mask path from `softmax_triton3.py`. -/
noncomputable def softmaxSpec
    (s : BlockState) (input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile s input_ptr row_stride n_cols BLOCK_SIZE
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        if HAS_MASK then
          Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil) shifted
            (softmaxMaskTile s mask_ptr row_stride n_cols BLOCK_SIZE)
        else shifted
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer cellwise correctness for `softmax_kernel`. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE HAS_MASK) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec s input_ptr mask_ptr row_stride n_cols BLOCK_SIZE HAS_MASK i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · cases HAS_MASK <;>
      simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            FloatDType.cast, hB] at hExec
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.sub, hB]
        congr
      · simp [hi]
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile, softmaxMaskTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.add, NumericDType.sub, hB]
        congr
      · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing cellwise correctness for `softmax_kernel`. -/
theorem softmax_kernel_compute_correct
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr mask_ptr row_stride n_cols BLOCK_SIZE HAS_MASK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_correct output_ptr input_ptr mask_ptr row_stride
    n_cols BLOCK_SIZE HAS_MASK s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `softmax_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
in-bounds lane holds `softmaxSpec` (including the optional additive-mask branch),
out-of-bounds lanes are preserved. -/
theorem softmax_kernel_output_summary
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s : BlockState) :
    (∃ alg, (softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * row_stride + i.val)))
      (expected := fun i =>
        softmaxSpec s input_ptr mask_ptr row_stride n_cols BLOCK_SIZE HAS_MASK i) := by
  refine ⟨?_, ?_⟩
  · simp [softmax_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact softmax_kernel_compute_correct output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE HAS_MASK s

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3

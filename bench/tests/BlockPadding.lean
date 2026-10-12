import VeriTile.Triton.Semantics.BlockPadding
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure

/-! Low-level semantic regressions for explicit versus unspecified padding.
The DSL migration is checked separately; these tests make no GPU claim. -/
namespace BlockPaddingTests
open VeriTile.Triton

private def ptr : Op .blockPtr [2] :=
  .makeBlockPtrDynOffsets "x" (.constNat 5) [3] [2] [2] [.constNat 2]

theorem real_unspecified (s : BlockState) :
    (evalOp (.load .real (.blockPtr ptr [0]) .none .unspecified) s).map
      (fun v => (v.data (0, PUnit.unit), v.data (1, PUnit.unit))) =
      some (some (s.readMem "x" 9), some (s.undef "x" 11)) := by
  simp [evalOp, ptr, TileShape.indexToList, BlockState.paddingValue,
    BlockState.unspecifiedValue]
  rfl

theorem zero_ignores_oracle (s : BlockState) :
    (evalOp (.load .real (.blockPtr ptr [0]) .none .zero) s).map
      (fun v => v.data (1, PUnit.unit)) = some (some 0) := by
  simp [evalOp, ptr, TileShape.indexToList, BlockState.defaultCarrier]
  rfl

theorem unchecked_reads_ignore_padding (s : BlockState) (padding : PaddingOption) :
    (evalOp (.load .real (.blockPtr ptr []) .none padding) s).map
      (fun v => v.data (1, PUnit.unit)) = some (some (s.readMem "x" 11)) := by
  simp [evalOp, ptr, TileShape.indexToList]
  rfl

theorem rounding_model_keeps_unspecified (R : RoundingModel) (s : BlockState) :
    (evalOpR R (.load .real (.blockPtr ptr [0]) .none .unspecified) s).map
      (fun v => v.data (1, PUnit.unit)) = some (some (s.undef "x" 11)) := by
  simp [evalOpR, ptr, TileShape.indexToList, BlockState.paddingValue,
    BlockState.unspecifiedValue]
  rfl

theorem nonzero_oracle_changes_tail (s : BlockState) (hu : s.undef "x" 11 = 7) :
    evalOp (.load .real (.blockPtr ptr [0]) .none .unspecified) s ≠
      evalOp (.load .real (.blockPtr ptr [0]) .none .zero) s := by
  intro h
  have hh := congrArg (fun v : Option (Tile .real [2]) =>
    v.map (fun tile => tile.data (1, PUnit.unit))) h
  simp [evalOp, ptr, TileShape.indexToList, BlockState.paddingValue,
    BlockState.unspecifiedValue, BlockState.defaultCarrier, hu] at hh
  have bad : (7 : ℝ) = 0 := Option.some.inj hh
  norm_num at bad

theorem structural_unspecified {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    (FP.Structural.evalOp M (some .fp32)
      (.load .bf16 (.blockPtr ptr [0]) .none .unspecified) s).map
      (fun v => (v (0, PUnit.unit), v (1, PUnit.unit))) =
      some ((s.mem "x" 9).read .bf16, s.undef .bf16 "x" 11) := by
  simp [FP.Structural.evalOp, FP.Structural.evalOffsets, ptr, TileShape.indexToList]

theorem compute_projection_preserves_padding :
    (ComputeOp.load .fp32 (.blockPtr ptr [0]) .none .unspecified).toAlgorithm? =
      .ok (.load .real (.blockPtr ptr [0]) .none .unspecified) := rfl

theorem structural_compute_load_keeps_precision {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    (FP.Structural.evalComputeOp M
      (.load .fp32 (.blockPtr ptr [0]) .none .unspecified) s).map
      (fun v => v (1, PUnit.unit)) = some (M.fp32Load (s.undef .real "x" 11)) := by
  simp [FP.Structural.evalComputeOp, FP.Structural.evalOp,
    FP.Structural.evalOffsets, ptr, TileShape.indexToList]
  rfl

theorem precision_erasure_preserves_padding :
    (Op.load .bf16 (.blockPtr ptr [0]) .none .unspecified).eraseDType =
      .load .real (.blockPtr ptr [0]) .none .unspecified := by
  simp [ptr, Op.eraseDType, VeriTile.Triton.eraseDType]

end BlockPaddingTests

import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def reductionBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  match ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body[1]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => (no.drop 7).take 3
  | _ => []

private theorem mul_exp_finite (z : ℝ) (a : WithBot ℝ) :
    WithBot.realMul (some z) (WithBot.realExp a) =
      some (z * (WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

 theorem reduction_step (Y X : RegionName) (M N B : Nat) (hB : 0 < B)
    (s : BlockState) (mf : TileIndex [B] → WithBot ℝ) (zf : TileIndex [B] → ℝ)
    (hm : s.regs .real [B] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B] "z" = some ⟨fun i ↦ some (zf i)⟩) :
    ∃ next, stepStmts (reductionBody Y X M N B) s = some next ∧
      next.mem = s.mem ∧ next.pids = s.pids ∧ next.numPids = s.numPids ∧
      next.undef = s.undef ∧
      next.regs .nat [] "pid_m" = s.regs .nat [] "pid_m" ∧
      next.regs .ptr [] "input_ptr" = s.regs .ptr [] "input_ptr" ∧
      next.regs .ptr [] "output_ptr" = s.regs .ptr [] "output_ptr" ∧
      next.regs .nat [] "previous_multiple" = s.regs .nat [] "previous_multiple" ∧
      next.regs .real [] "m" = some (Tile.scalar (globalMax hB mf)) ∧
      next.regs .real [] "z" = some (Tile.scalar (some (globalSum mf zf (globalMax hB mf)))) := by
  simp [reductionBody, softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hm, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, List.getElem_cons_zero, hB, ↓reduceDIte]
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · simp
  · simp
  · simp only [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro i
    rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    apply congrArg some
    apply Tile.ext
    intro i
    simp only [Tile.reduceSum, Tile.reduceSumDrop_data]
    change (∑ j : Fin B, WithBot.realMul (some (zf (j, PUnit.unit)))
      (WithBot.realExp (WithBot.realSub (mf (j, PUnit.unit))
        (globalMax hB mf)))) = some (globalSum mf zf (globalMax hB mf))
    simp only [mul_exp_finite]
    exact (WithBot.coe_sum Finset.univ _).symm

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

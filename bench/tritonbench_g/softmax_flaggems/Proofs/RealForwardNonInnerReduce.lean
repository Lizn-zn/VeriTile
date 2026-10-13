import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def reductionBody (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => (no.drop 3).take 3
  | _ => []

noncomputable def columnMax {B C : Nat} (hB : 0 < B)
    (mf : TileIndex [B, C] → WithBot ℝ) (k : Fin C) : WithBot ℝ :=
  Finset.univ.sup' (⟨⟨0, hB⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin B)).Nonempty)
    (fun i ↦ mf (i, k, PUnit.unit))

noncomputable def columnSum {B C : Nat} (mf : TileIndex [B, C] → WithBot ℝ)
    (zf : TileIndex [B, C] → ℝ) (mx : Fin C → WithBot ℝ) (k : Fin C) : ℝ :=
  ∑ i : Fin B, zf (i, k, PUnit.unit) *
    (WithBot.realExp (WithBot.realSub (mf (i, k, PUnit.unit)) (mx k))).unbotD 0

private theorem mul_exp_finite (z : ℝ) (a : WithBot ℝ) :
    WithBot.realMul (some z) (WithBot.realExp a) =
      some (z * (WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

 theorem reduction_step (Y X : RegionName) (M N K B C : Nat) (hB : 0 < B)
    (s : BlockState) (mf : TileIndex [B, C] → WithBot ℝ) (zf : TileIndex [B, C] → ℝ)
    (hm : s.regs .real [B, C] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B, C] "z" = some ⟨fun i ↦ some (zf i)⟩) :
    ∃ next, stepStmts (reductionBody Y X M N K B C) s = some next ∧
      next.mem = s.mem ∧ next.pids = s.pids ∧ next.numPids = s.numPids ∧
      next.undef = s.undef ∧
      next.regs .nat [] "pid_m" = s.regs .nat [] "pid_m" ∧
      next.regs .nat [C] "k_offsets" = s.regs .nat [C] "k_offsets" ∧
      next.regs .real [C] "m" = some ⟨fun i ↦ columnMax hB mf i.1⟩ ∧
      next.regs .real [C] "z" = some ⟨fun i ↦ some (columnSum mf zf (columnMax hB mf) i.1)⟩ := by
  simp [reductionBody, softmax_kernel_non_inner,
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
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩
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
    change (∑ j : Fin B, WithBot.realMul (some (zf (j, i.1, PUnit.unit)))
      (WithBot.realExp (WithBot.realSub (mf (j, i.1, PUnit.unit))
        (columnMax hB mf i.1)))) = some (columnSum mf zf (columnMax hB mf) i.1)
    simp only [mul_exp_finite]
    exact (WithBot.coe_sum Finset.univ _).symm

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

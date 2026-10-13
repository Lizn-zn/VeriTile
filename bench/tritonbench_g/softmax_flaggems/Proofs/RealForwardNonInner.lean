import bench.tritonbench_g.softmax_flaggems.Kernels
import bench.tritonbench_g.softmax_flaggems.Proofs.RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

private def branch (k : Kernel) : List Stmt :=
  match (k.body[3]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => no
  | _ => []

private def loopBody (stmts : List Stmt) (i : Nat) : List Stmt :=
  match (stmts[i]? : Option Stmt) with
  | some (.forRangeDyn _ _ _ _ body) => body
  | some (.forRange _ _ _ _ body) => body
  | _ => []

def pass1Body (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  loopBody (branch
    (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel) 2

theorem pass1Body_extract (Y X : RegionName) (M N K B C : Nat) :
    pass1Body Y X M N K B C =
      match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
      | some (.ifThenElse _ _ no) =>
        match (no[2]? : Option Stmt) with
        | some (.forRangeDyn _ _ _ _ body) => body
        | some (.forRange _ _ _ _ body) => body
        | _ => []
      | _ => [] := by
  unfold pass1Body branch
  cases h : ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
  | none => rfl
  | some st => cases st <;> rfl

noncomputable def score (s : BlockState) (X : RegionName)
    (row N K kbase start B C : Nat) (i : TileIndex [B, C]) : WithBot ℝ :=
  if start + i.1.val < N ∧ kbase + i.2.1.val < K then
    some (s.readMem X (row * N * K + (start + i.1.val) * K + (kbase + i.2.1.val)))
  else ⊥

private theorem option_max_eq (a b : WithBot ℝ) :
    @max (Option ℝ) Option.instMax a b = a ⊔ b := by
  cases a <;> cases b <;> rfl

private theorem select_max (a b : WithBot ℝ) :
    (if b < a then a else b) = a ⊔ b := by
  rw [← max_def_lt, max_comm]

private theorem exp_finite (a : WithBot ℝ) :
    WithBot.realExp a = some ((WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

private theorem update_z (m x : WithBot ℝ) (z : ℝ) :
    WithBot.realAdd
      (WithBot.realMul (some z) (WithBot.realExp (WithBot.realSub m (m ⊔ x))))
      (WithBot.realExp (WithBot.realSub x (m ⊔ x))) =
      some (RealMath.maskedOnlineStep (m, z) x).2 := by
  rw [exp_finite (WithBot.realSub m (m ⊔ x)),
    exp_finite (WithBot.realSub x (m ⊔ x))]
  rfl

 theorem pass1_step (Y X : RegionName) (M N K B C row kbase start : Nat)
    (s : BlockState) (mf : TileIndex [B, C] → WithBot ℝ) (zf : TileIndex [B, C] → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩)
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [B, C] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B, C] "z" = some ⟨fun i ↦ some (zf i)⟩) :
    ∃ next, stepStmts (pass1Body Y X M N K B C) s = some next ∧
      next.mem = s.mem ∧ next.pids = s.pids ∧ next.numPids = s.numPids ∧
      next.undef = s.undef ∧
      next.regs .nat [] "pid_m" = some (Tile.scalar row) ∧
      next.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩ ∧
      next.regs .real [B, C] "m" = some ⟨fun i ↦
        (RealMath.maskedOnlineStep (mf i, zf i) (score s X row N K kbase start B C i)).1⟩ ∧
      next.regs .real [B, C] "z" = some ⟨fun i ↦ some
        (RealMath.maskedOnlineStep (mf i, zf i) (score s X row N K kbase start B C i)).2⟩ := by
  simp [pass1Body, branch, loopBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    List.getElem?_cons_zero, List.getElem?_cons_succ]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hrow, hk, hstart, hm, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hrow
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hk
  · simp only [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      score, RealMath.maskedOnlineStep, option_max_eq]
    change (if score s X row N K kbase start B C i < mf i then mf i
      else score s X row N K kbase start B C i) = _
    exact select_max _ _
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    apply congrArg some
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      score, option_max_eq, update_z]
    simp only [select_max]
    exact update_z (mf i) (score s X row N K kbase start B C i) (zf i)

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

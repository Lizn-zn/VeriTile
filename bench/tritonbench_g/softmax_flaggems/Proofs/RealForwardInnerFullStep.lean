import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInner
import bench.tritonbench_g.softmax_flaggems.Proofs.RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

private def branch (k : Kernel) : List Stmt :=
  match (k.body[1]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => no
  | _ => []

private def loopBody (stmts : List Stmt) (i : Nat) : List Stmt :=
  match (stmts[i]? : Option Stmt) with
  | some (.forRangeDyn _ _ _ _ body) => body
  | some (.forRange _ _ _ _ body) => body
  | _ => []

def fullBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  loopBody (branch
    (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel) 5

theorem fullBody_extract (Y X : RegionName) (M N B : Nat) :
    fullBody Y X M N B =
      match ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body[1]? : Option Stmt) with
      | some (.ifThenElse _ _ no) =>
        match (no[5]? : Option Stmt) with
        | some (.forRangeDyn _ _ _ _ body) => body
        | some (.forRange _ _ _ _ body) => body
        | _ => []
      | _ => [] := by
  unfold fullBody branch
  cases h : ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body[1]? : Option Stmt) with
  | none => rfl
  | some st => cases st <;> rfl

noncomputable def fullScore (s : BlockState) (X : RegionName)
    (row N start B : Nat) (i : TileIndex [B]) : WithBot ℝ :=
  some (s.readMem X (row * N + (start + i.1.val)))

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

 theorem full_step (Y X : RegionName) (M N B row start : Nat)
    (s : BlockState) (mf : TileIndex [B] → WithBot ℝ) (zf : TileIndex [B] → ℝ)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hp : s.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)))
    (hstart : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .real [B] "m" = some ⟨mf⟩)
    (hz : s.regs .real [B] "z" = some ⟨fun i ↦ some (zf i)⟩) :
    ∃ next, stepStmts (fullBody Y X M N B) s = some next ∧
      next.mem = s.mem ∧ next.pids = s.pids ∧ next.numPids = s.numPids ∧
      next.undef = s.undef ∧
      next.regs .ptr [] "output_ptr" = s.regs .ptr [] "output_ptr" ∧
      next.regs .nat [] "previous_multiple" = s.regs .nat [] "previous_multiple" ∧
      next.regs .nat [] "pid_m" = some (Tile.scalar row) ∧
      next.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N)) ∧
      next.regs .real [B] "m" = some ⟨fun i ↦
        (RealMath.maskedOnlineStep (mf i, zf i) (fullScore s X row N start B i)).1⟩ ∧
      next.regs .real [B] "z" = some ⟨fun i ↦ some
        (RealMath.maskedOnlineStep (mf i, zf i) (fullScore s X row N start B i)).2⟩ := by
  simp [fullBody, branch, loopBody, softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    List.getElem?_cons_zero, List.getElem?_cons_succ]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hrow, hp, hstart, hm, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hrow
  · simpa only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true] using hp
  · simp only [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      fullScore, RealMath.maskedOnlineStep, option_max_eq]
    change (if fullScore s X row N start B i < mf i then mf i
      else fullScore s X row N start B i) = _
    exact select_max _ _
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    apply congrArg some
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      fullScore, option_max_eq, update_z]
    simp only [select_max]
    exact update_z (mf i) (fullScore s X row N start B i) (zf i)

theorem fullScore_eq_score (s : BlockState) (X : RegionName) (row N start B : Nat)
    (h : start + B ≤ N) (i : TileIndex [B]) :
    fullScore s X row N start B i = score s X row N start B i := by
  have hi := i.1.isLt
  simp [fullScore, score, show start + i.1.val < N by omega]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

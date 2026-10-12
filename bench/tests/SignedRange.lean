import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Memory.Typing.Checker
import VeriTile.Meta.StatementAudit

/-! Signed loop bounds are evaluated once. Negative indices, an empty signed
interval, missing bounds and body failures must agree across both interpreters.
These are exact control operations, independent of FP atomic assumptions. -/
namespace SignedRangeTests
open VeriTile.Triton

/-- Changing the original bound registers does not change the remaining
iterations: the index sequence is -3, -1, 1. -/
def changingBounds : ComputeKernel := triton {
  lo = $((-3 : Int))
  hi = $((2 : Int))
  stride = $((2 : Nat))
  total = $((0 : Int))
  for i in range(lo, hi, stride) {
    total = total + i
    lo = $((-99 : Int))
    hi = $((-100 : Int))
    stride = $((0 : Nat))
  }
}

theorem signed_lowering : changingBounds.surfaceBody = [
    .assign .int [] "lo" (.alg (.constInt (-3))),
    .assign .int [] "hi" (.alg (.constInt 2)),
    .assign .nat [] "stride" (.alg (.constNat 2)),
    .assign .int [] "total" (.alg (.constInt 0)),
    .forRangeInt "i" (.ref .int [] "lo") (.ref .int [] "hi")
      (.ref .nat [] "stride") [
        .assign .int [] "total" (.alg
          (.add .int .nil (.ref .int [] "total") (.ref .int [] "i"))),
        .assign .int [] "lo" (.alg (.constInt (-99))),
        .assign .int [] "hi" (.alg (.constInt (-100))),
        .assign .nat [] "stride" (.alg (.constNat 0))]] := rfl

noncomputable def realExecution (k : ComputeKernel) (s : BlockState) : Option BlockState :=
  k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)

set_option linter.unusedSimpArgs false in
set_option linter.unusedTactic false in
theorem real_keeps_negative_indices (s : BlockState) :
    (realExecution changingBounds s).map (fun t => t.regs .int [] "total") =
      some (some (Tile.scalar (-3))) := by
  simp only [realExecution, changingBounds, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, pure, bind, Option.bind_some, VeriTile.Triton.exec]
  iterate 8
    all_goals try simp only [stepStmts, stepStmt, VeriTile.Triton.evalOp]
    all_goals try simp [BlockState.setReg, Tile.bop, Tile.scalar, NumericDType.add,
      Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]
    all_goals try rw [stepForRangeIntAux]

set_option linter.unusedSimpArgs false in
set_option linter.unusedTactic false in
theorem fp_keeps_negative_indices {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    (FP.Structural.exec M changingBounds s).map (fun t => t.regs .int [] "total") =
      some (some (fun _ => (-3 : Int))) := by
  simp only [changingBounds, FP.Structural.exec]
  iterate 8
    all_goals try simp only [FP.Structural.run, FP.Structural.step,
      FP.Structural.evalExpr, FP.Structural.evalOp]
    all_goals try simp [FP.Structural.State.setReg, FP.Structural.numeric, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals try rw [FP.Structural.rangeInt]
  rfl

theorem empty_signed_interval (body : List Stmt) (s : BlockState) :
    stepStmt (.forRangeInt "i" (.constInt 0) (.constInt (-4)) (.constNat 2) body) s =
      some s := by
  rw [stepStmt]
  simp only [VeriTile.Triton.evalOp, bind, Option.bind_some, Tile.scalar]
  rw [stepForRangeIntAux]
  norm_num

theorem fp_empty_signed_interval {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (body : List ComputeStmt) (s : FP.Structural.State α) :
    FP.Structural.step M
      (.forRangeInt "i" (.constInt 0) (.constInt (-4)) (.constNat 2) body) s = some s := by
  rw [FP.Structural.step]
  change FP.Structural.rangeInt M "i" 0 (-4) 2 body s = some s
  rw [FP.Structural.rangeInt]
  norm_num

theorem missing_bound_fails (body : List Stmt) (s : BlockState)
    (h : s.regs .int [] "lo" = none) :
    stepStmt (.forRangeInt "i" (.ref .int [] "lo") (.constInt 4) (.constNat 1) body) s =
      none := by
  simp [VeriTile.Triton.evalOp, h]

theorem fp_missing_bound_fails {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (body : List ComputeStmt) (s : FP.Structural.State α)
    (h : s.regs .int [] "lo" = none) :
    FP.Structural.step M
      (.forRangeInt "i" (.ref .int [] "lo") (.constInt 4) (.constNat 1) body) s = none := by
  simp [FP.Structural.step, FP.Structural.evalOp, h]

theorem body_failure_is_preserved (s : BlockState)
    (h : s.regs .nat [] "missing" = none) :
    stepStmt (.forRangeInt "i" (.constInt (-2)) (.constInt 1) (.constNat 1)
      [.assign .nat [] "result" (.ref .nat [] "missing")]) s = none := by
  rw [stepStmt]
  simp only [VeriTile.Triton.evalOp, bind, Option.bind_some, Tile.scalar]
  rw [stepForRangeIntAux]
  norm_num
  simp only [stepStmts, stepStmt, VeriTile.Triton.evalOp]
  simp [BlockState.setReg, h]

theorem fp_body_failure_is_preserved {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α)
    (h : s.regs .nat [] "missing" = none) :
    FP.Structural.step M
      (.forRangeInt "i" (.constInt (-2)) (.constInt 1) (.constNat 1)
        [.assign .nat [] "result" (.alg (.ref .nat [] "missing"))]) s = none := by
  rw [FP.Structural.step]
  change FP.Structural.rangeInt M "i" (-2) 1 1
    [.assign .nat [] "result" (.alg (.ref .nat [] "missing"))] s = none
  rw [FP.Structural.rangeInt]
  norm_num
  simp only [FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp]
  simp [FP.Structural.State.setReg, h]

/-- Explicitly signed constexpr bounds use the same signed loop node. -/
def constantBounds : ComputeKernel := triton {
  for i in range($((-3 : Int)), $((2 : Int)), $(2)) {
    j = i
  }
}

theorem signed_constants_lowering : constantBounds.surfaceBody =
    [.forRangeInt "i" (.constInt (-3)) (.constInt 2) (.constNat 2)
      [.assign .int [] "j" (.alg (.ref .int [] "i"))]] := rfl

theorem checker_tracks_signed_index :
    (constantBounds.toAlgorithm?.map (fun k => k.checkStrict (fun _ => none))) =
      .ok (.ok ()) := by
  simp only [constantBounds, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, Except.map, bind, Except.bind]
  simp only [Kernel.checkStrict, Kernel.checkWith, StmtList.check, Stmt.check, Op.check,
    CheckCtx.setReg, Stmt.assignEntry, bind, Except.bind]
  rfl

theorem erasure_keeps_signed_range (idx : RegName) (start stop : Int)
    (stride : Nat) (body : List Stmt) :
    (Stmt.forRangeInt idx (.constInt start) (.constInt stop) (.constNat stride) body).eraseDType =
      .forRangeInt idx (.constInt start) (.constInt stop) (.constNat stride)
        (Stmt.eraseDTypeList body) := by
  simp [Stmt.eraseDType, Op.eraseDType]

#guard_msgs (drop info) in
#auditModuleAxioms
end SignedRangeTests

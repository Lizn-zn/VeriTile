import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace StaticPrintExecutionTests
open VeriTile.Triton FP.Structural

def printed : ComputeKernel := triton {
  tl.static_print("compile-time diagnostic")
  i = $((-2 : Int))
}

/-- A compile-time diagnostic contributes no runtime failure or memory effect. -/
theorem executes {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    FP.Structural.exec M printed s = some (s.setReg "i" .int [] (fun _ => -2)) := by
  simp [printed, FP.Structural.exec, run, step, FP.Structural.evalExpr,
    FP.Structural.evalOp]

#guard_msgs (drop info) in
#auditModuleAxioms
end StaticPrintExecutionTests

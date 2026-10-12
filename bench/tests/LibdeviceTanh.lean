import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.LibdeviceTanh
open VeriTile.Triton

def direct : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = libdevice.tanh(x)
}
def qualified : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = tl.extra.cuda.libdevice.tanh(x)
}
def aliasSource : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = triton_tanh(x)
}
def intrinsic : ComputeKernel := triton {
  x = (1.0).to(tl.float32)
  y = tl.tanh(x)
}
example : direct = qualified := rfl
example : direct = aliasSource := rfl
example : direct.surfaceBody ≠ intrinsic.surfaceBody := by
  intro h
  simp only [direct, intrinsic, ComputeKernel.surfaceBody, List.cons.injEq] at h
  cases h.2.1

/-- FP execution calls the distinct, unconstrained libdevice backend. -/
theorem backend {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Option ComputeDType) (s : FP.Structural.State α) (a : Op .real shape) :
    FP.Structural.evalOp M p (.libdeviceTanh a) s =
      ((FP.Structural.evalOp M p a s).map (fun v => M.unary p .libdeviceTanh ∘ v)) := by
  simp only [FP.Structural.evalOp, Option.map_eq_bind]
  rfl

example (a : Op .real shape) :
    (Op.libdeviceTanh a).eraseDType = Op.libdeviceTanh a.eraseDType := by rw [Op.eraseDType.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LibdeviceTanh

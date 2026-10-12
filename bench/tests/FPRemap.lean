import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Index-only rearrangement must preserve opaque values and operand failure.
The frontend uses this node when broadcasting a singleton mask dimension. -/
namespace FPRemapTests
open VeriTile.Triton FP.Structural

variable {α : Type} [Inhabited α]

/-- Repeated lanes retain their fp16 values; remapping introduces no cast or
floating arithmetic, whatever operations the backend supplies. -/
theorem broadcast_preserves_fp16 (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (v : Values α .fp16 [1, 3])
    (h : s.regs .fp16 [1, 3] "x" = some v) :
    FP.Structural.evalOp M p
      (.remap [2, 3] (fun i => (0, i.2)) (.ref .fp16 [1, 3] "x")) s =
      some (fun i => v (0, i.2)) := by
  simp [evalOp_unfold, h]

/-- The nested expression retains exactly the caller's compute precision. -/
theorem nested_precision_is_preserved (M : Algebra α) (p : Option ComputeDType)
    (s : State α) (v : Values α .real [1, 3])
    (h : s.regs .real [1, 3] "x" = some v) :
    FP.Structural.evalOp M p
      (.remap [2, 3] (fun i => (0, i.2)) (.exp (.ref .real [1, 3] "x"))) s =
      some (fun i => M.unary p .exp (v (0, i.2))) := by
  simp [evalOp_unfold, h]

theorem missing_mask_still_fails (M : Algebra α) (s : State α)
    (h : s.regs .bool [1, 3] "mask" = none) :
    FP.Structural.evalOp M none
      (.remap [2, 3] (fun i => (0, i.2)) (.ref .bool [1, 3] "mask")) s = none := by
  simp [evalOp_unfold, h]

/-- An empty output does not bypass evaluation of a missing source operand. -/
theorem empty_output_still_evaluates_operand (M : Algebra α) (s : State α)
    (h : s.regs .bool [1] "mask" = none) :
    FP.Structural.evalOp M none
      (.remap [0] (fun _ => (0, PUnit.unit)) (.ref .bool [1] "mask")) s = none := by
  simp [evalOp_unfold, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end FPRemapTests

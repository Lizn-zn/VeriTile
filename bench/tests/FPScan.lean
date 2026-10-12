import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.StatementAudit

/-! A distinguishing opaque backend checks the scan operation, axis, direction,
precision and payload. This is an interpreter regression, not a scan identity. -/
namespace FPScanTests
open VeriTile.Triton FP.Structural
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private def model : Algebra Nat where
  literal := fun _ _ _ => 7
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0
  scan := fun p {_} op axis direction v => some (fun i =>
    v i + axis.val * 10 + (match direction with | .reverse => 100 | .forward => 0) +
      (match op with | .sum => 1000 | .prod => 2000 | .max => 3000 | .min => 4000) +
      (if p = some .fp64 then 10000 else 0))

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def input : Op .real [2,3] := .full [2,3] (.const 7)

theorem preserves_kind_axis_direction_and_payload :
    FP.Structural.evalOp model none (.scan .prod ⟨1, by decide⟩ .reverse input) initial =
      some (fun _ => 2117) := by
  simp only [FP.Structural.evalOp, model, input, initial, Algebra.withDefaultPrecision, resolvePrecision]
  rfl

theorem default_precision_reaches_scan :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) none
      (.scan .sum ⟨0, by decide⟩ .forward input) initial = some (fun _ => 11007) := by
  simp only [FP.Structural.evalOp, model, input, initial, Algebra.withDefaultPrecision, resolvePrecision]
  rfl

theorem explicit_precision_wins :
    FP.Structural.evalOp (model.withDefaultPrecision .fp64) (some .fp32)
      (.scan .sum ⟨0, by decide⟩ .forward input) initial = some (fun _ => 1007) := by
  simp only [FP.Structural.evalOp, model, input, initial, Algebra.withDefaultPrecision, resolvePrecision]
  rfl

theorem unsupported_scan_fails :
    FP.Structural.evalOp { model with scan := fun _ {_} _ _ _ _ => none } none
      (.scan .sum ⟨0, by decide⟩ .forward input) initial = none := by
  simp only [FP.Structural.evalOp, model, input, initial, Algebra.withDefaultPrecision, resolvePrecision]
  rfl

theorem missing_input_fails :
    FP.Structural.evalOp model none
      (.scan .sum ⟨0, by decide⟩ .forward (.ref .real [2,3] "missing")) initial = none := by
  simp only [FP.Structural.evalOp, model, input, initial, Algebra.withDefaultPrecision, resolvePrecision]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPScanTests

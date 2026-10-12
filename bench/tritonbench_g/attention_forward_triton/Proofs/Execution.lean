import bench.tritonbench_g.attention_forward_triton.Proofs.LegacyRounding
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! A successful full-source FP execution in a total abstract backend. This
checks interpreter coverage, including pointer division/remainder, the loop,
both dot products, explicit casts and the final store. It does not establish a
numerical law or claim to emulate a GPU. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.Execution
open VeriTile.Triton FP.Structural

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fp32Bits := fun _ => 0
  fp32Load := fun _ => 0
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 0)
  dotWithOptions := fun _ _ {_} {_} {_} {_} _ _ => some (fun _ => 0)
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out 0)

private def initial : State Nat where
  mem := fun _ _ => .mk .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

set_option maxHeartbeats 1600000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

/-- Positive context and a nonempty loop, with the source's fixed 128/96 head
layout. In particular, this does not succeed merely by skipping the loop. -/
theorem full_kernel_executes : ∃ t, FP.Structural.exec model
    (attention_forward_triton_surface "Q" "K" "V" "QS" "KS" "Out"
      128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
      1 1 1 128 1 2 128 96 1) initial = some t := by
  simp (config := { maxSteps := 200000 }) [attention_forward_triton_surface, FP.Structural.exec, run, step,
    FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    model, initial, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, natLt, bop, FP.Structural.store, State.setReg,
    TileShape.axisDim]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.Execution

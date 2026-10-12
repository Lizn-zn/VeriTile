import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! The resume arm reads the fp16 output buffer before its fp32 conversion.
The nonzero diagnostic backend makes both a wrong storage channel and a
missing conversion observable. It is not a GPU numerical experiment. -/
namespace FPAttentionResumeStorageTests
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option linter.unusedSimpArgs false

private def source (candidate : Bool) :=
  (if candidate then TypedSource.reciprocal else TypedSource.original)
    "Q" "K" "V" "M" "O" "L" 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0

private def resume (candidate : Bool) :=
  ((Source.conditional? (source candidate).surfaceBody (fun _ => Bool.true)).getD
    ⟨[], .constBool Bool.false, [], [], []⟩).no

private def assignment (candidate : Bool) :=
  ((resume candidate).drop (Source.prefixBefore (resume candidate) "acc").length).head?

private def load : ComputeStmt :=
  .assign .real [1, 1] "acc" (.compute (.alg .fp32
    (.castFloat .fp16 .real
      (.load .fp16 (.blockPtr (.ref .blockPtr [1, 1] "O_block_ptr") []) .none))))

theorem resume_storage_precedes_cast (candidate : Bool) :
    assignment candidate = some load := by
  cases candidate <;> rfl

private def pointer : BlockPtr where
  region := "O"
  baseOffset := 0
  parentShape := [1, 1]
  blockShape := [1, 1]
  strides := [1, 1]
  offsets := [0, 0]

private def initial : State ℝ :=
  ({ mem := fun _ _ => .mk .fp16 5
     regs := fun _ _ _ => none
     pids := fun _ => 0
     numPids := fun _ => 1
     undef := fun d _ _ => defaultValue d } : State ℝ).setReg
      "O_block_ptr" .blockPtr [1, 1] (fun _ => pointer)

private def diagnostic : Algebra ℝ :=
  { FPModel.model with cast := fun _ _ _ x => x + 1 }

theorem resume_uses_stored_value_and_cast (candidate : Bool) :
    ((assignment candidate).bind (fun stmt => step (engine diagnostic) stmt initial)).bind
      (fun s => (s.regs .real [1, 1] "acc").map
        (fun v => v (0, 0, PUnit.unit))) = some 6 := by
  rw [resume_storage_precedes_cast]
  simp [load, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    engine, Algebra.withDefaultPrecision, resolvePrecision,
    diagnostic, FPModel.model, initial, pointer, State.setReg,
    TileShape.indexToList, BlockPtr.address, Cell.read, ofFloat, toFloat]
  norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end FPAttentionResumeStorageTests

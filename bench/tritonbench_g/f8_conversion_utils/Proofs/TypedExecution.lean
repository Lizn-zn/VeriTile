import bench.tritonbench_g.f8_conversion_utils.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000

def formatType : Format → FloatDType
  | .e4 => .f8e4
  | .e5 => .f8e5

def inputType : Input → FloatDType
  | .fp16 => .fp16
  | .fp32 => .real

def initial (dtype : TileDType) (value : Value ℝ dtype) : State ℝ where
  mem := fun r _ => if r="x" then .mk dtype value else .mk .int 9
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- A typed fp8 load and both fp16 stores execute from empty registers. This
is an inhabited abstract model, without a physical rounding assertion. -/
def toHalfExecution (format : Format) :
    {v : State ℝ // FP.Structural.exec (engine model)
      (toHalf format Bool.false "y" "x" 1 2)
      (initial (formatType format).toTileDType (ofFloat (α:=ℝ) (formatType format) (-3)))=some v} := by
  cases format
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [toHalf,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,evalExpr,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      numeric,natLt,bop,initial,formatType,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
    try rfl

def toFP8Execution (input : Input) (format : Format) :
    {v : State ℝ // FP.Structural.exec (engine model)
      (toFP8 input format Bool.false "y" "x" 1 2)
      (initial (inputType input).toTileDType (ofFloat (α:=ℝ) (inputType input) (-3)))=some v} := by
  cases input <;> cases format
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [toFP8,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,bop,initial,inputType,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
    try rfl

theorem to_half_candidate_executes (format : Format) :
    FP.Structural.exec (engine model) (toHalf format Bool.true "y" "x" 1 2)
      (initial (formatType format).toTileDType (ofFloat (α:=ℝ) (formatType format) (-3)))=
      some (toHalfExecution format).val := by
  have h := FP.ProfiledRewrite.sound (FPProof.to_half_equivalent format "y" "x" 1 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial (formatType format).toTileDType (ofFloat (α:=ℝ) (formatType format) (-3)))
    (by simp [FPProof.toHalfProgram])
  exact h.symm.trans (toHalfExecution format).property

theorem to_fp8_candidate_executes (input : Input) (format : Format) :
    FP.Structural.exec (engine model) (toFP8 input format Bool.true "y" "x" 1 2)
      (initial (inputType input).toTileDType (ofFloat (α:=ℝ) (inputType input) (-3)))=
      some (toFP8Execution input format).val := by
  have h := FP.ProfiledRewrite.sound (FPProof.to_fp8_equivalent input format "y" "x" 1 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial (inputType input).toTileDType (ofFloat (α:=ℝ) (inputType input) (-3)))
    (by simp [FPProof.toFP8Program])
  exact h.symm.trans (toFP8Execution input format).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedExecution

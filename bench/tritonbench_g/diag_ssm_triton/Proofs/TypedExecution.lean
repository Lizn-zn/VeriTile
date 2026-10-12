import bench.tritonbench_g.diag_ssm_triton.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def stored (storage : Storage) (x : ℝ) : Cell ℝ :=
  match storage with
  | .fp32 => .mk .real x
  | .fp16 => .mk .fp16 x
  | .bf16 => .mk .bf16 x

def initial (storage : Storage) : State ℝ where
  mem := fun _ _ => stored storage (-3)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Two recurrence steps and a masked tail; backward paths execute both
previous-state branches. This witnesses the abstract typed semantics, not
physical GPU rounding. -/
def forward_execution (storage : Storage) :
    {t : State ℝ // FP.Structural.exec (engine model)
      (TypedSource.forward storage Bool.false "s_ptr" "x_ptr" "lambda_ptr" "y_ptr" 2 1 3 4) (initial storage) = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    dsimp only [TypedSource.forward, ComputeKernel.surfaceBody, FP.Structural.exec]
    simp [run,step,loop_done,loop_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
      TileShape.dropInsertedIndex]
    try rfl

/-- Two recurrence steps and a masked tail; backward paths execute both
previous-state branches. This witnesses the abstract typed semantics, not
physical GPU rounding. -/
def backward_execution (storage : Storage) :
    {t : State ℝ // FP.Structural.exec (engine model)
      (TypedSource.backward storage Bool.false "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4) (initial storage) = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    dsimp only [TypedSource.backward, ComputeKernel.surfaceBody, FP.Structural.exec]
    simp [run,step,loop_done,loop_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
      TileShape.dropInsertedIndex]
    try rfl

/-- Two recurrence steps and a masked tail; backward paths execute both
previous-state branches. This witnesses the abstract typed semantics, not
physical GPU rounding. -/
def complexForward_execution  :
    {t : State ℝ // FP.Structural.exec (engine model)
      (TypedSource.complexForward Bool.false "s_ptr" "x_ptr" "y_ptr" "lambda_ptr" 2 1 3 4) (initial .fp32) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [TypedSource.complexForward, ComputeKernel.surfaceBody, FP.Structural.exec]
  simp [run,step,loop_done,loop_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
      TileShape.dropInsertedIndex]
  try rfl

/-- Two recurrence steps and a masked tail; backward paths execute both
previous-state branches. This witnesses the abstract typed semantics, not
physical GPU rounding. -/
def complexBackward_execution  :
    {t : State ℝ // FP.Structural.exec (engine model)
      (TypedSource.complexBackward Bool.false "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4) (initial .fp32) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  dsimp only [TypedSource.complexBackward, ComputeKernel.surfaceBody, FP.Structural.exec]
  simp [run,step,loop_done,loop_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
      TileShape.dropInsertedIndex]
  try rfl

theorem forward_candidate_executes (storage : Storage) :
    FP.Structural.exec (engine model)
      (TypedSource.forward storage Bool.true "s_ptr" "x_ptr" "lambda_ptr" "y_ptr" 2 1 3 4) (initial storage) =
        some (forward_execution storage).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent storage "s_ptr" "x_ptr" "lambda_ptr" "y_ptr" 2 1 3 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage) (by simp [FPProof.forwardProgram])
  exact h.symm.trans (forward_execution storage).property

theorem backward_candidate_executes (storage : Storage) :
    FP.Structural.exec (engine model)
      (TypedSource.backward storage Bool.true "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4) (initial storage) =
        some (backward_execution storage).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent storage "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage) (by simp [FPProof.backwardProgram])
  exact h.symm.trans (backward_execution storage).property

theorem complexForward_candidate_executes  :
    FP.Structural.exec (engine model)
      (TypedSource.complexForward Bool.true "s_ptr" "x_ptr" "y_ptr" "lambda_ptr" 2 1 3 4) (initial .fp32) =
        some (complexForward_execution ).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.complexForward_equivalent "s_ptr" "x_ptr" "y_ptr" "lambda_ptr" 2 1 3 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial .fp32) (by simp [FPProof.complexForwardProgram])
  exact h.symm.trans (complexForward_execution ).property

theorem complexBackward_candidate_executes  :
    FP.Structural.exec (engine model)
      (TypedSource.complexBackward Bool.true "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4) (initial .fp32) =
        some (complexBackward_execution ).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.complexBackward_equivalent "s_ptr" "lambda_ptr" "y_ptr" "grad_s_ptr" "grad_x_ptr" "grad_lambda_ptr" "grad_y_ptr" 2 1 3 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial .fp32) (by simp [FPProof.complexBackwardProgram])
  exact h.symm.trans (complexBackward_execution ).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedExecution

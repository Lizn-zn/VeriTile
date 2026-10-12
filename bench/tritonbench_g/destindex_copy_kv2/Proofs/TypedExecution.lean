import bench.tritonbench_g.destindex_copy_kv2.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def stored (storage : Storage) (x : ℝ) : Cell ℝ :=
  match storage with
  | .fp32 => .mk .real x
  | .fp16 => .mk .fp16 x
  | .bf16 => .mk .bf16 x

def initial (input : Storage) : State ℝ where
  mem := fun r _ => if r="dest" then .mk .int 2 else stored input (-3)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original (input output : Storage) :=
  kernel input output Bool.false "input" ⟨"dest"⟩ "output" 16 4 1 16 4 1 3 4 4

/-- Complete typed execution with head padding masked and all columns active, signed metadata
and independent input/output storage. This witnesses the abstract semantics,
not a physical GPU rounding measurement. -/
def execution (input output : Storage) :
    {t : State ℝ // FP.Structural.exec (engine model) (original input output) (initial input) = some t} := by
  cases input <;> cases output
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [original,kernel,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,bop,initial,stored,ofFloat,toFloat,Region.cast,
      FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write,
      TileShape.dropInsertedIndex]
    try rfl

theorem candidate_executes (input output : Storage) :
    FP.Structural.exec (engine model)
      (kernel input output Bool.true "input" ⟨"dest"⟩ "output" 16 4 1 16 4 1 3 4 4) (initial input) =
      some (execution input output).val := by
  have h := FP.ProfiledRewrite.sound (FPProof.equivalent input output
      "input" ⟨"dest"⟩ "output" 16 4 1 16 4 1 3 4 4)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial input) (by simp [FPProof.program])
  exact h.symm.trans (execution input output).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedExecution

import bench.tritonbench_g.destindex_copy.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.TypedExecution
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

def initial (nopeType ropeType : Storage) : State ℝ where
  mem := fun r _ => if r="dest" then .mk .int 2
    else if r="nope" then stored nopeType (-3) else stored ropeType 5
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def original (nopeType ropeType : Storage) :=
  kernel nopeType ropeType Bool.false "nope" "rope" ⟨"dest"⟩ "outNope" "outRope"
    4 7 1 2 9 1 4 7 1 2 9 1 2 3 4 2

/-- A complete typed execution from empty registers for every independent
storage pair, with a nonzero source row and a different signed destination row.
The consistent abstract backend witnesses execution, not physical GPU rounding. -/
def execution (nopeType ropeType : Storage) :
    {t : State ℝ // FP.Structural.exec (engine model) (original nopeType ropeType)
      (initial nopeType ropeType) = some t} := by
  cases nopeType <;> cases ropeType
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

theorem candidate_executes (nopeType ropeType : Storage) :
    FP.Structural.exec (engine model)
      (kernel nopeType ropeType Bool.true "nope" "rope" ⟨"dest"⟩ "outNope" "outRope"
        4 7 1 2 9 1 4 7 1 2 9 1 2 3 4 2) (initial nopeType ropeType) =
      some (execution nopeType ropeType).val := by
  have h := FP.ProfiledRewrite.sound (FPProof.equivalent nopeType ropeType
      "nope" "rope" ⟨"dest"⟩ "outNope" "outRope" 4 7 1 2 9 1 4 7 1 2 9 1 2 3 4 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial nopeType ropeType) (by simp [FPProof.program])
  exact h.symm.trans (execution nopeType ropeType).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.TypedExecution

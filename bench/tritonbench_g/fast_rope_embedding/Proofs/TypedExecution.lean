import bench.tritonbench_g.fast_rope_embedding.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input cosine sine : Storage) : State ℝ where
  mem := fun r _ => if r="q" then .mk (storageType input).toTileDType (ofFloat (storageType input) 2)
    else if r="c" then .mk (storageType cosine).toTileDType (ofFloat (storageType cosine) 3)
    else .mk (storageType sine).toTileDType (ofFloat (storageType sine) 1)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- A nonempty partial head group runs its final head from empty registers, with an
odd head size and a padded lane. All storage triples and both directions are
covered by this abstract execution witness, without a hardware-rounding claim. -/
def execution (input cosine sine : Storage) (backwards : Bool) :
    {v : State ℝ // FP.Structural.exec (engine model)
      (rope input cosine sine Bool.false "q" "c" "s" 35 3 3 2 7 5 4 backwards)
      (initial input cosine sine)=some v} := by
  cases input <;> cases cosine <;> cases sine <;> cases backwards
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp only [rope,ComputeKernel.surfaceBody,FP.Structural.exec]
    simp [run,step,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.allIndices,List.finRange_succ,State.write,TileShape.dropInsertedIndex]
    try rfl

theorem candidate_executes (input cosine sine : Storage) (backwards : Bool) :
    FP.Structural.exec (engine model)
      (rope input cosine sine Bool.true "q" "c" "s" 35 3 3 2 7 5 4 backwards)
      (initial input cosine sine)=some (execution input cosine sine backwards).val := by
  have h := FP.ProfiledRewrite.sound
    (FPProof.equivalent input cosine sine "q" "c" "s" 35 3 3 2 7 5 4 backwards)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial input cosine sine) (by simp [FPProof.program])
  exact h.symm.trans (execution input cosine sine backwards).property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedExecution

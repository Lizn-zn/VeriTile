import bench.tritonbench_g.flash_attn.Proofs.NormalizedSource
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def before (Q K V L O : RegionName) (p : Layout) : List Stmt :=
  (algorithm Q K V L O p).body.take 16

def entry (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    {t : BlockState // stepStmts (before Q K V L O p) s=some t} := by
  cases h : p.causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [before,algorithm,reduced,List.take_succ_cons,List.take_zero]
    simp (maxSteps := 300000) [stepStmts,stepStmt,evalOp,
      Option.bind,BlockState.setReg,Region.cast,h]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution

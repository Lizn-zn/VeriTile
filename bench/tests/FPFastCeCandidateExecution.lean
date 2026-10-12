import bench.tritonbench_g.fast_ce_loss.Proofs.TypedExecution
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 15000

theorem forward_candidate_executes (storage : Storage) (cap scale ignore : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (forward storage cap scale Bool.true "logits" "loss" "lse" ⟨"label"⟩ 3 4 4 1 2)
      (initial storage ignore Bool.false)=some v := by
  obtain ⟨v,hv⟩ := forward_executes storage cap scale ignore
  have h := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent storage cap scale "logits" "loss" "lse" ⟨"label"⟩ 3 4 4 1 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage ignore Bool.false) (by simp [FPProof.forwardProgram])
  exact ⟨v,h.symm.trans hv⟩

theorem chunked_candidate_executes (storage : Storage) (cap scale ignore later : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (chunked storage cap scale Bool.true "logits" "loss" "lse" ⟨"label"⟩ 3 2 4 2 1 2)
      (initial storage ignore later)=some v := by
  obtain ⟨v,hv⟩ := chunked_executes storage cap scale ignore later
  have h := FP.ProfiledRewrite.sound
    (FPProof.chunked_equivalent storage cap scale "logits" "loss" "lse" ⟨"label"⟩ 3 4 2 1 2 2)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage ignore later) (by simp [FPProof.chunkedProgram])
  exact ⟨v,h.symm.trans hv⟩

theorem backward_candidate_executes (storage : Storage) (cap scale ignore later : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (backward storage cap scale Bool.true "logits" "dloss" "lse" ⟨"label"⟩ 3 4 1 2 1 2)
      (initial storage ignore later)=some v := by
  obtain ⟨v,hv⟩ := backward_executes storage cap scale ignore later
  have h := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent storage cap scale "logits" "dloss" "lse" ⟨"label"⟩ 3 4 2 1 2 1)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage ignore later) (by simp [FPProof.backwardProgram])
  exact ⟨v,h.symm.trans hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.TypedExecution

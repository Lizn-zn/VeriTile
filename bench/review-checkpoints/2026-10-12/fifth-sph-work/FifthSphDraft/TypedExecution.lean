import FifthSphDraft.TypedForwardExecution
import FifthSphDraft.TypedBackwardExecution
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
theorem forward_candidate_executes (input destination : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      (forward input destination Bool.true "x" "y" 2 9 36 1 12) (initial input destination .fp32)=some v := by
  obtain ⟨v,hv⟩ := forward_executes input destination
  have h := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent input destination "x" "y" 2 9 36 1 12)
    PUnit model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial input destination .fp32) (by simp [FPProof.forwardProgram])
  exact ⟨v,h.symm.trans hv⟩


theorem backward_candidate_executes (input destination upstream : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      (backward input destination upstream Bool.true "x" "dx" "g" 2 9 36 1 12) (initial input destination upstream)=some v := by
  obtain ⟨v,hv⟩ := backward_executes input destination upstream
  have h := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent input destination upstream "x" "dx" "g" 2 9 36 1 12)
    PUnit model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial input destination upstream) (by simp [FPProof.backwardProgram])
  exact ⟨v,h.symm.trans hv⟩


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution

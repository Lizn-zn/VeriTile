import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFormula
import VeriTile.Meta.Disprove

/-! A checked distinction between the actual finite-penalty batch formula and
an exact causal mask. Both raw scores are zero; the past/current value is 0
and the future value is 1. The real result is strictly positive. This file is
an arithmetic refutation, not yet a refutation of a kernel IO specification:
the complete source-to-formula execution bridge is a separate obligation. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.FinitePenalty
open VeriTile.Triton
noncomputable section

def inputs : RealFormula.Inputs 2 2 1 where
  query := fun _ _ => 0
  key := fun _ _ _ => 0
  value := fun _ key _ => key.val
  queryScale := 1
  keyScale := fun _ => 1

theorem formula_value : RealFormula.expected inputs 0 2 1 3 0 0 =
    pow2 (-1000000) / (1 + pow2 (-1000000)) := by
  norm_num [RealFormula.expected, RealFormula.blocks, RealFormula.allVisits,
    RealFormula.firstVisits, RealFormula.secondVisits, RealFormula.visits,
    RealFormula.tripCount, RealFormula.blockPairs, List.finRange_succ,
    RealFormula.logit, RealFormula.score, RealFormula.penalty, RealFormula.loadedValue,
    inputs, Fin.sum_univ_succ]

theorem formula_positive : 0 < RealFormula.expected inputs 0 2 1 3 0 0 := by
  rw [formula_value]
  exact div_pos (pow2_pos _) (by have h := pow2_pos (-1000000); linarith)

/-- Exact masking would retain only key 0, whose value is zero. -/
refutation finite_penalty_is_exact_mask : RealFormula.expected inputs 0 2 1 3 0 0 = 0 := by
  disprove [ne_of_gt formula_positive]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.FinitePenalty

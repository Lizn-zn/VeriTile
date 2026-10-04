/- FP equivalence for the original mHC depth rank-one, zero-iteration case.
The final residual addition is commuted using the accepted fp32 atomic rule.
The exp weighting and all memory accesses are unchanged. -/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ReportedAdmission

namespace VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv

open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := FP.ReportedAdmission.fp32_add_commute

def originalKernel (tau : ℝ) : ComputeKernel :=
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchOutReg : Region .fp32 := ⟨"branch_out"⟩
  let hPostReg : Region .fp32 := ⟨"h_post"⟩
  let outReg : Region .fp32 := ⟨"out"⟩
  triton {
  b := tl.program_id(0)
  res_mix := tl.load(resMixReg + b)
  branch_out := tl.load(branchOutReg + b)
  h_post := tl.load(hPostReg)
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := res_mix + branch_mix
  tl.store(outReg + b, out)
}

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchOutReg : Region .fp32 := ⟨"branch_out"⟩
  let hPostReg : Region .fp32 := ⟨"h_post"⟩
  let outReg : Region .fp32 := ⟨"out"⟩
  triton {
  b := tl.program_id(0)
  res_mix := tl.load(resMixReg + b)
  branch_out := tl.load(branchOutReg + b)
  h_post := tl.load(hPostReg)
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := branch_mix + res_mix
  tl.store(outReg + b, out)
}

abbrev body := Spec.ProgramSyntax.body (Program := ComputeKernel)

def addFragment (lhs rhs : RegName) : List ComputeStmt :=
  [.assign .real [] "out"
    (.compute (.alg .fp32 (.add .real .nil
      (.ref .real [] lhs) (.ref .real [] rhs))))]

def originalAdd := addFragment "res_mix" "branch_mix"
def optimizedAdd := addFragment "branch_mix" "res_mix"
def beforeAdd (tau : ℝ) := (body (originalKernel tau)).take 5
def afterAdd (tau : ℝ) := (body (originalKernel tau)).drop 6

theorem original_decomposition (tau : ℝ) :
    body (originalKernel tau) = beforeAdd tau ++ originalAdd ++ afterAdd tau := rfl

theorem optimized_decomposition (tau : ℝ) :
    body (optimizedKernel tau) = beforeAdd tau ++ optimizedAdd ++ afterAdd tau := rfl

def addCommute : Spec.RuleEntry ComputeStmt := admitted.bind originalAdd optimizedAdd

structure Rules where
  add_comm : Spec.EvidenceValidated addCommute.rule addCommute.evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions ComputeStmt := [addCommute]
instance : CoeOut Rules (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

@[spec_rule] theorem admitted_add_commute (R : Rules) :
    Spec.Derivation R.assumptions originalAdd optimizedAdd :=
  .atom addCommute (by simp [Rules.assumptions])
    (admitted.admit originalAdd optimizedAdd R.add_comm)

specification mhc_depth_equiv (tau : ℝ) (R : Rules) :
    originalKernel tau ≡[R] optimizedKernel tau := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body (originalKernel tau)) (body (optimizedKernel tau))
  rw [original_decomposition, optimized_decomposition]
  exact .frame (beforeAdd tau) (afterAdd tau) (admitted_add_commute R)

#print_fp_assumptions mhc_depth_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv

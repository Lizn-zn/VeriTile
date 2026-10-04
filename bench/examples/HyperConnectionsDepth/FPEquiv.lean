import bench.examples.HyperConnectionsDepth.Kernels
/- FP equivalence for the original mHC depth rank-one, zero-iteration case.
The final residual addition is commuted using the accepted fp32 atomic rule.
The exp weighting and all memory accesses are unchanged. -/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := (FP.ScalarArithmetic.report .addCommute (by decide)).report

def addFragment (lhs rhs : RegName) : List ComputeStmt :=
  [.assign .real [] "out"
    (.compute (.alg .fp32 (.add .real .nil
      (.ref .real [] lhs) (.ref .real [] rhs))))]

def originalAdd := addFragment "res_mix" "branch_mix"
def optimizedAdd := addFragment "branch_mix" "res_mix"
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
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions mhc_depth_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsDepthFPEquiv

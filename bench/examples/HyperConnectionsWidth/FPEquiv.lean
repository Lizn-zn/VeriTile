import bench.examples.HyperConnectionsWidth.Kernels
/- FP equivalence for the original mHC width rank-one, zero-iteration case.
Two scalar products commute independently; their exp operands and all memory
accesses stay unchanged. This exercises composition of atomic rewrites. -/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := (FP.ScalarArithmetic.report .mulCommute (by decide)).report

/-- One multiplication, with the weight expression treated as an operand. -/
def mulFragment (tau : ℝ) (out logit : RegName) (swapped : Bool) : List ComputeStmt :=
  let weight : Op .real [] := .exp (.div .real .nil (.ref .real [] logit) (.const tau))
  let residual : Op .real [] := .ref .real [] "residual"
  [.assign .real [] out (.compute (.alg .fp32
    (if swapped then .mul .real .nil residual weight else .mul .real .nil weight residual)))]

def resCommute (tau : ℝ) := admitted.bind
  (mulFragment tau "res_mix" "h_res" Bool.false) (mulFragment tau "res_mix" "h_res" Bool.true)
def preCommute (tau : ℝ) := admitted.bind
  (mulFragment tau "branch_in" "h_pre" Bool.false) (mulFragment tau "branch_in" "h_pre" Bool.true)

structure Rules (tau : ℝ) where
  res_mul_comm : Spec.EvidenceValidated (resCommute tau).rule (resCommute tau).evidence
  pre_mul_comm : Spec.EvidenceValidated (preCommute tau).rule (preCommute tau).evidence

def Rules.assumptions {tau : ℝ} (_ : Rules tau) : Spec.Assumptions ComputeStmt :=
  [resCommute tau, preCommute tau]
instance {tau : ℝ} : CoeOut (Rules tau) (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

@[spec_rule] theorem admitted_res_commute {tau : ℝ} (R : Rules tau) :
    Spec.Derivation R.assumptions
      (mulFragment tau "res_mix" "h_res" Bool.false) (mulFragment tau "res_mix" "h_res" Bool.true) :=
  .atom (resCommute tau) (by simp [Rules.assumptions])
    (admitted.admit _ _ R.res_mul_comm)

@[spec_rule] theorem admitted_pre_commute {tau : ℝ} (R : Rules tau) :
    Spec.Derivation R.assumptions
      (mulFragment tau "branch_in" "h_pre" Bool.false) (mulFragment tau "branch_in" "h_pre" Bool.true) :=
  .atom (preCommute tau) (by simp [Rules.assumptions])
    (admitted.admit _ _ R.pre_mul_comm)

specification mhc_width_equiv (tau : ℝ) (R : Rules tau) :
    originalKernel tau ≡[R] optimizedKernel tau := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions mhc_width_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv

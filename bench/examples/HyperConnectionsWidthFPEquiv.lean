/- FP equivalence for the original mHC width rank-one, zero-iteration case.
Two scalar products commute independently; their exp operands and all memory
accesses stay unchanged. This exercises composition of atomic rewrites. -/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ReportedAdmission

namespace VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv

open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := FP.ReportedAdmission.fp32_mul_commute

def originalKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := tl.exp(h_res / $(tau)) * residual
  h_pre := tl.load(hPreReg)
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

def middleKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := residual * tl.exp(h_res / $(tau))
  h_pre := tl.load(hPreReg)
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := residual * tl.exp(h_res / $(tau))
  h_pre := tl.load(hPreReg)
  branch_in := residual * tl.exp(h_pre / $(tau))
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

abbrev body := Spec.ProgramSyntax.body (Program := ComputeKernel)

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

theorem original_decomposition (tau : ℝ) :
    body (originalKernel tau) = (body (originalKernel tau)).take 3 ++
      mulFragment tau "res_mix" "h_res" Bool.false ++ (body (originalKernel tau)).drop 4 := rfl

theorem middle_decomposition_res (tau : ℝ) :
    body (middleKernel tau) = (body (originalKernel tau)).take 3 ++
      mulFragment tau "res_mix" "h_res" Bool.true ++ (body (originalKernel tau)).drop 4 := rfl

theorem middle_decomposition_pre (tau : ℝ) :
    body (middleKernel tau) = (body (middleKernel tau)).take 5 ++
      mulFragment tau "branch_in" "h_pre" Bool.false ++ (body (middleKernel tau)).drop 6 := rfl

theorem optimized_decomposition (tau : ℝ) :
    body (optimizedKernel tau) = (body (middleKernel tau)).take 5 ++
      mulFragment tau "branch_in" "h_pre" Bool.true ++ (body (middleKernel tau)).drop 6 := rfl

specification mhc_width_equiv (tau : ℝ) (R : Rules tau) :
    originalKernel tau ≡[R] optimizedKernel tau := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body (originalKernel tau)) (body (optimizedKernel tau))
  apply Spec.Derivation.trans (middle := body (middleKernel tau))
  · rw [original_decomposition, middle_decomposition_res]
    exact .frame _ _ (admitted_res_commute R)
  · rw [middle_decomposition_pre, optimized_decomposition]
    exact .frame _ _ (admitted_pre_commute R)

#print_fp_assumptions mhc_width_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.HyperConnectionsWidthFPEquiv

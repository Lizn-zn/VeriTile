import bench.examples.AdamUpdateGridLaunch.Kernels
/- FP equivalence of the Adam-named Lion update. Only the final momentum
addition is commuted; masked in-place stores and the parameter update remain
as in the original kernel. This is a per-program equivalence. -/
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Float.ScalarArithmetic

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv
open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels
open VeriTile Triton
open scoped VeriTile.Spec

abbrev admitted := (FP.ScalarArithmetic.report .addCommute (by decide)).report


abbrev body := Spec.ProgramSyntax.body (Program := ComputeKernel)

/-- One addition; the momentum product is the unchanged left operand. -/
def addFragment (beta2 : ℝ) (B : Nat) (swapped : Bool) : List ComputeStmt :=
  let product : Op .real [B] :=
    .mul .real .scalarR (.ref .real [B] "diff") (.const beta2)
  let gradient : Op .real [B] := .ref .real [B] "grad"
  [.assign .real [B] "exp_avg" (.compute (.alg .fp32
    (if swapped then .add .real (.consSame .nil) gradient product
     else .add .real (.consSame .nil) product gradient)))]

def addCommute (beta2 : ℝ) (B : Nat) := admitted.bind
  (addFragment beta2 B Bool.false) (addFragment beta2 B Bool.true)

structure Rules (beta2 : ℝ) (B : Nat) where
  add_comm : Spec.EvidenceValidated (addCommute beta2 B).rule (addCommute beta2 B).evidence

def Rules.assumptions {beta2 : ℝ} {B : Nat} (_ : Rules beta2 B) :
    Spec.Assumptions ComputeStmt := [addCommute beta2 B]
instance {beta2 : ℝ} {B : Nat} :
    CoeOut (Rules beta2 B) (Spec.Assumptions (Spec.ProgramSyntax.Statement ComputeKernel)) :=
  ⟨Rules.assumptions⟩

@[spec_rule] theorem admitted_add_commute {beta2 : ℝ} {B : Nat} (R : Rules beta2 B) :
    Spec.Derivation R.assumptions
      (addFragment beta2 B Bool.false) (addFragment beta2 B Bool.true) :=
  .atom (addCommute beta2 B) (by simp [Rules.assumptions])
    (admitted.admit _ _ R.add_comm)

def beforeAdd (lr wd beta1 beta2 : ℝ) (n B : Nat) :=
  (body (originalKernel lr wd beta1 beta2 n B)).take 16

def afterAdd (lr wd beta1 beta2 : ℝ) (n B : Nat) :=
  (body (originalKernel lr wd beta1 beta2 n B)).drop 17

theorem original_decomposition (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    body (originalKernel lr wd beta1 beta2 n B) =
      beforeAdd lr wd beta1 beta2 n B ++ addFragment beta2 B Bool.false ++
      afterAdd lr wd beta1 beta2 n B := rfl

theorem optimized_decomposition (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    body (optimizedKernel lr wd beta1 beta2 n B) =
      beforeAdd lr wd beta1 beta2 n B ++ addFragment beta2 B Bool.true ++
      afterAdd lr wd beta1 beta2 n B := rfl

specification adam_update_equiv (lr wd beta1 beta2 : ℝ) (n B : Nat) (R : Rules beta2 B) :
    originalKernel lr wd beta1 beta2 n B ≡[R] optimizedKernel lr wd beta1 beta2 n B := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions
    (body (originalKernel lr wd beta1 beta2 n B)) (body (optimizedKernel lr wd beta1 beta2 n B))
  rw [original_decomposition, optimized_decomposition]
  exact .frame _ _ (admitted_add_commute R)

#print_fp_assumptions adam_update_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.AdamUpdateGridLaunchFPEquiv

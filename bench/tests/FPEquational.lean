/- Scalar-assumption and reduction-tree boundaries, independent of GPU runs. -/
import VeriTile.Triton.Float.Equational
import VeriTile.Triton.Float.ReportedAdmission
import VeriTile.Meta.StatementAudit

namespace FPEquationalTests
open VeriTile Triton FP.Equational

def commute := FP.ReportedAdmission.fp32_add_commute.bind commuteLHS commuteRHS

structure Rules where
  extra : Spec.Assumptions ComputeStmt
  comm : Spec.EvidenceValidated commute.rule commute.evidence
  assoc : Spec.Derivation (commute :: extra) associateLHS associateRHS

def rules (R : Rules) : Spec.Assumptions ComputeStmt := commute :: R.extra

theorem commutation (R : Rules) : Spec.Derivation (rules R) commuteLHS commuteRHS :=
  .atom commute (by simp [rules])
    (FP.ReportedAdmission.fp32_add_commute.admit _ _ R.comm)

theorem association (R : Rules) : Spec.Derivation (rules R) associateLHS associateRHS :=
  R.assoc

theorem reduction_reorder (R : Rules) (a b : SumTree α)
    (h : a.leaves.Perm b.leaves) : TermEq (rules R) a.eval b.eval :=
  a.equiv_of_perm (commutation R) (association R) b h

#print_fp_assumptions reduction_reorder

theorem opaque_term (R : Rules) (a b : Term α) (h : TermEq (rules R) a b) : TermEq (rules R) a b := h
#print_fp_assumptions opaque_term

theorem same_tree (a : SumTree α) : TermEq [] a.eval a.eval := .refl _
#print_fp_assumptions same_tree

theorem reassociation_only (R : Rules) (a b c : Term α) :
    TermEq (rules R) ((a.add b).add c) (a.add (b.add c)) :=
  .addAssociate (association R) a b c
#print_fp_assumptions reassociation_only

theorem inside_opaque_exp (R : Rules) (a b : Term α) :
    TermEq (rules R) (.app (.unary (some .fp32) .exp) [a.add b])
      (.app (.unary (some .fp32) .exp) [b.add a]) :=
  .context _ [] [] (.addCommute (commutation R) a b)
#print_fp_assumptions inside_opaque_exp

/-- This non-floating counting interpretation validates commutation and
association, but not additive identity or idempotent casts. -/
def cost : Symbol → Nat
  | .binary (some .fp32) .real .add => 1
  | _ => 2

def weight (a : Term α) : Nat := a.evaluate (fun _ => 1) (fun op args => cost op + args.sum)

theorem weight_preserved {R : Spec.Assumptions ComputeStmt} {a b : Term α} (h : TermEq R a b) :
    weight a = weight b := by
  apply TermEq.sound (fun _ => 1) (fun op args => cost op + args.sum) _ _ h
  · intro _ a b
    simp [cost, Nat.add_comm]
  · intro _ a b c
    simp [cost, Nat.add_assoc, Nat.add_left_comm]

theorem zero_cannot_be_removed (R : Spec.Assumptions ComputeStmt) :
    ¬ TermEq R ((Term.leaf (0 : Nat)).add (.app (.literal (some .fp32) .real 0) []))
      (.leaf 0) := by
  intro h
  have hw := weight_preserved h
  simp [weight, cost, Term.evaluate, Term.add] at hw

theorem repeated_cast_cannot_be_removed (R : Spec.Assumptions ComputeStmt) :
    ¬ TermEq R
      (.app (.cast (some .fp32) .real .bf16)
        [.app (.cast (some .fp32) .real .bf16) [.leaf (0 : Nat)]])
      (.app (.cast (some .fp32) .real .bf16) [.leaf 0]) := by
  intro h
  have hw := weight_preserved h
  simp [weight, cost, Term.evaluate] at hw

theorem no_rules_means_syntax_equality {Statement : Type} {a b : List Statement}
    (h : Spec.Derivation [] a b) : a = b := by
  induction h with
  | refl => rfl
  | atom _ hm _ => cases hm
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | frame _ _ _ ih => rw [ih]

theorem precision_cannot_be_erased (R : Spec.Assumptions ComputeStmt) :
    ¬ TermEq R ((Term.leaf (0 : Nat)).add (.leaf 1))
      (.app (.binary none .real .add) [.leaf 0, .leaf 1]) := by
  intro h
  have hw := weight_preserved h
  simp [weight, cost, Term.evaluate, Term.add] at hw

#guard_msgs (drop info) in
#auditModuleAxioms
end FPEquationalTests

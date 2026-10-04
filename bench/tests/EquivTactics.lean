import VeriTile.Meta.FPProve
import VeriTile.Meta.StatementAudit
import bench.examples.FlatVectorAdd.FPEquiv
import bench.examples.VectorAdd.FPEquiv
import bench.examples.FloatDTypeAdd.FPEquiv
import bench.examples.HyperConnectionsWidth.FPEquiv

/-!
Proof-producing decomposition and search: multiple differences, insertion,
deletion, symmetry, transitivity, model metadata and missing assumptions.
The abstract Nat statements test structure without any floating-point model.
-/
namespace VeriTile.Tests.EquivTactics
set_option linter.unusedVariables false
open VeriTile Triton
open scoped VeriTile.Spec

theorem two_differences (R : Spec.Assumptions Nat)
    (ha : Spec.Derivation R [2] [3]) (hb : Spec.Derivation R [4] [5]) :
    [0, 2, 4, 9] ≡[R] [0, 3, 5, 9] := by
  equiv_decompose
  · change Spec.Derivation R [2] [3]
    exact ha
  · change Spec.Derivation R [4] [5]
    exact hb

theorem repeated_anchors (R : Spec.Assumptions Nat)
    (ha : Spec.Derivation R [2] [3]) (hb : Spec.Derivation R [4] [5]) :
    [0, 2, 0, 4, 0] ≡[R] [0, 3, 0, 5, 0] := by
  equiv_decompose
  all_goals fp_prove

theorem insertion (R : Spec.Assumptions Nat) (h : Spec.Derivation R [] [3, 4]) :
    [0, 9] ≡[R] [0, 3, 4, 9] := by
  equiv_decompose
  change Spec.Derivation R [] [3, 4]
  fp_prove

theorem deletion (R : Spec.Assumptions Nat) (h : Spec.Derivation R [] [3, 4]) :
    [0, 3, 4, 9] ≡[R] [0, 9] := by
  equiv_decompose
  fp_prove

theorem intact_block (R : Spec.Assumptions Nat) (h : Spec.Derivation R [2, 4] [3, 5]) :
    [0, 2, 4, 9] ≡[R] [0, 3, 5, 9] := by
  equiv_decompose (split := false)
  change Spec.Derivation R [2, 4] [3, 5]
  fp_prove

theorem empty_identity (R : Spec.Assumptions Nat) : ([] : List Nat) ≡[R] [] := by
  equiv_decompose

theorem symbolic_identity (R : Spec.Assumptions Nat) (xs : List Nat) : xs ≡[R] xs := by
  equiv_decompose

theorem symbolic_obligation (R : Spec.Assumptions Nat) (xs ys : List Nat)
    (h : Spec.Derivation R xs ys) : xs ≡[R] ys := by
  equiv_decompose
  change Spec.Derivation R xs ys
  exact h

theorem transitive_reverse (R : Spec.Assumptions Nat)
    (h1 : Spec.Derivation R [2] [3]) (h2 : Spec.Derivation R [3] [4]) :
    Spec.Derivation R [4] [2] := by
  fail_if_success fp_prove (maxSteps := 1)
  fp_prove (maxSteps := 2)

theorem contextual_search (R : Spec.Assumptions Nat)
    (h1 : Spec.Derivation R [2] [3]) (h2 : Spec.Derivation R [3] [4]) :
    Spec.Derivation R [8, 2, 9] [8, 4, 9] := by
  fp_prove

-- Explicit hints can discharge guards from local premises through another lemma.
private theorem guarded_step (R : Spec.Assumptions Nat) (P Q : Prop)
    (h : P → Spec.Derivation R [2] [3]) (back : Q → P) (q : Q) :
    Spec.Derivation R [2] [3] := h (back q)

theorem hinted_step (R : Spec.Assumptions Nat) (P Q : Prop)
    (h : P → Spec.Derivation R [2] [3]) (back : Q → P) (q : Q) :
    Spec.Derivation R [2] [3] := by
  fp_prove [guarded_step]

theorem domain_is_required (R : Spec.Assumptions Nat) (P : Prop)
    (h : P → Spec.Derivation R [2] [3]) : True := by
  fail_if_success
    have : Spec.Derivation R [2] [3] := by fp_prove
  trivial

-- A custom execution view ensures body equality never erases signature or
-- private-context obligations, including a structural-view ProgramDerivation.
structure Program where
  port : Nat
  scratch : Nat
  code : List Nat

instance : Spec.ProgramSyntax Program where
  Statement := Nat
  Signature := Nat
  signature := Program.port
  body := Program.code
  structural := some Eq
  sameContext := fun a b => a.scratch = b.scratch

theorem metadata_preserved (R : Spec.Assumptions Nat) (a b c d : Nat)
    (ports : a = b) (context : c = d) (h : Spec.Derivation R [2] [3]) :
    (⟨a, c, [0, 2, 9]⟩ : Program) ≡[R] ⟨b, d, [0, 3, 9]⟩ := by
  equiv_decompose
  · change a = b
    exact ports
  · change c = d
    exact context
  · change Spec.Derivation R [2] [3]
    fp_prove

theorem structural_view (R : Spec.Assumptions Nat) (h : Spec.Derivation R [2] [3]) :
    Spec.ProgramDerivation Eq R (⟨0, 0, [8, 2, 9]⟩ : Program) ⟨0, 0, [8, 3, 9]⟩ := by
  equiv_decompose
  fp_prove

-- Failed attempts must not solve the goal, silently accept missing admissions,
-- drop changed metadata, or discard insertions/deletions.
theorem rejected_changes : True := by
  fail_if_success
    have : ([1] : List Nat) ≡[([] : Spec.Assumptions Nat)] [2] := by
      equiv_decompose
      fp_prove
  fail_if_success
    have : ([] : List Nat) ≡[([] : Spec.Assumptions Nat)] [2] := by
      equiv_decompose
      fp_prove
  fail_if_success
    have : (⟨0, 0, [2]⟩ : Program) ≡[[]] ⟨1, 0, [2]⟩ := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : (⟨0, 0, [2]⟩ : Program) ≡[[]] ⟨0, 1, [2]⟩ := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Merely having two-hop rewrites and cycles does not prove an unrelated goal.
theorem bounded_failure (R : Spec.Assumptions Nat)
    (h : Spec.Derivation R [2] [3]) : True := by
  fail_if_success
    have : Spec.Derivation R [2] [4] := by fp_prove
  trivial

open VeriTile.Bench.Examples

-- The actual registered fp32 atom can be found in an imported module.
theorem imported_fp_rule (B : Nat) (R : VectorAddFPEquiv.Rules B) :
    VectorAdd.Kernels.originalKernel B ≡[R] VectorAdd.Kernels.optimizedKernel B := by
  equiv_decompose
  fp_prove

-- In the two-site example, decomposition exposes both multiplication targets.
theorem two_fp_sites (tau : Real) (R : HyperConnectionsWidthFPEquiv.Rules tau) :
    HyperConnectionsWidth.Kernels.originalKernel tau ≡[R]
      HyperConnectionsWidth.Kernels.optimizedKernel tau := by
  equiv_decompose
  · fp_prove
  · fp_prove

def add64 (B : Nat) : List ComputeStmt :=
  [.assign .real [B] "out" (.compute (.alg .fp64
    (.add .real (.consSame .nil) (.ref .real [B] "y") (.ref .real [B] "x"))))]

theorem fp_precision_and_admission (B : Nat) (R : VectorAddFPEquiv.Rules B) : True := by
  fail_if_success
    have : Spec.Derivation R.assumptions (VectorAddFPEquiv.originalAdd B) (add64 B) := by
      fp_prove
  fail_if_success
    have : Spec.Derivation ([] : Spec.Assumptions ComputeStmt)
        (VectorAddFPEquiv.originalAdd B) (VectorAddFPEquiv.optimizedAdd B) := by
      fp_prove
  trivial

theorem output_cast_retained (B : Nat) (R : FloatDTypeAddFPEquiv.Rules B) : True := by
  fail_if_success
    have : FloatDTypeAdd.Kernels.originalKernel B ≡[R] VectorAdd.Kernels.optimizedKernel B := by
      equiv_decompose
      all_goals fp_prove
  trivial

theorem active_mask_retained (n B : Nat) (R : FlatVectorAddFPEquiv.Rules B) : True := by
  fail_if_success
    have : FlatVectorAdd.Kernels.originalKernel n B ≡[R]
        FlatVectorAdd.Kernels.optimizedKernel (n + 1) B := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- The unchanged printer reports the atom used by the generated proof.
/-- info: FP assumptions used by imported_fp_rule:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions imported_fp_rule

/-- info: FP assumptions used by two_fp_sites:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions two_fp_sites

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Tests.EquivTactics

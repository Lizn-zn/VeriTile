/- Specification/report regressions. Synthetic cases below are test fixtures,
not kernel acceptance experiments. In particular pendingConditional has an
uninhabitable atom-approval premise, which its report must expose as NOT_RUN. -/
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarOps

open VeriTile
open scoped VeriTile.Spec

namespace SpecificationSurface

@[spec_primitive] def identityPrimitive (n : Nat) : Nat := n
@[spec_primitive] def unusedPrimitive (n : Nat) : Nat := n + 1

@[spec_rule] theorem identityRule (n : Nat) : identityPrimitive n = n := rfl

-- A project namespace that resembles a trusted namespace must still be walked.
namespace Nat
theorem hiddenHelper (n : Nat) : identityPrimitive n = n := identityRule n
end Nat

specification ideal (n : Nat) (hN : n > 0) : Spec.Real (identityPrimitive n = n) := by
  exact Nat.hiddenHelper n

abbrev IdealAlias (p : Prop) := Spec.Real p
specification aliasedIdeal : IdealAlias True := True.intro

class Model where
  op : Nat → Nat
  identity : ∀ n, op n = n

theorem usingModel [m : Model] (n : Nat) : m.op n = n := m.identity n

specification modeled [m : Model] (n : Nat) : Spec.Real (m.op n = n) := usingModel n

-- Ensure optimized raw Expr.proj nodes, not just named accessor applications,
-- are covered by the inspector as well.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let deps ← liftCoreM <| VeriTile.Meta.specExprConsts env
    (.proj ``Model 1 (.const `dummyModel []))
  unless deps.contains ``Model.identity do
    throwError "raw structure projection was not recorded"

def testRule : Spec.AtomicRule Nat where
  lhs := [0, 1]
  rhs := [1, 0]
  contract := {
    ruleID := "ATOM-FIXTURE"
    instanceKey := "synthetic-fixture-only"
    configuration := Lean.Json.mkObj [
      ("test_only", .bool true), ("shape", .arr #[32]),
      ("dtype", .str "bf16"), ("accumulator", .str "fp32"),
      ("probe", .str "role-asymmetric-gaussian"),
      ("backend", .str "synthetic; no experiment performed")]
    warningPolicy := .passOnly }

def pending : Spec.RuleEntry Nat where
  rule := testRule
  evidence := { instanceKey := "synthetic-fixture-only" }

-- Approval is a separate atom-level operation, never the whole specification.
theorem admit (entry : Spec.RuleEntry Nat)
    (hValidated : Spec.EvidenceValidated entry.rule entry.evidence)
    (hIdentity : entry.evidence.instanceKey = entry.rule.contract.instanceKey)
    (hArtifact : entry.evidence.artifact.isSome = true)
    (hGates : Spec.PolicyAllows entry.rule.contract.warningPolicy entry.evidence) :
    Spec.AcceptedAtom entry := ⟨hValidated, hIdentity, hArtifact, hGates⟩

specification numerical (entry : Spec.RuleEntry Nat) (hAtom : Spec.AcceptedAtom entry) :
    Spec.FloatingPoint [entry] entry.rule.lhs entry.rule.rhs :=
  ⟨rfl, .atom entry (by simp) hAtom⟩

specification pendingConditional (hAtom : Spec.AcceptedAtom pending) :
    Spec.FloatingPoint [pending] [0, 1] [1, 0] := numerical pending hAtom

@[spec_rule] theorem localRule (entry : Spec.RuleEntry Nat)
    (hAtom : Spec.AcceptedAtom entry) :
    Spec.Derivation [entry] entry.rule.lhs entry.rule.rhs :=
  .atom entry (by simp) hAtom

specification usesLocalRule (entry : Spec.RuleEntry Nat) (hAtom : Spec.AcceptedAtom entry) :
    Spec.FloatingPoint [entry] ([9] ++ entry.rule.lhs ++ [8])
      ([9] ++ entry.rule.rhs ++ [8]) :=
  ⟨rfl, .frame ([9] : List Nat) ([8] : List Nat) (localRule entry hAtom)⟩

-- A public headline takes only its rule model. Hiding the bookkeeping behind
-- that model must not hide the atom or the admission premise from #print_spec.
structure Rules where
  entry : Spec.RuleEntry Nat
  admitted : Spec.AcceptedAtom entry

def Rules.assumptions (R : Rules) : Spec.Assumptions Nat := [R.entry]
instance : Coe Rules (Spec.Assumptions (Spec.ProgramSyntax.Statement (List Nat))) :=
  ⟨Rules.assumptions⟩

specification fromModel (R : Rules) :
    R.entry.rule.lhs ≡[R] R.entry.rule.rhs :=
  ⟨rfl, .atom R.entry (by simp [Rules.assumptions]) R.admitted⟩

example : ¬ Spec.AcceptedAtom pending := by
  intro h
  have impossible : false = true := h.artifactPresent
  cases impossible

def wrongIdentity : Spec.RuleEntry Nat where
  rule := testRule
  evidence := { instanceKey := "different-configuration"
                artifact := some "synthetic-not-an-experiment"
                bias := .pass, vars := .pass }

example : ¬ Spec.AcceptedAtom wrongIdentity := by
  intro h
  have distinct : wrongIdentity.evidence.instanceKey ≠ testRule.contract.instanceKey := by decide
  exact distinct h.instanceMatches

def failedGate : Spec.RuleEntry Nat where
  rule := testRule
  evidence := { instanceKey := "synthetic-fixture-only"
                artifact := some "synthetic-not-an-experiment"
                bias := .pass, vars := .fail }

example : ¬ Spec.AcceptedAtom failedGate := by
  intro h
  have impossible : false = true := h.gates.2
  cases impossible

-- If none of the table entries is admitted, even composition, symmetry and
-- contextual lifting cannot turn different syntax into an equivalence.
theorem no_unchecked_rewrite {rules : Spec.Assumptions Nat}
    (hNone : ∀ entry ∈ rules, ¬ Spec.AcceptedAtom entry)
    {lhs rhs : List Nat} (h : Spec.Derivation rules lhs rhs) : lhs = rhs := by
  induction h with
  | refl => rfl
  | atom entry member accepted => exact False.elim (hNone entry member accepted)
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | frame _ _ _ ih => rw [ih]

example (entry : Spec.RuleEntry Nat) (hAtom : Spec.AcceptedAtom entry) :
    Spec.FloatingPoint [entry] entry.rule.rhs entry.rule.rhs :=
  (numerical entry hAtom).symm.trans (numerical entry hAtom)

-- Statement-sequence equivalence must not silently change the IO signature.
structure Program where
  output : Nat
  code : List Nat

instance : Spec.ProgramSyntax Program where
  Statement := Nat
  Signature := Nat
  signature := Program.output
  body := Program.code

example (rules : Spec.Assumptions Nat) :
    ¬ Spec.FloatingPoint rules (Program.mk 0 []) (Program.mk 1 []) := by
  intro h
  have impossible : (0 : Nat) = 1 := h.sameSignature
  cases impossible

-- Concrete primitive annotations are found through helpers too.
def concreteHelper (x : Triton.FP.Value .bf16) :=
  Triton.FP.add Triton.FP.Config.ieeeValue x x

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let deps ← liftCoreM <| VeriTile.Meta.specDependencies env [``concreteHelper]
  unless deps.project.contains ``Triton.FP.add && deps.project.contains ``Triton.FP.round do
    throwError "concrete primitive dependency missing"
  unless VeriTile.Meta.specPrimitiveAttr.hasTag env ``Triton.FP.add do
    throwError "concrete primitive tag missing"

-- All 25 gate pairs, under both policies. WARN is never silently turned PASS;
-- FAIL, INCONCLUSIVE and NOT_RUN cannot establish acceptance.
example :
    [Spec.GateStatus.notRun, .pass, .warn, .fail, .inconclusive].all (fun a =>
      [Spec.GateStatus.notRun, .pass, .warn, .fail, .inconclusive].all (fun b =>
        (a.allowed .passOnly && b.allowed .passOnly) == (a == .pass && b == .pass) &&
        (a.allowed .allowWarn && b.allowed .allowWarn) ==
          ((a == .pass || a == .warn) && (b == .pass || b == .warn)))) = true := by decide

-- Existing keyword, attributes, and unwrapped statements still work.
@[simp] private specification «legacy headline» (n : Nat) : n + 0 = n := rfl

#print_spec ideal full
#print_spec aliasedIdeal
#print_spec modeled
#print_spec numerical
#print_spec pendingConditional
#print_spec usesLocalRule
#print_spec fromModel
#print_spec «legacy headline»
#auditModuleAxioms

end SpecificationSurface

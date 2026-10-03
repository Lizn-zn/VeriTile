"""Proof-directed FP assumption printing, independently of the declared table."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """import VeriTile.Meta.StatementAudit
open VeriTile
open scoped VeriTile.Spec

def entry (name scope : String) : Spec.RuleEntry Nat where
  rule := {
    lhs := [0], rhs := [1]
    contract := {
      ruleID := name, instanceKey := name ++ scope,
      configuration := .null, warningPolicy := .passOnly,
      description := scope } }
  evidence := { instanceKey := name ++ scope }

def used := entry "USED" "shape=32"
def unused := entry "UNUSED" "shape=32"

theorem step (rules : Spec.Assumptions Nat) (atom : Spec.RuleEntry Nat)
    (hm : atom ∈ rules) (ha : Spec.AcceptedAtom atom) :
    Spec.Derivation rules atom.rule.lhs atom.rule.rhs := .atom atom hm ha

theorem helper (ha : Spec.AcceptedAtom used) :
    Spec.FloatingPoint [used, unused] [0] [1] :=
  ⟨rfl, step [used, unused] used (by simp) ha⟩
"""


class FPAssumptionTests(unittest.TestCase):
    def run_lean(self, source, success=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'FPAssumptions.lean'
            path.write_text(PRELUDE + source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result.stdout

    def test_instantiated_helpers_composition_and_deduplication(self):
        output = self.run_lean("""
theorem composed (ha : Spec.AcceptedAtom used) :
    Spec.FloatingPoint [used, unused] [1] [1] :=
  (helper ha).symm.trans (helper ha)
#print_fp_assumptions composed
""")
        self.assertEqual(output.count('  used\n'), 1)
        self.assertNotIn('shape=', output)
        self.assertNotIn('unused', output)
        self.assertNotIn('unresolved', output)
        self.assertNotIn('bias', output)
        for hidden in ('Specification:', 'Claim:', 'Parameters', 'EvidenceValidated'):
            self.assertNotIn(hidden, output)

    def test_reflexivity_uses_no_atoms_from_its_table(self):
        output = self.run_lean("""
theorem identity : Spec.FloatingPoint [used, unused] [0] [0] := .refl _
#print_fp_assumptions identity
""")
        self.assertIn('FP assumptions used by identity:\n  none', output)
        self.assertNotIn('  used', output)

    def test_distinct_instances_with_same_id_are_not_collapsed(self):
        output = self.run_lean("""
def other := entry "USED" "shape=64"
theorem twoShapes (h32 : Spec.AcceptedAtom used) (h64 : Spec.AcceptedAtom other) :
    Spec.Derivation [used, other] [0] [0] :=
  .trans (step [used, other] used (by simp) h32)
    (.symm (step [used, other] other (by simp) h64))
#print_fp_assumptions twoShapes
""")
        self.assertEqual(output.count('  used\n'), 2)
        self.assertNotIn('shape=', output)

    def test_same_admission_at_two_rewrite_sites_is_printed_once(self):
        output = self.run_lean("""
def renamed : Spec.RuleEntry Nat :=
  { used with rule := { used.rule with lhs := [2], rhs := [3] } }
theorem twoSites (h : Spec.AcceptedAtom used) (h' : Spec.AcceptedAtom renamed) :
    Spec.Derivation [used, renamed] [0, 2] [1, 3] :=
  .trans (.frame [] [2] (step [used, renamed] used (by simp) h))
    (.frame [1] [] (step [used, renamed] renamed (by simp) h'))
#print_fp_assumptions twoSites
""")
        self.assertEqual(output.count('  used\n'), 1)

    def test_same_key_with_different_contract_or_evidence_stays_distinct(self):
        output = self.run_lean("""
def otherContract : Spec.RuleEntry Nat :=
  { used with rule := { used.rule with contract :=
    { used.rule.contract with configuration := .bool true } } }
def otherEvidence : Spec.RuleEntry Nat :=
  { used with evidence := { used.evidence with artifact := some "other-report" } }
theorem threeAdmissions
    (h : Spec.AcceptedAtom used)
    (hc : Spec.AcceptedAtom otherContract)
    (he : Spec.AcceptedAtom otherEvidence) :
    Spec.Derivation [used, otherContract, otherEvidence] [0] [1] :=
  .trans (step _ used (by simp) h)
    (.trans (.symm (step _ otherContract (by simp) hc))
      (step _ otherEvidence (by simp) he))
#print_fp_assumptions threeAdmissions
""")
        self.assertEqual(output.count('  used\n'), 3)

    def test_symbolic_atom_and_opaque_derivation_stay_explicit(self):
        output = self.run_lean("""
theorem symbolic (atom : Spec.RuleEntry Nat) (h : Spec.AcceptedAtom atom) :
    Spec.Derivation [atom] atom.rule.lhs atom.rule.rhs := .atom atom (by simp) h
#print_fp_assumptions symbolic
theorem conditional (h : Spec.FloatingPoint [used] [0] [1]) :
    Spec.FloatingPoint [used] [0] [1] := h
#print_fp_assumptions conditional
structure Model where
  proof : Spec.Derivation [used] [0] [1]
theorem projected (m : Model) : Spec.Derivation [used] [0] [1] := m.proof
#print_fp_assumptions projected
""")
        self.assertIn('atom [symbolic atom]', output)
        self.assertIn('unresolved FP proof: h', output)
        self.assertIn('unresolved FP proof: m.1', output)
        self.assertNotIn('  none', output)
        self.assertNotIn('  used\n', output)

    def test_non_fp_theorems_are_rejected(self):
        output = self.run_lean('theorem realClaim : Spec.Real True := True.intro\n'
                               '#print_fp_assumptions realClaim\n', success=False)
        self.assertIn('expected a floating-point equivalence or derivation', output)

    def test_generic_transport_cannot_hide_atoms_from_dependency_pruning(self):
        output = self.run_lean("""
theorem genericTransport {P : Prop} (h : P) : P := h
theorem transported (h : Spec.AcceptedAtom used) :
    Spec.Derivation [used] [0] [1] :=
  genericTransport (step [used] used (by simp) h)
#print_fp_assumptions transported
""")
        self.assertEqual(output.count('  used\n'), 1)
        self.assertNotIn('  none', output)

    def test_induction_hypotheses_do_not_hide_atoms_or_appear_as_external_premises(self):
        output = self.run_lean("""
theorem repeated (n : Nat) (ha : Spec.AcceptedAtom used) :
    Spec.FloatingPoint [used, unused] [0] [0] := by
  induction n with
  | zero => exact .refl _
  | succ n ih => exact (helper ha).trans ((helper ha).symm.trans ih)
#print_fp_assumptions repeated
""")
        self.assertEqual(output.count('  used\n'), 1)
        self.assertNotIn('unresolved', output)
        self.assertNotIn('  none', output)

    def test_weakening_reports_input_atoms_and_opaque_inputs_not_branch_binders(self):
        output = self.run_lean("""
theorem weaken {rules : Spec.Assumptions Nat} {lhs rhs : List Nat}
    (h : Spec.Derivation rules lhs rhs) :
    Spec.Derivation (rules ++ [unused]) lhs rhs := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (List.mem_append_left _ he) ha
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame beforeCode afterCode _ ih => exact .frame beforeCode afterCode ih
theorem weakened (h : Spec.AcceptedAtom used) :
    Spec.Derivation ([used] ++ [unused]) [0] [1] :=
  weaken (step [used] used (by simp) h)
#print_fp_assumptions weakened
theorem weakenedOpaque (h : Spec.Derivation [used] [0] [1]) :
    Spec.Derivation ([used] ++ [unused]) [0] [1] := weaken h
#print_fp_assumptions weakenedOpaque
""")
        self.assertEqual(output.count('  used\n'), 1)
        self.assertNotIn('symbolic atom', output)
        self.assertIn('unresolved FP proof: h', output)
        self.assertNotIn('  unused', output)
        self.assertNotIn('  none', output)

    def test_unpacking_a_symbolic_entry_does_not_hide_its_admission(self):
        output = self.run_lean("""
structure Packed where
  atom : Spec.RuleEntry Nat
  accepted : Spec.AcceptedAtom atom
theorem unpacked (p : Packed) :
    Spec.Derivation [p.atom] p.atom.rule.lhs p.atom.rule.rhs := by
  cases p with
  | mk atom accepted => exact .atom atom (by simp) accepted
#print_fp_assumptions unpacked
""")
        self.assertIn('[symbolic atom]', output)
        self.assertNotIn('  none', output)

    def test_structural_composition_keeps_atoms_and_opaque_steps_visible(self):
        output = self.run_lean("""
structure Code where
  body : List Nat
instance : Spec.ProgramSyntax Code where
  Statement := Nat
  Signature := Unit
  signature := fun _ => ()
  body := Code.body
  structural := some Eq
def c0 : Code := ⟨[0]⟩
def c1 : Code := ⟨[1]⟩
theorem structuralStart : Spec.FloatingPoint [used] c0 c0 :=
  Spec.FloatingPoint.ofStructural (structural := Eq) rfl rfl rfl
theorem numericalStep (h : Spec.AcceptedAtom used) : Spec.FloatingPoint [used] c0 c1 :=
  Spec.FloatingPoint.ofDerivation rfl trivial (step [used] used (by simp) h)
theorem combined (h : Spec.AcceptedAtom used) : Spec.FloatingPoint [used] c0 c1 :=
  structuralStart.trans (numericalStep h)
#print_fp_assumptions combined
theorem opaqueStep (h : Spec.ProgramDerivation (Program := Code) Eq [used] c0 c1) :
    Spec.FloatingPoint [used] c0 c1 := ⟨rfl, h⟩
#print_fp_assumptions opaqueStep
""")
        self.assertEqual(output.count('  used\n'), 1)
        self.assertIn('unresolved FP proof: h', output)
        self.assertNotIn('  none', output)


if __name__ == '__main__':
    unittest.main()

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


if __name__ == '__main__':
    unittest.main()

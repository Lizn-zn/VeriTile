"""Lean-backed regressions for the two specification meanings and their reports.

Build `VeriTile.Meta.StatementAudit` and `VeriTile.Triton.Float.ScalarOps`
first. These fixtures are not numerical acceptance experiments.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def lean(path):
    return subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                          text=True, capture_output=True, timeout=180)


class SpecificationSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = lean(ROOT / "bench/tests/SpecificationSurface.lean")
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.output = result.stdout

    def report(self, name):
        start = self.output.index(f"Specification: SpecificationSurface.{name}\n")
        end = self.output.find("\nSpecification: ", start)
        return self.output[start:] if end < 0 else self.output[start:end]

    def test_real_meaning_and_transitive_proof_dependencies(self):
        report = self.report("ideal")
        self.assertIn("Kind: REAL (explicit", report)
        self.assertIn("assumption hN: n > 0", report)
        self.assertIn("SpecificationSurface.identityRule", report)
        self.assertIn("SpecificationSurface.Nat.hiddenHelper", report)
        self.assertIn("Reachable execution primitives (1): [SpecificationSurface.identityPrimitive]", report)
        self.assertNotIn("unusedPrimitive", report)
        self.assertIn("Trusted library boundary:", report)
        self.assertIn("Kind: REAL (explicit", self.report("aliasedIdeal"))

    def test_model_hypotheses_are_not_confused_with_global_axioms(self):
        report = self.report("modeled")
        self.assertIn("class parameter m:", report)
        self.assertIn("SpecificationSurface.Model.identity:", report)
        self.assertIn("Transitive axioms: []", report)

    def test_equivalence_reports_implementations_and_atom_obligations(self):
        report = self.report("numerical")
        self.assertIn("Kind: FLOATING_POINT_EQUIVALENCE", report)
        self.assertIn("assumption hAtom: Spec.AcceptedAtom entry", report)
        self.assertIn("Implementation lhs: entry.rule.lhs", report)
        self.assertIn("Implementation rhs: entry.rule.rhs", report)
        for field in ("atom lhs", "atom rhs", "rule ID", "configuration", "warning policy",
                      "evidence key", "artifact", "bias gate", "vars gate"):
            self.assertIn(f"  {field}:", report)
        self.assertIn("conditional on all declared assumptions", report)
        self.assertIn("atomic validation obligation: Spec.EvidenceValidated", report)
        self.assertNotIn("algebraic obligation:", report)

    def test_pending_results_stay_pending_and_show_their_scope(self):
        report = self.report("pendingConditional")
        self.assertIn("bias gate: NOT_RUN", report)
        self.assertIn("vars gate: NOT_RUN", report)
        self.assertIn("artifact: none", report)
        self.assertIn('rule ID: "ATOM-FIXTURE"', report)
        for dimension in ("shape", "dtype", "accumulator", "probe", "backend"):
            self.assertIn(f'"{dimension}"', report)
        self.assertIn("assumption hAtom:", report)
        self.assertNotIn("bias gate: PASS", report)
        self.assertIn("synthetic; no experiment performed", report)

    def test_compact_view_preserves_pending_status_and_premises(self):
        source = (ROOT / 'bench/tests/SpecificationSurface.lean').read_text()
        source = source.split('@[spec_rule] theorem localRule')[0]
        source += '\n#print_spec pendingConditional\n#print_spec ideal\nend SpecificationSurface\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Compact.lean'
            path.write_text(source)
            result = lean(path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ATOM-FIXTURE [bias NOT_RUN; vars NOT_RUN]', result.stdout)
        self.assertIn('hAtom : Spec.AcceptedAtom pending', result.stdout)
        self.assertIn('hN : n > 0', result.stdout)
        self.assertNotIn('bias PASS', result.stdout)
        self.assertNotIn('atomic validation obligation:', result.stdout)
        self.assertNotIn('Dependency boundary:', result.stdout)

    def test_local_rule_is_a_scoped_acceptance_premise(self):
        report = self.report("usesLocalRule")
        self.assertIn('assumption hAtom: Spec.AcceptedAtom entry', report)
        self.assertIn("SpecificationSurface.localRule", report)
        self.assertIn("Symmetry, composition and common context are formal proof rules", report)
        self.assertIn("No IEEE value equality or whole-kernel two-gates result", report)

    def test_public_notation_keeps_model_assumptions_visible(self):
        report = self.report("fromModel")
        self.assertIn("Kind: FLOATING_POINT_EQUIVALENCE", report)
        self.assertIn("parameter R: Rules", report)
        self.assertIn("SpecificationSurface.Rules.admitted:", report)
        self.assertIn("Spec.AcceptedAtom", report)
        self.assertIn("Atom 1:", report)
        self.assertIn("atomic validation obligation:", report)
        self.assertIn("Transitive axioms:", report)
        self.assertNotIn("Nonstandard axioms:", report)
        self.assertNotIn("parameter experiment:", report)
        self.assertNotIn("parameter evidence:", report)

    def test_equivalence_notation_coexists_with_legacy_kernelio_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Notation.lean"
            path.write_text("""import VeriTile.Spec
import VeriTile.Triton.Memory.KernelSpec.Masked
open VeriTile Triton
open scoped VeriTile.Spec VeriTile.Triton.MaskedKernelIO₂
example (lhs rhs : MaskedKernelIO₂) (R : RoundingModel) :
    (lhs ≡[R] rhs) = MaskedKernelIO₂.Equiv lhs rhs R := rfl
example (lhs rhs : List Nat) (R : Spec.Assumptions Nat) :
    (lhs ≡[R] rhs) = Spec.FloatingPoint R lhs rhs := rfl
""")
            result = lean(path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_evidence_cannot_be_reused_for_a_different_atom(self):
        self.assert_rejected("""
example (left right : VeriTile.Spec.AtomicRule Nat) (e : VeriTile.Spec.Evidence)
    (h : VeriTile.Spec.EvidenceValidated left e) :
    VeriTile.Spec.EvidenceValidated right e := h
""")

    def test_equivalence_is_not_ieee_or_lean_equality(self):
        self.assert_rejected("""
example (rules : VeriTile.Spec.Assumptions Nat)
    (h : VeriTile.Spec.FloatingPoint rules [0] [1]) : (0 : Nat) = 1 := h
""")

    def test_pass_labels_do_not_admit_an_atom(self):
        self.assert_rejected("""
example (entry : VeriTile.Spec.RuleEntry Nat)
    (hBias : entry.evidence.bias = .pass) (hVars : entry.evidence.vars = .pass) :
    VeriTile.Spec.AcceptedAtom entry := by
  constructor
""")

    def test_syntax_only_proofs_still_report_operation_primitives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Syntax.lean"
            path.write_text("""import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Core.Ast
open VeriTile.Triton
def op : Op .real [] := .add .real .nil (.ref .real [] "x") (.ref .real [] "y")
specification headline : VeriTile.Spec.Real (op = op) := rfl
#print_spec headline full
""")
            result = lean(path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("VeriTile.Triton.Op.add", result.stdout)
            self.assertIn("VeriTile.Triton.Op.ref", result.stdout)

    def assert_rejected(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Rejected.lean"
            path.write_text("import VeriTile.Spec\n" + source)
            result = lean(path)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_registration_and_standard_axioms_still_pass(self):
        self.assertIn("Kind: REAL (legacy unwrapped", self.output)
        self.assertIn("Axiom audit: headlines=8", self.output)
        self.assertNotIn("sorryAx", self.output)
        self.assertNotIn("DISALLOWED", self.output)
        for fixture in ("StatementAudit.lean", "ModuleHeadlineAudit.lean"):
            result = lean(ROOT / "bench/tests" / fixture)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonstandard_axiom_hidden_by_a_helper_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Injected.lean"
            path.write_text("""import VeriTile.Meta.StatementAudit
private axiom injected : False
private theorem helper : False := injected
specification headline : VeriTile.Spec.Real False := helper
#print_spec headline
""")
            result = lean(path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Nonstandard axioms:", result.stdout)
            self.assertIn("injected", result.stdout)

    def test_plain_theorem_is_not_mislabeled_as_a_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NotHeadline.lean"
            path.write_text("import VeriTile.Meta.StatementAudit\n"
                            "theorem helper : True := True.intro\n#print_spec helper\n")
            result = lean(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a registered specification", result.stdout)


if __name__ == "__main__":
    unittest.main()

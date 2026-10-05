"""Actual rewrite operands, guarded admissions, and executable contexts."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "bench/tests/FPGuardedExamples.lean"


class GuardedExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = [line.removeprefix("import ") for line in FIXTURE.read_text().splitlines()
                   if line.startswith("import ")]
        result = subprocess.run(["lake", "build", *targets], cwd=ROOT,
                                text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GuardedExamples.lean"
            path.write_text(source)
            result = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_intermediate_domains_and_successful_runs(self):
        self.assertEqual(self.lean(FIXTURE.read_text()), "")

    def test_opaque_contextual_equivalence_remains_visible(self):
        output = self.lean('''
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Meta.StatementAudit
open VeriTile Triton FP.GuardedRewrite
open scoped VeriTile.Spec
theorem opaqueContext (R : Spec.Assumptions FP.GuardedFragment) (a b : Program)
    (hs : Spec.ProgramSyntax.signature a = Spec.ProgramSyntax.signature b)
    (h : Equivalent R a b) : a ≡[R] b :=
  Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl hs h
#print_fp_assumptions opaqueContext
''')
        self.assertIn("unresolved FP proof: h", output)
        self.assertNotIn("  none", output)


if __name__ == "__main__":
    unittest.main()

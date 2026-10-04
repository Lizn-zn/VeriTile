"""Selected report identities, precision and domains belong to the test layer."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ('FPAdmissionMetadata', 'FPScalarAdmission')


class AdmissionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = sorted({
            line.removeprefix('import ')
            for name in FIXTURES
            for line in (ROOT / f'bench/tests/{name}.lean').read_text().splitlines()
            if line.startswith('import ')
        })
        result = subprocess.run(['lake', 'build', *targets], cwd=ROOT,
                                text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_selected_reports_match_their_bindings(self):
        for name in FIXTURES:
            with self.subTest(fixture=name):
                result = subprocess.run(['lake', 'env', 'lean', f'bench/tests/{name}.lean'],
                                        cwd=ROOT, text=True, capture_output=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('warning:', result.stdout)


if __name__ == '__main__':
    unittest.main()

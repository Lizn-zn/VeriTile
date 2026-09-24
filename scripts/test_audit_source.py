"""Source-selection and failure-propagation regressions for structural audits."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source


class SourceSelectionTests(unittest.TestCase):
    def test_primary_kernel_and_helper_coverage_are_distinct(self):
        text = '''@triton.jit
def helper(x):
    return x + 1

@triton.jit
@triton.autotune(configs=[])
def other(
    X,
    N: tl.constexpr,
):
    x = tl.load(X)

@triton.jit
def _attn_fwd(X):
    preferred = tl.load(X)

def host():
    return None
'''
        selected = source.python_first_kernel_body(text)
        self.assertIn('preferred = tl.load(X)', selected)
        self.assertNotIn('def host', selected)
        self.assertEqual(source.python_jit_kernel_bodies(text), [selected])
        all_bodies = source.python_jit_kernel_bodies(text, all_jit=True)
        self.assertEqual(len(all_bodies), 3)
        self.assertIn('return x + 1', all_bodies[0])
        self.assertIn('x = tl.load(X)', all_bodies[1])

    def test_block_pointer_selection_is_explicit(self):
        text = '''@triton.jit
def pointer(X):
    p = tl.make_block_ptr(X, (16,), (1,), (0,), (16,), (0,))

@triton.jit
def memory(X):
    x = tl.load(X)
'''
        self.assertIn('tl.load', source.python_first_kernel_body(text))
        self.assertIn('tl.make_block_ptr', source.python_first_kernel_body(
            text, include_block_pointers=True))
        self.assertEqual(source.python_jit_names(text), {'pointer', 'memory'})
        self.assertEqual(source.python_jit_names_text(text), {'pointer', 'memory'})

    def test_empty_or_incomplete_sources_have_no_kernel(self):
        for text in ['', 'def host():\n    return 1\n', '@triton.jit\ndef kernel(\n']:
            with self.subTest(text=text):
                self.assertEqual(source.python_first_kernel_body(text), '')
                self.assertEqual(source.python_jit_kernel_bodies(text), [])
        self.assertEqual(source.lean_first_triton_body('def x := 1'), '')
        self.assertEqual(source.lean_triton_bodies('def x := 1'), [])

    def test_target_selection_ignores_later_non_kernel_prose(self):
        text = '''/- Port of `example.py`'s `helper`.
Headline: `example.py`'s `kernel`.
Translation note: `example.py`'s `X += offset`. -/
def kernel := triton { x = tl.load(X) }
'''
        preamble = source.lean_first_preamble(text)
        self.assertEqual(source.target_kernel_candidates(preamble),
                         ['helper', 'kernel', 'X += offset'])
        self.assertEqual(source.target_kernel_name(preamble, {'helper', 'kernel'}), 'kernel')
        self.assertIsNone(source.target_kernel_name(preamble, {'unrelated'}))

    def test_lean_bodies_keep_nested_braces_and_source_order(self):
        text = 'preamble\ndef a := triton { x = $({ value := 1 }) }\ndef b := triton { y = 2 }'
        self.assertEqual(source.lean_triton_bodies(text), [' x = $({ value := 1 }) ', ' y = 2 '])
        self.assertEqual(source.lean_first_triton_body(text), ' x = $({ value := 1 }) ')

    def test_call_views_preserve_nested_arguments(self):
        body = '''# tl.load(ignored)
x = tl.load(
    X + f(a, b), mask=g(i, [j, k]), other=0,
)
y = tl.load(Y)
'''
        whole = source.tl_calls(body, 'load')
        args = source.tl_calls(body, 'load', arguments_only=True)
        normalized = source.tl_calls(body, 'load', normalize_whitespace=True)
        self.assertEqual(len(whole), 2)
        self.assertEqual(whole[1], 'tl.load(Y)')
        self.assertEqual(args[1], 'Y')
        self.assertNotIn('\n', normalized[0])
        expected = ['X + f(a, b)', 'mask=g(i, [j, k])', 'other=0']
        self.assertEqual(source.split_top_level_args(args[0]), expected)
        self.assertEqual(source.split_top_level_args(whole[0], 'load'), expected)
        self.assertEqual(source.split_top_level_args(normalized[0], 'load'), expected)

    def test_nested_lean_comments_preserve_diagnostic_lines(self):
        text = 'theorem first := by\n/- outer\n/- nested -/\nstill comment -/\n-- line comment\ntheorem second := by\n'
        code = source.strip_lean_comments(text)
        self.assertEqual(code.count('\n'), text.count('\n'))
        self.assertEqual(code.splitlines()[5], 'theorem second := by')
        for hidden in ('outer', 'nested', 'still comment', 'line comment'):
            self.assertNotIn(hidden, code)


class AuditRuleIntegrationTests(unittest.TestCase):
    def run_rule(self, diagnostic, lean_body):
        # Run the production shell's actual rule body. A missing shared import
        # or changed call-selection option must not turn mismatches into passes.
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", script, re.S)
        matches = [block for block in blocks if diagnostic in block]
        self.assertEqual(len(matches), 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            port = root / 'fixture'
            port.mkdir()
            (port / 'fixture.py').write_text('''@triton.jit
def kernel(X, Y, mask):
    x = tl.load(X, mask=mask, other=0).to(tl.float32)
    tl.store(Y, x, mask=mask)
''')
            (port / 'Fixture.lean').write_text('def kernel := triton {\n' + lean_body + '\n}\n')
            return subprocess.run([sys.executable, '-', str(root)], input=matches[0],
                                  cwd=ROOT, text=True, capture_output=True, timeout=20)

    def test_matching_casts_and_masks_pass(self):
        body = '  x = tl.load(X, mask=mask, other=0).to(tl.float32)\n  tl.store(Y, x, mask=mask)'
        for diagnostic in ('.to(...) cast sequence mismatch', 'tl.{fn} mask presence mismatch'):
            with self.subTest(diagnostic=diagnostic):
                result = self.run_rule(diagnostic, body)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_cast_fails(self):
        result = self.run_rule('.to(...) cast sequence mismatch',
                               '  x = tl.load(X, mask=mask, other=0)\n  tl.store(Y, x, mask=mask)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cast sequence mismatch', result.stdout)

    def test_missing_store_mask_fails(self):
        result = self.run_rule('tl.{fn} mask presence mismatch',
                               '  x = tl.load(X, mask=mask, other=0).to(tl.float32)\n  tl.store(Y, x)')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('tl.store mask presence mismatch', result.stdout)


if __name__ == '__main__':
    unittest.main()

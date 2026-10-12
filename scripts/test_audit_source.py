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
    def test_store_whitespace_and_nested_values(self):
        for py_ws in ['', ' ', '\n ']:
            for lean_ws in ['', ' ', '\n ']:
                py = 'tl.store' + py_ws + '(out + row, a * (b + c), mask=active)'
                lean = 'tl.region out = $storage\ntl.store' + lean_ws + '(out + row, (a * (b + c)).to($storage), mask=active)'
                self.assertEqual(source.store_calls_with_explicit_destination_casts(py, lean),
                                 ['tl.store(out + row, (a * (b + c)).to($storage), mask=active)'])

    def test_store_whitespace_preserves_pointer_value_and_dtype(self):
        py = 'tl.store (out + row, value, mask=active)'
        lean = 'tl.region out = $storage\ntl.store (out + row, value.to($storage), mask=active)'
        for changed in [lean.replace('out + row', 'out + other'),
                        lean.replace('value.to', 'other.to'),
                        lean.replace('to($storage)', 'to(tl.float32)')]:
            self.assertEqual(source.store_calls_with_explicit_destination_casts(py, changed), [py])

    def test_load_default_splice_removes_only_surrounding_whitespace(self):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        start = script.index('def normalize_other(value: str) -> str:')
        end = script.index('\ndef other_values(', start)
        namespace = {'re': re}
        exec(script[start:end], namespace)
        normalize = namespace['normalize_other']
        for value in ['vob_end_id', '$(vob_end_id)', '$((vob_end_id : Int))',
                      '  $((vob_end_id  : Int))  ']:
            self.assertEqual(normalize(value), 'vob_end_id')
        for value in ['$((vob_start_id : Int))', '$((vob_end_id + 1 : Int))',
                      '$((0 : Int))']:
            self.assertNotEqual(normalize(value), 'vob_end_id')

    def test_store_parameter_offsets_and_compound_values(self):
        normalize = source.store_calls_with_explicit_destination_casts
        py = 'tl.store(out + block * rows + row, a * b, mask=active)'
        lean = 'tl.region out = $storage\ntl.store(out + block * $(rows) + row, (a * b).to($storage), mask=active)'
        self.assertEqual(normalize(py, lean),
                         ['tl.store(out + block * rows + row, (a * b).to($storage), mask=active)'])
        for changed in [lean.replace('$(rows)', '$(other_rows)'),
                        lean.replace('(a * b)', '(a + b)'),
                        lean.replace('.to($storage)', '.to(tl.float32)')]:
            self.assertEqual(normalize(py, changed), [py])
        py2 = 'tl.store(out, a * (b + c))'
        lean2 = 'tl.region out = tl.float32\ntl.store(out, (a * b + c).to(tl.float32))'
        self.assertEqual(normalize(py2, lean2), [py2])

    def test_self_advanced_store_pointer_retains_only_declared_type(self):
        normalize = source.store_calls_with_explicit_destination_casts
        call = 'tl.store(out + off, value, mask=active)'
        py = 'out = out + row * stride\n' + call
        lean = 'tl.region out = $storage\ntl.store(out + off, value.to($storage), mask=active)'
        self.assertEqual(normalize(py, lean),
                         ['tl.store(out + off, value.to($storage), mask=active)'])
        self.assertEqual(normalize(py, lean.replace('tl.region out = $storage', '')), [call])
        self.assertEqual(normalize('out = other + off\n' + py, lean), [call])
        self.assertEqual(normalize('other = out + off\nout = other + off\n' + call, lean), [call])

    def test_scalar_parameter_splices_keep_names_values_and_operations(self):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        start = script.index('def normalize_call(call: str) -> str:')
        end = script.index('\nfailures = []', start)
        namespace = {'re': re}
        exec(script[start:end], namespace)
        normalize = namespace['normalize_call']
        for ty in ['ℝ', 'Nat', 'Int']:
            self.assertEqual(normalize(f'tl.where(mask, $((alpha : {ty})), value)'),
                             normalize('tl.where(mask, alpha, value)'))
        base = normalize('tl.where(mask, alpha, value)')
        for value in ['$((beta : ℝ))', '$((alpha + 1 : ℝ))', '$((1 : ℝ))',
                      '$((alpha : SomeType))', 'alpha.to(tl.float16)']:
            self.assertNotEqual(base, normalize(f'tl.where(mask, {value}, value)'))

    def test_store_offset_preserves_destination_cast(self):
        normalize = source.store_calls_with_explicit_destination_casts
        for ptr in ['b + offset', '(b + offset)', 'b + row * stride + offset', 'b - offset']:
            py = f'tl.store({ptr}, value, mask=active)'
            lean = f'tl.region b = $storage\ntl.store({ptr}, value.to($storage), mask=active)'
            self.assertEqual(normalize(py, lean),
                             [f'tl.store({ptr}, value.to($storage), mask=active)'])
            self.assertEqual(normalize(py, lean.replace('value.to($storage)', 'value.to(tl.float32)')),
                             [py])
            self.assertEqual(normalize(py, lean.replace('value.to($storage)', 'other.to($storage)')),
                             [py])

    def test_store_offset_does_not_hide_unknown_roots_or_address_changes(self):
        normalize = source.store_calls_with_explicit_destination_casts
        for ptr in ['missing + offset', 'b * offset', 'f(b) + offset']:
            py = f'tl.store({ptr}, value, mask=active)'
            lean = f'tl.region b = $storage\ntl.store({ptr}, value.to($storage), mask=active)'
            self.assertEqual(normalize(py, lean), [py])
        py = 'tl.store(b + offset, value, mask=active)'
        lean = 'tl.region b = $storage\ntl.store(b + other_offset, value.to($storage), mask=active)'
        self.assertEqual(normalize(py, lean), [py])

    def test_block_load_padding_distinguishes_omission_from_zero(self):
        text = '''import triton.language as lang
@triton.jit
def kernel(p, axes, padding):
    a = lang.load(p, boundary_check=(0, 1))
    b = lang.load(p, boundary_check=(0,), padding_option="zero")
    c = lang.load(p, boundary_check=(0,), padding_option="")
    d = lang.load(p, boundary_check=(0,), padding_option="nan")
    e = lang.load(p, boundary_check=axes, padding_option=padding)
    f = lang.load(p, boundary_check=())
    g = lang.load(p, mask=True, other=0)
    h = something.load(p, boundary_check=(0,))
def host(p):
    return lang.load(p, boundary_check=(0,))
'''
        sites = source.python_block_load_padding(text)
        self.assertEqual([s['padding'] for s in sites],
                         ['unspecified', 'zero', 'unspecified', 'nan', 'dynamic_or_invalid'])
        self.assertEqual([s['line'] for s in sites], [4, 5, 6, 7, 8])
        self.assertTrue(all(s['kernel'] == 'kernel' for s in sites))
        self.assertEqual(sites[-1]['boundary_check'], 'axes')

    def test_original_statement_macro_handles_nested_calls_and_both_arm_orders(self):
        for tag, yes, no in [
            ('sumLeft', 'b_z = tl.sum(b_s, 0) + b_z', 'b_z += tl.sum(b_s, 0)'),
            ('original', 'b_z += tl.sum(b_s, 0)', 'b_z = tl.sum(b_s, 0) + b_z'),
        ]:
            text = f'''let update : TSyntax `tritonStmt ←
  if version.getId == `{tag} then `(tritonStmt| {yes})
  else `(tritonStmt| {no})
`(fun (s z : RegionName) (T BT : Nat) =>
  ComputeKernel.mk [s] [z] (triton {{
    if i_t >= $(0) {{ $update }}
  }}).surfaceBody)'''
            body = source.lean_first_triton_body(text)
            self.assertIn('b_z += tl.sum(b_s, 0)', body)
            self.assertNotIn('$update', body)
            self.assertNotIn('b_z = tl.sum', body)
            self.assertEqual(source.lean_triton_bodies(text), [body])
            self.assertIn('(T BT : Nat)', source.quoted_kernel_signature(text))
        broken = text.replace('else `(tritonStmt|', 'else unsupported (')
        with self.assertRaises(ValueError):
            source.lean_first_triton_body(broken)

    def test_region_continuations_do_not_become_statements(self):
        body = "tl.region x = DTYPE,\n  y = OUT_DTYPE\nz = tl.load(x)\ny += z"
        stripped = source.strip_region_declarations(body)
        self.assertNotIn('y = OUT_DTYPE', stripped)
        self.assertIn('z = tl.load(x)', stripped)
        self.assertIn('y += z', stripped)

    def test_preamble_supports_explicit_source_headers(self):
        self.assertEqual(source.target_kernel_name(
            '/-! Source: vector.py, vector_kernel. -/', {'vector_kernel'}), 'vector_kernel')
        self.assertEqual(source.target_kernel_name(
            '/-! Complete typed source for `backward` in\n`backward.py`. -/', {'backward'}), 'backward')

    def test_dtype_resolution_preserves_literal_and_wrong_precision_mismatches(self):
        lean = '''tl.region x = DTYPE, y = OUT_DTYPE
p_x = tl.make_block_ptr(base=x+base, shape=(T), strides=(1))
p_y = tl.make_block_ptr(base=y+base, shape=(T), strides=(1))
b_x = tl.load(p_x)
promoted = tl.load(p_x).to(tl.float32)
'''
        resolve = lambda text: source.resolve_source_dtype_attributes(text, lean)
        self.assertEqual(resolve('x.to(p_y.dtype.element_ty)'), 'x.to(OUT_DTYPE)')
        self.assertEqual(resolve('x.to(b_x.dtype)'), 'x.to(DTYPE)')
        self.assertNotEqual(resolve('x.to(b_x.dtype)'), 'x.to(OUT_DTYPE)')
        self.assertEqual(resolve('x.to(tl.float16)'), 'x.to(tl.float16)')
        self.assertEqual(resolve('x.to(missing.dtype)'), 'x.to(missing.dtype)')
        self.assertEqual(resolve('x.to(promoted.dtype)'), 'x.to(promoted.dtype)')
        ambiguous = lean + '\np_x = tl.make_block_ptr(base=y+base)\n'
        self.assertEqual(source.resolve_source_dtype_attributes('x.to(p_x.dtype.element_ty)', ambiguous),
                         'x.to(p_x.dtype.element_ty)')


    def test_implicit_store_cast_requires_the_actual_destination_dtype(self):
        python = 'out_ptrs = Out + off_o\ntl.store(out_ptrs, acc, mask=active)'
        lean = 'tl.region Out = $storage\ntl.store(out_ptrs, acc.to($storage), mask=active)'
        normalize = source.store_calls_with_explicit_destination_casts
        self.assertEqual(normalize(python, lean),
                         ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        self.assertEqual(normalize(python, lean.replace('acc.to($storage)', '(acc).to($storage:tritonDType)')),
                         ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        for invalid in [lean.replace('$storage),', 'tl.float32),'),
                        lean.replace('acc.to', 'other.to'),
                        lean.replace('out_ptrs,', 'different,')]:
            self.assertEqual(normalize(python, invalid), ['tl.store(out_ptrs, acc, mask=active)'])
        for ambiguous in [python.replace('tl.store', 'out_ptrs = Other + off_o\ntl.store'),
                          python.replace('tl.store', 'out_ptrs *= stride\ntl.store'),
                          python.replace('Out + off_o', 'choose(Out, Other)')]:
            self.assertEqual(normalize(ambiguous, lean), ['tl.store(out_ptrs, acc, mask=active)'])
        for advance in ['+=', '-=']:
            advanced = python.replace('tl.store', f'out_ptrs {advance} stride\ntl.store')
            self.assertEqual(normalize(advanced, lean),
                             ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        masked = lean.replace('mask=active', 'mask=wrong')
        self.assertIn('mask=active', normalize(python, masked)[0])
        self.assertNotIn('mask=wrong', normalize(python, masked)[0])

    def test_loaded_pointer_alias_dtype_requires_unique_bindings(self):
        lean = '''tl.region V = $storage, W = tl.float32
v_ptrs = V + off_v
next_ptrs = v_ptrs + step
v = tl.load(next_ptrs + offset, mask=active, other=0.0)
'''
        resolve = lambda text: source.resolve_source_dtype_attributes('p.to(v.dtype)', text)
        self.assertEqual(resolve(lean), 'p.to($storage)')
        for extra in ['v_ptrs = W + other', 'v_ptrs *= stride',
                      'v_ptrs = choose(V, W)', 'v_ptrs = next_ptrs + stride']:
            self.assertEqual(resolve(lean + extra + '\n'), 'p.to(v.dtype)')
        for advance in ['+=', '-=']:
            self.assertEqual(resolve(lean + f'v_ptrs {advance} stride\n'), 'p.to($storage)')
        self.assertNotEqual(resolve(lean), 'p.to(tl.float32)')

    def test_cast_sequence_keeps_macro_types_and_missing_casts(self):
        self.assertEqual(source.cast_sequence('p.to($storage:tritonDType)'), ['$storage'])
        self.assertEqual(source.cast_sequence('p.to($storage)'), ['$storage'])
        self.assertNotEqual(source.cast_sequence('p.to($storage)'), source.cast_sequence('p'))
        self.assertNotEqual(source.cast_sequence('p.to($storage)'), source.cast_sequence('p.to($other)'))
        self.assertNotEqual(source.cast_sequence('p.to(tl.float16)'), source.cast_sequence('p.to(tl.float32)'))

    def test_store_splice_category_keeps_the_selected_dtype(self):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        start = script.index('def normalize_call(call: str) -> str:')
        end = script.index('\nfailures = []', start)
        namespace = {'re': re}
        exec(script[start:end], namespace)
        normalize = namespace['normalize_call']
        plain = normalize('tl.store(p, v.to($valueTy))')
        typed = normalize('tl.store(p, v.to($valueTy:tritonDType))')
        self.assertEqual(plain, typed)
        self.assertNotEqual(typed, normalize('tl.store(p, v.to($finalTy:tritonDType))'))
        self.assertNotEqual(typed, normalize('tl.store(p, v.to(tl.float16))'))
        self.assertNotEqual(normalize('tl.store(p, v.to(tl.float16))'),
                            normalize('tl.store(p, v.to(tl.float32))'))

    def test_true_equality_is_not_a_bare_placeholder(self):
        for text in ['theorem empty : True := by trivial', '    True := by']:
            self.assertTrue(source.bare_true_proof(text))
        for text in ['    p = True := by simp', '    p ↔ True := by simp',
                     '    p = True := by rw [safety_definition]']:
            self.assertFalse(source.bare_true_proof(text))

    def test_named_jit_cast_does_not_include_other_kernels_or_host(self):
        text = """@triton.jit
def helper(X):
    x = tl.load(X).to(tl.float32)

@triton.jit
def target(X):
    x = tl.load(X)

def host(X):
    return X.to(tl.float32)
"""
        self.assertNotIn('.to(tl.float32)', source.python_named_jit_body(text, 'target'))
        self.assertIn('.to(tl.float32)', source.python_named_jit_body(text, 'helper'))
        with self.assertRaises(ValueError):
            source.python_named_jit_body(text, 'host')

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


class PointerAdvanceTests(unittest.TestCase):
    def test_pointer_advances_keep_storage_dtype(self):
        lean='tl.region out = tl.float16\np = out + off\np += step\np -= step\ntl.store(p, value.to(tl.float16))'
        py='p = out + off\np += step\np -= step\ntl.store(p, value)'
        self.assertEqual(source.resolve_source_dtype_attributes('value.to(p.dtype.element_ty)',lean),'value.to(tl.float16)')
        self.assertEqual(source.store_calls_with_explicit_destination_casts(py,lean),['tl.store(p, value.to(tl.float16))'])
    def test_ambiguous_or_unknown_pointer_stays_unresolved(self):
        for bindings in ['p = out + off\np = other + off', 'p = out + off\np *= step', 'p = out + off\np /= step', 'p += step','p = missing + off\np -= step']:
            lean='tl.region out = tl.float16, other = tl.bfloat16\n'+bindings+'\np += step\ntl.store(p, value.to(tl.float16))'
            py=bindings+'\np += step\ntl.store(p, value)'
            self.assertEqual(source.resolve_source_dtype_attributes('value.to(p.dtype.element_ty)',lean),'value.to(p.dtype.element_ty)',bindings)
            self.assertEqual(source.store_calls_with_explicit_destination_casts(py,lean),['tl.store(p, value)'],bindings)
    def test_advances_do_not_hide_changed_store(self):
        py='p = out + off\np += step\ntl.store(p, value, mask=active)'
        for call in ['tl.store(p, value.to(tl.float32), mask=active)','tl.store(p, other.to(tl.float16), mask=active)','tl.store(p + different, value.to(tl.float16), mask=active)','tl.store(p, value.to(tl.float16), mask=other)']:
            lean='tl.region out = tl.float16\np = out + off\np += step\n'+call
            self.assertNotEqual(source.store_calls_with_explicit_destination_casts(py,lean),[call])

if __name__ == '__main__':
    unittest.main()

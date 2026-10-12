"""Regressions for scope evidence and fail-closed review freshness."""
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
sys.path.insert(0, str(ROOT / 'scripts'))
import coverage_review as coverage
from spec_sheet import split_statement, split_decls, body_of, hypotheses, make_sheet
from record_coverage_ci import completed_record


@dataclass
class Row:
    file: str
    declaration: str
    coverage_level: str = 'unreviewed'
    blocker_family: str = 'none'
    issue: str = ''
    evidence: str = ''


class CoverageReviewTests(unittest.TestCase):
    def test_presentation_discovery_ignores_prose_and_literal_declarations(self):
        source = ('/- outer /- nested -/\nspecification is not code\n-/\n'
                  'def marker : String := "\nspecification fake : False\n"\n'
                  'def raw : String := r#"/-"#\n'
                  '/-- A real claim. -/\n'
                  'specification result : True := proof\n')
        decls = split_decls(source)
        self.assertEqual([d['name'] for d in decls], ['marker', 'raw', 'result'])
        result = decls[-1]
        self.assertEqual(result['start'], source.index('specification result'))
        self.assertEqual(result['doc'], '/-- A real claim. -/')

    def test_next_section_prose_does_not_leak_into_a_spec_definition(self):
        source = ('def expected : Nat := 1\n\n'
                  '/-- Supporting proofs.\nlemma takes symbolic arguments. -/\n'
                  'section Helpers\nvariable (n : Nat)\n'
                  'theorem helper : n = n := by rfl\nend Helpers\n')
        defs = split_decls(source)
        self.assertEqual(defs[0]['text'], 'def expected : Nat := 1')
        self.assertEqual([d['name'] for d in defs], ['expected', 'helper'])

    def test_split_sheet_keeps_module_preambles_out_of_kernel_bodies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'Correct.lean'
            source.with_name('Kernels.lean').write_text(
                'namespace Example\ndef kernel : Nat := 1\nend Example\n')
            source.write_text('import Example.Kernels\nimport Example.Proofs.Real\n'
                              'namespace Example\n'
                              'specification kernel_correctness : kernel = 1 := by rfl\n'
                              'end Example\n')
            sheet, stats = make_sheet(str(source), {})
            self.assertEqual(stats['headline'], 1)
            self.assertIn('def kernel : Nat := 1\n```', sheet)
            self.assertNotIn('import Example.', sheet)
            self.assertNotIn('end Example', sheet)

    def fixture(self, root):
        directory = root / 'bench/tritonbench_g/example'
        directory.mkdir(parents=True)
        lean = directory / 'Example.lean'
        lean.write_text('specification result : True := by trivial\n')
        python = directory / 'example.py'
        python.write_text('def kernel(x):\n    return x\n')
        row = Row(lean.relative_to(root).as_posix(), 'result')
        record = {'schema': 1, 'reviewed_on': '2026-09-24', 'basis': 'Statement review',
                  'ports': [{'file': row.file, 'python_file': python.relative_to(root).as_posix(),
                             'lean_sha256': coverage.digest(lean),
                             'python_sha256': coverage.digest(python),
                             'declarations': [{'declaration': 'result',
                                               'coverage_level': 'specialization',
                                               'numeric_model': 'mathematical',
                                               'scope': 'One selected model.',
                                               'python_functions': [{'name': 'kernel', 'line': 1}]}]}]}
        path = root / 'reviews.json'
        path.write_text(json.dumps(record))
        return row, record, path

    def test_source_changes_invalidate_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, record, path = self.fixture(root)
            coverage.load_reviews([row], root, path)
            for field in ('file', 'python_file'):
                target = root / record['ports'][0][field]
                original = target.read_text()
                target.write_text(original + '\n')
                with self.assertRaisesRegex(ValueError, 'review is stale'):
                    coverage.load_reviews([row], root, path)
                target.write_text(original)

    def test_missing_extra_and_duplicate_reviews_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, record, path = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'Missing coverage'):
                coverage.load_reviews([row, Row(row.file, 'new_result')], root, path)
            for name in ('result', 'unknown'):
                changed = copy.deepcopy(record)
                entry = copy.deepcopy(changed['ports'][0]['declarations'][0])
                entry['declaration'] = name
                changed['ports'][0]['declarations'].append(entry)
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'Unexpected/duplicate'):
                    coverage.load_reviews([row], root, path)

    def test_split_port_shared_source_and_proof_changes_invalidate_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, record, path = self.fixture(root)
            old = root / row.file
            correct = old.with_name('Correct.lean')
            old.rename(correct)
            row.file = correct.relative_to(root).as_posix()
            port = record['ports'][0]
            port['file'] = row.file
            (correct.parent / 'Proofs').mkdir()
            support = [correct.with_name('Kernels.lean'), correct.with_name('FPEquiv.lean'),
                       correct.parent / 'Proofs/Real.lean']
            for file in support:
                file.write_text('-- fixture\n')
            port['support_sha256'] = {p.relative_to(root).as_posix(): coverage.digest(p)
                                      for p in support}
            path.write_text(json.dumps(record))
            coverage.load_reviews([row], root, path)
            for file in support:
                original = file.read_text()
                file.write_text(original + '-- changed\n')
                with self.assertRaisesRegex(ValueError, 'shared source/proofs'):
                    coverage.load_reviews([row], root, path)
                file.write_text(original)

    def test_python_links_must_resolve_to_exact_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, record, path = self.fixture(root)
            record['ports'][0]['declarations'][0]['python_functions'][0]['line'] = 2
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Unresolved Python'):
                coverage.load_reviews([row], root, path)

    def test_review_cannot_upgrade_explicit_source_slice(self):
        row = Row('example.lean', 'result', 'precomputed_input_slice')
        entries = {(row.file, row.declaration): {'coverage_level': 'full_value_candidate'}}
        with patch.object(coverage, 'load_reviews', return_value=({}, entries)):
            with self.assertRaisesRegex(ValueError, 'conflicts with source'):
                coverage.apply_reviews([row])

    def test_named_arguments_and_defaults_do_not_cut_off_statement(self):
        source = ('specification result (n : Nat := 3) :\n'
                  '  Realizes (kernel := chosen n) (expected := fun i => i) := proof\n')
        statement, proof = split_statement(source)
        self.assertIn('(expected := fun i => i)', statement)
        self.assertEqual(proof.strip(), ':= proof')

    def test_let_values_and_comments_stay_in_statement(self):
        source = ('theorem result : (/- := by /- nested -/ -/ True) ∧ '
                  '("test := by" = "test := by") ∧\n'
                  '  (fun x => let y := x; y) = id := by simp\n')
        statement, proof = split_statement(source)
        self.assertIn('let y := x; y', statement)
        self.assertEqual(proof.strip(), ':= by simp')
        statement, proof = split_statement('specification result : io ⊨ fun xs => let y := xs; y := proof')
        self.assertTrue(statement.endswith('let y := xs; y'))
        self.assertEqual(proof, ':= proof')

    def test_where_definition_is_not_a_named_kernel_argument(self):
        source = 'def signature (n : Nat := 3) : KernelIO where\n  kernel := chosen n\n  B := n'
        header, value = split_statement(source)
        self.assertTrue(header.endswith(': KernelIO'))
        self.assertTrue(value.startswith('where'))
        self.assertIn('kernel := chosen n', body_of({'kind': 'def', 'text': source}))

    def test_lambdas_and_default_values_are_not_hypotheses(self):
        statement = ('specification result (n : Nat := 3) (h₀ : 0 < n) : '
                     'Realizes (write := (fun idx : Nat => idx))')
        self.assertEqual(hypotheses(statement), ['h₀ : 0 < n'])

    def test_missing_headline_keeps_index_statistics_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'NoHeadline.lean'
            source.write_text('def value : Nat := 1\n')
            sheet, stats = make_sheet(str(source), {})
            self.assertIn('public spec not located', sheet)
            self.assertEqual(stats['headline'], 0)
            for field in ('score', 'flat_reads', 'stmt_lines', 'hyps'):
                self.assertEqual(stats[field], 0)

    def test_only_completed_successful_corpus_runs_are_evidence(self):
        run = {'status': 'completed', 'conclusion': 'success',
               'path': '.github/workflows/bench-audit.yml', 'head_sha': 'a' * 40,
               'html_url': 'https://github.com/Lizn-zn/VeriTile/actions/runs/123'}
        job = {'status': 'completed', 'conclusion': 'success',
               'completed_at': '2026-09-24T06:13:56Z'}
        self.assertEqual(completed_record(123, run, [job])['commit'], 'a' * 40)
        for field, value in [('status', 'in_progress'), ('conclusion', 'failure'),
                             ('path', '.github/workflows/site.yml')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                completed_record(123, {**run, field: value}, [job])
        for jobs in [[], [{**job, 'conclusion': 'skipped'}],
                     [{**job, 'status': 'queued', 'completed_at': None}]]:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                completed_record(123, run, jobs)


if __name__ == '__main__':
    unittest.main()

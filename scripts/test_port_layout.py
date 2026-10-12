"""Split ports must not lose sources, FP entries or nested proof audit targets."""
from pathlib import Path
import tempfile
import unittest

from bench.port_layout import source_file, source_files, proof_files, specification_files


class PortLayoutTests(unittest.TestCase):
    def test_mixed_corpus_selects_both_contracts_and_nested_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, split = root / 'legacy', root / 'split'
            legacy.mkdir()
            (legacy / 'Legacy.lean').touch()
            (split / 'Proofs').mkdir(parents=True)
            for name in ('Kernels.lean', 'Correct.lean', 'FPEquiv.lean', 'Proofs/Real.lean'):
                (split / name).touch()
            self.assertEqual(source_file(split), split / 'Kernels.lean')
            self.assertEqual(len(source_files(root)), 2)
            self.assertEqual(set(specification_files(root)), {
                legacy / 'Legacy.lean', split / 'Correct.lean', split / 'FPEquiv.lean'})
            self.assertEqual(len(proof_files(split)), 4)
            self.assertIn(split / 'Proofs/Real.lean', proof_files(split))

    def test_missing_or_ambiguous_entry_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(source_file(root))
            (root / 'Kernels.lean').touch()
            with self.assertRaisesRegex(ValueError, 'Missing split-port entry'):
                source_file(root)
            for name in ('Correct.lean', 'FPEquiv.lean', 'Old.lean'):
                (root / name).touch()
            with self.assertRaisesRegex(ValueError, 'Unexpected split-port'):
                source_file(root)
            (root / 'Kernels.lean').unlink()
            with self.assertRaisesRegex(ValueError, 'Ambiguous legacy'):
                source_file(root)


if __name__ == '__main__':
    unittest.main()

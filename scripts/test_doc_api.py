"""Regression checks for documentation API extraction and Lean resolution."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('doc_api', ROOT / 'site/scripts/check-doc-api.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class DocApiTests(unittest.TestCase):
    def test_io_families_and_unicode_arities(self):
        names = {'KernelIO₁.Implements', 'KernelIO₂.Implements', 'KernelIO₃.Equiv',
                 'Masked2DKernelIO₁.Implements', 'KernelIO₃ₓ₂.Implements',
                 'StreamMetaGatherMasked3DKernelIO₃.ImplementsR',
                 'UKernelIO.Implements', 'KernelIO₉.missing',
                 'Masked2DKernelIO₁.not_a_real_lemma'}
        self.assertEqual(api.extract_names(' '.join(f'`{n}`' for n in names)), names)

    def test_schematic_exclusions(self):
        text = ('KernelIO₁.lean KernelIO₃.md ComputeKernel.atomic_add '
                'KernelIO₂.scatter_readback_* KernelIO₂.Implements{R} '
                '`forLoop_invariant` KernelIO₁.Implements')
        self.assertEqual(api.extract_names(text), {'forLoop_invariant', 'KernelIO₁.Implements'})

    def test_missing_io_lemma_is_rejected_by_lean(self):
        names = api.extract_names('`KernelIO₁.veritile_missing_doc_api` '
                                  '`Masked2DKernelIO₁.veritile_missing_doc_api`')
        self.assertEqual(len(names), 2)
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'DocApi.lean'
                path.write_text('import VeriTile.Triton.Memory.KernelSpec\n'
                                'open VeriTile.Triton\n#check ' + name + '\n')
                result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stdout)


if __name__ == '__main__':
    unittest.main()

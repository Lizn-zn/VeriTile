#!/usr/bin/env python3
"""Ask Lean to resolve public API references in the English documentation.

Run after lake build. This checks symbol existence, not prose or theorem scope.
"""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = (
    'ComputeCorrect', 'ComputeRefine', 'ComputeKernel', 'Kernel',
    'RoundingModel', 'BlockState', 'BlockPtr', 'BlockPtrSummary',
    'WriteFootprint', 'TensorView', 'ComparableDType', 'Tile', 'Offset',
)
# A schematic diagram and a theorem defined in a standalone bench file, which
# cannot be imported through the library prelude.
ILLUSTRATIONS = {'ComputeKernel.atomic_add', 'ComparableDType.real_gt_some_some_eq_true_iff'}
# Cover the entire IO family, including Unicode arities and shaped/masked/
# streamed variants. Deliberately also match unknown variants so typos fail.
PATTERN = re.compile(r'\b(?:' + '|'.join(PREFIXES) + r'|[A-Za-z_]*\w*KernelIO\w*|KernelIO\w*)\.[A-Za-z_][\w?\']*(?:\.[A-Za-z_][\w?\']*)*')

def extract_names(text):
    names = set()
    for match in PATTERN.finditer(text):
        name = match[0]
        # Ignore explicit glob families such as scatter_readback_*.
        if text[match.end():].startswith(('*', '{')) or name.endswith(('.lean', '.md')) or name in ILLUSTRATIONS:
            continue
        names.add(name)
    names.update(re.findall(r'`(for(?:Loop|Range)(?:Aux|Dyn)?_[A-Za-z_]+)`', text))
    return names

def main():
    names = set()
    pages = [p for p in (ROOT / 'documents').glob('*.md') if not p.stem.endswith('_zh')]
    pages.append(ROOT / 'README.md')
    pages += list((ROOT / 'site/src/content/docs/cookbook').glob('*.md'))
    for page in pages:
        names.update(extract_names(page.read_text()))
    program = 'import VeriTile.Triton\nimport VeriTile.Triton.Memory.KernelSpec\nopen VeriTile.Triton\n'
    program += '\n'.join('#check ' + name for name in sorted(names)) + '\n'
    with tempfile.TemporaryDirectory(prefix='veritile-doc-api-') as temp:
        path = Path(temp) / 'DocApi.lean'
        path.write_text(program)
        result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        return result.returncode
    print(f'{len(names)} documented public API references resolved by Lean.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

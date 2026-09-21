#!/usr/bin/env python3
"""Record real Lean checks for the homepage's two fixed VectorAdd variants.

Run after `lake build`. --check only checks freshness; it does not run Lean.
The intentionally failing variant is written to a temporary directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'bench/examples/VectorAdd.lean'
OUTPUT = ROOT / 'site/src/lib/vector-add-record.json'


def fingerprint():
    paths = [ROOT / p for p in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json',
                               'site/scripts/record-home-demo.py', 'bench/examples/VectorAdd.lean')]
    paths += list((ROOT / 'VeriTile').rglob('*.lean'))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stamp = fingerprint()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text())['fingerprint'] != stamp:
            raise SystemExit('Homepage Lean records are stale. Run lake build, then python3 site/scripts/record-home-demo.py.')
        print('Homepage Lean records match their source, library, toolchain, and recorder.')
        return

    lake = os.environ.get('VERITILE_LAKE', 'lake')
    lean = os.environ.get('VERITILE_LEAN', 'lean')
    text = SOURCE.read_text()
    original_line = '  out  := x + y'
    assert text.count(original_line) == 1
    kernel = re.search(r'def addKernel [^\n]* := (triton \{[\s\S]*?\n\})', text)[1]
    x, y = [1, 2, 3, 4], [4, 5, 6, 7]
    records = []

    def run(directory, filename, content, expected_code):
        path = directory / filename
        path.write_text(content)
        result = subprocess.run([lake, 'env', lean, str(path)], cwd=ROOT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        output = result.stdout.replace(str(directory) + '/', '')
        if result.returncode != expected_code:
            raise SystemExit(f'Unexpected result for {filename}: {result.returncode}\n{output}')
        print(f'{filename}: exit {result.returncode}', flush=True)
        return output

    with tempfile.TemporaryDirectory(prefix='veritile-home-demo-') as directory:
        directory = Path(directory)
        for variant, operator in [('add', '+'), ('subtract', '-')]:
            changed = text.replace(original_line, f'  out  := x {operator} y')
            output = run(directory, f'VectorAdd-{variant}.lean', changed, 0 if variant == 'add' else 1)
            if variant == 'subtract' and 'unsolved goals' not in output:
                raise SystemExit('The changed kernel must fail with a proof obligation, not an unrelated compiler error.')
            if variant == 'add' and ('sorry' in output or 'error:' in output):
                raise SystemExit('The original kernel must pass without proof placeholders.')

            # Prove that the displayed four values follow from the actual DSL
            # semantics for both operators. This is separate from the general
            # addition contract, which the subtraction variant fails to prove.
            actual = [a + b if operator == '+' else a - b for a, b in zip(x, y)]
            sample = changed[:changed.index('/-- A scatter-store')]
            if variant == 'subtract':
                sample = sample.replace('xs i + ys i', 'xs i - ys i')
                sample = sample.replace('NumericDType.add, NumericDType.mul',
                                        'NumericDType.add, NumericDType.sub, NumericDType.mul')
            vector = lambda values: '![' + ', '.join(map(str, values)) + ']'
            sample += f'''\n/-- The four values displayed by the homepage, checked against the DSL. -/
theorem displayed_values (s : BlockState)
    (hx : InputLoadedAt s ⟨"x"⟩ 4 (fun i => ({vector(x)} : Fin 4 → ℝ) i))
    (hy : InputLoadedAt s ⟨"y"⟩ 4 (fun i => ({vector(y)} : Fin 4 → ℝ) i)) :
    ∀ i : Fin 4,
      observeAt (exec (addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ 4) s) ⟨"out"⟩ 4 s.pid i
        = some (({vector(actual)} : Fin 4 → ℝ) i) := by
  intro i
  rw [add_kernel_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ 4 (by decide) s _ _ hx hy]
  fin_cases i <;> norm_num

#axiomsClean displayed_values
end VeriTile.Bench.Examples.VectorAdd
'''
            sample_output = run(directory, f'Sample-{variant}.lean', sample, 0)
            records.append({'id': variant, 'operator': operator, 'actual': actual,
                            'exitCode': 0 if variant == 'add' else 1,
                            'kernel': kernel.replace(original_line, f'  out  := x {operator} y'),
                            'diagnostics': output.strip(), 'sampleDiagnostics': sample_output.strip()})

    version = subprocess.check_output([lake, 'env', lean, '--version'], cwd=ROOT, text=True).strip()
    OUTPUT.write_text(json.dumps({'fingerprint': stamp, 'leanVersion': version,
                                 'source': 'bench/examples/VectorAdd.lean',
                                 'x': x, 'y': y, 'expected': [a + b for a, b in zip(x, y)],
                                 'variants': records}, indent=2, ensure_ascii=False) + '\n')
    print(f'Recorded both proof checks and both sample-value proofs in {OUTPUT.relative_to(ROOT)}.')


if __name__ == '__main__':
    main()

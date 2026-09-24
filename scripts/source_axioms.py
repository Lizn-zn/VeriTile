#!/usr/bin/env python3
"""List actual project axiom declarations from compiled Lean environments."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

INVENTORY = r'''
open Lean Elab Command in
elab "#sourceAxioms " "[" modules:ident,* "]" : command => do
  let wanted := modules.getElems.map Syntax.getId
  let env ← getEnv
  let mut rows : Array String := #[]
  for (name, info) in env.constants.toList do
    let .axiomInfo _ := info | continue
    let some idx := env.getModuleIdxFor? name | continue
    let origin := env.header.modules[idx.toNat]!.module
    unless wanted.contains origin do continue
    let file := String.intercalate "/" (origin.components.map Name.toString) ++ ".lean"
    let userName := ((privateToUserName? name).getD name).eraseMacroScopes
    rows := rows.push s!"{file}:{userName.getString!}"
  for row in rows.qsort (· < ·) do
    liftIO <| IO.println row
'''


def main() -> int:
    source_root = Path(sys.argv[1]).resolve()
    paths = sorted(source_root.rglob('*.lean'))
    if not paths:
        print(f'No Lean sources in {source_root}', file=sys.stderr)
        return 1
    modules = ['.'.join(p.relative_to(source_root.parent).with_suffix('').parts) for p in paths]
    env = dict(os.environ)
    command = ['lake', 'env', 'lean']
    if source_root == ROOT / 'VeriTile':
        # Include every source module, even a file outside the default targets.
        # Lake validates source hashes before we inspect the compiled objects.
        built = subprocess.run(['lake', 'build', *modules], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if built.returncode:
            print(built.stdout, file=sys.stderr)
            return built.returncode
    else:
        # Isolated, precompiled fixture modules used by regression tests.
        lean_path = subprocess.check_output(['lake', 'env', 'printenv', 'LEAN_PATH'],
                                            cwd=ROOT, text=True).strip()
        # Set this after `lake env`, which otherwise prepends the real project's
        # VeriTile directory and shadows fixture modules in the same namespace.
        command = ['lake', 'env', 'env',
                   'LEAN_PATH=' + str(source_root.parent) + os.pathsep + lean_path, 'lean']
    program = 'import Lean\n' + ''.join(f'import {module}\n' for module in modules)
    program += INVENTORY + '\n#sourceAxioms [' + ', '.join(modules) + ']\n'
    with tempfile.TemporaryDirectory(prefix='veritile-source-axioms-') as temp:
        path = Path(temp) / 'Inventory.lean'
        path.write_text(program)
        checked = subprocess.run([*command, str(path)], cwd=ROOT, env=env,
                                 text=True, capture_output=True)
    if checked.returncode:
        print(checked.stdout + checked.stderr, file=sys.stderr)
        return checked.returncode
    print(checked.stdout, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

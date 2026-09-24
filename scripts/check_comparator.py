#!/usr/bin/env python3
"""Replay checked-in proof targets with the official comparator.

Repository audits freeze the current trusted sources and replay their proofs;
they do not compare against a historical Git revision. prove.py separately
compares an agent's candidate against the pre-agent challenge.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from comparator_common import comparator_command, require_tools, snapshot_project, write_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from audit_trust_prep import prepare_source

# Alias actual Lean names instead of trying to parse private, quoted, or
# hygienic names through comparator's dotted-string config format.
INVENTORY = r'''
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let selected : Array Name := __SELECTED__
  let mut names : Array Name := #[]
  if selected.isEmpty then
    for (name, info) in env.constants.toList do
      if (env.getModuleIdxFor? name).isSome then continue
      if let .thmInfo _ := info then names := names.push name
  else
    names := selected
  names := names.qsort (·.toString < ·.toString)
  let mut aliases : Array String := #[]
  for i in [:names.size] do
    let name := names[i]!
    let some (.thmInfo info) := env.find? name
      | throwError "Comparator target is not a theorem: {name}"
    let wrapperName := Name.mkSimple s!"veritile_comparator_target_{i}"
    if env.contains wrapperName then throwError "Reserved comparator target name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := info.levelParams, type := info.type,
      value := mkConst name (info.levelParams.map Level.param)
    }
    aliases := aliases.push wrapperName.toString
  -- Definition-only regression fixtures still run comparator's primitive
  -- checks and kernel replay. Report zero original theorem targets explicitly.
  if aliases.isEmpty then
    let wrapperName := `veritile_comparator_empty_module
    if env.contains wrapperName then throwError "Reserved comparator target name: {wrapperName}"
    liftCoreM <| addDecl <| .thmDecl {
      name := wrapperName, levelParams := [], type := mkConst ``True,
      value := mkConst ``True.intro
    }
    aliases := #[wrapperName.toString]
  let report := Json.mkObj [
    ("theorems", toJson (names.map Name.toString)),
    ("aliases", toJson aliases)
  ]
  liftIO <| IO.FS.writeFile __REPORT__ report.compress
'''


def prepare(workspace):
    """One independent cache/source copy per batch, shared by its workers."""
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = workspace / 'ready.json'
        if ready.exists():
            state = json.loads(ready.read_text())
            if state['root'] != str(ROOT):
                raise ValueError('Comparator workspace belongs to another project')
            return state
        if (workspace / 'failed').exists():
            raise ValueError('Comparator workspace initialization previously failed')
        try:
            comparator = require_tools()
            judge = workspace / 'project'
            judge.mkdir()
            hashes = snapshot_project(ROOT, judge, libraries=('VeriTileComparator',), submodules=True)
            (judge / 'VeriTileComparator').mkdir()
            manifest = ROOT / 'scripts/kernel-manifest.tsv'
            (judge / 'scripts').mkdir(exist_ok=True)
            if manifest.exists():
                shutil.copyfile(manifest, judge / 'scripts/kernel-manifest.tsv')
                hashes['scripts/kernel-manifest.tsv'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (ROOT / 'Logs').mkdir(exist_ok=True)
            logs = Path(tempfile.mkdtemp(prefix='comparator-check-', dir=ROOT / 'Logs'))
            (logs / 'inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
            state = {'root': str(ROOT), 'comparator': comparator, 'logs': str(logs),
                     'comparator_sha256': hashlib.sha256(Path(comparator).read_bytes()).hexdigest()}
            ready.write_text(json.dumps(state))
            return state
        except Exception:
            (workspace / 'failed').touch()
            raise


def library_targets(manifest):
    modules, names = set(), []
    for line in manifest.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) >= 5 and fields[1].startswith('VeriTile/') and fields[4] == 'proven':
            modules.add(fields[1][:-5].replace('/', '.'))
            names.append(fields[2])
    if not names:
        raise ValueError('No proven library targets in the kernel manifest')
    return sorted(modules), sorted(set(names))


def check(workspace, source=None, trust=False, library=False):
    state = prepare(workspace)
    judge = workspace / 'project'
    logs = Path(state['logs'])
    if library:
        modules, selected = library_targets(judge / 'scripts/kernel-manifest.tsv')
        text = ''.join(f'import {module}\n' for module in modules)
        label, tag = 'proven library manifest', 'Library'
    else:
        relative = source.resolve().relative_to(ROOT).as_posix()
        original = judge / relative
        if source.suffix != '.lean' or not original.is_file():
            raise ValueError(f'Not a snapshotted Lean source: {relative}')
        text = original.read_text()
        if trust:
            text = prepare_source(text, relative, judge / 'scripts/kernel-manifest.tsv')
        selected = []
        label = relative
        tag = 'File_' + hashlib.sha256((relative + str(trust)).encode()).hexdigest()[:24]
    module = 'VeriTileComparator.' + tag
    job_logs = Path(tempfile.mkdtemp(prefix=tag + '-', dir=logs))
    metadata_dir = judge / '.lake' / 'comparator' / tag
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata_dir / 'targets.json'
    selection = '#[' + ', '.join(f'{json.dumps(n)}.toName' for n in selected) + ']'
    footer = INVENTORY.replace('__SELECTED__', selection).replace(
        '__REPORT__', json.dumps(str(metadata)))
    program = 'import Lean\n' + text + '\n' + footer
    (judge / 'VeriTileComparator' / (tag + '.lean')).write_text(program)
    (job_logs / 'Source.lean').write_text(program)
    # These are trusted repository sources, not an untrusted agent submission.
    # Elaboration discovers actual theorem objects, never regex-matched names.
    with (job_logs / 'build.log').open('w') as output:
        build = subprocess.run(['lake', 'build', module], cwd=judge,
                               stdout=output, stderr=subprocess.STDOUT)
    if build.returncode or not metadata.is_file():
        print((job_logs / 'build.log').read_text(), file=sys.stderr)
        raise ValueError(f'{label}: elaboration/inventory failed; logs: {job_logs}')
    inventory = json.loads(metadata.read_text())
    shutil.copyfile(metadata, job_logs / 'targets.json')
    config = write_config(judge, inventory['aliases'], challenge=module, solution=module,
                          filename=f'.lake/comparator/{tag}/config.json')
    shutil.copyfile(config, job_logs / 'config.json')
    with (job_logs / 'comparator.log').open('w') as output:
        result = subprocess.run(comparator_command(judge, state['comparator'], config),
                                stdout=output, stderr=subprocess.STDOUT)
    report = {**state, 'source': label, 'mode': 'source-replay',
              'theorem_count': len(inventory['theorems']),
              'comparator_exit': result.returncode, 'accepted': result.returncode == 0}
    (job_logs / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if result.returncode:
        print((job_logs / 'comparator.log').read_text(), file=sys.stderr)
        raise ValueError(f'{label}: comparator rejected; logs: {job_logs}')
    if trust:
        for line in (job_logs / 'build.log').read_text().splitlines():
            for marker in ('Axiom audit:', 'Spec audit:'):
                if marker in line:
                    print(line[line.index(marker):])
    print(f'Comparator: {label}: {len(inventory["theorems"])} theorem(s) accepted; logs: {job_logs}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=Path)
    group.add_argument('--library', action='store_true')
    group.add_argument('--check-tools', action='store_true')
    parser.add_argument('--trust', action='store_true', help='also run the standalone trust gates')
    parser.add_argument('--workspace', type=Path, help='share one snapshot within a batch')
    args = parser.parse_args()
    if args.check_tools:
        require_tools()
        print('Official comparator and sandbox tools are available.')
    elif args.workspace:
        check(args.workspace.resolve(), args.file, args.trust, args.library)
    else:
        with tempfile.TemporaryDirectory(prefix='veritile-comparator-') as temp:
            check(Path(temp), args.file, args.trust, args.library)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)

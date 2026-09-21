#!/usr/bin/env python3
"""Run autoprove, then judge the requested theorems with leanprover/comparator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
AXIOMS = ['propext', 'Quot.sound', 'Classical.choice']


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def snapshot_project(root, destination):
    """Copy trusted inputs before the agent runs; never share writable caches."""
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=root).decode().split('\0')
    hashes = {}
    for relative in sorted(set(paths)):
        if not relative or not (relative.endswith('.lean') or relative in {
            'lakefile.toml', 'lakefile.lean', 'lake-manifest.json', 'lean-toolchain'
        }):
            continue
        source = root / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    # Independent copies include the trusted dependency sources and their build
    # artifacts. Reflinks save space where supported; hard links/symlinks would
    # allow the agent's edits to alter the judge's inputs.
    if (root / '.lake').exists():
        subprocess.run(['cp', '-aL', '--reflink=auto', str(root / '.lake'),
                        str(destination / '.lake')], check=True)
    lakefile = destination / 'lakefile.toml'
    if not lakefile.exists() or (destination / 'lakefile.lean').exists():
        raise ValueError('The proof runner requires the project lakefile.toml.')
    with lakefile.open('a') as output:
        output.write('\n[[lean_lib]]\nname = "ComparatorChallenge"\n'
                     '\n[[lean_lib]]\nname = "ComparatorSolution"\n')
    return hashes


def write_config(directory, theorems):
    if not theorems or any(not name.strip() for name in theorems):
        raise ValueError('At least one nonempty, fully qualified theorem name is required.')
    config = {
        'challenge_module': 'ComparatorChallenge',
        'solution_module': 'ComparatorSolution',
        'theorem_names': list(dict.fromkeys(theorems)),
        'permitted_axioms': AXIOMS,
        'enable_nanoda': False,
    }
    path = directory / 'comparator.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    return path


def comparator_command(directory, comparator):
    # Follow upstream's AF_UNIX restriction as well as comparator's landrun
    # sandbox. Failure to start either is a failed check, never a fallback.
    return ['systemd-run', '--user', '--pipe', '--wait', '--collect',
            '--property=RestrictAddressFamilies=~AF_UNIX',
            '--setenv=PATH=' + os.environ['PATH'],
            '--working-directory=' + str(directory),
            'lake', 'env', comparator, str(directory / 'comparator.json')]


def main():
    parser = argparse.ArgumentParser(prog="scripts/prove.sh", description=__doc__)
    parser.add_argument('lean_file', type=Path)
    parser.add_argument('--theorem', action='append', required=True,
                        help='Fully qualified target; repeat for multiple theorems')
    parser.add_argument('--max-cycles', type=positive_int, default=5)
    parser.add_argument('--prompt', default='')
    args = parser.parse_args()
    target = args.lean_file.resolve(strict=True)
    target.relative_to(ROOT)
    if target.suffix != '.lean':
        parser.error('lean_file must be a .lean file inside this project')
    if any(not name.strip() for name in args.theorem):
        parser.error('--theorem cannot be empty')

    comparator = shutil.which(os.environ.get('COMPARATOR_BIN', 'comparator'))
    if not comparator:
        parser.error('Official comparator is required; see scripts/README.md for setup.')
    for tool in ('claude', 'lake', 'lean4export', 'landrun', 'systemd-run'):
        if not shutil.which(tool):
            parser.error(f'Missing {tool}; see scripts/README.md for setup.')
    # Detect unavailable user services before spending an agent attempt.
    subprocess.run(['systemd-run', '--user', '--pipe', '--wait', '--collect',
                    '--property=RestrictAddressFamilies=~AF_UNIX', 'true'], check=True)

    logs = ROOT / 'Logs'
    logs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=target.stem + '_', dir=logs))
    original = target.read_bytes()
    (run / 'Challenge.lean').write_bytes(original)
    start = time.monotonic()
    print(f'[START] {target.relative_to(ROOT)} (logs: {run})', flush=True)
    # The independent judge workspace is made BEFORE autoprove starts. Only the
    # resulting source file is admitted afterward; no agent-built oleans/configs.
    with tempfile.TemporaryDirectory(prefix='veritile-judge-') as temporary:
        judge = Path(temporary)
        hashes = snapshot_project(ROOT, judge)
        (judge / 'ComparatorChallenge.lean').write_bytes(original)
        config = write_config(judge, args.theorem)
        shutil.copyfile(config, run / 'comparator.json')
        (run / 'inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
        command = (f'/lean4:autoprove {shlex.quote(str(target))} '
                   f'--max-cycles={args.max_cycles} --commit=never '
                   '--planning=off --review-source=none')
        if args.prompt:
            command += ' ' + args.prompt
        with (run / 'agent.jsonl').open('w') as output:
            agent = subprocess.run([
                'claude', '-p', command, '--dangerously-skip-permissions',
                '--max-budget-usd', '10.00', '--output-format', 'stream-json',
                '--include-partial-messages', '--verbose'], cwd=ROOT, stdout=output)
        candidate = target.read_bytes()
        (run / 'Solution.lean').write_bytes(candidate)
        (judge / 'ComparatorSolution.lean').write_bytes(candidate)
        # Do not compile the submission in the judge outside comparator.
        with (run / 'comparator.txt').open('w') as output:
            result = subprocess.run(comparator_command(judge, comparator),
                                    stdout=output, stderr=subprocess.STDOUT)
        accepted = result.returncode == 0
        report = {
            'accepted': accepted, 'theorem_names': args.theorem,
            'agent_exit': agent.returncode, 'comparator_exit': result.returncode,
            'seconds': round(time.monotonic() - start, 2),
            'comparator': comparator,
            'comparator_sha256': hashlib.sha256(Path(comparator).read_bytes()).hexdigest(),
        }
        (run / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    # Agent text and exit status are diagnostics. Only comparator decides whether
    # these specified theorems are proved against the original trusted inputs.
    print(f'[{"SUCCESS" if accepted else "FAIL"}] {args.lean_file} '
          f'(comparator exit={result.returncode}, logs: {run})')
    return 0 if accepted else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[FAIL] {error}', file=sys.stderr)
        raise SystemExit(1)

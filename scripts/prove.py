#!/usr/bin/env python3
"""Run autoprove, then judge the requested theorems with leanprover/comparator."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from comparator_common import comparator_command, require_tools, snapshot_project, write_config

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


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

    comparator = require_tools()
    if not shutil.which('claude'):
        parser.error('Missing claude; see scripts/README.md for setup.')

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

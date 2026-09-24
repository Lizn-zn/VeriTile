#!/usr/bin/env python3
"""Record a completed GitHub corpus audit; never turn a pending run into evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'site/src/lib/coverage-ci.json'
REPO = 'Lizn-zn/VeriTile'


def completed_record(run_id, run, jobs):
    if (run['status'] != 'completed' or run['conclusion'] != 'success'
            or run['path'] != '.github/workflows/bench-audit.yml'
            or not re.fullmatch('[0-9a-f]{40}', run['head_sha'])
            or run['html_url'] != f'https://github.com/{REPO}/actions/runs/{run_id}'):
        raise ValueError('Only a successful completed corpus-audit run can be recorded.')
    if not jobs or any(job['status'] != 'completed' or job['conclusion'] != 'success'
                       or not job['completed_at'] for job in jobs):
        raise ValueError('Every job must have completed successfully.')
    return {'schema': 1, 'run_id': run_id, 'commit': run['head_sha'],
            'url': run['html_url'], 'completed_at': max(job['completed_at'] for job in jobs),
            'conclusion': 'success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', type=int)
    args = parser.parse_args()
    run = json.loads(subprocess.check_output(
        ['gh', 'api', f'repos/{REPO}/actions/runs/{args.run_id}'], text=True))
    jobs = json.loads(subprocess.check_output(
        ['gh', 'api', f'repos/{REPO}/actions/runs/{args.run_id}/jobs'], text=True))['jobs']
    data = completed_record(args.run_id, run, jobs)
    OUTPUT.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Recorded completed corpus audit {args.run_id} at {run["head_sha"]}.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error))

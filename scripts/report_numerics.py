#!/usr/bin/env python3
"""Maintain one current table of numerical rule decisions, max |z| and U."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import errno
import io
import json
import math
from pathlib import Path
import subprocess
import time

if __package__:
    from . import check_numerics as experiment
else:
    import check_numerics as experiment

COLUMNS = ("rule", "format", "replicates", "z", "B", "tau", "U", "bias", "vars", "u_kind",
           "accept", "decision", "state", "replayed", "reason")
DEFINITIONS = {
    'error_units': 'per-element output-format ULP at the rounded golden value; normalize before aggregation',
    'z_definition': 'maximum absolute z of local-ULP bucket means across all output buckets',
    'b_definition': 'maximum over buckets of abs(mean) + se_multiplier * std / sqrt(R), in local ULPs; not calibrated simultaneous/sequential coverage',
    'u_definition': 'upper estimate of amplification of peak local-ULP oracle errors with additive allowance 1; empirical_max is not a tail confidence bound',
}
TERMINAL = {"Succeeded", "Failed", "Stopped", "Deleted"}


def maximum_z(values):
    """Use the largest absolute channel statistic, preserving infinities."""
    if not values:
        return None
    numbers = [float(value) for value in values]
    if any(math.isnan(value) or value < 0 for value in numbers):
        raise ValueError("invalid absolute z statistic")
    value = max(numbers)
    return "+inf" if math.isinf(value) else value


def compatible_profile(profile):
    return {key: value for key, value in profile.items() if key != "rules"}


def collect(root, profile, cache=None, verify_all=False):
    """Include every requested rule/format, even when its bundle does not exist yet.

    A published import report triggers an independent replay once per unchanged
    bundle. Until replay, COMPLETE records show provisional z/U and pending accept.
    Smoke bundles never supply rows or acceptance to the formal table.
    """
    cache = {} if cache is None else cache
    expected = {(rule, fmt["name"]): {
        "rule": rule, "format": fmt["name"], "replicates": None, "z": None, "B": None, "tau": None, "U": None,
        "bias": None, "vars": None, "u_kind": None, "accept": None,
        "decision": "NOT_EVALUATED", "state": "PENDING", "replayed": False, "reason": "",
    } for rule in profile["rules"] for fmt in profile["formats"]}
    seen = set()
    for manifest_path in sorted(root.glob('*/manifest.json')):
        bundle = manifest_path.parent
        manifest = experiment.read_json(manifest_path)
        if manifest["smoke"]:
            continue
        if compatible_profile(manifest["profile"]) != compatible_profile(profile):
            raise ValueError(f"incompatible experiment configuration: {bundle}")
        records = {}
        fingerprint = [experiment.sha(manifest_path.read_bytes())]
        for name in manifest["entries"]:
            path = bundle / name / 'record.json'
            if path.exists():
                records[name] = experiment.read_json(path)
                fingerprint.append(experiment.sha(path.read_bytes()))
            else:
                fingerprint.append(None)
        imported = root / (bundle.name + '-report.json')
        verified = None
        verification_error = None
        if imported.exists() or verify_all:
            key = (str(bundle), tuple(fingerprint))
            if key not in cache:
                try:
                    cache[key] = (experiment.replay(bundle), None)
                except OSError as error:
                    if error.errno == errno.ESTALE:
                        raise
                    cache[key] = (None, str(error))
                except (ValueError, KeyError, TypeError) as error:
                    cache[key] = (None, str(error))
            verified, verification_error = cache[key]
        verified_rows = {} if verified is None else {
            (r['rule_id'], r['format']): r for r in verified['rows']}
        for fmt in manifest['profile']['formats']:
            for rule in manifest['profile']['rules']:
                identity = rule, fmt['name']
                if identity not in expected or identity in seen:
                    raise ValueError(f"unexpected or duplicate formal result: {identity}")
                seen.add(identity)
                row = expected[identity]
                record = records.get(f"{fmt['name']}__{rule}")
                if record is None:
                    continue
                if (record.get('rule_id'), record.get('format')) != identity:
                    raise ValueError(f"record identity mismatch: {identity}")
                row.update(state=record['state'], decision=record['decision'],
                           replicates=record.get('completed_replicates'), reason=record.get('reason', ''))
                if record['state'] != 'COMPLETE':
                    if record['state'] in {'UNSUPPORTED', 'NUMERIC_EVENT', 'ERROR'}:
                        row['accept'] = False
                    continue
                result = record['result']
                bias, magnitude = result['bias'], result['vars']
                if record['decision'] != result['decision']:
                    raise ValueError(f"record decision mismatch: {identity}")
                row.update(replicates=result['completed_replicates'],
                           z=maximum_z(bias.get('abs_z')), B=bias.get('upper'), tau=bias.get('tau'), U=magnitude.get('upper'),
                           bias=bias['status'], vars=magnitude['status'],
                           u_kind='empirical_max' if magnitude['empirical_fallback'] else magnitude['branch'])
                if verification_error:
                    row.update(state='REPLAY_ERROR', reason=verification_error)
                elif identity in verified_rows:
                    checked = verified_rows[identity]
                    if checked['decision'] != record['decision']:
                        raise ValueError(f"replay decision mismatch: {identity}")
                    row.update(replayed=True, accept=checked['decision'] in experiment.ACCEPTED)
                else:
                    row['reason'] = 'awaiting bundle replay'
    rows = list(expected.values())
    return {
        'rows': rows,
        'total': len(rows),
        'states': dict(Counter(r['state'] for r in rows)),
        'accepted': sum(r['accept'] is True for r in rows),
        'replayed': sum(r['replayed'] for r in rows),
        **DEFINITIONS,
        'complete': all(r['state'] in {'UNSUPPORTED', 'NUMERIC_EVENT', 'ERROR'} or r['replayed'] for r in rows),
    }


def display_number(value):
    if value is None:
        return '—'
    if value == '+inf':
        return '∞'
    return format(float(value), '.7g')


def write_current(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def publish(root, table):
    root.mkdir(parents=True, exist_ok=True)
    write_current(root / 'summary.json', json.dumps(table, indent=2, allow_nan=False) + '\n')
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(table['rows'])
    write_current(root / 'summary.csv', output.getvalue())
    lines = ['# Numerical rule results', '',
             f"{table['total']} instances; {table['replayed']} replayed; {table['accepted']} accepted.", '',
             'z = max |z| over output buckets (diagnostic only). U uses the configured magnitude gate.',
             'B = max(abs(mean) + se_multiplier * SE); bias PASS requires B <= tau, in local ULPs.',
             'Bias FAIL means an interval lies outside tolerance; INCONCLUSIVE means a boundary is crossed.',
             'The SE bands are engineering criteria, not calibrated simultaneous or optional-stopping confidence guarantees.',
             'Errors are normalized per element by the output-format ULP at the rounded golden value before aggregation.',
             'The magnitude gate uses peak normalized oracle errors and an additive allowance of 1 local ULP.',
             '`empirical_max` means an observed maximum, not a fitted tail confidence bound.',
             'Acceptance is statistical under the configured profile, not proof of strict floating-point equivalence.',
             'Accept is pending until CPU replay. Missing statistics are shown as —, never zero.', '',
             '| Rule | Format | R | z | B (ULP) | tau (ULP) | U | U type | Bias | Vars | Accept | State |',
             '|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for r in table['rows']:
        accept = 'yes' if r['accept'] is True else 'no' if r['accept'] is False else 'pending'
        values = [r['rule'], r['format'], str(r['replicates']) if r['replicates'] is not None else '—',
                  display_number(r['z']), display_number(r['B']), display_number(r['tau']),
                  display_number(r['U']), r['u_kind'] or '—',
                  r['bias'] or '—', r['vars'] or '—', accept, r['state']]
        lines.append('| ' + ' | '.join(values) + ' |')
    notes = [r for r in table['rows'] if r['reason'] and r['reason'] != 'awaiting bundle replay']
    if notes:
        lines.extend(['', '## Notes', ''])
        for r in notes:
            lines.append(f"- {r['rule']} / {r['format']}: {r['reason']}")
    write_current(root / 'summary.md', '\n'.join(lines) + '\n')


def refresh(root, profile, cache, verify_all=False, status=None, attempts=5, report_dir=None):
    """Reopen files after a transient shared-filesystem stale handle.

    A stale handle is not failed numerical evidence and must not poison the
    replay cache. Keep the current published table until a fresh read succeeds.
    """
    for attempt in range(attempts):
        try:
            table = collect(root, profile, cache, verify_all=verify_all)
            table['job_status'] = status
            publish(root if report_dir is None else report_dir, table)
            return table
        except OSError as error:
            if error.errno != errno.ESTALE or attempt + 1 == attempts:
                raise
            print('Shared filesystem handle expired; reopening result files.', flush=True)
            time.sleep(min(2 ** attempt, 5))


def job_status(job_id):
    proc = subprocess.run(['aliyun', 'pai-dlc', 'GetJob', '--region', 'ap-southeast-1',
                           '--JobId', job_id], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(proc.stdout)['Status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='directory containing formal bundles and import reports')
    parser.add_argument('--profile', type=Path, default=experiment.DEFAULT_PROFILE)
    parser.add_argument('--report-dir', type=Path, help='write current tables here; read bundles from root')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--job-id', help='DLC task to follow; terminal status triggers final replay')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--timeout', type=float, default=7800)
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error('interval and timeout must be positive')
    profile = experiment.validate_profile(deepcopy(experiment.load_module(args.profile).PROFILE))
    args.root.mkdir(parents=True, exist_ok=True)
    cache, previous = {}, None
    started = time.monotonic()
    while True:
        status = None
        if args.job_id:
            try:
                status = job_status(args.job_id)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(f'Job status temporarily unavailable: {type(error).__name__}', flush=True)
        done = status in TERMINAL
        try:
            table = refresh(args.root, profile, cache, verify_all=done or not args.watch,
                            status=status, report_dir=args.report_dir)
        except OSError as error:
            if error.errno != errno.ESTALE or not args.watch or time.monotonic() - started >= args.timeout:
                raise
            print('Shared filesystem still unavailable; retaining table and retrying.', flush=True)
            time.sleep(args.interval)
            continue
        progress = (table['states'], table['replayed'], table['accepted'], status)
        if progress != previous:
            print(json.dumps({'states': table['states'], 'replayed': table['replayed'],
                              'accepted': table['accepted'], 'job_status': status}), flush=True)
            previous = progress
        if not args.watch or done or table['complete']:
            return 0
        if time.monotonic() - started >= args.timeout:
            print('Watcher timed out; current partial table retained.', flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay both numerical catalogs and replace their current public reports.

The run directory contains original/atomic, supplement/atomic, GPU-generated
report directories, status.json and exit_code. Raw evidence stays outside Git.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

if __package__:
    from . import check_numerics as original, check_numerics_supplement as supplement
    from . import report_numerics as reporting
else:
    import check_numerics as original
    import check_numerics_supplement as supplement
    import report_numerics as reporting


def audit(bundle, table):
    manifest = original.read_json(bundle / 'manifest.json')
    rows = []
    for row in table['rows']:
        if row['state'] != 'COMPLETE' or row['accept']:
            continue
        record = original.read_json(bundle / f"{row['format']}__{row['rule']}" / 'record.json')
        bias, var = record['result']['bias'], record['result']['vars']
        means = np.asarray(bias.get('mean', []), dtype=float)
        rows.append({
            **{k: row[k] for k in ('rule', 'format', 'replicates', 'z', 'B', 'tau', 'U',
                                   'bias', 'vars', 'u_kind', 'decision', 'accept')},
            'failing_buckets': len(bias.get('failing_buckets', [])),
            'inconclusive_buckets': len(bias.get('inconclusive_buckets', [])),
            'bias_lower_bound': bias.get('lower'),
            'max_abs_mean_in_local_ulps': float(np.abs(means).max()) if means.size else None,
            'vars_interval': [2 * var['return_level'] - var['upper'], var['upper']]
                             if isinstance(var.get('upper'), (int, float)) else None,
        })
    return {'checked_by': 'Independent CPU replay of observations, stopping points, source/contract/PTX hashes',
            **reporting.DEFINITIONS, 'thresholds': manifest['profile']['gates'], 'rows': rows}


def publish(run, destination, hardware_model):
    status = original.read_json(run / 'status.json')
    if status['Status'] != 'Succeeded' or (run / 'exit_code').read_text().strip() != '0':
        raise ValueError('Both GPU runs must finish successfully before publication')
    job = {k: status.get(k) for k in ('JobId', 'DisplayName', 'Status', 'GmtRunningTime', 'GmtFinishTime')}
    with tempfile.TemporaryDirectory(prefix='cpu-replay-', dir=run) as temporary:
        staging = Path(temporary)
        main_root, supp_root = run / 'original', run / 'supplement'
        manifest = original.read_json(main_root / 'atomic/manifest.json')
        table = reporting.collect(main_root, manifest['profile'], verify_all=True)
        table['job_status'] = None
        if not table['complete'] or any(r['state'] in ('ERROR', 'REPLAY_ERROR') for r in table['rows']):
            raise ValueError('Original catalog failed independent replay')
        reporting.publish(staging / 'original', table)
        print('Original catalog independently replayed.', flush=True)
        replay = supplement.replay(supp_root / 'atomic')
        supplement.publish_report(supp_root / 'atomic', replay, staging / 'supplement')
        supp_table = original.read_json(staging / 'supplement/summary.json')
        if any(r['state'] in ('ERROR', 'PENDING') for r in supp_table['rows']):
            raise ValueError('Supplement catalog failed independent replay')
        print('Supplement catalog independently replayed.', flush=True)

        for name, root in [('original', main_root), ('supplement', supp_root)]:
            for filename in ('summary.md', 'summary.csv', 'summary.json'):
                if (root / 'report' / filename).read_bytes() != (staging / name / filename).read_bytes():
                    raise ValueError(f'GPU/CPU report disagreement: {name}/{filename}')

        runtime_name = manifest['backend']['target']['device']
        main_settings = {
            'profile': manifest['profile'], 'backend': manifest['backend'], 'sources': manifest['sources'],
            'hardware': {'model': hardware_model or runtime_name, 'runtime_name': runtime_name},
            'bundle_manifest_sha256': original.sha((main_root / 'atomic/manifest.json').read_bytes()),
            'job': job,
            'evidence': 'Fresh GPU samples from the frozen profile, followed by independent CPU replay; raw observations and PTX stay outside Git.',
        }
        supp_settings = original.read_json(supp_root / 'atomic/manifest.json')
        supp_settings['execution'] = {
            'job': job, 'hardware_model': hardware_model or runtime_name,
            'independent_cpu_replay': {'complete_rows': supp_table['replayed'], 'numpy': np.__version__,
                                       'tables_identical_to_gpu_environment': True},
            'manifest_sha256': original.sha((supp_root / 'atomic/manifest.json').read_bytes()),
        }
        settings = {'original': main_settings, 'supplement': supp_settings}
        tables = {'original': table, 'supplement': supp_table}
        # Finish all checks and audit construction before replacing current files.
        for name in tables:
            original.write_json(staging / name / 'experiment.json', settings[name])
            original.write_json(staging / name / 'warning_audit.json', audit(run / name / 'atomic', tables[name]))
        for name, current in tables.items():
            saved = run / name / 'cpu-report'
            saved.mkdir(exist_ok=True)
            target = destination / ('report' if name == 'original' else 'supplement/report')
            target.mkdir(parents=True, exist_ok=True)
            for source in (staging / name).iterdir():
                shutil.copyfile(source, saved / source.name)
            for filename in ('summary.md', 'summary.csv', 'summary.json', 'experiment.json', 'warning_audit.json'):
                reporting.write_current(target / filename, (staging / name / filename).read_text())
            print(f"Published {name}: {current['total']} rows, {current['accepted']} accepted.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--destination', type=Path, default=original.ROOT / 'experiments/floating_point')
    parser.add_argument('--hardware-model', help='Explicit physical model when the runtime reports a provider alias')
    args = parser.parse_args()
    try:
        publish(args.run.resolve(), args.destination.resolve(), args.hardware_model)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

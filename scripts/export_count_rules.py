#!/usr/bin/env python3
"""Bind the user-trusted bounded integer report as Lean data, without GPU replay."""
import argparse
from copy import deepcopy
from pathlib import Path

if __package__:
    from . import check_numerics_supplement as runner
    from .export_numerical_rules import lean_string, validate_accepted_bounds
else:
    import check_numerics_supplement as runner
    from export_numerical_rules import lean_string, validate_accepted_bounds

ROOT = runner.ROOT
REPORT = ROOT / 'experiments/floating_point/primitives/report'
OUTPUT = ROOT / 'VeriTile/Triton/Float/CountAdmission.lean'
RULES = ('COUNT-ZERO', 'COUNT-SUCCESSOR')
canonical = runner.original.registry.canonical_json


def load_report(directory):
    files = {name: runner.read_json(directory / name) for name in
             ('experiment.json', 'summary.json', 'admission.json')}
    manifest = files['experiment.json']['bundles']['counts']
    profile = runner.validate_profile(deepcopy(manifest['profile']))
    if manifest['sources'] != runner.source_hashes():
        raise ValueError('count report source hashes differ from the implementation')
    if manifest['bundle_version'] != runner.BUNDLE_VERSION or manifest['smoke'] is not False:
        raise ValueError('only formal count bundles can be imported')
    expected_format = dict(name='int32_fp32', input='int32', compute='fp32', accumulator='fp32', output='fp32')
    if profile['formats'] != [expected_format] or profile['rules'] != list(RULES):
        raise ValueError('count binding requires exactly the int32/fp32 zero and successor probes')
    dist = profile['distribution']
    if dist['family'] != 'uniform_integer' or dist['low'] != 0 or not 0 < dist['high'] <= 2**24:
        raise ValueError('count binding requires integer range 0 <= i < high <= 2^24')
    if profile['gates']['warning_policy'] != 'pass_only':
        raise ValueError('count binding requires pass_only')
    if manifest['entries'] != ['int32_fp32__' + rule for rule in RULES]:
        raise ValueError('count manifest entries disagree with profile')
    admission = files['admission.json']['counts']
    if admission['manifest_sha256'] != runner.sha(canonical(manifest) + b'\n'):
        raise ValueError('count manifest hash mismatch')
    rows = [r for r in files['summary.json']['rows'] if r['rule'] in RULES]
    accepted = admission['accepted']
    if len(rows) != 2 or {r['rule'] for r in rows} != set(RULES):
        raise ValueError('missing or duplicate count summary rows')
    if len(accepted) != 2 or {r['rule_id'] for r in accepted} != set(RULES):
        raise ValueError('both count atoms must have accepted report entries')
    by_rule = {r['rule_id']: r for r in accepted}
    recorded = admission['rows']
    if len(recorded) != 2 or {r['rule_id'] for r in recorded} != set(RULES):
        raise ValueError('missing or duplicate count admission rows')
    for record in recorded:
        if record != {k: by_rule[record['rule_id']][k] for k in record}:
            raise ValueError('accepted count entry disagrees with its recorded row')
    for row in rows:
        rule = row['rule']
        entry = by_rule[rule]
        if (row['accept'] is not True or row['replayed'] is not True or row['state'] != 'COMPLETE'
                or row['format'] != 'int32_fp32' or row['decision'] != 'ACCEPT'
                or row['bias'] != 'PASS' or row['vars'] != 'PASS'):
            raise ValueError('count summary is not dual-PASS and replayed')
        validate_accepted_bounds(row, profile)
        cap = ((profile['replicates_max'] + profile['batch'] - 1) // profile['batch']) * profile['batch']
        count = row['replicates']
        if type(count) is not int or not profile['replicates'] <= count <= cap or count % profile['batch']:
            raise ValueError('invalid count replicate total')
        cfg = entry['config']
        expected_domain = ('constant integer zero' if rule == 'COUNT-ZERO' else
                           f"integer a; 0 <= a < {dist['high']}; int32 a+1 cannot overflow")
        expected_relation = {**runner.supplemental.load_catalog()[rule],
                             'domain': expected_domain, 'final_cast': 'fp32'}
        if cfg['rule_id'] != rule or cfg['relation'] != expected_relation or entry['relation'] != expected_relation:
            raise ValueError('count relation or integer domain does not match the profile')
        if (entry['instance_key'] != runner.supplemental.instance_key(cfg)
                or cfg['backend'] != manifest['backend']):
            raise ValueError('count configuration identity mismatch')
        numerics = cfg['numerics']
        if (numerics['input_formats'] != dict(a='int32', b='int32', c='int32')
                or numerics['node_formats']['arithmetic'] != 'fp32'
                or numerics['output_formats'] != dict(out='fp32')):
            raise ValueError('count numerical precision mismatch')
        family = 'constant' if rule == 'COUNT-ZERO' else 'uniform_integer'
        roles = {} if rule == 'COUNT-ZERO' else {'a': dist}
        if cfg['probe']['family'] != family or cfg['probe']['roles'] != roles:
            raise ValueError('count sampling range mismatch')
        if (entry['format'] != 'int32_fp32' or entry['decision'] != 'ACCEPT'
                or entry['bias'] != 'PASS' or entry['vars'] != 'PASS'
                or entry['statistics']['decision'] != 'ACCEPT'
                or entry['statistics']['bias']['status'] != 'PASS'
                or entry['statistics']['vars']['status'] != 'PASS'
                or entry['replicates'] != count
                or entry['statistics']['bias']['upper'] != row['B']
                or entry['statistics']['vars']['upper'] != row['U']):
            raise ValueError('count admission disagrees with summary')
    hashes = {name: runner.sha((directory / name).read_bytes()) for name in files}
    return files, profile, by_rule, hashes, runner.sha(canonical(hashes))


def render(directory=REPORT):
    files, profile, entries, hashes, snapshot = load_report(directory)
    path = directory.resolve().relative_to(ROOT).as_posix() if directory.resolve().is_relative_to(ROOT) else str(directory.resolve())
    metadata = {'trust': 'user-trusted published report; no independent replay during export',
                'report_sha256': hashes, 'experiment': files['experiment.json']}
    lines = ['/- Generated by scripts/export_count_rules.py --trust-report.',
             '   Integer bounds are binding data, not an unbounded conversion axiom. -/',
             'import VeriTile.Triton.Float.ReportedRules', '',
             'namespace VeriTile.Triton.FP.CountAdmission', '',
             f'def snapshot : String := {lean_string(snapshot)}',
             f'def upperExclusive : Nat := {profile["distribution"]["high"]}',
             f'def reportMetadata : Lean.Json := (Lean.Json.parse {lean_string(canonical(metadata).decode())}).toOption.getD .null', '']
    for rule, name in zip(RULES, ('zero', 'successor')):
        entry = entries[rule]
        lines += [f'def {name} : ReportedRule where', f'  ruleID := {lean_string(rule)}',
                  '  format := "int32_fp32"',
                  f'  key := {lean_string(f"report:{snapshot}:int32_fp32:{rule}")}',
                  f'  artifact := {lean_string(f"{path}/summary.json#int32_fp32/{rule}")}',
                  '  configuration := Lean.Json.mkObj [("report", reportMetadata),',
                  f'    ("entry", (Lean.Json.parse {lean_string(canonical(entry).decode())}).toOption.getD .null)]',
                  f'  description := {lean_string(entry["relation"]["domain"])}',
                  f'  shape := {profile["shape"]}', f'  block := {profile["launch"]["block"]}',
                  '  input := "int32"', '  compute := "fp32"',
                  '  accumulator := "fp32"', '  output := "fp32"', '']
    lines += ['def all : List ReportedRule := [zero, successor]',
              'theorem accepted_count : all.length = 2 := rfl', '',
              'end VeriTile.Triton.FP.CountAdmission', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--trust-report', action='store_true', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        content = render(args.report)
        if args.check:
            if args.output.read_text() != content:
                raise ValueError('generated count table is stale')
        else:
            args.output.write_text(content)
        print(f'{"Checked" if args.check else "Generated"} {args.output}')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

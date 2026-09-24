#!/usr/bin/env python3
"""Validate explicit coverage reviews and expose their source evidence to the site.

This is documentation inventory, not proof discovery or a Lean verifier.
Proof selection/replay remains the elaborated-environment comparator gate.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / 'bench/tritonbench_g/coverage_review.json'
sys.path.insert(0, str(ROOT / 'scripts'))
from spec_sheet import split_decls, split_statement, strip_comments

LEVELS = {'full_value_candidate', 'specialization', 'precomputed_input_slice',
          'pre_rounding_slice', 'projection_only', 'blocked_summary',
          'public_summary_with_proof_gap'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(root: Path, name: str, suffix: str) -> Path:
    path = (root / name).resolve()
    if (not path.is_relative_to((root / 'bench/tritonbench_g').resolve())
            or path.suffix != suffix or not path.is_file()):
        raise ValueError(f'Invalid coverage source: {name}')
    return path


def load_reviews(rows, root=ROOT, path=REVIEWS):
    data = json.loads(path.read_text())
    if data.get('schema') != 1 or not data.get('basis') or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}', data.get('reviewed_on', '')):
        raise ValueError('Invalid coverage-review schema/date/basis')
    wanted = {(r.file, r.declaration) for r in rows}
    reviewed = {}
    files = set()
    for port in data['ports']:
        file = port['file']
        if file in files:
            raise ValueError(f'Duplicate reviewed file: {file}')
        files.add(file)
        lean = source_path(root, file, '.lean')
        python = source_path(root, port['python_file'], '.py')
        if lean.parent != python.parent:
            raise ValueError(f'Python source belongs to another port: {file}')
        for source, key in ((lean, 'lean_sha256'), (python, 'python_sha256')):
            if digest(source) != port[key]:
                raise ValueError(f'Coverage review is stale: {source.relative_to(root)}; '
                                 'review the changed source before updating its fingerprint')
        functions = {(n.name, n.lineno) for n in ast.parse(python.read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for entry in port['declarations']:
            key = (file, entry['declaration'])
            if key not in wanted or key in reviewed:
                raise ValueError(f'Unexpected/duplicate coverage declaration: {key}')
            if entry['coverage_level'] not in LEVELS or not entry['scope'].strip():
                raise ValueError(f'Missing explicit coverage scope/category: {key}')
            if entry['numeric_model'] not in {'mathematical', 'abstract_rounding'}:
                raise ValueError(f'Invalid numeric model: {key}')
            links = entry['python_functions']
            if not links or any((f['name'], f['line']) not in functions for f in links):
                raise ValueError(f'Unresolved Python function link: {key}')
            reviewed[key] = {**entry, 'python_file': port['python_file']}
    if set(reviewed) != wanted:
        raise ValueError(f'Missing coverage reviews: {sorted(wanted - set(reviewed))}')
    return data, reviewed


def apply_reviews(rows):
    _, reviewed = load_reviews(rows)
    output = []
    for row in rows:
        entry = reviewed[row.file, row.declaration]
        level = entry['coverage_level']
        # Explicit in-source blockers cannot be silently upgraded by a review.
        if row.coverage_level != 'unreviewed' and row.coverage_level != level:
            raise ValueError(f'Coverage review conflicts with source annotation: '
                             f'{row.file}::{row.declaration}')
        if level in {'precomputed_input_slice', 'pre_rounding_slice',
                     'public_summary_with_proof_gap', 'blocked_summary'}:
            family = row.blocker_family if row.blocker_family != 'none' else (
                'quantization-semantic-followup' if level == 'pre_rounding_slice'
                else 'proof-slice-precomputed-value')
            issue = row.issue or ('#158' if level == 'pre_rounding_slice' else '#152')
        else:
            family, issue = 'none', ''
        output.append(replace(row, coverage_level=level, blocker_family=family,
                              issue=issue, evidence=entry['scope']))
    return output


def declaration_index(text):
    return {d['name']: d for d in split_decls(text)}


def render_inventory(rows, root=ROOT, path=REVIEWS):
    data, reviews = load_reviews(rows, root, path)
    cache = {}
    output = []
    for row in rows:
        if row.file not in cache:
            text = (root / row.file).read_text()
            cache[row.file] = (text, declaration_index(text))
        text, decls = cache[row.file]
        if row.declaration not in decls:
            raise ValueError(f'Unresolved Lean declaration: {row.file}::{row.declaration}')
        decl = decls[row.declaration]
        statement, _ = split_statement(decl['text'])
        review = reviews[row.file, row.declaration]
        ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_'?!]*", strip_comments(statement)))
        refs = [d for n, d in decls.items() if n in ids
                and d['kind'] in {'def', 'abbrev', 'denotation'}]
        references = [dict(name=d['name'], line=text[:d['start']].count('\n') + 1)
                      for d in refs]
        # IO declarations are definitions written with `where`, not predicates
        # which merely pass a named `(kernel := ...)` argument.
        io = []
        for d in refs:
            _, body = split_statement(d['text'])
            if body.startswith('where'):
                match = re.search(r'\bkernel\s*:=\s*(.*?)(?=\n  \w[^\n]*:=|\Z)',
                                  body, re.S)
                if match:
                    io.append({'name': d['name'], 'kernel': match[1].strip(),
                               'definition': d['text']})
        output.append({**review, 'file': row.file,
                       'port': Path(row.file).parent.name,
                       'line': text[:decl['start']].count('\n') + 1,
                       'statement': statement, 'references': references,
                       'io': io, 'issue': row.issue,
                       'id': Path(row.file).parent.name + '--' + row.declaration})
    return {'schema': 1, 'reviewed_on': data['reviewed_on'], 'basis': data['basis'],
            'rows': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    from check_proof_gap_manifest import collect
    rows = collect()
    data = render_inventory(rows)
    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(f'{len(data["rows"])} coverage reviews; '
              f'{len({r.file for r in rows})} source pairs and Python links checked.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'Coverage inventory failed: {error}')

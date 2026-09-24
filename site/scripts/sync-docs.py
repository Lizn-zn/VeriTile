#!/usr/bin/env python3
"""Render repository design notes into Starlight; --check detects stale copies."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'site/src/content/docs'
BASE = '/VeriTile/'
GITHUB = 'https://github.com/Lizn-zn/VeriTile'
DOCS = [
    ('CodeOrganization', 'architecture/code-organization'),
    ('TritonSubset', 'architecture/triton-subset'),
    ('GpuMemoryModel', 'architecture/gpu-memory-model'),
    ('EraseDType', 'architecture/erase-dtype'),
    ('ConcurrencySemantics', 'architecture/concurrency-semantics'),
    ('MemorySafety', 'architecture/memory-safety'),
    ('SemanticCaveats', 'architecture/semantic-caveats'),
    ('ProofConventions', 'proofs/proof-conventions'),
    ('TheoremSurfaces', 'proofs/theorem-surfaces'),
    ('CorrectnessSurfaces', 'proofs/correctness-surfaces'),
    ('ApproxGeluPhiStrategy', 'proofs/approx-gelu-phi-strategy'),
    ('KernelManifest', 'proofs/kernel-manifest'),
    ('TrustAudit', 'proofs/trust-audit'),
]
ROUTES = {ROOT / f'documents/{name}.md': route + '/' for name, route in DOCS}


def rewrite_link(href, source):
    url = urlsplit(href)
    if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
        return href
    target = (source.parent / url.path).resolve()
    if not target.is_relative_to(ROOT) or not target.exists():
        raise ValueError(f'{source.relative_to(ROOT)}: missing link target {href}')
    if target in ROUTES:
        path = BASE + ROUTES[target]
        return urlunsplit(('', '', path, url.query, url.fragment))
    kind = 'tree' if target.is_dir() else 'blob'
    path = quote(target.relative_to(ROOT).as_posix(), safe='/')
    return f'{GITHUB}/{kind}/main/{path}' + (f'?{url.query}' if url.query else '') + (f'#{url.fragment}' if url.fragment else '')


def render(source):
    lines = source.read_text().splitlines()
    title = next(line[2:].strip() for line in lines if line.startswith('# '))
    body = []
    dropped_title = False
    for line in lines:
        if not dropped_title and line.startswith('# '):
            dropped_title = True
            continue
        body.append(line)
    # Rewrite prose links, leaving Lean and other fenced examples untouched.
    rendered = []
    fence = None
    for line in body:
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
        elif fence is None:
            line = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + rewrite_link(m[2], source) + m[3], line)
            line = re.sub(r'^(\[[^]]+\]:\s*)(\S+)', lambda m: m[1] + rewrite_link(m[2], source), line)
        rendered.append(line)
    text = '\n'.join(rendered).strip()
    return f'---\ntitle: {json.dumps(title, ensure_ascii=False)}\n---\n\n{text}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if a generated document differs')
    args = parser.parse_args()
    stale = []
    for source, route in ROUTES.items():
        dest = DEST / (route.rstrip('/') + '.md')
        expected = render(source)
        if not dest.exists() or dest.read_text() != expected:
            if args.check:
                stale.append(str(dest.relative_to(ROOT)))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(expected)
                print(f'Synced {dest.relative_to(ROOT)}')
    if stale:
        print('Documentation copies are stale. Run site/scripts/migrate-docs.sh:')
        print('\n'.join(stale))
        return 1
    print(f'{len(ROUTES)} documentation pages {"checked" if args.check else "synchronized"}.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

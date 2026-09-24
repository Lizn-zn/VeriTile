#!/usr/bin/env python3
"""Reject CJK text in tracked repository files and paths; allow math notation."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]')


def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    failures = []
    checked = 0
    for name in filter(None, names):
        if CJK.search(name):
            failures.append(f'{ascii(name)}: use an English path')
        path = ROOT / name
        if not path.is_file():
            continue  # Allow locally deleted files before they are staged.
        data = path.read_bytes()
        if b'\0' in data:
            continue
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            continue  # Binary assets are outside the text check.
        checked += 1
        for number, line in enumerate(text.splitlines(), 1):
            if match := CJK.search(line):
                failures.append(f'{name}:{number}: translate U+{ord(match[0]):04X} to English')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Repository language check passed: {checked} tracked text files, no CJK text.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

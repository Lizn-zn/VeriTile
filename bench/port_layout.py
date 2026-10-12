"""Discover legacy ports and split source/specification/proof directories.

This module selects files, never certifies a proof. Missing entry files and
ambiguous legacy sources are errors rather than silently skipped targets.
"""
from pathlib import Path
import argparse


def source_file(directory: Path) -> Path | None:
    if (directory / 'Kernels.lean').is_file():
        for name in ('Correct.lean', 'FPEquiv.lean'):
            if not (directory / name).is_file():
                raise ValueError(f'Missing split-port entry: {directory / name}')
        extra = set(directory.glob('*.lean')) - {
            directory / name for name in ('Kernels.lean', 'Correct.lean', 'FPEquiv.lean')}
        if extra:
            raise ValueError(f'Unexpected split-port entries: {sorted(extra)}')
        return directory / 'Kernels.lean'
    sources = sorted(directory.glob('*.lean'))
    if len(sources) > 1:
        raise ValueError(f'Ambiguous legacy port: {directory}')
    return sources[0] if sources else None


def source_files(root: Path) -> list[Path]:
    return [source for directory in sorted(root.iterdir()) if directory.is_dir()
            if (source := source_file(directory)) is not None]


def proof_files(directory: Path) -> list[Path]:
    source = source_file(directory)
    if source is None:
        return []
    return sorted(directory.rglob('*.lean'))


def specification_files(root: Path) -> list[Path]:
    files = []
    for source in source_files(root):
        if source.name == 'Kernels.lean':
            files.extend(source.parent / name for name in ('Correct.lean', 'FPEquiv.lean'))
        else:
            files.append(source)
    return sorted(files)


def combined_text(source: Path) -> str:
    """Source first, then entry/proof text, for existing whole-port scans."""
    return '\n'.join(p.read_text() for p in [source] +
                     [p for p in proof_files(source.parent) if p != source])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('source', 'sources', 'proofs', 'all-proofs', 'specs'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.kind == 'source':
        source = source_file(args.directory)
        files = [source] if source else []
    elif args.kind == 'sources':
        files = source_files(args.directory)
    elif args.kind == 'specs':
        files = specification_files(args.directory)
    elif args.kind == 'all-proofs':
        files = [p for source in source_files(args.directory)
                 for p in proof_files(source.parent)]
    else:
        files = proof_files(args.directory)
    for file in files:
        print(file)


if __name__ == '__main__':
    main()

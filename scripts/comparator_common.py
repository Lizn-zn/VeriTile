"""Shared official-comparator configuration, isolation, and tool checks."""
import hashlib
import json
import os
import shutil
import subprocess

AXIOMS = ['propext', 'Quot.sound', 'Classical.choice']


def snapshot_project(root, destination, libraries=("ComparatorChallenge", "ComparatorSolution"),
                     *, submodules=False):
    """Copy trusted inputs to an independent judge; never share writable caches."""
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
        for library in libraries:
            output.write(f'\n[[lean_lib]]\nname = "{library}"\n')
            if submodules:
                output.write(f'globs = ["{library}.+"]\n')
    return hashes


def write_config(directory, theorems, *, challenge="ComparatorChallenge",
                 solution="ComparatorSolution", filename="comparator.json"):
    if not theorems or any(not name.strip() for name in theorems):
        raise ValueError('At least one nonempty, fully qualified theorem name is required.')
    config = {
        'challenge_module': challenge,
        'solution_module': solution,
        'theorem_names': list(dict.fromkeys(theorems)),
        'permitted_axioms': AXIOMS,
        'enable_nanoda': False,
    }
    path = directory / filename
    path.write_text(json.dumps(config, indent=2) + '\n')
    return path


def comparator_command(directory, comparator, config=None):
    # Follow upstream's AF_UNIX restriction as well as comparator's landrun
    # sandbox. Failure to start either is a failed check, never a fallback.
    return ['systemd-run', '--user', '--pipe', '--wait', '--collect',
            '--property=RestrictAddressFamilies=~AF_UNIX',
            '--setenv=PATH=' + os.environ['PATH'],
            '--working-directory=' + str(directory),
            'lake', 'env', comparator, str(config or directory / 'comparator.json')]


def require_tools():
    comparator = shutil.which(os.environ.get('COMPARATOR_BIN', 'comparator'))
    if not comparator:
        raise ValueError('Official comparator is required; run scripts/setup-comparator.sh and add its bin directory to PATH.')
    for tool in ('lake', 'lean4export', 'landrun', 'systemd-run'):
        if not shutil.which(tool):
            raise ValueError(f'Missing {tool}; see scripts/README.md for setup.')
    subprocess.run(['systemd-run', '--user', '--pipe', '--wait', '--collect',
                    '--property=RestrictAddressFamilies=~AF_UNIX', 'true'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return comparator

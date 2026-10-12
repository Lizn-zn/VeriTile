"""Inventory block-load padding in the vendored JIT sources (no GPU imports).

An unspecified site is not itself a kernel bug: its lanes might be in bounds,
unobservable, or subject to an explicit unspecified-value contract. It does
require a corresponding distinction in the Lean translation and semantics.
"""
import argparse
import json
from pathlib import Path

from audit_source import python_block_load_padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit every checked load, including explicit zero padding")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sites = []
    for path in sorted((root / "bench/tritonbench_g").glob("*/*.py")):
        for site in python_block_load_padding(path.read_text()):
            sites.append({"file": str(path.relative_to(root)), **site})
    if args.json:
        print(json.dumps(sites, indent=2))
    else:
        unresolved = [site for site in sites if site["padding"] != "zero"]
        for site in unresolved:
            print(f'{site["file"]}:{site["line"]}: {site["kernel"]}: '
                  f'{site["padding"]}; boundary_check={site["boundary_check"]}')
        print(f'{len(sites)} checked loads; {len(unresolved)} without explicit zero padding '
              f'in {len({site["file"] for site in unresolved})} source files')


if __name__ == "__main__":
    main()

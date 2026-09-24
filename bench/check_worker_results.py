#!/usr/bin/env python3
"""Reject missing, duplicate, unexpected, or malformed worker results."""
from collections import Counter
import re
import sys


def validate(expected: str, results: str) -> None:
    targets = expected.splitlines()
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("expected targets must be nonempty and unique")
    actual = []
    for line in results.splitlines():
        match = re.fullmatch(r"  (?:ok    |FAIL  )(\S+)(?:   \(prep error\))?", line)
        if not match:
            raise ValueError(f"malformed worker result: {line!r}")
        actual.append(match[1])
    if Counter(actual) != Counter(targets):
        missing = sorted((Counter(targets) - Counter(actual)).elements())
        extra = sorted((Counter(actual) - Counter(targets)).elements())
        raise ValueError(f"incomplete worker results: missing={missing}, extra={extra}")


if __name__ == "__main__":
    try:
        validate(*sys.argv[1:])
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

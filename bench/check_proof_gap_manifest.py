#!/usr/bin/env python3
"""Check the TritonBench-G proof-gap manifest.

The #139 audit made every public Python path discoverable through an
`output_summary`.  Issue #146 tracks the stricter question: which of those
summaries are full value-level proofs, and which still depend on proof slices,
precomputed values, or explicit semantic blockers.

This script derives a conservative classification from the Lean source and
compares it with `bench/tritonbench_g/proof_gap_manifest.tsv`.  Use
`--write` after intentionally changing Lean summaries.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTS_ROOT = ROOT / "bench" / "tritonbench_g"
MANIFEST = PORTS_ROOT / "proof_gap_manifest.tsv"

ISSUE_BY_FAMILY = {
    "attention-softmax-accumulator": "#149",
    "attention-final-store-lift": "#161",
    "attention-online-softmax-recurrence": "#162",
    "fixed-width-int8-cast-semantics": "#154",
    "loss-reduction-aggregation": "#151",
    "bmm-final-store-accumulator": "#148",
    "dequant-matmul-cross-kernel-surface": "#148",
    "gemv-k-loop-accumulator": "#148",
    "iv-dependent-matmul-output-store": "#148",
    "matmul-activation-tail-accumulator": "#148",
    "matmul-dot-accumulator": "#148",
    "matmul-output-store-accumulator": "#148",
    "matmul-tma-output-store-accumulator": "#148",
    "proof-slice-precomputed-value": "#152",
    "quantization-semantic-followup": "#158",
    "rope-head-slice-lift": "#153",
    "rotary-2d-tile-value-lift": "#153",
    "recurrent-cumsum-loop": "#150",
    "reduction-layernorm-aggregation": "#151",
    "rotary-cache-path": "#153",
    "semantic-blocker": "#152",
}

SUMMARY_RE = re.compile(r"\b(?:theorem|abbrev)\s+([A-Za-z0-9_'.]*output_summary[A-Za-z0-9_'.]*)\b")

GAP_MARKERS = (
    "proof-oriented",
    "precomputed",
    "outside this slice",
    "outside the current",
    "outside this triton",
    "outside this proof",
    "outside this coverage",
    "not claimed",
    "not overclaimed",
    "represented by",
    "slice",
    "store slice",
    "final-store",
    "final store",
    "one-row",
    "one-block",
    "one-tile",
    "single-tile",
    "single-iteration",
    "current arithmetic layer",
    "llrint",
    "rounding",
    "packing",
)

FULL_VALUE_MARKERS = (
    "end-to-end",
    "without a precomputed",
    "no precomputed",
    "exact python-observable",
    "from `s` to `o`",
    "from `x` to `o`",
)


@dataclass(frozen=True)
class Summary:
    file: str
    declaration: str
    coverage_level: str
    blocker_family: str
    issue: str
    evidence: str


def context_for(lines: list[str], idx: int) -> str:
    start = max(0, idx - 18)
    end = min(len(lines), idx + 38)
    return "\n".join(lines[start:end])


def family_for(name: str, text: str) -> str:
    hay = f"{name}\n{text}".lower()
    if ("blocked_output_summary" in name.lower() or "blocked" in hay) and any(
        k in hay
        for k in (
            "llrint",
            "to(tl.int8)",
            ".to(tl.int8)",
            "int8",
            "quantize",
            "rounding/cast",
        )
    ):
        return "fixed-width-int8-cast-semantics"
    if "blocked_output_summary" in name.lower() or "blocked" in hay:
        return "semantic-blocker"
    if "dequantize_matmul" in hay:
        return "dequant-matmul-cross-kernel-surface"
    if "outside this triton" in hay and "matmul" in hay:
        return "matmul-dot-accumulator"
    if "batched_vecmat" in hay or "vecmat" in hay:
        return "gemv-k-loop-accumulator"
    if "bmm_chunk" in hay:
        return "bmm-final-store-accumulator"
    if "iv_dependent_matmul" in hay:
        return "iv-dependent-matmul-output-store"
    if "matmul_tma" in hay:
        return "matmul-tma-output-store-accumulator"
    if any(k in hay for k in ("matmul_leakyrelu", "matmul_autotune")):
        return "matmul-activation-tail-accumulator"
    if any(k in hay for k in ("matmul_kernel", "matmul_triton")):
        return "matmul-output-store-accumulator"
    if any(k in hay for k in ("attention", "attn", "softmax", "flash", "decode", "score", "prob")):
        if any(k in hay for k in ("final-store", "final store", "store slice", "precomputed")):
            return "attention-final-store-lift"
        return "attention-online-softmax-recurrence"
    if any(k in hay for k in ("matmul", "gemm", "dot", "bmm", "vecmat", "conv2d")):
        return "matmul-dot-accumulator"
    if any(k in hay for k in ("cumsum", "recurrent", "recurrence", "rwkv", "hgrn", "gla", "retention")):
        return "recurrent-cumsum-loop"
    if "int8" in hay and any(k in hay for k in ("unprojected", "cast semantics missing")):
        return "fixed-width-int8-cast-semantics"
    if any(k in hay for k in ("int8", "int4", "uint8", "quant", "llrint", "round", "pack")):
        return "quantization-semantic-followup"
    if any(k in hay for k in ("rope_transform", "rope_backward", "one-head proof", "head slices")):
        return "rope-head-slice-lift"
    if "rotary" in hay and any(
        k in hay
        for k in (
            "full 2d value proof",
            "one-row",
            "proof-oriented `o0`/`o1`",
            "cast-load simp extension",
        )
    ):
        return "rotary-2d-tile-value-lift"
    if any(k in hay for k in ("rotary", "rope", "cache", "kv")):
        return "rotary-cache-path"
    if any(k in hay for k in ("norm", "layernorm", "rms", "reduction", "mean", "max", "sum")):
        return "reduction-layernorm-aggregation"
    if any(k in hay for k in ("cross_entropy", "ce_loss", "kldiv", "logsumexp")):
        return "loss-reduction-aggregation"
    if "slice" in hay or "precomputed" in hay:
        return "proof-slice-precomputed-value"
    return "none"


def evidence_for(text: str, level: str) -> str:
    lower = text.lower()
    if level == "blocked_summary":
        for marker in ("blocked_output_summary", "blocked", "current arithmetic layer", "llrint"):
            if marker in lower:
                return marker
        return "explicit blocked summary"
    if level == "public_summary_with_proof_gap":
        for marker in GAP_MARKERS:
            if marker in lower:
                return marker
        return "summary lacks full-value marker"
    for marker in FULL_VALUE_MARKERS:
        if marker in lower:
            return marker
    return "no proof-gap marker in summary context"


def classify(name: str, text: str) -> tuple[str, str, str]:
    lower_name = name.lower()
    lower = text.lower()
    if "blocked_output_summary" in lower_name or re.search(r"\bblocked\b", lower):
        level = "blocked_summary"
    elif any(marker in lower for marker in GAP_MARKERS):
        level = "public_summary_with_proof_gap"
    else:
        level = "full_value_candidate"

    family = family_for(name, text) if level != "full_value_candidate" else "none"
    issue = ISSUE_BY_FAMILY[family] if level != "full_value_candidate" else ""
    return level, family, issue


def collect() -> list[Summary]:
    rows: list[Summary] = []
    for lean_file in sorted(PORTS_ROOT.glob("*/*.lean")):
        rel = lean_file.relative_to(ROOT).as_posix()
        lines = lean_file.read_text().splitlines()
        for i, line in enumerate(lines):
            match = SUMMARY_RE.search(line)
            if not match:
                continue
            name = match.group(1)
            ctx = context_for(lines, i)
            level, family, issue = classify(name, ctx)
            rows.append(
                Summary(
                    file=rel,
                    declaration=name,
                    coverage_level=level,
                    blocker_family=family,
                    issue=issue,
                    evidence=evidence_for(ctx, level),
                )
            )
    return rows


def write_manifest(rows: list[Summary]) -> None:
    with MANIFEST.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "file",
                "declaration",
                "coverage_level",
                "blocker_family",
                "issue",
                "evidence",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.file,
                    row.declaration,
                    row.coverage_level,
                    row.blocker_family,
                    row.issue,
                    row.evidence,
                ]
            )


def read_manifest() -> str:
    return MANIFEST.read_text() if MANIFEST.exists() else ""


def render(rows: list[Summary]) -> str:
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(
        ["file", "declaration", "coverage_level", "blocker_family", "issue", "evidence"]
    )
    for row in rows:
        writer.writerow(
            [row.file, row.declaration, row.coverage_level, row.blocker_family, row.issue, row.evidence]
        )
    return buf.getvalue()


def print_summary(rows: list[Summary]) -> None:
    by_level: dict[str, int] = {}
    by_family: dict[str, int] = {}
    files = {row.file for row in rows}
    for row in rows:
        by_level[row.coverage_level] = by_level.get(row.coverage_level, 0) + 1
        if row.coverage_level != "full_value_candidate":
            by_family[row.blocker_family] = by_family.get(row.blocker_family, 0) + 1

    print(f"proof-gap manifest summaries: {len(rows)} declarations across {len(files)} files")
    for key in sorted(by_level):
        print(f"  {key}: {by_level[key]}")
    for key in sorted(by_family):
        print(f"  blocker {key}: {by_family[key]}")


def validate_rows(rows: list[Summary]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    valid_levels = {
        "full_value_candidate",
        "public_summary_with_proof_gap",
        "blocked_summary",
    }
    for row in rows:
        key = (row.file, row.declaration)
        if key in seen:
            errors.append(f"duplicate manifest key: {row.file}::{row.declaration}")
        seen.add(key)
        if row.coverage_level not in valid_levels:
            errors.append(f"{row.file}::{row.declaration}: bad coverage_level {row.coverage_level}")
        if row.coverage_level == "full_value_candidate":
            if row.issue:
                errors.append(f"{row.file}::{row.declaration}: full candidate should not link an issue")
            if row.blocker_family != "none":
                errors.append(f"{row.file}::{row.declaration}: full candidate should use blocker_family=none")
        else:
            if row.blocker_family == "none":
                errors.append(f"{row.file}::{row.declaration}: proof gap needs blocker_family")
            expected_issue = ISSUE_BY_FAMILY.get(row.blocker_family)
            if row.issue != expected_issue:
                errors.append(
                    f"{row.file}::{row.declaration}: {row.blocker_family} must link {expected_issue}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite the manifest from Lean source")
    args = parser.parse_args()

    rows = collect()
    errors = validate_rows(rows)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    generated = render(rows)
    if args.write:
        write_manifest(rows)
        print_summary(rows)
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
        return 0

    if read_manifest() != generated:
        print(f"{MANIFEST.relative_to(ROOT)} is stale; run bench/check_proof_gap_manifest.py --write")
        return 1

    print_summary(rows)
    print("ok proof-gap manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

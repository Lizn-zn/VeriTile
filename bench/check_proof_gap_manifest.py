#!/usr/bin/env python3
"""Check the TritonBench-G proof-gap manifest.

The #139 audit made every public Python path discoverable through an
`output_summary`.  Issue #146 tracks the stricter question: which of those
summaries are full value-level proofs, and which still depend on proof slices,
precomputed values, or explicit semantic blockers.

This script derives a conservative classification from the Lean source and
compares it with `bench/tritonbench_g/proof_gap_manifest.tsv`.  Use
`--write` after intentionally changing Lean summaries.

Classification uses hard signals only, in priority order: an explicit
`coverage:` annotation in the summary docstring, an explicit
`blocked_output_summary` declaration name, and a genuinely self-referential
`expected` (shared detector with `scripts/spec_sheet.py`).  Docstring prose is
deliberately not keyword-sniffed: it misfiled genuine summaries whose
docstrings merely describe slices, precomputed input caches, or trusted carry
boundaries.  Headline discovery mirrors `scripts/spec_sheet.py`'s tiers so
every kernel file contributes rows.
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
    "attention-backward-score-reduction": "#162",
    "attention-context-decode-reduction": "#162",
    "attention-final-store-lift": "#161",
    "attention-forward-online-softmax-recurrence": "#162",
    "attention-fwd-triton1-bo-bhpre-producers": "#165",
    "attention-score-probability-reduction": "#162",
    "context-attention-mistral-sliding-window-acc-store": "#167",
    "context-attention-nopad-varlen-acc-store": "#167",
    "context-attention-streaming-acc-store": "#167",
    "dense-attention-online-softmax-recurrence": "#199",
    "dense-attention-acc-store": "#166",
    "flash-decode-normalized-vector-store": "#168",
    "flash-decode-llama-stage2-normalization": "#171",
    "flash-decode-llama-running-max-recurrence": "#181",
    "flash-decode-phi-stage2-normalization": "#172",
    "flash-decode-phi-running-max-recurrence": "#175",
    "flash-decode-phi-masked-accumulator-recurrence": "#176",
    "flash-decode-phi-normalization-store": "#177",
    "attention-online-softmax-recurrence": "#162",
    "fixed-width-int8-cast-semantics": "#154",
    "loss-reduction-aggregation": "#151",
    "chunk-delta-forward-recurrence-store": "#190",
    "layernorm-backward-residual-recompute-aggregation": "#191",
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
    "chunk-cumsum-carry-fold": "#185",
    "decay-cumsum-scan-fold": "#186",
    "gla-output-tile-producer": "#188",
    "recurrent-cumsum-loop": "#150",
    "recurrent-state-loop-fold": "#187",
    "reverse-cumsum-directional-scan": "#94",
    "reduction-layernorm-aggregation": "#151",
    "rotary-cache-path": "#153",
    "semantic-blocker": "#152",
    "token-attention-reduction": "#162",
    "self-referential-output-value": "#146",
}

def strip_lean_comments(text: str) -> str:
    """Blank out `--` line comments and (nested) `/- ... -/` block comments,
    preserving line structure, so keyword-at-line-start decl scans cannot
    match prose (e.g. a docstring line starting with "specification of...")."""
    out: list[str] = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if depth == 0 and two == "--":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth > 0 and two == "-/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth > 0:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


SUMMARY_RE = re.compile(r"\b(?:theorem|specification|abbrev)\s+([A-Za-z0-9_'.]*output_summary[A-Za-z0-9_'.]*)\b")

# Public-headline discovery keys on the `specification` declaration keyword
# (VeriTile/Meta/Specification.lean): a headline IS a `specification` decl.
# This retires the name-suffix HEADLINE_TIERS heuristics (which could be
# shadowed by helper suffixes) — mirroring scripts/spec_sheet.py.

DECL_LINE_RE = re.compile(
    r"^(?:noncomputable\s+|private\s+)*(?:theorem|specification|abbrev)\s+([A-Za-z0-9_'.]+)")
SPEC_LINE_RE = re.compile(
    r"^(?:noncomputable\s+|private\s+)*specification\s+([A-Za-z0-9_'.]+)")

# Explicit per-summary coverage annotation, written in the summary docstring:
#   coverage: public_summary_with_proof_gap family=<blocker-family> -- <evidence>
#   coverage: blocked_summary family=<blocker-family> -- <evidence>
#   coverage: full_value_candidate -- <evidence>
# An explicit annotation takes priority over every heuristic. Use it when a
# summary is genuinely partial (slice-/tile-scoped) or when the default
# classification is wrong; prose in docstrings is otherwise NOT sniffed for
# gap keywords (that heuristic misfiled genuine summaries whose docstrings
# merely *describe* slices or trusted boundaries).
COVERAGE_ANNOTATION_RE = re.compile(
    r"coverage:\s*(full_value_candidate|public_summary_with_proof_gap|blocked_summary)"
    r"(?:\s+family=([A-Za-z0-9_-]+))?"
    r"(?:\s*--\s*([^\n]*))?")

# A "self-referential" output value is a `def` whose body re-executes the kernel
# and reads the kernel's own store back out (`match exec ... | some s' => s'.readMem ...`).
# When a summary's `expected` is such a value, the value half of its
# `ComputeCorrect.Realizes` is a tautology (`output = output`): it proves only
# surface-lowering + progress, NOT that the computed values are correct. Such
# summaries must be recorded as proof gaps, not `full_value_candidate`.
SELF_REF_EXEC_RE = re.compile(r"\bmatch\s+exec\b")
DECL_SPLIT_RE = re.compile(
    r"\n(?=(?:noncomputable\s+|private\s+)*(?:def|theorem|specification|abbrev)\s|/--|@\[|namespace\s|end\s)")
_DECL_HEAD_RE = re.compile(
    r"^(?:noncomputable\s+|private\s+)*(def|theorem|specification|abbrev)\s+([A-Za-z0-9_']+)")
DEF_NAME_RE = re.compile(r"^(?:noncomputable\s+|private\s+)*def\s+([A-Za-z0-9_']+)")


def find_self_ref_value_defs(full_text: str) -> set[str]:
    """Names of value `def`s whose body re-reads the kernel's own execution."""
    names: set[str] = set()
    for block in DECL_SPLIT_RE.split(full_text):
        m = DEF_NAME_RE.match(block.lstrip("\n"))
        if m and SELF_REF_EXEC_RE.search(block):
            names.add(m.group(1))
    return names


def build_decl_blocks(full_text: str) -> dict[str, tuple[str, str]]:
    """Map each declaration name to `(kind, block_text)` (kind ∈ def/theorem/abbrev)."""
    blocks: dict[str, tuple[str, str]] = {}
    for block in DECL_SPLIT_RE.split(full_text):
        m = _DECL_HEAD_RE.match(block.lstrip("\n"))
        if m:
            blocks.setdefault(m.group(2), (m.group(1), block))
    return blocks


def _alias_target(block: str, decl_blocks: dict[str, tuple[str, str]]) -> str | None:
    """For an `abbrev … := TARGET …` block, return the aliased declaration name."""
    m = re.search(r":=\s*([A-Za-z_][A-Za-z0-9_'.]*)", block)
    if not m:
        return None
    head = m.group(1).split(".")[0]
    return head if head in decl_blocks else None


def resolved_summary_text(name: str, decl_blocks: dict[str, tuple[str, str]]) -> str:
    """Text of the summary declaration, following `abbrev` alias chains to the
    underlying theorem that actually carries the `expected :=` clause."""
    texts: list[str] = []
    seen: set[str] = set()
    cur: str | None = name
    while cur and cur not in seen and cur in decl_blocks:
        seen.add(cur)
        kind, block = decl_blocks[cur]
        texts.append(block)
        cur = _alias_target(block, decl_blocks) if kind == "abbrev" else None
    return "\n".join(texts)


def summary_expected_is_self_ref(text: str, self_ref_names: set[str]) -> bool:
    """True when the summary's `expected :=` refers to a self-referential value."""
    if not self_ref_names:
        return False
    marker = text.find("expected :=")
    region = text[marker:] if marker != -1 else text
    return any(re.search(r"\b" + re.escape(n) + r"\b", region) for n in self_ref_names)

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
    for j in range(idx - 1, start - 1, -1):
        if DECL_LINE_RE.match(lines[j]):
            start = j + 1
            break
    for j in range(idx - 1, start - 1, -1):
        if lines[j].lstrip().startswith("/--"):
            start = j
            break
    end = min(len(lines), idx + 38)
    for j in range(idx + 1, end):
        if DECL_LINE_RE.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end])


def family_for(name: str, text: str) -> str:
    lname = name.lower()
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
    if "attention_fwd_triton1" in hay:
        return "attention-fwd-triton1-bo-bhpre-producers"
    if "attention_kernel" in hay and "online-softmax recurrence" in hay:
        return "dense-attention-online-softmax-recurrence"
    if "attention_kernel" in hay:
        return "dense-attention-acc-store"
    if "context_attn_mistral" in hay:
        return "context-attention-mistral-sliding-window-acc-store"
    if "context_attn_nopad" in hay:
        return "context-attention-nopad-varlen-acc-store"
    if "context-attention" in hay:
        return "context-attention-streaming-acc-store"
    if "flash_decode2_llama" in lname and "running_max" in lname:
        return "flash-decode-llama-running-max-recurrence"
    if "flash_decode2_llama" in hay:
        return "flash-decode-llama-stage2-normalization"
    if "flash_decode2_phi" in lname and "running_max" in lname:
        return "flash-decode-phi-running-max-recurrence"
    if "flash_decode2_phi" in lname and "masked_accumulator" in lname:
        return "flash-decode-phi-masked-accumulator-recurrence"
    if "flash_decode2_phi" in lname and "normalization_store" in lname:
        return "flash-decode-phi-normalization-store"
    if "flash_decode2_phi" in hay:
        return "flash-decode-phi-stage2-normalization"
    if "flash_decode2" in hay:
        return "flash-decode-normalized-vector-store"
    if "triton_attention_bwd" in lname:
        return "attention-backward-score-reduction"
    if any(k in lname for k in ("token_attn", "token_softmax", "softmax_reducev")):
        return "token-attention-reduction"
    if any(k in lname for k in ("attention_score", "ksoftmax", "mixed_sparse_attention", "block_sparse_attn")):
        return "attention-score-probability-reduction"
    if any(k in lname for k in ("context_attn_bloom", "context_attn_fwd", "context_attn_llama")):
        return "attention-context-decode-reduction"
    if any(
        k in lname
        for k in (
            "attention_forward",
            "attention_fwd_triton2",
            "attention_fwd_triton3",
            "attn_fwd",
            "chunk_gated_attention",
            "flash_attn",
            "lightning_attention",
            "triton_attention_forward",
        )
    ):
        return "attention-forward-online-softmax-recurrence"
    if any(k in hay for k in ("attention", "attn", "softmax", "flash", "decode", "score", "prob")):
        if any(k in hay for k in ("final-store", "final store", "store slice", "precomputed")):
            return "attention-final-store-lift"
        return "attention-online-softmax-recurrence"
    if any(k in hay for k in ("matmul", "gemm", "dot", "bmm", "vecmat", "conv2d")):
        return "matmul-dot-accumulator"
    if "reversed_cumsum" in hay:
        return "reverse-cumsum-directional-scan"
    if "chunk_gla_simple" in hay:
        return "gla-output-tile-producer"
    if "decay_cumsum" in hay:
        return "decay-cumsum-scan-fold"
    if any(k in hay for k in ("chunk_cumsum", "chunked_cumsum")):
        return "chunk-cumsum-carry-fold"
    if any(k in hay for k in ("chunk_gate_recurrence", "fused_recurrent_hgrn", "fused_recurrent_rwkv6")):
        return "recurrent-state-loop-fold"
    if "chunk_delta_fwd" in lname:
        return "chunk-delta-forward-recurrence-store"
    if any(k in hay for k in ("cumsum", "recurrent", "recurrence", "rwkv", "hgrn", "gla", "retention")):
        return "recurrent-cumsum-loop"
    if "layer_norm_ops_bwd_plain_bias" in lname and any(
        k in lname for k in ("residual", "recompute")
    ):
        return "layernorm-backward-residual-recompute-aggregation"
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


def evidence_for(text: str, level: str, self_ref: bool = False) -> str:
    lower = text.lower()
    if level == "blocked_summary":
        for marker in ("blocked_output_summary", "current arithmetic layer", "llrint"):
            if marker in lower:
                return marker
        return "explicit blocked summary"
    if level == "public_summary_with_proof_gap":
        if self_ref:
            return "self-referential expected (= executed kernel output)"
        return "explicit coverage annotation"
    for marker in FULL_VALUE_MARKERS:
        if marker in lower:
            return marker
    return "no proof-gap marker in summary context"


def classify(name: str, text: str, self_ref: bool = False) -> tuple[str, str, str, str | None]:
    """Classify a public summary. Signals, in priority order:

    1. an explicit `coverage:` annotation in the summary docstring;
    2. an explicit `blocked_output_summary` declaration name;
    3. a genuinely self-referential `expected` (the one hard value-gap signal,
       shared with `scripts/spec_sheet.py`);
    4. otherwise `full_value_candidate`.

    Docstring prose is deliberately NOT sniffed for gap keywords: honest
    docstrings of genuine summaries routinely *describe* proof slices,
    precomputed input caches, or trusted carry boundaries, and keyword
    matching misfiled 17 genuine summaries (e.g. `slice` matching the kernel
    *name* `bgmv_expand_slice`, `precomputed` matching "without a
    precomputed"). Genuinely partial summaries must say so explicitly via
    `coverage:`. Returns `(level, family, issue, explicit_evidence)`.
    """
    ann = COVERAGE_ANNOTATION_RE.search(text)
    if ann:
        level = ann.group(1)
        if level == "full_value_candidate":
            family = "none"
        else:
            family = ann.group(2) or family_for(name, text)
        issue = ISSUE_BY_FAMILY.get(family, "") if level != "full_value_candidate" else ""
        return level, family, issue, (ann.group(3) or "").strip() or "explicit coverage annotation"

    if "blocked_output_summary" in name.lower():
        level = "blocked_summary"
    elif self_ref:
        level = "public_summary_with_proof_gap"
    else:
        level = "full_value_candidate"

    if level == "full_value_candidate":
        family = "none"
    else:
        family = family_for(name, text)
        if family == "none" and self_ref:
            family = "self-referential-output-value"
    issue = ISSUE_BY_FAMILY.get(family, "") if level != "full_value_candidate" else ""
    return level, family, issue, None


def headline_names(full_text: str) -> set[str]:
    """Per-file public headline declarations: every `specification` decl
    (the keyword marks the public spec surface), plus every
    `*output_summary*` decl for backward compatibility."""
    decls: list[str] = []
    picked: set[str] = set()
    for raw in strip_lean_comments(full_text).splitlines():
        m = DECL_LINE_RE.match(raw)
        if not m:
            continue
        decls.append(m.group(1))
        if SPEC_LINE_RE.match(raw):
            picked.add(m.group(1))
    picked.update(n for n in decls if SUMMARY_RE.search(f"theorem {n} "))
    return picked


def collect() -> list[Summary]:
    rows: list[Summary] = []
    for lean_file in sorted(PORTS_ROOT.glob("*/*.lean")):
        rel = lean_file.relative_to(ROOT).as_posix()
        full_text = lean_file.read_text()
        self_ref_names = find_self_ref_value_defs(full_text)
        decl_blocks = build_decl_blocks(full_text)
        lines = full_text.splitlines()
        wanted = headline_names(full_text)
        emitted: set[str] = set()
        for i, line in enumerate(lines):
            match = DECL_LINE_RE.match(line)
            if not match or match.group(1) not in wanted or match.group(1) in emitted:
                continue
            name = match.group(1)
            emitted.add(name)
            ctx = context_for(lines, i)
            # Resolve `abbrev` aliases to the underlying theorem so a summary that
            # merely aliases a `…_compute_correct` is checked against that target's
            # `expected :=` clause, not just the alias line.
            self_ref = summary_expected_is_self_ref(
                ctx + "\n" + resolved_summary_text(name, decl_blocks), self_ref_names)
            level, family, issue, explicit_evidence = classify(name, ctx, self_ref)
            rows.append(
                Summary(
                    file=rel,
                    declaration=name,
                    coverage_level=level,
                    blocker_family=family,
                    issue=issue,
                    evidence=explicit_evidence
                        if explicit_evidence is not None
                        else evidence_for(ctx, level, self_ref),
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

# Spec sheets — per-kernel specification review cards

These files are a **review aid for the *specification*** of each `tritonbench_g`
kernel, not the proof. The trustworthiness of a verified kernel rests entirely
on whether its `expected` closed form actually says what the kernel is supposed
to compute — and that closed form is otherwise scattered across a 1000+ line
proof file. Each sheet here collapses one kernel into a single self-contained
card so a human can audit the spec without spelunking.

## What a sheet contains

- **Python source** link — the kernel being specified.
- **Public theorem statement(s)** — verbatim, with the docstring. These are the
  headline `*_output_summary` / `*_compute_correct` / `*_closed_form_correct`
  theorems (the public spec), preferring the dimension-general `*_general`
  variant when one exists.
- **Assumptions / layout contracts** — the theorem's hypotheses (contiguity,
  shape, `≠` aliasing side-conditions) pulled into one bullet list.
- **Closed-form spec defs (transitive)** — the `expected` value's full
  definition closure, *plus* the kernel's DSL `surface` port, so the spec and
  the kernel sit side-by-side for cross-checking.
- **Self-ref audit flags** — a ⚠ fires if any def in the spec's trust path
  references `exec` / `produced*Value` / `*SurfaceValue` / executed-state
  `readMem`, i.e. the "closed form" would be defined in terms of the executed
  kernel output (tautological / fake). **A clean sheet has no such flag.**

## INDEX.md

Ranks all kernels by a **review-cost proxy**

    score = 3·(transitive spec defs) + (flat-offset reads) + (statement lines) + (hypotheses)

hardest-to-audit first. The `flat-offset reads` column counts
`readMem R (… * … + …)` patterns — the main readability tax (flat offset
arithmetic the reviewer must decode against the kernel strides).

## Regenerating

These sheets are **generated artifacts**; do not hand-edit them. After changing
any kernel spec, regenerate:

    python3 scripts/spec_sheet.py            # → bench/tritonbench_g/spec-sheets/

To print a single kernel's sheet to stdout (e.g. while reviewing a diff):

    python3 scripts/spec_sheet.py bench/tritonbench_g/<kernel>/<File>.lean

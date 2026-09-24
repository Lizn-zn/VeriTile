# VeriTile project website

Public project homepage and documentation, built with Astro +
[Starlight](https://starlight.astro.build/).
Published in English, deployed to
[lizn-zn.github.io/VeriTile/](https://lizn-zn.github.io/VeriTile/) via the
`Website` GitHub Actions workflow.

## Quick start

```bash
./site/scripts/dev.sh           # dev server on http://localhost:4321/VeriTile/
./site/scripts/dev.sh build     # one-shot production build to dist/
./site/scripts/dev.sh preview   # build + serve the production output
```

Requires [Bun](https://bun.sh) for dependency installation and Node.js 22.12+
for Astro 7. Run from a complete repository checkout: the homepage reads
the benchmark inventory and vector-add example at build time.

The lockfile uses Astro 7 / Starlight 0.42 / Sharp 0.35. Compatible transitive
overrides keep devalue, js-yaml, postcss-selector-parser, and smol-toml above
their patched minimums; remove an override only after the upstream dependency
graph resolves to patched versions without it. Run `bun audit` and the build
and link checks after dependency changes. Registry severity is not evidence
of a deployed exploit: this site serves static output, while build-time input
processing remains part of the dependency review.

## Layout

```text
site/
├── astro.config.mjs            ← site URL, base path, sidebar, language
├── package.json                ← dev / build / preview / astro scripts
├── public/                     ← static files copied as-is
├── scripts/
│   ├── dev.sh                  ← one-touch dev server (entry point)
│   ├── migrate-docs.sh         ← re-sync content from ../documents/
│   ├── sync-docs.py            ← generation and --check for stale copies
│   ├── check-site.py           ← built-page links and repository paths
│   ├── check-doc-api.py        ← Lean resolution of documented public names
│   └── record-home-demo.py     ← real Lean records for the interactive example
└── src/
    ├── components/Hero.astro   ← project homepage
    ├── components/ProjectStory.astro ← project contributions and their evidence
    ├── components/KernelDemo.astro ← fixed kernel variants, sample values, proof records
    ├── components/Header.astro ← homepage navigation; standard header on docs
    ├── components/Footer.astro ← homepage footer; standard footer on docs
    ├── components/CorpusStats.astro ← build-time inventory table
    ├── lib/project-data.ts     ← counts and code excerpts from the repository
    ├── styles/theme.css        ← engineering-notebook theme tokens
    └── content/docs/           ← English pages (no locale prefix)
```

## Authoring

Homepage copy is maintained in `src/lib/home-story.ts` and
`src/lib/home-copy.ts`; `src/components/Hero.astro`
renders the shared layout and `ProjectStory.astro` renders the contributions.
Write all website content in English, using descriptive headings and factual prose. Name the
feature, theorem, workflow, or result directly; avoid slogans, rhetorical
questions, and paired promotional sentences.
The displayed port count comes from directories containing both Python and
Lean files; it is an inventory count, not a claim that all hardware behavior
is verified. The vector-add excerpt is read directly from its Lean source.
The status pages use the same inventory and link to the full audit records.

### Homepage structure and sources

The homepage uses warm paper colors, an amber accent, restrained monospace
typography, and a narrower reading column. Its three main topics follow the
original deck: Triton-style kernel syntax embedded in Lean with explicit translation choices, mathematical correctness and optimization equivalence, and a lemma
library with agent proof tooling. Each has an accompanying source example or
workflow; detailed semantics and assumptions live in the documentation.
FlashAttention and the benchmark audit provide concrete results.
Copyable build commands close the page.
`src/lib/home-story.ts` contains this narrative; `home-copy.ts`
contains the interactive example and shared actions.

The Python and DSL excerpts come from `bench/examples/VectorAdd.lean`.
The language comparison shows only the kernel bodies, with corresponding names
matched for readability and the aligned, unmasked scope stated in the caption.
The interactive demo retains the exact verified kernel text.
The View proof links open the full Lean contracts and proofs in
`bench/tritonbench_g/logsumexp_fwd/LogsumexpFwd.lean` and
`bench/examples/StableLogSumExpEquiv.lean`.
Both cards use log-sum-exp. The correctness formula is explicitly the unscaled,
per-block active-lane case. The equivalence example compares direct and
maximum-shifted LSE, with real intermediate arithmetic and a shared rounded bf16
output store; it has no private scratch.
The Correctness and Equivalence cards share one template and matching
Computation / Memory rows. Their short relations use explicitly schematic names:
`lse_kernel ⊨ lse_spec` on the left and `direct_lse ≡ stable_lse` on the right.
These are presentation labels, not aliases defined in Lean. Both LSE examples
have matching Proof details and View proof controls. Example
visuals, model assumptions, and preconditions are inside the details; full Lean
statements are available through the source links. The LSE equivalence contract compares output values
at the shared output address and preserves every other cell; it does not require
identical intermediate operations.
The featured FlashAttention link covers the forward causal/non-causal
reference contracts with their boundary/D-tail conditions.
The benchmark shows paired-port inventory and unique theorem statements from
`proof_gap_manifest.tsv`, with prominent links to the recorded completion
audit and per-kernel scope. Counts do not label every statement fully proved.

The agent section uses two panels: reusable lemmas on the left, proof generation
on the right. Four lemma groups link directly to the loop-invariant, matrix
multiplication, masked-store, and address-injectivity modules. The workflow
starts with a kernel and specification, shows the agent / Lean feedback cycle,
and labels the checked proof as the successful outcome. Running instructions,
retry limits, logs, and separately invoked artifact audits live behind the
Proving guide and Proof audits links. The diagram describes the workflow; it
does not display a live run or claim a particular generated proof.
VectorAdd is available in an expandable example under the two proof surfaces.

The demo switches between the original `x + y` and a fixed `x - y` mutation.
It displays **recorded** Lean results, not an in-browser Lean execution. The
original complete example must pass; the unchanged proof of the mutated
kernel must fail with unsolved goals. Two additional proofs establish the
displayed four-lane outputs directly from the DSL semantics. Sample values
illustrate one input; the original theorem quantifies over all inputs allowed
by its preconditions. Both results include the raw Lean diagnostics.

To regenerate the committed record after changing the Lean sources:

```bash
lake build
python3 site/scripts/record-home-demo.py
python3 site/scripts/record-home-demo.py --check
```

`src/lib/vector-add-record.json` stores these results. Its fingerprint covers
the example, library Lean sources, toolchain, dependency manifest, lakefile,
and recorder. The homepage build refuses a stale record. Failed variants are
created in a temporary directory; the canonical example is never mutated.
Regeneration requires Python 3 and the Lean dependencies; building the site
from a current record does not require Lean.

The variant selector supports keyboard navigation and announces changes to
screen readers. With JavaScript disabled, the original example and proof
remain readable; source and reproduction links remain available.

`../VeriTile-Deck.pptx` informs the positioning and examples. Its slide sequence
is not the website structure: proof automation and semantic boundaries live
in the overview and architecture docs, upstream bug reports live in project
status, and development plans are linked from the roadmap.

Corpus counts are derived from source rather than the deck's old snapshot.
Log-sum-exp is scoped to its per-block contracts, and LSE/FlashAttention link
to their current proofs. Audit records document axiom dependencies and
rounding assumptions; the homepage does not claim full hardware verification.

Each page in `src/content/docs/` is a Markdown / MDX file with frontmatter:

```yaml
---
title: Page title
description: One-line summary used in nav previews and SEO.
---
```

Internal links use absolute paths **with the `/VeriTile/` base prefix**:

```markdown
[Cookbook](/VeriTile/cookbook/)
```

The base prefix is currently hard-coded so the same Markdown serves dev
(`http://localhost:4321/VeriTile/...`) and prod
(`https://lizn-zn.github.io/VeriTile/...`).

Bench / scripts links should point to GitHub:

```markdown
[`bench/check_ports.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/check_ports.sh)
```

## Re-syncing from `documents/`

The English files in `documents/` are the source of truth for the design notes; the
architecture & proofs sections of this site are copies. To re-sync after
upstream edits:

```bash
cd site && ./scripts/migrate-docs.sh
```

The script extracts the H1 as `title:`, drops the bilingual switcher line,
and resolves relative links against the original document. Published design
notes link to their English site pages with the `/VeriTile/` prefix; other
repository files link to GitHub. Fenced code is preserved. It covers 13 design
notes in English, including semantic caveats and trust audits.

Checks:

```bash
./site/scripts/migrate-docs.sh --check  # generated pages match their sources
python3 site/scripts/check-site.py     # after building site/dist
python3 site/scripts/check-doc-api.py  # after lake build
```

The site workflow checks synchronization and built links. The existing artifact
gate checks that documented public API names resolve in Lean. These checks do
not establish the truth of every prose claim or compile schematic proof
fragments; changes to theorem meanings still require documentation review.

## Deployment

Pushing to `main` with changes under `site/**`, `documents/**`,
`bench/tritonbench_g/**`, the Lean toolchain, or the vector-add showcase triggers
`.github/workflows/site.yml`, which runs `bun install` + `bun run build`
and uploads `site/dist/` as a Pages artifact. First-time setup needs the
repo's **Settings → Pages → Source** set to **"GitHub Actions"**; after
that, every push is automatic.

The workflow can also be triggered manually from the Actions tab
(`workflow_dispatch`).

## Coverage table and complete tutorial

`/proofs/coverage/` displays every registered TritonBench-G headline with its
exact statement, relevant IO definitions, source links, numeric model, and
reviewed scope. Search and filters enhance an already readable HTML table.
`bench/coverage_review.py` validates `bench/tritonbench_g/coverage_review.json`
against complete Lean/Python source fingerprints and exact declaration/function
identities. Python 3 is required during site builds; Lean is not. After changing
a port, review its statements and excluded operations before updating its
fingerprints and regenerating `proof_gap_manifest.tsv`. Never refresh hashes
merely to silence the validator.

The historical completed corpus audit in `src/lib/coverage-ci.json` is pinned
to an exact commit and linked from every row. Refresh it only after a newer
full audit succeeds:

```bash
python3 scripts/record_coverage_ci.py RUN_ID  # from repository root; requires gh
```

Pending, failed, and skipped runs cannot be recorded. This record is historical
evidence, not a claim that later source commits passed; the page links to newer
runs. Building the coverage table does not replay proofs.

`/cookbook/vector-add-walkthrough/` walks from the Python kernel through the
checked DSL/IO contract, full proof, trust checks, and rejected mutation. Its
code excerpts are extracted from the executable `bench/examples/VectorAdd.lean`
by `src/lib/vector-add-tutorial.ts`; missing anchors fail the site build.

Typography separates reading paragraphs (`--vt-text-body`), expanded-example
explanations (`--vt-text-explanation`), supporting text and controls
(`--vt-text-small`), and code (`--vt-text-code`). At the default root size,
the homepage uses 14px paragraphs with a 1.65 line height, 13px expanded-example
explanations, 14px controls, and 12px code with a 1.7 line height. The hero title
scales from 32px on phones to a 48px desktop maximum. Documentation reading sizes
retain their existing scale and a 14px small-text/inline-code floor. All sizes
scale with user font preferences; mobile layouts wrap or stack content rather
than shrinking type further.

Homepage emphasis uses 600-weight headings and selected key claims in the
primary ink color. Proof outcomes have a status-colored edge and a 14px,
600-weight label, with their conditions and recorded-check provenance alongside.
Example entry points use the warm accent and a distinct background or border.
Supporting prose and code retain the type sizes above.

# Author feedback resolution — 2026-09-24

This follows the eight findings and six product/documentation suggestions in
`VeriTile-author-feedback-2026-09-23.zip`, which reviewed revision `bb54250b`.
The fixes cover projection-failure preservation, elaborated-environment
discovery, and mandatory official comparator checks. They also add explicit
statement-scope reviews, a browsable table, the complete tutorial, and
corrections to historical proof claims.

## Findings

| Feedback | Resolution and evidence |
|---|---|
| 1. KernelIO accepted projection failures through an empty program | Public IO constructors require successful projection. Exact, rounding, equivalence, denotation, dtype erasure, pipeline, and chain paths preserve rejection. `bench/tests/KernelIOProjection.lean` checks unsupported effects and composition. |
| 2. Worker failures could report success | Both runners validate concurrency, propagate launcher/worker failures, and require exactly one result per selected target. `scripts/test_audit_gates.py` injects launch failure, killed workers, missing, duplicate, partial, and unexpected output. |
| 3. Regex discovery missed kernels/specs | Lean environment audits discover elaborated declarations and their result types, including multiline/custom/macro declarations. They reject circular specs against actual kernel constants; IO-wrapper arguments do not count as kernels. Module audit fixtures and real runner tests cover these cases. |
| 4. Proof scope was overclaimed | All 345 current headlines have explicit reviews bound to 173 Lean/Python source pairs. The inventory distinguishes 8 original-kernel candidates, 296 configured models/stages, 37 precomputed-input slices, 3 pre-rounding slices, and 1 blocked surface. Conflicting source annotations, missing reviews, and stale fingerprints fail checks. |
| 5. Case-insensitive spec-sheet collisions | Output names include the unique port directory. Case-folded collisions are rejected before writing. The named-argument/default/let splitter now preserves complete statements; a file without a headline can still produce an index entry. |
| 6. Private axioms bypassed the source whitelist | The whitelist enumerates actual elaborated axiom declarations. Tests cover modifiers, multiline declarations, macro-generated axioms, and strings/comments that defeated textual scanning. |
| 7. Low contrast and hidden mobile theme control | Light/dark muted text colors were corrected, explanatory captions enlarged, and the theme selector retained on narrow screens. Real Chromium checks cover 375px layout and theme selection. |
| 8. Vulnerable website dependency set | Compatible direct/transitive dependencies and lockfile were updated. Website CI audits the locked set before building and checking links. Deployment remains static GitHub Pages. |

## Product and documentation suggestions

| Suggestion | Resolution |
|---|---|
| Put the trust boundary near the main claim | Homepage lead says “Lean-checked contracts for a typed Triton-style model” and links semantic scope. |
| Explain the real DSL workflow | Homepage acknowledges Lean wrapping/translation choices. The complete [VectorAdd tutorial](https://lizn-zn.github.io/VeriTile/cookbook/vector-add-walkthrough/) extracts code from the executable showcase and reproduces an accepted proof and rejected mutation. |
| Browse coverage by theorem | The [coverage table](https://lizn-zn.github.io/VeriTile/proofs/coverage/) provides search, scope/numeric filters, exact theorem statements and IO definitions, Python/Lean links, limits, and an exact-commit completed audit record. It works without JavaScript. |
| Preserve recorded-demo labels/freshness | The fixed addition/subtraction demo remains explicitly recorded; source/library/toolchain fingerprint drift fails the build. |
| Correct build claims and resource guidance | The README distinguishes the default library, full library, and standalone checks, with a smoke command and first-build estimates. |
| Linux comparator integration and library smoke tests | Comparator integration runs on Linux with the real tools. Artifact CI checks representative vector-addition, reduction, and quantization ports after library changes. All proof-compilation audit entry points require the official comparator. |

## Evidence boundaries

The GeLU Taylor-20 remainder is still a disclosed, whitelisted axiom. Samples
and evaluations at polynomial extrema are now explicitly described as
checkpoint evidence; they are not an interval certificate or a proof of the
residual bound. The trust and research notes agree.

Coverage review concerns the proposition actually stated, not an independent
proof of the Python-to-model translation. Lower-level `Realizes` claims can
be partial-correctness claims; exec-existentials add termination but can omit
a frame; IO contracts carry their explicit memory conditions. Mathematical
execution and abstract rounding are not concrete GPU/IEEE certification.

`site/src/lib/coverage-ci.json` records only a completed successful full corpus
audit at its exact commit. `scripts/record_coverage_ci.py` rejects pending,
failed, and skipped runs. A later source review or site build does not turn
that historical result into proof evidence for a newer commit.

## Reproduction

Run from the repository root with the pinned Lean toolchain and dependencies:

```bash
lake build VeriTile VeriTileFull
python3 -m unittest discover -s scripts -p 'test_coverage_review.py'
python3 bench/coverage_review.py
python3 bench/check_proof_gap_manifest.py
python3 -m unittest discover -s scripts -p 'test_doc_api.py'
python3 site/scripts/check-doc-api.py
python3 site/scripts/sync-docs.py --check
python3 site/scripts/record-home-demo.py --check
```

The following require the official comparator, exporter, and sandbox tools
on `PATH`, as described in `scripts/README.md`:

```bash
python3 -m unittest discover -s scripts -p 'test_audit_gates.py'
scripts/check-artifact.sh
bench/audit_tritonbench_g.sh
```

The full corpus audit is substantially slower than library and website
checks. Its run status must be reported separately from those faster checks.

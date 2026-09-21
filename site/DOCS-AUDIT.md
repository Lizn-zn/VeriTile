# Documentation audit — 2026-09-17

Scope: the public website, its repository design-note sources, and the
README / tooling references linked by the site. The Lean implementation was
used as the authority for interfaces and modeled guarantees.

## Corrected

- Replaced nonexistent `ComputeCorrect.Realizes` references with the appropriate
  exact or rounding-model API; corrected the `.toRealizes_without_Rounding`
  bridge and the explicit `R` argument to `RefinesAt`.
- Distinguished universally quantified `ComputeRefine.Realizes` from fixed-model
  `Refines R` and `io ⊨[R] f`; documented partial correctness versus the
  termination/output/frame guarantees in the KernelIO contracts.
- Documented `round_real` and `round_idem`, abstract cast/store rounding, and
  the remaining IEEE-754 boundary.
- Updated the implemented flat-memory bridge and atomic slices; corrected
  relocated files, reduction helper names, and standalone helper ownership.
- Marked the GELU strategy as research and linked the current axiom whitelist.
- Added semantic-caveat and trust-audit pages in both languages.
- Replaced the old migration script with repeatable source synchronization,
  correct localized links, and a check mode.

## Verification

- All generated design-note copies match their sources (26 pages).
- Production site build succeeds (51 pages).
- Built internal links, anchor targets, and linked repository files resolve.
- Public API names extracted from English design notes, cookbook pages, and
  README are resolved by the pinned Lean toolchain.
- `bench/examples/VectorAdd.lean` elaborates, including its axiom and statement
  audits. `bench/tests/TritonSmoke.lean` and `bench/tests/LoopInvariant.lean`
  elaborate too. These runs emit existing unused-simp-argument warnings.
- Site CI now checks source synchronization and links. The existing artifact
  gate also checks documented API names.

## Limits

This audit does not rerun the complete benchmark proof/trust suite or the
full artifact release gate. Schematic proof fragments with placeholders are
identified as templates rather than standalone compilable programs. API-name
resolution is not a semantic proof of the prose; changes to assumptions and
theorem meanings still need review. External issue resolution is left to the
linked upstream reports.

## Website language update — 2026-09-20

The public website now publishes English only. The Chinese site copies and
language navigation were removed, and source synchronization covers the 13
English design notes. The original audit counts above describe the earlier
bilingual build.

The updated production build contains 26 pages, including the error page.
All 1,369 internal and repository links resolve. Generated pages use English
and contain no Chinese text or language selector; Chinese routes and sitemap
entries are absent. Repository design-note sources remain the authority for
the English pages.

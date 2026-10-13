# kldiv_compute

- Source file: `kldiv_compute.py`
- Corpus: TritonBench-G v1
- Status: source, precision, real IO/frame, safety, full FP equivalence and execution witnesses pass. The final dependency-aware build passes 3538 jobs; public axiom checks use only standard Lean axioms, with no FP assumptions. Three contracts are registered as `test-gap`; independent replay remains pending. See [source_scope.md](source_scope.md) for modeling boundaries.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

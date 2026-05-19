---
title: Architecture & semantics
description: How VeriTile's pieces fit together — the layers, the supported Triton subset, and the semantic models that proofs rely on.
---

This section is the **specification side** of VeriTile: how the codebase is
laid out, which Triton features the DSL admits, and what semantic models the
proofs commit to.

## In this section

- [Code organization](/VeriTile/architecture/code-organization/) — three layers, what
  lives where, and the typical change shape when you add an operator, a
  bridge lemma, or a new kernel transcription.
- [Triton subset](/VeriTile/architecture/triton-subset/) — the Triton-like surface
  syntax that's actually embedded, with the language features that are
  modelled vs. deliberately out-of-scope.
- [Erase + dtype](/VeriTile/architecture/erase-dtype/) — how typed `Op` terms project
  through `toAlgorithm?` to a mathematical (`ℝ` / `ℤ`) channel for algorithmic
  proofs.
- [GPU memory model](/VeriTile/architecture/gpu-memory-model/) — what the kernel
  semantics assume about regions, pointers, and reads/writes.
- [Concurrency semantics](/VeriTile/architecture/concurrency-semantics/) — the
  serialized projection model, and what's deliberately *not* claimed.
- [Memory safety](/VeriTile/architecture/memory-safety/) — the safety side of
  `tl.load` / `tl.store`, masks, and bounds.

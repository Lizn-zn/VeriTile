---
title: Roadmap
description: Where VeriTile is headed next — substrate work, bench expansion, and open design questions.
---

The roadmap is structured around **substrate work that unblocks bench
expansion** rather than a kernel-by-kernel checklist. A single substrate
unlock (e.g. `tl.dot` algorithmic model) often closes 10+ bench entries
at once, so the leverage is in the substrate.

:::caution[Draft]
This page is a working draft. Priority ordering and timing reflect the
current best-guess against open issue clusters but are not committed
plans. Adjust against the
[live issue list](https://github.com/Lizn-zn/VeriTile/issues) before
making project-level decisions.
:::

## Top-of-funnel substrate gaps

The bench coverage census on the [status page](/VeriTile/status/) names eight
substrate gaps that, together, account for most unproven bench entries.
Roughly in expected-leverage order:

### 1. `tl.dot` algorithmic model

**Unblocks:** all 10 matmul kernels, 14 flash / attention families,
`attn_fwd_causal`, chunk kernels with `tl.dot`. Likely the single
highest-leverage substrate item in the project.

**Shape of the work:** Compute-layer model of typed `tl.dot` against
`Real` matrix multiplication, plus the loop-induction characterization
needed when `tl.dot` sits inside a `forLoop_inv` body. Tracked in #93,
referenced by #128 / #135 and others.

### 2. Streaming softmax invariant chain

**Unblocks:** attention forward kernels, `token_softmax_*`,
`token_attn_*`. Builds on #1 since the kernels combine `tl.dot` with
streaming softmax.

**Shape of the work:** promote the `(m, l, O)` accumulator to a reusable
invariant in `Semantics/StreamingAccumulator.lean` (issue #112).

### 3. forLoop-wrapped store proofs

**Unblocks:** `rope_embedding`, `fast_rope_embedding`,
`kv_cache_filling`, `kv_cache_copy`. Existing slice proofs cover one
iteration; full-loop proofs need `forLoop_inv` with per-iteration
invariants over interleaved stores.

**Shape of the work:** offset-set disjointness helpers analogous to the
existing cross-region strip, but for same-region interleaved stores.
Estimated 100–300 lines per kernel after the helper lands.

### 4. Multi-block layer norm / RMS norm

**Unblocks:** `rmsnorm_implementation` (#122), `layernorm_fwd_triton`
(#123), `layer_norm_ops` (#133). NOT substrate-blocked — the helpers
exist (`forRange_inv` / `forLoop_inv`); the work is cross-loop scalar
threading. Multi-week per kernel; precedent in
`VeriTile/Examples/LayerNormKernels.lean` and `OnlineSoftmax.lean`.

### 5. `tl.cumsum` direction-aware semantics

**Unblocks:** recurrent / cumsum kernels with backward / reverse
direction (issue #94). Plus the kernels that compose cumsum with
streaming.

### 6. Int rounding + packed int4 / int8

**Unblocks:** `int8_quantization` (issue #129), quantized KV per-block
scales (#137). Needs Compute-layer `llrint` / int8-cast build-out plus
the int4 packing semantics (issue #95).

### 7. Signed pointer arithmetic

**Unblocks:** `conv2d` padded (issue #130). Needs a typed `Int` pointer
model.

### 8. Constexpr branch case-splitting

**Unblocks:** `rotary_transform` (4-branch `Stmt.ifThenElse`),
`layer_norm_ops` full forward (4 Bool flags → 16 combinations). Tactic
work rather than substrate.

## Near-term bench expansion

Outside the substrate work, the **43 README-only scaffolds** in
`bench/tritonbench_g/` are kernels with no `.py` / `.lean` yet. Some are
parked behind substrate (attention families pending #93 / #135), others
are independently approachable. Open question: which scaffolds, if any,
should jump ahead of the substrate queue?

## DSL / surface evolution

Listed because they affect every bench port, not because they're
imminent:

- **Reduce / eliminate `$(REGION)` antiquote** for the paste-in goal —
  context-aware resolution where possible.
- **`tl.toReal(_)` removal** — Triton uses implicit type promotion;
  VeriTile's explicit form is paste-in-hostile.
- **Mixed `ℝ` / `Nat` arithmetic** — Triton accepts; VeriTile currently
  rejects. Tradeoff with channel separation.
- **Unknown kwarg policy** — currently hard-fails on `cache_modifier=`
  / `eviction_policy=`. May need to relax to "warn + ignore".

## Open design questions

Items where the cleaner answer is not yet committed:

1. **Refinement bench coverage** — `ComputeRefine.Realizes` has zero
   uses in `bench/` despite the surface existing. Are there
   "kernel ↔ kernel" pairs in the corpus that would benefit from
   refinement proofs, or is the surface only paying its way for
   `VeriTile/Examples/` work?
2. **Manifest as source of truth** — `KernelManifest.md` documents
   what's tracked; should bench status (slice vs. full vs. substrate-
   blocked) be encoded *in* the manifest with CI enforcement, instead
   of recomputed from `grep`?
3. **Verso architecture deck** — the slide deck in `verso/` is
   maintained separately. Should it live in this site as an `/overview/`
   page, or stay independent because it has its own toolchain
   (`v4.30.0-rc2`)?
4. **i18n parity** — every page on this site is duplicated EN/ZH today.
   Worth it long-term, or should some pages (changelog, raw API
   reference) be EN-only?

## How priorities are set

Status here is descriptive — what's open, what's blocked. Priorities are
set off-site (issue triage, project conversations) and these notes are
adjusted to follow. If you find this page lagging the actual direction,
the right fix is to update this page, not to change direction silently.

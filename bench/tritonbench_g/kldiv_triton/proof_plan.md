# KLDiv Triton Full-Source Proof Plan

> Execute sequentially with one capped Lean compiler. No subagents, commits,
> shared edits, Lake builds, or global registry edits are authorized here.

**Goal:** Prove independent real IO/safety/frame and whole-kernel FP equivalence
for the actual complete forward and backward kernels, preserving historical APIs.

**Architecture:** Three public Lean entries share faithful typed source. Local
proof modules separate independent formulas, source projection, signed loop
invariants, actual stores/frames, precision regressions and numerical execution.
The reduction contract describes one scalar per executed chunk, not a row sum.

**Tech Stack:** Lean 4, mathlib, VeriTile Triton DSL and real/structural interpreters.

## Approved source scope

Forward floating source profiles independently select six prediction and output
carriers, and fp32/fp64 target storage for direct log/exp. Both log flags and every
Nat reduction mode retain the complete source loop. Backward retains its two
pre-loop assignments, six target/output carriers in the default multiplication
branch (fp8 targets are an abstract extension, excluded from source acceptance
because nonzero weak scalar construction is unavailable), and fp32/fp64 target
with six outputs in the exp branch. Runtime offsets
use signed Int registers; natural row strides and positive B delimit the initial
mathematical address domain. Signed negative strides and native fixed-width
arithmetic are separate obligations, never silently reinterpreted as proved.

Zero-row forward reduction performs no store. For N>0 the nonzero reduction
branch stores ceil(N/B) scalars at row*loss_stride + chunk. The Python wrapper's
one-scalar-per-row allocation does not satisfy general multi-block bounds.

## Steps

- [x] Preserve a legacy snapshot and audit every declaration header/docstring (51, no mismatches).
- [x] Split historical source, helpers and specifications without changing them.
- [x] Add complete typed forward and backward sources plus exact address-commuting candidates.
- [x] Check exact load, mixed-arithmetic, reduction, signed-loop and store ASTs.
- [x] Prove independent pointwise, padded-block and backward mathematical formulas.
- [x] Prove actual initialization and body execution, then arbitrary-length signed-loop invariants.
- [x] Prove actual active bounds, trace safety, flat execution/readback and full frames for the canonical mathematical source. Typed projection and candidate execution/safety transport gates also pass.
- [ ] Prove full FP state/failure equivalence without numerical rewrite assumptions.
- [ ] Check nonempty multi-block execution witnesses, reduction chunk footprints and empty-row behavior.
- [ ] Gate every leaf, public entry and axiom audit sequentially; preserve all historical headers/docs.
- [ ] Remove the redundant historical top-level module only after all replacements pass.
- [ ] Obtain root read-only review and hand off dependency integration; report unsupported source profiles.

No proof may assume successful execution or desired output values. Real log is
totalized; real padding simplifications do not claim IEEE behavior. FP proofs
retain arithmetic on inactive lanes and arbitrary backend behavior.

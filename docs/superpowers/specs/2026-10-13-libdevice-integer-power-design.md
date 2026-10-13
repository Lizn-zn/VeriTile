# Preserve libdevice integer-power semantics

## Purpose and source evidence

The full `rmsnorm_triton` source calls `libdevice.pow(x.to(tl.float32), 2)`.
Pinned Triton 3.5.1 dispatches the fp32/int32 overload to `__nv_powif` and the
fp64/int32 overload to `__nv_powi`. The current DSL expands literal exponent two
to multiplication. That expansion can support an ideal-real square proof but
cannot identify the source floating operation in an arbitrary backend.

Primary source: https://github.com/triton-lang/triton/blob/v3.5.1/third_party/nvidia/language/cuda/libdevice.py

The existing all-corpus proof plan requires faithful source operations and
explicit precision. This change supplies the missing primitive; it does not
admit an equality between native integer power and multiplication.

## Alternatives and decision

1. Add a general integer-exponent libdevice operation. This preserves both
   operands and broadcasting, supports signed exponents, and follows the
   existing AST/backend/erasure architecture. Recommended.
2. Add a unary square-intrinsic operation for exponent two. This is smaller,
   but leaves other integer exponents unrepresented and adds a special-case
   backend interface that would need replacement later.
3. Add only a compute-layer intrinsic with no algorithm projection. This
   preserves execution identity but blocks the existing source-to-real proof
   workflow and requires a second projection design.

## Chosen representation

Add `Op.libdevicePowi` with a real carrier base, an Int exponent and explicit
broadcasting. Existing typed precision scopes record the actual fp32/fp64
compute profile. Keep the exponent integer; do not convert it to floating pow.

The structural backend gets an optional precision-indexed callback accepting a
base value and integer exponent. Evaluate the actual operands and call that
callback. An absent callback returns execution failure. There is no fallback
to multiplication or to floating-exponent pow. The precision adapter forwards
the resolved compute dtype.

The exact-real and abstract-rounding interpreters use integer power (`zpow`)
with the framework's existing handling of bottom values and result rounding.
Erasure retains the operation. Exact-real lemmas can prove exponent two equals
multiplication; this fact is confined to the mathematical channel.

The DSL preserves libdevice integer-exponent calls as this node, including the
literal-two case used by `rmsnorm_triton`. Floating-exponent calls retain their
separate operation. Historical source-independent mathematical models may keep
an explicitly documented square model, with exact-real bridges where needed;
historical theorem headers and docstrings remain unchanged.

The native overload's exponent type is int32. Literal two fits it directly;
other integer widths must not be silently relabeled as that overload. The Int
carrier models the exponent value, not fixed-width overflow. Dynamic exponent
bindings require a retained int32 source annotation or an explicit checked
conversion; unsupported source dispatch must remain a visible error.

## Integration and ownership

Root coordinates shared files after the current RMSNorm implementation build
finishes. A single owner edits Core/Ast, Semantics/EvalOp, Float/Structural,
Float/ExecutionProfile, Float/EvalOpR, Float/Erasure, memory safety/locality/
flattening/typing modules, and DSL expansion. Other agents keep working in
disjoint ports. No unrelated refactor, commit, push, GPU rerun or admission-table
change is part of this extension.

## Validation

- Exact AST checks distinguish integer power from multiplication and floating
  pow, retaining exponent, broadcast and local compute precision.
- A backend regression returns distinct results for the three operations and
  proves the integer-power callback is selected; absence must fail.
- Nonempty exact-real execution exercises integer power; prove the square
  bridge with ordinary Lean proofs and standard axioms only.
- Extend all affected semantics, typing, safety and flattening proofs rather
  than adding unproved assumptions or skipping new cases.
- Compile the shared library and affected DSL regressions, then both full
  `rmsnorm_triton` source variants and their independent real/FP contracts.
- Source fidelity and model distinctions receive an independent review.

Native code-generation behavior and IEEE error bounds remain separate from
backend-parametric equivalence. This change must not turn a mathematical square
identity into a floating rewrite admission.

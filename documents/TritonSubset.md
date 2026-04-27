# Supported Triton Subset

This document records the Triton subset currently embedded in Lean by VeriTile,
as well as the parts that are not supported yet.

## Currently Supported

- `tl.load($(REGION) + offs)` and `tl.store($(REGION) + offs, value)` with
  gather/scatter tile offsets — the surface syntax mirrors Triton's
  pointer-plus-offset form. Scalar-pointer sugar `tl.load($(REGION))` /
  `tl.store($(REGION), value)` desugars to offset `0`.
  Note: `$(REGION)` is a Lean `RegionName` term, not a CUDA pointer; see
  [RP1](../Notes/research_problem_pointer_vs_named_region.md).
- `tl.arange(N)` and `tl.arange(start, end)`
- `tl.broadcast`, `tl.full`
- scalar `ℝ` constants (numeric literals) and `Nat` address constants (`$(...)`
  antiquote); see [RP2](../Notes/research_problem_address_typing.md) for the
  `ℝ` / `Nat` channel split.
- `tl.exp`, `tl.log`, `tl.maximum`
- arithmetic: `+`, `-`, `*`, `/` (dispatched by carrier type — `ℝ × ℝ` and
  `Nat × Nat` only; mixed-type expressions are an `evalOp` error)
- reductions: `tl.max`, `tl.sum` (over `ℝ` tiles only)
- `tl.program_id`, `tl.constexpr`
- assignment and `tl.store` statements

## Known Limitations

- `forLoop` exists in the AST but is not operational yet.
- `tl.dot`, atomics, async copy, masking, and multi-block execution are future
  work.
- Floating-point arithmetic is modeled over `ℝ`; IEEE-754 fidelity is outside
  the current proof model.
- Memory regions (`$(REGION)`) are statically named, not first-class
  pointers — `if cond then xReg else yReg` style cannot be expressed
  ([RP1](../Notes/research_problem_pointer_vs_named_region.md)).

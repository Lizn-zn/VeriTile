# Supported Triton Subset

This document records the Triton subset currently embedded in Lean by VeriTile,
as well as the parts that are not supported yet.

## Currently Supported

- `tl.load`, `tl.store` with gather/scatter tile offsets
- `tl.arange`, `tl.broadcast`, `tl.full`
- scalar and tensor constants
- `tl.exp`, `tl.log`, `tl.maximum`
- arithmetic: `+`, `-`, `*`, `/`
- reductions: `tl.max`, `tl.sum`
- `tl.program_id`, `tl.constexpr`
- assignment and store statements

## Known Limitations

- `forLoop` exists in the AST but is not operational yet.
- `tl.dot`, atomics, async copy, masking, and multi-block execution are future
  work.
- Floating-point arithmetic is modeled over `Real`; IEEE-754 fidelity is outside
  the current proof model.

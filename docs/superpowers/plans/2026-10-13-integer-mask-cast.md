# Integer mask cast repair

The full TritonBench-G goal includes `masked_add_cuda`. Its Python tests pass
int64 and int32 mask buffers and call `.to(tl.int1)` after loading. The existing
bool-channel proof does not cover those buffers. Preserve that legacy statement;
the source-facing proof must retain the integer load and test nonzero truthiness.

Root owns the shared DSL repair under the existing completion plan. Reuse the
integer nonzero conversion already used by `tl.where`; no new execution axiom or
floating-point admission is needed. Changing Python to use bool buffers would
change the requested source scope. Rewriting the source cast as a comparison in
the port would require a separate translation exception, so preserving the cast
syntax and correcting its typed expansion is preferable.

- [x] Add a source regression for a declared integer load followed by int1 cast.
  Check that the expanded expression contains an integer load under an integer
  nonzero comparison, including offsetted pointers and parenthesized syntax.
- [x] Resolve storage from the pointer's declared region/element type before
  applying a bool load-cast assignment. Reuse `Op.ne` and scalar zero broadcasting.
  Keep legacy untyped bool-load expansion unchanged. Do not claim floating-point
  casts to bool are supported by this integer-specific repair.
- [x] Check the failing regression before repair, then compile the changed DSL
  module and its imports and rerun it. Check existing signed-load and integer
  `tl.where` regressions to detect interference with their storage inference.
- [ ] Add a typed source for the complete masked-add kernel and derive both real
  and FP contracts over actual integer mask memory. Preserve the historical bool
  contract and every original theorem signature.
- [ ] Register the new public source scope only after its execution, frame,
  precision and source checks pass. Independent comparator acceptance remains a
  separate required gate.

# Remaining conversion and width obligations

The Python wrapper forwards independently supplied source and destination
buffers without checking dtype equality. Triton stores implicitly convert to
the destination element type ([official store documentation](https://triton-lang.org/main/python-api/generated/triton.language.store.html)).
Homogeneous discrete copies need no payload arithmetic or conversion and are
handled separately from the 36 floating input/output profiles.

The current shared DSL cannot yet faithfully represent every mixed pair:

- `v = tl.load(src); tl.store(dst, v.to(tl.int1))` rejects the variable cast:
  the boolean cast expansion only accepts a directly nested `tl.load`.
- A floating `v.to(tl.int32)` or `v.to(tl.int64)` currently reaches the shared
  `.real, .int` branch and emits `Op.castRealToInt8`. This is not a faithful
  general int32/int64 destination conversion.
- A floating `v.to(tl.uint32)` reaches the floating-to-Nat branch, which emits
  the placeholder register `__compute_float_to_nat_cast_unprojected__`.
- Signed widths share the mathematical Int carrier, and unsigned widths share
  Nat. Narrowing conversions, signed/unsigned reinterpretation or wraparound,
  and fixed-width address arithmetic are not established by these carriers.

These cases are not included in the new homogeneous discrete family. They
remain source-supported obligations for the overall port, requiring coordinated
shared semantics work. No local source rewrite is used to hide the missing
conversions. The relevant expansion branches are in
`VeriTile/Triton/DSL/Expansion/Main.lean`, the `methodCast?` handler.

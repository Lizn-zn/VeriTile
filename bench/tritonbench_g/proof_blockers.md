# TritonBench-G Proof Blockers

No current TritonBench-G port exposes an explicit algorithm-layer `hAlg`
correctness blocker.

The mechanical audit still remains a translation-consistency gate, not a
substitute for future human line review against `review_criteria.md`.

## Translation-Surface Blockers

### Typed region parameter flow into the DSL surface

Several TritonBench-G ports still need explicit typed-region antiquotations in
the `triton { ... }` body, for example:

```lean
tl.load($((index_ptr : Region .nat)) + indices, ...)
```

The desired review-criteria surface is the Python-shaped form:

```lean
tl.load(index_ptr + indices, ...)
```

This currently fails because the `triton` macro recognizes region dtypes from
typed antiquotations, but a bare identifier bound by an enclosing Lean
parameter such as `(index_ptr : Region .nat)` is not available to the macro as
a typed pointer. Removing the ascription in `index_select_cat` produces a
`nat expression: dtype mismatch` expansion error.

This is a DSL typed-region inference gap, not a per-port proof gap. Closing it
requires teaching the DSL elaboration/expansion layer to preserve typed
`Region d` information for bare region parameters, or moving the relevant
surface expansion from macro-only syntax into an elaborator that can inspect
the local context.

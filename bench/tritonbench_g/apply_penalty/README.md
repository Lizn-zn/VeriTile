# apply_penalty

Source: [apply_penalty.py](apply_penalty.py), TritonBench-G v1.
Each program updates one logit row in place, gathering token ids and counts from
a cumsum window and scattering the penalized values back to those token ids.

- [Kernels.lean](Kernels.lean): complete original `apply_penalty` and reciprocal
  candidate. Only `cur_logits / cur_repetition` in the positive `tl.where` arm
  becomes `cur_logits * (1 / cur_repetition)`. The sign comparison, other arm,
  count promotion, metadata, masks, addresses and store stay unchanged.
- [Correct.lean](Correct.lean): both sources implement the independent formula
  `(if logit > 0 then logit / repetition else logit * repetition)
  - count * frequency - presence`, using the initially loaded values.
  The contract includes termination, scalar-slot and active-lane address bounds,
  and preservation of all other memory. Per-lane output values require distinct
  **active** token addresses; padded inactive ids may repeat.
- [FPEquiv.lean](FPEquiv.lean): complete fp32 equivalence using `div_mul_rcp`.
  Actual gathered logits must be finite and the loaded repetition scalar finite
  and nonzero at `rep_logits`, including evaluated masked lanes. Negative logits
  and negative penalties are allowed. Both `tl.where` arms are evaluated, with
  no assumption about which one is selected. Full state and failure are preserved.
- [Proofs/Real.lean](Proofs/Real.lean): real execution, readback, frame and address
  safety lemmas. FP composition stays in the short public entry.

The FP proof uses the generic ExpressionRewrite congruence lemmas through
`equiv_decompose; fp_prove`. They compose the admitted division identity inside
an expression, then preserve its assignment and the remaining source. These
structural lemmas are proved in Lean; they are not extra experimental atoms.
FP contextual equality retains scatter order even for repeated token ids;
independent per-lane real formulas need the active-index injectivity condition.

Both proof layers share Kernels.lean and do not import each other. Block size and
strides are symbolic. As in the source, `stride_logit_s` is unused because rows
are contiguous. Host metadata construction and launch composition are outside
these per-program contracts. No IEEE identity or speedup is claimed.

Current scopes and source fingerprints: [coverage_review.json](../coverage_review.json).

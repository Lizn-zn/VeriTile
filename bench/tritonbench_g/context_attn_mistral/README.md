# context_attn_mistral

- Source: `context_attn_mistral.py::_fwd_kernel`.
- `Kernels.lean`: complete original and scale-first candidate for fp16/bf16/fp32 storage. Only the operands of qk * sm_scale are swapped.
- `Correct.lean`: both sources equal an independent initial-array natural-exp weighted sum in real semantics. It includes finite causal/window sentinel weights, successful finite flat execution, address bounds and a full tagged-cell frame.
- `FPEquiv.lean`: complete program equivalence using only admitted `mul_commute`, checked on every actual fp32 qk/scale operand before the masks. Dot-plus-add order, natural exp, both guards, casts, accesses and failure are preserved.

The real formula preserves the finite -1e9 causal/window scores. Padded values
are zero but their scores still contribute to the denominator. The all-sentinel
center replacement and zero-denominator guard are part of the proved source.
The formula uses all key blocks reached by the source range, including its
rounded tail; it does not substitute ideal zero-weight masking.

Dimensions, strides, three program axes, grouped heads, signed metadata and
signed window widths remain symbolic. The real contract requires positive BN,
nonnegative source starts, valid active addresses and injective active output
stores. The start condition preserves signed K/V pointer advances before
conversion to natural offsets. All loads precede the sole store, so inputs
may alias output. Experimental shapes do not restrict the derivation.

`Proofs/` contains source decomposition, scalar real identities, execution,
loop invariants, memory bounds and finite-memory transport. A real witness uses
nonzero batch/head/block IDs, grouped heads, three key blocks and a tail. A
consistent abstract FP backend also witnesses a nonempty domain and full
source execution. These are model-level results; they do not establish GPU
execution, IEEE bitwise equality, fixed-width index-overflow safety or
concurrent launch safety.

Validation passed: both repository proof closures (3570 jobs), source/type
regressions, source-only corpus gates, 23 audit/layout tests and independent
comparator/trust replay of all 97 files across these two cases. Final target
inventories and source fingerprints match `Logs/comparator-check-b23tm5xw`.

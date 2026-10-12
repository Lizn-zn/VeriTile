# chunk_bwd_dqkg

**Checked with an explicit padding premise:** the seven block-pointer loads
retain their actual unspecified padding. Both complete real specifications
propagate the existing zero-oracle IO condition through the real source proof;
the complete FP equivalence preserves arbitrary unspecified values.

`Proofs/UndefinedTail.lean` proves that this real condition cannot simply be
removed. With T=K=BT=BK=BV=16, V=1, all-zero actual arrays and one-valued
unspecified V/H/DO/DH lanes, the full original source stores dq[0,0]=15 while
the independent whole-V formula is zero. `UndefinedTailLoop.lean` supplies the
actual loaded tiles and value-loop execution. This is a Lean real-semantics
counterexample, not a GPU measurement or a claim about concrete compiler values.

Source: `chunk_bwd_dqkg.py`, function `chunk_simple_gla_bwd_kernel_dqkg`.
Both real specifications and the complete FP equivalence were checked for
the scope below.

- `Kernels.lean`: complete original and dot-left candidate. The candidate only
  swaps the operands of the final fp32 dq addition. All casts, dots, reductions,
  masks, loop iterations and stores remain in context.
- `Correct.lean`: two real specifications with independent whole-V dq/dk/dg
  formulas, successful flat execution, active access bounds and full-cell frame.
  BV must be positive and row stride at least K; output regions are distinct.
  There is no V≤BV restriction, K≥BK restriction or divisibility requirement.
- `FPEquiv.lean`: complete typed program equivalence. Its only numerical
  assumption is `add_commute`, checked on the actual reached fp32 carry and
  typed dot result after score conversion.
- `Proofs/`: source projection, complete loop, real formulas, bounds, memory
  frame, flat IO and FP composition. `LegacyReal.lean` retains reused execution
  helpers and the earlier single-value-block headline.

q/k and v/dout each share a storage choice; h and dh choices are independent.
Each choice can be fp16, bf16 or fp32. Gates and dg use fp32. Python dtype
attributes in casts are resolved explicitly to DTYPE, OUT_DTYPE or tl.float32.
The actual typed AST is regression-checked, including history casts before dot,
score conversion and output conversion.

The source's H/DH parent shape is `(V, NT*K)`, with strides `(1, s_h_t)`.
The proof preserves that mask, including active accesses beyond a logical K
tail. `Proofs/HostSafety.lean` refutes the first bundled Python fixture's
history allocation safety: active offset 32768 lies outside 32768 elements.
No source mask or allocation is silently repaired. The real specification
requires sufficient active-address bounds and zero unspecified values for
loads whose Python source omits padding. Host launch composition, fixed-width
address overflow and concrete IEEE behavior are separate claims.

Validation includes complete successful original/candidate executions in a
consistent abstract FP model: two value iterations, mixed storage, nonzero
launch coordinates, partial time/key/value tiles, all three output values and
preserved masked lanes. `bench/tests/ChunkBwdDqkgBounds.lean` checks a K=3,
BK=4 tail: a masked address aliases the next row, but all actual writes remain
injective. These are Lean regressions, not GPU experiments.

Floating sums retain the input dtype unless explicitly promoted; an omitted
axis flattens the whole tile before one reduction. The shared interpreter uses
an opaque typed reduction backend. Only the real projection equates flat and
nested sums. Source references are the repository's pinned Triton 3.5.1
[sum implementation](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/standard.py)
and [reduction lowering](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/semantic.py).

The final targeted build passes 3,469 jobs, including real correctness, FP
equivalence, typed execution, the host-bounds refutation and the padding
counterexample. Both source/precision and active-address regressions pass.
Coverage review (380 declarations / 201 files) and the proof-gap manifest pass.
Independent comparator/trust replay accepts **24 files**, **150 named
proof targets** and **1,124 kernel theorem targets**; all **1,161 snapshot
input hashes** match the workspace. Receipt: `Logs/comparator-check-hrl4w00r`.

Earlier validation receipts are historical; the receipt above covers the repaired
padding semantics. This does not claim a whole-corpus build: other cases remain
pending, including the separate chunk-delta source-padding repair.

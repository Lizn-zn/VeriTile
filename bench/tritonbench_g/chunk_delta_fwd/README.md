# chunk_delta_fwd

Source: `chunk_delta_fwd.py`, `chunk_delta_rule_fwd_kernel_h`.

The three public entries are:

- `Kernels.lean`: complete original and sum-left candidate. Only the outer
  `b_h += b_h_cumsum` changes to `b_h = b_h_cumsum + b_h`. Both loops, optional
  initial/final states, pre-update history, corrected-value stores, all casts,
  boundary checks and TF32-disabled dots are retained.
- `Correct.lean`: independent real recurrence and both complete flat IO
  specifications, including successful execution, all three output buffers,
  active-address bounds and preservation of every other full tagged cell.
- `FPEquiv.lean`: complete conditional FP equivalence using only `add_commute`.
  The finite-operand checks occur after the history store and the whole inner
  loop at every reached outer carry update.

Supporting execution, safety, source bridges and arithmetic are in `Proofs/`.
The old `BC=BT` proof and precomputed-state store specifications remain in
`Proofs/LegacyReal.lean`; they are not the current public contracts.

## Real contract

The independent recurrence freezes H within each outer chunk, stores that
pre-update H, computes `C_c(t,j) = V(t,j) - sum_e D(t,e)*H_c(e,j)`, and adds
`sum_t K(t,e)*C_c(t,j)` to H. The correction sums over logical K, not padded BK.
The update window starts at `c*BT` and has width `ceil(BT/BC)*BC`, checked by
global T. Adjacent windows can overlap when BC does not divide BT. Corrected
row t is finally written by chunk `t/BT`; substituting a disjoint BT-row
recurrence is refuted by `bench/tests/ChunkDeltaRealExecution.lean`.

Dimensions, strides, storage choices and both state flags remain symbolic.
The real contract requires:

- `pid0=0`, `K≤BK`, positive BT/BC and `T≤NT*BT`;
- active read/write addresses within their extents, active output addresses
  distinct within each output, and disjoint physical allocation intervals;
- K/D/V separated from history and corrected V; history and corrected V
  distinct, and enabled final storage separated from those outputs;
- zero values for omitted, unspecified padding. Seed may alias outputs because
  initialization reads it before stores; final storage may alias inputs after
  the recurrence finishes.

This permits partial K/V/time tiles and overlapping inner windows. It is a
per-program real specification with mathematical integer addresses, not a
proof of host launch composition, concrete IEEE correctness or performance.

## Why the padding premise is necessary

All four checked block loads retain `.unspecified`; the source does not request
zero padding. `Proofs/UndefinedTail.lean` executes the complete original with
T=1, K=V=BT=BC=BK=BV=16, one outer chunk, no initial state and final storage
enabled. Actual memory is all zero; unspecified K/V tail lanes are one.
The final valid `H[0,0]` is 15, while the independent input-array formula is 0.
`refutation final_state_without_zero_padding` rejects removing that premise.
A separate T=16 array describes the loaded tiles only; the actual source keeps
T=1, including its stores. This is a checked real-model witness, not a GPU
measurement or a rejection of an FP atom.

## Validation

The targeted build passes 3,474 jobs. Three Lean regressions pass: source and
precision checks, multiple inner-block execution, and complete execution for
all four state-flag pairs. The latter also checks overlapping-row overwrite,
full-cell frame, active-address bounds, rejection of an undersized history
buffer and the complete candidate. An abstract typed execution exercises all
three tails and mixed fp16/bf16/fp32 storage; its dots are model operations,
not GPU measurements.

The 173-port source gates, scope review (380 declarations / 202 files),
proof-gap manifest and specification sheets pass. The source-checker regression
suite passes 18 tests, including preservation of dtype mismatches when a Lean
splice has an explicit quotation category. Independent comparator/trust replay accepts **31 files**, **179 named proof
targets** and **1,271 kernel theorem targets**. All **1,171 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-4rbqaiz0`.
The case is accepted within the contracts and limits above.

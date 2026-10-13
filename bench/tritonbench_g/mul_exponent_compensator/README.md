# mul_exponent_compensator

- Source file: `mul_exponent_compensator.py`
- Corpus: TritonBench-G v1
- Status: both real specifications and the FP equivalence compile with module
  axiom checks. Independent comparator replay remains pending on this host.

`Kernels.lean` contains the complete original source and a candidate that moves
the constant scalar to the left of multiplication. Loads, stores, index
arithmetic and the constexpr assignment are unchanged; no performance claim
is made.

`Correct.lean` proves the independent formula `xs i * 2^112` for both sources,
with successful flat execution and preservation of every other memory cell.
The existing positive block-size condition is retained. `Proofs/Real.lean`
contains execution, access safety and frame lemmas; it imports no FP proof.

`FPEquiv.lean` selects fp32 input/arithmetic and the wrapper's fp32 output, and
uses only `mul_commute`. The wrapper itself does not constrain input dtype. Its domain
requires the actual loaded operands and the interpreted scalar to be finite
after the shared prefix. It preserves the entire execution result, including
registers and memory, and agrees on failure. It is conditional on admitted
scalar rules, not a claim of unconditional bitwise IEEE equivalence.

The kernel has no tail mask: the full read/write windows must fit their
allocations. All sizes and program IDs remain symbolic. Host scheduling and
fixed-width integer overflow are outside these per-program contracts.

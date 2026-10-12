# chunk_gla_simple

Source: `chunk_gla_simple.py`, `chunk_simple_gla_fwd_kernel_o`.

- `Kernels.lean`: complete source and a candidate that puts `scale` on the left of the final multiplication. Q/K/H share a compatible fp32, fp16 or bf16 dot-input type; V and G are independently selected, and output storage is explicit. All three `allow_tf32=False` dots, the score-to-V cast, gate subtraction, intrinsic `tl.exp`, boundary checks and output cast are retained.
- `Correct.lean`: two real specifications for the complete source, including every K block. The independent formula is gated Q·H plus the causal Q·K·V contribution, followed by scaling. The conclusions include successful finite flat-memory execution, the active output values and preservation of every other tagged memory cell.
- `FPEquiv.lean`: complete typed program equivalence. The only FP atom is `mul_commute`, applied to the actual final fp32 total and scale after the common prefix. No exponential identity or real erasure is used in this proof.

`Proofs/` contains the source decomposition, numerical erasure, execution and memory arguments. The real and FP proofs use the same implementations. The three public specifications are `Correct.original_correct`, `Correct.scaleFirst_correct` and `FPEquiv.scale_equiv`.

The real contract requires noncolliding active output addresses, disjoint flat allocation intervals, and allocation bounds for the source-active Q/K/H loads in every K block, the gate/value loads and the output store. Inputs may alias the output: this kernel performs every load before its sole store. K, block sizes, strides and program IDs remain parameters; there is no `K = BK` restriction.

The original checked loads omit `padding_option`. Their unspecified values are retained in the source. The real flat contract explicitly chooses a zero unspecified-value oracle; it is not an unconditional result for arbitrary padding. The supplied history is an input array, not an assumed producer output. The Python test history lacks the launch's chunk dimension, so these per-JIT proofs do not certify the surrounding host allocation or launch. Backend shape restrictions, cross-program races, IEEE bitwise equivalence and performance are outside this contract.

A real regression checks two K blocks, partial time/key/value tiles and nonzero program IDs: both complete flat executions store `12` and preserve a neighboring integer-tagged `99`. `Proofs/TypedExecution.lean` separately checks a nonempty mixed-storage FP execution under a named abstract model satisfying the admitted rules; this is not a GPU measurement.

Accepted in `../verification_round.md`. The integrated build and concrete real regression pass, as do source translation and coverage checks. The current independent comparator/trust receipt is `Logs/comparator-check-usy6q186`: all 13 checked files, 107 named declarations and 1,234 kernel theorem targets pass, with 1,344 matching source/configuration fingerprints. The batch includes this case's eleven Lean modules, its real regression and the preceding case's comment-only helper update.

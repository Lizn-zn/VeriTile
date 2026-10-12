# attention_score

Source: `attention_score.py`, `_score_kernel`. The complete source and both proof
entries pass compilation, independent comparator replay and trust checks.

- `Kernels.lean`: complete original and scalar-left candidate. Q/K/Out have
  fp16 storage; M and accumulation use fp32. The candidate changes only
  `qk * qk_scale` to `qk_scale * qk`.
- `Correct.lean`: independent finite-input column formula for both versions,
  named strided Q/K/M/Out windows and masks. The contract proves successful
  flat execution, active-address safety, masked output values and preservation
  of every other memory cell.
- `FPEquiv.lean`: complete program equivalence using only `mul_commute`.
  The domain checks finite actual qk lanes and qk_scale at every reached fp32
  multiplication. Casts, dots, exp2, stores and execution failure are preserved.

M is a supplied normalization input, with no assumption about an omitted
producer. Each output sums base-2 weights over `ceil(ROUND_CTX / B)` query
blocks. The exact source scale constant, signed distance, regular/complement
windows, all four flags and grouped-head strides remain explicit. The
query-tail selection is controlled by IS_EVEN_N, as in Python, independently
of the IS_EVEN_M-controlled Q load.

The source wrapper uses a common query/key block size B. Empty loops and
partial final query blocks are covered. Checked Q/K loads require only active
addresses; unchecked loads require the complete displayed windows. M loads
are unmasked, so the final partial block still needs B normalizer cells.
K is loaded even if the query loop is empty. Output allocation and writes
follow the actual key mask. The unused output inner stride is retained.

The reviewed dtype transcription specializes the Python test configuration:
region declarations state Q/K/Out fp16 and M fp32, and `Out.type.element_ty`
is written as `tl.float16`. Explicit input/output region lists are retained.
`bench/tests/FPAttentionScoreSource.lean` checks these precision boundaries,
the signed distance and the query-tail condition. The lexical source gate
records this explicit exception; it does not verify Python alias resolution.

`Proofs/Real*.lean` connects source initialization, Q/K/M reads, arithmetic,
loop invariants, writeback, address safety and flat IO. Intermediate register
facts are proved from execution. They are not user input assumptions.
`Proofs/TypedExecution.lean` provides consistent nonempty abstract FP execution
and domain witnesses for all flag combinations, independently of the real proof.

The old natural-subtraction window is preserved in
`Proofs/LegacyNaturalWindow.lean`. It turns query 0/key 1 into distance 0 rather
than -1 and includes a future key in a positive regular window. Those old
statements concern the historical Lean program and are absent from current
public specifications. A source regression retains this disagreement.

Boundary regressions cover ROUND_CTX=3/B=2, the key-even-controlled query
mask, inactive output preservation, the fourth unmasked M read and the K read
before an empty loop. These are Lean model checks, not GPU experiments.
The results are per-program; host launch composition, concrete IEEE identities,
fixed-width integer overflow and performance are outside their scope.
Experimental admission shape does not restrict the symbolic proof dimensions.

Validation: the migrated benchmark library builds (3,677 jobs); all three
source/execution/IO regressions and 29 inventory/layout/shard tests pass.
Final replay accepts all 33 case/regression files, with all 198 named targets
present in the actual inventories. Evidence: `Logs/comparator-check-3r3pe16j`.

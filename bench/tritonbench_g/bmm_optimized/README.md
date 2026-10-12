# bmm_optimized

- Source: `bmm_optimized.py`, `bmm_kernel`.
- Status: complete real/FP contracts, regressions and independent replay accepted.
- Public entries: `Kernels.lean`, `Correct.lean`, `FPEquiv.lean`.
- Internal execution and bridge proofs: `Proofs/`.

The historical proof retained in `Proofs/Legacy.lean` covers only the fully masked branch, assumes zero
undefined values and proves active output readback. It does not provide the
full flat IO/safety/frame contract required by this review round.

The complete source in `Kernels.lean` retains all eight divisible-dimension mask
combinations, both CTA-ordering branches and the final short CTA group. The
fp32/fp16/bf16 variants keep input/output storage, fp32 dot accumulation and
`allow_tf32=false`. Triton's implicit store conversion to the output pointer
precision is explicit in the Lean source. Masked loads still omit `other`;
no zero fill has been inserted into the implementation.

The dot-first candidate commutes only each reached `o + dot(a,b)` addition.
Its proved typed-program rewrite uses only `add_commute`, with finite actual
carry/dot results checked after the loads and pointer updates. A consistent
abstract model supplies complete executions through the shortened CTA group
and two K blocks. This is a nonempty-domain witness, not GPU arithmetic.
Source and execution regressions check mask elision, omitted `other`, TF32
options, actual trip counts, store dtype, nonzero writes and masked cells.

The full real prefix, all mask/load branches, every K-loop iteration and the
terminal store now have execution proofs. Under explicit zero unspecified
load values and the appropriate K divisibility when its mask is elided,
`Proofs/RealExecution.lean` connects the output to an independent matrix
product over initial memory and preserves every other tagged memory cell.

`Proofs/UndefinedTail.lean` proves why the unspecified-value condition cannot
simply be removed: a supported 32×32 tile with K=33 and all-one input writes
808 rather than 33 when masked loads return five. The counterexample executes
the complete original/candidate source, including the final store. It concerns
the real interpreter's unspecified-load semantics, not a GPU measurement.

Whole-source memory safety is proved in `Proofs/RealSafety.lean`.
`Proofs/RealMemory.lean` pins both runtime grid dimensions to the host launch,
uses divisibility to justify omitted M/N masks, and proves active output
addresses distinct, including the final shortened CTA group.

`Correct.lean` connects both complete sources to independent initial-array
mathematical formulas through full flat IO, successful execution, active
allocation bounds and preservation of all other typed memory cells. The
contract retains the runtime-grid and zero-unspecified-value conditions.
`FPEquiv.lean` separately exposes the admitted FP equivalence.

`bench/tests/BmmOptimizedComplete.lean` checks a 65×48 output with K=33 and
32×32 tiles: the final short CTA group, M/N/K tails, both addition orders,
nonzero output and preservation of adjacent fp16-tagged memory cells, through
both region and flat execution. This is a Lean semantic regression.

The three benchmark/example libraries build successfully (4055 jobs).
All five focused regressions pass, as do the 28 layout/coverage/source tests
and all source gates. The reviewed inventory has 373 declarations across 197
public files. Independent comparator/trust replay accepts all 34 selected
files: all 28 case modules, the shared pointwise IO core and five regressions.
Every one of the 165 named declarations is in the replay inventory;
1275 kernel theorem targets and all 1055 snapshot input hashes are checked.
Receipt: `Logs/comparator-check-_3eonbb6`. These checks certify the displayed Lean
contracts and their dependencies, not concrete GPU execution.

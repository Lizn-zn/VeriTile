# `sin_computation`

`Kernels.lean` retains the legacy `sin_kernel` declaration and adds a full typed original/candidate family. `Correct.lean` retains the legacy headline and proves an independent lane formula `Real.sin (xs i)` for both complete typed variants, with successful flat execution, active-address bounds, and a frame for every other cell. `FPEquiv.lean` proves equality of the complete floating execution, including registers, memory and failure.

The supported family has 12 input/output profiles (fp32 or fp64 input × six independent supplied-output formats), each with original and candidate. The six represented output storage formats are fp32, fp64, fp16, bf16, f8e4nv and f8e5. There is no input cast and no `other` value on the masked load. The source sine preserves its fp32/fp64 precision. Explicit store casts expose the implicit conversion to a different output-pointer type; each precision boundary is checked by exact AST regression.

The candidate only changes `pid * BLOCK_SIZE` into `BLOCK_SIZE * pid`. Its FP theorem has no numerical atoms and no finite-input/domain assumption: every floating operation and its order stay identical under an arbitrary structural backend. `Proofs/Execution.lean` supplies nonempty successful exact-real numerical executions for every supported profile and both variants, with nonzero program id and a preserved masked neighbor. This witness backend is not an IEEE implementation or a target-GPU acceptance result.

The source restriction follows pinned Triton **v3.5.1**: [`math.py` lines 129–135](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/math.py#L129-L135) checks sine inputs against fp32/fp64 and returns the same tensor type; [`language/__init__.py` lines 119–120](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/__init__.py#L119-L120) reexports that function as `tl.sin`. Half, bfloat16 and fp8 inputs are rejected by this uncast source, not silently promoted or claimed covered. Integer/bool storage and other fp8 formats are outside this floating family; availability of fp8 output conversion on a particular GPU remains a target boundary.

The real flat-memory bridge uses its standard `undef = 0` launch invariant. This is not a source `other=0` argument: the typed load retains unspecified inactive values, and the FP theorem allows arbitrary undefined-value behavior. The common load/store mask prevents inactive lanes from becoming active outputs.

The real theorem quantifies over arbitrary tile size, element count, program id, active input values and valid flat allocations; it does not assume a full tile or precompute the input slice. Input/output region aliases are allowed by the region proof because the load precedes the store; distinct named regions use the flat allocation's disjoint placement. Natural address arithmetic abstracts fixed-width overflow. Host views/byte offsets, launches, cross-program races and target compilation are not proved. The wrapper launches `(n_elements,)` programs with `BLOCK_SIZE=4`; the per-program proof includes the additional fully masked programs.

The two sine directories have separate source declarations and public contracts; one does not stand in for coverage of the other. Independent comparator replay is pending.

The combined four-public-entry build for both sine ports passed all 3455 jobs.
All six public contracts are registered with current source/support hashes as
`test-gap`; the two FP assumption reports contain no numerical atoms.

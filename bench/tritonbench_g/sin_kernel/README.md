# `sin_kernel`

`Kernels.lean` retains the legacy `kernel_function` declaration and adds a full typed original/candidate family. `Correct.lean` retains the legacy headline and proves an independent lane formula `Real.sin (xs i)` for both complete typed variants, with successful flat execution, active-address bounds, and a frame for every other cell. `FPEquiv.lean` proves equality of the complete floating execution, including registers, memory and failure.

The supported family has two correlated input/output profiles (fp32 and fp64), each with original and candidate; the wrapper allocates output with `empty_like`. The wrapper-correlated output formats are fp32 and fp64. There is no input cast and no `other` value on the masked load. The source sine preserves its fp32/fp64 precision. The same-dtype store preserves the selected source precision, checked by exact AST regression.

The candidate only changes `pid * BLOCK_SIZE` into `BLOCK_SIZE * pid`. Its FP theorem has no numerical atoms and no finite-input/domain assumption: every floating operation and its order stay identical under an arbitrary structural backend. `Proofs/Execution.lean` supplies nonempty successful exact-real numerical executions for every supported profile and both variants, with nonzero program id and a preserved masked neighbor. This witness backend is not an IEEE implementation or a target-GPU acceptance result.

The source restriction follows pinned Triton **v3.5.1**: [`math.py` lines 129–135](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/math.py#L129-L135) checks sine inputs against fp32/fp64 and returns the same tensor type; [`language/__init__.py` lines 119–120](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/__init__.py#L119-L120) reexports that function as `tl.sin`. Half, bfloat16 and fp8 inputs are rejected by this uncast source, not silently promoted or claimed covered. Integer/bool and narrow floating inputs are outside this source-supported family. Direct low-level calls with a differently typed output are outside the wrapper-correlated contract.

The real flat-memory bridge uses its standard `undef = 0` launch invariant. This is not a source `other=0` argument: the typed load retains unspecified inactive values, and the FP theorem allows arbitrary undefined-value behavior. The common load/store mask prevents inactive lanes from becoming active outputs.

The real theorem quantifies over arbitrary tile size, element count, program id, active input values and valid flat allocations; it does not assume a full tile or precompute the input slice. Input/output region aliases are allowed by the region proof because the load precedes the store; distinct named regions use the flat allocation's disjoint placement. Natural address arithmetic abstracts fixed-width overflow. Host views/byte offsets, launches, cross-program races and target compilation are not proved. The wrapper launches `cdiv(n_elements, BLOCK_SIZE)` programs with `BLOCK_SIZE=1024`; its empty-input launch is a host boundary.

The two sine directories have separate source declarations and public contracts; one does not stand in for coverage of the other. Independent comparator replay is pending.

The combined four-public-entry build for both sine ports passed all 3455 jobs.
All six public contracts are registered with current source/support hashes as
`test-gap`; the two FP assumption reports contain no numerical atoms.

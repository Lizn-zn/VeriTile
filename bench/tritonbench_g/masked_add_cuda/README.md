# masked_add_cuda

`Kernels.lean` retains the complete source and commutes the final addition in
its candidate: `grad + p * alpha` becomes `p * alpha + grad`. Loads, stores,
integer-mask conversion, arithmetic precision and evaluation order inside the
product are preserved. All nine independent fp32/fp16/bf16 gradient/data storage pairs have signed
int32 or int64 mask storage. The earlier boolean-buffer specialization and its public
contracts remain separate; it does not stand in for Python integer masks.

`Correct.lean` proves the real formula `old_grad + p * alpha` independently for
both source versions. The signed mask is a genuine integer input channel:
exactly zero permits a write, and every nonzero value vetoes it. Contracts
include successful execution, active-address safety, in-place input/output
wiring and preservation of every memory cell outside the actual write set.
Dimensions, block size, program IDs and all mask values are symbolic, including
empty tiles and tails. The fp64-region transcription also has a real contract, with the FP translation
limitation below.

`FPEquiv.lean` compares complete typed executions with fp32 arithmetic under the existing
`add_commute` atom. The guard evaluates the actual loaded gradient and actual
product under the source's fp32 profile after the integer-mask prefix, requiring
both to be finite on every evaluated lane. It preserves registers, all memory
and failed executions. A separate theorem covers the legacy boolean profile.
`Proofs/Source.lean` checks every storage combination for signed integer mask
loads compared with exactly zero, explicit fp32 arithmetic, and the exact final
store carrier and conversion. There is no boolean-load substitution.

The wrapper asserts matching strides but passes no strides to the kernel. These
proofs cover the kernel's actual contiguous offset window `pid * B + lane`, not
a logical strided-tensor interpretation. Launch scheduling and fixed-width
address overflow are outside the per-program contract.

Runtime parameter typing is explicit: Python floating `alpha` arrives as an
fp32 runtime scalar in the repository's pinned Triton 3.5.1, unlike a weak
constexpr literal. The source annotates that fp32 scalar before arithmetic,
which promotes fp16/bf16 data to fp32. See the versioned
[Triton runtime specialization](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/runtime/jit.py#L336)
and [scalar promotion semantics](https://triton-lang.org/main/python-api/triton-semantics.html).

Translation-surface exception: the DSL does not retain the gradient pointer's
storage dtype at the final store after the register has promoted to fp32. The
source spells out the implicit fp16/bf16 output conversion once; fp32 stores
remain uncast. Exact store AST regressions cover every original/candidate pair.

Profile gaps remain explicit. Float64-region loads lose their compute profile
in the current DSL. Parenthesized explicit fp64 load casts retain that profile
but then reject the mixed fp64 operand / fp32 runtime-alpha expression. The
fp64-region transcription is therefore verified only mathematically, not claimed
as a faithful FP source. Even after translation repair, the admitted arithmetic
atoms are fp32-specific. Mixed fp64 storage, other mask storage, and arbitrary
integer-valued runtime alpha profiles remain outside the new FP contracts;
the test's integer zero scalar is exactly representable by the fp32 zero case.
Independent comparator replay remains pending on this host.

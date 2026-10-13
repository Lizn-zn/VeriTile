# swiglu_triton

- Source file: `swiglu_triton.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The complete source definitions are in `Kernels.lean`, legacy
execution/safety/frame helpers in `Proofs/Real.lean`, and preserved public
contracts in `Correct.lean`. `FPEquiv.lean` contains both full-state equivalence
contracts. All nineteen legacy declaration headers and docstrings remain
unchanged; the old compatibility module has been removed.

The typed source family covers six floating storage formats (fp32, fp16, bf16,
fp64, fp8e4nv, fp8e5), with 36 independent A/B forward combinations and
216 independent A/B/DC backward combinations. C follows A storage; backward
writes to the original A and B buffers. The candidate commutes only the three
signed Int row-address products. Input casts, masks, source sigmoid, raw B/DC
storage, multiplication order and output conversion remain explicit. In
particular, backward computes `dc * (silu(a)*(1-sigmoid(a))+sigmoid(a)) * b`;
it does not multiply dc and b before the fp32 expression.

The pure mathematical layer defines independent forward and backward scalar
formulas with sigmoid written as 1/(1+exp(-a)). It bridges to both complete
real IO contracts without assuming outputs or successful execution. Typed
whole-row contracts require positive BLOCK_SIZE and n_cols≤BLOCK_SIZE;
backward requires distinct A/B output regions because its second store would
otherwise overwrite the first. Legacy active-tile contracts remain separate.

All source pointers use the same symbolic stride, exactly as Python does.
The wrapper's views, stride compatibility, next-power-of-two block choice,
launch/allocation and cross-program composition remain host boundaries.
Signed arithmetic uses mathematical Int; negative strides and fixed-width
integer overflow are not modeled. Integer/boolean and other storage formats,
arbitrary byte-level alias reinterpretation and target-specific fp8 support
remain outside the current typed profile.

All helper and public file compiler gates pass. Independent source review found
no blocking defect. Exact AST regressions check all storage loads/stores and
the distinct mixed-fp64 scopes in the backward products. Numerical witnesses
use an exact-real backend with explicit casts, check both variants and preserve
masked sentinel cells; they do not certify hardware rounding. The combined
four-entry SwiGLU Lake gate passed (3459 jobs); independent comparator replay
remains pending.

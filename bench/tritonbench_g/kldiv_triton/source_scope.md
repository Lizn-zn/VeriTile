# Source obligations for full-loop verification

The forward source casts program id to int64 and applies three independent
row strides before its signed column loop. Reduction zero stores each block
elementwise under the active mask. Every other reduction mode instead reduces
the current block, stores one scalar without a mask, and increments the loss
pointer by one. It does not accumulate a whole-row sum inside the kernel.

Consequently, a row with multiple blocks writes multiple scalar cells. The
wrapper allocates only one scalar per row for these reduction modes, so the
source can overlap another row's output or exceed the allocation. A faithful
per-program contract must describe the actual chunk-output footprint and
required bounds; it cannot assume the wrapper allocation is sufficient or
replace these stores with one final whole-row sum. For example, three columns
with block size two produce two scalar stores in the modeled direct call.

Both input loads explicitly use zero on inactive lanes. The default formula
is `target * (log(target) - prediction)` without an epsilon clamp; the alternate
formula is `exp(target) * (target - prediction)`. Reductions include inactive
lane arithmetic. The exact-real totalized logarithm does not justify an IEEE
claim that a zero target times log zero vanishes.

Backward retains its pre-loop offsets/mask assignments and the complete signed
loop, loading target and writing the buffer named `input_ptr`. It implements
`target * -1` or `-exp(target)` according to the flag. Host scaling and reduction
postprocessing remain separate.

Required verification includes all source branches and blocks, exact precision,
actual trace safety and frames for the actual footprints, independently stated
real formulas, full backend original/candidate equivalence and nonempty
multi-block witnesses. Historical declarations/docstrings remain untouched.
This source review does not certify the port or its host launch.

Pinned [Triton v3.5.1 math frontend](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/python/triton/language/math.py)
accepts only fp32/fp64 tensors at `exp` and `log`. Both forward branches call
these operations directly on the loaded target, so source-facing floating
profiles require fp32/fp64 target storage. The logarithmic-target backward
branch has the same restriction. Its default multiplication branch must be
checked separately. These are source-inspection constraints, not observations
of native GPU compilation.

### Nonzero weak literals and fp8 backward input

The default backward expression is `target * -1`. The pinned frontend selects
`target`'s dtype for a weak integer scalar and creates `-1` directly in that dtype.
The baseline DSL expanded this literal as `0 - 1`, incorrectly widening half
input and changing a same-storage output carrier; a minimal exact-AST regression
reproduced the issue. A contextual literal repair now passes the shared
regressions. This port's full-profile precision checks also pass. Runtime
`-tl.exp(target)` retains its actual subtraction operation.

For fp8 target storage this source expression has a further frontend limitation:
nonzero `scalar_constant` requires a `get_<dtype>` builder method, but the pinned
builder binds nonzero floating constructors only for bf16/fp16/fp32/fp64. Its fp8
bindings construct types only. `Proofs/SourceDomain.lean` records the four
source-supported default-backward target types separately from the six-type
abstract AST extension. This is compiler-source evidence, not a native GPU test.
Forward fp8 prediction storage is converted by the fp32/fp64 target arithmetic;
fp8 output storage likewise remains a distinct conversion path.

Primary references: [semantic.py v3.5.1](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L540-L549)
and [ir.cc v3.5.1](https://github.com/triton-lang/triton/blob/v3.5.1/python/src/ir.cc#L848-L930).

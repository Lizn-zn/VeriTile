# bgmv_shrink_kernel

Complete for the real and admitted-FP contracts below. Compilation and
independent comparator/trust replay pass for the current sources.

- `Kernels.lean`: the complete original `_bgmv_shrink_kernel` and a candidate
  changing only `accumulator * scaling` to `scaling * accumulator`.
- `Correct.lean`: both precision-erased real IO contracts and an independent
  rank-slice GEMV formula over initial inputs.
- `FPEquiv.lean`: whole-program FP equivalence, with actual operand guards and
  the assumption printer. Its only admitted atom is `mul_commute`.
- `Proofs/`: source connections, execution, loop, bounds, flat-memory and IO
  proofs, plus nonempty execution witnesses. `RealFoundation.lean` retains
  earlier active surfaces only for reusable mathematical lemmas.

The source is `bgmv_shrink_kernel.py`'s `_bgmv_shrink_kernel` (line 8). Input
and weights share fp16 or bf16 storage; output storage is explicit. The source
keeps signed metadata, the -1 skip, every masked K-loop iteration, fp32
scaling and the actual `SPLIT_K == 1` store/atomic choice. Products retain
input/weight precision before fp32 reduction and accumulation. Both output
operands explicitly retain the conversion that Triton inserts for the output
pointer. `xk_stride` remains an unused parameter: X is addressed contiguously
as `batch*xm_stride + current_k`. `tl.max_contiguous` is a value-preserving
layout hint. The early return becomes a non-sentinel active guard; macro
specialization selects storage dtype.

## Real contract

For split ID `s`, the rank indices are
`k(c,e) = c*BLOCK_K*SPLIT_K + e + s*BLOCK_K`, with
`c < ceil(K/(BLOCK_K*SPLIT_K))` and `e < BLOCK_K`. Each active output row gets
`scaling * sum(k<K, X[k]*W[n,k])`. A single split stores this contribution;
multiple splits atomically add it to the cell's initial output value.

The contract quantifies over symbolic dimensions, strides and inputs. It
requires positive BLOCK_K*SPLIT_K, positive output stride and N≤BLOCK_N; K
need not divide the loop step. Signed metadata must be -1 or nonnegative.
Four disjoint flat allocations hold X, W, output and metadata; output appears
once even when read by atomic addition. The input pins and bounds cover only
rank indices visited by this split, all N weight/output rows, and metadata.
Input loads remain active independently of N. The -1 branch needs only its
metadata cell and skips every data window.

Both contracts include successful flat execution, the independent formula,
and preservation of all typed cells outside actual output writes. Initial
output is pinned only for the atomic branch. No output equation or prior
successful execution is a premise. `RealIOExecution` instantiates the entire
contract from empty registers and four concrete allocations: K=7/BK=2,
N=3/BN=4, batch/slot 1 and nonunit output stride. Four store iterations produce
[420,1176,1932]; two atomic iterations with split ID 1 produce [335,715,1095].
Only flat[141], flat[143], flat[145] may change. Masked input/weight padding
may exceed allocations without being read; masked output and all other cells
stay unchanged. The signed sentinel preserves all flat memory.

## FP contract

The domain checks finite actual fp32 accumulator and scaling operands after
the complete reached non-sentinel prefix and K loop. No reduction or atomic
operation is rewritten. The interpreter's atomic backend is separate from
ordinary floating addition and selected by storage dtype; each lane reads
memory updated by earlier lanes. Ordinary addition assumptions impose no
atomic law. Masks, output conversion, state and execution failure are retained.

`TypedExecution` demonstrates both complete sources and both tails in a
consistent abstract model with bf16 input/weights and fp16 output. For its
K=7/BK=2, N=3/BN=4 fixture, store produces [12,12,12]; atomic addition changes
initial output 2 to [8,8,8]. The fourth cell stays 2. The model interprets each
opaque reduction as one, so this is an FP execution/domain witness, separate
from the real GEMV values above. `FPBgmvShrinkSource.lean` checks the source's
precision, cast, mask, split-stride and output-branch structure.

These are per-program statements. Cross-program atomic interleavings,
launch-wide summation, fixed-width address overflow and concrete IEEE/error
claims are outside their scope. Experimental configurations select the
admitted scaling atom; they do not restrict the symbolic proof dimensions.
No GPU rerun is needed for this already admitted relation.

## Validation

On 2026-10-09, `lake build TritonBenchGVerified TritonBenchSpecExamples`
passes (3,916 jobs), as do the 28 layout/source-audit/coverage tests and the
typed-source regression. Source gates, coverage fingerprints and the
proof-gap manifest agree. Independent comparator/trust replay accepts all
20 Lean files in this directory plus `bench/tests/FPBgmvShrinkSource.lean`:
145 named targets, 661 kernel theorem targets and 889 matching snapshot
input hashes. Evidence: `Logs/comparator-check-se2cakdp` at the repository
root. These checks certify the displayed proof scope, not a GPU launch.

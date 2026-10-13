# log_softmax

- Source file: `log_softmax.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The full source/real/FP migration has passed local integration. All 38 historical declaration
headers and docstrings are preserved across `Kernels.lean`, `Proofs/Real.lean`
and `Correct.lean`. The historical compatibility entry was removed after the
public entries passed their file gates and the 38-declaration preservation audit.
The directory now has the strict three-entry layout.

The new typed sources retain both complete JIT bodies, both program axes,
two-dimensional masks, explicit fp32 input casts, reduction and operation order,
and destination conversions. Forward has 36 independent floating input/output
profiles; backward has 216 independent saved-output/upstream/destination profiles.
The wrapper's correlated backward destination is included. Exact load, formula
and destination-cast AST checks pass for every profile and both variants.
Real erasure, complete successful flat IO, arbitrary-padding region execution
and full raw-memory frames pass their file gates. Both public FP equivalences
pass and print no numerical assumptions; their complete state and failure
relations allow arbitrary aliases and undefined values. Independent-formula
public assembly now passes both LSP and file compilation: two flat IO contracts
and two arbitrary-padding raw-region contracts. All eight public declarations
use only the standard Lean axioms. Concrete execution witnesses and the final two-entry build pass (3454 jobs).
All eight public contracts are registered with current source/support hashes
as `test-gap`; independent comparator replay remains pending.

`Proofs/Address.lean` proves injectivity on active lanes with positive K, allowing
BLOCK_N to exceed N. The historical whole-tile injectivity hypotheses are retained
unchanged. Forward must retain `log(exp(x-max)/sum)`, including its source order.
Backward loads omit `other=`, so the reduction includes arbitrary unspecified
padding in the region model; zero padding is an explicit flat-model premise,
not a property of the Python source. Independent comparator replay is pending.

`Proofs/Execution.lean` runs both actual source variants at M=3, N=2, K=2,
BLOCK_M=2, BLOCK_N=4 with program ids (1,1). Forward mixed fp64/bf16 storage
yields log(1/2) at addresses 9 and 11. Backward fp64/fp16/bf16 storage yields
(-4-2u, -2-2u) for arbitrary real gradient padding u, including u=0 and u=5.
Masked addresses 13 and 15 remain seven despite overlapping masked tile
addresses. This numerical backend uses actual real exp/log and reductions,
with identity casts; it is not an IEEE or GPU-compiler validation.

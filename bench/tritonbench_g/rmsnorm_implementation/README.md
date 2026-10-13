# rmsnorm_implementation

The complete dependency-aware real/FP public-entry build passes (3447 jobs). All four public contracts and the separate trace-safety theorem use only standard axioms; the FP assumption list is empty. Source/support fingerprints are validated and the contracts are registered as `test-gap`; independent comparator replay remains pending.

The typed source retains both `range(0, N_SIZE, BLOCK_N_SIZE)` loops, both program IDs, all seven strides, every load and the masked in-loop store. Its 216 profiles independently choose input, weight and output storage from fp32, fp64, fp16, bf16 and the two supported fp8 carriers. The Python wrapper uses the subset whose output storage equals input storage.

Pass one converts input to fp32 before accumulating squares. The sum, division by N and square root stay fp32. Pass two reloads input without that cast, divides by the fp32 standard deviation, then multiplies by the uncast weight. Those operations can promote to fp64 independently; precision checks inspect the local scopes and the final output conversion. The weight load retains its missing `other` argument. Input loads retain their explicit zero padding.

The candidate commutes the natural multiplication forming the first term of `offset_m`. It retains both complete loops. The FP contract compares complete executions for an arbitrary backend, including failures, undefined masked values, aliases, registers and all memory. It requires no numerical rewrite atom. Numerical examples use an exact-real backend with actual square root and addition, and identity casts; they are not IEEE rounding results.

The independent real formula is

`x[t,j] / sqrt(sum_{u,e : u*B+e<N} x[u,e]^2 / N + eps) * w[t,j]`.

The exact-real contract covers successful flat execution, active bounds, every active output and the full frame. `Proofs/Safety.lean` separately proves actual execution-trace safety for both complete erased-real programs by transporting local access safety and execution through the variance loop and the address-prefix rewrite. It assumes positive block size, positive output column stride and output distinct from both inputs, allowing arbitrary logical row lengths across multiple tiles. The real projection uses zero undefined padding as its stated memory-model convention; the source weight load has not acquired a zero-padding argument. Positive epsilon implies a positive denominator; the general real theorem retains Lean's total division convention.

The pointer geometry currently uses nonnegative natural strides and addresses, as in the retained legacy proof. Negative signed strides are an explicit untranslated boundary, not covered by this contract. Cross-program scheduling and races remain outside this per-program result. The old mathematical `RoundingModel` contracts are preserved without interpreting their erased-real cast identity as an IEEE guarantee.

`Proofs/Precision.lean` checks both source loops and load/arithmetic/store profiles. `Proofs/Execution.lean` supplies two-tile mixed-storage executions for both candidates and both fp64-promotion locations (input division or later weight multiplication), with both program IDs nonzero, nonunit strides and a masked final lane. All 216 storage combinations pass exact AST checks for both variants. Four full numerical executions pass, and all 110 legacy declaration headers and adjacent docstrings are retained exactly. Independent source/formula review found no blocking defect; final repository integration and native source replay remain separate gates.

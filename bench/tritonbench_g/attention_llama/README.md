# attention_llama

Streaming natural-exp attention from `attention_llama.py`, `_fwd_kernel`.
Both K/V storage formats and causal modes share three public entries:

- `Kernels.lean`: complete fp16 and int8/E5M2 originals and carry-first
  candidates. The sole rewrite is `sum(p,1)+l_prev` → `l_prev+sum(p,1)`.
- `Correct.lean`: the real input/output windows, masks, launch conditions,
  independent weighted-sum formula and two specifications. `candidate` selects
  either implementation; `causal` selects either source branch.
- `FPEquiv.lean`: complete conditional equivalence for each storage format,
  with visible per-iteration domains. Both print only `add_commute`.

The real formula is Σ exp(scale·Q·K)·V / Σ exp(scale·Q·K). In causal mode,
key k is visible when `k + start_position ≤ query_row`, exactly as in the
source. H is query length, N_CTX is key length, and N_HEAD is head count.

The real contracts prove successful execution in a disjoint flat allocation,
active output values and preservation of every other cell. All buffer strides
remain symbolic; output row/column offsets must be injective. BM, BN and the
number of streamed blocks T are positive. Noncausal mode requires N_CTX=BN*T;
causal mode requires `(pid0+1)*BN+start_position=BN*T ≤ N_CTX` and
`start_position ≤ pid0*BM`. These are per-program launch conditions, separate
from access masks and allocation bounds.

For FP8, K/V stay original signed integer cells. Finite E5M2 bytes are decoded
inside execution and in the independent mathematical formula; no prepared
real buffer is supplied. Q and Out are distinct from K/V. Exact raw-byte frame
preservation includes negative zero, whose encoding a real value would lose.

The FP theorem preserves fp16 loads/casts/store, fp32 accumulation, tl.exp,
dots, reductions, source masks, causal branches and failures. FP8 additionally
preserves both int8 loads, E5M2 bitcasts and fp16 conversions. Its domain checks
finite actual sum(p) and rescaled l_prev after the score/exp prefix of every
reached iteration. It assumes no exp, dot or reduction identity and requires
no new GPU experiment. Abstract nonempty execution/domain witnesses cover both
storage formats and both causal flags; they are not GPU measurements.

Source details and limits:

- Positional zero padding is spelled `other=`, Q.dtype.element_ty resolves to
  fp16, and implicit output conversion is explicit. Storage constexprs become
  two complete sources. These documented transcription differences account
  for the textual scan exception; source regression tests inspect the AST.
- The source score mask uses local offs_n, not block_n_offs. The real formula
  therefore requires whole streamed blocks. Address safety covers partial
  tails, but this is not a conventional-attention proof for those tails.
- Empty visible causal rows are outside the real formula contract. The source
  predicate shrinks visibility with start_position; it does not add a cache
  prefix. The Python host grid, race analysis and GPU performance are separate.

`Proofs/RealIO.lean` and `FP8RealIO.lean` assemble the independent input-array
contracts from source execution, formula, safety and flat-memory proofs.
`Proofs/FP.lean` and `FP8.lean` use equiv_decompose/fp_prove for the numerical
rewrite. The real and FP proof stacks do not depend on each other. The former
single-file port has been superseded; its mathematical derivations are retained
and extended in `Proofs/RealKernel.lean`.

Validation passed: TritonBenchGVerified (3,646 jobs), five focused Lean fixtures,
source gates, coverage fingerprints/manifest and 41 distinct inventory/source
checks. Independent comparator and trust checks accepted 39 files, 255 named theorem targets and 1394 kernel theorem targets,
with all 745 snapshot hashes matching. Logs: `Logs/comparator-check-pnw4wkeh`.
This completes the directory for its displayed per-program contracts;
progress and retained scope are recorded in `../verification_round.md`.

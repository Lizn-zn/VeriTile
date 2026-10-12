# chunk_gla_fwd

Five complete JIT sources from `chunk_gla_fwd.py`, with real correctness and
conditional floating-point equivalence for each source/candidate pair.
**Accepted after complete source, build and independent proof checks.**

- [Kernels.lean](Kernels.lean): all five source kernels, including literal early
  returns, and candidates changing one arithmetic operation each.
- [Correct.lean](Correct.lean): ten real specifications, with input arrays,
  output addresses, masks, bounds, independent formulas and short proofs.
- [FPEquiv.lean](FPEquiv.lean): five full-program equivalence specifications,
  the actual reached-operand domains and the atoms used.
- `Proofs/`: execution, loop, frame, safety, projection and FP derivations;
  executable domain witnesses and the padding counterexample.

The obsolete output-only `ChunkGlaFwd.lean` has been replaced. Neither proof
entry imports the other. Dimensions, strides, scale and launch IDs remain
parameters; experimental shape does not restrict the derivations.

| Kernel | Candidate change | FP atom |
|---|---|---|
| Output consumer (`TypedOutput`) | Final carry/dot addition commuted | `add_commute` |
| Inter-subblock (`TypedInter`) | Carry/dot addition commuted in every K iteration | `add_commute` |
| Intra-subblock (`TypedIntra`) | Scale multiplication commuted in causal `where` | `mul_commute` |
| Split intra (`TypedSplit`) | Scale multiplication commuted after the source zero/add reduction | `mul_commute` |
| Merge (`TypedMerge`) | Carry/load addition commuted in every NK iteration | `add_commute` |

## Real contracts

Every `io ⊨ formula` concludes successful execution of the **whole erased
source**, the formula on initial memory, and preservation of every tagged cell
outside the masked output addresses. The finite-memory adapter requires
disjoint allocations, bounds on actual active accesses, and the explicit
unspecified-value oracle `s.undef = fun _ _ => 0`. The original block loads
still omit `padding_option`; the transcription does not insert zero padding.

The output formula combines all K-block gated history contractions with the
causal product of input A and V:

```
O[i,p] = Σ_{kb<cdiv(K,BK), e<BK}
           ((Q[i,k] * scale) * exp(G[i,k])) * H[k,p]
         + Σ_j tril(A)[i,j] * V[j,p],  k = kb*BK+e
```

The actual source strides and checked-parent zero extension are displayed in
the specification. Output addresses must be injective. A and H are inputs;
the formula does not assume another kernel produced them correctly.

The inter-subblock formula sums
`Q[i,k] * K[j,k] * exp(G[i,k]-G[j,k]) * scale` across the full K loop. The proof
cancels the separate flat gate reference using real exponential identities.
Both skip branches, partial rows/features and K=0 remain covered. All reads
precede the final store, so input/output aliasing is allowed.

Intra-subblock and split formulas are causal Q/K contractions. Cached matrix
rows use `timeStride`; streamed key/gate rows use `K`, exactly as in the source.
The unsplit case requires `K <= BK` and `columns <= BT`. The split case covers
one K tile and keeps the runtime head count and `T * BC` temporary-slice stride;
it does not require `K <= BK`. Both require key/gate regions distinct from
output because those inputs are read after earlier column stores. Cached Q
may alias output. Their region-level proofs permit arbitrary matrix padding:
invalid features multiply the source's explicitly zero-masked key. Only real
arithmetic uses that cancellation; no corresponding FP rule is assumed.

The merge formula sums every NK slice of the initial temporary input. Input
and output may alias because all reads precede the store. Active output
addresses must not collide. An early return preserves memory, and NK=0 writes
zero to active outputs. Its region-level formula also permits arbitrary
padding; the common flat adapter retains its zero-oracle premise.

## FP domains and source fidelity

Query/key/value/gate storage independently select fp32, fp16 or bf16 as exposed
by each source. History/output casts stay explicit; the temporary split buffer
and the consumer's A input are fp32, matching the host allocation. The consumer
keeps both query casts, the history conversion, default-option history dot and
`allow_tf32=False` final dot. Gate subtraction precision, intrinsic `tl.exp`,
reduction order, pointer advances, masks, padding and stores are preserved.

Domains evaluate the actual operands after the original prefix, inside every
reached branch and loop iteration. Both `where` arms are evaluated, so all
lanes are checked. No final-output equation is a premise. The split rewrite
retains the source `0 + sum`; it does not use an unadmitted zero identity.
The proof preserves complete state and execution failure under the selected
atoms; it is not an IEEE bitwise identity or an end-to-end error bound.

`Typed*Execution.lean` supplies complete, nonempty executions in explicitly
consistent abstract models, including mixed storage, multiple iterations,
partial tails, early returns and empty loops. These establish satisfiable
domains and are not GPU measurements.

## Counterexample and host boundaries

[Proofs/UndefinedOutput.lean](Proofs/UndefinedOutput.lean) refutes removing the
consumer's zero unspecified-value condition. With
`T=1, K=17, V=16, BT=16, BK=32, BV=16`, unit scale and zero program IDs,
valid Q/H/V/A values are one, G is zero, and permitted unspecified values are
one outside G. The complete erased kernel stores **33**, while its independent
zero-extended formula is **18**. All active accesses satisfy the declared
bounds. This covers both versions and all supported storage selections.
It is a model counterexample, not a GPU observation or an FP admission result.

The Python host allocates the split temporary with BT rows although kernel
slice addresses use T rows. `host_temporary_allocation_rejected` in
`bench/tests/ChunkGLAMergeReal.lean` proves a concrete extent failure: extent
4096 cannot contain an active read at 8192. The provided H test tensor also
omits the consumer's chunk axis. The per-kernel contracts expose these bounds;
they do not repair or certify host allocation, launch composition, races or
fixed-width address overflow.

## Validation

The ten `bench/tests/{FPChunkGLA*,ChunkGLA*Real}.lean` regressions check source
precision/casts/intrinsics, rewrite-domain refusal, full source executions,
nonzero launch IDs, tails, early returns, zero iterations, finite memory and
untouched tagged cells. The real examples distinguish matrix and flat strides.

All 72 case modules and ten regressions compile. The FP entry prints only the
atoms in the table. Source inventory gates and all five call/control-flow
pair checks pass. Independent comparator/trust replay accepted all 82 files:
414 named declarations were covered by 4,870 kernel theorem targets, with
1,333 source/configuration fingerprints matching at verification. The receipt
is `Logs/comparator-check-2_3bwsqe` in the repository root.

Earlier receipts for pre-consolidation files remain historical; the complete
case receipt above records the accepted consolidation. Scope and subsequent
work are tracked in `../verification_round.md`.

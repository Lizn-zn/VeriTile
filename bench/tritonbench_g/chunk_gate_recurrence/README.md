# chunk_gate_recurrence

Source: [chunk_gate_recurrence.py](chunk_gate_recurrence.py), TritonBench-G v1.
Both complete forward and backward sources have independent real correctness
and admitted FP equivalence proofs. The current whole-case independent replay
and trust checks pass.

## Reader entries

- [Kernels.lean](Kernels.lean): complete forward/backward Triton sources and
  their sum-left candidates. The candidates commute only the addition in
  `acc*d + S` or `Dacc*d + DS`, including backward's final DL update.
- [Correct.lean](Correct.lean): four real specifications, independent closed
  forms, initial-array reads, output windows, access bounds and aliasing rules.
- [FPEquiv.lean](FPEquiv.lean): two complete FP equivalence specifications,
  reached-update domain checks and the actual atom list, `add_commute`.

Source, gate and seed/incoming storage independently select fp32, fp16 or
bf16. Every load, storage conversion, output cast, whole-tile DG reduction,
branch, pointer update and store is retained. `offs_k/offs_v` in the backward
port name Python's repeated arange expressions. The optional seed is
specialized by a Boolean; dimensions and program IDs remain symbolic.

## Real correctness

Forward writes `O[0] = seed` (or zero), then
`O[t+1] = O[t]*D[t] + S[t]`. Its specification expands this into a seed/product/
sum formula over the initial arrays. Both versions require N>0, BK<=K and
BV<=V, with S/D separate from O. The seed may alias O because it is consumed
before the first store. The last S/D chunk is never loaded, so N=1 requires
no source/gate allocation.

Backward starts from `C[N] = 0`, uses `C[t] = C[t+1]*D[t] + DS[t]`, and writes
`DI[t] = C[t+1]`, `DG[t] = sum(C[t+1]*S[t])` for t<N-1, then `DL = C[0]`.
The public formula is an independent suffix sum/product; DG contracts all
BK×BV lanes. S is the saved forward output supplied by Python. Both versions
require N>0, positive BK/BV fitting K/V, DI/DG distinct from one another and
from S/D/DS, and DL distinct from DI/DG. DL may alias an input because its
store follows all loads. N=1 only writes DL=DS[0] and needs no S/DI/DG
allocation unless a live output aliases one of those regions.

Both contracts conclude successful whole-source execution in a disjoint flat
allocation and preservation of every other tagged memory cell. Bounds apply
to actual accesses; [RealForwardBounds](Proofs/RealForwardBounds.lean) and
[RealBackwardBounds](Proofs/RealBackwardBounds.lean) derive them from whole-block
fit and allocation lengths. The final unused S/DI/DG pointer values may differ
between region and flat execution, while all registers needed by the DL
postlude and all memory remain in correspondence. No head-index restriction,
supplied carry, FP atom or padding-oracle premise is added to the real contract.

The mathematical-address, per-program model does not compose the CUDA grid.
Python's last-slice DI/DG zeroing and DG reduction across programs are host
operations, not stores made by the backward kernel.

## FP equivalence

Both specifications use only `add_commute`. Their guards check the actual
fp32 product and promoted incoming operand at every reached update, after
storage conversion. Backward also checks the separate DL update against the
prefix containing the entire original loop. The proofs preserve the existing
products, reductions, casts and memory context; experiment shapes select the
atom and do not constrain symbolic proof dimensions.

## Proofs and checks

Execution, arithmetic, bounds and flat correspondence live in [Proofs](Proofs).
[LegacyReal.lean](Proofs/LegacyReal.lean) retains the historical source bridge
and materialized-carry slices; the public specifications use the complete
initialization/loop/postlude proofs, not those slice contracts.

The public entries compile and print only `add_commute` for both FP results.
`bench/tests/ChunkGatePublic.lean` exercises all four public specifications at
N=1, optional seed/output aliasing and absent DI/DG allocations, with a nonzero
initial padding oracle. The existing real/flat/typed regressions check three
chunks, nonuniform 2×2 tiles, mixed storage, nonzero program IDs, complete DG
contractions, input/output aliasing, unchanged final slices and tagged cells,
and storage promotion/casts. Source gates and coverage inventory checks pass.

Independent comparator/trust replay accepts **41 files**, **284 named proof targets** and **1521 kernel theorem targets**.
All **1,212 snapshot input hashes** match the checked workspace.
Receipt: `Logs/comparator-check-tou10lkr`. This covers all 36 case modules and five focused
regressions, including the public specifications and every named proof target.
Corpus acceptance is now 29 complete / 155 pending. Earlier milestone receipts
and scope are retained in [verification_round.md](../verification_round.md).

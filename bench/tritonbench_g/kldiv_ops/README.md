# kldiv_ops

- Source file: `kldiv_ops.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

Full-loop completion is in progress. [source_scope.md](source_scope.md) records
the two source branches and reduction behavior. `Proofs/Math.lean` passes its
compiler gate and module axiom audit: the independent padded-block prefix equals
the whole-row formula once the positive-stride blocks cover all columns. This
does not yet establish source execution or add a completed public contract.
The runtime-epsilon AST regression also passes, preserving fp32 promotion before
the default branch's logarithm.
`Proofs/Algorithm.lean` now also passes compilation and its module axiom audit:
both full signed-loop control flows lower successfully, preserving every branch
and the backward pre-loop assignments. Source execution correctness remains pending.
`Proofs/Initialize.lean` and `Proofs/Finish.lean` pass their compiler and module
axiom checks: both actual initialization prefixes execute with memory preserved,
and the forward post-loop statement stores the accumulated scalar exactly when
reduction is nonzero. The complete loop invariant remains to be proved.
`Proofs/BackwardStep.lean` passes compilation and its module axiom audit: the
actual signed offsets, masked zero loads, both gradient branches and cached-value
scatter execute for one complete block. The address-injectivity, active-lane
readback and raw-memory frame lemmas in `Proofs/BackwardMemory.lean` also pass.
The combined source-block IO theorem also passes: it proves actual successful
execution, the independent gradient formula for every active output, and the
full raw-memory frame. It allows arbitrary input/output aliases within that
block because loads precede stores. The complete loop invariant and public
full-kernel contracts remain pending.
`Proofs/BackwardState.lean` also passes: scatter preserves registers, program
identifiers, program counts and the undefined-value handler. `BackwardInvariant`
now passes too: every source block extends the verified prefix and preserves
the input region under the explicit region-separation hypothesis. The new
`BackwardLoop` now passes its compilation and module axiom audit: the actual
positive-stride signed loop reaches complete row coverage, and the covered
invariant implies the pointwise formula at every logical column.
`BackwardExecution` now passes compilation and its module axiom audit. It connects
the source initialization and signed loop to the
whole-row gradient formula for positive block size and distinct input/output
regions, including zero-length rows and partial final blocks. This is a complete
real algorithm IO/frame theorem; typed-source projection, access safety and
flat-memory/public contracts are still pending, so the port is not counted complete.
`ForwardStep` now passes compilation and module axiom audit: the actual signed
offsets, both masked loads, epsilon-clamped/log-target formulas and scalar block
sum are checked. `ForwardMath` also passes compilation and module axiom audit:
the cached values equal the independent padded-loss formula, and their sum equals
the independent block loss.
`ForwardFinish`, `ForwardBlock`, `ForwardReductionInvariant`,
`ForwardReductionLoop` and `ForwardReductionExecution` now pass compilation and
module axiom audits. The complete positive-stride loop accumulates all blocks
before its one scalar store, allowing output/input aliases; zero columns still
produce the source's final zero store. `ForwardElementInvariant`,
`ForwardElementLoop` and `ForwardElementExecution` also pass: separate output
and input regions yield the independent loss at every logical column, with
the complete raw-memory frame, including zero and partial-block rows.
`BackwardSafety` also passes compilation and module axiom audit: it follows
the actual masked load/store states and complete signed loop from explicit
active-address bounds. The complete twenty-module algorithm/execution/backward-safety
chain also passes after the shared negative-constexpr precision correction.

The new typed `Kernels.lean` and `Proofs/SourceDomain.lean` pass compilation
after the shared literal fix; the domain module also passes its axiom audit.
A static inventory check confirms exactly 216 default/72 log-target forward
instances and 36 default/12 log-target backward instances, with no missing or
duplicate storage tuples. The source-domain predicate records both raw-exp
input restrictions and the fp64 reduction-carry rejection described in
`source_scope.md`. Neither this inventory nor the draft predicate certifies
native compilation or full typed correctness.
The shared DSL now retains the weak scalar type for `target * -1`, verified by
four supported floating-type regressions and a narrow-storage check. The
port-specific full precision audit is still pending. The source-domain predicate
distinguishes its 36 formal default-backward pairs from the 24 pairs whose
target type has a nonzero scalar builder in pinned Triton.

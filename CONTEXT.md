# VeriTile

A Lean 4 verifier that proves an accelerated GPU kernel consistent with a golden
reference. Kernels written in supported DSLs (Triton, Tilelang) lower onto one shared
substrate, where proofs run on a real-valued algorithm layer.

## Language

### Verification layers

**Algorithm layer**:
The erased real-valued (ℝ/ℤ/ℕ) view of a kernel that Lean proofs reason about. No IEEE
rounding, NaN, or denormals.
_Avoid_: math layer, spec layer

**Compute gap**:
The difference between a kernel's concrete floating-point behavior and its algorithm
layer. Checked externally (test-backed contracts), never bridged by a Lean theorem.
_Avoid_: FP gap, numeric error

**Rounding model**:
The abstract, magnitude-free account of where rounding events occur in a kernel. It
captures rounding structure inside Lean while magnitudes remain the compute gap's job.

### Properties

**Fusion correctness**:
The property that a fused kernel exactly implements its stage pipeline on the declared
outputs, at the algorithm layer. It says nothing about memory outside the declared
outputs and nothing about floating-point magnitudes — the FP effect of eliminating an
intermediate store/load (a removed rounding site) is the rounding model's and the
compute gap's business.
_Avoid_: fusion equivalence, kernel fusion soundness

**Stage pipeline**:
The sequential composition of the stage kernels a fusion replaces — run the first
stage, then the next, each seeing the previous stage's memory. It is the golden
reference a fused kernel is proved against.
_Avoid_: unfused kernel, kernel chain

**Stage kernel**:
One kernel of the original multi-kernel computation that a fusion collapses; stages
communicate only through intermediate regions.
_Avoid_: sub-kernel, component kernel

### Foundation

**Foundation**:
The DSL-agnostic stack every property and every frontend rests on: the neutral core,
memory reasoning, correctness surfaces, the rounding model, launch/grid, the concurrency
substrate, and the pure-math and mechanism libraries.
_Avoid_: framework, platform

**Neutral core**:
The single DSL-agnostic IR (typed operator/statement/kernel representation) and its
operational semantics that all frontends lower onto. Obtained by generalizing the
original Triton-branded core, governed by the union policy.
_Avoid_: Triton core, common operators library

**Union policy**:
The neutral core's inclusion rule: any construct from any supported DSL that has clean
algorithm-layer semantics belongs in the core (one closed type, no frontend extension
points). Provenance is documentation, not types.
_Avoid_: intersection core, extensible core

**Frontend**:
A DSL adapter contributing only surface syntax and lowering onto the neutral core.
Frontends own no semantic nodes and no proof surfaces.
_Avoid_: DSL layer, dialect

**Per-kernel glue**:
Kernel-specific code sitting on top of the foundation: the DSL transcription, local
spec/load/offset helpers, and the kernel's correctness theorem. The three-layer
discipline (pure math / mechanisms / glue) decides what may live here.

### Memory

**Region**:
A named buffer in the flat semantic memory; the unit of addressing, bounds, and
footprint reasoning. Cell offsets are logical, not byte addresses.
_Avoid_: buffer, allocation

**Memory-space tag**:
Region metadata naming the memory scope (global / shared / fragment). Visibility and
safety rules may refer to it; the semantic memory stays flat regardless.
_Avoid_: address space, memory hierarchy level

**World layer**:
The multi-rank state (rank → per-rank memory) layered above the single-rank block
state for async/distributed verification. Peer addressing (put/get/signal) is expressed
against it; single-GPU proofs never see it.
_Avoid_: global state, cluster state

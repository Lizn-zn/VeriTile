/-
VeriTile.Triton.Concurrency.Async

Proof-facing async/TMA sequentialization contract for #12/#67.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-!
This module records the first async/TMA proof boundary. It intentionally does
not add async/TMA AST nodes, does not change `Kernel.exec`, and does not add
shared-memory or in-flight operation state.

The intended architecture mirrors the limited `atomic_add` slice:

* Compute-facing async/TMA syntax may be added later.
* `ComputeKernel.toAlgorithm?` may project disciplined patterns to AlgKernel
  markers.
* AlgKernel proofs eliminate those markers into ordinary mathematical specs.
-/

/-- Projection failure categories for async/TMA-shaped effects. -/
inductive AsyncProjectionFailure where
  | unmatchedIssueWait
  | readBeforeWait
  | ambiguousDestination
  | overlappingDestinations
  | crossProgramSharedMemory
  | requiresMemoryHierarchy
  | unsupportedPattern
  deriving Repr, BEq, DecidableEq

namespace AsyncProjectionFailure

def message : AsyncProjectionFailure → String
  | .unmatchedIssueWait => "async issue/wait pairs are unmatched"
  | .readBeforeWait => "async destination is read before the matching wait"
  | .ambiguousDestination => "async destination ownership is ambiguous"
  | .overlappingDestinations => "overlapping async destinations need an explicit order"
  | .crossProgramSharedMemory => "cross-program shared-memory communication is unsupported"
  | .requiresMemoryHierarchy => "explicit shared memory/TMA state requires #65"
  | .unsupportedPattern => "async/TMA pattern is not recognized"

end AsyncProjectionFailure

namespace Kernel

/--
Every async issue has a matching wait in the program slice.

This is a contract hook for future async/TMA AST nodes. It is `True` for the
current language because no async nodes exist yet.
-/
def AsyncIssueWaitMatched (_k : Kernel) : Prop :=
  True

/--
No consumer reads an async destination before the matching wait makes it
logically visible.
-/
def AsyncNoReadBeforeWait (_k : Kernel) : Prop :=
  True

/--
The program slice has unambiguous ownership of each async destination.

This excludes patterns that need cross-program shared-memory communication or
a scheduler/interleaving model.
-/
def AsyncDestinationOwned (_k : Kernel) : Prop :=
  True

/--
Overlapping async writes are either ordered by the recognized pattern or
rejected during projection.
-/
def AsyncOverlapsOrdered (_k : Kernel) : Prop :=
  True

/--
Minimal discipline required before an async/TMA-shaped compute kernel may be
projected into an AlgKernel marker.

The fields are intentionally factored so future implementation slices can
replace the `True` hooks above with concrete syntactic or semantic checks
without changing theorem statements that consume `AsyncDiscipline`.
-/
structure AsyncDiscipline (k : Kernel) : Prop where
  issueWaitMatched : AsyncIssueWaitMatched k
  noReadBeforeWait : AsyncNoReadBeforeWait k
  destinationOwned : AsyncDestinationOwned k
  overlapsOrdered : AsyncOverlapsOrdered k

/--
`source` projects to an AlgKernel async/TMA marker.

This is a relation rather than a function because the concrete marker AST node
is intentionally deferred. Future PRs should instantiate this relation with a
specific `Stmt` marker once a real async/TMA surface exists.
-/
def AsyncProjectsToMarker (_source marker : Kernel) : Prop :=
  AsyncDiscipline marker

/--
An AlgKernel async/TMA marker is eliminated into a sequential mathematical
target kernel or theorem surface.

This is the async/TMA analogue of proving that an AlgKernel `atomicAdd` marker
matches a mathematical sum.
-/
def AsyncMarkerEliminatesTo (_marker target : Kernel) : Prop :=
  True ∧ AsyncDiscipline target

/--
Complete proof-facing sequentialization package for #67.

`source` is the async/TMA-shaped algorithm projection, `marker` is the
proof-facing AlgKernel marker, and `target` is the ordinary mathematical
behavior after eliminating the marker.
-/
structure AsyncSequentialization (source marker target : Kernel) : Prop where
  disciplined : AsyncDiscipline source
  projectsToMarker : AsyncProjectsToMarker source marker
  markerEliminatesTo : AsyncMarkerEliminatesTo marker target

/--
The theorem shape #67 wants future concrete async-copy/TMA proofs to refine:
once a disciplined source projects to an AlgKernel marker, the marker can be
eliminated into the mathematical target behavior.
-/
theorem asyncCopy_wait_sequentializes {source marker target : Kernel}
    (h : AsyncSequentialization source marker target) :
    AsyncMarkerEliminatesTo marker target :=
  h.markerEliminatesTo

end Kernel

end VeriTile.Triton

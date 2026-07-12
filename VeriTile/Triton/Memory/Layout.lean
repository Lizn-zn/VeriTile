/-
VeriTile.Triton.Memory.Layout

Spec-interface bundles for flat-memory specifications. A flat-memory
headline quantifies over (1) a well-formed allocation that covers the
kernel's footprint and (2) a launch state; before these bundles each
specification restated the same six lines of layout bookkeeping
(`Disjoint`, extent closure, per-region extent bounds, `undef` cleanliness).
Bundling them keeps every remaining hypothesis line about the *content* of
the spec: which arrays are loaded where, and what closed form the output
realizes.
-/

import VeriTile.Triton.Memory.Flatten
import VeriTile.Meta.Specification

namespace VeriTile.Triton

/-- A well-formed flat allocation covering `footprint`: regions are
disjoint, extents vanish outside the region set, and each
`(region, size)` pair of the footprint fits inside its region's extent. -/
structure FlatLayout (footprint : List (RegionName × Nat)) where
  alloc : FlatAlloc
  disjoint : alloc.Disjoint
  closed : ∀ r, r ∉ alloc.regions → alloc.extent r = 0
  covers : ∀ p ∈ footprint, p.2 ≤ alloc.extent p.1

namespace FlatLayout

variable {footprint : List (RegionName × Nat)} (L : FlatLayout footprint)

/-- Extent bound for a footprint entry, keyed by the pair. -/
theorem covers_mem {r : RegionName} {k : Nat}
    (h : (r, k) ∈ footprint) : k ≤ L.alloc.extent r :=
  L.covers (r, k) h

/-- Any region with a positive extent is allocated (contrapositive of
`closed`) — this discharges the `r ∈ regions` side conditions of the
bridge's read-back lemmas without a separate membership hypothesis. -/
theorem mem_regions {r : RegionName} (h : 0 < L.alloc.extent r) :
    r ∈ L.alloc.regions := by
  by_contra hn
  exact absurd (L.closed r hn) (Nat.pos_iff_ne_zero.mp h)

end FlatLayout

/-- A launch state: what a kernel actually starts from — every memory cell
well-defined (no uninitialized poison at launch). Execution may mark
`undef` as it runs, so this is a property of *initial* states only; making
it a subtype keeps it out of every specification's hypothesis list. -/
def LaunchState := {s : BlockState // s.undef = fun _ _ => 0}

instance : Coe LaunchState BlockState := ⟨Subtype.val⟩

/-- The launch-state cleanliness fact, in the form the bridge consumes. -/
theorem LaunchState.undef_eq (s : LaunchState) :
    (s : BlockState).undef = fun _ _ => 0 := s.2

/-- The current program id, read through the coercion — so specifications
write `s.pid`, not `(s : BlockState).pid`. -/
def LaunchState.pid (s : LaunchState) : Nat := (s : BlockState).pid

end VeriTile.Triton

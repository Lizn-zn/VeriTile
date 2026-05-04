/-
VeriTile.Triton.Concurrency.Discipline

First-slice discipline predicates for #81 producer-consumer reasoning.
-/

import VeriTile.Triton.Concurrency.Trace

namespace VeriTile.Triton

namespace EffectTraceEvent

def IsAsyncCopy (event : EffectTraceEvent) : Prop :=
  match event.event with
  | .async (.copy _ _) => True
  | _ => False

def IsAsyncWait (event : EffectTraceEvent) : Prop :=
  match event.event with
  | .async .wait => True
  | _ => False

def IsBarrierArrive (name : String) (event : EffectTraceEvent) : Prop :=
  match event.event with
  | .barrier (.arrive name') => name' = name
  | _ => False

def IsBarrierWait (name : String) (event : EffectTraceEvent) : Prop :=
  match event.event with
  | .barrier (.wait name') => name' = name
  | _ => False

end EffectTraceEvent

namespace EffectTrace

/-- Every concrete permission in an ownership map is valid. -/
def OwnershipValid {Perm : Type} [PermissionModel Perm]
    (ownership : OwnershipMap Perm) : Prop :=
  ∀ cell perm, ownership cell = some perm → PermissionModel.valid perm

/-- Ownership transfer over a selected footprint.

This first slice records the proof obligation without committing to a concrete
permission implementation. Richer permission instances can strengthen what it
means for the same permission to be transferable. -/
def OwnershipTransfer {Perm : Type} [PermissionModel Perm]
    (before after : OwnershipMap Perm) (moved : WriteFootprint) : Prop :=
  ∀ cell, moved cell →
    ∃ perm, before cell = some perm ∧ after cell = some perm ∧
      PermissionModel.valid perm

/-- Race-freedom over owned cells as a proof-facing predicate. -/
def NoRaceOnOwnedCells {Perm : Type} [PermissionModel Perm]
    (ownership : OwnershipMap Perm) (writes : WriteFootprint) : Prop :=
  ∀ cell, writes cell → ∃ perm, ownership cell = some perm ∧ PermissionModel.valid perm

/-- Every async wait is justified by a prior async copy in the effect trace. -/
def AsyncVisibility (trace : EffectTrace) : Prop :=
  ∀ wait, wait ∈ trace →
    wait.IsAsyncWait →
      ∃ copy, copy ∈ trace ∧ copy.IsAsyncCopy ∧ HappensBefore trace copy wait

/-- Every barrier wait is justified by a prior arrive on the same barrier name. -/
def BarrierDiscipline (trace : EffectTrace) : Prop :=
  ∀ wait name, wait ∈ trace →
    wait.IsBarrierWait name →
      ∃ arrive, arrive ∈ trace ∧ arrive.IsBarrierArrive name ∧
        HappensBefore trace arrive wait

/-- No async/barrier discipline obligations for an empty trace. -/
@[simp] theorem asyncVisibility_nil : AsyncVisibility [] := by
  intro wait hmem
  cases hmem

/-- No barrier discipline obligations for an empty trace. -/
@[simp] theorem barrierDiscipline_nil : BarrierDiscipline [] := by
  intro wait name hmem
  cases hmem

end EffectTrace

end VeriTile.Triton

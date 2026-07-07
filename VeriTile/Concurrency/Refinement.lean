/-
VeriTile.Concurrency.Refinement

Final-state refinement surface for #81.
-/

import VeriTile.Concurrency.Discipline

namespace VeriTile

namespace ConcurrentTrace

/-- Final-state refinement between two state transformers.

This is intentionally weaker than trace inclusion or bisimulation. It is the
first theorem surface for #81 producer-consumer correctness: every successful
concurrent execution has a successful sequential execution with the same final
memory. -/
def RefinesSequential
    (concurrent sequential : BlockState → Option BlockState) : Prop :=
  ∀ s cFinal, concurrent s = some cFinal →
    ∃ sFinal, sequential s = some sFinal ∧ cFinal.mem = sFinal.mem

/-- Reflexivity for deterministic final-state refinement. -/
theorem refinesSequential_refl (f : BlockState → Option BlockState) :
    RefinesSequential f f := by
  intro s cFinal h
  exact ⟨cFinal, h, rfl⟩

end ConcurrentTrace

end VeriTile


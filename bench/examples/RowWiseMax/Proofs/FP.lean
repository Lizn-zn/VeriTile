import VeriTile.Triton.Float.Kernel
import bench.examples.RowWiseMax.Kernels
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit

/-! FP execution and scalar composition lemmas for RowWiseMax.
Public IO, numerical domains and the equivalence specification are in ../FPEquiv.lean. -/

namespace VeriTile.Bench.Examples.RowWiseMaxFPEquiv
open VeriTile.Bench.Examples.RowWiseMax.Kernels
open VeriTile Triton
open FP.Structural

set_option maxHeartbeats 1200000

def reducedValue {α : Type} (M : Algebra α) (B : Nat) (xs : Fin B → α) : α :=
  M.reduceMax none (shape := [B]) ⟨0, by simp⟩ Bool.false
    (fun i => xs i.1) PUnit.unit

theorem original_run {α : Type} [Inhabited α] (M : Algebra α)
    (nCol B : Nat) (hB : 0 < B) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * nCol + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (rowWiseMaxKernel "x" "y" nCol B) s = some t ∧
      t.mem "y" (s.pids 0) = Cell.mk .real (reducedValue M B xs) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [rowWiseMaxKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.axisDim, TileShape.allIndices,
    TileShape.eraseAxis,
    hB, hx, reducedValue]
  intro r o hmiss
  exact (State.write_other _ "y" r (s.pids 0) o _ hmiss).trans rfl

theorem inlined_run {α : Type} [Inhabited α] (M : Algebra α)
    (nCol B : Nat) (hB : 0 < B) (s : State α) (xs : Fin B → α)
    (hx : ∀ i : Fin B, (s.mem "x" (s.pids 0 * nCol + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (inlinedKernel "x" "y" nCol B) s = some t ∧
      t.mem "y" (s.pids 0) = Cell.mk .real (reducedValue M B xs) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ o ≠ s.pids 0) → t.mem r o = s.mem r o) := by
  simp [inlinedKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, TileShape.axisDim, TileShape.allIndices,
    TileShape.eraseAxis,
    hB, hx, reducedValue]
  intro r o hmiss
  exact (State.write_other _ "y" r (s.pids 0) o _ hmiss).trans rfl

open scoped VeriTile.Spec

/- Internal execution contracts mirror the public fields in FPEquiv.lean.
The entry's final theorem checks their definitional equality. -/
namespace Execution

def originalIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseMaxKernel "x" "y" nCol B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := id

def inlinedIO (nCol B : Nat) : KernelIO₁ :=
  (originalIO nCol B).withKernel (inlinedKernel "x" "y" nCol B)


open scoped VeriTile.Spec

/-- Compose execution, numerical laws and the memory frames. -/
theorem equivalent (nCol B : Nat) (hB : 0 < B) :
    IO₁Equiv (originalIO nCol B) (inlinedIO nCol B) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [IO₁PrivateScratch, originalIO]
  · simp [IO₁PrivateScratch, inlinedIO, originalIO]
  · intro α _ M s
    let xs : Fin B → α := fun i => (s.mem "x" (s.pids 0 * nCol + i.val)).read .real
    obtain ⟨a, ha, hva, hfa⟩ := original_run M nCol B hB s xs (fun _ => rfl)
    obtain ⟨b, hb, hvb, hfb⟩ := inlined_run M nCol B hB s xs (fun _ => rfl)
    refine ⟨a, b, ha, hb, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := by have := i.isLt; change i.val < 1 at this; omega
      simpa [originalIO, inlinedIO, hi] using hva.trans hvb.symm
    · intro r o ho _
      apply hfa r o
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by simpa [originalIO] using ho ⟨0, by change 0 < 1; decide⟩)
    · intro r o ho _
      apply hfb r o
      rcases ho with hr | ho
      · exact Or.inl hr
      · exact Or.inr (by simpa [inlinedIO, originalIO] using ho ⟨0, by change 0 < 1; decide⟩)

end Execution

end VeriTile.Bench.Examples.RowWiseMaxFPEquiv

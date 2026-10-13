import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealFlatFrame
open VeriTile Triton

def footprint (A : FlatAlloc) (P : WriteFootprint) : WriteFootprint :=
  fun a ↦ a.1 = A.flat ∧ ∃ r o, P (r, o) ∧ a.2 = A.addr r o

theorem transport (A : FlatAlloc) (P : WriteFootprint) (s t : BlockState)
    (h : BlockState.WriteWithin P s t) :
    BlockState.WriteWithin (footprint A P) (A.flattenState s) (A.flattenState t) := by
  intro r off hnot
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState]
    unfold FlatAlloc.readFlat
    cases hdec : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨r, o⟩ := cell
      obtain ⟨_, haddr, _⟩ := A.decode_sound hdec
      change A.trCell (s.mem r o) = A.trCell (t.mem r o)
      apply congrArg A.trCell
      apply h r o
      intro hp
      exact hnot ⟨rfl, r, o, hp, haddr⟩
  · simp only [FlatAlloc.flattenState, if_neg hr]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealFlatFrame

import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroComplexBackward
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroIO
open VeriTile VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 600000

/-- Real zero-time IO. The static access bounds are checked against a finite
allocation even when all time windows are empty. The conclusion includes
source trace safety, successful flattened execution, exact zero output cells
and the full tagged-cell frame. -/
def Implements (kernel : ComputeKernel)
    (bounds : RegionBounds → BlockState → Prop)
    (writes : FlatAlloc → BlockState → RegionName → Nat → Prop) : Prop :=
  ∀ (A : FlatAlloc), A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ u : BlockState, bounds A.extent u →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u) = some v ∧
      (∀ r o, writes A u r o → v.mem r o = MemCell.real 0) ∧
      (∀ r o, ¬ writes A u r o → v.mem r o = (A.flattenState u).mem r o)

theorem zero_readback {v : BlockState} {r : RegionName} {o : Nat}
    (h : v.mem r o = MemCell.real 0) : v.readMem r o = 0 := by
  unfold BlockState.readMem
  rw [h]
  rfl

theorem forward_correct (storage : Storage) (swapped : Bool) (s x l y : RegionName)
    (batch dim block : Nat) :
    Spec.Real (Implements
      (TypedSource.forward storage swapped s x l y 0 batch dim block).eraseDType
      (fun b u => ZeroSafety.forwardBounds b u s l batch dim block)
      (fun _ _ _ _ => False)) := by
  intro A _hd _hc u hb
  refine ⟨ZeroSafety.forward_safe_typed storage swapped A.extent u s x l y batch dim block hb,?_⟩
  obtain ⟨v,he,hm⟩ := ZeroTime.forward_memory storage swapped A s x l y batch dim block (A.flattenState u)
  exact ⟨v,he,fun _ _ h => h.elim,fun r o _ => hm r o⟩

theorem complexForward_correct (swapped : Bool) (s x y l : RegionName)
    (batch dim block : Nat) :
    Spec.Real (Implements
      (TypedSource.complexForward swapped s x y l 0 batch dim block).eraseDType
      (fun b u => ZeroSafety.complexForwardBounds b u s l batch dim block)
      (fun _ _ _ _ => False)) := by
  intro A _hd _hc u hb
  refine ⟨ZeroSafety.complex_forward_safe_typed swapped A.extent u s x y l batch dim block hb,?_⟩
  obtain ⟨v,he,hm⟩ := ZeroTime.complex_forward_memory swapped A s x y l batch dim block (A.flattenState u)
  exact ⟨v,he,fun _ _ h => h.elim,fun r o _ => hm r o⟩

theorem backward_correct (storage : Storage) (swapped : Bool) (s l y gs gx gl gy : RegionName)
    (batch dim block : Nat) :
    Spec.Real (Implements
      (TypedSource.backward storage swapped s l y gs gx gl gy 0 batch dim block).eraseDType
      (fun b u => ZeroSafety.backwardBounds b u l gs gl batch dim block)
      (fun A u => ZeroBackward.written A u batch dim block gs gl)) := by
  intro A _hd _hc u hb
  refine ⟨ZeroSafety.backward_safe_typed storage swapped A.extent u s l y gs gx gl gy batch dim block hb,?_⟩
  obtain ⟨v,he,hm⟩ := ZeroBackward.backward_memory storage swapped A s l y gs gx gl gy batch dim block (A.flattenState u)
  have hp : ZeroBackward.written A (A.flattenState u) batch dim block gs gl =
      ZeroBackward.written A u batch dim block gs gl := rfl
  rw [hp] at hm
  refine ⟨v,he,?_,?_⟩
  · intro r o hw; rw [hm,if_pos hw]
  · intro r o hw; rw [hm,if_neg hw]

theorem complexBackward_correct (swapped : Bool) (s l y gs gx gl gy : RegionName)
    (batch dim block : Nat) :
    Spec.Real (Implements
      (TypedSource.complexBackward swapped s l y gs gx gl gy 0 batch dim block).eraseDType
      (fun b u => ZeroSafety.complexBackwardBounds b u l gs gl batch dim block)
      (fun A u => ZeroComplexBackward.written A u batch dim block gs gl)) := by
  intro A _hd _hc u hb
  refine ⟨ZeroSafety.complex_backward_safe_typed swapped A.extent u s l y gs gx gl gy batch dim block hb,?_⟩
  obtain ⟨v,he,hm⟩ := ZeroComplexBackward.complex_backward_memory swapped A s l y gs gx gl gy batch dim block (A.flattenState u)
  have hp : ZeroComplexBackward.written A (A.flattenState u) batch dim block gs gl =
      ZeroComplexBackward.written A u batch dim block gs gl := rfl
  rw [hp] at hm
  refine ⟨v,he,?_,?_⟩
  · intro r o hw; rw [hm,if_pos hw]
  · intro r o hw; rw [hm,if_neg hw]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ZeroIO

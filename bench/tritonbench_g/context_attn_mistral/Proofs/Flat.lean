import bench.tritonbench_g.context_attn_mistral.Proofs.WholeSafety
import bench.tritonbench_g.context_attn_mistral.Proofs.FlattenShape
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.Flat
open VeriTile.Triton Formula Program EpilogueResult
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (r : Regions) (initial : BlockState) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, Writes p r initial rg off ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (r : Regions) (initial u v : BlockState)
    (hf : ∀ rg off, ¬ Writes p r initial rg off → v.mem rg off = u.mem rg off) :
    BlockState.WriteWithin (footprint A p r initial) (A.flattenState u) (A.flattenState v) := by
  intro rg off hn
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      symm
      apply hf
      intro hh
      exact hn ⟨rfl,rg',off',hh,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

/-- Finite flat execution, exact real formula, and a full-cell write frame.
Changing the region oracle to the flat adapter's zero oracle changes neither
input arrays nor initial flat state; all masked source loads use other=0. -/
theorem correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ)
    (initial : BlockState) (hBN : 0 < p.BN)
    (hi : ∀ i j, activeLane p r initial i → activeLane p r initial j →
      offset p r initial i = offset p r initial j → i = j)
    (hb : AddressBounds.Valid p r (launch initial) (inputs initial r) A.extent) :
    ∃ final,
      exec (A.flattenKernel (kernel p r storage swapped scale).eraseDType.toAlgKernel) (A.flattenState initial) = some final ∧
      (∀ i, activeLane p r initial i → final.readMem A.flat (A.addr r.out (offset p r initial i)) =
        WholeExecution.output p r initial scale i.1 i.2.1) ∧
      BlockState.WriteWithin (footprint A p r initial) (A.flattenState initial) final := by
  let u : BlockState := { initial with undef := fun _ _ => 0 }
  obtain ⟨v,he,ho,hf⟩ := EpilogueResult.correct p r storage swapped scale u hBN hb.advance hi
  have hflat := A.exec_flatten hd hcov _ u
    (WholeSafety.complete p r storage swapped scale u hBN A.extent hb)
    (FlattenShape.supported p r storage swapped scale) rfl
  change exec _ u = some v at he
  rw [he,Option.map_some] at hflat
  refine ⟨A.flattenState v,hflat,?_,flatten_frame A p r initial u v hf⟩
  intro i hi'
  have hbound := AddressBounds.inside_toNat (hb.output i.1 i.2.1 hi')
  have hr : r.out ∈ A.regions := by
    by_contra hn
    rw [hcov r.out hn] at hbound
    exact Nat.not_lt_zero _ hbound
  change offset p r initial i < A.extent r.out at hbound
  rw [A.flattenState_readMem hd v hr hbound]
  exact ho i hi'

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.Flat

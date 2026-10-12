import bench.tritonbench_g.cross_entropy2.Proofs.BackwardExecution
import bench.tritonbench_g.cross_entropy2.Proofs.BackwardSafety
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardFlat
open VeriTile.Triton BackwardFormula BackwardProgram
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 8000

def writes (p : Layout) (r : Regions) (u : BlockState) : WriteFootprint := fun cell =>
  ∃ i : Fin p.B, active p (launch u) i.val ∧ r.gradient=cell.1 ∧
    gradientAddress p (launch u) i.val=cell.2

def footprint (A : FlatAlloc) (p : Layout) (r : Regions) (u : BlockState) : WriteFootprint := fun cell =>
  cell.1=A.flat ∧ ∃ region offset, writes p r u (region,offset) ∧ A.addr region offset=cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (r : Regions) (u v : BlockState)
    (h : ∀ region offset, (∀ i : Fin p.B, active p (launch u) i.val →
      ¬ (r.gradient=region ∧ gradientAddress p (launch u) i.val=offset)) →
        v.mem region offset=u.mem region offset) :
    BlockState.WriteWithin (footprint A p r u) (A.flattenState u) (A.flattenState v) := by
  intro region offset hn
  by_cases hr : region=A.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode offset with
    | none => rfl
    | some cell =>
      obtain ⟨rg,off⟩ := cell
      obtain ⟨_,he,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      symm
      apply h
      intro i hi hc
      exact hn ⟨rfl,rg,off,⟨i,hi,hc⟩,he.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

/-- The complete typed source, after real erasure, executes over finite flat
memory and preserves every non-output cell. Explicit masked padding makes
the result independent of the original unspecified-value oracle. -/
theorem correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r=0)
    (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (initial : BlockState)
    (hb : BackwardSafety.Bounds p r (launch initial) (inputs initial r) A.extent) :
    ∃ final,
      exec (A.flattenKernel (source p r storage swapped).eraseDType.toAlgKernel)
        (A.flattenState initial)=some final ∧
      (∀ i : Fin p.B, active p (launch initial) i.val →
        final.readMem A.flat (A.addr r.gradient (gradientAddress p (launch initial) i.val)) =
          expected p (launch initial) (inputs initial r) i.val) ∧
      BlockState.WriteWithin (footprint A p r initial) (A.flattenState initial) final := by
  let u : BlockState := { initial with undef := fun _ _ => 0 }
  obtain ⟨v,he,hv,hf⟩ := BackwardExecution.correct p r u
  have hs := (executes_as_base p r storage swapped u).trans he
  have hflat := A.exec_flatten hd hcov _ u
    (safe_as_base p r storage swapped A.extent u (BackwardSafety.traceSafe p r _ u hb))
    (flatten_as_base p r storage swapped (BackwardSafety.flattenOk p r)) rfl
  rw [hs,Option.map_some] at hflat
  refine ⟨A.flattenState v,hflat,?_,flatten_frame A p r u v hf⟩
  intro i hi
  have hbound := hb.gradient i hi
  have hrg : r.gradient ∈ A.regions := by
    by_contra hn
    rw [hcov r.gradient hn] at hbound
    exact Nat.not_lt_zero _ hbound
  rw [A.flattenState_readMem hd v hrg hbound]
  exact hv i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardFlat

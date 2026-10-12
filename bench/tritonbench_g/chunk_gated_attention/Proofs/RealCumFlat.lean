import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumSafety

/-! Both complete cumulative sources in disjoint flat allocations. Only active
addresses require allocation, and every other tagged cell is preserved.
The flattening bridge's zero oracle is internal: flattenState resets it, and
the independent observed prefix formula holds for arbitrary region padding. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumFlat
open VeriTile.Triton RealCumProgram RealCumExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (c : Nat) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = A.flat ∧ ∃ i, tileActive p l c i ∧ A.addr o (offset p l c i) = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (c : Nat) (o : RegionName)
    (u v : BlockState) (hf : BlockState.WriteWithin (RealCumExecution.footprint p l c o) u v) :
    BlockState.WriteWithin (footprint A p l c o) (A.flattenState u) (A.flattenState v) := by
  intro rg off hmiss
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      rintro ⟨rfl,i,hi,rfl⟩
      exact hmiss ⟨rfl,i,hi,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType)
    (u : BlockState) (hinj : NoCollision p (launch u) (chunk u))
    (hb : RealCumSafety.Bounds p (launch u) (chunk u) s o A.extent) :
    ∃ v, exec (A.flattenKernel (RealCumSource.source swapped dtype out_dtype s o
        p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      (∀ i, tileActive p (launch u) (chunk u) i →
        v.readMem A.flat (A.addr o (offset p (launch u) (chunk u) i)) =
          expected p (launch u) (chunk u) (u.readMem s) i) ∧
      BlockState.WriteWithin (footprint A p (launch u) (chunk u) o) (A.flattenState u) v := by
  let u0 : BlockState := { u with undef := fun _ _ => 0 }
  obtain ⟨v,he,ho,hf⟩ := RealCumExecution.source_executes p s o swapped dtype out_dtype u0 hinj
  have hs := RealCumSafety.source_safe p s o swapped dtype out_dtype u0 A.extent hb
  have hflat := A.exec_flatten hd hcov _ u0 hs
    (RealCumSafety.source_flatten_ok p s o swapped dtype out_dtype) rfl
  rw [he,Option.map_some] at hflat
  refine ⟨A.flattenState v,hflat,?_,flatten_frame A p (launch u) (chunk u) o u0 v hf⟩
  intro i hi
  have hoff := hb.output i hi
  have hr : o ∈ A.regions := by
    by_contra hn
    rw [hcov o hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [A.flattenState_readMem hd v hr hoff]
  exact ho i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumFlat

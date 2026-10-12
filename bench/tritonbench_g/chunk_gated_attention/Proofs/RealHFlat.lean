import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHSafety

/-! Both complete H kernels in disjoint finite flat allocations. The explicit
zero-unspecified-value premise is retained from the region theorem. Active
history/final outputs match the independent closed form, and every other whole
tagged cell is preserved. Bounds and separation are on actual source accesses. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFlat
open VeriTile.Triton RealHMath RealHProgram RealHAccessSafety
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, RealHExecution.footprint p l r storeFinal (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) (u v : BlockState)
    (hf : BlockState.WriteWithin (RealHExecution.footprint p l r storeFinal) u v) :
    BlockState.WriteWithin (footprint A p l r storeFinal) (A.flattenState u) (A.flattenState v) := by
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
      intro hhit
      exact hmiss ⟨rfl,rg',off',hhit,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

structure Outputs (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (gateK hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a gateK hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a gateK hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (launch u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (launch u) (RealHFinal.base p (launch u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hg : ∀ off, u.readMem r.gate off = a.gate off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off)
    (hb : RealHSafety.Bounds p (launch u) r gateK hasSeed storeFinal A.extent) :
    ∃ v, exec (A.flattenKernel (source p r dot gate history seed final gateK hasSeed storeFinal swapped).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧ Outputs A p (launch u) r a gateK hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint A p (launch u) r storeFinal) (A.flattenState u) v := by
  obtain ⟨v,he,ho,hf⟩ := RealHExecution.executes p r dot gate history seed final gateK hasSeed storeFinal swapped a u
    hu hs hfsep hhi hfi hk hv hg hseed
  have hflat := A.exec_flatten hd hcov _ u
    (RealHSafety.source_safe p r dot gate history seed final gateK hasSeed storeFinal swapped u hu hs hhi A.extent hb)
    (RealHSafety.source_flatten_ok p r dot gate history seed final gateK hasSeed storeFinal swapped) hu
  rw [he,Option.map_some] at hflat
  have hmem (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      exact Nat.not_lt_zero _ hbound
    exact A.flattenState_readMem hd v hr hbound
  refine ⟨A.flattenState v,hflat,⟨?_,?_⟩,flatten_frame A p _ r storeFinal u v hf⟩
  · intro c e j hc ha
    have hoff : historyAddress p (launch u) c (keyIndex p (launch u) e.val) (valueIndex p (launch u) j.val) < A.extent r.history :=
      (hb.loop c hc).history e j ha
    rw [hmem _ _ hoff]
    exact ho.history c e j hc ha
  · intro hon e j ha
    have hoff : stateAddress p (launch u) (keyIndex p (launch u) e.val) (valueIndex p (launch u) j.val) < A.extent r.final := by
      simpa only [stateAddress,keyIndex,valueIndex,Nat.mul_one] using hb.final hon e j ha
    rw [hmem _ _ hoff]
    exact ho.final hon e j ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHFlat

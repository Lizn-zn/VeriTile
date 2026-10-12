import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealSafety
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealCandidate

/-! Complete execution of both sources in one flat allocation. History,
corrected values and optional final state match the independent recurrence;
every other full tagged cell is preserved. Omitted-padding values, active
address bounds, separation and address injectivity remain explicit premises. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealFlat
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, RealExecution.footprint p l r storeFinal (rg,off) ∧
    A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
    (storeFinal : Bool) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealExecution.footprint p l r storeFinal) s t) :
    BlockState.WriteWithin (footprint A p l r storeFinal) (A.flattenState s) (A.flattenState t) := by
  intro rg off hmiss
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState, if_pos rfl, FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hhit
      exact hmiss ⟨rfl,rg',off',hhit,heq.symm⟩
  · simp only [FlatAlloc.flattenState, if_neg hr]

structure Outputs (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (a₀ : Nat → ℝ)
    (useSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT →
    RealStateStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.history (RealOuterStep.historyAddress p l c e.val j.val)) =
      RealWindow.expected p l a (RealInit.seed p l a₀) useSeed c e.val (valueIndex p l j.val)
  corrected : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    v.readMem A.flat (A.addr r.corrected (valueAddress p l t j.val)) =
      RealMath.corrected (RealValues.arrays p l a (RealInit.seed p l a₀)) p.K
        (RealWindow.expected p l a (RealInit.seed p l a₀) useSeed (t / p.BT)) t (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealStateStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.final (RealStateStore.address p l (RealInit.base p l) p.V e.val j.val)) =
      RealWindow.expected p l a (RealInit.seed p l a₀) useSeed p.NT e.val (valueIndex p l j.val)

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (swapped useSeed storeFinal : Bool) (a : Inputs) (a₀ : Nat → ℝ) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hk : u.pids 0 = 0) (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC)
    (hcover : p.T ≤ p.NT * p.BT) (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final)
    (hvi : RealInnerLoop.NoCollision p (RealInit.launch u))
    (hhi : RealOuterLoop.HistoryNoCollision p (RealInit.launch u))
    (hfi : storeFinal = Bool.true → RealStateStore.NoCollision p (RealInit.launch u)
      (RealInit.base p (RealInit.launch u)) p.V)
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hseed : ∀ off, u.readMem r.initial off = a₀ off) 
    (hb : RealSafety.Bounds p (RealInit.launch u) r useSeed storeFinal A.extent) :
    ∃ v, exec (A.flattenKernel
        (RealCandidate.source p r kd value initial final swapped useSeed storeFinal).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      Outputs A p (RealInit.launch u) r a a₀ useSeed storeFinal v ∧
      BlockState.WriteWithin (footprint A p (RealInit.launch u) r storeFinal) (A.flattenState u) v := by
  obtain ⟨v,he,ho,hf⟩ := RealExecution.executes p r kd value initial final useSeed storeFinal a a₀ u
    hu hk hK hBT hBC hcover hs hfsep hvi hhi hfi hkey hdelta hvalue hseed
  have hflat := A.exec_flatten hd hcov _ u
    (RealSafety.source_safe p r kd value initial final useSeed storeFinal u hu hk hK hBC hs hvi hhi A.extent hb)
    (RealSafety.source_flatten_ok p r kd value initial final useSeed storeFinal) hu
  rw [he, Option.map_some] at hflat
  have hmem (rg : RegionName) (off : Nat) (ho : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at ho
      exact Nat.not_lt_zero _ ho
    exact A.flattenState_readMem hd v hr ho
  refine ⟨A.flattenState v, ?_, ⟨?_,?_,?_⟩, flatten_frame A p _ r storeFinal u v hf⟩
  · cases swapped
    · exact hflat
    · rw [← RealCandidate.flat_execution_eq]
      exact hflat
  · intro c e j hc ha
    rw [hmem r.history (RealOuterStep.historyAddress p (RealInit.launch u) c e.val j.val)
      (hb.loop.history c hc e j ha)]
    exact ho.history c e j hc ha
  · intro t j ha
    rw [hmem _ _ (hb.loop.inner.output t j ha)]
    exact ho.corrected t j ha
  · intro hon e j ha
    rw [hmem _ _ (hb.final hon e j ha)]
    exact ho.final hon e j ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealFlat

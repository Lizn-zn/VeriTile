import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateSafety

/-! Both complete H kernels in disjoint finite flat allocations. The explicit
zero-unspecified-value premise is retained from the region theorem. Active
history/final outputs match the independent closed form, and every other whole
tagged cell is preserved. Bounds and separation are on actual source accesses. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHFlat
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
    (hasSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealHStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.final (stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val)

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealHStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final)
    (hhi : RealHLoop.HistoryNoCollision p (launch u))
    (hfi : storeFinal = Bool.true → RealHStore.NoCollision p (launch u) (RealHFinal.base p (launch u)) p.V 1)
    (hk : ∀ off, u.readMem r.key off = a.key off)
    (hv : ∀ off, u.readMem r.value off = a.value off)
    (hseed : ∀ off, u.readMem r.seed off = a.seed off)
    (hb : RealHSafety.Bounds p (launch u) r hasSeed storeFinal A.extent) :
    ∃ v, exec (A.flattenKernel (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧ Outputs A p (launch u) r a hasSeed storeFinal v ∧
      BlockState.WriteWithin (footprint A p (launch u) r storeFinal) (A.flattenState u) v := by
  obtain ⟨v,he,ho,hf⟩ := RealHExecution.executes p r dot seed final hasSeed storeFinal swapped a u
    hu hs hfsep hhi hfi hk hv hseed
  have hflat := A.exec_flatten hd hcov _ u
    (RealHSafety.source_safe p r dot seed final hasSeed storeFinal swapped u hu hs hhi A.extent hb)
    (RealHSafety.source_flatten_ok p r dot seed final hasSeed storeFinal swapped) hu
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
    have hoff : historyAddress p (launch u) c (keyIndex p (launch u) e.val) (valueIndex p (launch u) j.val) < A.extent r.history := by
      simpa [historyAddress,keyIndex,valueIndex,Nat.mul_one] using (hb.loop c hc).history e j ha
    rw [hmem _ _ hoff]
    exact ho.history c e j hc ha
  · intro hon e j ha
    have hoff : stateAddress p (launch u) (keyIndex p (launch u) e.val) (valueIndex p (launch u) j.val) < A.extent r.final := by
      simpa only [stateAddress,keyIndex,valueIndex,Nat.mul_one] using hb.final hon e j ha
    rw [hmem _ _ hoff]
    exact ho.final hon e j ha

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHFlat

/-! Explicit IO for pre-update history and optional final state. Input arrays
come only from initial memory. Successful full execution, independent outputs
and preservation of all other tagged cells are conclusions of Implements. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealHIO
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

structure Outputs (p : Layout) where
  history : Nat → Fin p.BK → Fin p.BV → ℝ
  final : Fin p.BK → Fin p.BV → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  history : RegionName
  final : RegionName
  storeFinal : Bool
  historyWindow : Launch → Nat → Fin p.BK → Fin p.BV → Nat
  finalWindow : Launch → Fin p.BK → Fin p.BV → Nat
  stateMask : Launch → Fin p.BK → Fin p.BV → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  (cell.1 = io.history ∧ ∃ c e j, c < p.NT ∧ io.stateMask l e j ∧ cell.2 = io.historyWindow l c e j) ∨
  (io.storeFinal = Bool.true ∧ cell.1 = io.final ∧ ∃ e j, io.stateMask l e j ∧ cell.2 = io.finalWindow l e j)

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- Zero unspecified lanes are explicit: the source does not request zero
padding. Disjoint allocation intervals and actual active-address bounds are
also required. No prepared carry, history or final output is supplied. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Outputs p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ c e j, c < p.NT → io.stateMask (launch s) e j →
        v.readMem A.flat (A.addr io.history (io.historyWindow (launch s) c e j)) =
          (f (launch s) (io.read s)).history c e j) ∧
      (io.storeFinal = Bool.true → ∀ e j, io.stateMask (launch s) e j →
        v.readMem A.flat (A.addr io.final (io.finalWindow (launch s) e j)) =
          (f (launch s) (io.read s)).final e j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) :
    (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgorithm? =
      .ok (source p r dot seed final hasSeed storeFinal swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealHSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool) : IO p where
  kernel := (source p r dot seed final hasSeed storeFinal swapped).eraseDType
  projection := projection p r dot seed final hasSeed storeFinal swapped
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  finalWindow := fun l e j => stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)
  stateMask := fun l e j => RealHStore.active p l e.val j.val

def expected (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : Outputs p :=
  { history := fun c e j => closed p l a hasSeed c (keyIndex p l e.val) (valueIndex p l j.val)
    final := fun e j => closed p l a hasSeed p.NT (keyIndex p l e.val) (valueIndex p l j.val) }

theorem implements (p : Layout) (r : Regions) (dot : Storage)
    (seed final : FloatDType) (hasSeed storeFinal swapped : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    sourceIO p r dot seed final hasSeed storeFinal swapped ⊨ expected p hasSeed := by
  intro A hd hcov s hu hv hb
  obtain ⟨v,he,ho,hf⟩ := RealHFlat.source_executes A hd hcov p r dot seed final
    hasSeed storeFinal swapped ⟨s.readMem r.key,s.readMem r.value,s.readMem r.seed⟩ s
    hu hs hfsep hv.1 hv.2 (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hb
  refine ⟨v,he,ho.history,ho.final,?_⟩
  intro region off hn
  apply hf region off
  simpa only [IO.writes,IO.regionWrites,sourceIO,RealHFlat.footprint,RealHExecution.footprint,
    WriteFootprint.union,RealHLoop.footprint,RealHLoop.address,RealHStep.historyBase,
    RealHFinal.footprint,RealHStore.footprint,RealHStore.address,RealHFinal.base,
    historyAddress,stateAddress,Nat.mul_one] using hn

/-- Connect an independently presented recurrence to the complete execution
contract by induction, retaining both history and optional final outputs. -/
theorem implements_recurrence (p : Layout) (r : Regions) (dot : Storage)
    (seedType finalType : FloatDType) (hasSeed storeFinal swapped : Bool)
    (f : Launch → Inputs → Nat → Nat → Nat → ℝ)
    (hzero : ∀ l a k v, f l a 0 k v = seed p l a hasSeed k v)
    (hstep : ∀ l a c k v, f l a (c+1) k v = carryFactor p l c * f l a c k v + contribution p l a c k v)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    sourceIO p r dot seedType finalType hasSeed storeFinal swapped ⊨
      (fun l a => { history := fun c e j => f l a c (keyIndex p l e.val) (valueIndex p l j.val)
                    final := fun e j => f l a p.NT (keyIndex p l e.val) (valueIndex p l j.val) }) := by
  have hclosed : ∀ l a c k v, f l a c k v = closed p l a hasSeed c k v := by
    intro l a c k v
    induction c with
    | zero => exact hzero l a k v
    | succ c ih => rw [hstep,ih,closed_step]
  simpa only [hclosed,expected] using implements p r dot seedType finalType hasSeed storeFinal swapped hs hfsep

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealHIO

#guard_msgs (drop info) in
#auditModuleAxioms

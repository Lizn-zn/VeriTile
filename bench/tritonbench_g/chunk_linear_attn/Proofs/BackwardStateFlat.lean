import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardStateSafety
import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardStateExecution

/-! Complete iteration and loop access safety. Register addresses follow the
actual execution, including the history store before K/V loads. The previously
proved loop invariant supplies every reached state; no whole-program execution
or output equation is assumed as a precondition. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhSafety
open VeriTile.Triton
open RealHMath (Layout Launch)
open RealHProgram (times keyPtr valuePtr historyPtr load2)
open RealHTiles (StateValue stateTile)
open RealHPointers RealHAccessSafety RealDhMath RealDhProgram RealDhTiles
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat)
    (bounds : RegionBounds) : Prop where
  key : KeyBounds p l r.query c bounds
  value : ValueBounds p l r.incoming c bounds
  history : HistoryBounds p l r.history c bounds

def Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop :=
  ∀ c, c < p.NT → StepBounds p l r c bounds

theorem prelude_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (scale : ℝ)
    (f : StateValue p) (bounds : RegionBounds) (u : BlockState) (hb : StepBounds p l r c bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (RealDhStep.prelude p r scale) u := by
  unfold RealDhStep.prelude
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.query c u hih hik hit))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.incoming c _
    (by simp [hih]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (history_eval p l r.history c _
    (by simp [hih]) (by simp [hik]) (by simp [hiv]) (by simp [hit])))]
  refine ⟨by simp [Stmt.TraceSafe,historyPtr,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds (storeCarry p :: _) (RealDhStep.pointers p l r c u)
  erw [TraceProofs.cons_of_run_iff (RealDhStore.store_eval p l r.history (RealDhStep.historyBase p l c)
    p.historyKey 1 f "p_dh" _ (by simp [RealDhStep.pointers,hh]) (by simp [RealDhStep.pointers]; rfl))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.history (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV
      p.historyKey 1 (l.key*p.BK) (l.value*p.BV) "p_dh" "b_dh" bounds _ hb.history
      (by simp [RealDhStep.pointers,historyPointer])
  apply TraceProofs.assign_cons (load2_safe r.query (l.head*p.keyHead) p.K p.T p.BK p.BT
    p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT) "p_q" bounds _ hb.key
    (by simp [RealDhStore.scatter,RealDhStep.pointers,keyPointer]))
  intro key
  apply TraceProofs.assign_cons (by simp [scaledQuery,Op.SafeAt])
  intro scaled
  apply TraceProofs.assign_cons (load2_safe r.incoming (l.head*p.valueHead) p.T p.V p.BT p.BV
    p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV) "p_do" bounds _ hb.value
    (by simp [RealDhStore.scatter,RealDhStep.pointers,valuePointer]))
  intro value
  exact Stmt.TraceSafeList.nil_intro

theorem inner_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (c : Nat) (scale : ℝ) (swapped : Bool) (f : StateValue p) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealDhStep.Separate r)
    (bounds : RegionBounds) (hb : StepBounds p l r c bounds)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hk : ∀ off, u.readMem r.query off = a.query off)
    (hv : ∀ off, u.readMem r.incoming off = a.incoming off)
    (hh : u.regs .real [p.BK,p.BV] "b_dh" = some (stateTile p f)) :
    Stmt.TraceSafeList bounds (iterationBody p r scale swapped) u := by
  have he := RealDhStep.prefix_executes p l r a (fun _ _ => 0) c scale f u hs hu hik hiv hih hit hk hv hh
  have hp := prelude_safe p l r c scale f bounds u hb hik hiv hih hit hh
  rw [RealDhStep.iteration_parts]
  apply TraceProofs.append_safe _ u hp
  intro v hv
  rw [he] at hv
  cases hv
  apply TraceProofs.all_safe
  cases swapped <;> simp [Stmt.TraceSafe,update,RealDhProgram.carry,dot,Op.SafeAt,Op.SafeAt.eq_def]

theorem iteration_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealDhStep.Separate r) (bounds : RegionBounds) (hb : Bounds p l r bounds)
    (n : Nat) (u : BlockState) (hn : n < p.NT) (h : RealDhLoop.Holds initial p l r a scale n u) :
    Stmt.TraceSafeList bounds (RealDhProgram.iteration p r scale swapped)
      (u.setReg "j" .nat [] (Tile.scalar n)) := by
  rw [RealDhProgram.iteration,TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (RealDhLoop.reverseIndex_eval p n _ (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe,reverseIndex,Op.SafeAt],?_⟩
  apply inner_safe p l r a (p.NT-1-n) scale swapped (expected p l a scale n)
    ((u.setReg "j" .nat [] (Tile.scalar n)).setReg "i_t" .nat [] (Tile.scalar (p.NT-1-n)))
    (h.undef.trans hu) hs bounds (hb _ (by omega))
  · simpa using h.key
  · simpa using h.value
  · simpa using h.head
  · simp
  · simpa using h.inputQuery
  · simpa using h.inputGradient
  · simpa using h.carry

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (swapped : Bool) (hu : initial.undef = fun _ _ => 0)
    (hs : RealDhStep.Separate r) (hinj : RealDhLoop.HistoryNoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (u : BlockState)
    (h : RealDhLoop.Holds initial p l r a scale 0 u) :
    Stmt.TraceSafe bounds (RealDhLoop.statement p r scale swapped) u := by
  rw [RealDhLoop.statement,Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := RealDhLoop.Holds initial p l r a scale)
  · intro n v hn hv
    exact iteration_safe initial p l r a scale swapped hu hs bounds hb n v hn hv
  · intro n v w hn hv he
    obtain ⟨w',he',hw⟩ := RealDhLoop.iteration initial p l r a scale swapped hu hs hinj n v hn hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

theorem source_safe (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool)
    (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : RealDhStep.Separate r)
    (hinj : RealDhLoop.HistoryNoCollision p (launch u))
    (bounds : RegionBounds) (hb : Bounds p (launch u) r bounds) :
    (source p r query incoming out scale swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := launch u
  let a : Inputs := ⟨u.readMem r.query,u.readMem r.incoming⟩
  let start := RealDhInit.registers p u
  have hi := RealDhInit.invariant p r a scale u (fun _ => rfl) (fun _ => rfl)
  have hei := RealDhInit.executes p u
  have hloop := loop_safe u p l r a scale swapped hu hs hinj bounds hb start hi
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe _ u (by
    apply TraceProofs.all_safe
    simp [initialization,Stmt.TraceSafe,Op.SafeAt])
  intro v hv
  rw [hei] at hv
  cases hv
  change Stmt.TraceSafeList bounds [RealDhLoop.statement p r scale swapped] start
  exact Stmt.TraceSafeList.cons_intro hloop (fun _ _ => Stmt.TraceSafeList.nil_intro)

theorem source_flatten_ok (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r query incoming out scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [StmtList.FlattenOk,Stmt.FlattenOk,body,initialization,
    RealDhProgram.iteration,iterationBody,reverseIndex,storeCarry,RealDhProgram.carry,dot,update,scaledQuery,
    keyPtr,valuePtr,historyPtr,times,load2,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhSafety

/-! Both reverse-state kernels over finite disjoint flat allocations. Outputs
are sums over strictly later chunks; all other tagged memory cells are preserved.
Only actual active accesses need bounds, and padding remains an explicit premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhFlat
open VeriTile.Triton
open RealHMath (Layout Launch historyAddress keyIndex valueIndex)
open RealDhMath RealDhProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
     : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, RealDhLoop.footprint p l r (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions)
     (u v : BlockState)
    (hf : BlockState.WriteWithin (RealDhLoop.footprint p l r) u v) :
    BlockState.WriteWithin (footprint A p l r) (A.flattenState u) (A.flattenState v) := by
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
    (scale : ℝ) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT → RealDhStore.active p l e.val j.val →
    v.readMem A.flat (A.addr r.history (historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val))) =
      closed p l a scale c (keyIndex p l e.val) (valueIndex p l j.val)

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) (a : Inputs) (u : BlockState)
    (hu : u.undef = fun _ _ => 0) (hs : RealDhStep.Separate r)
    (hhi : RealDhLoop.HistoryNoCollision p (launch u))
    (hk : ∀ off, u.readMem r.query off = a.query off)
    (hv : ∀ off, u.readMem r.incoming off = a.incoming off)
    (hb : RealDhSafety.Bounds p (launch u) r A.extent) :
    ∃ v, exec (A.flattenKernel (source p r query incoming out scale swapped).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧ Outputs A p (launch u) r a scale v ∧
      BlockState.WriteWithin (footprint A p (launch u) r) (A.flattenState u) v := by
  obtain ⟨v,he,ho,hf⟩ := RealDhExecution.executes p r query incoming out scale swapped a u
    hu hs hhi hk hv
  have hflat := A.exec_flatten hd hcov _ u
    (RealDhSafety.source_safe p r query incoming out scale swapped u hu hs hhi A.extent hb)
    (RealDhSafety.source_flatten_ok p r query incoming out scale swapped) hu
  rw [he,Option.map_some] at hflat
  have hmem (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState v).readMem A.flat (A.addr rg off) = v.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      exact Nat.not_lt_zero _ hbound
    exact A.flattenState_readMem hd v hr hbound
  refine ⟨A.flattenState v,hflat,⟨?_⟩,flatten_frame A p _ r u v hf⟩
  · intro c e j hc ha
    have hoff : historyAddress p (launch u) c (keyIndex p (launch u) e.val) (valueIndex p (launch u) j.val) < A.extent r.history := by
      simpa [historyAddress,keyIndex,valueIndex,Nat.mul_one] using (hb c hc).history e j ha
    rw [hmem _ _ hoff]
    exact ho c e j hc ha

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhFlat

/-! IO for the reverse-state output. Inputs are read only from the initial
memory; success, the independent sum, and the whole-cell frame are conclusions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhIO
open VeriTile.Triton
open RealHMath (Layout Launch historyAddress keyIndex valueIndex)
open RealDhMath RealDhProgram
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

abbrev Values (p : Layout) := Nat → Fin p.BK → Fin p.BV → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  output : RegionName
  window : Launch → Nat → Fin p.BK → Fin p.BV → Nat
  mask : Launch → Fin p.BK → Fin p.BV → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = io.output ∧ ∃ c e j, c < p.NT ∧ io.mask l e j ∧ cell.2 = io.window l c e j

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- The source leaves padding unspecified. This real formula assumes the
explicit zero oracle, finite disjoint allocations and active-address bounds. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Values p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ v, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some v ∧
      (∀ c e j, c < p.NT → io.mask (launch s) e j →
        v.readMem A.flat (A.addr io.output (io.window (launch s) c e j)) =
          f (launch s) (io.read s) c e j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) v

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r query incoming out scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r query incoming out scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealDhSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) : IO p where
  kernel := (source p r query incoming out scale swapped).eraseDType
  projection := projection p r query incoming out scale swapped
  read := fun s => ⟨s.readMem r.query,s.readMem r.incoming⟩
  valid := fun s => RealDhLoop.HistoryNoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  output := r.history
  window := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  mask := fun l e j => RealDhStore.active p l e.val j.val

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : Values p :=
  fun c e j => closed p l a scale c (keyIndex p l e.val) (valueIndex p l j.val)

theorem implements (p : Layout) (r : Regions) (query incoming : Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) (hs : RealDhStep.Separate r) :
    sourceIO p r query incoming out scale swapped ⊨ expected p scale := by
  intro A hd hcov s hu hv hb
  obtain ⟨v,he,ho,hf⟩ := RealDhFlat.source_executes A hd hcov p r query incoming out scale swapped
    ⟨s.readMem r.query,s.readMem r.incoming⟩ s hu hs hv (fun _ => rfl) (fun _ => rfl) hb
  refine ⟨v,he,ho.history,?_⟩
  intro region off hn
  apply hf region off
  simpa only [IO.writes,IO.regionWrites,sourceIO,RealDhFlat.footprint,RealDhLoop.footprint,
    RealDhLoop.address,RealDhStep.historyBase,RealDhStore.address,historyAddress,Nat.mul_one] using hn

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealDhIO

#guard_msgs (drop info) in
#auditModuleAxioms

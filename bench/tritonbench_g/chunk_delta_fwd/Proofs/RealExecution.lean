import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealFinal

/-! Complete real execution of the typed original after precision erasure.
Both state switches, every outer/inner iteration and all three output buffers
are composed here. The formula exposes the source's rounded-up window width;
flat-memory bounds and IO packaging are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealExecution
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def source (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : ComputeKernel :=
  TypedSource.original kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal

private theorem legacy_body_parts (p : Layout) (r : Regions) (useSeed storeFinal : Bool) :
    (chunk_delta_rule_fwd_h_surface r.key r.value r.delta r.corrected r.history r.initial r.final
      p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
      p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType.toAlgKernel.body =
      cdfPrologue r.initial p.K p.V p.BK p.BV useSeed ++ [RealOuterLoop.statement p r] ++
        cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal := by
  obtain ⟨alg,hp⟩ := chunk_delta_rule_fwd_h_surface_toAlgorithm_supported
    r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal
  have hb := chunk_delta_fwd_body_split r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal
  simp only [ComputeKernel.toAlgKernel, hp] at hb
  simp only [ComputeKernel.eraseDType, hp, ComputeKernel.toAlgKernel_fromKernelBody, Kernel.eraseDType]
  rw [hb]
  simp only [cdfPrologue, cdfEpilogue, cdfOuterBody, chunkDeltaInnerBody, cdfStopOp,
    RealOuterLoop.statement, RealOuterStep.body, List.cons_append, List.nil_append, List.map_cons, List.map_nil,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]
  rfl

theorem body_parts (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) :
    (source p r kd value initial final useSeed storeFinal).eraseDType.toAlgKernel.body =
      cdfPrologue r.initial p.K p.V p.BK p.BV useSeed ++ [RealOuterLoop.statement p r] ++
        cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal := by
  unfold source
  rw [RealSource.original_body]
  exact legacy_body_parts p r useSeed storeFinal

def footprint (p : Layout) (l : Launch) (r : Regions) (storeFinal : Bool) : WriteFootprint :=
  WriteFootprint.union (RealOuterLoop.footprint p l r) (RealFinal.footprint p l r storeFinal)

structure Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (a₀ : Nat → ℝ)
    (useSeed storeFinal : Bool) (v : BlockState) : Prop where
  history : ∀ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT →
    RealStateStore.active p l e.val j.val →
    v.readMem r.history (RealOuterStep.historyAddress p l c e.val j.val) =
      RealWindow.expected p l a (RealInit.seed p l a₀) useSeed c e.val (valueIndex p l j.val)
  corrected : ∀ (t : Nat) (j : Fin p.BV), activeValue p l t j.val →
    v.readMem r.corrected (valueAddress p l t j.val) =
      RealMath.corrected (RealValues.arrays p l a (RealInit.seed p l a₀)) p.K
        (RealWindow.expected p l a (RealInit.seed p l a₀) useSeed (t / p.BT)) t (valueIndex p l j.val)
  final : storeFinal = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV), RealStateStore.active p l e.val j.val →
    v.readMem r.final (RealStateStore.address p l (RealInit.base p l) p.V e.val j.val) =
      RealWindow.expected p l a (RealInit.seed p l a₀) useSeed p.NT e.val (valueIndex p l j.val)

theorem executes (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) (a : Inputs) (a₀ : Nat → ℝ) (u : BlockState)
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
    (hseed : ∀ off, u.readMem r.initial off = a₀ off) :
    ∃ v, exec (source p r kd value initial final useSeed storeFinal).eraseDType.toAlgKernel u = some v ∧
      Outputs p (RealInit.launch u) r a a₀ useSeed storeFinal v ∧
      BlockState.WriteWithin (footprint p (RealInit.launch u) r storeFinal) u v := by
  let l := RealInit.launch u
  let start := RealInit.initial p r a₀ useSeed u
  have heinit := RealInit.executes p r a₀ useSeed u hu hseed
  obtain ⟨mid,heloop,hm⟩ := RealOuterLoop.executes u p l r a (RealInit.seed p l a₀) useSeed
    hu hk hK hBC hs hvi hhi start (RealInit.invariant p r a a₀ useSeed u hk hkey hdelta hvalue)
  obtain ⟨frozen,hreg,hrep⟩ := hm.state
  let v := RealFinal.result p l r storeFinal frozen mid
  have hefinal := RealFinal.executes p l r storeFinal frozen mid hm.key hm.value hm.head hreg
  have hf := RealFinal.frame p l r storeFinal frozen mid
  have preserved (rg : RegionName) (off : Nat) (hr : storeFinal = Bool.true → rg ≠ r.final) :
      mid.readMem rg off = v.readMem rg off := by
    apply RealOuterLoop.read_of_mem_eq
    exact hf rg off (fun h => hr h.1 h.2.1)
  refine ⟨v,?_,⟨?_,?_,?_⟩,?_⟩
  · rw [exec,body_parts]
    rw [List.append_assoc,stepStmts.append_some heinit]
    change stepStmts (RealOuterLoop.statement p r :: cdfEpilogue r.final p.K p.V p.BK p.BV storeFinal) start = some v
    rw [stepStmts.cons_some heloop]
    exact hefinal
  · intro c e j hc ha
    rw [← preserved r.history _ (fun h => (hfsep h).1)]
    exact hm.history c e j hc ha
  · intro t j ha
    rw [← preserved r.corrected _ (fun h => (hfsep h).2)]
    have hq : t / p.BT < p.NT := (Nat.div_lt_iff_lt_mul hBT).2 (Nat.lt_of_lt_of_le ha.1 hcover)
    have hlo : t / p.BT * p.BT ≤ t := Nat.div_mul_le_self t p.BT
    have hupper : t < (t / p.BT + 1) * p.BT := by
      have hm := Nat.mod_lt t hBT
      have hd := Nat.mod_add_div t p.BT
      rw [Nat.mul_comm p.BT] at hd
      simp only [Nat.add_mul,Nat.one_mul]
      omega
    exact hm.corrected (t / p.BT) t j hq ha hlo hupper
  · intro hon e j ha
    subst storeFinal
    rw [RealFinal.readback p l r frozen mid (hfi rfl) e j ha]
    apply hrep e j
    · simpa [RealStateStore.active,keyIndex,l,RealInit.launch,hk] using ha.1
    · exact ha.2
  · intro rg off hn
    exact (hm.frame rg off (fun h => hn (Or.inl h))).trans
      (hf rg off (fun h => hn (Or.inr h)))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealExecution

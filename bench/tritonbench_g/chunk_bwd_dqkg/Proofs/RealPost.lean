import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealLoop
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealSource

/-! Complete real execution and independent output readback for arbitrary V.
The full value loop supplies its four total sums to the existing post-loop
execution proof. The latter's summation parameter is V; the actual source
still uses the original BV and executes every required iteration.

This is the region-execution component. RealSafety, RealFrame and RealFlat
supply active access bounds, the full cell frame and flat-memory execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealPost
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

-- Initial-memory view only; all independent formulas still live in RealMath.
def inputs (s : BlockState) (q k v h g dout dh : RegionName) : Inputs where
  q := s.readMem q
  k := s.readMem k
  v := s.readMem v
  h := s.readMem h
  g := s.readMem g
  dout := s.readMem dout
  dh := s.readMem dh

section
variable (s : BlockState) (p : Layout) (q k v h g dout dh : RegionName)

-- V here is only the formula's summation extent; it does not change the
-- actual kernel's BV or the number of value-loop iterations.
theorem dq_acc (i : Fin p.BT) (e : Fin p.BK) :
    dqAcc s h dout p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.T p.K p.V p.BT p.BK p.V p.NT i.val e.val =
    (total p (RealLoop.launch s) (inputs s q k v h g dout dh)).dq i e := by
  simp only [dqAcc, total, dqTerm, Finset.sum_range, voElem, hElem,
    value, history, RealLoop.launch, inputs]

theorem dk_acc (i : Fin p.BT) (e : Fin p.BK) :
    dkAcc s v dh p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.T p.K p.V p.BT p.BK p.V p.NT i.val e.val =
    (total p (RealLoop.launch s) (inputs s q k v h g dout dh)).dk i e := by
  simp only [dkAcc, total, dkTerm, Finset.sum_range, voElem, hElem,
    value, history, RealLoop.launch, inputs]

theorem ds_acc (i j : Fin p.BT) :
    dsAcc s v dout p.s_v_h p.s_v_t p.T p.V p.BT p.V i.val j.val =
    (total p (RealLoop.launch s) (inputs s q k v h g dout dh)).ds i j := by
  simp only [dsAcc, total, dsTerm, Finset.sum_range, voElem,
    value, RealLoop.launch, inputs]

theorem dg_acc :
    dgLastAcc s h dh p.s_h_h p.s_h_t p.K p.V p.BK p.V p.NT =
    (total p (RealLoop.launch s) (inputs s q k v h g dout dh)).dgLast := by
  simp only [dgLastAcc, total, dgLastTerm, Finset.sum_range, hElem,
    history, RealLoop.launch, inputs]

theorem dq_formula (i : Fin p.BT) (e : Fin p.BK) :
    dqSpec s g k v h dout p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.V p.NT i.val e.val =
    RealMath.dq p (RealLoop.launch s) (inputs s q k v h g dout dh) i e := by
  simp only [dqSpec, dqGated, dsMasked, RealMath.dq, gatedDq, score, total,
    dqAcc, dsAcc, dqTerm, dsTerm, value, history, gate, queryKey,
    Finset.sum_range, qkElem, gElem, voElem, hElem, RealLoop.launch, inputs]

theorem dk_formula (i : Fin p.BT) (e : Fin p.BK) :
    dkSpec s g q v dout dh p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.V p.NT i.val e.val =
    RealMath.dk p (RealLoop.launch s) (inputs s q k v h g dout dh) i e := by
  simp only [dkSpec, dkGated, dsMasked, RealMath.dk, gatedDk, score, total,
    dkAcc, dsAcc, dkTerm, dsTerm, value, history, gate, lastGate, queryKey,
    Finset.sum_range, qkElem, gElem, gLastElem, voElem, hElem, RealLoop.launch, inputs]

theorem dg_formula (i : Fin p.BT) :
    dgSpec s g q k v h dout dh p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.V p.NT i.val =
    RealMath.dg p (RealLoop.launch s) (inputs s q k v h g dout dh) i := by
  simp only [dgSpec, dgLastFinal, RealMath.dg, lastCorrection, dqSpec, dkSpec,
    dqGated, dkGated, dsMasked, RealMath.dq, RealMath.dk, gatedDq, gatedDk, score, total,
    dqAcc, dkAcc, dsAcc, dgLastAcc, dqTerm, dkTerm, dsTerm, dgLastTerm,
    value, history, gate, lastGate, queryKey, Finset.sum_range, qkElem,
    gElem, gLastElem, voElem, hElem, RealLoop.launch, inputs]

/-- The original typed source executes every value block and stores the
independent whole-V formulas. Flat allocation bounds and frame are separate
obligations. The explicit oracle premise supplies zero for unspecified
checked-off block-pointer lanes. -/
theorem original_executes (qk valueStorage hist dhist : TypedSource.Storage)
    (dqOut dkOut dgOut : RegionName) (hu : s.undef = fun _ _ => 0) (hBV : 0 < p.BV)
    (hDqDk : dqOut ≠ dkOut) (hDqDg : dqOut ≠ dgOut) (hDkDg : dkOut ≠ dgOut)
    (hInj : ∀ i j : TileIndex [p.BT, p.BK],
      (s.pids 1 * p.BT + i.1.val < p.T ∧ s.pids 0 * p.BK + i.2.1.val < p.K) →
      (s.pids 1 * p.BT + j.1.val < p.T ∧ s.pids 0 * p.BK + j.2.1.val < p.K) →
      cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK i =
        cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK j → i = j) :
    ∃ sf, exec
      (RealSource.source qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
        p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
        p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel s = some sf ∧
      (∀ i : TileIndex [p.BT, p.BK],
        s.pids 1 * p.BT + i.1.val < p.T ∧ s.pids 0 * p.BK + i.2.1.val < p.K →
        sf.readMem dqOut (cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK i) =
          RealMath.dq p (RealLoop.launch s) (inputs s q k v h g dout dh) i.1 i.2.1) ∧
      (∀ i : TileIndex [p.BT, p.BK],
        s.pids 1 * p.BT + i.1.val < p.T ∧ s.pids 0 * p.BK + i.2.1.val < p.K →
        sf.readMem dkOut (cbdQkAddr s p.s_k_h p.s_k_t p.BT p.BK i) =
          RealMath.dk p (RealLoop.launch s) (inputs s q k v h g dout dh) i.1 i.2.1) ∧
      (∀ i : TileIndex [p.BT], s.pids 1 * p.BT + i.1.val < p.T →
        sf.readMem dgOut (cbdDgAddr s p.T p.BT i) =
          RealMath.dg p (RealLoop.launch s) (inputs s q k v h g dout dh) i.1) := by
  let regions : RealStep.Regions := ⟨v, h, dout, dh⟩
  let a := inputs s q k v h g dout dh
  obtain ⟨setup, t, hsetup, hprefix, hmem, hi⟩ :=
    RealLoop.prologue_and_loop s p regions g a hu
      (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  obtain ⟨setup', hsetup', _, _, _, hik, hit, hib, hnb, hoi, hg, hgl,
    _, _, _, _, hgz⟩ := cbdPreLoop_run g p.T p.BT p.BK s (congrFun hu g)
  have heq : setup' = setup := Option.some.inj (hsetup'.symm.trans hsetup)
  subst setup'
  obtain ⟨hdq, hdk, hds, hdgl⟩ := RealLoop.full_sums setup t p (RealLoop.launch s) a hBV hi
  have hmemory : t.mem = s.mem := hi.mem.trans hmem
  obtain ⟨sf, htail, hq, hk, hgrad⟩ := cbdPostLoop_run q k dqOut dkOut dgOut g v h dout dh s t
    p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
    p.T p.K p.V p.BT p.BK p.V p.NT
    ⟨congrFun (hi.undef.trans ((cbdPreLoop_undef g p.T p.BT p.BK s setup hsetup).trans hu)) q,
     congrFun (hi.undef.trans ((cbdPreLoop_undef g p.T p.BT p.BK s setup hsetup).trans hu)) k⟩
    (fun off => by simp only [BlockState.readMem, hmemory])
    (fun off => by simp only [BlockState.readMem, hmemory])
    hDqDk hDqDg hDkDg hInj
    ((hi.fixed .nat [] "i_k" (by decide)).trans hik)
    ((hi.fixed .nat [] "i_t" (by decide)).trans hit)
    ((hi.fixed .nat [] "i_bh" (by decide)).trans hib)
    ((hi.fixed .nat [] "n_bh" (by decide)).trans hnb)
    ((hi.fixed .nat [p.BT] "o_i" (by decide)).trans hoi)
    ((hi.fixed .real [p.BT] "b_g" (by decide)).trans hg)
    ((hi.fixed .real [] "b_g_last" (by decide)).trans hgl)
    ((hi.fixed .real [p.BT] "b_dg" (by decide)).trans hgz)
    (by simpa only [cbdDgLastTile, dg_acc s p q k v h g dout dh] using hdgl)
    (by simpa only [cbdDsTile, ds_acc s p q k v h g dout dh] using hds)
    (by simpa only [cbdDqTile, dq_acc s p q k v h g dout dh] using hdq)
    (by simpa only [cbdDkTile, dk_acc s p q k v h g dout dh] using hdk)
  refine ⟨sf, ?_, ?_, ?_, ?_⟩
  · rw [exec, RealSource.real_body_matches, cbd_body_eq]
    change stepStmts
      ((cbdPreLoop g p.T p.BT p.BK ++ [RealLoop.statement p regions]) ++
        cbdPostLoop q k dqOut dkOut dgOut p.s_k_h p.s_k_t p.scale p.T p.K p.BT p.BK) s = _
    rw [stepStmts.append_some hprefix]
    exact htail
  · intro i hs
    exact (hq i hs).trans (dq_formula s p q k v h g dout dh i.1 i.2.1)
  · intro i hs
    exact (hk i hs).trans (dk_formula s p q k v h g dout dh i.1 i.2.1)
  · intro i hs
    exact (hgrad i hs).trans (dg_formula s p q k v h g dout dh i.1)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealPost

import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealMemorySafety

/-! The complete source prologue is safe at the actual constructed pointers.
Only the selected Q load mode and the INIT=0 carry loads require read bounds.
Grouped-head metadata and pointer registers are established by source execution. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetyPrologue
open VeriTile.Triton RealAccess RealRegisterSafety RealMemorySafety RealPrologue

structure Bounds (BM BN BD : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds) : Prop where
  query : BlockBounds bounds
    (RealPointers.queryPointer BM BD p.layout (RealHeader.coordinates s p.heads p.kvHeads)
      (RealHeader.bases (RealHeader.coordinates s p.heads p.kvHeads) p.strides))
    (RealAccumulation.checks p.evenM)
  center : p.init = 0 → ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.M
  denominator : p.init = 0 → ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.L
  output : p.init = 0 → ∀ i, RealExecution.outputAddress BM BD p s i < bounds p.layout.Out

theorem prologue_safe (BM BN BD : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds)
    (hb : Bounds BM BN BD p s bounds) : Stmt.TraceSafeList bounds (code BM BN BD p) s := by
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  let mp := RealPointers.rowPointer BM p.layout.roundedContext p.layout.M c
  let lp := RealPointers.rowPointer BM p.layout.roundedContext p.layout.L c
  let op := RealPointers.outputPointer BM BD p.layout c b
  let qp := RealPointers.queryPointer BM BD p.layout c b
  obtain ⟨sh, eh, hc, hoff, _⟩ := RealHeader.executes s p.heads p.kvHeads p.strides
  obtain ⟨sp, ep, hp, _⟩ := RealPointers.executes sh BM BN BD p.layout c b hc hoff
  obtain ⟨sc, ec, _, _, _, fc⟩ := RealInit.carry_executes sp BM BD p.init mp lp op hp.center hp.denominator hp.output
  obtain ⟨ss, es, _, fs⟩ := RealInit.scale_executes sc p.scale
  have hqp : ss.regs .blockPtr [BM, BD] "Q_block_ptr" = some qp :=
    (fs.regs _ _ _ (by simp)).trans ((fc.regs _ _ _ (by simp)).trans hp.query)
  obtain ⟨sq, eq, _, _⟩ := RealInit.query_executes ss BM BD p.evenM qp hqp
  have hcarry : Stmt.TraceSafe bounds (RealInit.carryCode BM BD p.init) sp := by
    apply carry_safe bounds sp BM BD p.init mp lp op hp.center hp.denominator hp.output hb.center hb.denominator
    intro hi j _
    exact hb.output hi j
  simp only [code, List.append_assoc, List.singleton_append]
  apply append_of_run (header_safe p.heads p.kvHeads p.strides bounds s) eh
  apply append_of_run (pointers_safe BM BN BD p.layout bounds sh) ep
  apply cons_of_run hcarry ec
  apply append_of_run (scale_safe p.scale bounds sc) es
  apply cons_of_run (query_safe bounds ss BM BD p.evenM qp hqp hb.query) eq
  apply Stmt.TraceSafeList.cons_intro (window_safe _ _ _ _ _ _ _ _ bounds sq)
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetyPrologue

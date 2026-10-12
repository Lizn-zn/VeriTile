import bench.tritonbench_g.fast_rope_embedding.Proofs.Prelude
import bench.tritonbench_g.fast_rope_embedding.Proofs.StepSafety
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Safety
open VeriTile.Triton TypedSource Code
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

structure Bounds (u : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (bounds : RegionBounds) : Prop where
  cosine : ∀ j : Fin B,j.val<H/2 → Reference.coefficientAddress u.pid CR seq j.val<bounds C
  sine : ∀ j : Fin B,j.val<H/2 → Reference.coefficientAddress u.pid SR seq j.val<bounds S
  first : ∀ k,Reference.start (u.pids 1)≤k → k<Reference.stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → Reference.first u.pid QR H k j.val<bounds Q
  second : ∀ k,Reference.start (u.pids 1)≤k → k<Reference.stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → Reference.second u.pid QR H k j.val<bounds Q

theorem prelude_safe (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) (bounds : RegionBounds)
    (u : BlockState) (hb : Bounds u Q C S QR CR SR seq H heads B bounds) :
    Stmt.TraceSafeList bounds (prelude Q C S QR CR SR seq H heads B backwards) u := by
  cases backwards
  all_goals
    simp [prelude,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
      Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
      NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
      FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,IntegralDType.mod,Reference.coefficient,Reference.coefficientAddress,Reference.signedSine]


  all_goals exact ⟨hb.sine,hb.cosine⟩

theorem flatten_ok (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    (Code.original Q C S QR CR SR seq H heads B backwards).FlattenOk := by
  unfold Kernel.FlattenOk
  cases backwards
  all_goals simp [Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]

/-- Coefficient loads are required even for empty head groups. The loop only
requires allocations for the active halves of heads that actually execute. -/
theorem trace_safe (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) (bounds : RegionBounds)
    (u : BlockState) (hb : Bounds u Q C S QR CR SR seq H heads B bounds) :
    Stmt.TraceSafeList bounds (Code.original Q C S QR CR SR seq H heads B backwards).body u := by
  rw [Code.decomposition]
  apply TraceProofs.append_safe _ u (prelude_safe Q C S QR CR SR seq H heads B backwards bounds u hb)
  intro v hv
  obtain ⟨v',hv',hc,hs,he,hp,hm⟩ := Prelude.executes Q C S QR CR SR seq H heads B backwards u
  have hvv : v'=v := Option.some.inj (hv'.symm.trans hv)
  subst v'
  have hrange := TraceProofs.range_safe
    (P := fun k st => Reference.start (u.pids 1)≤k ∧
      Context st u.pid H B (Reference.coefficient u C CR seq H B)
        (fun j => Reference.signedSine backwards (Reference.coefficient u S SR seq H B j)))
    (bounds:=bounds) (index:="k") (stop:=Reference.stop (u.pids 1) heads) (stride:=1)
    (body:=body Q C S QR CR SR seq H heads B backwards)
    (fun k st hlt hst => StepSafety.trace_safe Q C S QR CR SR seq H heads B u.pid k backwards
      _ _ st bounds hst.2 (hb.first k hst.1 hlt) (hb.second k hst.1 hlt))
    (fun k st t _ hst hr => ⟨by omega,
      (Step.preserves_context Q C S QR CR SR seq H heads B u.pid k backwards _ _ st t hst.2 hr).1⟩)
    (Reference.start (u.pids 1)) v ⟨le_rfl,hc⟩
  apply Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simpa [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,hs,he] using hrange

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Safety

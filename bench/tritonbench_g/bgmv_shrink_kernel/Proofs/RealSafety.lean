import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealSetupSafety
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealTailSafety

/-! Access bounds for the complete signed source. Only actual active loads
and writes need allocation; the -1 branch requires its metadata cell only. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSafety
open VeriTile.Triton RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The input vector is read independently of the N mask. Weight loads use
both N and K masks; the same output bounds cover stores and atomic updates. -/
structure ActiveBounds (bounds : RegionBounds) (p : Parameters) (s : BlockState)
    (slot : Nat) : Prop where
  input : ∀ (c : Fin (RealLoop.numKIters p.K (p.BK * p.splits))) (e : Fin p.BK),
    RealLoop.kIdx s p.BK p.splits c.val e.val < p.K →
    s.pids 1 * p.inputBatchStride + RealLoop.kIdx s p.BK p.splits c.val e.val < bounds p.input
  weights : ∀ (c : Fin (RealLoop.numKIters p.K (p.BK * p.splits))) (n : Fin p.BN) (e : Fin p.BK),
    n.val < p.N → RealLoop.kIdx s p.BK p.splits c.val e.val < p.K →
    p.slotStride * slot + n.val * p.rowStride +
      RealLoop.kIdx s p.BK p.splits c.val e.val * p.colStride < bounds p.weights
  output : ∀ n : Fin p.BN, n.val < p.N → RealTail.offset p (s.pids 1) n.val < bounds p.output

theorem active_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState) (slot : Nat)
    (swapped : Bool)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = Int.ofNat slot)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices))
    (hS : 0 < p.BK * p.splits) (hb : ActiveBounds bounds p s slot) :
    Stmt.TraceSafeListR .triv bounds (p.body swapped) s := by
  unfold Parameters.body
  refine Stmt.TraceSafeListR.append_intro _ _
    (RealSetupSafety.prelude_safe bounds p s hmeta) (fun t ht => ?_)
  rw [stepStmtsR_triv, prelude_exec p s (Int.ofNat slot) hm] at ht
  obtain rfl := Option.some.inj ht
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  have hg : evalOp Parameters.guard (afterPrelude s (Int.ofNat slot)) = some (Tile.scalar Bool.true) := by
    have hn : Int.ofNat slot ≠ -1 := by change (slot : ℤ) ≠ -1; omega
    simpa only [decide_eq_true hn] using guard_eval (afterPrelude s (Int.ofNat slot))
      (Int.ofNat slot) (by simp [afterPrelude])
  simp only [Stmt.TraceSafeR, Parameters.guard, Op.SafeAtR.eq_def, true_and, evalOpR_triv]
  change (match evalOp Parameters.guard (afterPrelude s (Int.ofNat slot)) with
    | some v => if v.data PUnit.unit then Stmt.TraceSafeListR .triv bounds
        (p.setup ++ p.loop :: p.tail swapped) (afterPrelude s (Int.ofNat slot)) else True
    | none => True)
  rw [hg]
  simp only [Tile.scalar_data, if_true]
  refine Stmt.TraceSafeListR.append_intro _ _
    (RealSetupSafety.setup_safe bounds p _) (fun u hu => ?_)
  rw [stepStmtsR_triv, setup_exec p s slot] at hu
  obtain rfl := Option.some.inj hu
  refine Stmt.TraceSafeListR.cons_intro
    (RealAccessSafety.loop_safe bounds p s slot hS hb.input hb.weights) (fun st he => ?_)
  obtain ⟨counter, loopState, hloop, _, hinv⟩ := loop_exec p s slot hS
  rw [stepStmtR_triv, hloop] at he
  obtain rfl := Option.some.inj he
  exact RealTailSafety.tail_safe bounds p swapped loopState (s.pids 1) hinv.2.2.2.1 hb.output

theorem source_flatten_ok (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) :
    (p.source storage out_dtype swapped).eraseDType.toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk, Parameters.source_body]
  cases swapped
  all_goals simp [Parameters.body, Parameters.prelude, Parameters.setup, Parameters.guard,
    Parameters.weightPointer, Parameters.loop, Parameters.tail, Parameters.scale,
    shrinkLoopBody, shrinkTailCommon, shrinkStoreStmt, shrinkAtomicStmt,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def, MemAccess.FlattenOkC, MaskOpt.FlattenOkC]

theorem source_active_safe (bounds : RegionBounds) (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) (s : BlockState) (slot : Nat)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = Int.ofNat slot)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices))
    (hS : 0 < p.BK * p.splits) (hb : ActiveBounds bounds p s slot) :
    (p.source storage out_dtype swapped).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  simpa only [Kernel.TraceSafeR, Parameters.source_body] using active_safe bounds p s slot swapped hm hmeta hS hb

theorem source_sentinel_safe (bounds : RegionBounds) (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) (s : BlockState)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = -1)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices)) :
    (p.source storage out_dtype swapped).eraseDType.toAlgKernel.TraceSafeR .triv bounds s := by
  simpa only [Kernel.TraceSafeR, Parameters.source_body] using RealSetupSafety.sentinel_safe bounds p swapped s hm hmeta

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSafety

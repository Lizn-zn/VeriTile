import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealAccessSafety

/-! Bounds for both source output branches. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTailSafety
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem writes_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState) (batch : Nat)
    (hp : s.regs .ptr [p.BN] "c_ptr" = some ⟨fun i => (p.output, RealTail.offset p batch i.1.val)⟩)
    (hm : s.regs .bool [p.BN] "c_mask" = some ⟨fun i => decide (i.1.val < p.N)⟩)
    (hb : ∀ n : Fin p.BN, n.val < p.N → RealTail.offset p batch n.val < bounds p.output) :
    Stmt.TraceSafeR .triv bounds (shrinkStoreStmt p.BN) s ∧
      Stmt.TraceSafeR .triv bounds (shrinkAtomicStmt p.BN) s := by
  have ha : ((MemAccess.ptr (Op.ref .ptr [p.BN] "c_ptr")) : MemAccess .real [p.BN]).ActiveAddressSafeR .triv bounds s
      (((MaskOpt.mask (Op.ref .bool [p.BN] "c_mask")) : MaskOpt .real [p.BN]).ActiveR .triv s) := by
    simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR,
      evalOpR_triv, evalOp_ref]
    intro ptrs hptrs i hi
    rw [hp] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmask, hmi⟩ := hi
    rw [hm] at hmask
    obtain rfl := Option.some.inj hmask
    exact hb i.1 (of_decide_eq_true hmi)
  constructor <;> simpa only [shrinkStoreStmt, shrinkAtomicStmt, Stmt.TraceSafeR,
    MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR.eq_def, true_and] using ha

theorem tail_safe (bounds : RegionBounds) (p : Parameters) (swapped : Bool)
    (s : BlockState) (batch : Nat)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hb : ∀ n : Fin p.BN, n.val < p.N → RealTail.offset p batch n.val < bounds p.output) :
    Stmt.TraceSafeListR .triv bounds (p.tail swapped) s := by
  unfold Parameters.tail shrinkTailCommon
  simp only [List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append]
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun s1 hs1 => ?_)
  · cases swapped <;> simp [Parameters.scale, Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv hs1
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 hs2 => ?_)
  obtain ⟨v2, hv2, rfl⟩ := stepStmtR_assign_inv hs2
  simp only [evalOpR_triv, evalOp] at hv2
  obtain rfl := Option.some.inj hv2
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s3 hs3 => ?_)
  obtain ⟨v3, hv3, rfl⟩ := stepStmtR_assign_inv hs3
  rw [evalOpR_triv, cptr_tail_eval p.output p.outputBatchStride p.outputStride p.BN batch _
    (by simpa using hcb) (by simp)] at hv3
  obtain rfl := Option.some.inj hv3
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s4 hs4 => ?_)
  obtain ⟨v4, hv4, rfl⟩ := stepStmtR_assign_inv hs4
  rw [evalOpR_triv, cmask_eval p.N p.BN _ (by simp)] at hv4
  obtain rfl := Option.some.inj hv4
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  set st := (((s.setReg "accumulator" .real [p.BN] v1).setReg "offset_cn" .nat [p.BN]
    (Tile.vec (fun n => n.val))).setReg "c_ptr" .ptr [p.BN]
    ⟨fun i => (p.output, RealTail.offset p batch i.1.val)⟩).setReg "c_mask" .bool [p.BN]
    ⟨fun i => decide (i.1.val < p.N)⟩
  have hw := writes_safe bounds p st batch (by simp [st]) (by simp [st]) hb
  have hc : evalOpR .triv (Op.eq .nat .nil (.constNat p.splits) (.constNat 1)) st =
      some (Tile.scalar (decide (p.splits = 1))) := by
    simp only [evalOpR_triv, evalOp, Tile.cop, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rfl
  change Stmt.TraceSafeR .triv bounds
    (.ifThenElse (.eq .nat .nil (.constNat p.splits) (.constNat 1))
      [shrinkStoreStmt p.BN] [shrinkAtomicStmt p.BN]) st
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and, hc, Tile.scalar_data]
  split
  · exact Stmt.TraceSafeListR.cons_intro hw.1 (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  · exact Stmt.TraceSafeListR.cons_intro hw.2 (fun _ _ => Stmt.TraceSafeListR.nil_intro)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTailSafety

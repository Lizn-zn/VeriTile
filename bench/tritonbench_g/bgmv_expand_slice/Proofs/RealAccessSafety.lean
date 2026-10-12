import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealProgram

/-! Active output addresses are shared by the optional old-output load and
the store. Only lanes selected by the actual c_mask need bounds. Identity
rounding is used solely to reuse the library's flat-memory safety bridge. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealAccessSafety
open VeriTile.Triton RealBlock RealFoundation
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem output_address_safe (bounds : RegionBounds) (s : BlockState) (out : RegionName)
    (base stride snl BN c : Nat)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hb : ∀ i : Fin BN, c*BN+i.val < snl → base+(c*BN+i.val)*stride < bounds out) :
    (MemAccess.ptr (outputAddress stride BN) : MemAccess .real [BN]).ActiveAddressSafeR
      .triv bounds s ((MaskOpt.mask (.ref .bool [BN] "c_mask") : MaskOpt .real [BN]).ActiveR .triv s) := by
  simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR, evalOpR_triv]
  intro ptrs hptrs i hactive
  rw [show evalOp (outputAddress stride BN) s =
    some (Tile.vec (fun j => (out,base+(c*BN+j.val)*stride))) from
      cptrs_eval out base stride BN c s hp hn] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨mask, hmask, hi⟩ := hactive
  rw [evalOp_ref, hm] at hmask
  obtain rfl := Option.some.inj hmask
  exact hb i.1 (by simpa [Tile.vec] using hi)

theorem old_output_safe (bounds : RegionBounds) (s : BlockState) (out : RegionName)
    (base stride snl BN c : Nat)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hb : ∀ i : Fin BN, c*BN+i.val < snl → base+(c*BN+i.val)*stride < bounds out) :
    (oldOutputLoad stride BN).SafeAtR .triv bounds s := by
  simp only [oldOutputLoad, outputAddress, Op.SafeAtR.eq_def, true_and, and_true]
  exact output_address_safe bounds s out base stride snl BN c hp hn hm hb

theorem store_safe (bounds : RegionBounds) (s : BlockState) (out : RegionName)
    (base stride snl BN c : Nat)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hb : ∀ i : Fin BN, c*BN+i.val < snl → base+(c*BN+i.val)*stride < bounds out) :
    (outputStore stride BN).TraceSafeR .triv bounds s := by
  simp only [outputStore, outputAddress, Stmt.TraceSafeR, MemAccess.SafeAtR,
    MaskOpt.SafeAtR, Op.SafeAtR.eq_def, true_and, and_true]
  exact output_address_safe bounds s out base stride snl BN c hp hn hm hb

theorem update_safe (bounds : RegionBounds) (s : BlockState) (out : RegionName)
    (base stride snl BN c BK : Nat) (add swapped : Bool)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hb : ∀ i : Fin BN, c*BN+i.val < snl → base+(c*BN+i.val)*stride < bounds out) :
    (update stride BN BK add swapped).TraceSafeR .triv bounds s := by
  cases add <;> simp only [update, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    evalOpR_triv, evalOp, Tile.scalar_data, Bool.false_eq_true, if_false, if_true, true_and]
  · apply Stmt.TraceSafeListR.of_forall
    intro st hst t
    simp only [List.mem_singleton] at hst
    subst st
    simp [Stmt.TraceSafeR, reduction, Op.SafeAtR.eq_def]
  · refine Stmt.TraceSafeListR.cons_intro
      (by simpa only [Stmt.TraceSafeR] using
        old_output_safe bounds s out base stride snl BN c hp hn hm hb) (fun t _ => ?_)
    apply Stmt.TraceSafeListR.of_forall
    intro st hst u
    simp only [List.mem_singleton] at hst
    subst st
    cases swapped <;> simp [Stmt.TraceSafeR, accumulated, reduction, Op.SafeAtR.eq_def]

theorem tail_safe (bounds : RegionBounds) (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl c BK : Nat) (add swapped : Bool)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hs : evalOp (reduction BN BK) s = some ⟨fun i => some (sum i.1)⟩)
    (hb : ∀ i : Fin BN, c*BN+i.val < snl → base+(c*BN+i.val)*stride < bounds out) :
    Stmt.TraceSafeListR .triv bounds (tail stride BN BK add swapped) s := by
  unfold tail
  refine Stmt.TraceSafeListR.cons_intro
    (update_safe bounds s out base stride snl BN c BK add swapped hp hn hm hb) (fun t ht => ?_)
  rw [stepStmtR_triv, update_exec s sum out base stride snl c BK add swapped hp hn hm hs] at ht
  obtain rfl := Option.some.inj ht
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  apply store_safe bounds _ out base stride snl BN c
  · cases add <;> simpa [afterUpdate] using hp
  · cases add <;> simpa [afterUpdate] using hn
  · cases add <;> simpa [afterUpdate] using hm
  · exact hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealAccessSafety

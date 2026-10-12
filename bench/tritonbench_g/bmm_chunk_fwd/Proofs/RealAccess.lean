import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealLoads
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealStore

/-! Safety of the actual masked pointer operations. Active predicates are
obtained by evaluating source masks in the current state; bounds constrain
only reached addresses, independently of the numerical values. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealAccess
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

theorem masked_address_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape) (mask : Op .bool shape) (bits : Tile .bool shape)
    (hp : s.regs .ptr shape name = some ptrs) (hm : evalOp mask s = some bits)
    (hb : ∀ i, bits.data i = Bool.true → (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr shape name)).ActiveAddressSafe bounds s
      ((MaskOpt.mask (dtype := .real) mask).Active s) := by
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ptr he i hi
  rw [evalOp_ref, hp] at he
  cases he
  obtain ⟨ms, he, hi⟩ := hi
  rw [hm] at he
  cases he
  exact hb i hi

theorem a_safe (bounds : RegionBounds) (s : BlockState) (BM BK K length c : Nat)
    (rows : Fin BM → Nat) (ptrs : Tile .ptr [BM, BK])
    (hp : s.regs .ptr [BM, BK] "a_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c))
    (hb : ∀ (i : Fin BM) (k : Fin BK), rows i < length → k.val < K - c * BK →
      (ptrs.data (i,k,PUnit.unit)).2 < bounds (ptrs.data (i,k,PUnit.unit)).1) :
    (RealLoads.aLoad BM BK K).SafeAt bounds s := by
  rw [RealLoads.aLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealLoads.aMask, RealLoads.remaining, Op.SafeAt.eq_def], ?_⟩
  apply masked_address_safe bounds s "a_ptrs" ptrs _ _ hp
    (RealLoads.a_mask s BM BK K length c rows hr hk hl hc)
  rintro ⟨i,k,⟨⟩⟩ ha
  have h : rows i < length ∧ k.val < K - c * BK := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb i k h.1 h.2

theorem b_safe (bounds : RegionBounds) (s : BlockState) (BN BK K length c : Nat)
    (columns : Fin BN → Nat) (ptrs : Tile .ptr [BK, BN])
    (hp : s.regs .ptr [BK, BN] "b_ptrs" = some ptrs)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "k" = some (Tile.scalar c))
    (hb : ∀ (k : Fin BK) (j : Fin BN), k.val < K - c * BK → columns j < length →
      (ptrs.data (k,j,PUnit.unit)).2 < bounds (ptrs.data (k,j,PUnit.unit)).1) :
    (RealLoads.bLoad BN BK K).SafeAt bounds s := by
  rw [RealLoads.bLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealLoads.bMask, RealLoads.remaining, Op.SafeAt.eq_def], ?_⟩
  apply masked_address_safe bounds s "b_ptrs" ptrs _ _ hp
    (RealLoads.b_mask s BN BK K length c columns hn hk hl hc)
  rintro ⟨k,j,⟨⟩⟩ ha
  have h : k.val < K - c * BK ∧ columns j < length := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb k j h.1 h.2

theorem output_address_safe (bounds : RegionBounds) (s : BlockState) (BM BN chunk : Nat)
    (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] "out_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hb : ∀ (i : Fin BM) (j : Fin BN), rows i < chunk → columns j < chunk →
      (ptrs.data (i,j,PUnit.unit)).2 < bounds (ptrs.data (i,j,PUnit.unit)).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr [BM, BN] "out_ptrs")).ActiveAddressSafe bounds s
      ((MaskOpt.mask (dtype := .real) (RealStore.mask BM BN chunk)).Active s) := by
  apply masked_address_safe bounds s "out_ptrs" ptrs _ _ hp (RealStore.mask_eval s BM BN chunk rows columns hr hn)
  rintro ⟨i,j,⟨⟩⟩ ha
  have h : rows i < chunk ∧ columns j < chunk := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb i j h.1 h.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealAccess

import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealLoads
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealResidual

/-! Safety of the actual masked pointer operations. Active predicates are
obtained by evaluating source masks in the current state; bounds constrain
only reached addresses, independently of the numerical values. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealAccess
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

theorem dout_safe (bounds : RegionBounds) (s : BlockState) (BM BCS chunk length c : Nat)
    (rows : Fin BM → Nat) (ptrs : Tile .ptr [BM, BCS])
    (hp : s.regs .ptr [BM, BCS] "dout_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hk : s.regs .nat [BCS] "offs_cs" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "cs" = some (Tile.scalar c))
    (hb : ∀ (i : Fin BM) (k : Fin BCS), rows i < chunk → k.val < length - c * BCS →
      (ptrs.data (i,k,PUnit.unit)).2 < bounds (ptrs.data (i,k,PUnit.unit)).1) :
    (RealLoads.doutLoad BM BCS chunk).SafeAt bounds s := by
  rw [RealLoads.doutLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealLoads.doutMask, RealLoads.remaining, Op.SafeAt.eq_def], ?_⟩
  apply masked_address_safe bounds s "dout_ptrs" ptrs _ _ hp
    (RealLoads.dout_mask s BM BCS chunk length c rows hr hk hl hc)
  rintro ⟨i,k,⟨⟩⟩ ha
  have h : rows i < chunk ∧ k.val < length - c * BCS := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb i k h.1 h.2

theorem a_safe (bounds : RegionBounds) (s : BlockState) (BN BCS K length c : Nat)
    (columns : Fin BN → Nat) (ptrs : Tile .ptr [BCS, BN])
    (hp : s.regs .ptr [BCS, BN] "a_ptrs" = some ptrs)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hk : s.regs .nat [BCS] "offs_cs" = some (Tile.vec Fin.val))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hc : s.regs .nat [] "cs" = some (Tile.scalar c))
    (hb : ∀ (k : Fin BCS) (j : Fin BN), k.val < length - c * BCS → columns j < K →
      (ptrs.data (k,j,PUnit.unit)).2 < bounds (ptrs.data (k,j,PUnit.unit)).1) :
    (RealLoads.aLoad BN BCS K).SafeAt bounds s := by
  rw [RealLoads.aLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealLoads.aMask, RealLoads.remaining, Op.SafeAt.eq_def], ?_⟩
  apply masked_address_safe bounds s "a_ptrs" ptrs _ _ hp
    (RealLoads.a_mask s BN BCS K length c columns hn hk hl hc)
  rintro ⟨k,j,⟨⟩⟩ ha
  have h : k.val < length - c * BCS ∧ columns j < K := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb k j h.1 h.2

theorem output_address_safe (bounds : RegionBounds) (s : BlockState) (BM BN K length : Nat)
    (name : RegName) (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] name = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hb : ∀ (i : Fin BM) (j : Fin BN), rows i < length → columns j < K →
      (ptrs.data (i,j,PUnit.unit)).2 < bounds (ptrs.data (i,j,PUnit.unit)).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr [BM, BN] name)).ActiveAddressSafe bounds s
      ((MaskOpt.mask (dtype := .real) (RealResidual.mask BM BN K)).Active s) := by
  apply masked_address_safe bounds s name ptrs _ _ hp (RealResidual.mask_eval s BM BN K length rows columns hr hn hl)
  rintro ⟨i,j,⟨⟩⟩ ha
  have h : rows i < length ∧ columns j < K := by simpa only [Tile.mat, decide_eq_true_eq] using ha
  exact hb i j h.1 h.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealAccess

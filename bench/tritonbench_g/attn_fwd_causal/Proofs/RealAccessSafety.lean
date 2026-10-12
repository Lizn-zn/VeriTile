import bench.tritonbench_g.attn_fwd_causal.Proofs.RealInputPins
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealAccess

/-! Bounds for each actual memory operation, with the source masks retained.
In particular, a padded K/V tile still performs an unmasked key-scale read.
These local lemmas are composed along execution in the whole-program safety
proof; they do not assume safe execution as an input contract. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealAccessSafety
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 800000

structure Windows (c : Config) (Out : RegionName) (BM BN BD stage : Nat)
    (bounds : RegionBounds) : Prop where
  query : ∀ (row : Fin BM) (d : Fin BD), c.startM*BM+row.val < c.context → d.val < c.activeHead →
    c.base+(c.startM*BM+row.val)*c.strideM+d.val*c.strideD < bounds c.Q
  key : ∀ v ∈ RealFormula.allVisits c.startM BM BN stage, ∀ (j : Fin BN) (d : Fin BD),
    v.start+j.val < c.context → d.val < c.activeHead →
    c.base+d.val+j.val*c.strideKN+c.headDim*v.start < bounds c.K
  value : ∀ v ∈ RealFormula.allVisits c.startM BM BN stage, ∀ (j : Fin BN) (d : Fin BD),
    v.start+j.val < c.context → d.val < c.activeHead →
    c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*v.start < bounds c.V
  queryScale : c.queryScaleBase+c.startM < bounds c.QS
  keyScale : ∀ v ∈ RealFormula.allVisits c.startM BM BN stage,
    c.keyScaleBase+v.start/BN < bounds c.KS
  output : ∀ i, RealEpilogue.active c BM BD i = Bool.true →
    RealEpilogue.address c BM BD i < bounds Out

theorem masked_address_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape) (mask : Op .bool shape) (bits : Tile .bool shape)
    (hp : s.regs .ptr shape name = some ptrs) (hm : evalOp mask s = some bits)
    (hb : ∀ i, bits.data i = Bool.true → (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr shape name)).ActiveAddressSafe bounds s
      ((MaskOpt.mask (dtype := .real) mask).Active s) := by
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ptr hp' i hi
  rw [evalOp_ref, hp] at hp'
  cases hp'
  obtain ⟨ms, he, hi⟩ := hi
  rw [hm] at he
  cases he
  exact hb i hi

theorem masked_load_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape) (mask : Op .bool shape) (bits : Tile .bool shape)
    (hp : s.regs .ptr shape name = some ptrs) (hm : evalOp mask s = some bits)
    (hs : mask.SafeAt bounds s)
    (hb : ∀ i, bits.data i = Bool.true → (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (Op.load .real (.ptr (.ref .ptr shape name)) (.mask mask)).SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  exact ⟨by simp [Op.SafeAt.eq_def], hs,
    masked_address_safe bounds s name ptrs mask bits hp hm hb⟩

theorem query_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BD : Nat)
    (hp : s.regs .ptr [BM,BD] "Q_ptrs" = some (RealPointers.queryPointer c BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (RealPointers.rows BM (c.startM*BM)))
    (hb : ∀ (r : Fin BM) (d : Fin BD), c.startM*BM+r.val < c.context → d.val < c.activeHead →
      c.base+(c.startM*BM+r.val)*c.strideM+d.val*c.strideD < bounds c.Q) :
    (Op.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs"))
      (.mask (RealInit.queryMask BM BD c.context c.activeHead))).SafeAt bounds s := by
  apply masked_load_safe bounds s "Q_ptrs" _ _ _ hp
    (RealInit.eval_mask s BM BD c.startM c.context c.activeHead hr)
    (by simp [RealInit.queryMask, Op.SafeAt.eq_def])
  rintro ⟨r,d,⟨⟩⟩ hi
  have h : c.startM*BM+r.val < c.context ∧ d.val < c.activeHead := by
    simpa only [RealInit.mask, decide_eq_true_eq] using hi
  exact hb r d h.1 h.2

theorem key_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (name : RegName) (hp : s.regs .ptr [BD,BN] name = some (keyPointer c BD BN start))
    (hm : s.regs .bool [BD,BN] "k_mask" = some (keyMask BD BN c.context c.activeHead start))
    (hb : ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.context → d.val < c.activeHead →
      c.base+d.val+j.val*c.strideKN+c.headDim*start < bounds c.K) :
    (Op.load .real (.ptr (.ref .ptr [BD,BN] name)) (.mask (.ref .bool [BD,BN] "k_mask"))).SafeAt bounds s := by
  apply masked_load_safe bounds s name _ _ _ hp (by simpa only [evalOp_ref] using hm)
    (by simp [Op.SafeAt.eq_def])
  rintro ⟨d,j,⟨⟩⟩ hi
  have h : start+j.val < c.context ∧ d.val < c.activeHead := by
    simpa only [keyMask, decide_eq_true_eq] using hi
  exact hb j d h.1 h.2

theorem value_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (name : RegName) (hp : s.regs .ptr [BN,BD] name = some (valuePointer c BN BD start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hb : ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.context → d.val < c.activeHead →
      c.base+j.val*c.strideM+d.val*c.strideD+c.headDim*start < bounds c.V) :
    Stmt.TraceSafe bounds (RealBody.loadValue name BN BD c.context c.activeHead) s := by
  rw [RealBody.loadValue, Stmt.TraceSafe]
  apply masked_load_safe bounds s name _ _ _ hp
    (eval_value_mask s BN BD c.context c.activeHead start hn hs)
    (by simp [RealBody.valueMask, Op.SafeAt.eq_def])
  rintro ⟨j,d,⟨⟩⟩ hi
  have h : start+j.val < c.context ∧ d.val < c.activeHead := by
    simpa only [valueMask, decide_eq_true_eq] using hi
  exact hb j d h.1 h.2

theorem query_scale_safe (bounds : RegionBounds) (s : BlockState) (c : Config)
    (hp : s.regs .ptr [] "Q_scale_ptr" = some (RealPointers.queryScalePointer c))
    (hb : c.queryScaleBase+c.startM < bounds c.QS) :
    (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none).SafeAt bounds s :=
  pointer_load_safe bounds s "Q_scale_ptr" _ hp (fun _ => hb)

theorem key_scale_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN start : Nat)
    (name : RegName) (hp : s.regs .ptr [] name = some (keyScalePointer c BN start))
    (hb : c.keyScaleBase+start/BN < bounds c.KS) :
    (Op.load .real (.ptr (.ref .ptr [] name)) .none).SafeAt bounds s :=
  pointer_load_safe bounds s name _ hp (fun _ => hb)

theorem output_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (Out : RegionName)
    (BM BD : Nat)
    (hp : s.regs .ptr [BM,BD] "O_block_ptr" = some (RealPointers.outputPointer c Out BM BD))
    (hr : s.regs .nat [BM] "offs_m" = some (RealPointers.rows BM (c.startM*BM)))
    (hb : ∀ i, RealEpilogue.active c BM BD i = Bool.true →
      RealEpilogue.address c BM BD i < bounds Out) :
    Stmt.TraceSafe bounds (RealSource.store BM BD c.context c.activeHead) s := by
  rw [RealSource.store, Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt, Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def],
    by simp [MaskOpt.SafeAt, RealInit.queryMask, Op.SafeAt.eq_def], ?_⟩
  exact masked_address_safe bounds s "O_block_ptr" _ _ _ hp
    (RealInit.eval_mask s BM BD c.startM c.context c.activeHead hr) hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealAccessSafety

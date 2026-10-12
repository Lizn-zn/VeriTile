import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSource
import VeriTile.Triton.Memory.TraceProofs

/-! Bounds for the output consumer's actual active accesses. Every K block has
separate Q/G/H windows; V, A and O use the source postlude's windows. No numerical
output or successful source execution is a caller premise. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSafety
open VeriTile.Triton RealOutput
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def MatrixBounds (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (bounds : RegionBounds) : Prop :=
  ∀ (i : Fin nr) (j : Fin nc), offR+i.val < rows ∧ offC+j.val < cols →
    base+(offR+i.val)*sr+(offC+j.val)*sc < bounds r

def pointer (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat) : Tile .blockPtr [nr,nc] :=
  ⟨fun _ => BlockPtr.mk r base [rows,cols] [nr,nc] [sr,sc] [offR,offC]⟩

theorem load_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (pointer r base rows cols nr nc sr sc offR offC)) :
    (Op.load .real (.blockPtr (.ref .blockPtr [nr,nc] name) [0,1]) .none .unspecified).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

theorem store_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name value : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (pointer r base rows cols nr nc sr sc offR offC)) :
    Stmt.TraceSafe bounds (.store .real [nr,nc] (.blockPtr (.ref .blockPtr [nr,nc] name) [0,1])
      (.ref .real [nr,nc] value) .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

structure StepBounds (s : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t T K V BT BK BV kb : Nat) (bounds : RegionBounds) : Prop where
  query : MatrixBounds q (s.pids 2*s_k_h) T K BT BK s_k_t 1 (s.pids 1*BT) (kb*BK) bounds
  gate : MatrixBounds g (s.pids 2*s_k_h) T K BT BK s_k_t 1 (s.pids 1*BT) (kb*BK) bounds
  history : MatrixBounds h (s.pids 2*s_h_h+s.pids 1*K*V) K V BK BV s_h_t 1 (kb*BK) (s.pids 0*BV) bounds

structure PostBounds (s : BlockState) (v o A : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) (bounds : RegionBounds) : Prop where
  value : MatrixBounds v (s.pids 2*s_v_h) T V BT BV s_v_t 1 (s.pids 1*BT) (s.pids 0*BV) bounds
  output : MatrixBounds o (s.pids 2*s_v_h) T V BT BV s_v_t 1 (s.pids 1*BT) (s.pids 0*BV) bounds
  attention : MatrixBounds A (s.pids 2*T*BT) T BT BT BT BT 1 (s.pids 1*BT) 0 bounds

structure Bounds (s : BlockState) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV : Nat) (bounds : RegionBounds) : Prop where
  loop : ∀ kb, kb < numKB K BK → StepBounds s q g h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV kb bounds
  post : PostBounds s v o A s_v_h s_v_t T V BT BV bounds

theorem step_safe (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV i : Nat)
    (bounds : RegionBounds) (hb : StepBounds s0 q g h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV i bounds)
    (t : BlockState)
    (hiv : t.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 0)))
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar (s0.pids 1)))
    (hibh : t.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2)))
    (hk : t.regs .nat [] "i_k" = some (Tile.scalar i)) :
    Stmt.TraceSafeList bounds (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV) t := by
  unfold cgfLoopBody
  -- 1-3. the three block pointers, rebuilt from `i_k`
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval q t _ _ _ [T, K] [BT, BK] [s_k_t, 1]
      (s0.pids 2 * s_k_h) (s0.pids 1 * BT) (i * BK)
      (cgf_mulConst_eval t "i_bh" (s0.pids 2) s_k_h hibh)
      (cgf_mulConst_eval t "i_t" (s0.pids 1) BT hit)
      (cgf_mulConst_eval t "i_k" i BK hk)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval g _ _ _ _ [T, K] [BT, BK] [s_k_t, 1]
      (s0.pids 2 * s_k_h) (s0.pids 1 * BT) (i * BK)
      (cgf_mulConst_eval _ "i_bh" (s0.pids 2) s_k_h (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (cgf_mulConst_eval _ "i_k" i BK (by simpa using hk))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (s0.pids 2 * s_h_h + s0.pids 1 * K * V) (i * BK) (s0.pids 0 * BV)
      (cgf_hBase_eval _ s_h_h K V (s0.pids 2) (s0.pids 1)
        (by simpa using hibh) (by simpa using hit))
      (cgf_mulConst_eval _ "i_k" i BK (by simpa using hk))
      (cgf_mulConst_eval _ "i_v" (s0.pids 0) BV (by simpa using hiv))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe q (s0.pids 2*s_k_h) T K BT BK s_k_t 1
    (s0.pids 1*BT) (i*BK) "p_q" bounds _ hb.query (by simp [pointer]))
  intro qval
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro scaled
  apply TraceProofs.assign_cons (load_safe g (s0.pids 2*s_k_h) T K BT BK s_k_t 1
    (s0.pids 1*BT) (i*BK) "p_g" bounds _ hb.gate (by simp [pointer]))
  intro gval
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro gated
  apply TraceProofs.assign_cons (load_safe h (s0.pids 2*s_h_h+s0.pids 1*K*V) K V BK BV s_h_t 1
    (i*BK) (s0.pids 0*BV) "p_h" bounds _ hb.history (by simp [pointer]))
  intro history
  apply Stmt.TraceSafeList.cons_intro ?_ (by intro _ _; exact Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt],?_⟩
  split
  · split
    · apply TraceProofs.all_safe
      simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
    · trivial
  · trivial

theorem post_safe (s0 : BlockState) (v o A : RegionName)
    (s_v_h s_v_t T V BT BV : Nat)
    (bounds : RegionBounds) (hb : PostBounds s0 v o A s_v_h s_v_t T V BT BV bounds)
    (t : BlockState)
    (hiv : t.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 0)))
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar (s0.pids 1)))
    (hibh : t.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2))) :
    Stmt.TraceSafeList bounds (cgfPostLoop v o A s_v_h s_v_t T V BT BV) t := by
  unfold cgfPostLoop
  -- 1-3. the three block pointers
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval v t _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval t "i_bh" (s0.pids 2) s_v_h hibh)
      (cgf_mulConst_eval t "i_t" (s0.pids 1) BT hit)
      (cgf_mulConst_eval t "i_v" (s0.pids 0) BV hiv)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval _ "i_bh" (s0.pids 2) s_v_h (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (cgf_mulConst_eval _ "i_v" (s0.pids 0) BV (by simpa using hiv))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval A _ _ _ _ [T, BT] [BT, BT] [BT, 1]
      (s0.pids 2 * T * BT) (s0.pids 1 * BT) 0
      (cgf_aBase_eval _ T BT (s0.pids 2) (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (evalOp_constNat 0 _)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe v (s0.pids 2*s_v_h) T V BT BV s_v_t 1
    (s0.pids 1*BT) (s0.pids 0*BV) "p_v" bounds _ hb.value (by simp [pointer]))
  intro value
  apply TraceProofs.assign_cons (load_safe A (s0.pids 2*T*BT) T BT BT BT BT 1
    (s0.pids 1*BT) 0 "p_A" bounds _ hb.attention (by simp [pointer]))
  intro attention
  apply TraceProofs.assign_cons (by simp [Op.SafeAt])
  intro masked
  apply TraceProofs.assign_cons (by simp [Op.SafeAt,Op.SafeAt.eq_def])
  intro output
  apply Stmt.TraceSafeList.cons_intro
    (store_safe o (s0.pids 2*s_v_h) T V BT BV s_v_t 1
      (s0.pids 1*BT) (s0.pids 0*BV) "p_o" "b_o" bounds _ hb.output (by simp [pointer]))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
    (bounds : RegionBounds)
    (hb : ∀ kb, kb < numKB K BK → StepBounds s0 q g h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV kb bounds)
    (t : BlockState)
    (hi : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV 0 t) :
    Stmt.TraceSafe bounds (.forRangeDyn "i_k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat K) (.constNat BK)) (.constNat 1)) (.constNat BK))
      (.constNat 1) (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)) t := by
  rw [Stmt.TraceSafe]
  simp only [Op.SafeAt,evalOp,Option.bind,Tile.scalar,Tile.bop_data,
    NumericDType.add,NumericDType.sub,NumericDType.div,Broadcast.leftIndex,Broadcast.rightIndex,true_and]
  apply TraceProofs.range_safe (P := cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)
  · intro i u hn hu
    obtain ⟨_,_,_,hiv,hit,hibh,_⟩ := hu
    apply step_safe s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i bounds (hb i hn)
    · simpa using hiv
    · simpa using hit
    · simpa using hibh
    · simp
  · intro i u w hn hu he
    obtain ⟨w',he',hw⟩ := cgfLoopBody_run s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i
      (u.setReg "i_k" .nat [] (Tile.scalar i)) (Nat.succ_le_of_lt hn) (by simp)
      (cgfInv_setReg_k s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i i u hu)
    rw [he] at he'
    cases he'
    exact hw
  · exact hi

theorem source_safe (s : BlockState) (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)
    (bounds : RegionBounds)
    (hb : Bounds s q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV bounds)
    (hu : s.undef = fun _ _ => 0) :
    (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨t0,hpre,hm,hp,hiv,hit,hibh,hms,hbo,hundef⟩ := cgfPreLoop_run s BT BV
  have hi : cgfInv s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV 0 t0 :=
    ⟨Nat.zero_le _,hm,hp,hiv,hit,hibh,hms,by rw [hbo,cgfAccTile_zero],hundef.trans hu⟩
  obtain ⟨t1,he,ht⟩ := cgfLoop_collapse s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV t0 hi
  rw [Kernel.TraceSafe,RealOutputSource.storage_erasure]
  change Stmt.TraceSafeList bounds
    (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body s
  rw [cgf_body_eq,List.append_assoc]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [cgfPreLoop,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro u heq
    rw [hpre] at heq
    cases heq
    change Stmt.TraceSafeList bounds (_ :: cgfPostLoop v o A s_v_h s_v_t T V BT BV) t0
    rw [TraceProofs.cons_of_run_iff he]
    refine ⟨loop_safe s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV bounds hb.loop t0 hi,?_⟩
    obtain ⟨_,_,_,iv,it,ih,_⟩ := ht
    exact post_safe s v o A s_v_h s_v_t T V BT BV bounds hb.post t1 iv it ih

theorem source_flatten_ok (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,RealOutputSource.storage_erasure]
  change StmtList.FlattenOk
    (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body
  rw [cgf_body_eq]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,cgfPreLoop,cgfLoopBody,cgfPostLoop,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSafety

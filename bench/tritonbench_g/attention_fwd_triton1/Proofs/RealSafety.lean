import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealCandidate
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety of the all-flag source. These obligations concern actual
load/store addresses; numeric output equalities are not safety premises. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealSafety
open VeriTile.Triton
open RealBranch
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1600000
set_option maxRecDepth 10000

private theorem checked_cons {bounds stmt rest s t}
    (hrun : stepStmt stmt s = some t) (hsafe : Stmt.TraceSafe bounds stmt s)
    (htail : Stmt.TraceSafeList bounds rest t) :
    Stmt.TraceSafeList bounds (stmt :: rest) s := by
  exact Stmt.TraceSafeList.cons_intro hsafe (fun u hu => by
    have heq := Option.some.inj (hrun.symm.trans hu)
    subst u
    exact htail)

private theorem safe_list (bounds : RegionBounds) (code : List Stmt)
    (hs : ∀ stmt ∈ code, ∀ st, Stmt.TraceSafe bounds stmt st) (s : BlockState) :
    Stmt.TraceSafeList bounds code s := by
  induction code generalizing s with
  | nil => exact Stmt.TraceSafeList.nil_intro
  | cons stmt rest ih =>
    refine Stmt.TraceSafeList.cons_intro (hs stmt List.mem_cons_self s) (fun t _ => ?_)
    exact ih (fun stmt hm => hs stmt (List.mem_cons_of_mem _ hm)) t

private theorem if_safe (bounds : RegionBounds) (cond : Op .bool [])
    (lhs rhs : List Stmt) (s : BlockState) (hc : cond.SafeAt bounds s)
    (hl : Stmt.TraceSafeList bounds lhs s) (hr : Stmt.TraceSafeList bounds rhs s) :
    Stmt.TraceSafe bounds (.ifThenElse cond lhs rhs) s := by
  rw [Stmt.TraceSafe]
  refine ⟨hc, ?_⟩
  cases evalOp cond s with
  | none => trivial
  | some val =>
    dsimp only
    split <;> assumption

theorem update_safe (bounds : RegionBounds) (BT BD : Nat) (IFCOND : Bool) (s : BlockState) :
    Stmt.TraceSafe bounds (update BT BD IFCOND) s := by
  have ha (st : BlockState) : Stmt.TraceSafeList bounds (accumulate BT BD) st := by
    apply safe_list
    intro stmt hm u
    simp only [accumulate, List.mem_cons, List.not_mem_nil, or_false] at hm
    rcases hm with rfl | rfl <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def]
  apply if_safe bounds _ _ _ s (by simp [Op.SafeAt.eq_def]) ?_ (ha s)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  apply if_safe bounds _ _ _ s (by simp [Op.SafeAt.eq_def]) ?_ (ha s)
  apply safe_list
  intro stmt hm u
  simp only [List.mem_singleton] at hm
  subst stmt
  simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

private theorem if_preserves (cond : Op .bool []) (lhs rhs : List Stmt)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (s t : BlockState)
    (hl : ∀ st, stepStmts lhs s = some st → st.regs dtype shape name = s.regs dtype shape name)
    (hr : ∀ st, stepStmts rhs s = some st → st.regs dtype shape name = s.regs dtype shape name)
    (h : stepStmt (.ifThenElse cond lhs rhs) s = some t) :
    t.regs dtype shape name = s.regs dtype shape name := by
  simp only [stepStmt] at h
  cases hc : evalOp cond s with
  | none => simp [hc] at h
  | some val =>
    simp only [hc, bind, Option.bind_some] at h
    split at h
    · exact hl t h
    · exact hr t h

theorem update_pointer (BT BD : Nat) (IFCOND : Bool) (s t : BlockState)
    (h : stepStmt (update BT BD IFCOND) s = some t) :
    t.regs .blockPtr [BT, BD] "p_o" = s.regs .blockPtr [BT, BD] "p_o" := by
  have ha (s t : BlockState) (h : stepStmts (accumulate BT BD) s = some t) :
      t.regs .blockPtr [BT, BD] "p_o" = s.regs .blockPtr [BT, BD] "p_o" := by
    obtain ⟨u, hu, htail⟩ := TraceProofs.cons_inv h
    obtain ⟨v, _, rfl⟩ := TraceProofs.assign_inv hu
    obtain ⟨u', hu', htail⟩ := TraceProofs.cons_inv htail
    obtain ⟨v', _, rfl⟩ := TraceProofs.assign_inv hu'
    simp only [stepStmts.nil, Option.some.injEq] at htail
    subst t
    simp
  apply if_preserves _ _ _ _ _ _ s t ?_ (ha s) h
  intro st hst
  obtain ⟨u, hu, htail⟩ := TraceProofs.cons_inv hst
  simp only [stepStmts.nil, Option.some.injEq] at htail
  subst st
  apply if_preserves _ _ _ _ _ _ s u ?_ (ha s) hu
  intro st hst
  obtain ⟨v, hv, htail⟩ := TraceProofs.cons_inv hst
  obtain ⟨val, _, rfl⟩ := TraceProofs.assign_inv hv
  simp only [stepStmts.nil, Option.some.injEq] at htail
  subst st
  simp

private theorem load_safe (bounds : RegionBounds) (st : BlockState) (shape : TileShape)
    (ptr : RegName) (ptrs : Tile .blockPtr shape)
    (hp : st.regs .blockPtr shape ptr = some ptrs)
    (hb : ∀ i, BlockPtr.address (ptrs.data i) (TileShape.indexToList shape i) < bounds (ptrs.data i).region) :
    (Op.load .real (.blockPtr (.ref .blockPtr shape ptr) []) .none .unspecified).SafeAt bounds st := by
  simp only [Op.SafeAt.eq_def, MaskOpt.Active, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe]
  refine ⟨trivial, trivial, by simp [Op.MemorySafe], ?_⟩
  intro ps hps i _
  rw [evalOp_ref, hp] at hps
  obtain rfl := Option.some.inj hps
  intro _
  exact hb i

private theorem store_safe (bounds : RegionBounds) (st : BlockState) (shape : TileShape)
    (ptr val : RegName) (ptrs : Tile .blockPtr shape)
    (hp : st.regs .blockPtr shape ptr = some ptrs)
    (hb : ∀ i, BlockPtr.address (ptrs.data i) (TileShape.indexToList shape i) < bounds (ptrs.data i).region) :
    Stmt.TraceSafe bounds (.store .real shape (.blockPtr (.ref .blockPtr shape ptr) [])
      (.ref .real shape val) .none) st := by
  simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, MemAccess.SafeAt, MaskOpt.SafeAt,
    MaskOpt.Active, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  refine ⟨trivial, trivial, trivial, by simp [Op.MemorySafe], ?_⟩
  intro ps hps i _
  rw [evalOp_ref, hp] at hps
  obtain rfl := Option.some.inj hps
  intro _
  exact hb i

private theorem save_safe (bounds : RegionBounds) (STORE : Bool) (st : BlockState)
    (shape : TileShape) (ptr val : RegName) (ptrs : Tile .blockPtr shape)
    (hp : st.regs .blockPtr shape ptr = some ptrs)
    (hb : STORE = Bool.true → ∀ i,
      BlockPtr.address (ptrs.data i) (TileShape.indexToList shape i) < bounds (ptrs.data i).region) :
    Stmt.TraceSafe bounds (.ifThen (.constBool STORE)
      [.store .real shape (.blockPtr (.ref .blockPtr shape ptr) [])
        (.ref .real shape val) .none]) st := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
  cases STORE with
  | «false» => simp [evalOp]
  | «true» =>
    simp only [evalOp, Tile.scalar, ↓reduceIte]
    exact Stmt.TraceSafeList.cons_intro (store_safe bounds st shape ptr val ptrs hp (hb rfl))
      (fun _ _ => Stmt.TraceSafeList.nil_intro)

/-- Follow every actual access in a chunk. The optional H bounds are
required only when STORE is enabled. Dot operands and branch results do
not impose any additional memory-access conditions. -/
theorem body_safe (bounds : RegionBounds)
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) (s sin : BlockState) (c : Nat)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar c))
    (hbh : sin.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c))
    (hbQ : ∀ (i : Fin BT) (j : Fin BD), s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds Q)
    (hbK : ∀ (i : Fin BD) (j : Fin BT), s.pids 0 * s_qh + i.val + (c * BT + j.val) * BD < bounds K)
    (hbV : ∀ (i : Fin BT) (j : Fin BD), s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds V)
    (hbO : ∀ (i : Fin BT) (j : Fin BD), s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds O)
    (hbH : STORE = Bool.true → ∀ (i j : Fin BD),
      s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val < bounds H) :
    Stmt.TraceSafeList bounds
      (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND) sin := by
  unfold aft1LoopBodyG
  -- p_q, p_k, p_v, p_h, p_o
  refine checked_cons (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval Q sin _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval sin "i_bh" (s.pids 0) s_qh hibh)
      (aft1_mulConst_eval sin "i" c BT hi) (by simp))) (by
      rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
      intro off ho
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
      rcases ho with rfl | rfl <;> simp [Op.SafeAt.eq_def]) ?_
  refine checked_cons (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval K _ _ _ _ [BD, NT * BT] [BD, BT] [1, BD]
      (s.pids 0 * s_qh) 0 (c * BT)
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh])) (by simp)
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])))) (by
      rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
      intro off ho
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
      rcases ho with rfl | rfl <;> simp [Op.SafeAt.eq_def]) ?_
  refine checked_cons (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval V _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp))) (by
      rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
      intro off ho
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
      rcases ho with rfl | rfl <;> simp [Op.SafeAt.eq_def]) ?_
  refine checked_cons (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval H _ _ _ _ [NT * BD, BD] [BD, BD] [s_ht, 1]
      (s.pids 0 * s_hh) (c * BD) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_hh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BD (by simp [hi])) (by simp))) (by
      rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
      intro off ho
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
      rcases ho with rfl | rfl <;> simp [Op.SafeAt.eq_def]) ?_
  refine checked_cons (stepStmt_assign_eq_some
    (aft1_makeBlockPtr_2d_eval O _ _ _ _ [NT * BT, BD] [BT, BD] [BD, 1]
      (s.pids 0 * s_qh) (c * BT) 0
      (aft1_mulConst_eval _ "i_bh" (s.pids 0) s_qh (by simp [hibh]))
      (aft1_mulConst_eval _ "i" c BT (by simp [hi])) (by simp))) (by
      rw [Stmt.TraceSafe, Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt.eq_def], ?_⟩
      intro off ho
      simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
      rcases ho with rfl | rfl <;> simp [Op.SafeAt.eq_def]) ?_
  refine checked_cons (RealStore.save_executes STORE _ H "p_h" "b_h"
    (NT * BD) BD BD BD (s.pids 0 * s_hh) (c * BD) s_ht
    (aft1RecStateG s K V s_qh BT BD c) (by simp) (by simpa [aft1BhTileG] using hbh)) ?_ ?_
  · apply save_safe bounds STORE _ [BD, BD] "p_h" "b_h"
      ⟨fun _ => BlockPtr.mk H (s.pids 0 * s_hh) [NT * BD, BD] [BD, BD] [s_ht, 1] [c * BD, 0]⟩ (by simp)
    intro hs i
    simpa using hbH hs i.1 i.2.1
  -- Only address registers are needed below; loaded numeric values stay abstract.
  refine TraceProofs.assign_cons ?_ (fun q => ?_)
  · apply load_safe bounds _ [BT, BD] "p_q"
      ⟨fun _ => BlockPtr.mk Q (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1] [c * BT, 0]⟩ (by simp)
    intro i
    simpa using hbQ i.1 i.2.1
  refine TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def]) (fun scaled => ?_)
  refine TraceProofs.assign_cons ?_ (fun k => ?_)
  · apply load_safe bounds _ [BD, BT] "p_k"
      ⟨fun _ => BlockPtr.mk K (s.pids 0 * s_qh) [BD, NT * BT] [BD, BT] [1, BD] [0, c * BT]⟩ (by simp)
    intro i
    simpa using hbK i.1 i.2.1
  refine TraceProofs.assign_cons ?_ (fun v => ?_)
  · apply load_safe bounds _ [BT, BD] "p_v"
      ⟨fun _ => BlockPtr.mk V (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1] [c * BT, 0]⟩ (by simp)
    intro i
    simpa using hbV i.1 i.2.1
  refine TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def]) (fun scores => ?_)
  refine TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def]) (fun output => ?_)
  change Stmt.TraceSafeList bounds [update BT BD IFCOND, _] _
  refine Stmt.TraceSafeList.cons_intro (update_safe bounds BT BD IFCOND _) (fun t ht => ?_)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  refine store_safe bounds t [BT, BD] "p_o" "b_o"
    ⟨fun _ => BlockPtr.mk O (s.pids 0 * s_qh) [NT * BT, BD] [BT, BD] [BD, 1] [c * BT, 0]⟩ ?_ ?_
  · rw [update_pointer BT BD IFCOND _ t ht]
    simp
  · intro i
    simpa using hbO i.1 i.2.1

private theorem append_safe (bounds : RegionBounds) (before after : List Stmt)
    (s : BlockState) (hbefore : Stmt.TraceSafeList bounds before s)
    (hafter : ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t) :
    Stmt.TraceSafeList bounds (before ++ after) s := by
  induction before generalizing s with
  | nil => exact hafter s stepStmts.nil
  | cons stmt rest ih =>
    rw [Stmt.TraceSafeList] at hbefore
    refine Stmt.TraceSafeList.cons_intro hbefore.1 (fun t ht => ?_)
    exact ih t (by simpa only [ht] using hbefore.2)
      (fun u hu => hafter u ((stepStmts.cons_some ht).trans hu))

private theorem range_safe (bounds : RegionBounds) (index : RegName) (stop stride : Nat)
    (body : List Stmt) (P : Nat → BlockState → Prop)
    (hstep : ∀ c s, c < stop → P c s →
      Stmt.TraceSafeList bounds body (s.setReg index .nat [] (Tile.scalar c)) ∧
      ∃ t, stepStmts body (s.setReg index .nat [] (Tile.scalar c)) = some t ∧ P (c + stride) t) :
    ∀ cur s, P cur s → Stmt.forRangeTraceSafe bounds index cur stop stride body s
  | cur, s, h => by
    rw [Stmt.forRangeTraceSafe]
    split
    · trivial
    · split
      · obtain ⟨hsafe, t, hrun, hinv⟩ := hstep cur s ‹cur < stop› h
        refine ⟨hsafe, ?_⟩
        rw [hrun]
        exact range_safe bounds index stop stride body P hstep (cur + stride) t hinv
      · trivial
  termination_by cur _ _ => stop - cur
  decreasing_by omega

private theorem stop_eval (BT NT : Nat) (hBT : 0 < BT) (s : BlockState) :
    evalOp (aft1StopOpG BT NT) s = some (Tile.scalar NT) := by
  simp only [aft1StopOpG, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  intro z
  simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.div, NumericDType.sub, NumericDType.add]
  show (NT * BT + BT - 1) / BT = NT
  have heq : NT * BT + BT - 1 = BT * NT + (BT - 1) := by
    rw [Nat.mul_comm NT BT]
    omega
  rw [heq, Nat.mul_add_div hBT]
  have : (BT - 1) / BT = 0 := Nat.div_eq_of_lt (by omega)
  omega

/-- Address safety of the complete original source, with precisely the
additional H write window needed when STORE is enabled. -/
theorem original_safe (bounds : RegionBounds)
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) (s : BlockState) (hBT : 0 < BT) (hstride : BD ≤ s_ht)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O)
    (hbQ : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds Q)
    (hbK : ∀ c, c < NT → ∀ (i : Fin BD) (j : Fin BT),
      s.pids 0 * s_qh + i.val + (c * BT + j.val) * BD < bounds K)
    (hbV : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds V)
    (hbO : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds O)
    (hbH : STORE = Bool.true → ∀ c, c < NT → ∀ (i j : Fin BD),
      s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val < bounds H) :
    Kernel.TraceSafe bounds (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s := by
  rw [RealProjection.original_matches_transcription]
  simp only [Kernel.TraceSafe, attention_fwd_triton1_body_splitG]
  obtain ⟨s0, hpre, hinv⟩ := RealLoop.prologue Q K V H O s_qh s_hh s_ht scale BT BD STORE s
  apply append_safe bounds (aft1PrologueG BD) _ s
  · apply safe_list
    intro stmt hm u
    simp only [aft1PrologueG, List.mem_cons, List.not_mem_nil, or_false] at hm
    rcases hm with rfl | rfl <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def]
  · intro t ht
    have heq := Option.some.inj (hpre.symm.trans ht)
    subst t
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
    rw [Stmt.TraceSafe]
    refine ⟨by simp [Op.SafeAt.eq_def], by simp [aft1StopOpG, Op.SafeAt.eq_def],
      by simp [Op.SafeAt.eq_def], ?_⟩
    rw [evalOp_constNat, stop_eval BT NT hBT s0, evalOp_constNat]
    change Stmt.forRangeTraceSafe bounds "i" 0 NT 1 _ s0
    apply range_safe bounds "i" NT 1 _
      (RealLoop.Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s) ?_ 0 s0 hinv
    intro c st hc h
    refine ⟨?_, ?_⟩
    · apply body_safe bounds Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND
        s (st.setReg "i" .nat [] (Tile.scalar c)) c
        (by simpa using h.head) (by simp) (by simpa using h.carry)
        (hbQ c hc) (hbK c hc) (hbV c hc) (hbO c hc)
      exact fun hs => hbH hs c hc
    · exact RealLoop.invariant_step Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND
        s hQO hKO hVO hQH hKH hVH hHO hstride c st h

theorem candidate_safe (bounds : RegionBounds)
    (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat)
    (STORE IFCOND : Bool) (s : BlockState) (hBT : 0 < BT) (hstride : BD ≤ s_ht)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O)
    (hbQ : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds Q)
    (hbK : ∀ c, c < NT → ∀ (i : Fin BD) (j : Fin BT),
      s.pids 0 * s_qh + i.val + (c * BT + j.val) * BD < bounds K)
    (hbV : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds V)
    (hbO : ∀ c, c < NT → ∀ (i : Fin BT) (j : Fin BD),
      s.pids 0 * s_qh + (c * BT + i.val) * BD + j.val < bounds O)
    (hbH : STORE = Bool.true → ∀ c, c < NT → ∀ (i j : Fin BD),
      s.pids 0 * s_hh + (c * BD + i.val) * s_ht + j.val < bounds H) :
    Kernel.TraceSafe bounds (TypedSource.scalarLeft Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s := by
  apply (RealCandidate.same_safety bounds Q K V H O s_qh s_hh s_ht scale
    BT BD NT STORE IFCOND s).mpr
  exact original_safe bounds Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND s
    hBT hstride hQO hKO hVO hQH hKH hVH hHO hbQ hbK hbV hbO hbH

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealSafety

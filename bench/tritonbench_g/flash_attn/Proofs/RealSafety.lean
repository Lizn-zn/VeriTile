import bench.tritonbench_g.flash_attn.Proofs.RealLoopInvariant
import bench.tritonbench_g.flash_attn.Proofs.RealPost
import VeriTile.Triton.Memory.TraceProofs

/-! Full trace safety from bounds on the exact read and write windows. The
read window includes causally discarded keys and the entire final KV tile. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

elab "normalize_flash_safety" : tactic => do
  let g ← Lean.Elab.Tactic.getMainGoal
  let ty ← g.getType
  let args := ty.getAppArgs
  let rec norm (fuel : Nat) (e : Lean.Expr) : Lean.MetaM Lean.Expr := do
    let e ← Lean.Meta.whnf e
    match fuel with
    | 0 => return e
    | fuel+1 =>
      if e.isAppOfArity ``List.cons 3 then
        let es := e.getAppArgs
        let tail ← norm fuel es[2]!
        return Lean.mkAppN e.getAppFn (es.set! 2 tail)
      else return e
  let body ← norm 40 args[1]!
  Lean.Elab.Tactic.replaceMainGoal [←g.change (Lean.mkAppN ty.getAppFn (args.set! 1 body))]

structure AddressBounds (p : Layout) (s : BlockState) (Q K V L O : RegionName)
    (bounds : RegionBounds) : Prop where
  reads : ∀ r o, readWindow p Q K V (s.pids 0) (s.pids 1) r o → o < bounds r
  writes : ∀ r o, writeWindow p L O (s.pids 0) (s.pids 1) r o → o < bounds r

private theorem before_pure (Q K V L O : RegionName) (p : Layout) (bounds : RegionBounds)
    (s : BlockState) :
    Stmt.TraceSafeList bounds ((before Q K V L O p).take 12) s := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  normalize_flash_safety
  apply TraceProofs.all_safe
  intro st hs t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [Stmt.TraceSafe,Op.SafeAt,List.forall_mem_cons,
    List.forall_mem_nil,and_self,and_true,true_and]
  all_goals simp

private theorem before_tail_pure (Q K V L O : RegionName) (p : Layout)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds ((before Q K V L O p).drop 13) s := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  normalize_flash_safety
  apply TraceProofs.all_safe
  intro st hs t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl
  all_goals simp only [Stmt.TraceSafe,eq_mpr_eq_cast,eq_mp_eq_cast,cast_eq,
    Op.SafeAt,and_self,and_true,true_and]

private def preload (Q K V L O : RegionName) (p : Layout) (s : BlockState) :
    {t : BlockState // stepStmts ((before Q K V L O p).take 12) s = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [before,algorithm,reduced,List.take_succ_cons,List.take_zero]
  simp (maxSteps := 300000) [stepStmts,stepStmt,evalOp,Option.bind,BlockState.setReg,Region.cast]
  try rfl

theorem before_safe (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (bounds : RegionBounds)
    (hb : ∀ i : Fin p.BLOCK_M, ∀ d : Fin p.DIM,
      qOffset p (s.pids 0) (s.pids 1) i.val d.val < bounds Q) :
    Stmt.TraceSafeList bounds (before Q K V L O p) s := by
  rw [←List.take_append_drop 12 (before Q K V L O p)]
  apply TraceProofs.append_safe
  · exact before_pure Q K V L O p bounds s
  · intro t ht
    have he : t = (preload Q K V L O p s).val :=
      Option.some.inj (ht.symm.trans (preload Q K V L O p s).property)
    subst t
    have tail := before_tail_pure Q K V L O p bounds
    simp only [before,algorithm,reduced,List.take_succ_cons,List.take_zero,
      List.drop_succ_cons,List.drop_zero] at tail ⊢
    apply Stmt.TraceSafeList.cons_intro
    · simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,Op.MemorySafe.eq_def,MaskOpt.Active]
      simp [evalOp,preload,Region.cast,-Tile.scalar_eta]
      simpa [TileShape.indexToList,BlockPtr.address_2d_row_offset,qOffset,base,row,
        Tile.scalar,Tile.bop,NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex] using hb
    · intro u _
      exact tail u

theorem body_tail_safe (p : Layout) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds ((body p).drop 2) s := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  normalize_flash_safety
  apply TraceProofs.all_safe
  intro st hs t
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe,Stmt.TraceSafeList,Op.SafeAt.eq_def,evalOp]
  all_goals split <;> simp

theorem body_safe (p : Layout) (s : BlockState) (bounds : RegionBounds)
    (kp : Tile .blockPtr [p.DIM,p.BLOCK_N]) (vp : Tile .blockPtr [p.BLOCK_N,p.DIM])
    (hkp : s.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr" = some kp)
    (hvp : s.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr" = some vp)
    (hbK : ∀ i, (kp.data i).address (TileShape.indexToList _ i) < bounds (kp.data i).region)
    (hbV : ∀ i, (vp.data i).address (TileShape.indexToList _ i) < bounds (vp.data i).region) :
    Stmt.TraceSafeList bounds (body p) s := by
  rw [← List.take_append_drop 2 (body p)]
  apply TraceProofs.append_safe
  · simp only [body,algorithm,reduced,List.getElem?_cons_zero,List.getElem?_cons_succ,
      List.take_succ_cons,List.take_zero]
    simpa [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,Op.MemorySafe.eq_def,
      MaskOpt.Active,stepStmt,evalOp,Option.bind,Region.cast,hkp,hvp] using And.intro hbK hbV
  · intro t _
    exact body_tail_safe p bounds t

theorem loop_safe (Q K V L O : RegionName) (p : Layout) (s₀ t : BlockState)
    (bounds : RegionBounds) (hb : AddressBounds p s₀ Q K V L O bounds)
    (hBN : 0 < p.BLOCK_N) (h : PrefixInvariant p s₀ Q K V 0 t) :
    Stmt.TraceSafe bounds (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat p.BLOCK_N) (body p)) t := by
  simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, true_and, evalOp_ref, evalOp_constNat, h.lo, h.hi]
  let P := fun n st => ∃ c, n = c*p.BLOCK_N ∧ PrefixInvariant p s₀ Q K V c st
  apply TraceProofs.range_safe (P := P)
  · intro n st hn ⟨c,hc,hinv⟩
    subst n
    have hclt := (Geometry.tile_start_lt (hi p (s₀.pids 0)) p.BLOCK_N c hBN).mpr hn
    have hset := hinv.set_start_n p s₀ Q K V c st (c*p.BLOCK_N)
    apply body_safe p _ bounds _ _ hset.key hset.value
    · intro i
      change ((keyPointer p K (s₀.pids 1) (c*p.BLOCK_N)).data i).address
        (TileShape.indexToList _ i) < bounds K
      rw [keyPointer_address]
      exact hb.reads K _ (Or.inr (Or.inl ⟨rfl,
        ⟨c*p.BLOCK_N+i.2.1.val, Geometry.tile_lane_bound _ _ _ hclt i.2.1⟩,i.1,rfl⟩))
    · intro i
      change ((valuePointer p V (s₀.pids 1) (c*p.BLOCK_N)).data i).address
        (TileShape.indexToList _ i) < bounds V
      rw [valuePointer_address]
      exact hb.reads V _ (Or.inr (Or.inr ⟨rfl,
        ⟨c*p.BLOCK_N+i.1.val, Geometry.tile_lane_bound _ _ _ hclt i.1⟩,i.2.1,rfl⟩))
  · intro n st u hn ⟨c,hc,hinv⟩ hrun
    subst n
    have hset := hinv.set_start_n p s₀ Q K V c st (c*p.BLOCK_N)
    obtain ⟨v,hv,hnext⟩ := invariant_body_step p s₀ Q K V c _ hBN hset (by simp)
    have he : v = u := Option.some.inj (hv.symm.trans hrun)
    subst v
    exact ⟨c+1, by simp [Nat.add_mul], hnext⟩
  · exact ⟨0, by simp, h⟩

private theorem store_pointer_regs {d sh} (ptr : Op .ptr sh) (value : Op d sh)
    (s u : BlockState) (h : stepStmt (.store d sh (.ptr ptr) value .none) s = some u) :
    u.regs = s.regs := by
  simp only [stepStmt] at h
  cases hv : evalOp value s with
  | none => simp [hv] at h
  | some values =>
    cases hp : evalOp ptr s with
    | none => simp [hv,hp] at h
    | some ptrs =>
      simp only [hv,hp,Option.bind_eq_bind,Option.bind_some] at h
      cases h
      funext dtype shape name
      exact BlockState.foldl_writeMemTypedAt_masked_regs d _ _ _ (fun _ => Bool.true) _ s dtype shape name

theorem after_safe (Q K V L O : RegionName) (p : Layout) (s₀ t : BlockState)
    (bounds : RegionBounds) (c : Nat) (h : PrefixInvariant p s₀ Q K V c t)
    (hL : ∀ i : Fin p.BLOCK_M, lOffset p (s₀.pids 0) (s₀.pids 1) i.val < bounds L)
    (hO : ∀ i : TileIndex [p.BLOCK_M,p.DIM],
      oOffset p (s₀.pids 0) (s₀.pids 1) i.1.val i.2.1.val < bounds O) :
    Stmt.TraceSafeList bounds (after Q K V L O p) t := by
  normalize_flash_safety
  apply TraceProofs.assign_cons
  · simp only [Op.SafeAt.eq_def, and_true, true_and]
  · intro normalized
    apply TraceProofs.assign_eval_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
    · intro offsets he
      have hoff : offsets = Tile.vec (fun i => (L, lOffset p (s₀.pids 0) (s₀.pids 1) i.val)) := by
        simpa [-Tile.scalar_eta,evalOp,h.bshead,h.offm,Tile.bop,Tile.scalar,Tile.vec,
          NumericDType.add,NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex,
          lOffset,row,Tile.ptrAdd,Region.cast] using he.symm
      subst offsets
      apply Stmt.TraceSafeList.cons_intro
      · simp [Stmt.TraceSafe,Op.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn,MemAccess.SafeAt,MaskOpt.SafeAt,Op.MemorySafe,MaskOpt.Active,evalOp,hL,-Tile.scalar_eta]
      · intro u hu
        have hregs := store_pointer_regs _ _ _ _ hu
        apply TraceProofs.assign_eval_cons
        · simp [Op.SafeAt,List.forall_mem_cons,List.forall_mem_nil]
        · intro ptr hp
          simp [evalOp,hregs,h.startm,h.base,-Tile.scalar_eta] at hp
          subst ptr
          apply Stmt.TraceSafeList.cons_intro
          · simp [Stmt.TraceSafe,Op.SafeAt,MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,
              memAccessActiveAddressSafe,Op.MemorySafe,MaskOpt.Active,evalOp,-Tile.scalar_eta]
            simpa [TileShape.indexToList,BlockPtr.address_2d_row_offset,Tile.scalar,Tile.bop,
              NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex,oOffset,base,row] using hO
          · intro _ _
            exact Stmt.TraceSafeList.nil_intro

theorem algorithm_safe (Q K V L O : RegionName) (p : Layout) (s : BlockState)
    (bounds : RegionBounds) (hb : AddressBounds p s Q K V L O bounds)
    (hBN : 0 < p.BLOCK_N) :
    (algorithm Q K V L O p).TraceSafe bounds s := by
  change Stmt.TraceSafeList bounds (algorithm Q K V L O p).body s
  rw [decomposition]
  apply TraceProofs.append_safe
  · apply before_safe
    intro i d
    exact hb.reads Q _ (Or.inl ⟨rfl,i,d,rfl⟩)
  · intro initial he
    have hei : initial = (entry Q K V L O p s).val :=
      Option.some.inj (he.symm.trans (entry Q K V L O p s).property)
    subst initial
    have hinit := entry_invariant p s Q K V L O
    obtain ⟨t,ht,hinv⟩ := loop_invariant p s Q K V _ hBN hinit
    rw [TraceProofs.cons_of_run_iff ht]
    refine ⟨loop_safe Q K V L O p s _ bounds hb hBN hinit, ?_⟩
    apply after_safe Q K V L O p s t bounds _ hinv
    · intro i
      exact hb.writes L _ (Or.inl ⟨rfl,i,rfl⟩)
    · intro i
      exact hb.writes O _ (Or.inr ⟨rfl,i.1,i.2.1,rfl⟩)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution

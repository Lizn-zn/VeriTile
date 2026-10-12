import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealRegisterSafety

/-! Safety of the actual Q/K/V load choices, INIT carry loads and END stores.
Pointer registers are supplied by verified source execution. Only addresses of
accesses taken by the source flag branch require bounds. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealMemorySafety
open VeriTile.Triton RealAccess RealRegisterSafety

theorem load_choice_safe (bounds : RegionBounds) (s : BlockState) (shape : TileShape)
    (pointer target : RegName) (even : Nat) (ptrs : Tile .blockPtr shape)
    (hp : s.regs .blockPtr shape pointer = some ptrs)
    (hb : BlockBounds bounds ptrs (RealAccumulation.checks even)) :
    Stmt.TraceSafe bounds
      (.ifThenElse (.ne .nat .nil (.constNat even) (.constNat 0))
        [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) []) .none .unspecified)]
        [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) [0, 1]) .none)]) s := by
  rw [if_nonzero_safe]
  by_cases he : even ≠ 0
  · rw [if_pos he]
    apply Stmt.TraceSafeList.cons_intro
    · simpa only [Stmt.TraceSafe] using block_load_safe (padding := .unspecified) bounds s pointer ptrs [] hp
        (by simpa only [RealAccumulation.checks, if_pos he] using hb)
    · intro _ _; exact Stmt.TraceSafeList.nil_intro
  · rw [if_neg he]
    apply Stmt.TraceSafeList.cons_intro
    · simpa only [Stmt.TraceSafe] using block_load_safe bounds s pointer ptrs [0, 1] hp
        (by simpa only [RealAccumulation.checks, if_neg he] using hb)
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem carry_safe (bounds : RegionBounds) (s : BlockState) (BM BD init : Nat)
    (mp lp : Tile .ptr [BM]) (op : Tile .blockPtr [BM, BD])
    (hm : s.regs .ptr [BM] "m_ptrs" = some mp)
    (hl : s.regs .ptr [BM] "l_ptrs" = some lp)
    (ho : s.regs .blockPtr [BM, BD] "O_block_ptr" = some op)
    (hbM : init = 0 → ∀ i, (mp.data i).2 < bounds (mp.data i).1)
    (hbL : init = 0 → ∀ i, (lp.data i).2 < bounds (lp.data i).1)
    (hbO : init = 0 → BlockBounds bounds op []) :
    Stmt.TraceSafe bounds (RealInit.carryCode BM BD init) s := by
  rw [RealInit.carryCode, if_nonzero_safe]
  by_cases hi : init ≠ 0
  · rw [if_pos hi]
    apply list_safe
    simp [Op.SafeAt.eq_def]
  · have hz : init = 0 := by omega
    rw [if_neg hi]
    apply TraceProofs.assign_cons (pointer_load_safe bounds s "m_ptrs" mp hm (hbM hz))
    intro m
    apply TraceProofs.assign_cons (pointer_load_safe bounds _ "l_ptrs" lp
      (by simp [hl]) (hbL hz))
    intro l
    apply TraceProofs.assign_cons (block_load_safe (padding := .unspecified) bounds _ "O_block_ptr" op []
      (by simp [ho]) (hbO hz))
    intro _
    exact Stmt.TraceSafeList.nil_intro

theorem query_safe (bounds : RegionBounds) (s : BlockState) (BM BD even : Nat)
    (qp : Tile .blockPtr [BM, BD]) (hq : s.regs .blockPtr [BM, BD] "Q_block_ptr" = some qp)
    (hb : BlockBounds bounds qp (RealAccumulation.checks even)) :
    Stmt.TraceSafe bounds (RealInit.queryCode BM BD even) s :=
  load_choice_safe bounds s [BM, BD] "Q_block_ptr" "q" even qp hq hb

theorem load_key_safe (bounds : RegionBounds) (s : BlockState) (BD BN even : Nat)
    (kp : Tile .blockPtr [BD, BN]) (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hb : BlockBounds bounds kp (RealAccumulation.checks even)) :
    Stmt.TraceSafe bounds (RealBody.loadKey BD BN even) s :=
  load_choice_safe bounds s [BD, BN] "K_block_ptr" "k" even kp hk hb

theorem accumulate_safe (bounds : RegionBounds) (s : BlockState) (BM BN BD even : Nat)
    (vp : Tile .blockPtr [BN, BD]) (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp)
    (hb : BlockBounds bounds vp (RealAccumulation.checks even)) :
    Stmt.TraceSafeList bounds (RealBody.accumulate BM BN BD even) s := by
  apply Stmt.TraceSafeList.cons_intro (load_choice_safe bounds s [BN, BD] "V_block_ptr" "v" even vp hv hb)
  intro t _
  exact accumulate_tail_safe BM BN BD even bounds t

theorem end_branch_safe (bounds : RegionBounds) (s : BlockState) (BM BD finish : Nat) (reciprocal : Bool)
    (lp : Tile .ptr [BM]) (hl : s.regs .ptr [BM] "l_ptrs" = some lp)
    (hb : finish = 0 → ∀ i, (lp.data i).2 < bounds (lp.data i).1) :
    Stmt.TraceSafe bounds (RealEpilogue.endBranch BM BD finish reciprocal) s := by
  rw [RealEpilogue.endBranch, if_nonzero_safe]
  by_cases he : finish ≠ 0
  · rw [if_pos he]; exact normalize_safe BM BD reciprocal bounds s
  · rw [if_neg he]
    apply Stmt.TraceSafeList.cons_intro (pointer_store_safe bounds s "l_ptrs" "l_i" lp hl (hb (by omega)))
    intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem epilogue_safe (bounds : RegionBounds) (s : BlockState) (BM BD finish : Nat) (reciprocal : Bool)
    (c : RealLoopFormula.Carry BM BD) (M L : RegionName) (addr : TileIndex [BM] → Nat) (bp : BlockPtr)
    (hc : RealEpilogue.Registers c s)
    (hm : s.regs .ptr [BM] "m_ptrs" = some (⟨fun i => (M, addr i)⟩ : Tile .ptr [BM]))
    (hl : s.regs .ptr [BM] "l_ptrs" = some (⟨fun i => (L, addr i)⟩ : Tile .ptr [BM]))
    (ho : s.regs .blockPtr [BM, BD] "O_block_ptr" = some (⟨fun _ => bp⟩ : Tile .blockPtr [BM, BD]))
    (hbM : ∀ i, addr i < bounds M) (hbL : finish = 0 → ∀ i, addr i < bounds L)
    (hbO : ∀ i, RealEpilogue.blockAddress BM BD bp i < bounds bp.region) :
    Stmt.TraceSafeList bounds (RealEpilogue.code BM BD finish reciprocal) s := by
  let se := RealEpilogue.branchState s finish c L addr
  let sm := RealStores.writeTile se M addr (fun i => RealEpilogue.centerValue finish c i.1)
  have erun := RealEpilogue.branch_executes s finish reciprocal c L addr hc hl
  have hpM : se.regs .ptr [BM] "m_ptrs" = some (⟨fun i => (M, addr i)⟩ : Tile .ptr [BM]) :=
    (RealEpilogue.branch_other_register s finish c L addr _ _ _ (by simp)).trans hm
  have hregs := RealEpilogue.branch_registers s finish c L addr hc
  have em : stepStmt (RealEpilogue.centerStore BM) se = some sm := by
    simpa only [RealEpilogue.center_value] using RealStores.pointer_store se [BM] "m_ptrs" "m_i"
      M addr (RealEpilogue.centerTile (RealEpilogue.savedCarry finish c)) hpM hregs.center
  rw [RealEpilogue.code]
  apply cons_of_run (end_branch_safe bounds s BM BD finish reciprocal _ hl hbL) erun
  apply cons_of_run (pointer_store_safe bounds se "m_ptrs" "m_i" _ hpM hbM) em
  apply Stmt.TraceSafeList.cons_intro
  · apply block_store_safe bounds sm "O_block_ptr" "acc" (⟨fun _ => bp⟩ : Tile .blockPtr [BM, BD])
    · rw [RealStores.write_tile_regs]
      exact (RealEpilogue.branch_other_register s finish c L addr _ _ _ (by simp)).trans ho
    · intro i _; exact hbO i
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealMemorySafety

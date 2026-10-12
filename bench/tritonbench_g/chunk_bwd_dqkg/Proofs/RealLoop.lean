import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealStep

/-! Complete real value-axis loop, including multiple iterations and a partial
last block. The invariant keeps all four accumulators, the full typed memory,
unspecified-value function, launch coordinates and every untouched register.
RealSource connects the typed implementation, RealPost proves the three stored
formulas, and RealSafety/RealFrame/RealFlat complete the memory contract. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealLoop
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (a : Inputs)
    (count : Nat) (s : BlockState) : Prop where
  count_le : count ≤ blocks p
  mem : s.mem = initial.mem
  undef : s.undef = initial.undef
  pids : s.pids = initial.pids
  numPids : s.numPids = initial.numPids
  fixed : ∀ dtype shape name, name ∉ "i_v" :: RealStep.writes →
    s.regs dtype shape name = initial.regs dtype shape name
  dq : s.regs .real [p.BT, p.BK] "b_dq" =
    some (Tile.mat fun i j => some ((carries p l a count).dq i j))
  dk : s.regs .real [p.BT, p.BK] "b_dk" =
    some (Tile.mat fun i j => some ((carries p l a count).dk i j))
  ds : s.regs .real [p.BT, p.BT] "b_ds" =
    some (Tile.mat fun i j => some ((carries p l a count).ds i j))
  dg : s.regs .real [1] "b_dg_last" =
    some (Tile.vec fun _ => some (carries p l a count).dgLast)

theorem initial (s : BlockState) (p : Layout) (l : Launch) (a : Inputs)
    (hq : s.regs .real [p.BT, p.BK] "b_dq" = some (cbdZero [p.BT, p.BK]))
    (hk : s.regs .real [p.BT, p.BK] "b_dk" = some (cbdZero [p.BT, p.BK]))
    (hs : s.regs .real [p.BT, p.BT] "b_ds" = some (cbdZero [p.BT, p.BT]))
    (hg : s.regs .real [1] "b_dg_last" = some (cbdZero [1])) :
    Holds s p l a 0 s := by
  refine ⟨Nat.zero_le _, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_⟩
  · intros; rfl
  · simpa [carries, prefix_zero, cbdZero, Tile.mat] using hq
  · simpa [carries, prefix_zero, cbdZero, Tile.mat] using hk
  · simpa [carries, prefix_zero, cbdZero, Tile.mat] using hs
  · simpa [carries, prefix_zero, cbdZero, Tile.vec] using hg

theorem iteration (s₀ : BlockState) (p : Layout) (l : Launch) (r : RealStep.Regions)
    (a : Inputs)
    (hu : s₀.undef = fun _ _ => 0)
    (hmv : ∀ off, s₀.readMem r.v off = a.v off)
    (hmh : ∀ off, s₀.readMem r.h off = a.h off)
    (hmo : ∀ off, s₀.readMem r.dout off = a.dout off)
    (hmd : ∀ off, s₀.readMem r.dh off = a.dh off)
    (hik : s₀.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : s₀.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : s₀.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (c : Nat) (s : BlockState) (hc : c < blocks p) (h : Holds s₀ p l a c s) :
    ∃ t, stepStmts (RealStep.code p r) (s.setReg "i_v" .nat [] (Tile.scalar c)) = some t ∧
      Holds s₀ p l a (c + 1) t := by
  let entry := s.setReg "i_v" .nat [] (Tile.scalar c)
  have hkey : entry.regs .nat [] "i_k" = some (Tile.scalar l.key) := by
    simpa [entry, hik] using h.fixed .nat [] "i_k" (by decide)
  have htime : entry.regs .nat [] "i_t" = some (Tile.scalar l.time) := by
    simpa [entry, hit] using h.fixed .nat [] "i_t" (by decide)
  have hhead : entry.regs .nat [] "i_bh" = some (Tile.scalar l.head) := by
    simpa [entry, hib] using h.fixed .nat [] "i_bh" (by decide)
  have emem : entry.mem = s₀.mem := h.mem
  have ev (off : Nat) : entry.readMem r.v off = a.v off := by
    simpa only [BlockState.readMem, emem] using hmv off
  have eh (off : Nat) : entry.readMem r.h off = a.h off := by
    simpa only [BlockState.readMem, emem] using hmh off
  have eo (off : Nat) : entry.readMem r.dout off = a.dout off := by
    simpa only [BlockState.readMem, emem] using hmo off
  have ed (off : Nat) : entry.readMem r.dh off = a.dh off := by
    simpa only [BlockState.readMem, emem] using hmd off
  have he := RealStep.executes entry p l r a c (carries p l a c)
    (by exact h.undef.trans hu) ev eh eo ed hkey htime hhead (by simp [entry])
    (by simpa [entry] using h.dq) (by simpa [entry] using h.dk)
    (by simpa [entry] using h.ds) (by simpa [entry] using h.dg)
  refine ⟨_, he, Nat.succ_le_of_lt hc, h.mem, h.undef, h.pids, h.numPids, ?_, ?_, ?_, ?_, ?_⟩
  · intro dtype shape name hn
    have hn' := hn
    simp only [List.mem_cons, not_or] at hn'
    rw [RealStep.register_frame _ _ _ _ _ _ _ _ _ _ hn'.2]
    simpa [entry, BlockState.setReg, hn'.1] using h.fixed dtype shape name hn
  all_goals simp [RealStep.result, ← partial_succ]

def statement (p : Layout) (r : RealStep.Regions) : Stmt :=
  .forRangeDyn "i_v" (.constNat 0) (cbdStopOp p.V p.BV) (.constNat 1) (RealStep.code p r)

theorem stop_eval (s : BlockState) (p : Layout) :
    evalOp (cbdStopOp p.V p.BV) s = some (Tile.scalar (blocks p)) := by
  simp only [cbdStopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  intro z
  simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
    NumericDType.div, NumericDType.sub, NumericDType.add, blocks]

theorem executes (s₀ : BlockState) (p : Layout) (l : Launch) (r : RealStep.Regions)
    (a : Inputs)
    (hu : s₀.undef = fun _ _ => 0)
    (hmv : ∀ off, s₀.readMem r.v off = a.v off)
    (hmh : ∀ off, s₀.readMem r.h off = a.h off)
    (hmo : ∀ off, s₀.readMem r.dout off = a.dout off)
    (hmd : ∀ off, s₀.readMem r.dh off = a.dh off)
    (hik : s₀.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : s₀.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : s₀.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hq : s₀.regs .real [p.BT, p.BK] "b_dq" = some (cbdZero [p.BT, p.BK]))
    (hk : s₀.regs .real [p.BT, p.BK] "b_dk" = some (cbdZero [p.BT, p.BK]))
    (hs : s₀.regs .real [p.BT, p.BT] "b_ds" = some (cbdZero [p.BT, p.BT]))
    (hg : s₀.regs .real [1] "b_dg_last" = some (cbdZero [1])) :
    ∃ t, stepStmt (statement p r) s₀ = some t ∧ Holds s₀ p l a (blocks p) t := by
  have resolve : stepStmt (statement p r) s₀ =
      stepForRangeAux "i_v" 0 (blocks p) 1 (RealStep.code p r) s₀ := by
    simp only [statement, stepForRangeAux.forRangeDyn_unfold, stop_eval,
      evalOp_constNat, bind, Option.bind_some]
    rfl
  rw [resolve]
  obtain ⟨final, t, he, hf, ht⟩ := forRangeAux_inv
    (idx := "i_v") (stop := blocks p) (step := 1) (P := Holds s₀ p l a)
    (by decide) (fun c s hc h => iteration s₀ p l r a hu hmv hmh hmo hmd hik hit hib c s hc h)
    0 s₀ (initial s₀ p l a hq hk hs hg)
  have hfinal : final = blocks p := Nat.le_antisymm ht.count_le hf
  subst final
  exact ⟨t, he, ht⟩

/-- The four completed carries have the independent sums over all V values.
There is no `V ≤ BV` assumption and no requirement that V divide BV. -/
theorem full_sums (s₀ s : BlockState) (p : Layout) (l : Launch) (a : Inputs)
    (hBV : 0 < p.BV) (h : Holds s₀ p l a (blocks p) s) :
    s.regs .real [p.BT, p.BK] "b_dq" = some (Tile.mat fun i j => some ((total p l a).dq i j)) ∧
    s.regs .real [p.BT, p.BK] "b_dk" = some (Tile.mat fun i j => some ((total p l a).dk i j)) ∧
    s.regs .real [p.BT, p.BT] "b_ds" = some (Tile.mat fun i j => some ((total p l a).ds i j)) ∧
    s.regs .real [1] "b_dg_last" = some (Tile.vec fun _ => some (total p l a).dgLast) := by
  have hvalues := And.intro h.dq (And.intro h.dk (And.intro h.ds h.dg))
  simpa only [partial_complete p l a hBV] using hvalues

def launch (s : BlockState) : Launch := ⟨s.pids 0, s.pids 1, s.pids 2, s.numPids 2⟩

/-- The zero carries and grid registers are reached by executing the actual
prologue. They are not assumed as an already-executed whole loop. -/
theorem prologue_and_loop (s : BlockState) (p : Layout) (r : RealStep.Regions)
    (g : RegionName) (a : Inputs)
    (hu : s.undef = fun _ _ => 0)
    (hmv : ∀ off, s.readMem r.v off = a.v off)
    (hmh : ∀ off, s.readMem r.h off = a.h off)
    (hmo : ∀ off, s.readMem r.dout off = a.dout off)
    (hmd : ∀ off, s.readMem r.dh off = a.dh off) :
    ∃ setup final,
      stepStmts (cbdPreLoop g p.T p.BT p.BK) s = some setup ∧
      stepStmts (cbdPreLoop g p.T p.BT p.BK ++ [statement p r]) s = some final ∧
      setup.mem = s.mem ∧ Holds setup p (launch s) a (blocks p) final := by
  obtain ⟨setup, hsetup, _, _, hm, hik, hit, hib, _, _, _, _, hq, hk, hs, hg, _⟩ :=
    cbdPreLoop_run g p.T p.BT p.BK s (congrFun hu g)
  obtain ⟨final, hfinal, hi⟩ := executes setup p (launch s) r a
    ((cbdPreLoop_undef g p.T p.BT p.BK s setup hsetup).trans hu)
    (fun off => (hm r.v off).trans (hmv off))
    (fun off => (hm r.h off).trans (hmh off))
    (fun off => (hm r.dout off).trans (hmo off))
    (fun off => (hm r.dh off).trans (hmd off)) hik hit hib hq hk hs hg
  refine ⟨setup, final, hsetup, ?_, ?_, hi⟩
  · rw [stepStmts.append_some hsetup, stepStmts.cons_some hfinal, stepStmts.nil]
  · exact storeFree_stepStmts_mem _ _ _ (by simp [cbdPreLoop, storeFree]) hsetup

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealLoop

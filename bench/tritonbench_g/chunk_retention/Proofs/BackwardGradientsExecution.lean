import bench.tritonbench_g.chunk_retention.Proofs.BackwardGradientsProgram

/-! One complete value-block iteration, with actual loads followed by the
three carry updates and the dV store. Q, K and the upper causal score stay in
registers. Only inputs reloaded by later iterations need separation from dV. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVStep
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs)
open RealQKVProgram RealQKVTiles RealQKVPointers RealQKVAlgebra RealMatrix RealMatrixStore
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

structure Fixed (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (u : BlockState) : Prop where
  key : u.regs .nat [] "i_k" = some (Tile.scalar l.key)
  chunk : u.regs .nat [] "i_t" = some (Tile.scalar l.chunk)
  head : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  heads : u.regs .nat [] "n_bh" = some (Tile.scalar l.heads)
  queryWeights : u.regs .real [p.BT] "d_q" = some (queryWeights p l scale)
  keyWeights : u.regs .real [p.BT] "d_k" = some (keyWeights p l)
  scoreWeights : u.regs .real [p.BT,p.BT] "d_s" = some (tile (scoreWeights p l scale))
  query : u.regs .real [p.BK,p.BT] "b_q" = some (tile (q p l a))
  keys : u.regs .real [p.BT,p.BK] "b_k" = some (tile (k p l a))
  score : u.regs .real [p.BT,p.BT] "b_s" = some (tile (s p l a scale))

structure Carries (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState) : Prop where
  ds : u.regs .real [p.BT,p.BT] "b_ds" = some (tile (cs p l a n))
  dq : u.regs .real [p.BT,p.BK] "b_dq" = some (tile (cq p l a scale n))
  dk : u.regs .real [p.BT,p.BK] "b_dk" = some (tile (ck p l a n))

structure Reads (r : Regions) (a : Inputs) (u : BlockState) : Prop where
  value : ∀ off, u.readMem r.value off = a.value off
  incoming : ∀ off, u.readMem r.incoming off = a.incoming off
  history : ∀ off, u.readMem r.history off = a.history off
  gradient : ∀ off, u.readMem r.historyGradient off = a.historyGradient off

structure Separate (r : Regions) : Prop where
  value : r.value ≠ r.dv
  incoming : r.incoming ≠ r.dv
  history : r.history ≠ r.dv
  gradient : r.historyGradient ≠ r.dv

def valueWindow (p : Layout) (l : Launch) (r : Regions) (n : Nat) : Window p.BT p.BV :=
  ⟨r.dv,(l.key*l.heads+l.head)*p.valueHead,p.T,p.V,p.valueTime,p.valueFeature,l.chunk*p.BT,n*p.BV⟩

def pointers (p : Layout) (l : Launch) (r : Regions) (n : Nat) (u : BlockState) : BlockState :=
  (((((u.setReg "p_v" .blockPtr [p.BT,p.BV] (valuePointer p l r.value n)).setReg
    "p_h" .blockPtr [p.BV,p.BK] (historyPointer p l r.history n)).setReg
    "p_do" .blockPtr [p.BT,p.BV] (valuePointer p l r.incoming n)).setReg
    "p_dh" .blockPtr [p.BK,p.BV] (historyGradientPointer p l r.historyGradient n)).setReg
    "p_dv" .blockPtr [p.BT,p.BV] (dvPointer p l r.dv n))

def loaded (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (n : Nat) (u : BlockState) : BlockState :=
  ((((pointers p l r n u).setReg "b_v" .real [p.BT,p.BV] (tile (v p l a n))).setReg
    "b_do" .real [p.BT,p.BV] (tile (o p l a n))).setReg
    "b_h" .real [p.BV,p.BK] (tile (h p l a n))).setReg
    "b_dh" .real [p.BK,p.BV] (tile (dh p l a n))

def prepared (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (n : Nat) (u : BlockState) : BlockState :=
  ((((loaded p l r a n u).setReg "b_ds" .real [p.BT,p.BT] (tile (cs p l a (n+1)))).setReg
    "b_dq" .real [p.BT,p.BK] (tile (cq p l a scale (n+1)))).setReg
    "b_dk" .real [p.BT,p.BK] (tile (ck p l a (n+1)))).setReg
    "b_dv" .real [p.BT,p.BV] (tile (dv p l a scale n))

def prelude (p : Layout) (r : Regions) (scale : ℝ) : List Stmt := (iteration p r scale).take 13

theorem iteration_parts (p : Layout) (r : Regions) (scale : ℝ) :
    iteration p r scale = prelude p r scale ++ [store2 p.BT p.BV "p_dv" "b_dv"] := rfl

theorem prefix_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (n : Nat) (u : BlockState) (hf : Fixed p l a scale u) (hc : Carries p l a scale n u)
    (hm : Reads r a u) (hu : u.undef = fun _ _ => 0)
    (hn : u.regs .nat [] "i_v" = some (Tile.scalar n)) :
    stepStmts (prelude p r scale) u = some (prepared p l r a scale n u) := by
  unfold prelude iteration
  simp only [List.take_succ_cons,List.take_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.value n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_eval p l r.history n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_eval p l r.incoming n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (historyGradient_eval p l r.historyGradient n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (dv_eval p l r.dv n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn]) (by simp [hf.heads])))]
  change stepStmts (.assign .real [p.BT,p.BV] "b_v" _ :: _) (pointers p l r n u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.value a n "p_v" _
    (by simpa [pointers] using hm.value) (by simpa [pointers] using congrFun hu r.value) (by simp [pointers]))) ]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (incoming_load p l r.incoming a n "p_do" _
    (by simpa [pointers] using hm.incoming) (by simpa [pointers] using congrFun hu r.incoming) (by simp [pointers]))) ]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (history_load p l r.history a n "p_h" _
    (by simpa [pointers] using hm.history) (by simpa [pointers] using congrFun hu r.history) (by simp [pointers]))) ]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (historyGradient_load p l r.historyGradient a n "p_dh" _
    (by simpa [pointers] using hm.gradient) (by simpa [pointers] using congrFun hu r.historyGradient) (by simp [pointers]))) ]
  change stepStmts (.assign .real [p.BT,p.BT] "b_ds" _ :: _) (loaded p l r a n u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_ds p l a scale n _
    (by simp [loaded,pointers,hc.ds]) (by simp [loaded,pointers]) (by simp [loaded,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_dq p l a scale n _
    (by simp [loaded,pointers,hc.dq]) (by simp [loaded,pointers]) (by simp [loaded,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (update_dk p l a scale n _
    (by simp [loaded,pointers,hc.dk]) (by simp [loaded,pointers]) (by simp [loaded,pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (output_dv p l a scale n _
    (by simp [loaded,pointers,hf.keys]) (by simp [loaded,pointers]) (by simp [loaded,pointers,hf.score]) (by simp [loaded,pointers]) (by simp [loaded,pointers,hf.keyWeights])))]
  rw [stepStmts.nil]
  rfl

structure Result (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (n : Nat) (u w : BlockState) : Prop where
  fixed : Fixed p l a scale w
  carries : Carries p l a scale (n+1) w
  reads : Reads r a w
  output : ∀ (i : Fin p.BT) (j : Fin p.BV), (valueWindow p l r n).active i.val j.val →
    w.readMem r.dv ((valueWindow p l r n).address i.val j.val) = dv p l a scale n i j
  frame : BlockState.WriteWithin (valueWindow p l r n).footprint u w
  undef : w.undef = u.undef

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (n : Nat) (u : BlockState) (hf : Fixed p l a scale u) (hc : Carries p l a scale n u)
    (hm : Reads r a u) (hu : u.undef = fun _ _ => 0)
    (hn : u.regs .nat [] "i_v" = some (Tile.scalar n))
    (hs : Separate r) (hinj : (valueWindow p l r n).NoCollision) :
    ∃ w, stepStmts (iteration p r scale) u = some w ∧ Result p l r a scale n u w := by
  let pre := prepared p l r a scale n u
  let w := (valueWindow p l r n).scatter (dv p l a scale n) pre
  have he := prefix_executes p l r a scale n u hf hc hm hu hn
  refine ⟨w,?_,⟨?_,?_,?_,?_,?_,?_⟩⟩
  · rw [iteration_parts,stepStmts.append_some he,
      stepStmts.cons_some (store_eval (valueWindow p l r n) (dv p l a scale n) "p_dv" "b_dv" pre
        (by simp [pre,prepared]) (by simp [pre,prepared,loaded,pointers]; rfl)),stepStmts.nil]
  · constructor
    · simpa [w,pre,prepared,loaded,pointers] using hf.key
    · simpa [w,pre,prepared,loaded,pointers] using hf.chunk
    · simpa [w,pre,prepared,loaded,pointers] using hf.head
    · simpa [w,pre,prepared,loaded,pointers] using hf.heads
    · simpa [w,pre,prepared,loaded,pointers] using hf.queryWeights
    · simpa [w,pre,prepared,loaded,pointers] using hf.keyWeights
    · simpa [w,pre,prepared,loaded,pointers] using hf.scoreWeights
    · simpa [w,pre,prepared,loaded,pointers] using hf.query
    · simpa [w,pre,prepared,loaded,pointers] using hf.keys
    · simpa [w,pre,prepared,loaded,pointers] using hf.score
  · constructor <;> simp [w,pre,prepared]
  · constructor
    · intro off
      exact (other_region (valueWindow p l r n) _ pre r.value hs.value off).trans (hm.value off)
    · intro off
      exact (other_region (valueWindow p l r n) _ pre r.incoming hs.incoming off).trans (hm.incoming off)
    · intro off
      exact (other_region (valueWindow p l r n) _ pre r.history hs.history off).trans (hm.history off)
    · intro off
      exact (other_region (valueWindow p l r n) _ pre r.historyGradient hs.gradient off).trans (hm.gradient off)
  · exact readback _ _ pre hinj
  · exact frame _ _ pre
  · simp only [w,scatter_undef]
    rfl

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVStep

/-! The complete value-block loop. Each dV tile is written once and retained;
three carries are independent sums over all finished blocks of initial arrays. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVLoop
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram RealQKVTiles RealQKVStep RealMatrix RealMatrixStore
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def NoCollision (p : Layout) (l : Launch) (r : Regions) : Prop :=
  ∀ (c d : Nat) (i e : Fin p.BT) (j f : Fin p.BV), c < numBlocks p → d < numBlocks p →
    (valueWindow p l r c).active i.val j.val → (valueWindow p l r d).active e.val f.val →
    (valueWindow p l r c).address i.val j.val = (valueWindow p l r d).address e.val f.val →
    c = d ∧ i = e ∧ j = f

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  ∃ n, n < numBlocks p ∧ (valueWindow p l r n).footprint cell

structure Holds (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (n : Nat) (u : BlockState) : Prop where
  count_le : n ≤ numBlocks p
  fixed : Fixed p l a scale u
  carries : Carries p l a scale n u
  reads : Reads r a u
  outputs : ∀ (c : Nat) (i : Fin p.BT) (j : Fin p.BV), c < n → (valueWindow p l r c).active i.val j.val →
    u.readMem r.dv ((valueWindow p l r c).address i.val j.val) = dv p l a scale c i j
  frame : BlockState.WriteWithin (footprint p l r) initial u
  undef : u.undef = initial.undef

theorem iteration (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (hu : initial.undef = fun _ _ => 0) (hs : Separate r) (hinj : NoCollision p l r)
    (n : Nat) (u : BlockState) (hn : n < numBlocks p) (h : Holds initial p l r a scale n u) :
    ∃ w, stepStmts (RealQKVProgram.iteration p r scale) (u.setReg "i_v" .nat [] (Tile.scalar n)) = some w ∧
      Holds initial p l r a scale (n+1) w := by
  let entry := u.setReg "i_v" .nat [] (Tile.scalar n)
  have hci : (valueWindow p l r n).NoCollision := by
    intro i e j f hi he heq
    exact (hinj n n i e j f hn hn hi he heq).2
  have hfix : Fixed p l a scale entry := by
    rcases h.fixed with ⟨hk,ht,hh,hnh,hwq,hwk,hws,hq,hk',hs'⟩
    constructor <;> simp [entry, hk, ht, hh, hnh, hwq, hwk, hws, hq, hk', hs']
  have hcarry : Carries p l a scale n entry := by
    rcases h.carries with ⟨hds,hdq,hdk⟩
    constructor <;> simp [entry, hds, hdq, hdk]
  have hread : Reads r a entry := ⟨h.reads.value,h.reads.incoming,h.reads.history,h.reads.gradient⟩
  obtain ⟨w,he,hw⟩ := RealQKVStep.executes p l r a scale n entry hfix hcarry hread
    (h.undef.trans hu) (by simp [entry]) hs hci
  refine ⟨w,he,Nat.succ_le_of_lt hn,hw.fixed,hw.carries,hw.reads,?_,?_,hw.undef.trans h.undef⟩
  · intro c i j hc ha
    by_cases hcn : c = n
    · subst c
      exact hw.output i j ha
    · have hc' : c < n := by omega
      have hf := hw.frame r.dv ((valueWindow p l r c).address i.val j.val) (by
        rintro ⟨_,e,f,hactive,haddr⟩
        exact hcn (hinj c n i e j f (Nat.lt_trans hc' hn) hn ha hactive haddr).1)
      unfold BlockState.readMem
      rw [← hf]
      exact h.outputs c i j hc' ha
  · intro region off hf
    exact (h.frame region off hf).trans (hw.frame region off (fun hh => hf ⟨n,hn,hh⟩))

theorem bound_eval (p : Layout) (u : BlockState) :
    evalOp (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.V) (.constNat p.BV)) (.constNat 1))
      (.constNat p.BV)) u = some (Tile.scalar (numBlocks p)) := by
  simp only [evalOp_div,evalOp_sub,evalOp_add,evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem executes (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (hu : initial.undef = fun _ _ => 0) (hs : Separate r) (hinj : NoCollision p l r)
    (u : BlockState) (h : Holds initial p l r a scale 0 u) :
    ∃ w, stepStmt (loop p r scale) u = some w ∧ Holds initial p l r a scale (numBlocks p) w := by
  obtain ⟨last,w,he,hend,hw⟩ := forRangeDyn_inv (idx := "i_v") (start := 0) (stop := numBlocks p) (step := 1)
    (P := Holds initial p l r a scale) (body := RealQKVProgram.iteration p r scale)
    (evalOp_constNat 0 u) (bound_eval p u) (evalOp_constNat 1 u) (by decide : 1 ≠ 0) h
    (fun n u hn h => iteration initial p l r a scale hu hs hinj n u hn h)
  have hlast : last = numBlocks p := Nat.le_antisymm hw.count_le hend
  subst last
  exact ⟨w,he,hw⟩

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVLoop

/-! Initial Q/K loads and the upper causal score, before any output write. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVInit
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs)
open RealQKVProgram RealQKVTiles RealQKVPointers RealQKVAlgebra RealQKVStep RealMatrix
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def pointers (p : Layout) (r : Regions) (scale : ℝ) (u : BlockState) : BlockState :=
  ((RealQKVDecay.state p scale u).setReg "p_q" .blockPtr [p.BK,p.BT] (queryPointer p (launch u) r.query)).setReg
    "p_k" .blockPtr [p.BT,p.BK] (keyPointer p (launch u) r.key)

def loaded (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) : BlockState :=
  ((pointers p r scale u).setReg "b_q" .real [p.BK,p.BT] (tile (q p (launch u) a))).setReg
    "b_k" .real [p.BT,p.BK] (tile (k p (launch u) a))

def scored (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) : BlockState :=
  (loaded p r a scale u).setReg "b_s" .real [p.BT,p.BT] (tile (s p (launch u) a scale))

def initial (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) : BlockState :=
  (((scored p r a scale u).setReg "b_dq" .real [p.BT,p.BK] (tile (fun _ _ => 0))).setReg
    "b_dk" .real [p.BT,p.BK] (tile (fun _ _ => 0))).setReg
    "b_ds" .real [p.BT,p.BT] (tile (fun _ _ => 0))

theorem executes (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hq : ∀ off, u.readMem r.query off = a.query off)
    (hk : ∀ off, u.readMem r.key off = a.key off) :
    stepStmts (initialization p r scale) u = some (initial p r a scale u) := by
  have hsplit : initialization p r scale = RealQKVDecay.prologue p scale ++
      (initialization p r scale).drop 12 := rfl
  rw [hsplit,stepStmts.append_some (RealQKVDecay.executes p scale u)]
  unfold initialization
  simp only [List.drop_succ_cons,List.drop_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_eval p (launch u) r.query _
    (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p (launch u) r.key _
    (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch])))]
  change stepStmts (.assign .real [p.BK,p.BT] "b_q" _ :: _) (pointers p r scale u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (query_load p (launch u) r.query a "p_q" _
    (by simpa [pointers,RealQKVDecay.state] using hq) (by simpa [pointers,RealQKVDecay.state] using congrFun hu r.query)
    (by simp [pointers])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p (launch u) r.key a "p_k" _
    (by simpa [pointers,RealQKVDecay.state] using hk) (by simpa [pointers,RealQKVDecay.state] using congrFun hu r.key)
    (by simp [pointers])))]
  change stepStmts (.assign .real [p.BT,p.BT] "b_s" _ :: _) (loaded p r a scale u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (score_eval p (launch u) a scale _
    (by simp [loaded]) (by simp [loaded]) (by simp [loaded,pointers,RealQKVDecay.state])))]
  change stepStmts (.assign .real [p.BT,p.BK] "b_dq" _ :: _) (scored p r a scale u) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BK] (.const 0)) _ =
    some (tile (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BK] (.const 0)) _ =
    some (tile (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BT] (.const 0)) _ =
    some (tile (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl)),stepStmts.nil]
  rfl

theorem invariant (p : Layout) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hm : Reads r a u) :
    RealQKVLoop.Holds u p (launch u) r a scale 0 (initial p r a scale u) := by
  constructor
  · exact Nat.zero_le _
  · constructor <;> simp [initial,scored,loaded,pointers,RealQKVDecay.state,launch]
  · constructor
    all_goals
      simp [initial]
      apply congrArg tile
      funext i j
      simp [cs,cq,ck,RealQKVMath.dsSum,RealQKVMath.dqSum,RealQKVMath.dkSum]
  · constructor
    · intro off; simpa only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_readMem] using hm.value off
    · intro off; simpa only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_readMem] using hm.incoming off
    · intro off; simpa only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_readMem] using hm.history off
    · intro off; simpa only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_readMem] using hm.gradient off
  · intro c _ _ hc; omega
  · intro rg off _
    simp only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_mem]
  · funext rg off
    simp only [initial,scored,loaded,pointers,RealQKVDecay.state,BlockState.setReg_undef]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVInit

/-! Lower causal score-gradient mask, final dQ/dK contractions and both stores.
Readback requires distinct outputs, while the frame tracks the exact windows. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVFinal
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram RealQKVTiles RealQKVPointers RealQKVAlgebra RealQKVStep RealMatrix RealMatrixStore
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def keyWindow (p : Layout) (l : Launch) (region : RegionName) : Window p.BT p.BK :=
  ⟨region,l.head*p.keyHead,p.T,p.K,p.keyTime,p.keyFeature,l.chunk*p.BT,l.key*p.BK⟩

def prepared (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) : BlockState :=
  (((((u.setReg "b_ds" .real [p.BT,p.BT] (tile (ds p l a scale))).setReg
    "b_dq" .real [p.BT,p.BK] (tile (dq p l a scale))).setReg
    "b_dk" .real [p.BT,p.BK] (tile (dk p l a scale))).setReg
    "p_dq" .blockPtr [p.BT,p.BK] (keyPointer p l r.dq)).setReg
    "p_dk" .blockPtr [p.BT,p.BK] (keyPointer p l r.dk))

def output (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) : BlockState :=
  (keyWindow p l r.dk).scatter (dk p l a scale)
    ((keyWindow p l r.dq).scatter (dq p l a scale) (prepared p l r a scale u))

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  WriteFootprint.union (keyWindow p l r.dq).footprint (keyWindow p l r.dk).footprint

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (swapped : Bool) (u : BlockState) (hf : Fixed p l a scale u) (hc : Carries p l a scale (numBlocks p) u) :
    stepStmts (finalization p r scale swapped) u = some (output p l r a scale u) := by
  unfold finalization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (final_ds p l a scale u hf.scoreWeights hc.ds))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (final_dq p l a scale swapped _
    (by simp [hc.dq]) (by simp) (by simp [hf.keys]) (by simp [hf.queryWeights])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (final_dk p l a scale _
    (by simp [hc.dk]) (by simp [hf.query]) (by simp) (by simp [hf.keyWeights])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.dq _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_eval p l r.dk _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk])))]
  change stepStmts [store2 p.BT p.BK "p_dq" "b_dq",store2 p.BT p.BK "p_dk" "b_dk"]
    (prepared p l r a scale u) = _
  rw [stepStmts.cons_some (store_eval (keyWindow p l r.dq) (dq p l a scale) "p_dq" "b_dq" _
    (by simp [prepared]) (by simp [prepared]; rfl))]
  rw [stepStmts.cons_some (store_eval (keyWindow p l r.dk) (dk p l a scale) "p_dk" "b_dk" _
    (by simp [prepared]) (by simp [prepared]; rfl)),stepStmts.nil]
  rfl

theorem frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r) u (output p l r a scale u) := by
  intro region off hf
  have hq := RealMatrixStore.frame (keyWindow p l r.dq) (dq p l a scale) (prepared p l r a scale u)
    region off (fun h => hf (Or.inl h))
  have hk := RealMatrixStore.frame (keyWindow p l r.dk) (dk p l a scale)
    ((keyWindow p l r.dq).scatter (dq p l a scale) (prepared p l r a scale u))
    region off (fun h => hf (Or.inr h))
  exact hq.trans hk

theorem query_readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hs : r.dq ≠ r.dk) (hi : (keyWindow p l r.dq).NoCollision)
    (i : Fin p.BT) (e : Fin p.BK) (ha : (keyWindow p l r.dq).active i.val e.val) :
    (output p l r a scale u).readMem r.dq ((keyWindow p l r.dq).address i.val e.val) = dq p l a scale i e := by
  unfold output
  rw [other_region _ _ _ r.dq hs]
  exact readback _ _ _ hi i e ha

theorem key_readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (u : BlockState)
    (hi : (keyWindow p l r.dk).NoCollision)
    (i : Fin p.BT) (e : Fin p.BK) (ha : (keyWindow p l r.dk).active i.val e.val) :
    (output p l r a scale u).readMem r.dk ((keyWindow p l r.dk).address i.val e.val) = dk p l a scale i e := by
  exact readback _ _ _ hi i e ha

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVFinal

/-! Both complete typed gradient sources after real erasure. All three output
formulas refer to initial arrays, and the frame retains every other tagged cell.
No successful execution, prepared carry or output equation is a precondition. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVExecution
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram RealMatrixStore
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

structure Separate (r : Regions) : Prop where
  loop : RealQKVStep.Separate r
  query_key : r.dq ≠ r.dk
  query_value : r.dq ≠ r.dv
  key_value : r.dk ≠ r.dv

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  WriteFootprint.union (RealQKVLoop.footprint p l r) (RealQKVFinal.footprint p l r)

structure Outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (w : BlockState) : Prop where
  query : ∀ (i : Fin p.BT) (e : Fin p.BK), RealQKVMath.keyActive p l i.val e.val →
    w.readMem r.dq (RealQKVMath.keyAddress p l i.val e.val) = RealQKVMath.dq p l a scale i.val e.val
  key : ∀ (i : Fin p.BT) (e : Fin p.BK), RealQKVMath.keyActive p l i.val e.val →
    w.readMem r.dk (RealQKVMath.keyAddress p l i.val e.val) = RealQKVMath.dk p l a scale i.val e.val
  value : ∀ (c : Nat) (i : Fin p.BT) (j : Fin p.BV), c < numBlocks p → RealQKVMath.valueActive p l c i.val j.val →
    w.readMem r.dv (RealQKVMath.dvAddress p l c i.val j.val) = RealQKVMath.dv p l a scale c i.val j.val

theorem executes (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool)
    (a : Inputs) (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : Separate r)
    (hvi : RealQKVLoop.NoCollision p (launch u) r)
    (hki : (RealQKVFinal.keyWindow p (launch u) r.dq).NoCollision)
    (hq : ∀ off, u.readMem r.query off = a.query off)
    (hk : ∀ off, u.readMem r.key off = a.key off) (hm : RealQKVStep.Reads r a u) :
    ∃ w, exec (source p r storage scale swapped).eraseDType.toAlgKernel u = some w ∧
      Outputs p (launch u) r a scale w ∧ BlockState.WriteWithin (footprint p (launch u) r) u w := by
  let l := launch u
  let start := RealQKVInit.initial p r a scale u
  have hei := RealQKVInit.executes p r a scale u hu hq hk
  obtain ⟨mid,hel,hm⟩ := RealQKVLoop.executes u p l r a scale hu hs.loop hvi start
    (RealQKVInit.invariant p r a scale u hm)
  let w := RealQKVFinal.output p l r a scale mid
  have hef := RealQKVFinal.executes p l r a scale swapped mid hm.fixed hm.carries
  have hf := RealQKVFinal.frame p l r a scale mid
  refine ⟨w,?_,⟨?_,?_,?_⟩,?_⟩
  · rw [exec,source_body,body,List.append_assoc,stepStmts.append_some hei]
    change stepStmts (loop p r scale :: finalization p r scale swapped) start = some w
    rw [stepStmts.cons_some hel]
    exact hef
  · intro i e ha
    exact RealQKVFinal.query_readback p l r a scale mid hs.query_key hki i e ha
  · intro i e ha
    exact RealQKVFinal.key_readback p l r a scale mid hki i e ha
  · intro c i j hc ha
    have hframe := hf r.dv (RealQKVMath.dvAddress p l c i.val j.val) (by
      rintro (⟨hr,_⟩ | ⟨hr,_⟩)
      · exact hs.query_value hr.symm
      · exact hs.key_value hr.symm)
    unfold BlockState.readMem
    rw [← hframe]
    exact hm.outputs c i j hc ha
  · intro region off hn
    exact (hm.frame region off (fun h => hn (Or.inl h))).trans (hf region off (fun h => hn (Or.inr h)))

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVExecution

#guard_msgs (drop info) in
#auditModuleAxioms

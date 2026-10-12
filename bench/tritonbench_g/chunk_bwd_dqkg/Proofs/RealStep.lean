import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.LegacyReal
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealMath

/-! One real value-block iteration at an arbitrary loop index and with four
arbitrary incoming accumulators. This is the complete twelve-statement body
of the existing real lowering, not the old zero-carry, index-zero special case.
The typed source and its reduction lowering are connected separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealStep
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Regions where
  v : RegionName
  h : RegionName
  dout : RegionName
  dh : RegionName

def code (p : Layout) (r : Regions) : List Stmt :=
  cbdLoopBody r.v r.h r.dout r.dh p.s_v_h p.s_v_t p.s_h_h p.s_h_t
    p.T p.K p.V p.BT p.BK p.BV p.NT

def vPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :
    Tile .blockPtr [p.BT, p.BV] :=
  ⟨fun _ =>
    { region := r, baseOffset := l.head * p.s_v_h,
      parentShape := [p.T, p.V], blockShape := [p.BT, p.BV],
      strides := [p.s_v_t, 1], offsets := [l.time * p.BT, c * p.BV] }⟩

def hPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :
    Tile .blockPtr [p.BV, p.BK] :=
  ⟨fun _ =>
    { region := r, baseOffset := l.head * p.s_h_h,
      parentShape := [p.V, p.NT * p.K], blockShape := [p.BV, p.BK],
      strides := [1, p.s_h_t], offsets := [c * p.BV, l.time * p.K + l.key * p.BK] }⟩

def vTile (p : Layout) (l : Launch) (a : Nat → ℝ) (c : Nat) : Tile .real [p.BT, p.BV] :=
  Tile.mat fun r v => some (value p l a r.val (c * p.BV + v.val))

def hTile (p : Layout) (l : Launch) (a : Nat → ℝ) (c : Nat) : Tile .real [p.BV, p.BK] :=
  Tile.mat fun v e => some (history p l a (c * p.BV + v.val) e.val)

def result (s : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (a : Inputs) (c : Nat) (acc : Accumulators p.BT p.BK) : BlockState :=
  let next := add acc (block p l a c)
  let t1 := s.setReg "p_v" .blockPtr [p.BT, p.BV] (vPointer p l r.v c)
  let t2 := t1.setReg "p_h" .blockPtr [p.BV, p.BK] (hPointer p l r.h c)
  let t3 := t2.setReg "p_do" .blockPtr [p.BT, p.BV] (vPointer p l r.dout c)
  let t4 := t3.setReg "p_dh" .blockPtr [p.BV, p.BK] (hPointer p l r.dh c)
  let t5 := t4.setReg "b_v" .real [p.BT, p.BV] (vTile p l a.v c)
  let t6 := t5.setReg "b_do" .real [p.BT, p.BV] (vTile p l a.dout c)
  let t7 := t6.setReg "b_h" .real [p.BV, p.BK] (hTile p l a.h c)
  let t8 := t7.setReg "b_dh" .real [p.BV, p.BK] (hTile p l a.dh c)
  let t9 := t8.setReg "b_dg_last" .real [1] (Tile.vec fun _ => some next.dgLast)
  let t10 := t9.setReg "b_ds" .real [p.BT, p.BT] (Tile.mat fun i j => some (next.ds i j))
  let t11 := t10.setReg "b_dq" .real [p.BT, p.BK] (Tile.mat fun i j => some (next.dq i j))
  t11.setReg "b_dk" .real [p.BT, p.BK] (Tile.mat fun i j => some (next.dk i j))

def writes : List RegName :=
  ["p_v", "p_h", "p_do", "p_dh", "b_v", "b_do", "b_h", "b_dh",
    "b_dg_last", "b_ds", "b_dq", "b_dk"]

theorem register_frame (s : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (a : Inputs) (c : Nat) (acc : Accumulators p.BT p.BK)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hn : name ∉ writes) :
    (result s p l r a c acc).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, List.mem_nil_iff, or_false, not_or] at hn
  simp [result, BlockState.setReg, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1,
    hn.2.2.2.2.1, hn.2.2.2.2.2.1, hn.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.1,
    hn.2.2.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.2.2.1,
    hn.2.2.2.2.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.2.2.2.2]

theorem mul_ref (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

theorem add_nat (t : BlockState) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a t = some (Tile.scalar va))
    (hb : evalOp b t = some (Tile.scalar vb)) :
    evalOp (Op.add .nat Broadcast.nil a b) t = some (Tile.scalar (va + vb)) := by
  rw [evalOp_add]
  simp only [ha, hb, Option.bind_some, Option.bind_eq_bind]
  rfl

theorem make_pointer (R : RegionName) (d0 d1 st0 st1 B0 B1 : Nat)
    (baseOp o0 o1 : Op .nat []) (t : BlockState) (base f0 f1 : Nat)
    (hbase : evalOp baseOp t = some (Tile.scalar base))
    (h0 : evalOp o0 t = some (Tile.scalar f0))
    (h1 : evalOp o1 t = some (Tile.scalar f1)) :
    evalOp (Op.makeBlockPtrDynOffsets R baseOp [d0, d1] [B0, B1] [st0, st1] [o0, o1]) t
      = some (⟨fun _ : TileIndex [B0, B1] =>
          { region := R, baseOffset := base, parentShape := [d0, d1],
            blockShape := [B0, B1], strides := [st0, st1], offsets := [f0, f1] }⟩
          : Tile .blockPtr [B0, B1]) := by
  rw [makeBlockPtr2_eval]
  simp only [hbase, h0, h1, List.mapM_cons, List.mapM_nil, Option.bind_some,
    Option.pure_def, Option.bind_eq_bind, Tile.scalar_data]

theorem load_value (s : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (a : Nat → ℝ) (c : Nat)
    (hu : s.undef = fun _ _ => 0)
    (hm : ∀ off, s.readMem r off = a off)
    (hp : s.regs .blockPtr [p.BT, p.BV] name = some (vPointer p l r c)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BT, p.BV] name) [0, 1]) .none .unspecified) s =
      some (vTile p l a c) := by
  rw [evalOp_load_padding_of_undef_zero _ _ _ _ _ hu]
  simp only [evalOp, evalOp_ref, hp]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [vPointer, vTile, Tile.mat, value, TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real, decide_eq_true_eq, Nat.mul_one]
  split <;> simp_all [BlockState.defaultCarrier]

theorem load_history (s : BlockState) (p : Layout) (l : Launch)
    (r : RegionName) (name : RegName) (a : Nat → ℝ) (c : Nat)
    (hu : s.undef = fun _ _ => 0)
    (hm : ∀ off, s.readMem r off = a off)
    (hp : s.regs .blockPtr [p.BV, p.BK] name = some (hPointer p l r c)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BV, p.BK] name) [0, 1]) .none .unspecified) s =
      some (hTile p l a c) := by
  rw [evalOp_load_padding_of_undef_zero _ _ _ _ _ hu]
  simp only [evalOp, evalOp_ref, hp]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [hPointer, hTile, Tile.mat, history, TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real, decide_eq_true_eq, Nat.mul_one, Nat.add_assoc]
  split <;> simp_all [BlockState.defaultCarrier]

theorem coe_sum {n : Nat} (f : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e => (some (f e) : WithBot ℝ)) = some (∑ e : Fin n, f e) := by
  show (Finset.univ.sum fun e => ((f e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

theorem coe_sum2 {m n : Nat} (f g : Fin m → Fin n → ℝ) :
    (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
        @Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          Option.map₂ (fun x y : ℝ => x * y) (some (f a e)) (some (g a e)))
      = some (∑ a : Fin m, ∑ e : Fin n, f a e * g a e) := by
  rw [show (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
        @Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          Option.map₂ (fun x y : ℝ => x * y) (some (f a e)) (some (g a e)))
      = (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
          (some (∑ e : Fin n, f a e * g a e) : WithBot ℝ))
      from Finset.sum_congr rfl fun a _ => coe_sum _]
  exact coe_sum _

/-- The real projection expresses the flattened reduction as a double sum. No corresponding FP
identity is asserted by this lemma. -/
theorem dg_add (s : BlockState) (p : Layout) (l : Launch) (a : Inputs) (c : Nat)
    (acc : ℝ)
    (ha : s.regs .real [1] "b_dg_last" = some (Tile.vec fun _ => some acc))
    (hh : s.regs .real [p.BV, p.BK] "b_h" = some (hTile p l a.h c))
    (hd : s.regs .real [p.BV, p.BK] "b_dh" = some (hTile p l a.dh c)) :
    evalOp (Op.add .real Broadcast.scalarR (Op.ref .real [1] "b_dg_last")
      (Op.reduceSum (shape := [p.BV * p.BK]) ⟨0, by simp⟩ Bool.false
        (Op.reshape [p.BV * p.BK]
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [p.BV, p.BK] "b_h") (Op.ref .real [p.BV, p.BK] "b_dh"))))) s =
      some (Tile.vec fun _ => some (acc + (block p l a c).dgLast)) := by
  erw [evalOp_add, evalOp_ref, ha, evalOp_reduceSum, evalOp, evalOp_mul,
    evalOp_ref, hh, evalOp_ref, hd]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [hTile, Tile.mat_data, Tile.vec, Tile.bop_data, Tile.reduceSum_reshape_two, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Broadcast.leftIndex, Broadcast.rightIndex, block, blockSum, dgLastTerm]
  erw [coe_sum2]
  rfl

theorem dot_add (s : BlockState) (M N K : Nat) (name : RegName)
    (acc : Fin M → Fin N → ℝ) (x : Fin M → Fin K → ℝ) (y : Fin K → Fin N → ℝ)
    (xo : Op .real [M, K]) (yo : Op .real [K, N])
    (ha : s.regs .real [M, N] name = some (Tile.mat fun i j => some (acc i j)))
    (hx : evalOp xo s = some (Tile.mat fun i j => some (x i j)))
    (hy : evalOp yo s = some (Tile.mat fun i j => some (y i j))) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [M, N] name) (Op.dot (batch := []) xo yo)) s =
      some (Tile.mat fun i j => some (acc i j + ∑ k : Fin K, x i k * y k j)) := by
  erw [evalOp_add, evalOp_ref, ha, evalOp_dot, hx, hy]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data M K N _ _ i j (x i) (fun k => y k j) (fun _ => rfl) (fun _ => rfl)]
  rfl

theorem executes (s : BlockState) (p : Layout) (l : Launch) (r : Regions)
    (a : Inputs) (c : Nat) (acc : Accumulators p.BT p.BK)
    (hu : s.undef = fun _ _ => 0)
    (hmv : ∀ off, s.readMem r.v off = a.v off)
    (hmh : ∀ off, s.readMem r.h off = a.h off)
    (hmo : ∀ off, s.readMem r.dout off = a.dout off)
    (hmd : ∀ off, s.readMem r.dh off = a.dh off)
    (hik : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hiv : s.regs .nat [] "i_v" = some (Tile.scalar c))
    (hdq : s.regs .real [p.BT, p.BK] "b_dq" = some (Tile.mat fun i j => some (acc.dq i j)))
    (hdk : s.regs .real [p.BT, p.BK] "b_dk" = some (Tile.mat fun i j => some (acc.dk i j)))
    (hds : s.regs .real [p.BT, p.BT] "b_ds" = some (Tile.mat fun i j => some (acc.ds i j)))
    (hdg : s.regs .real [1] "b_dg_last" = some (Tile.vec fun _ => some acc.dgLast)) :
    stepStmts (code p r) s = some (result s p l r a c acc) := by
  unfold code cbdLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.v p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ s
      (l.head * p.s_v_h) (l.time * p.BT) (c * p.BV)
      (mul_ref s "i_bh" l.head p.s_v_h hib)
      (mul_ref s "i_t" l.time p.BT hit)
      (mul_ref s "i_v" c p.BV hiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.h p.V (p.NT * p.K) 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h) (c * p.BV) (l.time * p.K + l.key * p.BK)
      (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))
      (add_nat _ _ _ (l.time * p.K) (l.key * p.BK)
        (mul_ref _ "i_t" l.time p.K (by simp [hit]))
        (mul_ref _ "i_k" l.key p.BK (by simp [hik])))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.dout p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ _
      (l.head * p.s_v_h) (l.time * p.BT) (c * p.BV)
      (mul_ref _ "i_bh" l.head p.s_v_h (by simp [hib]))
      (mul_ref _ "i_t" l.time p.BT (by simp [hit]))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.dh p.V (p.NT * p.K) 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h) (c * p.BV) (l.time * p.K + l.key * p.BK)
      (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
      (mul_ref _ "i_v" c p.BV (by simp [hiv]))
      (add_nat _ _ _ (l.time * p.K) (l.key * p.BK)
        (mul_ref _ "i_t" l.time p.K (by simp [hit]))
        (mul_ref _ "i_k" l.key p.BK (by simp [hik])))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_value _ p l r.v "p_v" a.v c (by simpa using hu) (by intro off; simp [hmv]) (by simp [vPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_value _ p l r.dout "p_do" a.dout c (by simpa using hu) (by intro off; simp [hmo]) (by simp [vPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_history _ p l r.h "p_h" a.h c (by simpa using hu) (by intro off; simp [hmh]) (by simp [hPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_history _ p l r.dh "p_dh" a.dh c (by simpa using hu) (by intro off; simp [hmd]) (by simp [hPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dg_add _ p l a c acc.dgLast (by simp [hdg]) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BT p.BV "b_ds" acc.ds
      (fun i k => value p l a.dout i.val (c * p.BV + k.val))
      (fun k j => value p l a.v j.val (c * p.BV + k.val)) _ _
      (by simp [hds]) (by simp [evalOp_ref, vTile])
      (by erw [evalOp_transpose]; simp [evalOp_ref, vTile]; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BK p.BV "b_dq" acc.dq
      (fun i k => value p l a.dout i.val (c * p.BV + k.val))
      (fun k j => history p l a.h (c * p.BV + k.val) j.val) _ _
      (by simp [hdq]) (by simp [evalOp_ref, vTile]) (by simp [evalOp_ref, hTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BK p.BV "b_dk" acc.dk
      (fun i k => value p l a.v i.val (c * p.BV + k.val))
      (fun k j => history p l a.dh (c * p.BV + k.val) j.val) _ _
      (by simp [hdk]) (by simp [evalOp_ref, vTile]) (by simp [evalOp_ref, hTile])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealStep

import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealStep

/-! A real-semantics witness for a partial V tile. Actual inputs are all zero,
but unspecified V/H/DO/DH lanes are one. A separate 16-value array describes
these loaded tiles; it is only a proof device and never changes the source V=1.
The 16×16 tile sizes are retained throughout the actual execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.UndefinedTailLoop
open VeriTile.Triton RealMath RealStep
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def p : Layout := ⟨16,16,1,16,16,16,1,256,16,256,16,256,16,1⟩
def wide : Layout := { p with V := 16 }
def l : Launch := ⟨0,0,0,1⟩
def r : Regions := ⟨"v","h","do","dh"⟩
def sample (off : Nat) : ℝ := if off % 16 = 0 then 0 else 1
def a : Inputs := ⟨fun _ => 0, fun _ => 0, sample, sample, fun _ => 0, sample, sample⟩
def acc : Accumulators p.BT p.BK := ⟨fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, 0⟩
def initial : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun rg _ => if rg ∈ (["v","h","do","dh"] : List RegionName) then 1 else 0

/-- Array used solely to express the four loaded tiles and resulting sums. -/
def model : BlockState := { initial with
  mem := fun rg off => .real (if rg ∈ (["v","h","do","dh"] : List RegionName) then sample off else 0) }

def result (s : BlockState) : BlockState :=
  let next := add acc (block wide l a 0)
  let t1 := s.setReg "p_v" .blockPtr [p.BT, p.BV] (vPointer p l r.v 0)
  let t2 := t1.setReg "p_h" .blockPtr [p.BV, p.BK] (hPointer p l r.h 0)
  let t3 := t2.setReg "p_do" .blockPtr [p.BT, p.BV] (vPointer p l r.dout 0)
  let t4 := t3.setReg "p_dh" .blockPtr [p.BV, p.BK] (hPointer p l r.dh 0)
  let t5 := t4.setReg "b_v" .real [p.BT, p.BV] (vTile wide l a.v 0)
  let t6 := t5.setReg "b_do" .real [p.BT, p.BV] (vTile wide l a.dout 0)
  let t7 := t6.setReg "b_h" .real [p.BV, p.BK] (hTile wide l a.h 0)
  let t8 := t7.setReg "b_dh" .real [p.BV, p.BK] (hTile wide l a.dh 0)
  let t9 := t8.setReg "b_dg_last" .real [1] (Tile.vec fun _ => some next.dgLast)
  let t10 := t9.setReg "b_ds" .real [p.BT, p.BT] (Tile.mat fun i j => some (next.ds i j))
  let t11 := t10.setReg "b_dq" .real [p.BT, p.BK] (Tile.mat fun i j => some (next.dq i j))
  t11.setReg "b_dk" .real [p.BT, p.BK] (Tile.mat fun i j => some (next.dk i j))


theorem load_value (s : BlockState) (rg : RegionName) (name : RegName)
    (hm : ∀ off, s.readMem rg off = 0) (hu : s.undef rg = fun _ => 1)
    (hp : s.regs .blockPtr [16,16] name = some (vPointer p l rg 0)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [16,16] name) [0,1]) .none .unspecified) s =
      some (vTile wide l sample 0) := by
  simp only [evalOp, evalOp_ref, hp]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [vPointer, vTile, p, wide, l, Tile.mat, value, sample,
    TileShape.indexToList, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real, BlockState.paddingValue, BlockState.unspecifiedValue,
    hm, hu, Nat.mod_eq_of_lt j.isLt, i.isLt, j.isLt, Nat.lt_one_iff]
  split <;> rfl

theorem load_history (s : BlockState) (rg : RegionName) (name : RegName)
    (hm : ∀ off, s.readMem rg off = 0) (hu : s.undef rg = fun _ => 1)
    (hp : s.regs .blockPtr [16,16] name = some (hPointer p l rg 0)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [16,16] name) [0,1]) .none .unspecified) s =
      some (hTile wide l sample 0) := by
  simp only [evalOp, evalOp_ref, hp]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [hPointer, hTile, p, wide, l, Tile.mat, history, sample,
    TileShape.indexToList, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real, BlockState.paddingValue, BlockState.unspecifiedValue,
    hm, hu, Nat.mod_eq_of_lt i.isLt, i.isLt, j.isLt, Nat.lt_one_iff]
  split <;> rfl

theorem executes (s : BlockState)
    (hu : s.undef = initial.undef)
    (hmv : ∀ off, s.readMem r.v off = 0)
    (hmh : ∀ off, s.readMem r.h off = 0)
    (hmo : ∀ off, s.readMem r.dout off = 0)
    (hmd : ∀ off, s.readMem r.dh off = 0)
    (hik : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar l.time))
    (hib : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hiv : s.regs .nat [] "i_v" = some (Tile.scalar 0))
    (hdq : s.regs .real [p.BT, p.BK] "b_dq" = some (Tile.mat fun i j => some (acc.dq i j)))
    (hdk : s.regs .real [p.BT, p.BK] "b_dk" = some (Tile.mat fun i j => some (acc.dk i j)))
    (hds : s.regs .real [p.BT, p.BT] "b_ds" = some (Tile.mat fun i j => some (acc.ds i j)))
    (hdg : s.regs .real [1] "b_dg_last" = some (Tile.vec fun _ => some acc.dgLast)) :
    stepStmts (code p r) s = some (result s) := by
  unfold code cbdLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.v p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ s
      (l.head * p.s_v_h) (l.time * p.BT) (0 * p.BV)
      (mul_ref s "i_bh" l.head p.s_v_h hib)
      (mul_ref s "i_t" l.time p.BT hit)
      (mul_ref s "i_v" 0 p.BV hiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.h p.V (p.NT * p.K) 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h) (0 * p.BV) (l.time * p.K + l.key * p.BK)
      (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
      (mul_ref _ "i_v" 0 p.BV (by simp [hiv]))
      (add_nat _ _ _ (l.time * p.K) (l.key * p.BK)
        (mul_ref _ "i_t" l.time p.K (by simp [hit]))
        (mul_ref _ "i_k" l.key p.BK (by simp [hik])))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.dout p.T p.V p.s_v_t 1 p.BT p.BV _ _ _ _
      (l.head * p.s_v_h) (l.time * p.BT) (0 * p.BV)
      (mul_ref _ "i_bh" l.head p.s_v_h (by simp [hib]))
      (mul_ref _ "i_t" l.time p.BT (by simp [hit]))
      (mul_ref _ "i_v" 0 p.BV (by simp [hiv]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (make_pointer r.dh p.V (p.NT * p.K) 1 p.s_h_t p.BV p.BK _ _ _ _
      (l.head * p.s_h_h) (0 * p.BV) (l.time * p.K + l.key * p.BK)
      (mul_ref _ "i_bh" l.head p.s_h_h (by simp [hib]))
      (mul_ref _ "i_v" 0 p.BV (by simp [hiv]))
      (add_nat _ _ _ (l.time * p.K) (l.key * p.BK)
        (mul_ref _ "i_t" l.time p.K (by simp [hit]))
        (mul_ref _ "i_k" l.key p.BK (by simp [hik])))))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_value _ r.v "p_v" (by intro off; simp [hmv]) (by simpa [initial, r] using congrFun hu r.v) (by simp [vPointer, p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_value _ r.dout "p_do" (by intro off; simp [hmo]) (by simpa [initial, r] using congrFun hu r.dout) (by simp [vPointer, p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_history _ r.h "p_h" (by intro off; simp [hmh]) (by simpa [initial, r] using congrFun hu r.h) (by simp [hPointer, p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_history _ r.dh "p_dh" (by intro off; simp [hmd]) (by simpa [initial, r] using congrFun hu r.dh) (by simp [hPointer, p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dg_add _ wide l a 0 acc.dgLast (by simp [hdg]) (by simp [wide, p, a]) (by simp [wide, p, a])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BT p.BV "b_ds" acc.ds
      (fun i k => value wide l a.dout i.val (0 * p.BV + k.val))
      (fun k j => value wide l a.v j.val (0 * p.BV + k.val)) _ _
      (by simp [hds]) (by simp [evalOp_ref, vTile, wide, p, a])
      (by erw [evalOp_transpose]; simp [evalOp_ref, vTile, wide, p, a]; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BK p.BV "b_dq" acc.dq
      (fun i k => value wide l a.dout i.val (0 * p.BV + k.val))
      (fun k j => history wide l a.h (0 * p.BV + k.val) j.val) _ _
      (by simp [hdq]) (by simp [evalOp_ref, vTile, wide, p, a]) (by simp [evalOp_ref, hTile, wide, p, a])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dot_add _ p.BT p.BK p.BV "b_dk" acc.dk
      (fun i k => value wide l a.v i.val (0 * p.BV + k.val))
      (fun k j => history wide l a.dh (0 * p.BV + k.val) j.val) _ _
      (by simp [hdk]) (by simp [evalOp_ref, vTile, wide, p, a]) (by simp [evalOp_ref, hTile, wide, p, a])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.UndefinedTailLoop

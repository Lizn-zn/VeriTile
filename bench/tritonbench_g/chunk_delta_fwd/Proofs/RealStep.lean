import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealProgram
import VeriTile.Triton.Memory.Frame.Basic

/-! A full inner-loop iteration from its incoming registers and independent
input arrays. The step executes all four pointers, three loads, the correction,
the masked V_new store and the accumulation. Frozen H and incoming accumulation
are arbitrary; tails and nonzero inner offsets are retained. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealStep
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

theorem key_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (outer inner : Nat) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hm : ∀ off, u.readMem r off = a.key off)
    (hp : u.regs .blockPtr [p.BK,p.BC] "p_k" = some (keyPointer p l r outer inner)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BK,p.BC] "p_k") [0,1]) .none .unspecified) u =
      some (keyTile p l a outer inner) := by
  rw [load_bp_2d_ref r u "p_k" (l.head * p.keyHead) p.K p.T p.BK p.BC p.keyDim p.keyTime
    (l.key * p.BK) (timeBase p outer inner) hu hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,c,⟨⟩⟩
  simp only [keyTile, Tile.mat, keyValue, keyIndex, keyAddress, hm]
  split <;> rfl

theorem delta_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (outer inner : Nat) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hm : ∀ off, u.readMem r off = a.delta off)
    (hp : u.regs .blockPtr [p.BC,p.BK] "p_d" = some (deltaPointer p l r outer inner)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BC,p.BK] "p_d") [0,1]) .none .unspecified) u =
      some (deltaTile p l a outer inner) := by
  rw [load_bp_2d_ref r u "p_d" (l.head * p.keyHead) p.T p.K p.BC p.BK p.keyTime p.keyDim
    (timeBase p outer inner) (l.key * p.BK) hu hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨c,e,⟨⟩⟩
  simp only [deltaTile, Tile.mat, deltaValue, keyIndex, deltaAddress, hm]
  split <;> rfl

theorem value_load (p : Layout) (l : Launch) (r : RegionName) (a : Inputs)
    (outer inner : Nat) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hm : ∀ off, u.readMem r off = a.value off)
    (hp : u.regs .blockPtr [p.BC,p.BV] "p_v" = some (valuePointer p l r outer inner)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BC,p.BV] "p_v") [0,1]) .none .unspecified) u =
      some (valueTile p l a outer inner) := by
  rw [load_bp_2d_ref r u "p_v" (l.head * p.valueHead) p.T p.V p.BC p.BV p.valueTime p.valueDim
    (timeBase p outer inner) (l.value * p.BV) hu hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨c,j,⟨⟩⟩
  simp only [valueTile, Tile.mat, inputValue, activeValue, valueIndex, valueAddress, hm]
  split <;> rfl

theorem correction (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) (u : BlockState)
    (hv : u.regs .real [p.BC,p.BV] "b_v" = some (valueTile p l a outer inner))
    (hd : u.regs .real [p.BC,p.BK] "b_d" = some (deltaTile p l a outer inner))
    (hh : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen)) :
    evalOp (.sub .real Broadcast.nil.consSame.consSame (.ref .real [p.BC,p.BV] "b_v")
      (.dot (batch := []) (.ref .real [p.BC,p.BK] "b_d") (.ref .real [p.BK,p.BV] "b_h")
        {allowTF32 := some Bool.false})) u = some (correctedTile p l a outer inner frozen) := by
  exact cdfSubDot_op_eval u "b_v" "b_d" "b_h" _ _ _
    (fun c j => inputValue p l a (timeBase p outer inner + c.val) j.val)
    (fun c e => deltaValue p l a (timeBase p outer inner + c.val) e.val)
    frozen hv hd hh (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)

theorem accumulation (p : Layout) (l : Launch) (a : Inputs) (outer inner : Nat)
    (frozen acc : StateValue p) (u : BlockState)
    (ha : u.regs .real [p.BK,p.BV] "b_h_cumsum" = some (stateTile p acc))
    (hk : u.regs .real [p.BK,p.BC] "b_k" = some (keyTile p l a outer inner))
    (hv : u.regs .real [p.BC,p.BV] "b_v" = some (correctedTile p l a outer inner frozen)) :
    evalOp (.add .real Broadcast.nil.consSame.consSame (.ref .real [p.BK,p.BV] "b_h_cumsum")
      (.dot (batch := []) (.ref .real [p.BK,p.BC] "b_k") (.ref .real [p.BC,p.BV] "b_v")
        {allowTF32 := some Bool.false})) u = some (stateTile p (nextAccum p l a outer inner frozen acc)) := by
  exact cdfAccDot_op_eval u "b_h_cumsum" "b_k" "b_v" _ _ _ acc
    (fun e c => keyValue p l a (timeBase p outer inner + c.val) e.val)
    (fun c j => correctedValue p l a frozen (timeBase p outer inner + c.val) j)
    ha hk hv (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)

theorem store_eval (p : Layout) (l : Launch) (out : RegionName) (a : Inputs) (outer inner : Nat)
    (frozen : StateValue p) (u : BlockState)
    (hv : u.regs .real [p.BC,p.BV] "b_v" = some (correctedTile p l a outer inner frozen))
    (hp : u.regs .blockPtr [p.BC,p.BV] "p_v_new" = some (valuePointer p l out outer inner)) :
    stepStmt (.store .real [p.BC,p.BV] (.blockPtr (.ref .blockPtr [p.BC,p.BV] "p_v_new") [0,1])
      (.ref .real [p.BC,p.BV] "b_v") .none) u = some (scatter p l out a outer inner frozen u) := by
  unfold stepStmt scatter
  simp only [evalOp_ref, hv, hp, Option.bind, Option.map]
  apply congrArg some
  apply congrArg (fun f => List.foldl f u (TileShape.allIndices [p.BC,p.BV]))
  funext st idx
  rcases idx with ⟨c,j,⟨⟩⟩
  simp only [valuePointer, TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, activeValue, valueIndex, valueAddress]
  by_cases h : timeBase p outer inner + c.val < p.T ∧ l.value * p.BV + j.val < p.V
  · simp only [h, decide_true, if_true, BlockState.writeMemTyped_real]; rfl
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

/-- No alias or injectivity assumption is needed to compute the source step.
Such assumptions are used separately for input preservation and readback. -/
theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (outer inner : Nat)
    (frozen acc : StateValue p) (u : BlockState)
    (hu : u.undef = fun _ _ => 0)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar outer))
    (hic : u.regs .nat [] "i_c" = some (Tile.scalar inner))
    (hkey : ∀ off, u.readMem r.key off = a.key off)
    (hdelta : ∀ off, u.readMem r.delta off = a.delta off)
    (hvalue : ∀ off, u.readMem r.value off = a.value off)
    (hstate : u.regs .real [p.BK,p.BV] "b_h" = some (stateTile p frozen))
    (hacc : u.regs .real [p.BK,p.BV] "b_h_cumsum" = some (stateTile p acc)) :
    stepStmts (innerBody p r) u = some (result p l r a outer inner frozen acc u) := by
  unfold innerBody chunkDeltaInnerBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.key u _ _ _ [p.K,p.T] [p.BK,p.BC] [p.keyDim,p.keyTime]
      (l.head * p.keyHead) (l.key * p.BK) (timeBase p outer inner)
      (mulConst_eval u "i_bh" l.head p.keyHead hih)
      (mulConst_eval u "i_k" l.key p.BK hik)
      (addMulMul_eval u "i_t" "i_c" outer p.BT inner p.BC hit hic)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.delta _ _ _ _ [p.T,p.K] [p.BC,p.BK] [p.keyTime,p.keyDim]
      (l.head * p.keyHead) (timeBase p outer inner) (l.key * p.BK)
      (mulConst_eval _ "i_bh" l.head p.keyHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_k" l.key p.BK (by simp [hik]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p outer inner) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.corrected _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p outer inner) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" outer p.BT inner p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load p l r.key a outer inner _ (by simpa using hu)
    (by intro off; simp [hkey]) (by simp [keyPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (delta_load p l r.delta a outer inner _ (by simpa using hu)
    (by intro off; simp [hdelta]) (by simp [deltaPointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load p l r.value a outer inner _ (by simpa using hu)
    (by intro off; simp [hvalue]) (by simp [valuePointer])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (correction p l a outer inner frozen _
    (by simp) (by simp) (by simp [hstate])))]
  rw [stepStmts.cons_some (store_eval p l r.corrected a outer inner frozen _
    (by simp) (by simp [valuePointer]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (accumulation p l a outer inner frozen acc _
    (by simp [scatter, hacc]) (by simp [scatter]) (by simp [scatter])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealStep

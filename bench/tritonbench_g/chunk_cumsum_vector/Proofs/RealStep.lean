import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealProgram
import VeriTile.Triton.Semantics.BlockPadding

/-! Exact evaluation of one complete time chunk with arbitrary incoming carry.
Loads, triangular dot, masked store and the post-store carry update all occur
in source order. Checked-off lanes retain arbitrary unspecified values from
the initial oracle. No output formula is a premise of the execution lemma. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealStep
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def address (p : Layout) (l : Launch) (t j : Nat) : Nat :=
  l.head * p.headStride + t * p.timeStride + (l.feature * p.BS + j) * p.channelStride

def active (p : Layout) (l : Launch) (t j : Nat) : Prop :=
  t < p.T ∧ l.feature * p.BS + j < p.S
instance (p : Layout) (l : Launch) (t j : Nat) : Decidable (active p l t j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def column (p : Layout) (l : Launch) (a : Nat → ℝ) (j : Fin p.BS) (t : Nat) : ℝ :=
  if l.feature * p.BS + j.val < p.S then a (address p l t j.val) else 0

def value (p : Layout) (l : Launch) (a padding : Nat → ℝ) (t : Nat) (j : Fin p.BS) : ℝ :=
  if active p l t j.val then a (address p l t j.val) else padding (address p l t j.val)

def pointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :
    Tile .blockPtr [p.BT, p.BS] :=
  ⟨fun _ =>
    { region := r, baseOffset := l.head * p.headStride,
      parentShape := [p.T, p.S], blockShape := [p.BT, p.BS],
      strides := [p.timeStride, p.channelStride], offsets := [c * p.BT, l.feature * p.BS] }⟩

def inputTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) : Tile .real [p.BT, p.BS] :=
  Tile.mat fun i j => some (value p l a padding (c * p.BT + i.val) j)

def chunkPrefix (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : Fin p.BT) (j : Fin p.BS) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Fin p.BT => k.val ≤ i.val),
    value p l a padding (c * p.BT + k.val) j

def nextCarry (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (acc : Fin p.BS → ℝ) (j : Fin p.BS) : ℝ :=
  acc j + ∑ k : Fin p.BT, value p l a padding (c * p.BT + k.val) j

def outputTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (acc : Fin p.BS → ℝ) : Tile .real [p.BT, p.BS] :=
  Tile.mat fun i j => some (acc j + chunkPrefix p l a padding c i j)

def scatter (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT, p.BS]).foldl (fun st idx =>
    if active p l (c * p.BT + idx.1.val) idx.2.1.val then
      st.writeMem o (address p l (c * p.BT + idx.1.val) idx.2.1.val)
        (acc idx.2.1 + chunkPrefix p l a padding c idx.1 idx.2.1)
    else st) s

def beforeStore (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) : BlockState :=
  let u1 := u.setReg "p_s" .blockPtr [p.BT, p.BS] (pointer p l s c)
  let u2 := u1.setReg "p_z" .blockPtr [p.BT, p.BS] (pointer p l o c)
  let u3 := u2.setReg "b_s" .real [p.BT, p.BS] (inputTile p l a padding c)
  u3.setReg "b_c" .real [p.BT, p.BS] (outputTile p l a padding c acc)

def result (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) : BlockState :=
  (scatter p l o a padding c acc (beforeStore p l s o a padding c acc u)).setReg "b_z" .real [p.BS]
    (Tile.vec fun j => some (nextCarry p l a padding c acc j))

theorem mul_ref (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

theorem pointer_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hs : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (pointerOp p r) u = some (pointer p l r c) := by
  rw [pointerOp, makeBlockPtr2_eval]
  simp only [mul_ref u _ _ _ hs, mul_ref u _ _ _ hh, mul_ref u _ _ _ ht,
    List.mapM_cons, List.mapM_nil, Option.bind_some, Option.pure_def,
    Option.bind_eq_bind, Tile.scalar_data]
  rfl

theorem load_eval (p : Layout) (l : Launch) (r : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) (hu : ∀ off, u.undef r off = padding off)
    (hm : ∀ off, u.readMem r off = a off)
    (hp : u.regs .blockPtr [p.BT, p.BS] "p_s" = some (pointer p l r c)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_s") [0, 1]) .none .unspecified) u =
      some (inputTile p l a padding c) := by
  simp only [evalOp, evalOp_ref, hp]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [pointer, inputTile, Tile.mat, value, active, address,
    TileShape.indexToList, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real, decide_eq_true_eq]
  by_cases hi : c * p.BT + i.val < p.T <;>
    by_cases hj : l.feature * p.BS + j.val < p.S <;>
      simp only [hi, hj, hm, ite_true, ite_false, and_true, and_false, false_and,
        BlockState.paddingValue, BlockState.unspecifiedValue, pointer, hu]

theorem dot_value (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : Fin p.BT) (j : Fin p.BS) :
    (Tile.dot [] (lowerTriTile p.BT) (inputTile p l a padding c)).data (i,j,PUnit.unit) =
      some (chunkPrefix p l a padding c i j) := by
  have h := dotLowerTri_sum_if_some p.BT p.BS
    (fun k j => value p l a padding (c * p.BT + k.val) j) (fun _ _ => True) i j
  simpa only [inputTile, Tile.mat, ite_true, and_true, chunkPrefix] using h

theorem add_dot_eval (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (acc : Fin p.BS → ℝ) (u : BlockState)
    (ha : u.regs .real [p.BS] "b_z" = some (Tile.vec fun j => some (acc j)))
    (hm : u.regs .real [p.BT, p.BT] "m_s" = some (lowerTriTile p.BT))
    (hb : u.regs .real [p.BT, p.BS] "b_s" = some (inputTile p l a padding c)) :
    evalOp (.add .real (.consL (.consSame .nil))
      (.expandDim ⟨0, by simp⟩ (.ref .real [p.BS] "b_z"))
      (.dot (batch := []) (.ref .real [p.BT, p.BT] "m_s") (.ref .real [p.BT, p.BS] "b_s")
        {allowTF32 := some Bool.false})) u = some (outputTile p l a padding c acc) := by
  rw [evalOp_add]
  erw [evalOp_expandDim, evalOp_ref, ha, evalOp_dot, evalOp_ref, hm, evalOp_ref, hb]
  simp only [Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim,
    TileShape.dropInsertedIndex, Tile.vec, NumericDType.add, dot_value, outputTile,
    Tile.mat, WithBot.realAdd, Option.map₂_some_some]
  erw [dot_value]
  rfl

theorem coe_sum {n : Nat} (f : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e => (some (f e) : WithBot ℝ)) = some (∑ e : Fin n, f e) := by
  show (Finset.univ.sum fun e => ((f e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

theorem carry_eval (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (acc : Fin p.BS → ℝ) (u : BlockState)
    (ha : u.regs .real [p.BS] "b_z" = some (Tile.vec fun j => some (acc j)))
    (hb : u.regs .real [p.BT, p.BS] "b_s" = some (inputTile p l a padding c)) :
    evalOp (carryOp p Bool.false) u =
      some (Tile.vec fun j => some (nextCarry p l a padding c acc j)) := by
  rw [carryOp, if_neg Bool.false_ne_true, evalOp_add]
  erw [evalOp_ref, ha, evalOp_reduceSum, evalOp_ref, hb]
  simp only [Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  rintro ⟨j,⟨⟩⟩
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.vec,
    Tile.reduceSum_false, Tile.reduceSumDrop_data, NumericDType.add, inputTile, Tile.mat,
    TileShape.insertAxisIndex, nextCarry]
  erw [coe_sum]
  rfl

theorem store_eval (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT, p.BS] "p_z" = some (pointer p l o c))
    (hv : u.regs .real [p.BT, p.BS] "b_c" = some (outputTile p l a padding c acc)) :
    stepStmt (.store .real [p.BT, p.BS] (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_z") [0, 1])
      (.ref .real [p.BT, p.BS] "b_c") .none) u = some (scatter p l o a padding c acc u) := by
  simp only [stepStmt, evalOp_ref, hp, hv, Option.bind_some, Option.bind_eq_bind,
    Bool.true_and, pointer, TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets, decide_eq_true_eq]
  rfl

/-- Every statement of a chunk succeeds for arbitrary loaded input and incoming
carry. The final store remains before the source's nonnegative-index branch. -/
theorem executes (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (acc : Fin p.BS → ℝ) (u : BlockState) (hu : ∀ off, u.undef s off = padding off)
    (his : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hm : ∀ off, u.readMem s off = a off)
    (htri : u.regs .real [p.BT, p.BT] "m_s" = some (lowerTriTile p.BT))
    (hacc : u.regs .real [p.BS] "b_z" = some (Tile.vec fun j => some (acc j))) :
    stepStmts (loopBody p s o Bool.false) u = some (result p l s o a padding c acc u) := by
  unfold loopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l s c u his hih hit))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l o c _
    (by simp [his]) (by simp [hih]) (by simp [hit])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (load_eval p l s a padding c _ (by simpa using hu)
    (by intro off; simp [hm]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (add_dot_eval p l a padding c acc _
    (by simp [hacc]) (by simp [htri]) (by simp)))]
  rw [stepStmts.cons_some (store_eval p l o a padding c acc _ (by simp) (by simp))]
  change stepStmts [.ifThen _ _] (scatter p l o a padding c acc (beforeStore p l s o a padding c acc u)) = _
  have hc : evalOp (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
      (scatter p l o a padding c acc (beforeStore p l s o a padding c acc u)) = some (Tile.scalar Bool.true) := by
    have hr : (scatter p l o a padding c acc (beforeStore p l s o a padding c acc u)).regs .nat [] "i_t" =
        some (Tile.scalar c) := by simp [scatter, beforeStore, hit]
    rw [evalOp]
    simp only [evalOp_ref, hr, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro i
    change decide (0 ≤ c) = Bool.true
    simp
  rw [stepStmts, stepStmt, hc]
  simp only [Option.bind_some, Option.bind_eq_bind, Tile.scalar_data, ↓reduceIte]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (carry_eval p l a padding c acc _
    (by simp [scatter, beforeStore, hacc]) (by simp [scatter, beforeStore])))]
  simp only [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealStep

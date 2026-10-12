import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumProgram

/-! Whole cumulative-source execution with arbitrary unspecified real values.
The valid chunk-local prefix uses only preceding valid time lanes, so its
formula depends solely on initial input memory. All loads precede the store;
source/output aliasing is permitted. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumExecution
open VeriTile.Triton RealCumProgram
open ChunkCumsumVector.RealStep (pointer inputTile chunkPrefix address active)
open ChunkCumsumVector (lowerTriTile)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def tileActive (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) : Prop :=
  active p l (c*p.BT+i.1.val) i.2.1.val
instance (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) :
    Decidable (tileActive p l c i) := inferInstanceAs (Decidable (active _ _ _ _))

def offset (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) : Nat :=
  address p l (c*p.BT+i.1.val) i.2.1.val

def NoCollision (p : Layout) (l : Launch) (c : Nat) : Prop :=
  ∀ i j, tileActive p l c i → tileActive p l c j → offset p l c i = offset p l c j → i = j

def expected (p : Layout) (l : Launch) (c : Nat) (a : Nat → ℝ) (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Fin p.BT => k.val ≤ i.1.val),
    a (address p l (c*p.BT+k.val) i.2.1.val)

def initial (p : Layout) (u : BlockState) : BlockState :=
  let u1 := u.setReg "i_s" .nat [] (Tile.scalar (u.pids 0))
  let u2 := u1.setReg "i_t" .nat [] (Tile.scalar (u.pids 1))
  let u3 := u2.setReg "i_bh" .nat [] (Tile.scalar (u.pids 2))
  let u4 := u3.setReg "o_i" .nat [p.BT] ⟨fun i => i.1.val⟩
  u4.setReg "m_s" .real [p.BT,p.BT] (lowerTriTile p.BT)

def outputTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) : Tile .real [p.BT,p.BS] :=
  Tile.mat fun i j => some (chunkPrefix p l a padding c i j)

def prepared (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) : BlockState :=
  let u1 := u.setReg "p_s" .blockPtr [p.BT,p.BS] (pointer p l s c)
  let u2 := u1.setReg "p_o" .blockPtr [p.BT,p.BS] (pointer p l o c)
  let u3 := u2.setReg "b_s" .real [p.BT,p.BS] (inputTile p l a padding c)
  u3.setReg "b_o" .real [p.BT,p.BS] (outputTile p l a padding c)

def scatter (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT,p.BS]).foldl (fun st i =>
    if tileActive p l c i then st.writeMem o (offset p l c i) (chunkPrefix p l a padding c i.1 i.2.1)
    else st) u

def result (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) : BlockState :=
  scatter p l o a padding c (prepared p l s o a padding c u)

def footprint (p : Layout) (l : Launch) (c : Nat) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ i, tileActive p l c i ∧ cell.2 = offset p l c i

theorem initialization (p : Layout) (u : BlockState) :
    stepStmts (initCode p) u = some (initial p u) := by
  unfold initCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ = some
    (⟨fun i : TileIndex [p.BT] => i.1.val⟩ : Tile .nat [p.BT]) from by rw [evalOp]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ChunkCumsumVector.RealInit.triangular_eval p _ (by simp)))]
  rw [stepStmts.nil]
  rfl

theorem base_eval (p : Layout) (swapped : Bool) (u : BlockState) (head : Nat)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar head)) :
    evalOp (baseOp p swapped) u = some (Tile.scalar (head*p.headStride)) := by
  cases swapped
  all_goals simp only [baseOp,Bool.false_eq_true,↓reduceIte,evalOp_mul,evalOp_ref,hh,
    evalOp_constNat,Option.bind_some,Option.bind_eq_bind]
  · rfl
  · change some (Tile.scalar (dtype := .nat) (p.headStride*head)) = _
    rw [Nat.mul_comm]

theorem pointer_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (swapped : Bool)
    (u : BlockState) (hs : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (pointerOp p r swapped) u = some (pointer p l r c) := by
  rw [pointerOp,makeBlockPtr2_eval]
  simp only [base_eval p swapped u l.head hh,ChunkCumsumVector.RealStep.mul_ref u _ _ _ hs,
    ChunkCumsumVector.RealStep.mul_ref u _ _ _ ht,List.mapM_cons,List.mapM_nil,
    Option.bind_some,Option.pure_def,Option.bind_eq_bind,Tile.scalar_data]
  rfl

theorem dot_eval (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (u : BlockState)
    (hm : u.regs .real [p.BT,p.BT] "m_s" = some (lowerTriTile p.BT))
    (hb : u.regs .real [p.BT,p.BS] "b_s" = some (inputTile p l a padding c)) :
    evalOp (.dot (batch := []) (.ref .real [p.BT,p.BT] "m_s") (.ref .real [p.BT,p.BS] "b_s")
      {allowTF32 := some Bool.false}) u = some (outputTile p l a padding c) := by
  rw [evalOp_dot]
  erw [evalOp_ref,hm,evalOp_ref,hb]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  exact ChunkCumsumVector.RealStep.dot_value p l a padding c i j

theorem store_eval (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_o" = some (pointer p l o c))
    (hv : u.regs .real [p.BT,p.BS] "b_o" = some (outputTile p l a padding c)) :
    stepStmt (.store .real [p.BT,p.BS] (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_o") [0,1])
      (.ref .real [p.BT,p.BS] "b_o") .none) u = some (scatter p l o a padding c u) := by
  simp only [stepStmt,evalOp_ref,hp,hv,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq]
  rfl

theorem computes (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (swapped : Bool) (u : BlockState)
    (hu : ∀ off, u.undef s off = padding off)
    (hs : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hm : ∀ off, u.readMem s off = a off)
    (htri : u.regs .real [p.BT,p.BT] "m_s" = some (lowerTriTile p.BT)) :
    stepStmts (computeCode p s o swapped) u = some (result p l s o a padding c u) := by
  unfold computeCode
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l s c swapped u hs hh ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l o c swapped _
    (by simp [hs]) (by simp [hh]) (by simp [ht])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ChunkCumsumVector.RealStep.load_eval p l s a padding c _ (by simpa using hu)
      (by intro off; simp [hm]) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (dot_eval p l a padding c _ (by simp [htri]) (by simp)))]
  rw [stepStmts.cons_some (store_eval p l o a padding c _ (by simp) (by simp)),stepStmts.nil]
  rfl

theorem prefix_active (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : TileIndex [p.BT,p.BS]) (hi : tileActive p l c i) :
    chunkPrefix p l a padding c i.1 i.2.1 = expected p l c a i := by
  unfold chunkPrefix expected
  apply Finset.sum_congr rfl
  intro k hk
  have hk := (Finset.mem_filter.mp hk).2
  have ha : active p l (c*p.BT+k.val) i.2.1.val := ⟨by have := hi.1; omega,hi.2⟩
  simp only [ChunkCumsumVector.RealStep.value,ha,ite_true]

theorem scatter_frame (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) :
    BlockState.WriteWithin (footprint p l c o) u (scatter p l o a padding c u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,i,hi,rfl⟩

theorem source_executes (p : Layout) (s o : RegionName) (swapped : Bool)
    (dtype out_dtype : FloatDType) (u : BlockState)
    (hinj : NoCollision p (launch u) (chunk u)) :
    ∃ v, exec (RealCumSource.source swapped dtype out_dtype s o p.headStride p.timeStride
        p.channelStride p.T p.S p.BT p.BS).eraseDType.toAlgKernel u = some v ∧
      (∀ i, tileActive p (launch u) (chunk u) i →
        v.readMem o (offset p (launch u) (chunk u) i) = expected p (launch u) (chunk u) (u.readMem s) i) ∧
      BlockState.WriteWithin (footprint p (launch u) (chunk u) o) u v := by
  let pre := prepared p (launch u) s o (u.readMem s) (u.undef s) (chunk u) (initial p u)
  refine ⟨scatter p (launch u) o (u.readMem s) (u.undef s) (chunk u) pre,?_,?_,?_⟩
  · rw [exec,source_body,body,stepStmts.append_some (initialization p u)]
    exact computes p (launch u) s o (u.readMem s) (u.undef s) (chunk u) swapped (initial p u)
      (fun _ => rfl) (by simp [initial,launch]) (by simp [initial,launch])
      (by simp [initial,chunk]) (fun _ => rfl) (by simp [initial])
  · intro i hi
    have hr := BlockState.scatter_readback_prop_masked_nd_of_true (region := o) pre
      (offset p (launch u) (chunk u))
      (fun j => chunkPrefix p (launch u) (u.readMem s) (u.undef s) (chunk u) j.1 j.2.1)
      (tileActive p (launch u) (chunk u)) i hi (fun j hj he => hinj j i hj hi he)
    exact hr.trans (prefix_active p (launch u) (u.readMem s) (u.undef s) (chunk u) i hi)
  · exact scatter_frame p (launch u) o (u.readMem s) (u.undef s) (chunk u) pre

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumExecution

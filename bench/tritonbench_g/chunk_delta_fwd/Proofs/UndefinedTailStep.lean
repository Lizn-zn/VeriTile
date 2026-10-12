import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealStep

/-! One actual inner step with T=1 and 16×16 dot tiles. Real inputs are zero,
but omitted K/V padding is one. A separate T=16 array describes the loaded
tiles only; it never changes the source's T=1 pointer masks or output store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.UndefinedTailStep
open VeriTile.Triton RealProgram
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def p : Layout := ⟨16,16,1,16,16,1,256,16,1,1,16,16,16,16,16,16,1⟩
def wide : Layout := {p with T := 16}
def l : Launch := ⟨0,0,0⟩
def r : Regions := ⟨"k","d","v","v_new","h","h0","ht"⟩
def sample (off : Nat) : ℝ := if off < 16 then 0 else 1
def a : Inputs := ⟨sample,fun _ => 0,sample⟩
def zero : Inputs := ⟨fun _ => 0,fun _ => 0,fun _ => 0⟩
def initial : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun rg _ => if rg = r.key ∨ rg = r.value then 1 else 0

def beforeStore (s : BlockState) : BlockState :=
  let s1 := s.setReg "p_k" .blockPtr [16,16] (keyPointer p l r.key 0 0)
  let s2 := s1.setReg "p_d" .blockPtr [16,16] (deltaPointer p l r.delta 0 0)
  let s3 := s2.setReg "p_v" .blockPtr [16,16] (valuePointer p l r.value 0 0)
  let s4 := s3.setReg "p_v_new" .blockPtr [16,16] (valuePointer p l r.corrected 0 0)
  let s5 := s4.setReg "b_k" .real [16,16] (keyTile wide l a 0 0)
  let s6 := s5.setReg "b_d" .real [16,16] (deltaTile wide l a 0 0)
  let s7 := s6.setReg "b_v" .real [16,16] (valueTile wide l a 0 0)
  s7.setReg "b_v" .real [16,16] (correctedTile wide l a 0 0 (fun _ _ => 0))

def result (s : BlockState) : BlockState :=
  (scatter p l r.corrected zero 0 0 (fun _ _ => 0) (beforeStore s)).setReg
    "b_h_cumsum" .real [16,16] (stateTile wide (nextAccum wide l a 0 0 (fun _ _ => 0) (fun _ _ => 0)))

theorem key_load (s : BlockState) (hu : s.undef r.key = fun _ => 1)
    (hm : ∀ off, s.readMem r.key off = 0)
    (hp : s.regs .blockPtr [16,16] "p_k" = some (keyPointer p l r.key 0 0)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [16,16] "p_k") [0,1]) .none .unspecified) s =
      some (keyTile wide l a 0 0) := by
  simp only [evalOp, evalOp_ref, hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨e,c,⟨⟩⟩
  have h : e.val + c.val*16 < 16 ↔ c.val = 0 := by omega
  simp [keyPointer,keyTile,keyValue,keyIndex,keyAddress,timeBase,p,wide,l,a,sample,Tile.mat,
    TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real,BlockState.paddingValue,BlockState.unspecifiedValue,
    hm,hu,e.isLt,c.isLt,Nat.lt_one_iff,h]
  split <;> rfl

theorem delta_load (s : BlockState) (hu : s.undef r.delta = fun _ => 0)
    (hm : ∀ off, s.readMem r.delta off = 0)
    (hp : s.regs .blockPtr [16,16] "p_d" = some (deltaPointer p l r.delta 0 0)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [16,16] "p_d") [0,1]) .none .unspecified) s =
      some (deltaTile wide l a 0 0) := by
  simp only [evalOp,evalOp_ref,hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨c,e,⟨⟩⟩
  simp [deltaPointer,deltaTile,deltaValue,keyIndex,timeBase,p,wide,l,a,Tile.mat,
    TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real,BlockState.paddingValue,BlockState.unspecifiedValue,hm,hu]

theorem value_load (s : BlockState) (hu : s.undef r.value = fun _ => 1)
    (hm : ∀ off, s.readMem r.value off = 0)
    (hp : s.regs .blockPtr [16,16] "p_v" = some (valuePointer p l r.value 0 0)) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [16,16] "p_v") [0,1]) .none .unspecified) s =
      some (valueTile wide l a 0 0) := by
  simp only [evalOp,evalOp_ref,hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨c,j,⟨⟩⟩
  have h : c.val*16+j.val < 16 ↔ c.val = 0 := by omega
  simp [valuePointer,valueTile,inputValue,activeValue,valueIndex,valueAddress,timeBase,p,wide,l,a,sample,Tile.mat,
    TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    BlockState.readMemValue_real,BlockState.paddingValue,BlockState.unspecifiedValue,
    hm,hu,c.isLt,j.isLt,Nat.lt_one_iff,h]
  split <;> rfl

theorem store_eval (s : BlockState)
    (hv : s.regs .real [16,16] "b_v" = some (correctedTile wide l a 0 0 (fun _ _ => 0)))
    (hp : s.regs .blockPtr [16,16] "p_v_new" = some (valuePointer p l r.corrected 0 0)) :
    stepStmt (.store .real [16,16] (.blockPtr (.ref .blockPtr [16,16] "p_v_new") [0,1])
      (.ref .real [16,16] "b_v") .none) s = some (scatter p l r.corrected zero 0 0 (fun _ _ => 0) s) := by
  unfold stepStmt scatter
  simp only [evalOp_ref,hv,hp,Option.bind,Option.map]
  apply congrArg some
  apply congrArg (fun f => List.foldl f s (TileShape.allIndices [16,16]))
  funext st idx
  rcases idx with ⟨c,j,⟨⟩⟩
  by_cases hc : c.val = 0
  · simp [valuePointer,correctedTile,correctedValue,Tile.mat,inputValue,activeValue,valueIndex,
      valueAddress,timeBase,deltaValue,keyIndex,p,wide,l,a,zero,sample,hc,j.isLt,
      TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
      BlockState.writeMemTyped_real]
  · simp [valuePointer,activeValue,valueIndex,timeBase,p,l,TileShape.indexToList,
      BlockPtr.inBounds_2d_offsets,Nat.lt_one_iff,hc]

theorem executes (u : BlockState) (hu : u.undef = initial.undef)
    (hik : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hiv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hih : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hit : u.regs .nat [] "i_t" = some (Tile.scalar 0))
    (hic : u.regs .nat [] "i_c" = some (Tile.scalar 0))
    (hkey : ∀ off, u.readMem r.key off = 0)
    (hdelta : ∀ off, u.readMem r.delta off = 0)
    (hvalue : ∀ off, u.readMem r.value off = 0)
    (hstate : u.regs .real [16,16] "b_h" = some (stateTile wide (fun _ _ => 0)))
    (hacc : u.regs .real [16,16] "b_h_cumsum" = some (stateTile wide (fun _ _ => 0))) :
    stepStmts (innerBody p r) u = some (result u) := by
  unfold innerBody chunkDeltaInnerBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.key u _ _ _ [p.K,p.T] [p.BK,p.BC] [p.keyDim,p.keyTime]
      (l.head * p.keyHead) (l.key * p.BK) (timeBase p 0 0)
      (mulConst_eval u "i_bh" l.head p.keyHead hih)
      (mulConst_eval u "i_k" l.key p.BK hik)
      (addMulMul_eval u "i_t" "i_c" 0 p.BT 0 p.BC hit hic)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.delta _ _ _ _ [p.T,p.K] [p.BC,p.BK] [p.keyTime,p.keyDim]
      (l.head * p.keyHead) (timeBase p 0 0) (l.key * p.BK)
      (mulConst_eval _ "i_bh" l.head p.keyHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" 0 p.BT 0 p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_k" l.key p.BK (by simp [hik]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p 0 0) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" 0 p.BT 0 p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.corrected _ _ _ _ [p.T,p.V] [p.BC,p.BV] [p.valueTime,p.valueDim]
      (l.head * p.valueHead) (timeBase p 0 0) (l.value * p.BV)
      (mulConst_eval _ "i_bh" l.head p.valueHead (by simp [hih]))
      (addMulMul_eval _ "i_t" "i_c" 0 p.BT 0 p.BC (by simp [hit]) (by simp [hic]))
      (mulConst_eval _ "i_v" l.value p.BV (by simp [hiv]))))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (key_load _ (by simpa [initial,r] using congrFun hu r.key)
    (by intro off; simp [hkey]) (by simp [keyPointer,p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (delta_load _ (by simpa [initial,r] using congrFun hu r.delta)
    (by intro off; simp [hdelta]) (by simp [deltaPointer,p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (value_load _ (by simpa [initial,r] using congrFun hu r.value)
    (by intro off; simp [hvalue]) (by simp [valuePointer,p])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (RealStep.correction wide l a 0 0 (fun _ _ => 0) _
    (by simp [wide,p]) (by simp [wide,p]) (by simp [wide,p,hstate])))]
  erw [stepStmts.cons_some (store_eval _
    (by simp [wide,p]) (by simp [valuePointer,p]))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (RealStep.accumulation wide l a 0 0 (fun _ _ => 0) (fun _ _ => 0) _
    (by simp [scatter,wide,p,hacc]) (by simp [scatter,wide,p]) (by simp [scatter,wide,p])))]
  rw [stepStmts.nil]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.UndefinedTailStep

import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSafety
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputCandidate
import VeriTile.Meta.Disprove

/-! Removing the zero-unspecified-value condition changes valid stored outputs.
This complete-source real execution uses T=1, K=17, V=16 and dot tiles
BT=16, BK=32, BV=16. Valid Q/H/V/A cells are one, G is zero, and unspecified
Q/H/V/A lanes may also be one. The original and candidate then store 33;
the independent zero-extended contraction is 18. Actual access bounds are
checked. This is a real-semantics counterexample, not GPU or FP evidence. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.UndefinedOutput
open VeriTile.Triton RealOutput
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def constant (m n : Nat) (v : ℝ) : Tile .real [m,n] := Tile.mat fun _ _ => some v

theorem load_constant (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BR BS strideT strideS rowOff colOff : Nat) (v : ℝ)
    (hreg : s.regs .blockPtr [BR,BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows,cols] [BR,BS] [strideT,strideS] [rowOff,colOff]⟩)
    (hm : ∀ off, s.readMem rg off = v) (hu : ∀ off, s.undef rg off = v) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr [BR,BS] name) [0,1]) .none .unspecified) s =
      some (constant BR BS v) := by
  simp only [evalOp,evalOp_ref,hreg,bind,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [constant,Tile.mat,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,BlockState.readMemValue_real,BlockState.paddingValue,
    BlockState.unspecifiedValue,hm,hu]

theorem dot_constant (m k n : Nat) :
    Tile.dot [] (constant m k 1) (constant k n 1) = constant m n k := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [Tile.dot_nil_data]
  simp only [constant,Tile.mat,Option.map₂_some_some,one_mul]
  change (∑ _ : Fin k, ((1 : ℝ) : WithBot ℝ)) = ((k : ℝ) : WithBot ℝ)
  rw [← WithBot.coe_sum]
  simp

def pointer (rg : RegionName) (rows cols br bc stride : Nat) : Tile .blockPtr [br,bc] :=
  ⟨fun _ => BlockPtr.mk rg 0 [rows,cols] [br,bc] [stride,1] [0,0]⟩
def pointed (s : BlockState) :=
  ((s.setReg "p_q" .blockPtr [16,32] (pointer "q" 1 17 16 32 17)).setReg
    "p_g" .blockPtr [16,32] (pointer "g" 1 17 16 32 17)).setReg
    "p_h" .blockPtr [32,16] (pointer "h" 17 16 32 16 16)
def loaded (s : BlockState) :=
  (((((pointed s).setReg "b_q" .real [16,32] (constant 16 32 1)).setReg
    "b_q" .real [16,32] (constant 16 32 1)).setReg
    "b_g" .real [16,32] (constant 16 32 0)).setReg
    "b_qg" .real [16,32] (constant 16 32 1)).setReg
    "b_h" .real [32,16] (constant 32 16 1)
def result (s : BlockState) := (loaded s).setReg "b_o" .real [16,16] (constant 16 16 32)

theorem loop_body (s : BlockState)
    (hq : ∀ off, s.readMem "q" off = 1) (huq : ∀ off, s.undef "q" off = 1)
    (hg : ∀ off, s.readMem "g" off = 0) (hug : ∀ off, s.undef "g" off = 0)
    (hh : ∀ off, s.readMem "h" off = 1) (huh : ∀ off, s.undef "h" off = 1)
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar 0))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar 0))
    (hb : s.regs .nat [] "i_bh" = some (Tile.scalar 0))
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar 0))
    (ho : s.regs .real [16,16] "b_o" = some (constant 16 16 0)) :
    stepStmts (cgfLoopBody "g" "h" "q" 17 17 272 16 1 1 17 16 16 32 16) s = some (result s) := by
  unfold cgfLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "q" s _ _ _ [1,17] [16,32] [17,1] 0 0 0
      (by simpa using cgf_mulConst_eval s "i_bh" 0 17 hb)
      (by simpa using cgf_mulConst_eval s "i_t" 0 16 ht)
      (by simpa using cgf_mulConst_eval s "i_k" 0 32 hk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "g" _ _ _ _ [1,17] [16,32] [17,1] 0 0 0
      (by simpa using cgf_mulConst_eval _ "i_bh" 0 17 (by simpa using hb))
      (by simpa using cgf_mulConst_eval _ "i_t" 0 16 (by simpa using ht))
      (by simpa using cgf_mulConst_eval _ "i_k" 0 32 (by simpa using hk))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "h" _ _ _ _ [17,16] [32,16] [16,1] 0 0 0
      (by simpa using cgf_hBase_eval _ 272 17 16 0 0 (by simpa using hb) (by simpa using ht))
      (by simpa using cgf_mulConst_eval _ "i_k" 0 32 (by simpa using hk))
      (by simpa using cgf_mulConst_eval _ "i_v" 0 16 (by simpa using hv))))]
  change stepStmts (.assign _ _ "b_q" _ :: _) (pointed s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_constant "q" _ "p_q" 0 1 17 16 32 17 1 0 0 1 (by simp [pointed,pointer])
      (by simpa [pointed] using hq) (by simpa [pointed] using huq)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.mul .real .scalarR (.ref .real [16,32] "b_q") (.const 1)) _ = some (constant 16 32 1) from by
      rw [evalOp_mul,evalOp_ref,BlockState.setReg_same,evalOp_const]
      apply congrArg some
      apply Tile.ext
      intro ix
      simp [-Tile.scalar_eta,constant,Tile.mat,Tile.bop,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_constant "g" _ "p_g" 0 1 17 16 32 17 1 0 0 0 (by simp [pointed,pointer])
      (by simpa [pointed] using hg) (by simpa [pointed] using hug)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (.mul .real (.consSame (.consSame .nil)) (.ref .real [16,32] "b_q") (.exp (.ref .real [16,32] "b_g"))) _ =
      some (constant 16 32 1) from by
      simp only [evalOp,evalOp_ref,BlockState.setReg_same,BlockState.setReg_ne_name,ne_eq,
        String.reduceEq,not_false_eq_true,Option.bind_eq_bind,Option.bind_some]
      apply congrArg some
      apply Tile.ext
      intro ix
      simp [constant,Tile.mat,Tile.bop,Tile.uop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,WithBot.realExp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_constant "h" _ "p_h" 0 17 16 32 16 16 1 0 0 1 (by simp [pointed,pointer])
      (by simpa [pointed] using hh) (by simpa [pointed] using huh)))]
  change stepStmts [.ifThen _ _] (loaded s) = _
  have hc : evalOp (.ge .nat .nil (.ref .nat [] "i_k") (.constNat 0)) (loaded s) = some (Tile.scalar Bool.true) := by
    simp only [evalOp,evalOp_ref,loaded,pointed,BlockState.setReg_ne_name,ne_eq,String.reduceEq,
      not_false_eq_true,hk,Option.bind_eq_bind,Option.bind_some]
    rfl
  have hd : evalOp (.dot (batch := []) (.ref .real [16,32] "b_qg") (.ref .real [32,16] "b_h")) (loaded s) =
      some (Tile.dot [] (constant 16 32 1) (constant 32 16 1)) := by
    erw [evalOp_dot]
    simp [evalOp_ref,loaded]
  simp only [stepStmts,stepStmt]
  rw [hc]
  simp only [Tile.scalar,Option.bind_some,Option.bind_eq_bind,↓reduceIte]
  rw [evalOp_add]
  erw [hd]
  rw [evalOp_ref]
  have hb_o : (loaded s).regs .real [16,16] "b_o" = some (constant 16 16 0) := by simp [loaded,pointed,ho]
  rw [hb_o]
  simp only [Option.bind_some,Option.bind_eq_bind]
  change some ((loaded s).setReg "b_o" .real [16,16]
    (Tile.bop NumericDType.real.add (.consSame (.consSame .nil)) (constant 16 16 0)
      (Tile.dot [] (constant 16 32 1) (constant 32 16 1)))) = _
  rw [dot_constant]
  congr 2
  apply Tile.ext
  intro ix
  simp [constant,Tile.mat,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.add]

def causal : Tile .real [16,16] := Tile.mat fun i j => some (if j.val ≤ i.val then (1 : ℝ) else 0)
def value (i : Fin 16) : ℝ := 32 + ∑ e : Fin 16, if e.val ≤ i.val then (1 : ℝ) else 0
def output : Tile .real [16,16] := Tile.mat fun i _ => some (value i)

theorem causal_eval (s : BlockState)
    (hm : s.regs .bool [16,16] "m_s" = some (cgfMsTile 16))
    (ha : s.regs .real [16,16] "b_A" = some (constant 16 16 1)) :
    evalOp (.where (.ref .bool [16,16] "m_s") (.ref .real [16,16] "b_A")
      (.broadcast (.const 0.0) [16,16])) s = some causal := by
  simp only [evalOp,evalOp_ref,hm,ha,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [causal,constant,Tile.mat,Tile.select,cgfMsTile]
  split <;> norm_num

theorem final_add (s : BlockState)
    (ho : s.regs .real [16,16] "b_o" = some (constant 16 16 32))
    (ha : s.regs .real [16,16] "b_A" = some causal)
    (hv : s.regs .real [16,16] "b_v" = some (constant 16 16 1)) :
    evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [16,16] "b_o")
      (.dot (options := {allowTF32 := some Bool.false}) (batch := [])
        (.ref .real [16,16] "b_A") (.ref .real [16,16] "b_v"))) s = some output := by
  rw [evalOp_add,evalOp_ref,ho]
  erw [evalOp_dot,evalOp_ref,ha,evalOp_ref,hv]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,constant,Tile.mat,output,value,
    Tile.dot_nil_data,causal,Option.map₂_some_some,mul_one,NumericDType.add]
  erw [Tile.dot_nil_data]
  simp only [Option.map₂_some_some,mul_one]
  have hs := (WithBot.coe_sum Finset.univ (fun e : Fin 16 => if e.val ≤ i.val then (1 : ℝ) else 0)).symm
  erw [hs]
  rfl

def postPointers (s : BlockState) :=
  ((s.setReg "p_v" .blockPtr [16,16] (pointer "v" 1 16 16 16 16)).setReg
    "p_o" .blockPtr [16,16] (pointer "o" 1 16 16 16 16)).setReg
    "p_A" .blockPtr [16,16] (pointer "A" 1 16 16 16 16)
def postCalculated (s : BlockState) :=
  ((((postPointers s).setReg "b_v" .real [16,16] (constant 16 16 1)).setReg
    "b_A" .real [16,16] (constant 16 16 1)).setReg "b_A" .real [16,16] causal).setReg
    "b_o" .real [16,16] output

def scatter (s : BlockState) :=
  (TileShape.allIndices [16,16]).foldl (fun u ix =>
    if ix.1.val < 1 then u.writeMem "o" (ix.1.val*16+ix.2.1.val) (value ix.1) else u) s

theorem store_executes (s : BlockState)
    (hp : s.regs .blockPtr [16,16] "p_o" = some (pointer "o" 1 16 16 16 16))
    (hv : s.regs .real [16,16] "b_o" = some output) :
    stepStmt (.store .real [16,16] (.blockPtr (.ref .blockPtr [16,16] "p_o") [0,1])
      (.ref .real [16,16] "b_o") .none) s = some (scatter s) := by
  simp only [stepStmt,evalOp_ref,hp,hv,scatter]
  apply congrArg some
  apply congrArg (fun f => (TileShape.allIndices [16,16]).foldl f s)
  funext u ix
  rcases ix with ⟨i,j,⟨⟩⟩
  simp [pointer,TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,j.isLt,
    BlockState.writeMemTyped_real,output,Tile.mat]

theorem post_body (s : BlockState)
    (hv : ∀ off, s.readMem "v" off = 1) (huv : ∀ off, s.undef "v" off = 1)
    (ha : ∀ off, s.readMem "A" off = 1) (hua : ∀ off, s.undef "A" off = 1)
    (hiv : s.regs .nat [] "i_v" = some (Tile.scalar 0))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar 0))
    (hib : s.regs .nat [] "i_bh" = some (Tile.scalar 0))
    (hms : s.regs .bool [16,16] "m_s" = some (cgfMsTile 16))
    (hbo : s.regs .real [16,16] "b_o" = some (constant 16 16 32)) :
    stepStmts (cgfPostLoop "v" "o" "A" 16 16 1 16 16 16) s = some (scatter (postCalculated s)) := by
  unfold cgfPostLoop
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "v" s _ _ _ [1,16] [16,16] [16,1] 0 0 0
      (by simpa using cgf_mulConst_eval s "i_bh" 0 16 hib)
      (by simpa using cgf_mulConst_eval s "i_t" 0 16 hit)
      (by simpa using cgf_mulConst_eval s "i_v" 0 16 hiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "o" _ _ _ _ [1,16] [16,16] [16,1] 0 0 0
      (by simpa using cgf_mulConst_eval _ "i_bh" 0 16 (by simpa using hib))
      (by simpa using cgf_mulConst_eval _ "i_t" 0 16 (by simpa using hit))
      (by simpa using cgf_mulConst_eval _ "i_v" 0 16 (by simpa using hiv))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval "A" _ _ _ _ [1,16] [16,16] [16,1] 0 0 0
      (by simpa using cgf_aBase_eval _ 1 16 0 (by simpa using hib))
      (by simpa using cgf_mulConst_eval _ "i_t" 0 16 (by simpa using hit))
      (by rw [evalOp_constNat])))]
  change stepStmts (.assign _ _ "b_v" _ :: _) (postPointers s) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_constant "v" _ "p_v" 0 1 16 16 16 16 1 0 0 1 (by simp [postPointers,pointer])
      (by simpa [postPointers] using hv) (by simpa [postPointers] using huv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_constant "A" _ "p_A" 0 1 16 16 16 16 1 0 0 1 (by simp [postPointers,pointer])
      (by simpa [postPointers] using ha) (by simpa [postPointers] using hua)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (causal_eval _ (by simp [postPointers,hms]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (final_add _ (by simp [postPointers,hbo]) (by simp) (by simp)))]
  change stepStmts [_] (postCalculated s) = _
  rw [stepStmts.cons_some (store_executes _ (by simp [postCalculated,postPointers]) (by simp [postCalculated])),stepStmts.nil]


def initial : BlockState where
  mem := fun r _ => MemCell.real (if r = "g" then 0 else 1)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun r _ => if r = "g" then 0 else 1

def kernel := chunk_gla_fwd_o_surface "q" "v" "g" "h" "o" "A" 17 17 16 16 272 16 1 1 17 16 16 32 16

theorem scatter_value (s : BlockState) : (scatter s).readMem "o" 0 = 33 := by
  have hh : (scatter s).readMem "o" 0 = value 0 := by
    apply BlockState.scatter_readback_prop_masked_nd_of_true (shape := [16,16]) s
      (fun ix => ix.1.val*16+ix.2.1.val) (fun ix => value ix.1) (fun ix => ix.1.val < 1)
      (0,0,PUnit.unit) (by decide)
    rintro ⟨i,j,⟨⟩⟩ hi he
    change i.val < 1 at hi
    change i.val*16+j.val = 0 at he
    have hi0 : i = 0 := Fin.ext (by omega)
    have hj0 : j = 0 := Fin.ext (by omega)
    subst i
    subst j
    rfl
  rw [hh]
  norm_num [value,Fin.sum_univ_succ]

theorem kernel_executes : ∃ sf, exec kernel.toAlgKernel initial = some sf ∧ sf.readMem "o" 0 = 33 := by
  obtain ⟨s,hpre,hm,hp,hiv,hit,hib,hms,hbo,hu⟩ := cgfPreLoop_run initial 16 16
  let entry := s.setReg "i_k" .nat [] (Tile.scalar 0)
  have he : stepStmts (cgfLoopBody "g" "h" "q" 17 17 272 16 1 1 17 16 16 32 16) entry = some (result entry) := by
    apply loop_body
    all_goals simp [entry,BlockState.readMem,hm,hu,initial,hiv,hit,hib,hbo,constant,Tile.mat,Region.mk.injEq]
  have hl : stepStmt (.forRangeDyn "i_k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat 17) (.constNat 32)) (.constNat 1)) (.constNat 32))
      (.constNat 1) (cgfLoopBody "g" "h" "q" 17 17 272 16 1 1 17 16 16 32 16)) s = some (result entry) := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp,Option.bind_eq_bind,Option.bind_some]
    change stepForRangeAux "i_k" 0 1 1 (cgfLoopBody "g" "h" "q" 17 17 272 16 1 1 17 16 16 32 16) s = _
    rw [stepForRangeAux]
    simp only [show (1 : Nat) ≠ 0 from by decide,Nat.reduceLT,↓reduceIte,he,bind,Option.bind_some,Nat.reduceAdd]
    rw [he]
    change stepForRangeAux "i_k" 1 1 1 (cgfLoopBody "g" "h" "q" 17 17 272 16 1 1 17 16 16 32 16) (result entry) = _
    rw [stepForRangeAux]
    rfl
  have hpost : stepStmts (cgfPostLoop "v" "o" "A" 16 16 1 16 16 16) (result entry) =
      some (scatter (postCalculated (result entry))) := by
    apply post_body
    all_goals simp [result,loaded,pointed,entry,BlockState.readMem,hm,hu,initial,hiv,hit,hib,hms,Region.mk.injEq]
  refine ⟨scatter (postCalculated (result entry)),?_,scatter_value _⟩
  rw [exec,kernel,cgf_body_eq,List.append_assoc,stepStmts.append_some hpre]
  rw [List.singleton_append,stepStmts.cons_some hl,hpost]


def extent (r : RegionName) : Nat :=
  if r = "q" ∨ r = "g" then 17 else if r = "h" then 272
  else if r = "v" ∨ r = "o" ∨ r = "A" then 16 else 0

theorem bounds : RealOutputSafety.Bounds initial "q" "v" "g" "h" "o" "A"
    17 17 16 16 272 16 1 17 16 16 32 16 extent := by
  constructor
  · intro kb hk
    have hk0 : kb = 0 := by norm_num [numKB] at hk; omega
    subst kb
    constructor
    all_goals simp [RealOutputSafety.MatrixBounds,initial,extent,Region.mk.injEq]
    all_goals intros; omega
  · constructor
    all_goals simp [RealOutputSafety.MatrixBounds,initial,extent,Region.mk.injEq]
    all_goals intros; omega

theorem mathematical_value :
    cgfOutput initial "q" "v" "g" "h" "A" 17 17 16 16 272 16 1 1 17 16 16 32 16 0 0 = 18 := by
  norm_num [cgfOutput,numKB,Fin.sum_univ_succ,qgElem,hGuarded,vGuarded,aMasked,tIndex,kIndex,vIndex,
    qElem,gElem,hElem,vElem,aElem,initial,BlockState.readMem,Region.mk.injEq]
  norm_num [show ("h" : String) ≠ "g" from by decide,
    show ("q" : String) ≠ "g" from by decide,
    show ("v" : String) ≠ "g" from by decide,
    show ("A" : String) ≠ "g" from by decide]

/-- Precision erasure connects the witness to all supported storage selections
and both source orders; no unspecified-value premise is used. -/
theorem source_executes (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (swapped : Bool) : ∃ sf,
    exec (TypedOutput.source query value gate swapped dtype out_dtype "q" "v" "g" "h" "o" "A"
      17 17 16 16 272 16 1 1 17 16 16 32 16).eraseDType.toAlgKernel initial = some sf ∧
    sf.readMem "o" 0 = 33 := by
  obtain ⟨sf,he,hv⟩ := kernel_executes
  have ht : exec (TypedOutput.source query value gate Bool.false dtype out_dtype "q" "v" "g" "h" "o" "A"
      17 17 16 16 272 16 1 1 17 16 16 32 16).eraseDType.toAlgKernel initial = some sf := by
    rw [exec,RealOutputSource.storage_erasure]
    exact he
  refine ⟨sf,?_,hv⟩
  cases swapped
  · exact ht
  · exact RealOutputCandidate.candidate_executes_like query value gate dtype out_dtype "q" "v" "g" "h" "o" "A"
      17 17 16 16 272 16 1 1 17 16 16 32 16 initial sf ht

/-- The zero-extended real formula cannot hold unconditionally, even with
valid active-access bounds and an ordinary first-tile launch. -/
refutation output_without_zero_fill (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (swapped : Bool) :
    ∀ s t : BlockState,
      RealOutputSafety.Bounds s "q" "v" "g" "h" "o" "A"
        17 17 16 16 272 16 1 17 16 16 32 16 extent →
      s.pids = (fun _ => 0) →
      exec (TypedOutput.source query value gate swapped dtype out_dtype "q" "v" "g" "h" "o" "A"
        17 17 16 16 272 16 1 1 17 16 16 32 16).eraseDType.toAlgKernel s = some t →
      t.readMem "o" 0 = cgfOutput s "q" "v" "g" "h" "A" 17 17 16 16 272 16 1 1 17 16 16 32 16 0 0 := by
  intro h
  obtain ⟨sf,he,hv⟩ := source_executes query value gate dtype out_dtype swapped
  have hc := h initial sf bounds rfl he
  rw [hv,mathematical_value] at hc
  norm_num at hc

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.UndefinedOutput

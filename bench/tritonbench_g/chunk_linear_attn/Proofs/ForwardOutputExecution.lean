import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardOutputProgram

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
open ChunkGlaSimple.Real (msGe_op_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def iteration (p : Layout) (r : Regions) : List Stmt :=
  linearOutputLoopBody r.query r.key r.value r.history p.keyHead p.keyTime p.keyDimension p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV

def tripCount (p : Layout) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK)
def loop (p : Layout) (r : Regions) : Stmt :=
  .forRangeDyn "i_k" (.constNat 0) (tripCount p) (.constNat 1) (iteration p r)

structure Holds (p : Layout) (r : Regions) (s0 : BlockState) (n : Nat) (u : BlockState) : Prop where
  limit : n ≤ numBlocks p
  memory : u.mem = s0.mem
  padding : u.undef = s0.undef
  pids : u.pids = s0.pids
  valueReg : u.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 0))
  timeReg : u.regs .nat [] "i_t" = some (Tile.scalar (s0.pids 1))
  headReg : u.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2))
  maskReg : u.regs .bool [p.BT,p.BT] "m_s" = some (maskTile p)
  outputReg : u.regs .real [p.BT,p.BV] "b_o" = some (carryO p (launch s0) (inputs r s0) n)
  scoreReg : u.regs .real [p.BT,p.BT] "b_s" = some (carryS p (launch s0) (inputs r s0) n)

theorem loop_step (p : Layout) (r : Regions) (s0 : BlockState)
    (hu : s0.undef = fun _ _ => 0) (n : Nat) (u : BlockState)
    (hn : n < numBlocks p) (hi : Holds p r s0 n u) :
    ∃ v, stepStmts (iteration p r) (u.setReg "i_k" .nat [] (Tile.scalar n)) = some v ∧
      Holds p r s0 (n+1) v := by
  let t := u.setReg "i_k" .nat [] (Tile.scalar n)
  have ht_l : launch t = launch s0 := by simp [launch,t,hi.pids]
  have ht_a : inputs r t = inputs r s0 := by
    change inputs r u = inputs r s0
    unfold inputs
    congr 1 <;> funext off <;> simp only [BlockState.readMem,hi.memory]
  obtain ⟨v,he,hp,hm,hpad,hv,ht,hh,hmask,ho,hs⟩ :=
    linearOutputLoopBody_steps r.query r.key r.value r.history p.keyHead p.keyTime p.keyDimension p.historyHead p.historyKey
      p.T p.K p.V p.BT p.BK p.BV n t (carryO p (launch s0) (inputs r s0) n)
      (carryS p (launch s0) (inputs r s0) n)
      (by simpa [t,hi.pids] using hi.valueReg) (by simpa [t,hi.pids] using hi.timeReg)
      (by simpa [t,hi.pids] using hi.headReg) (by simp [t])
      (by simpa [t] using hi.outputReg) (by simpa [t] using hi.scoreReg)
      (show u.undef = fun _ _ => 0 from hi.padding.trans hu)
  refine ⟨v,he,⟨Nat.succ_le_of_lt hn,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · exact hm.trans hi.memory
  · exact hpad.trans hi.padding
  · exact hp.trans hi.pids
  · simpa [t,hi.pids] using hv
  · simpa [t,hi.pids] using ht
  · simpa [t,hi.pids] using hh
  · simpa [t,hi.maskReg] using hmask
  · rw [ho,query_tile p r t n,history_tile p r t n,ht_l,ht_a,output_update]
  · rw [hs,query_tile p r t n,key_tile p r t n,ht_l,ht_a,score_update]

theorem loop_executes (p : Layout) (r : Regions) (s0 : BlockState)
    (hu : s0.undef = fun _ _ => 0) (u : BlockState) (hi : Holds p r s0 0 u) :
    ∃ v, stepStmt (loop p r) u = some v ∧ Holds p r s0 (numBlocks p) v := by
  have htrip : evalOp (tripCount p) u = some (Tile.scalar (numBlocks p)) := by
    simp only [tripCount,evalOp_div,evalOp_sub,evalOp_add,evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
    rfl
  obtain ⟨n,v,he,hn,hv⟩ := forRangeDyn_inv (idx := "i_k") (start := 0) (stop := numBlocks p) (step := 1)
    (P := Holds p r s0) (body := iteration p r) (evalOp_constNat 0 u) htrip (evalOp_constNat 1 u)
    (by decide) hi (loop_step p r s0 hu)
  have hn' : n = numBlocks p := Nat.le_antisymm hv.limit hn
  subst n
  exact ⟨v,he,hv⟩

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
open ChunkGlaSimple.Real (msGe_op_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def setup (p : Layout) : List Stmt :=
  [.assign .nat [] "i_v" (.programId 0),
   .assign .nat [] "i_t" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .bool [p.BT,p.BT] "m_s"
      (.ge .nat (.consR (.consL .nil))
        (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BT] "o_i"))
        (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BT] "o_i"))),
   .assign .real [p.BT,p.BV] "b_o" (.full [p.BT,p.BV] (.const 0)),
   .assign .real [p.BT,p.BT] "b_s" (.full [p.BT,p.BT] (.const 0))]

def start (p : Layout) (s : BlockState) : BlockState :=
  ((((((s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))).setReg "i_t" .nat [] (Tile.scalar (s.pids 1))).setReg
    "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg "o_i" .nat [p.BT] (Tile.vec fun i => i.val)).setReg
    "m_s" .bool [p.BT,p.BT] (maskTile p)).setReg
    "b_o" .real [p.BT,p.BV] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV])).setReg
    "b_s" .real [p.BT,p.BT] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])

theorem setup_executes (p : Layout) (s : BlockState) :
    stepStmts (setup p) s = some (start p s) := by
  unfold setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.arange p.BT) _ =
    some (Tile.vec fun i => i.val) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msGe_op_eval _ p.BT (by simp) (by simp) "o_i" (Tile.vec fun i => i.val) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BV] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BV]) from by simp [evalOp]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BT] (.const 0)) _ =
    some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT]) from by simp [evalOp])),stepStmts.nil]
  rfl

theorem start_holds (p : Layout) (r : Regions) (s : BlockState) : Holds p r s 0 (start p s) := by
  constructor
  · exact Nat.zero_le _
  · rfl
  · rfl
  · rfl
  · simp [start]
  · simp [start]
  · simp [start]
  · simp [start]
  · simp [start,carryO,Tile.mat,qhSum]
  · simp [start,carryS,Tile.mat,qkSum]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
open ChunkGlaSimple.Real (whereMask_op_eval makeBlockPtr_2d_eval load_bp_2d_ref mulConst_eval accScale_op_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def tail (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  (linearOutputFront r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV scale).drop 8

def outputPointer (p : Layout) (l : Launch) (r : Regions) : Tile .blockPtr [p.BT,p.BV] :=
  ⟨fun _ => BlockPtr.mk r.output (l.head*p.valueHead) [p.T,p.V] [p.BT,p.BV] [p.valueTime,p.valueDimension]
    [l.chunk*p.BT,l.value*p.BV]⟩

theorem tail_executes (p : Layout) (r : Regions) (scale : ℝ) (s0 u : BlockState)
    (hu : s0.undef = fun _ _ => 0) (hi : Holds p r s0 (numBlocks p) u) :
    ∃ v, stepStmts (tail p r scale) u = some v ∧ v.mem = s0.mem ∧ v.undef = s0.undef ∧ v.pids = s0.pids ∧
      v.regs .blockPtr [p.BT,p.BV] "p_o" = some (outputPointer p (launch s0) r) ∧
      v.regs .real [p.BT,p.BV] "b_o" = some (resultTile p (launch s0) (inputs r s0) scale) := by
  have hmem (rg : RegionName) (off : Nat) : u.readMem rg off = s0.readMem rg off := by
    simp only [BlockState.readMem,hi.memory]
  have hpad : u.undef = fun _ _ => 0 := hi.padding.trans hu
  unfold tail linearOutputFront
  simp only [List.drop_succ_cons,List.drop_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (whereMask_op_eval _ p.BT "m_s" "b_s" (maskTile p) (carryS p (launch s0) (inputs r s0) (numBlocks p))
      (by simp [hi.maskReg]) (by simp [hi.scoreReg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,p.valueDimension] (s0.pids 2*p.valueHead)
      (s0.pids 1*p.BT) (s0.pids 0*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hi.headReg])) (mulConst_eval _ "i_t" _ _ (by simp [hi.timeReg]))
      (mulConst_eval _ "i_v" _ _ (by simp [hi.valueReg]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.output _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,p.valueDimension] (s0.pids 2*p.valueHead)
      (s0.pids 1*p.BT) (s0.pids 0*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hi.headReg])) (mulConst_eval _ "i_t" _ _ (by simp [hi.timeReg]))
      (mulConst_eval _ "i_v" _ _ (by simp [hi.valueReg]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref r.value _ "p_v" (s0.pids 2*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueDimension
      (s0.pids 1*p.BT) (s0.pids 0*p.BV) (by simp) (by simpa using hpad)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accScale_op_eval _ p.BT p.BT p.BV scale "b_o" "b_s" "b_v"
      (carryO p (launch s0) (inputs r s0) (numBlocks p)) (maskedS p (launch s0) (inputs r s0))
      (valueTile p (launch s0) (inputs r s0))
      (by simp [hi.outputReg]) (by simp [maskedS]) (by
        simp only [BlockState.setReg_same]
        apply congrArg some
        apply Tile.ext
        rintro ⟨i,j,⟨⟩⟩
        simp only [valueTile,value,active,outputAddress,row,column,launch,inputs,Tile.mat,
          BlockState.setReg_readMem,hmem,mul_one]
        split_ifs <;> rfl)))]
  rw [stepStmts.nil]
  refine ⟨_,rfl,hi.memory,hi.padding,hi.pids,?_,?_⟩
  · simp [outputPointer,launch]
  · simp [resultTile]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def scatter (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT,p.BV]).foldl (fun u i =>
    if active p l i.1.val i.2.1.val then
      u.writeMem r.output (outputAddress p l i.1.val i.2.1.val) (expected p l a scale i.1.val i.2.1.val) else u) s

def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ i j : TileIndex [p.BT,p.BV], active p l i.1.val i.2.1.val → active p l j.1.val j.2.1.val →
    outputAddress p l i.1.val i.2.1.val = outputAddress p l j.1.val j.2.1.val → i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = r.output ∧ ∃ i : TileIndex [p.BT,p.BV],
    active p l i.1.val i.2.1.val ∧ cell.2 = outputAddress p l i.1.val i.2.1.val

theorem store_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hp : s.regs .blockPtr [p.BT,p.BV] "p_o" = some (outputPointer p l r))
    (ho : s.regs .real [p.BT,p.BV] "b_o" = some (resultTile p l a scale)) :
    stepStmt (linearOutputStoreStmt r.output p.BT p.BV) s = some (scatter p l r a scale s) := by
  simp only [linearOutputStoreStmt,stepStmt,evalOp_ref,hp,ho,Option.bind_eq_bind,Option.bind_some,
    outputPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,Bool.true_and,decide_eq_true_eq,Nat.mul_one]
  apply congrArg some
  unfold scatter
  congr 1
  funext u i
  obtain ⟨i,j,⟨⟩⟩ := i
  simp only [BlockState.writeMemTyped_real]
  erw [result_value]
  rfl

theorem scatter_readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState)
    (hinj : NoCollision p l) (i : TileIndex [p.BT,p.BV]) (hi : active p l i.1.val i.2.1.val) :
    (scatter p l r a scale s).readMem r.output (outputAddress p l i.1.val i.2.1.val) =
      expected p l a scale i.1.val i.2.1.val := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true s _ _ _ i hi
    (fun j hj he => hinj j i hj hi he)

theorem scatter_frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (s : BlockState) :
    BlockState.WriteWithin (footprint p l r) s (scatter p l r a scale s) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,i,hi,rfl⟩

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def kernel (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) : ComputeKernel :=
  forwardO dot value out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV

theorem source_body (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) :
    (kernel p r dot value out scale).eraseDType.toAlgKernel.body =
      setup p ++ [loop p r] ++ tail p r scale ++ [linearOutputStoreStmt r.output p.BT p.BV] := by
  unfold kernel forwardO
  rw [OutputSource.storage_erasure,OutputSource.body_split]
  rfl

theorem region_executes (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (s : BlockState) (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s)) :
    ∃ t, exec (kernel p r dot value out scale).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        t.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r) s t := by
  obtain ⟨v,hl,hv⟩ := loop_executes p r s hu (start p s) (start_holds p r s)
  obtain ⟨u,ht,hm,hpad,hp,hptr,ho⟩ := tail_executes p r scale s v hu hv
  have he : exec (kernel p r dot value out scale).eraseDType.toAlgKernel s =
      some (scatter p (launch s) r (inputs r s) scale u) := by
    show stepStmts _ _ = _
    rw [source_body]
    simp only [List.append_assoc]
    rw [stepStmts.append_some (setup_executes p s)]
    have hlone : stepStmts [loop p r] (start p s) = some v := by
      rw [stepStmts.cons_some hl,stepStmts.nil]
    rw [stepStmts.append_some hlone]
    rw [stepStmts.append_some ht,stepStmts.cons_some (store_executes p (launch s) r (inputs r s) scale u hptr ho),
      stepStmts.nil]
  refine ⟨_,he,fun i hi => scatter_readback p (launch s) r (inputs r s) scale u hinj i hi,?_⟩
  intro region off hn
  simpa only [hm] using scatter_frame p (launch s) r (inputs r s) scale u region off hn

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def candidate (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) : ComputeKernel :=
  forwardOScaleFirst dot value out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV

def before (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  (linearOutputFront r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV scale).take 12

def after (p : Layout) (r : Regions) (scale : ℝ) : List Stmt :=
  (linearOutputFront r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV scale).drop 13 ++ [linearOutputStoreStmt r.output p.BT p.BV]

def totalOp (p : Layout) : Op .real [p.BT,p.BV] :=
  .add .real (.consSame (.consSame .nil)) (.ref .real [p.BT,p.BV] "b_o")
    (.dot (batch := []) (.ref .real [p.BT,p.BT] "b_s") (.ref .real [p.BT,p.BV] "b_v")
      {allowTF32 := some Bool.false})

theorem original_split (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) :
    (kernel p r dot value out scale).eraseDType.toAlgKernel.body =
      before p r scale ++ .assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarR (totalOp p) (.const scale)) ::
      after p r scale := by
  unfold kernel forwardO
  rw [OutputSource.storage_erasure,OutputSource.body_split]
  rfl

theorem candidate_split (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) :
    (candidate p r dot value out scale).eraseDType.toAlgKernel.body =
      before p r scale ++ .assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarL (.const scale) (totalOp p)) ::
      after p r scale := by
  unfold candidate forwardOScaleFirst
  rw [OutputSource.storage_erasure]
  simp only [forwardOSource, Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float,VeriTile.Triton.eraseDType]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType,eraseDType_float,VeriTile.Triton.eraseDType]
  rfl

theorem scaling_commutes (p : Layout) (scale : ℝ) (u : BlockState) :
    stepStmt (.assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarR (totalOp p) (.const scale))) u =
    stepStmt (.assign .real [p.BT,p.BV] "b_o" (.mul .real .scalarL (.const scale) (totalOp p))) u := by
  simp only [stepStmt,evalOp_mul,evalOp_const]
  cases hv : evalOp (totalOp p) u
  · rfl
  rename_i v
  simp only [Option.bind_eq_bind,Option.bind_some]
  congr 2
  apply Tile.ext
  intro i
  cases h : v.data i <;>
    simp [Tile.bop,Tile.scalar,-Tile.scalar_eta,Broadcast.leftIndex,Broadcast.rightIndex,NumericDType.mul,
      WithBot.realMul,h,mul_comm]

theorem candidate_executes_like (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType)
    (scale : ℝ) (s t : BlockState)
    (he : exec (kernel p r dot value out scale).eraseDType.toAlgKernel s = some t) :
    exec (candidate p r dot value out scale).eraseDType.toAlgKernel s = some t := by
  rw [exec,original_split] at he
  rw [exec,candidate_split]
  obtain ⟨u,hbefore,ht⟩ := stepStmts.append_some_iff.mp he
  apply stepStmts.append_some_iff.mpr
  refine ⟨u,hbefore,?_⟩
  simpa only [stepStmts,scaling_commutes p scale u] using ht

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput
open VeriTile.Triton OutputMath
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def program (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  if swapped then candidate p r dot value out scale else kernel p r dot value out scale

def post (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  (tail p r scale).take 4 ++ .assign .real [p.BT,p.BV] "b_o"
    (if swapped then .mul .real .scalarL (.const scale) (totalOp p)
      else .mul .real .scalarR (totalOp p) (.const scale)) :: after p r scale

theorem program_body (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (program p r dot value out scale swapped).eraseDType.toAlgKernel.body =
      setup p ++ loop p r :: post p r scale swapped := by
  cases swapped
  · simp only [program,Bool.false_eq_true,↓reduceIte,source_body]
    rfl
  · simp only [program,↓reduceIte,candidate_split]
    rfl

theorem program_executes (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (s : BlockState) (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s)) :
    ∃ t, exec (program p r dot value out scale swapped).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        t.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r) s t := by
  obtain ⟨t,he,ho,hf⟩ := region_executes p r dot value out scale s hu hinj
  refine ⟨t,?_,ho,hf⟩
  cases swapped
  · exact he
  · exact candidate_executes_like p r dot value out scale s t he

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.RealOutput

#guard_msgs (drop info) in
#auditModuleAxioms

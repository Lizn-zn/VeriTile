import bench.tritonbench_g.rotary_emb.Proofs.Arithmetic

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- A finite real masked store is the same scatter used by the independent
memory specification. Inactive values are unrestricted. -/
theorem real_store_scatter {shape : TileShape} (X : RegionName)
    (offsetExpr : Op .nat shape) (valueExpr : Op .real shape) (maskExpr : Op .bool shape)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : BlockState)
    (hOffset : evalOp offsetExpr s = some ⟨offset⟩)
    (hValue : evalOp valueExpr s = some ⟨fun i ↦ some (value i)⟩)
    (hMask : evalOp maskExpr s = some ⟨fun i ↦ decide (active i)⟩) :
    stepStmt (.store .real shape (.region X offsetExpr) valueExpr (.mask maskExpr)) s =
      some (Scatter.one X offset value active s) := by
  classical
  simp [stepStmt, hOffset, hValue, hMask, Scatter.one,
    FloatDType.storeValue]

theorem real_ptr_store_scatter {shape : TileShape} (X : RegionName)
    (ptrExpr : Op .ptr shape) (valueExpr : Op .real shape) (maskExpr : Op .bool shape)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : BlockState)
    (hPtr : evalOp ptrExpr s = some ⟨fun i ↦ (X,offset i)⟩)
    (hValue : evalOp valueExpr s = some ⟨fun i ↦ some (value i)⟩)
    (hMask : evalOp maskExpr s = some ⟨fun i ↦ decide (active i)⟩) :
    stepStmt (.store .real shape (.ptr ptrExpr) valueExpr (.mask maskExpr)) s =
      some (Scatter.one X offset value active s) := by
  classical
  simp [stepStmt, hPtr, hValue, hMask, Scatter.one,
    FloatDType.storeValue]

theorem real_masked_store_scatter {shape : TileShape} (X : RegionName)
    (offsetExpr : Op .nat shape) (valueExpr : Op .real shape) (maskExpr : Op .bool shape)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : BlockState)
    (hOffset : evalOp offsetExpr s = some ⟨offset⟩)
    (hValue : evalOp valueExpr s = some ⟨fun i ↦ some (if active i then value i else 0)⟩)
    (hMask : evalOp maskExpr s = some ⟨fun i ↦ decide (active i)⟩) :
    stepStmt (.store .real shape (.region X offsetExpr) valueExpr (.mask maskExpr)) s =
      some (Scatter.one X offset value active s) := by
  rw [real_store_scatter X _ _ _ offset (fun i ↦ if active i then value i else 0)
    active s hOffset hValue hMask]
  congr 1
  apply Scatter.one_value_congr
  intro i hi
  simp only [hi, ↓reduceIte]

theorem real_masked_ptr_store_scatter {shape : TileShape} (X : RegionName)
    (ptrExpr : Op .ptr shape) (valueExpr : Op .real shape) (maskExpr : Op .bool shape)
    (offset : TileIndex shape → Nat) (value : TileIndex shape → ℝ)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : BlockState)
    (hPtr : evalOp ptrExpr s = some ⟨fun i ↦ (X,offset i)⟩)
    (hValue : evalOp valueExpr s = some ⟨fun i ↦ some (if active i then value i else 0)⟩)
    (hMask : evalOp maskExpr s = some ⟨fun i ↦ decide (active i)⟩) :
    stepStmt (.store .real shape (.ptr ptrExpr) valueExpr (.mask maskExpr)) s =
      some (Scatter.one X offset value active s) := by
  rw [real_ptr_store_scatter X _ _ _ offset (fun i ↦ if active i then value i else 0)
    active s hPtr hValue hMask]
  congr 1
  apply Scatter.one_value_congr
  intro i hi
  simp only [hi, ↓reduceIte]

def qStoreBody (p : Parameters) (Q K Cos Sin : RegionName) : List Stmt :=
  ((mathKernel p Q K Cos Sin).body.drop 18).take 2

noncomputable def qEvenState (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : BlockState :=
  Scatter.one Q (p.qAddress s Bool.false) (p.expectedQ s Q Cos Sin Bool.false)
    (p.activeQ s) (qArithmeticState p s Q Cos Sin)

noncomputable def qStoreState (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : BlockState :=
  Scatter.pair Q (p.qAddress s) (p.expectedQ s Q Cos Sin) (p.activeQ s)
    (qArithmeticState p s Q Cos Sin)

macro "solveRotaryQStore" : tactic => `(tactic|
  simp [evalOp.eq_def, qEvenState, qArithmeticState, qLoadState, qIndexState,
    prefixState, Parameters.dataOffsets, Parameters.resultValues, Parameters.qAddress,
    Parameters.expectedQ, Parameters.activeQ, address, trigAddress, sequence,
    head, dimension, active, range, dims, tile_elementwise, Bool.and_eq_true])

theorem q_even_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmt (mathKernel p Q K Cos Sin).body[18]! (qArithmeticState p s Q Cos Sin) =
      some (qEvenState p s Q Cos Sin) := by
  classical
  first
  | apply real_masked_store_scatter Q _ _ _ (p.qAddress s Bool.false)
      (p.expectedQ s Q Cos Sin Bool.false) (p.activeQ s) (qArithmeticState p s Q Cos Sin)
  | apply real_masked_ptr_store_scatter Q _ _ _ (p.qAddress s Bool.false)
      (p.expectedQ s Q Cos Sin Bool.false) (p.activeQ s) (qArithmeticState p s Q Cos Sin)
  all_goals solveRotaryQStore

theorem q_odd_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmt (mathKernel p Q K Cos Sin).body[19]! (qEvenState p s Q Cos Sin) =
      some (qStoreState p s Q Cos Sin) := by
  classical
  first
  | apply real_masked_store_scatter Q _ _ _ (p.qAddress s Bool.true)
      (p.expectedQ s Q Cos Sin Bool.true) (p.activeQ s) (qEvenState p s Q Cos Sin)
  | apply real_masked_ptr_store_scatter Q _ _ _ (p.qAddress s Bool.true)
      (p.expectedQ s Q Cos Sin Bool.true) (p.activeQ s) (qEvenState p s Q Cos Sin)
  all_goals solveRotaryQStore

theorem q_store_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    stepStmts (qStoreBody p Q K Cos Sin) (qArithmeticState p s Q Cos Sin) =
      some (qStoreState p s Q Cos Sin) := by
  change stepStmts [(mathKernel p Q K Cos Sin).body[18]!,
    (mathKernel p Q K Cos Sin).body[19]!] (qArithmeticState p s Q Cos Sin) = _
  rw [stepStmts.cons_some (q_even_store_run p Q K Cos Sin s),
    stepStmts.cons_some (q_odd_store_run p Q K Cos Sin s)] <;> rfl

theorem q_store_state_mem (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) :
    (qStoreState p s Q Cos Sin).mem =
      (Scatter.pair Q (p.qAddress s) (p.expectedQ s Q Cos Sin) (p.activeQ s) s).mem :=
  Scatter.pair_mem_congr Q _ _ _ _ _ (q_arithmetic_mem p s Q Cos Sin)

@[simp] theorem q_store_state_pids (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) : (qStoreState p s Q Cos Sin).pids = s.pids := by
  simp only [qStoreState, Scatter.pair_pids, q_arithmetic_pids]

theorem q_store_state_read_other (p : Parameters) (s : BlockState)
    (Q Cos Sin R : RegionName) (hRQ : R≠Q) (o : Nat) :
    (qStoreState p s Q Cos Sin).readMem R o = s.readMem R o := by
  rw [show (qStoreState p s Q Cos Sin).readMem R o =
    (qArithmeticState p s Q Cos Sin).readMem R o from
    Scatter.pair_read_other Q R hRQ _ _ _ _ o]
  rfl

theorem q_store_state_readback (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) (hGeometry : p.Geometry s)
    (odd : Bool) (i : p.Lane) (hi : p.activeQ s i) :
    (qStoreState p s Q Cos Sin).readMem Q (p.qAddress s odd i) =
      p.expectedQ s Q Cos Sin odd i :=
  Scatter.pair_readback Q _ _ _ _ hGeometry.1 odd i hi

theorem q_store_state_frame (p : Parameters) (s : BlockState)
    (Q K Cos Sin : RegionName) :
    BlockState.WriteWithin (p.footprint s Q K)
      (qArithmeticState p s Q Cos Sin) (qStoreState p s Q Cos Sin) := by
  apply Scatter.pair_frame
  intro odd i hi
  exact Or.inl ⟨rfl,i,odd,hi,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath

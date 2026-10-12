import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealStores

/-! Real execution of both END normalization variants and every M/L/Out store.
The real division is total, including denominator zero. Floating equivalence
uses its separate admitted rule and guards; none are assumed here. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogue
open VeriTile.Triton RealTiles RealLoopFormula RealStores
noncomputable section

def denominatorExpr (BM : Nat) : Op .real [BM, 1] := .expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")

def quotientExpr (BM BD : Nat) (reciprocal : Bool) : Op .real [BM, BD] :=
  if reciprocal then
    .mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
      (.div .real .scalarL (.const 1.0) (denominatorExpr BM))
  else .div .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc") (denominatorExpr BM)

def normalizeCode (BM BD : Nat) (reciprocal : Bool) : List Stmt :=
  [.assign .real [BM] "m_i" (.add .real (.consSame .nil) (.ref .real [BM] "m_i") (.log2 (.ref .real [BM] "l_i"))),
   .assign .real [BM, BD] "acc" (quotientExpr BM BD reciprocal)]

def denominatorStore (BM : Nat) : Stmt :=
  .store .real [BM] (.ptr (.ref .ptr [BM] "l_ptrs")) (.ref .real [BM] "l_i") .none

def centerStore (BM : Nat) : Stmt :=
  .store .real [BM] (.ptr (.ref .ptr [BM] "m_ptrs")) (.ref .real [BM] "m_i") .none

def outputStore (BM BD : Nat) : Stmt :=
  .store .real [BM, BD] (.blockPtr (.ref .blockPtr [BM, BD] "O_block_ptr") []) (.ref .real [BM, BD] "acc") .none

def endBranch (BM BD finish : Nat) (reciprocal : Bool) : Stmt :=
  .ifThenElse (.ne .nat .nil (.constNat finish) (.constNat 0)) (normalizeCode BM BD reciprocal) [denominatorStore BM]

def code (BM BD finish : Nat) (reciprocal : Bool) : List Stmt :=
  [endBranch BM BD finish reciprocal, centerStore BM, outputStore BM BD]

def centerTile {BM BD : Nat} (c : Carry BM BD) : Tile .real [BM] := ⟨fun i => c.center i.1⟩
def denominatorTile {BM BD : Nat} (c : Carry BM BD) : Tile .real [BM] := finite (fun i => c.denominator i.1)

structure Registers {BM BD : Nat} (c : Carry BM BD) (s : BlockState) : Prop where
  center : s.regs .real [BM] "m_i" = some (centerTile c)
  denominator : s.regs .real [BM] "l_i" = some (denominatorTile c)
  numerator : s.regs .real [BM, BD] "acc" = some (finite c.numerator)

def normalized {BM BD : Nat} (c : Carry BM BD) : Carry BM BD where
  center := fun row => (c.center row).map (fun m => m + Real.log (c.denominator row) / Real.log 2)
  denominator := c.denominator
  numerator := fun i => c.numerator i / c.denominator i.1

def normalizeState {BM BD : Nat} (s : BlockState) (c : Carry BM BD) : BlockState :=
  (s.setReg "m_i" .real [BM] (centerTile (normalized c))).setReg "acc" .real [BM, BD] (finite (normalized c).numerator)

theorem center_eval {BM BD : Nat} (s : BlockState) (c : Carry BM BD) (h : Registers c s) :
    evalOp (.add .real (.consSame .nil) (.ref .real [BM] "m_i") (.log2 (.ref .real [BM] "l_i"))) s =
      some (centerTile (normalized c)) := by
  simp only [evalOp.eq_def, h.center, h.denominator, bind, pure, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨row, ⟨⟩⟩
  cases hm : c.center row <;>
    simp [Tile.bop_data, Tile.uop, centerTile, denominatorTile, finite, normalized, NumericDType.add,
      WithBot.realLog2, hm]
  all_goals rfl

theorem quotient_eval {BM BD : Nat} (s : BlockState) (c : Carry BM BD) (reciprocal : Bool)
    (hd : s.regs .real [BM] "l_i" = some (denominatorTile c))
    (ha : s.regs .real [BM, BD] "acc" = some (finite c.numerator)) :
    evalOp (quotientExpr BM BD reciprocal) s = some (finite (normalized c).numerator) := by
  cases reciprocal with
  | «false» =>
    simp only [quotientExpr, Bool.false_eq_true, if_false, denominatorExpr, evalOp.eq_def,
      hd, ha, bind, pure, Option.bind_some]
    rfl
  | «true» =>
    simp only [quotientExpr, if_true, denominatorExpr, evalOp.eq_def, hd, ha, bind, pure, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨row, lane, ⟨⟩⟩
    change (some (c.numerator (row, lane, PUnit.unit) * (1.0 / c.denominator row)) : WithBot ℝ) =
      some (c.numerator (row, lane, PUnit.unit) / c.denominator row)
    norm_num [div_eq_mul_inv]

theorem normalize_executes {BM BD : Nat} (s : BlockState) (c : Carry BM BD) (reciprocal : Bool)
    (h : Registers c s) :
    stepStmts (normalizeCode BM BD reciprocal) s = some (normalizeState s c) := by
  let sm := s.setReg "m_i" .real [BM] (centerTile (normalized c))
  have hq := quotient_eval sm c reciprocal (by simp [sm, h.denominator]) (by simp [sm, h.numerator])
  rw [normalizeCode, stepStmts.cons_some (stepStmt_assign_eq_some (center_eval s c h)),
    stepStmts.cons_some (stepStmt_assign_eq_some hq), stepStmts.nil]
  rfl

def centerValue {BM BD : Nat} (finish : Nat) (c : Carry BM BD) (row : Fin BM) : ℝ :=
  (RealFormula.writeback (decide (finish ≠ 0)) (rowSeed c row 0)).maxValue

def outputValue {BM BD : Nat} (finish : Nat) (c : Carry BM BD) (i : TileIndex [BM, BD]) : ℝ :=
  (RealFormula.writeback (decide (finish ≠ 0)) (rowSeed c i.1 (c.numerator i))).output

def savedCarry {BM BD : Nat} (finish : Nat) (c : Carry BM BD) : Carry BM BD :=
  if finish ≠ 0 then normalized c else c

theorem center_value {BM BD : Nat} (finish : Nat) (c : Carry BM BD) (i : TileIndex [BM]) :
    FloatDType.real.storeValue ((centerTile (savedCarry finish c)).data i) = centerValue finish c i.1 := by
  rw [centerValue, RealFormula.writeback_center_matches]
  by_cases h : finish ≠ 0
  · simp only [savedCarry, if_pos h, centerTile, normalized, decide_eq_true h, if_true, rowSeed]
    cases c.center i.1 <;> rfl
  · simp [savedCarry, h, centerTile, rowSeed]

theorem output_value {BM BD : Nat} (finish : Nat) (c : Carry BM BD) (i : TileIndex [BM, BD]) :
    FloatDType.real.storeValue ((finite (savedCarry finish c).numerator).data i) = outputValue finish c i := by
  by_cases h : finish ≠ 0 <;> simp [finite, FloatDType.storeValue, savedCarry, normalized,
    outputValue, RealFormula.writeback, rowSeed, h]

def branchState {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (L : RegionName) (rowAddress : TileIndex [BM] → Nat) : BlockState :=
  if finish ≠ 0 then normalizeState s c else writeTile s L rowAddress (fun i => c.denominator i.1)

theorem branch_executes {BM BD : Nat} (s : BlockState) (finish : Nat) (reciprocal : Bool)
    (c : Carry BM BD) (L : RegionName) (rowAddress : TileIndex [BM] → Nat)
    (h : Registers c s)
    (hl : s.regs .ptr [BM] "l_ptrs" = some (⟨fun i => (L, rowAddress i)⟩ : Tile .ptr [BM])) :
    stepStmt (endBranch BM BD finish reciprocal) s = some (branchState s finish c L rowAddress) := by
  have hd := pointer_store s [BM] "l_ptrs" "l_i" L rowAddress (denominatorTile c) hl h.denominator
  have hs (yes no : List Stmt) :
      stepStmt (.ifThenElse (.ne .nat .nil (.constNat finish) (.constNat 0)) yes no) s =
        if finish ≠ 0 then stepStmts yes s else stepStmts no s := by
    conv_lhs => unfold stepStmt
    simp only [evalOp.eq_def, bind, Option.bind_some, Tile.scalar, Tile.cop, ComparableDType.ne]
    simp
  rw [endBranch, hs]
  by_cases he : finish ≠ 0
  · rw [if_pos he, branchState, if_pos he]
    exact normalize_executes s c reciprocal h
  · erw [if_neg he, stepStmts.cons_some hd, stepStmts.nil, branchState, if_neg he]
    rfl

theorem branch_registers {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (L : RegionName) (rowAddress : TileIndex [BM] → Nat) (h : Registers c s) :
    Registers (savedCarry finish c) (branchState s finish c L rowAddress) := by
  by_cases he : finish ≠ 0
  · simp only [branchState, savedCarry, if_pos he]
    constructor <;> simp [normalizeState, denominatorTile, normalized, h.denominator]
  · simp only [branchState, savedCarry, if_neg he]
    exact ⟨by simpa using h.center, by simpa using h.denominator, by simpa using h.numerator⟩

theorem branch_other_register {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (L : RegionName) (rowAddress : TileIndex [BM] → Nat) (dt shape name)
    (hn : name ≠ "m_i" ∧ name ≠ "acc") :
    (branchState s finish c L rowAddress).regs dt shape name = s.regs dt shape name := by
  by_cases he : finish ≠ 0 <;> simp [branchState, he, normalizeState, hn.1, hn.2]

theorem branch_control {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (L : RegionName) (rowAddress : TileIndex [BM] → Nat) :
    (branchState s finish c L rowAddress).pids = s.pids ∧
      (branchState s finish c L rowAddress).undef = s.undef := by
  by_cases he : finish ≠ 0
  · rw [branchState, if_pos he]
    exact ⟨rfl, rfl⟩
  · simpa [branchState, he] using write_tile_control s L rowAddress (fun i => c.denominator i.1)

def blockAddress (BM BD : Nat) (bp : BlockPtr) : TileIndex [BM, BD] → Nat :=
  fun i => bp.address (TileShape.indexToList [BM, BD] i)

/-- Internal operational state for proving the three stores. Output readback
below is stated using the independent mathematical writeback values. -/
def writtenState {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr) : BlockState :=
  let se := branchState s finish c L rowAddress
  let sm := writeTile se M rowAddress (fun i => centerValue finish c i.1)
  writeTile sm bp.region (blockAddress BM BD bp) (outputValue finish c)

theorem executes {BM BD : Nat} (s : BlockState) (finish : Nat) (reciprocal : Bool) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr)
    (h : Registers c s)
    (hm : s.regs .ptr [BM] "m_ptrs" = some (⟨fun i => (M, rowAddress i)⟩ : Tile .ptr [BM]))
    (hl : s.regs .ptr [BM] "l_ptrs" = some (⟨fun i => (L, rowAddress i)⟩ : Tile .ptr [BM]))
    (ho : s.regs .blockPtr [BM, BD] "O_block_ptr" = some (⟨fun _ => bp⟩ : Tile .blockPtr [BM, BD])) :
    stepStmts (code BM BD finish reciprocal) s = some (writtenState s finish c M L rowAddress bp) := by
  let se := branchState s finish c L rowAddress
  let sm := writeTile se M rowAddress (fun i => centerValue finish c i.1)
  have he := branch_executes s finish reciprocal c L rowAddress h hl
  have hr := branch_registers s finish c L rowAddress h
  have em : stepStmt (centerStore BM) se = some sm := by
    have hp := (branch_other_register s finish c L rowAddress .ptr [BM] "m_ptrs" (by simp)).trans hm
    have hs := pointer_store se [BM] "m_ptrs" "m_i" M rowAddress (centerTile (savedCarry finish c)) hp hr.center
    simpa only [center_value] using hs
  have eo : stepStmt (outputStore BM BD) sm = some (writtenState s finish c M L rowAddress bp) := by
    have hp : sm.regs .blockPtr [BM, BD] "O_block_ptr" = some (⟨fun _ => bp⟩ : Tile .blockPtr [BM, BD]) := by
      rw [write_tile_regs]
      exact (branch_other_register s finish c L rowAddress .blockPtr [BM, BD] "O_block_ptr" (by simp)).trans ho
    have hv : sm.regs .real [BM, BD] "acc" = some (finite (savedCarry finish c).numerator) := by
      rw [write_tile_regs]
      exact hr.numerator
    have hs := block_store sm [BM, BD] "O_block_ptr" "acc" bp.region bp (finite (savedCarry finish c).numerator) rfl hp hv
    simpa only [output_value] using hs
  rw [code, stepStmts.cons_some he, stepStmts.cons_some em, stepStmts.cons_some eo, stepStmts.nil]

theorem written_other_register {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr) (dt shape name)
    (hn : name ≠ "m_i" ∧ name ≠ "acc") :
    (writtenState s finish c M L rowAddress bp).regs dt shape name = s.regs dt shape name := by
  simp only [writtenState, write_tile_regs, branch_other_register _ _ _ _ _ _ _ _ hn]

theorem written_control {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr) :
    (writtenState s finish c M L rowAddress bp).pids = s.pids ∧
      (writtenState s finish c M L rowAddress bp).undef = s.undef := by
  have he := branch_control s finish c L rowAddress
  have hm := write_tile_control (branchState s finish c L rowAddress) M rowAddress (fun i => centerValue finish c i.1)
  have ho := write_tile_control
    (writeTile (branchState s finish c L rowAddress) M rowAddress (fun i => centerValue finish c i.1))
    bp.region (blockAddress BM BD bp) (outputValue finish c)
  exact ⟨ho.1.trans (hm.1.trans he.1), ho.2.trans (hm.2.trans he.2)⟩

theorem written_frame {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr) (r : RegionName) (o : Nat)
    (hM : r ≠ M ∨ ∀ i, o ≠ rowAddress i)
    (hL : finish ≠ 0 ∨ r ≠ L ∨ ∀ i, o ≠ rowAddress i)
    (hO : r ≠ bp.region ∨ ∀ i, o ≠ blockAddress BM BD bp i) :
    (writtenState s finish c M L rowAddress bp).mem r o = s.mem r o := by
  rw [writtenState, write_tile_frame _ _ _ _ r o hO, write_tile_frame _ _ _ _ r o hM]
  by_cases he : finish ≠ 0
  · simp [branchState, he, normalizeState]
  · rw [branchState, if_neg he]
    exact write_tile_frame _ _ _ _ r o (hL.resolve_left he)

theorem center_readback {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr)
    (hi : Function.Injective rowAddress) (hMO : M ≠ bp.region) (i : TileIndex [BM]) :
    (writtenState s finish c M L rowAddress bp).readMem M (rowAddress i) = centerValue finish c i.1 := by
  rw [writtenState, write_tile_read_other _ _ _ _ M (rowAddress i) (Or.inl hMO), write_tile_readback _ _ _ _ hi]

theorem output_readback {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr)
    (hi : Function.Injective (blockAddress BM BD bp)) (i : TileIndex [BM, BD]) :
    (writtenState s finish c M L rowAddress bp).readMem bp.region (blockAddress BM BD bp i) = outputValue finish c i :=
  write_tile_readback _ _ _ _ hi i

theorem denominator_readback {BM BD : Nat} (s : BlockState) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr)
    (hi : Function.Injective rowAddress) (hLM : L ≠ M) (hLO : L ≠ bp.region) (i : TileIndex [BM]) :
    (writtenState s 0 c M L rowAddress bp).readMem L (rowAddress i) = c.denominator i.1 := by
  rw [writtenState, write_tile_read_other _ _ _ _ L (rowAddress i) (Or.inl hLO),
    write_tile_read_other _ _ _ _ L (rowAddress i) (Or.inl hLM)]
  simp only [branchState, ne_eq, not_true_eq_false, if_false]
  exact write_tile_readback _ _ _ _ hi i

theorem finished_preserves_L {BM BD : Nat} (s : BlockState) (finish : Nat) (c : Carry BM BD)
    (M L : RegionName) (rowAddress : TileIndex [BM] → Nat) (bp : BlockPtr)
    (he : finish ≠ 0) (hLM : L ≠ M) (hLO : L ≠ bp.region) (o : Nat) :
    (writtenState s finish c M L rowAddress bp).mem L o = s.mem L o :=
  written_frame s finish c M L rowAddress bp L o (Or.inl hLM) (Or.inl he) (Or.inl hLO)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealEpilogue

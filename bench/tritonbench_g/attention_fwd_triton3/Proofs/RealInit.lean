import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoopSource

/-! Real execution of the INIT/continuation carry, exact source scale and
Q load. Pointer construction and window bounds are separate prologue parts.
RealInitSource checks these fragments against both complete typed sources. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInit
open VeriTile.Triton RealTiles RealScores RealAccumulation RealLoopFormula
noncomputable section

def carryCode (BM BD init : Nat) : Stmt :=
  .ifThenElse (.ne .nat .nil (.constNat init) (.constNat 0))
    [.assign .real [BM] "m_i" (.add .real .scalarR (.full [BM] (.const 0)) .negInf),
     .assign .real [BM] "l_i" (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)),
     .assign .real [BM, BD] "acc" (.full [BM, BD] (.const 0))]
    [.assign .real [BM] "m_i" (.load .real (.ptr (.ref .ptr [BM] "m_ptrs")) .none),
     .assign .real [BM] "l_i" (.load .real (.ptr (.ref .ptr [BM] "l_ptrs")) .none),
     .assign .real [BM, BD] "acc" (.load .real (.blockPtr (.ref .blockPtr [BM, BD] "O_block_ptr") []) .none .unspecified)]

def scaleCode (scale : ℝ) : List Stmt :=
  [.assign .real [] "qk_scale" (.const scale),
   .assign .real [] "qk_scale" (.mul .real .nil (.ref .real [] "qk_scale") (.const 1.4426950408889634))]

def queryCode (BM BD even : Nat) : Stmt :=
  .ifThenElse (.ne .nat .nil (.constNat even) (.constNat 0))
    [.assign .real [BM, BD] "q" (.load .real (.blockPtr (.ref .blockPtr [BM, BD] "Q_block_ptr") []) .none .unspecified)]
    [.assign .real [BM, BD] "q" (.load .real (.blockPtr (.ref .blockPtr [BM, BD] "Q_block_ptr") [0, 1]) .none)]

def pointerValues {shape : TileShape} (s : BlockState) (ptrs : Tile .ptr shape) : TileIndex shape → ℝ :=
  fun i => s.readMem (ptrs.data i).1 (ptrs.data i).2

def seed {BM BD : Nat} (s : BlockState) (init : Nat)
    (mp lp : Tile .ptr [BM]) (op : Tile .blockPtr [BM, BD]) : Carry BM BD where
  center := fun row => if init ≠ 0 then ⊥ else some (pointerValues s mp (row, PUnit.unit))
  denominator := fun row => if init ≠ 0 then 1 else pointerValues s lp (row, PUnit.unit)
  numerator := fun i => if init ≠ 0 then 0 else values s op [] i

theorem eval_pointer_load (s : BlockState) (shape : TileShape) (name : RegName)
    (ptrs : Tile .ptr shape) (hp : s.regs .ptr shape name = some ptrs) :
    evalOp (.load .real (.ptr (.ref .ptr shape name)) .none) s = some (finite (pointerValues s ptrs)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind, if_true, BlockState.readMemValue_real]
  rfl

theorem carry_executes (s : BlockState) (BM BD init : Nat)
    (mp lp : Tile .ptr [BM]) (op : Tile .blockPtr [BM, BD])
    (hm : s.regs .ptr [BM] "m_ptrs" = some mp)
    (hl : s.regs .ptr [BM] "l_ptrs" = some lp)
    (ho : s.regs .blockPtr [BM, BD] "O_block_ptr" = some op) :
    ∃ s', stepStmt (carryCode BM BD init) s = some s' ∧
      s'.regs .real [BM] "m_i" = some (⟨fun i => (seed s init mp lp op).center i.1⟩ : Tile .real [BM]) ∧
      s'.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => (seed s init mp lp op).denominator i.1)) ∧
      s'.regs .real [BM, BD] "acc" = some (finite (seed s init mp lp op).numerator) ∧
      Frame ["m_i", "l_i", "acc"] s s' := by
  unfold carryCode
  have hbranch (yes no : List Stmt) :
      stepStmt (.ifThenElse (.ne .nat .nil (.constNat init) (.constNat 0)) yes no) s =
        if init ≠ 0 then stepStmts yes s else stepStmts no s := by
    conv_lhs => unfold stepStmt
    simp only [evalOp.eq_def, bind, Option.bind, Tile.cop_data, Tile.scalar, ComparableDType.ne]
    simp
  rw [hbranch]
  by_cases hi : init ≠ 0
  · rw [if_pos hi]
    have hmi : evalOp (.add .real .scalarR (.full [BM] (.const 0)) .negInf) s =
        some (⟨fun _ => ⊥⟩ : Tile .real [BM]) := by simp [evalOp.eq_def]; rfl
    let sm := s.setReg "m_i" .real [BM] (⟨fun _ => ⊥⟩ : Tile .real [BM])
    have hli : evalOp (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)) sm =
        some (finite (fun _ : TileIndex [BM] => 1)) := by
      norm_num [evalOp.eq_def, finite]
      apply Tile.ext
      rintro ⟨row, ⟨⟩⟩
      simp [Tile.bop_data, NumericDType.add, Tile.scalar, -Tile.scalar_eta]
    let sl := sm.setReg "l_i" .real [BM] (finite (fun _ => 1))
    have hac : evalOp (.full [BM, BD] (.const 0)) sl =
        some (finite (fun _ : TileIndex [BM, BD] => 0)) := by simp [evalOp.eq_def, finite]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some hmi), stepStmts.cons_some (stepStmt_assign_eq_some hli),
      stepStmts.cons_some (stepStmt_assign_eq_some hac), stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
    · simp [sl, sm, seed, hi]
    · simp [sl, seed, hi]
    · simp [seed, hi]
    · exact ((Frame.setReg _ s "m_i" _ _ _ (by simp)).trans
        (Frame.setReg _ sm "l_i" _ _ _ (by simp))).trans (Frame.setReg _ sl "acc" _ _ _ (by simp))
  · rw [if_neg hi]
    let sm := s.setReg "m_i" .real [BM] (finite (pointerValues s mp))
    let sl := sm.setReg "l_i" .real [BM] (finite (pointerValues sm lp))
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (eval_pointer_load s [BM] "m_ptrs" mp hm)),
      stepStmts.cons_some (stepStmt_assign_eq_some (eval_pointer_load sm [BM] "l_ptrs" lp (by simp [sm, hl]))),
      stepStmts.cons_some (stepStmt_assign_eq_some (eval_unchecked_load sl [BM, BD] "O_block_ptr" op (by simp [sl, sm, ho]) .unspecified)),
      stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
    · simp [sl, sm, seed, hi, finite, pointerValues]
    · simp [sl, sm, seed, hi, finite, pointerValues]
    · simp only [BlockState.setReg_same, seed, if_neg hi]
      exact congrArg some (congrArg finite (RealIteration.values_of_same_memory sl s op [] rfl))
    · exact ((Frame.setReg _ s "m_i" _ _ _ (by simp)).trans
        (Frame.setReg _ sm "l_i" _ _ _ (by simp))).trans (Frame.setReg _ sl "acc" _ _ _ (by simp))

/-- Keep the source's exact decimal scale; it is not replaced by 1/log(2). -/
theorem scale_executes (s : BlockState) (scale : ℝ) :
    ∃ s', stepStmts (scaleCode scale) s = some s' ∧
      s'.regs .real [] "qk_scale" = some (Tile.scalar (some (scale * 1.4426950408889634))) ∧
      Frame ["qk_scale"] s s' := by
  let ss := s.setReg "qk_scale" .real [] (Tile.scalar (some scale))
  have hm : evalOp (.mul .real .nil (.ref .real [] "qk_scale") (.const 1.4426950408889634)) ss =
      some (Tile.scalar (some (scale * 1.4426950408889634))) := by simp [evalOp.eq_def, ss]; rfl
  simp only [scaleCode]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_const scale s)),
    stepStmts.cons_some (stepStmt_assign_eq_some hm), stepStmts.nil]
  exact ⟨_, rfl, by simp, (Frame.setReg _ s "qk_scale" _ _ _ (by simp)).trans
    (Frame.setReg _ ss "qk_scale" _ _ _ (by simp))⟩

theorem query_executes (s : BlockState) (BM BD even : Nat) (qp : Tile .blockPtr [BM, BD])
    (hq : s.regs .blockPtr [BM, BD] "Q_block_ptr" = some qp) :
    ∃ s', stepStmt (queryCode BM BD even) s = some s' ∧
      s'.regs .real [BM, BD] "q" = some (finite (values s qp (checks even))) ∧ Frame ["q"] s s' := by
  rw [queryCode, load_choice_executes s [BM, BD] "Q_block_ptr" "q" even qp hq]
  exact ⟨_, rfl, by simp, Frame.setReg _ s "q" _ _ _ (by simp)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealInit

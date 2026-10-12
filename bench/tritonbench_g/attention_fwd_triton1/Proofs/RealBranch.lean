import bench.tritonbench_g.attention_fwd_triton1.Proofs.Legacy
import VeriTile.Meta.StatementAudit

/-! Real arithmetic for the optional first-chunk shortcut. The loop starts
with a zero carry, so omitting its contribution on chunk zero preserves the
linear-attention formula. These lemmas do not assert an FP identity. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealBranch
open VeriTile.Triton
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 800000

/-- The two assignments used on an ordinary chunk. -/
def accumulate (BT BD : Nat) : List Stmt :=
  [ .assign .real [BT, BD] "b_o"
      (.add .real Broadcast.nil.consSame.consSame (.ref .real [BT, BD] "b_o")
        (@Op.dot [] BT BD BD (.ref .real [BT, BD] "b_q")
          (.ref .real [BD, BD] "b_h") { allowTF32 := some Bool.false })),
    .assign .real [BD, BD] "b_h"
      (.add .real Broadcast.nil.consSame.consSame (.ref .real [BD, BD] "b_h")
        (@Op.dot [] BD BT BD (.ref .real [BD, BT] "b_k")
          (.ref .real [BT, BD] "b_v") { allowTF32 := some Bool.false })) ]

/-- The actual nested constexpr/runtime branch, including the first-chunk
assignment rather than an assumed equality of its results. -/
def update (BT BD : Nat) (IFCOND : Bool) : Stmt :=
  .ifThenElse (.constBool IFCOND)
    [.ifThenElse (.eq .nat .nil (.ref .nat [] "i") (.constNat 0))
      [.assign .real [BD, BD] "b_h"
        (@Op.dot [] BD BT BD (.ref .real [BD, BT] "b_k")
          (.ref .real [BT, BD] "b_v") { allowTF32 := some Bool.false })]
      (accumulate BT BD)]
    (accumulate BT BD)

theorem update_is_source (Q K V H O : RegionName)
    (s_qh s_hh s_ht : Nat) (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) :
    (aft1LoopBodyG Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND).drop 12 =
      [update BT BD IFCOND,
       .store .real [BT, BD] (.blockPtr (.ref .blockPtr [BT, BD] "p_o") [])
         (.ref .real [BT, BD] "b_o") .none] := rfl

theorem first_output (s : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD : Nat) :
    aft1BoLocalTileG s Q K V s_qh scale BT BD 0 =
      aft1BoTileG s Q K V s_qh scale BT BD 0 := by
  ext idx
  simp [aft1BoLocalTileG, aft1BoTileG, aft1OutG, aft1RecOutG, aft1RecStateG]

theorem first_carry (s st : BlockState) (K V : RegionName) (s_qh BT BD : Nat)
    (hk : st.regs .real [BD, BT] "b_k" = some (aft1BkTileG s K s_qh BT BD 0))
    (hv : st.regs .real [BT, BD] "b_v" = some (aft1BvTileG s V s_qh BT BD 0)) :
    evalOp (@Op.dot [] BD BT BD (.ref .real [BD, BT] "b_k")
      (.ref .real [BT, BD] "b_v") { allowTF32 := some Bool.false }) st =
      some (aft1BhTileG s K V s_qh BT BD 1) := by
  rw [aft1_dot_op_eval st "b_k" "b_v"
    (aft1BkTileG s K s_qh BT BD 0) (aft1BvTileG s V s_qh BT BD 0)
    (fun d' tk => aft1KCellG s K s_qh BT BD 0 d'.val tk.val)
    (fun tk d => aft1VCellG s V s_qh BT BD 0 tk.val d.val)
    hk hv (fun _ _ => rfl) (fun _ _ => rfl)]
  congr 1
  ext idx
  simp [aft1BhTileG, aft1RecStateG]

noncomputable def afterUpdate (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool) : BlockState :=
  let output := if IFCOND = Bool.true ∧ c = 0 then st
    else st.setReg "b_o" .real [BT, BD] (aft1BoTileG s Q K V s_qh scale BT BD c)
  output.setReg "b_h" .real [BD, BD] (aft1BhTileG s K V s_qh BT BD (c + 1))

/-- Evaluate either branch for arbitrary chunk dimensions. The inputs are
register facts obtained from the preceding loads and dot products. -/
theorem update_executes (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool)
    (hi : st.regs .nat [] "i" = some (Tile.scalar c))
    (hq : st.regs .real [BT, BD] "b_q" = some (aft1BqTileG s Q s_qh scale BT BD c))
    (hk : st.regs .real [BD, BT] "b_k" = some (aft1BkTileG s K s_qh BT BD c))
    (hv : st.regs .real [BT, BD] "b_v" = some (aft1BvTileG s V s_qh BT BD c))
    (ho : st.regs .real [BT, BD] "b_o" = some (aft1BoLocalTileG s Q K V s_qh scale BT BD c))
    (hh : st.regs .real [BD, BD] "b_h" = some (aft1BhTileG s K V s_qh BT BD c)) :
    stepStmt (update BT BD IFCOND) st =
      some (afterUpdate s st Q K V s_qh scale BT BD c IFCOND) := by
  have hacc : stepStmts (accumulate BT BD) st =
      some ((st.setReg "b_o" .real [BT, BD] (aft1BoTileG s Q K V s_qh scale BT BD c)).setReg "b_h" .real [BD, BD] (aft1BhTileG s K V s_qh BT BD (c + 1))) := by
    unfold accumulate
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (aft1_bo_full_eqG s st Q K V s_qh scale BT BD c ho hq hh))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (aft1_bh_succ_eqG s _ K V s_qh BT BD c (by simp [hh]) (by simp [hk]) (by simp [hv])))]
    exact stepStmts.nil
  cases IFCOND with
  | «false» =>
    simpa only [update, aft1_ifThenElse_false_else, afterUpdate, Bool.false_eq_true,
      false_and, ↓reduceIte] using hacc
  | «true» =>
    have houter (thenB elseB : List Stmt) :
        stepStmt (.ifThenElse (.constBool Bool.true) thenB elseB) st = stepStmts thenB st := by
      simp [stepStmt, evalOp]
    rw [update, houter]
    have heq : evalOp (.eq .nat .nil (.ref .nat [] "i") (.constNat 0)) st =
        some (Tile.scalar (decide (c = 0))) := by
      simp only [evalOp, hi]
      rfl
    by_cases hc : c = 0
    · subst c
      have hfirst := first_carry s st K V s_qh BT BD hk hv
      simp only [stepStmts, stepStmt, heq, decide_true, bind, Option.bind_some,
        Tile.scalar, hfirst, Nat.zero_add, afterUpdate, and_self, ↓reduceIte]
      erw [hfirst]
      rfl
    · simp only [stepStmts, stepStmt, heq, hc, decide_false, bind, Option.bind_some,
        Tile.scalar, Bool.false_eq_true, afterUpdate, and_false, ↓reduceIte]
      rw [hacc]

theorem afterUpdate_output (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool)
    (ho : st.regs .real [BT, BD] "b_o" = some (aft1BoLocalTileG s Q K V s_qh scale BT BD c)) :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).regs .real [BT, BD] "b_o" =
      some (aft1BoTileG s Q K V s_qh scale BT BD c) := by
  unfold afterUpdate
  split_ifs with h
  · rcases h with ⟨_, rfl⟩
    simpa [first_output] using ho
  · simp

@[simp] theorem afterUpdate_mem (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool) :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).mem = st.mem := by
  unfold afterUpdate
  split_ifs <;> rfl

@[simp] theorem afterUpdate_readMem (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool) (rg : RegionName) (off : Nat) :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).readMem rg off = st.readMem rg off := by
  simp only [BlockState.readMem, afterUpdate_mem]

@[simp] theorem afterUpdate_pids (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool) :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).pids = st.pids := by
  unfold afterUpdate
  split_ifs <;> rfl

@[simp] theorem afterUpdate_carry (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool) :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).regs .real [BD, BD] "b_h" =
      some (aft1BhTileG s K V s_qh BT BD (c + 1)) := by
  simp only [afterUpdate, BlockState.setReg_same]

@[simp] theorem afterUpdate_otherReg (s st : BlockState) (Q K V : RegionName)
    (s_qh : Nat) (scale : ℝ) (BT BD c : Nat) (IFCOND : Bool)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (ho : name ≠ "b_o") (hh : name ≠ "b_h") :
    (afterUpdate s st Q K V s_qh scale BT BD c IFCOND).regs dtype shape name =
      st.regs dtype shape name := by
  unfold afterUpdate
  split_ifs <;> simp [ho, hh]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealBranch

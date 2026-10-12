import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealProbabilities
import VeriTile.Triton.Semantics.BlockPtrEval

/-! Real execution of K/V loads and the source accumulation suffix. Block
pointer metadata is retained explicitly, so later IO proofs can supply the
actual strides, bounds and initial offsets instead of assuming loaded outputs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccumulation
open VeriTile.Triton
open RealTiles RealScores
open scoped BigOperators
noncomputable section

def checks (even : Nat) : List Nat := if even ≠ 0 then [] else [0, 1]

/-- The real memory view read by a block pointer; checked out-of-bounds lanes
use the interpreter's zero boundary value. Active-address safety is separate. -/
def values {shape : TileShape} (s : BlockState) (ptrs : Tile .blockPtr shape)
    (checked : List Nat) : TileIndex shape → ℝ := fun i =>
  let bp := ptrs.data i
  let index := TileShape.indexToList shape i
  if bp.inBounds index checked then s.readMem bp.region (bp.address index) else 0

theorem eval_load (s : BlockState) (shape : TileShape) (name : RegName)
    (ptrs : Tile .blockPtr shape) (checked : List Nat)
    (hp : s.regs .blockPtr shape name = some ptrs) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr shape name) checked) .none) s =
      some (finite (values s ptrs checked)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  simp only [finite, values]
  split <;> simp_all [BlockState.readMemValue_real, BlockState.defaultCarrier]

/-- Without boundary-check axes every lane is active, so no padding value
is selected. Preserve the source's unspecified tag without assuming it is zero. -/
theorem eval_unchecked_load (s : BlockState) (shape : TileShape) (name : RegName)
    (ptrs : Tile .blockPtr shape) (hp : s.regs .blockPtr shape name = some ptrs)
    (padding : PaddingOption) :
    evalOp (.load .real (.blockPtr (.ref .blockPtr shape name) []) .none padding) s =
      some (finite (values s ptrs [])) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  simp [finite, values, BlockPtr.inBounds, BlockState.readMemValue_real]

theorem load_choice_executes (s : BlockState) (shape : TileShape)
    (pointer target : RegName) (even : Nat) (ptrs : Tile .blockPtr shape)
    (hp : s.regs .blockPtr shape pointer = some ptrs) :
    stepStmt (.ifThenElse (.ne .nat .nil (.constNat even) (.constNat 0))
      [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) []) .none .unspecified)]
      [.assign .real shape target (.load .real (.blockPtr (.ref .blockPtr shape pointer) [0, 1]) .none)]) s =
      some (s.setReg target .real shape (finite (values s ptrs (checks even)))) := by
  conv_lhs => unfold stepStmt
  simp only [evalOp.eq_def, bind, Option.bind, Tile.cop_data, Tile.scalar, ComparableDType.ne,
    decide_eq_true_eq]
  by_cases h : even ≠ 0
  · rw [if_pos h, stepStmts.cons_some (stepStmt_assign_eq_some (eval_unchecked_load s shape pointer ptrs hp .unspecified)),
      stepStmts.nil]
    simp [checks, h]
  · rw [if_neg h, stepStmts.cons_some (stepStmt_assign_eq_some (eval_load s shape pointer ptrs [0, 1] hp)),
      stepStmts.nil]
    simp [checks, h]

theorem load_key_executes (s : BlockState) (BD BN even : Nat)
    (ptrs : Tile .blockPtr [BD, BN])
    (hp : s.regs .blockPtr [BD, BN] "K_block_ptr" = some ptrs) :
    stepStmt (RealBody.loadKey BD BN even) s =
      some (s.setReg "k" .real [BD, BN] (finite (values s ptrs (checks even)))) :=
  load_choice_executes s [BD, BN] "K_block_ptr" "k" even ptrs hp

theorem eval_accumulate (s : BlockState) (BM BN BD : Nat)
    (acc : TileIndex [BM, BD] → ℝ) (p : TileIndex [BM, BN] → ℝ) (v : TileIndex [BN, BD] → ℝ)
    (ha : s.regs .real [BM, BD] "acc" = some (finite acc))
    (hp : s.regs .real [BM, BN] "p" = some (finite p))
    (hv : s.regs .real [BN, BD] "v" = some (finite v)) :
    evalOp (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BD] "acc")
      (.dot (batch := []) (.ref .real [BM, BN] "p") (.ref .real [BN, BD] "v"))) s =
      some (finite (fun i : TileIndex [BM, BD] => acc i +
        ∑ j : Fin BN, p (i.1, j, PUnit.unit) * v (j, i.2.1, PUnit.unit))) := by
  simp only [evalOp.eq_def, List.nil_append, ha, hp, hv, bind, Option.bind]
  erw [dot_finite BM BD BN p v]
  rfl

/-- Execute all five source accumulation statements: the actual V load,
P/V dot, maximum carry and both pointer advances. The output value is a sum
over the memory view, not a premise giving the result of a kernel execution. -/
theorem accumulate_executes (s : BlockState) (BM BN BD even : Nat)
    (acc : TileIndex [BM, BD] → ℝ) (p : TileIndex [BM, BN] → ℝ) (mij : Tile .real [BM])
    (kp : Tile .blockPtr [BD, BN]) (vp : Tile .blockPtr [BN, BD])
    (ha : s.regs .real [BM, BD] "acc" = some (finite acc))
    (hp : s.regs .real [BM, BN] "p" = some (finite p))
    (hm : s.regs .real [BM] "m_ij" = some mij)
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp) :
    ∃ s', stepStmts (RealBody.accumulate BM BN BD even) s = some s' ∧
      s'.regs .real [BM, BD] "acc" = some (finite (fun i : TileIndex [BM, BD] => acc i +
        ∑ j : Fin BN, p (i.1, j, PUnit.unit) * values s vp (checks even) (j, i.2.1, PUnit.unit))) ∧
      s'.regs .real [BM] "m_i" = some mij ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" =
        some (⟨fun i => (kp.data i).advance [0, BN]⟩ : Tile .blockPtr [BD, BN]) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" =
        some (⟨fun i => (vp.data i).advance [BN, 0]⟩ : Tile .blockPtr [BN, BD]) ∧
      Frame ["v", "acc", "m_i", "V_block_ptr", "K_block_ptr"] s s' := by
  let v := values s vp (checks even)
  let sv := s.setReg "v" .real [BN, BD] (finite v)
  let next := fun i : TileIndex [BM, BD] => acc i +
    ∑ j : Fin BN, p (i.1, j, PUnit.unit) * v (j, i.2.1, PUnit.unit)
  let sa := sv.setReg "acc" .real [BM, BD] (finite next)
  let sm := sa.setReg "m_i" .real [BM] mij
  let newV : Tile .blockPtr [BN, BD] := ⟨fun i => (vp.data i).advance [BN, 0]⟩
  let sp := sm.setReg "V_block_ptr" .blockPtr [BN, BD] newV
  have hload := load_choice_executes s [BN, BD] "V_block_ptr" "v" even vp hv
  have hadd := eval_accumulate sv BM BN BD acc p v (by simp [sv, ha]) (by simp [sv, hp]) (by simp [sv])
  have hmi : evalOp (.ref .real [BM] "m_ij") sa = some mij := by simp [sa, sv, hm]
  have hvp : evalOp (.advanceBlockPtr (.ref .blockPtr [BN, BD] "V_block_ptr") [BN, 0]) sm =
      some newV := by simp [sm, sa, sv, hv, newV]
  have hkp : evalOp (.advanceBlockPtr (.ref .blockPtr [BD, BN] "K_block_ptr") [0, BN]) sp =
      some (⟨fun i => (kp.data i).advance [0, BN]⟩ : Tile .blockPtr [BD, BN]) := by
    simp [sp, sm, sa, sv, hk]
  simp only [RealBody.accumulate]
  rw [stepStmts.cons_some hload, stepStmts.cons_some (stepStmt_assign_eq_some hadd),
    stepStmts.cons_some (stepStmt_assign_eq_some hmi), stepStmts.cons_some (stepStmt_assign_eq_some hvp),
    stepStmts.cons_some (stepStmt_assign_eq_some hkp), stepStmts.nil]
  refine ⟨_, rfl, by simp [sp, sm, sa, next, v], by simp [sp, sm], by simp, by simp [sp, newV], ?_⟩
  exact ((((Frame.setReg _ s "v" _ _ _ (by simp)).trans
    (Frame.setReg _ sv "acc" _ _ _ (by simp))).trans
    (Frame.setReg _ sa "m_i" _ _ _ (by simp))).trans
    (Frame.setReg _ sm "V_block_ptr" _ _ _ (by simp))).trans
    (Frame.setReg _ sp "K_block_ptr" _ _ _ (by simp))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccumulation

import bench.tritonbench_g.kldiv_ops.Proofs.BackwardStep
import bench.tritonbench_g.kldiv_ops.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm FullBackward
open scoped BigOperators
set_option maxHeartbeats 1000000

def body (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) (reduction : Nat) :
    List Stmt :=
  match ((forward Y T O sy st so N B eps logTarget reduction).body[6]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []

noncomputable def value {B : Nat} (s : BlockState) (R : RegionName)
    (base N : Nat) (current : Int) (j : TileIndex [B]) : ℝ :=
  if active N current j then s.readMem R (base + (offset current j).toNat) else 0

noncomputable def lossValue {B : Nat} (s : BlockState) (Y T : RegionName)
    (yBase tBase N : Nat) (current : Int) (eps : ℝ) (logTarget : Bool) (j : TileIndex [B]) : ℝ :=
  FullMath.loss logTarget eps (value s Y yBase N current j) (value s T tBase N current j)

noncomputable def loadedState {B : Nat} (s : BlockState) (Y T : RegionName)
    (yBase tBase N : Nat) (current : Int) (eps : ℝ) (logTarget : Bool) : BlockState :=
  ((((s.setReg "offsets" .int [B] ⟨offset current⟩).setReg "mask" .bool [B]
    ⟨active N current⟩).setReg "y" .real [B]
    ⟨fun j ↦ some (value s Y yBase N current j)⟩).setReg "y_true" .real [B]
    ⟨fun j ↦ some (value s T tBase N current j)⟩).setReg "loss" .real [B]
    ⟨fun j ↦ some (lossValue s Y T yBase tBase N current eps logTarget j)⟩

/-- Both masked loads and the chosen scalar formula run on the actual signed-offset state. -/
theorem load_compute (s : BlockState) (Y T O : RegionName)
    (sy st so N B yBase tBase : Nat) (current : Int) (eps : ℝ) (logTarget : Bool) (reduction : Nat)
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)) :
    stepStmts ((body Y T O sy st so N B eps logTarget reduction).take 5) s =
      some (loadedState (B := B) s Y T yBase tBase N current eps logTarget) := by
  cases logTarget <;>
    simp [body, forward, forwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp.eq_def, hI, hY, hT, hBase,
      loadedState, lossValue, value, FullMath.loss, offset, active,
      Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop, Tile.scalar, Tile.vec, -Tile.scalar_eta,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      BlockState.readMemValue_real, Region.cast, Except.bind, bind]
  all_goals norm_num [WithBot.realExp, WithBot.realLog, offset, active]
  all_goals simp only [apply_ite]
  all_goals simp [WithBot.realExp, WithBot.realLog, offset, active]
  all_goals congr 1
  all_goals ext j
  all_goals dsimp
  all_goals split_ifs <;> simp_all [WithBot.realExp, WithBot.realLog, max_def]
  all_goals simp_all only [WithBot.some_eq_coe, WithBot.coe_lt_coe, WithBot.coe_le_coe]
  all_goals simp_all
  all_goals split_ifs <;> first | rfl | (exfalso; linarith)

/-- The scalar reduction is the sum of the actual computed block lanes, including padding. -/
theorem sum_eval {B : Nat} (s : BlockState) (f : Fin B → ℝ)
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (f j.1)⟩) :
    evalOp (.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (.ref .real [B] "loss")) s =
      some (Tile.scalar (some (∑ j : Fin B, f j))) := by
  rw [evalOp_reduceSum, evalOp_ref, hLoss]
  apply congrArg some
  ext idx
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  change @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
    (fun j ↦ (some (f j) : WithBot ℝ)) = some (∑ j : Fin B, f j)
  exact withBot_sum_some _

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms

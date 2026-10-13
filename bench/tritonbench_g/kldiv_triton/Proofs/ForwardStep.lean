import bench.tritonbench_g.kldiv_triton.Proofs.BackwardStep
import bench.tritonbench_g.kldiv_triton.Proofs.Math
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton FullAlgorithm FullBackward
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def body (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    List Stmt :=
  match ((forward Y T O sy st so N B logTarget reduction).body[5]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []

noncomputable def value {B : Nat} (s : BlockState) (R : RegionName)
    (base N : Nat) (current : Int) (j : TileIndex [B]) : ℝ :=
  if active N current j then s.readMem R (base + (offset current j).toNat) else 0

noncomputable def lossValue {B : Nat} (s : BlockState) (Y T : RegionName)
    (yBase tBase N : Nat) (current : Int) (logTarget : Bool) (j : TileIndex [B]) : ℝ :=
  FullMath.loss logTarget (value s Y yBase N current j) (value s T tBase N current j)

noncomputable def loadedState {B : Nat} (s : BlockState) (Y T : RegionName)
    (yBase tBase N : Nat) (current : Int) (logTarget : Bool) : BlockState :=
  ((((s.setReg "offsets" .int [B] ⟨offset current⟩).setReg "mask" .bool [B]
    ⟨active N current⟩).setReg "y" .real [B]
    ⟨fun j ↦ some (value s Y yBase N current j)⟩).setReg "y_true" .real [B]
    ⟨fun j ↦ some (value s T tBase N current j)⟩).setReg "loss" .real [B]
    ⟨fun j ↦ some (lossValue s Y T yBase tBase N current logTarget j)⟩

/-- Both masked loads and the chosen scalar formula run on the actual signed-offset state. -/
theorem load_compute (s : BlockState) (Y T O : RegionName)
    (sy st so N B yBase tBase : Nat) (current : Int) (logTarget : Bool) (reduction : Nat)
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)) :
    stepStmts ((body Y T O sy st so N B logTarget reduction).take 5) s =
      some (loadedState (B := B) s Y T yBase tBase N current logTarget) := by
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
  all_goals split_ifs <;> simp_all [WithBot.realExp, WithBot.realLog]

/-- The scalar reduction is the sum of the actual computed block lanes, including padding. -/
theorem sum_eval {B : Nat} (s : BlockState) (f : Fin B → ℝ)
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (f j.1)⟩) :
    evalOp (.reduceSum (0 : Fin [B].length) Bool.false (.ref .real [B] "loss")) s =
      some (Tile.scalar (some (∑ j : Fin B, f j))) := by
  rw [evalOp_reduceSum, evalOp_ref, hLoss]
  apply congrArg some
  ext idx
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  change @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
    (fun j ↦ (some (f j) : WithBot ℝ)) = some (∑ j : Fin B, f j)
  exact withBot_sum_some _


def finish (N B reduction : Nat) : List Stmt :=
  [.ifThenElse (.eq .nat .nil (.constNat reduction) (.constNat 0))
    [.store .real [B]
      (.ptr (.ptrAdd .scalarL (.ref .ptr [] "O") (.castIntToNat (.ref .int [B] "offsets"))))
      (.ref .real [B] "loss") (.mask (.ref .bool [B] "mask"))]
    [.assign .real [] "loss"
      (.reduceSum (0 : Fin [B].length) Bool.false (.ref .real [B] "loss")),
     .store .real [] (.ptr (.ref .ptr [] "O")) (.ref .real [] "loss") .none,
     .assign .ptr [] "O" (.ptrAdd .nil (.ref .ptr [] "O") (.constNat 1))]]

theorem finish_source (Y T O : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) :
    (body Y T O sy st so N B logTarget reduction).drop 5 = finish N B reduction := by
  rfl

noncomputable def reducedState {B : Nat} (s : BlockState) (O : RegionName)
    (base : Nat) (values : Fin B → ℝ) : BlockState :=
  ((s.setReg "loss" .real [] (Tile.scalar (some (∑ j : Fin B, values j)))).writeMem O base
    (∑ j : Fin B, values j)).setReg "O" .ptr [] (Tile.scalar (O, base + 1))

/-- Every nonzero reduction mode performs an unmasked scalar store followed by pointer advance. -/
theorem reduced_finish (s : BlockState) (O : RegionName) (N B base reduction : Nat)
    (values : Fin B → ℝ) (hReduction : reduction ≠ 0)
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, base)))
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (values j.1)⟩) :
    stepStmts (finish N B reduction) s = some (reducedState s O base values) := by
  simp [finish, stepStmts, stepStmt, evalOp.eq_def, hLoss, hO, hReduction,
    ComparableDType.eq, Tile.bop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
    reducedState, BlockState.writeMemTyped_real, Region.cast,
    TileShape.allIndices, List.finRange_succ, Option.bind_eq_bind, Option.bind]
  simp [Tile.reduceSumDrop, TileShape.insertAxisIndex, TileShape.axisDim,
    Tile.scalar, -Tile.scalar_eta]
  congr 1

/-- Cached block values are stored elementwise only at the original source mask. -/
theorem element_finish (s : BlockState) (O : RegionName) (N B base : Nat)
    (current : Int) (values : TileIndex [B] → ℝ)
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, base)))
    (hOffsets : s.regs .int [B] "offsets" = some ⟨offset current⟩)
    (hMask : s.regs .bool [B] "mask" = some ⟨active N current⟩)
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (values j)⟩) :
    stepStmts (finish N B 0) s = some (scatter s O base N current values) := by
  simp [finish, stepStmts, stepStmt, evalOp.eq_def, hO, hOffsets, hMask, hLoss,
    scatter, ComparableDType.eq, Tile.bop, Tile.uop, Tile.ptrAdd,
    Tile.scalar, -Tile.scalar_eta, BlockState.writeMemTyped_real, Region.cast]

noncomputable def resultState {B : Nat} (s : BlockState) (Y T O : RegionName)
    (yBase tBase oBase N : Nat) (current : Int) (logTarget : Bool) (reduction : Nat) :
    BlockState :=
  let middle := loadedState (B := B) s Y T yBase tBase N current logTarget
  let values := lossValue (B := B) s Y T yBase tBase N current logTarget
  if reduction = 0 then scatter middle O oBase N current values
  else reducedState middle O oBase (fun j ↦ values (j, PUnit.unit))

/-- Successful execution of the whole source block, including the chosen actual store path. -/
theorem step (s : BlockState) (Y T O : RegionName)
    (sy st so N B yBase tBase oBase : Nat) (current : Int) (logTarget : Bool) (reduction : Nat)
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, oBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val)) :
    stepStmts (body Y T O sy st so N B logTarget reduction) s =
      some (resultState (B := B) s Y T O yBase tBase oBase N current logTarget reduction) := by
  rw [← List.take_append_drop 5 (body Y T O sy st so N B logTarget reduction),
    ProgramRewrite.append, load_compute s Y T O sy st so N B yBase tBase current
      logTarget reduction hI hY hT hBase, Option.bind_some, finish_source]
  by_cases hr : reduction = 0
  · subst reduction
    exact element_finish _ O N B oBase current _
      (by simpa [loadedState] using hO) (by simp [loadedState])
      (by simp [loadedState]) (by simp [loadedState])
  · simpa only [resultState, hr, if_false] using
      reduced_finish (loadedState (B := B) s Y T yBase tBase N current logTarget)
        O N B oBase reduction
        (fun j ↦ lossValue s Y T yBase tBase N current logTarget (j, PUnit.unit)) hr
        (by simpa [loadedState] using hO) (by simp [loadedState])

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms

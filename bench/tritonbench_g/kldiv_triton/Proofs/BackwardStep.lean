import bench.tritonbench_g.kldiv_triton.Proofs.Algorithm
import VeriTile.Triton.KernelLemmas.ScatterStore
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 1000000

/-- The independently stated backward formula, before host gradient scaling. -/
noncomputable def gradient (logTarget : Bool) (target : ℝ) : ℝ :=
  if logTarget then -Real.exp target else -target

def body (T O : RegionName) (st so N B : Nat) (logTarget : Bool) : List Stmt :=
  match ((backward T O st so N B logTarget).body[5]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []

def offset {B : Nat} (current : Int) (j : TileIndex [B]) : Int := current + j.1.val

def active {B : Nat} (N : Nat) (current : Int) (j : TileIndex [B]) : Bool :=
  decide (offset current j < (N : Int))

noncomputable def targetValue {B : Nat} (s : BlockState) (T : RegionName)
    (targetBase N : Nat) (current : Int) (j : TileIndex [B]) : ℝ :=
  if active N current j then s.readMem T (targetBase + (offset current j).toNat) else 0

noncomputable def loadedState {B : Nat} (s : BlockState) (T : RegionName)
    (targetBase N : Nat) (current : Int) (logTarget : Bool) : BlockState :=
  (((s.setReg "offsets" .int [B] ⟨offset current⟩).setReg "mask" .bool [B]
    ⟨active N current⟩).setReg "target" .real [B]
    ⟨fun j ↦ some (targetValue s T targetBase N current j)⟩).setReg "res" .real [B]
    ⟨fun j ↦ some (gradient logTarget (targetValue s T targetBase N current j))⟩

noncomputable def scatter {B : Nat} (s : BlockState) (O : RegionName)
    (outputBase N : Nat) (current : Int) (values : TileIndex [B] → ℝ) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun state j ↦
    if active N current j then
      state.writeMem O (outputBase + (offset current j).toNat) (values j)
    else state) s

/-- Actual masked loads, branch arithmetic and stores for one complete column block. -/
theorem step (s : BlockState) (T O : RegionName) (st so N B targetBase outputBase : Nat)
    (current : Int) (logTarget : Bool)
    (hI : s.regs .int [] "i" = some (Tile.scalar current))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, targetBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase))) :
    stepStmts (body T O st so N B logTarget) s =
      some (scatter (B := B) (loadedState (B := B) s T targetBase N current logTarget)
        O outputBase N current
        (fun j ↦ gradient logTarget (targetValue s T targetBase N current j))) := by
  cases logTarget <;>
    simp [body, backward, backwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      stepStmts, stepStmt, evalOp.eq_def, hI, hT, hO,
      scatter, loadedState, targetValue, gradient, offset, active,
      Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop, Tile.scalar, -Tile.scalar_eta,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      BlockState.writeMemTyped_real, BlockState.readMemValue_real,
      Region.cast, Except.bind, bind]
  all_goals norm_num [WithBot.realExp, offset, active]
  all_goals simp only [apply_ite]
  all_goals simp [WithBot.realExp, offset, active]
  all_goals try rfl
  congr 1
  · funext state j
    split_ifs <;> simp_all
  · congr 1
    ext j
    dsimp
    split_ifs <;> simp_all

end VeriTile.Bench.TritonBenchG.KldivTriton.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms

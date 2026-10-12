import bench.examples.LogExp.Kernels
import VeriTile.Triton.Memory.KernelSpec.Basic
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for LogExp.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.Correct
open Triton
open scoped VeriTile.Triton.KernelIO₁

set_option maxHeartbeats 1600000

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

theorem original_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (originalKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [originalKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

theorem optimized_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (optimizedKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [optimizedKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

theorem original_traceSafe (xReg yReg : RegionName) (B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (originalKernel xReg yReg B).toAlgKernel s := by
  simp only [BlockState.pid_eq] at hx hy
  simp [Kernel.TraceSafe, originalKernel, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

theorem optimized_traceSafe (xReg yReg : RegionName) (B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (optimizedKernel xReg yReg B).toAlgKernel s := by
  simp only [BlockState.pid_eq] at hx hy
  simp [Kernel.TraceSafe, optimizedKernel, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

theorem original_run (xReg yReg : RegionName) (B : Nat)
    (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem xReg (s.pid * B + i.val) = xs i) :
    ∃ t, exec (originalKernel xReg yReg B).toAlgKernel s = some t ∧
      (∀ i : Fin B, t.readMem yReg (s.pid * B + i.val) = xs i) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [originalKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hx, FloatDType.cast]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = yReg
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem optimized_run (xReg yReg : RegionName) (B : Nat)
    (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem xReg (s.pid * B + i.val) = xs i) :
    ∃ t, exec (optimizedKernel xReg yReg B).toAlgKernel s = some t ∧
      (∀ i : Fin B, t.readMem yReg (s.pid * B + i.val) = xs i) ∧
      (∀ (r : RegionName) o, (r ≠ yReg ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [B] => s.pids 0 * B + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hx ⊢
  simp [optimizedKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hx, WithBot.realExp, WithBot.realLog, FloatDType.cast]
    split <;> simp_all
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = yReg
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

end VeriTile.Bench.Examples.LogExp.Correct

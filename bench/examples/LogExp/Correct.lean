import bench.examples.LogExp.Kernels
import VeriTile.Triton.Memory.KernelSpec.Basic
import VeriTile.Meta.StatementAudit

/-!
Real correctness of the two programs in Kernels.lean: output[i] = input[i].
KernelIO₁ interprets their mathematical projection, erasing compute precision;
this proof uses the real log/exp identities and no admitted FP assumptions.
Both specifications include successful execution and preservation of memory
outside the output tile, for every block size and in-bounds program window.
-/

noncomputable section
namespace VeriTile.Bench.Examples.LogExp.Correct
open Triton
open scoped VeriTile.Triton.KernelIO₁

set_option maxHeartbeats 1600000

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- The original source, under the real interpretation of KernelIO₁. -/
def originalIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ where
  kernel := originalKernel xReg yReg B
  inp := xReg
  out := yReg
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The piecewise candidate shares its source with the FP proof. -/
def optimizedIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ :=
  { originalIO xReg yReg B with kernel := optimizedKernel xReg yReg B, projection := by rfl }

private theorem original_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (originalKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [originalKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

private theorem optimized_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (optimizedKernel xReg yReg B).toAlgKernel.FlattenOk := by
  simp [optimizedKernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

private theorem original_traceSafe (xReg yReg : RegionName) (B : Nat)
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

private theorem optimized_traceSafe (xReg yReg : RegionName) (B : Nat)
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

private theorem original_run (xReg yReg : RegionName) (B : Nat)
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

private theorem optimized_run (xReg yReg : RegionName) (B : Nat)
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

/-- The original log-exp source implements the identity function over ℝ. -/
specification original_correct (xReg yReg : RegionName) (B : Nat) :
    Spec.Real (originalIO xReg yReg B ⊨ fun xs i => xs i) := by
  refine KernelIO₁.Implements.intro _ (original_flattenOk xReg yReg B) ?_ ?_
  · intro bounds s hx hy _
    exact original_traceSafe xReg yReg B bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := original_run xReg yReg B s xs hx
    exact ⟨t, ht, hv, fun r o ho _ => hf r o ho⟩

/-- The piecewise candidate implements the same identity function over ℝ. -/
specification optimized_correct (xReg yReg : RegionName) (B : Nat) :
    Spec.Real (optimizedIO xReg yReg B ⊨ fun xs i => xs i) := by
  refine KernelIO₁.Implements.intro _ (optimized_flattenOk xReg yReg B) ?_ ?_
  · intro bounds s hx hy _
    exact optimized_traceSafe xReg yReg B bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := optimized_run xReg yReg B s xs hx
    exact ⟨t, ht, hv, fun r o ho _ => hf r o ho⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.LogExp.Correct

import bench.examples.FloatDTypeSoftmax.Kernels
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Triton.Math.Softmax
import VeriTile.Meta.StatementAudit

/-!
Real correctness of the same fp32-load, fp64-work implementations used by
FPEquiv.lean. Erasure removes precision and casts, but retains the explicit
`x32` load and `x` assignment. Both versions compute exp(x[i]) / sum(exp(x)).
-/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect
open VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

private theorem erased_projection (ck : ComputeKernel)
    (h : ck.toAlgorithm? = .ok ck.toAlgKernel) :
    ck.eraseDType.toAlgorithm? = .ok ck.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h]
  simp

/-- The erased fp32 divide kernel's projected body, pinned to its literal
statement list: the plain per-element-divide stable softmax — nine
statements (`pid`, `offs`, `x32`-load, `x`-copy, running `max`, `exp`, `sum`, `y = e/s`,
raw store), with the `.to(tl.float64)`/`.to(tl.float32)` casts erased. Every
obligation below rewrites the erased kernel's projection along this equation
first, then runs the plain computational walk on the literal body. -/
private theorem floatStable_erased_toAlg (xReg yReg : RegionName) (N : Nat) :
    (floatStableSoftmaxKernel xReg yReg N).eraseDType.toAlgKernel
      = Kernel.mk [xReg] [yReg]
          [Stmt.assign TileDType.nat [] "pid" (Op.programId 0),
           Stmt.assign TileDType.nat [N] "offs"
             (Op.add NumericDType.nat Broadcast.scalarL
               (Op.mul NumericDType.nat Broadcast.nil
                 (Op.ref TileDType.nat [] "pid") (Op.constNat N))
               (Op.arange N)),
           Stmt.assign TileDType.real [N] "x32"
             (Op.load TileDType.real
               (MemAccess.region xReg (Op.ref TileDType.nat [N] "offs"))
               MaskOpt.none),
           Stmt.assign TileDType.real [N] "x" (Op.ref TileDType.real [N] "x32"),
           Stmt.assign TileDType.real [] "m"
             (Op.reduceMax (shape := [N]) (0 : Fin (0 + 1)) Bool.false
               (Op.ref TileDType.real [N] "x")),
           Stmt.assign TileDType.real [N] "e"
             (Op.libdeviceExp (Op.sub NumericDType.real Broadcast.scalarR
               (Op.ref TileDType.real [N] "x") (Op.ref TileDType.real [] "m"))),
           Stmt.assign TileDType.real [] "s"
             (Op.reduceSum (shape := [N]) (0 : Fin (0 + 1)) Bool.false
               (Op.ref TileDType.real [N] "e")),
           Stmt.assign TileDType.real [N] "y"
             (Op.div NumericDType.real Broadcast.scalarR
               (Op.ref TileDType.real [N] "e") (Op.ref TileDType.real [] "s")),
           Stmt.store TileDType.real [N]
             (MemAccess.region yReg (Op.ref TileDType.nat [N] "offs"))
             (Op.ref TileDType.real [N] "y") MaskOpt.none] := by
  simp [floatStableSoftmaxKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]
    try rfl

/-- The erased fp32 reciprocal kernel's projected body, pinned to its literal
statement list: the plain reciprocal-form stable softmax — ten statements
(the same prefix, then `inv_s = 1/s`, `y = e·inv_s`, raw store), casts
erased. -/
private theorem floatRecip_erased_toAlg (xReg yReg : RegionName) (N : Nat) :
    (floatSoftmaxRecipKernel xReg yReg N).eraseDType.toAlgKernel
      = Kernel.mk [xReg] [yReg]
          [Stmt.assign TileDType.nat [] "pid" (Op.programId 0),
           Stmt.assign TileDType.nat [N] "offs"
             (Op.add NumericDType.nat Broadcast.scalarL
               (Op.mul NumericDType.nat Broadcast.nil
                 (Op.ref TileDType.nat [] "pid") (Op.constNat N))
               (Op.arange N)),
           Stmt.assign TileDType.real [N] "x32"
             (Op.load TileDType.real
               (MemAccess.region xReg (Op.ref TileDType.nat [N] "offs"))
               MaskOpt.none),
           Stmt.assign TileDType.real [N] "x" (Op.ref TileDType.real [N] "x32"),
           Stmt.assign TileDType.real [] "m"
             (Op.reduceMax (shape := [N]) (0 : Fin (0 + 1)) Bool.false
               (Op.ref TileDType.real [N] "x")),
           Stmt.assign TileDType.real [N] "e"
             (Op.libdeviceExp (Op.sub NumericDType.real Broadcast.scalarR
               (Op.ref TileDType.real [N] "x") (Op.ref TileDType.real [] "m"))),
           Stmt.assign TileDType.real [] "s"
             (Op.reduceSum (shape := [N]) (0 : Fin (0 + 1)) Bool.false
               (Op.ref TileDType.real [N] "e")),
           Stmt.assign TileDType.real [] "inv_s"
             (Op.div NumericDType.real Broadcast.nil (Op.const 1)
               (Op.ref TileDType.real [] "s")),
           Stmt.assign TileDType.real [N] "y"
             (Op.mul NumericDType.real Broadcast.scalarR
               (Op.ref TileDType.real [N] "e") (Op.ref TileDType.real [] "inv_s")),
           Stmt.store TileDType.real [N]
             (MemAccess.region yReg (Op.ref TileDType.nat [N] "offs"))
             (Op.ref TileDType.real [N] "y") MaskOpt.none] := by
  simp [floatSoftmaxRecipKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]
    try rfl

private theorem div_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (floatStableSoftmaxKernel xReg yReg B).eraseDType.toAlgKernel.FlattenOk := by
  rw [floatStable_erased_toAlg]
  simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
private theorem div_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (floatStableSoftmaxKernel xReg yReg B).eraseDType.toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  rw [floatStable_erased_toAlg]
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

private theorem div_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (floatStableSoftmaxKernel "x" "y" B).eraseDType.toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hinj : Function.Injective (fun i : TileIndex [n+1] => s.pids 0 * (n+1) + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  rw [floatStable_erased_toAlg]
  simp only [BlockState.pid_eq] at hx ⊢
  simp [exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp only [hx]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

private theorem recip_flattenOk (xReg yReg : RegionName) (B : Nat) :
    (floatSoftmaxRecipKernel xReg yReg B).eraseDType.toAlgKernel.FlattenOk := by
  rw [floatRecip_erased_toAlg]
  simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
private theorem recip_traceSafe (xReg yReg : RegionName) (B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds (floatSoftmaxRecipKernel xReg yReg B).eraseDType.toAlgKernel s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  rw [floatRecip_erased_toAlg]
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hx hy
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hx,
    fun a => lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) hy⟩

private theorem recip_region_run (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * B + i.val) = xs i) :
    ∃ s1, exec (floatSoftmaxRecipKernel "x" "y" B).eraseDType.toAlgKernel s = some s1 ∧
      (∀ i : Fin B, s1.readMem "y" (s.pid * B + i.val) = naiveSpec xs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin B, o ≠ s.pid * B + i.val) →
        s1.mem r o = s.mem r o) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hinj : Function.Injective (fun i : TileIndex [n+1] => s.pids 0 * (n+1) + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  rw [floatRecip_erased_toAlg]
  simp only [BlockState.pid_eq] at hx ⊢
  simp [exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp only [hx, ← div_eq_mul_inv]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

def divIO (B : Nat) : KernelIO₁ where
  kernel := (floatStableSoftmaxKernel "x" "y" B).eraseDType
  projection := erased_projection _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_div_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (divIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact div_flattenOk "x" "y" B
  · intro bounds s hx hy _
    exact div_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := div_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (floatSoftmaxRecipKernel "x" "y" B).eraseDType
  projection := erased_projection _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_recip_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (recipIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact recip_flattenOk "x" "y" B
  · intro bounds s hx hy _
    exact recip_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := recip_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect

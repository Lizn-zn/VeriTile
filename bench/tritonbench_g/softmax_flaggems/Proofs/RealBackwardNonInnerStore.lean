import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def pass2Body (P G Y : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((branch P G Y M N K B C Bool.false)[7]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | some (.forRangeDyn _ _ _ _ body) => body
  | _ => []

abbrev tileActive := @RealForwardNonInner.tileActive
noncomputable abbrev scatter := @RealForwardNonInner.scatter

noncomputable def storeValue {B C : Nat} (s : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (z : Fin C → ℝ) (i : TileIndex [B,C]) : ℝ :=
  tileSample s P row N K kbase start i *
    (tileSample s G row N K kbase start i - z i.2.1)

noncomputable def storeMid {B C : Nat} (s : BlockState) (P G : RegionName)
    (row N K kbase start : Nat) (z : Fin C → ℝ) : BlockState :=
  let a := s.setReg "mask" .bool [B,C] ⟨tileActive N K kbase start⟩
  let b := a.setReg "out_tile" .real [B,C] ⟨fun i ↦ some (tileSample s P row N K kbase start i)⟩
  let c := b.setReg "out_grad_tile" .real [B,C] ⟨fun i ↦ some (tileSample s G row N K kbase start i)⟩
  c.setReg "in_grad_tile" .real [B,C] ⟨fun i ↦ some (storeValue s P G row N K kbase start z i)⟩

noncomputable def advance {B C : Nat} (s : BlockState)
    (row N K kbase start : Nat) : BlockState :=
  (s.setReg "offsets_n" .nat [B] ⟨fun i ↦ start+B+i.1.val⟩).setReg
    "offsets" .nat [B,C] ⟨tileOffset row N K kbase (start+B)⟩

theorem pass2_step (P G Y : RegionName) (M N K B C row kbase start : Nat)
    (s : BlockState) (z : Fin C → ℝ)
    (hk : s.regs .nat [C] "offsets_k" = some ⟨fun i ↦ kbase+i.1.val⟩)
    (hn : s.regs .nat [B] "offsets_n" = some ⟨fun i ↦ start+i.1.val⟩)
    (ho : s.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase start⟩)
    (hz : s.regs .real [C] "scale" = some ⟨fun i ↦ some (z i.1)⟩) :
    stepStmts (pass2Body P G Y M N K B C) s =
      some (advance (B := B) (C := C)
        (scatter (B := B) (C := C) (storeMid (B := B) s P G row N K kbase start z) Y
          (tileOffset row N K kbase start) (storeValue s P G row N K kbase start z)
          (tileActive N K kbase start)) row N K kbase start) := by
  simp [pass2Body, branch, source, softmax_backward_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hk, hn, ho, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  have hmask : (⟨fun i ↦ decide (start+i.1.val<N) &&
      decide (kbase+i.2.1.val<K)⟩ : Tile .bool [B,C]) =
      ⟨tileActive N K kbase start⟩ := by
    apply Tile.ext
    intro i
    by_cases hn : start+i.1.val<N <;> by_cases hk : kbase+i.2.1.val<K <;>
      simp [tileActive, RealForwardNonInner.tileActive, hn, hk]
  have hite (p : Prop) [Decidable p] (a b : ℝ) :
      (if p then some a else some b) = some (if p then a else b) := by
    split_ifs <;> rfl
  simp [storeMid, advance, scatter, RealForwardNonInner.scatter, storeValue,
    Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt, BlockState.paddingValue,
    BlockState.readMemValue_real, BlockState.writeMemTyped_real,
    tileSample, sample, tileOffset, offset, hite, hk, hn, ho, hmask,
    RealForwardNonInner.tileActive, tileActive]

  congr 1
  · congr 1
    · congr 1
      · funext st i
        split_ifs <;> simp_all
      · congr 1
        · congr 1
          · congr 1
            apply Tile.ext
            intro i
            dsimp only
            split_ifs <;> rfl
          · apply Tile.ext
            intro i
            dsimp only
            split_ifs <;> rfl
        · apply Tile.ext
          intro i
          dsimp only
          split_ifs <;> rfl
    · apply Tile.ext
      intro i
      simp [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
  · apply Tile.ext
    intro i
    simp [tileOffset, offset]
    ring

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

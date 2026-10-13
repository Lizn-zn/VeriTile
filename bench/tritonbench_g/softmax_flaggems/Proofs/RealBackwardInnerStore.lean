import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerScatter

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def pass2Body (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  match ((branch P G Y M N B C Bool.false)[7]? : Option Stmt) with
  | some (.forRange _ _ _ _ body) => body
  | some (.forRangeDyn _ _ _ _ body) => body
  | _ => []

def tileActive {B C : Nat} (M N rbase start : Nat) (i : TileIndex [C,B]) : Bool :=
  decide (rbase+i.1.val<M ∧ start+i.2.1.val<N)
noncomputable abbrev scatter := @RealForwardNonInner.scatter

noncomputable def storeValue {B C : Nat} (s : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (z : Fin C → ℝ) (i : TileIndex [C,B]) : ℝ :=
  tileSample s P M N rbase start i *
    (tileSample s G M N rbase start i - z i.1)

noncomputable def storeMid {B C : Nat} (s : BlockState) (P G : RegionName)
    (M N rbase start : Nat) (z : Fin C → ℝ) : BlockState :=
  let a := s.setReg "mask" .bool [C,B] ⟨tileActive M N rbase start⟩
  let b := a.setReg "out_tile" .real [C,B] ⟨fun i ↦ some (tileSample s P M N rbase start i)⟩
  let c := b.setReg "out_grad_tile" .real [C,B] ⟨fun i ↦ some (tileSample s G M N rbase start i)⟩
  c.setReg "in_grad_tile" .real [C,B] ⟨fun i ↦ some (storeValue s P G M N rbase start z i)⟩

noncomputable def advance {B C : Nat} (s : BlockState)
    (M N rbase start : Nat) : BlockState :=
  (s.setReg "n_offsets" .nat [B] ⟨fun i ↦ start+B+i.1.val⟩).setReg
    "offsets" .nat [C,B] ⟨tileOffset N rbase (start+B)⟩

theorem pass2_step (P G Y : RegionName) (M N B C rbase start : Nat)
    (s : BlockState) (z : Fin C → ℝ)
    (hk : s.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩)
    (hn : s.regs .nat [B] "n_offsets" = some ⟨fun i ↦ start+i.1.val⟩)
    (ho : s.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase start⟩)
    (hz : s.regs .real [C] "scale" = some ⟨fun i ↦ some (z i.1)⟩) :
    stepStmts (pass2Body P G Y M N B C) s =
      some (advance (B := B) (C := C)
        (scatter (B := C) (C := B) (storeMid (B := B) s P G M N rbase start z) Y
          (tileOffset N rbase start) (storeValue s P G M N rbase start z)
          (tileActive M N rbase start)) M N rbase start) := by
  simp [pass2Body, branch, source, softmax_backward_kernel_inner,
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
  have hmask : (⟨fun i ↦ decide (rbase+i.1.val<M) &&
      decide (start+i.2.1.val<N)⟩ : Tile .bool [C,B]) =
      ⟨tileActive M N rbase start⟩ := by
    apply Tile.ext
    intro i
    by_cases hn : start+i.2.1.val<N <;> by_cases hk : rbase+i.1.val<M <;>
      simp [tileActive, hn, hk]
  have hite (p : Prop) [Decidable p] (a b : ℝ) :
      (if p then some a else some b) = some (if p then a else b) := by
    split_ifs <;> rfl
  simp [storeMid, advance, scatter, RealForwardNonInner.scatter, storeValue,
    Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt, BlockState.paddingValue,
    BlockState.readMemValue_real, BlockState.writeMemTyped_real,
    tileSample, sample, tileOffset, offset, hite, hk, hn, ho, hmask,
    tileActive]

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
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner

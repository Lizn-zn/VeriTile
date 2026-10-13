import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def oneTileBody (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.true).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
  | some (.ifThenElse _ yes _) => yes
  | _ => []

noncomputable def oneTileMid (s : BlockState) (Y X : RegionName)
    (row N K kbase B C : Nat) (hB : 0 < B) : BlockState :=
  let off := tileOffset (B := B) (C := C) row N K kbase 0
  let inp := score s X row N K kbase 0 B C
  let mx := batchMax s X row N K kbase B C hB
  let z := batchSum s X row N K kbase B C hB
  let e := fun i : TileIndex [B, C] ↦ WithBot.realExp (WithBot.realSub (inp i) (mx i.2.1))
  let s₁ := s.setReg "n_offsets" .nat [B] ⟨fun i ↦ i.1.val⟩
  let s₂ := s₁.setReg "offset" .nat [B, C] ⟨off⟩
  let s₃ := s₂.setReg "mask" .bool [B, C] ⟨tileActive N K kbase 0⟩
  let s₄ := s₃.setReg "input_ptrs" .ptr [B, C] ⟨fun i ↦ (X, off i)⟩
  let s₅ := s₄.setReg "inp" .real [B, C] ⟨inp⟩
  let s₆ := s₅.setReg "m" .real [C] ⟨fun i ↦ mx i.1⟩
  let s₇ := s₆.setReg "e" .real [B, C] ⟨e⟩
  let s₈ := s₇.setReg "z" .real [C] (Tile.reduceSum 0 Bool.false ⟨e⟩)
  let s₉ := s₈.setReg "out" .real [B, C] ⟨fun i ↦ WithBot.realDiv (e i) (some (z i.2.1))⟩
  s₉.setReg "output_ptrs" .ptr [B, C] ⟨fun i ↦ (Y, off i)⟩

private theorem exp_finite (x : WithBot ℝ) :
    WithBot.realExp x = some (x.elim 0 Real.exp) := by
  cases x <;> rfl

theorem oneTile_step (Y X : RegionName) (M N K B C row kbase : Nat) (hB : 0 < B)
    (s : BlockState)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩) :
    ∃ mid, stepStmts (oneTileBody Y X M N K B C) s =
        some (scatter (B := B) (C := C) mid Y (tileOffset row N K kbase 0)
          (storeValue s X row N K kbase 0 (batchMax s X row N K kbase B C hB)
            (batchSum s X row N K kbase B C hB)) (tileActive N K kbase 0)) ∧
      mid.mem = s.mem ∧ mid.pids = s.pids ∧ mid.numPids = s.numPids ∧ mid.undef = s.undef := by
  simp [oneTileBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hrow, hk,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, List.getElem_cons_zero, hB, ↓reduceDIte]
  refine ⟨oneTileMid s Y X row N K kbase B C hB, ?_, rfl, rfl, rfl, rfl⟩
  simp [oneTileMid, scatter, tileOffset, tileActive, storeValue, score, batchMax, batchSum, columnMax, TileShape.insertAxisIndex,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop_data,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, ComparableDType.lt,
    BlockState.writeMemTyped_real, BlockState.readMemValue_real, exp_finite, ← WithBot.coe_sum]
  have hmask : (⟨fun i ↦ decide (i.1.val < N) &&
      decide (kbase + i.2.1.val < K)⟩ : Tile .bool [B, C]) =
      ⟨tileActive N K kbase 0⟩ := by
    apply Tile.ext
    intro i
    by_cases hn : i.1.val < N <;>
      by_cases hk' : kbase + i.2.1.val < K <;> simp [tileActive, hn, hk']
  rw [hmask]
  simp only [Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
    List.getElem_cons_zero]
  congr 1
  congr 10 <;> funext i <;> simp [tileOffset, score]
  rfl

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

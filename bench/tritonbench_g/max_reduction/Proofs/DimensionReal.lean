import bench.tritonbench_g.max_reduction.Proofs.Source
import bench.tritonbench_g.max_reduction.Proofs.Math
import bench.tritonbench_g.max_reduction.Proofs.DimensionStore

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- Independent masked row values, read from the complete original input
window. Padding is bottom, not a finite numerical initializer. -/
noncomputable def rowElement (M N BM BN pid : Nat)
    (xs : TileIndex [BM, BN] → ℝ) (i : Fin BM) (j : Fin BN) : WithBot ℝ :=
  if pid * BM + i.val < M ∧ j.val < N then some (xs (i, j, PUnit.unit)) else ⊥

noncomputable def rowMaximum (M N BM BN pid : Nat)
    (xs : TileIndex [BM, BN] → ℝ) (i : Fin BM) : ℝ :=
  WithBot.unbotD 0 ((Finset.univ : Finset (Fin BN)).sup (rowElement M N BM BN pid xs i))

/-- A left-to-right mathematical argmax; strict improvement retains the first
index on ties. It depends only on the supplied input values and row mask. -/
noncomputable def rowMaximumIndex (M N BM BN pid : Nat)
    (xs : TileIndex [BM, BN] → ℝ) (i : Fin BM) : Nat :=
  if h : 0 < BN then
    ((List.finRange BN).foldl (fun best j =>
      if rowElement M N BM BN pid xs i best < rowElement M N BM BN pid xs i j
      then j else best) ⟨0,h⟩).val
  else 0

theorem dimension_input_of_window (inp : RegionName) (M N K BM BN : Nat)
    (s : BlockState) (xs : TileIndex [BM, BN] → ℝ)
    (h : ∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      s.readMem inp ((s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1) =
        xs (i,j,PUnit.unit)) :
    maxKernelInputTile s inp M N K BM BN =
      ⟨fun ij => rowElement M N BM BN (s.pids 0) xs ij.1 ij.2.1⟩ := by
  apply congrArg Tile.mk
  funext ij
  rcases ij with ⟨i,j,u⟩
  cases u
  by_cases hr : s.pids 0 * BM + i.val < M
  · by_cases hc : j.val < N
    · simp [maxKernelInputTile, rowElement, maxInpElem, hr, hc, h i j hr hc]
    · simp [maxKernelInputTile, rowElement, hr, hc]
      rfl
  · simp [maxKernelInputTile, rowElement, hr]
    rfl

theorem dimension_value_formula (inp : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s : BlockState) (xs : TileIndex [BM, BN] → ℝ)
    (h : ∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      s.readMem inp ((s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1) =
        xs (i,j,PUnit.unit)) (i : Fin BM) :
    maxKernelValueSpec s inp M N K BM BN i = rowMaximum M N BM BN (s.pids 0) xs i := by
  unfold maxKernelValueSpec
  rw [dimension_input_of_window inp M N K BM BN s xs h]
  simp [rowMaximum, Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, hN, Finset.sup'_eq_sup]
  rfl

theorem dimension_index_formula (inp : RegionName) (M N K BM BN : Nat)
    (s : BlockState) (xs : TileIndex [BM, BN] → ℝ)
    (h : ∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      s.readMem inp ((s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1) =
        xs (i,j,PUnit.unit)) (i : Fin BM) :
    maxKernelIndexSpec s inp M N K BM BN i = rowMaximumIndex M N BM BN (s.pids 0) xs i := by
  unfold maxKernelIndexSpec
  rw [dimension_input_of_window inp M N K BM BN s xs h]
  simp [rowMaximumIndex, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]

theorem dimension_terminates (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s : BlockState) :
    ∃ t, exec (mathDimension Bool.false inp ov oi M N K BM BN) s = some t := by
  simp [exec, mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, Option.bind,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hN]

theorem dimension_flattenOk (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk, mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, Option.bind,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem dimension_traceSafe (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (bounds : RegionBounds) (s : BlockState)
    (hi : ∀ i : Fin BM, ∀ j : Fin BN, s.pids 0 * BM + i.val < M → j.val < N →
      (s.pids 0 * BM + i.val) * N * K + j.val * K + s.pids 1 < bounds inp)
    (hv : ∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      (s.pids 0 * BM + i.val) * K + s.pids 1 < bounds ov)
    (hx : ∀ i : Fin BM, s.pids 0 * BM + i.val < M →
      (s.pids 0 * BM + i.val) * K + s.pids 1 < bounds oi) :
    (mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel.TraceSafe bounds s := by
  simp [Kernel.TraceSafe, mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, Option.bind,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active,
    MemAccess.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hN]
  exact ⟨hi, hv, hx⟩

theorem dimension_readback (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s t : BlockState)
    (he : exec (mathDimension Bool.false inp ov oi M N K BM BN) s = some t)
    (hInj : Function.Injective (fun i : Fin BM => maxKernelOutOffset s K BM i))
    (hRegions : ov ≠ oi) (i : Fin BM) (ha : s.pids 0 * BM + i.val < M) :
    t.readMem ov (maxKernelOutOffset s K BM i) = maxKernelValueSpec s inp M N K BM BN i ∧
    t.readMemValue .int oi (maxKernelOutOffset s K BM i) =
      Int.ofNat (maxKernelIndexSpec s inp M N K BM BN i) := by
  simp [exec, mathDimension, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, Option.bind,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hN] at he
  rw [← he]
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BM] => (s.pids 0 * BM + idx.1.val) * K + s.pids 1) := by
    intro a b h
    have hab : a.1 = b.1 := hInj h
    rcases a with ⟨a,u⟩
    rcases b with ⟨b,v⟩
    cases u
    cases v
    simp_all
  constructor
  · rw [DimensionStore.foldl_writeMemTyped_intAt_prop_masked_readMem_other_region]
    · simp only [maxKernelOutOffset]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
      simp [ha, maxKernelValueSpec, maxKernelInputTile, maxInpElem,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, hN]
      congr
    · intro j hj hjActive heq
      exact hRegions heq
  · simp only [maxKernelOutOffset]
    rw [DimensionStore.scatter_readback_int_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
    simp [ha, maxKernelIndexSpec, maxKernelInputTile, maxInpElem,
      Tile.argMaxDrop, Tile.argBestDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, hN]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction

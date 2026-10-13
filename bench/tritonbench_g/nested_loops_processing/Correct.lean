import bench.tritonbench_g.nested_loops_processing.Proofs.Legacy
import bench.tritonbench_g.nested_loops_processing.Proofs.Contracts
import bench.tritonbench_g.nested_loops_processing.Proofs.RealExecution
import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionWitness

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
open VeriTile Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-! ## Dimension-general output summary

The full kernel body is stride-parametric: the lowering and every per-store
copy-correctness fact above hold over symbolic `(stride_m, stride_n)`. The
summary below bundles, over arbitrary strides, (a) that the full nested-loop
surface lowers and (b) the complete genuine copy-correctness sweep — all 24
`2×2` block stores of the three nested `range(0, 2)` loops, each carrying its own
`2×2`-offset injectivity hypothesis and asserting a verbatim copy from the
shifted INPUT region to the shifted OUTPUT region (`expected` reads input
memory, never `exec(...).readMem`). The Python wrapper's contiguous square
layouts (`stride_m ∈ {8, 4, 16, 2}`, `stride_n = 1`) are instances. -/
specification nested3_output_summary_general
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s : BlockState)
    (hInj0 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx))
    (hInj2 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 2 idx))
    (hInj4 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 4 idx))
    (hInj6 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 6 idx))
    (hInj8 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 8 idx))
    (hInj10 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 10 idx))
    (hInj12 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 12 idx))
    (hInj14 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 14 idx))
    (hInj16 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 16 idx))
    (hInj18 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 18 idx))
    (hInj20 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 20 idx))
    (hInj22 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 22 idx))
    (hInj24 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 24 idx))
    (hInj26 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 26 idx))
    (hInj28 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 28 idx))
    (hInj30 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 30 idx))
    (hInj32 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 32 idx))
    (hInj34 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 34 idx))
    (hInj36 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 36 idx))
    (hInj38 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 38 idx))
    (hInj40 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 40 idx))
    (hInj42 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 42 idx))
    (hInj44 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 44 idx))
    (hInj46 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 46 idx)) :
    -- (a) The full nested-loop surface lowers to the algorithm layer.
    (∃ alg, (nested3 in_ptr out_ptr stride_m stride_n).toAlgorithm? =
      Except.ok alg) ∧
    -- (b1) Initial a1 store: verbatim copy at shift 0.
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffset stride_m stride_n idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffset stride_m stride_n idx)) ∧
    -- (b2) First a2 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a2_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 2 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b3) First a3 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_first_a3_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 4 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 4 idx)) ∧
    -- (b4) Second k: a1 reused (in 0 → out 6).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 6)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 6 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b5) Second k: a2 (in 2 → out 8).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 2 8)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 8 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b6) Second k: a3 (in 6 → out 10).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 6 10)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 10 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 6 idx)) ∧
    -- (b7) Second j, first k: a1 reused (in 0 → out 12).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 12)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 12 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b8) Second j, first k: a2 (in 8 → out 14).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 14)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 14 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b9) Second j, first k: a3 (in 10 → out 16).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 10 16)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 16 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 10 idx)) ∧
    -- (b10) Second j, second k: a1 reused (in 0 → out 18).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 18)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 18 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b11) Second j, second k: a2 reused (in 8 → out 20).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 20)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 20 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b12) Second j, second k: a3 (in 12 → out 22).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 12 22)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 22 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 12 idx)) ∧
    -- (b13) Second i, first j, first k: a1 (in 14 → out 24).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 24)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 24 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b14) Second i, first j, first k: a2 (in 16 → out 26).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 26)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 26 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b15) Second i, first j, first k: a3 (in 18 → out 28).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 18 28)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 28 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 18 idx)) ∧
    -- (b16) Second i, first j, second k: a1 reused (in 14 → out 30).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 30)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 30 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b17) Second i, first j, second k: a2 reused (in 16 → out 32).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 32)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 32 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b18) Second i, first j, second k: a3 (in 20 → out 34).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 20 34)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 34 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 20 idx)) ∧
    -- (b19) Second i, second j, first k: a1 reused (in 14 → out 36).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 36)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 36 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b20) Second i, second j, first k: a2 (in 22 → out 38).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 38)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 38 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b21) Second i, second j, first k: a3 (in 24 → out 40).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 24 40)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 40 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 24 idx)) ∧
    -- (b22) Second i, second j, second k: a1 reused (in 14 → out 42).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 42)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 42 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b23) Second i, second j, second k: a2 reused (in 22 → out 44).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 44)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 44 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b24) Second i, second j, second k: final a3 (in 26 → out 46).
    ComputeCorrect.Realizes_without_Rounding
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 26 46)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 46 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 26 idx)) := by
  refine ⟨nested3_surface_toAlgorithm_supported in_ptr out_ptr stride_m stride_n,
    nested3_first_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj0,
    nested3_first_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj2,
    nested3_first_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj4,
    nested3_second_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj6,
    nested3_second_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj8,
    nested3_second_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj10,
    nested3_second_j_first_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj12,
    nested3_second_j_first_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj14,
    nested3_second_j_first_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj16,
    nested3_second_j_second_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj18,
    nested3_second_j_second_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj20,
    nested3_second_j_second_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj22,
    nested3_second_i_first_j_first_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj24,
    nested3_second_i_first_j_first_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj26,
    nested3_second_i_first_j_first_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj28,
    nested3_second_i_first_j_second_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj30,
    nested3_second_i_first_j_second_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj32,
    nested3_second_i_first_j_second_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj34,
    nested3_second_i_second_j_first_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj36,
    nested3_second_i_second_j_first_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj38,
    nested3_second_i_second_j_first_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj40,
    nested3_second_i_second_j_second_k_a1_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj42,
    nested3_second_i_second_j_second_k_a2_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj44,
    nested3_second_i_second_j_second_k_a3_store_compute_correct in_ptr out_ptr stride_m stride_n s hInj46⟩

/-! ## ════════ `⊨` IO face for the `a1` 2×2 store ════════

The first loop body's `a1` store is a straight-line, *unmasked* 2×2 tile memcpy:
read and write share the strided address `i·stride_m + j·stride_n`. That is
`MaskedTileKernelIO₁`'s shape at the degenerate mask `True`. -/

section IOFace

/-- Cell-level frame of an unmasked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame_unmasked {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl, BlockState.writeMem_mem, if_neg ?_]
      rintro ⟨h1, h2⟩
      rcases hc with h | h
      · exact h h1
      · exact h hd List.mem_cons_self h2.symm

theorem a1Store_flattenOk (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat) :
    ((nested3_first_a1_store in_ptr out_ptr stride_m stride_n).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [nested3_first_a1_store, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem a1Store_terminates (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s : BlockState) :
    ∃ s1, exec (nested3_first_a1_store in_ptr out_ptr stride_m stride_n) s
      = some s1 := by
  simp [exec, nested3_first_a1_store, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    TileShape.insertAxis, TileShape.dropInsertedIndex]

theorem a1Store_frame (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s s' : BlockState)
    (hExec : exec (nested3_first_a1_store in_ptr out_ptr stride_m stride_n) s
      = some s') :
    ∀ (r : RegionName) (n : Nat),
      (r ≠ out_ptr ∨ ∀ idx : TileIndex [2, 2],
        n ≠ matrixOffset stride_m stride_n idx) →
      s'.mem r n = s.mem r n := by
  intro r n hcond
  simp only [matrixOffset, rowIndex, colIndex] at hcond
  simp [exec, nested3_first_a1_store, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Option.bind, Option.map,
    Tile.bop, Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame_unmasked (region := out_ptr)
    (fun i : TileIndex [2, 2] => i.1.val * stride_m + i.2.1.val * stride_n)
    _ r n (TileShape.allIndices [2, 2]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ => Ne.symm (h idx)

theorem a1Store_traceSafe (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [2, 2],
      matrixOffset stride_m stride_n idx < bounds in_ptr)
    (hout : ∀ idx : TileIndex [2, 2],
      matrixOffset stride_m stride_n idx < bounds out_ptr) :
    ((nested3_first_a1_store in_ptr out_ptr stride_m stride_n).toAlgKernel).TraceSafe
      bounds s := by
  simp only [matrixOffset, rowIndex, colIndex] at hin hout
  simp [Kernel.TraceSafe, nested3_first_a1_store, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    TileShape.insertAxis, TileShape.dropInsertedIndex]
  and_intros
  all_goals try simp [Op.SafeAt.eq_def]
  -- the four lanes of the 2×2 window, on both buffers
  all_goals try (simpa using hin (0, 0, PUnit.unit))
  all_goals try (simpa using hin (0, 1, PUnit.unit))
  all_goals try (simpa using hin (1, 0, PUnit.unit))
  all_goals try (simpa using hin (1, 1, PUnit.unit))
  all_goals try (simpa using hout (0, 0, PUnit.unit))
  all_goals try (simpa using hout (0, 1, PUnit.unit))
  all_goals try (simpa using hout (1, 0, PUnit.unit))
  all_goals try (simpa using hout (1, 1, PUnit.unit))

/-- Region-model run of the `a1` store. -/
theorem a1Store_region_run (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx))
    (xs : TileIndex [2, 2] → ℝ)
    (hx : ∀ idx : TileIndex [2, 2],
      s₀.readMem in_ptr (matrixOffset stride_m stride_n idx) = xs idx) :
    ∃ s1, exec (nested3_first_a1_store in_ptr out_ptr stride_m stride_n) s₀
        = some s1
      ∧ (∀ idx : TileIndex [2, 2],
          s1.readMem out_ptr (matrixOffset stride_m stride_n idx) = xs idx)
      ∧ (∀ (r : RegionName) (n : Nat),
          (r ≠ out_ptr ∨ ∀ idx : TileIndex [2, 2],
            n ≠ matrixOffset stride_m stride_n idx) →
          s1.mem r n = s₀.mem r n) := by
  obtain ⟨s1, hexec⟩ := a1Store_terminates in_ptr out_ptr stride_m stride_n s₀
  refine ⟨s1, hexec, fun idx => ?_,
    a1Store_frame in_ptr out_ptr stride_m stride_n s₀ s1 hexec⟩
  rw [nested3_first_a1_store_correct in_ptr out_ptr stride_m stride_n s₀ s1
    hOutInj hexec idx, hx idx]

/-- IO signature of the `a1` 2×2 store. -/
def a1StoreIO (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat) :
    MaskedTileKernelIO₁ where
  kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n
  inp := in_ptr
  out := out_ptr
  shape := [2, 2]
  read := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  write := fun _pid idx => idx.1.val * stride_m + idx.2.1.val * stride_n
  mask := fun _pid _idx => True

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `nested_loops_processing.py`'s first
`a1` writeback: for every disjoint flat placement of `in_ptr` / `out_ptr` whose
2×2 strided window is in bounds, and every launch state whose `in_ptr` window
holds `xs`, the kernel terminates, every lane of the `out_ptr` window holds
`xs idx`, and every other memory cell is unchanged.

The store is unmasked, so the mask is `True` and the readback is total on the
window. General in both strides; honest side-condition = window injectivity, the
same hypothesis the per-write-map summary takes (`stride_m = stride_n` aliases
the two diagonal lanes). The tile extents are literal `2 × 2` because the Python
kernel's `tl.arange(0, 2)` are literals. -/
specification nested3_first_a1_store_io_correctness
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)) :
    a1StoreIO in_ptr out_ptr stride_m stride_n ⊨ fun _pid xs idx => xs idx := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact a1Store_flattenOk in_ptr out_ptr stride_m stride_n
  · intro bounds s h1 h2
    exact a1Store_traceSafe in_ptr out_ptr stride_m stride_n bounds s
      (fun idx => h1 idx trivial) (fun idx => h2 idx trivial)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      a1Store_region_run in_ptr out_ptr stride_m stride_n s₀ hOutInj xs
        (fun idx => hx idx trivial)
    exact ⟨s1, hexec, fun idx _ => hval idx, fun r n hc => hframe r n (by
      rcases hc with hr | hn
      · exact Or.inl hr
      · exact Or.inr fun idx => hn idx trivial)⟩

/-! ### The rounding face

The store is a pure copy: no arithmetic on the loaded window, and no `.to(...)`
on the store — the tile stays `.real` end to end. So the slice is **cast-free**
— every statement steps identically under `stepStmtsR R` and `stepStmts` — and
the exact run transports to `execR R` for *every* rounding model. -/

/-- The slice is cast-free: `execR R` is the exact stepper. -/
private theorem a1Store_castFree (R : RoundingModel) (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) (s : BlockState) :
    execR R ((nested3_first_a1_store in_ptr out_ptr stride_m stride_n).toAlgKernel) s
      = exec ((nested3_first_a1_store in_ptr out_ptr stride_m stride_n).toAlgKernel) s := by
  simp [execR, exec, nested3_first_a1_store, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmtsR, stepStmts,
    stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp, evalOp.eq_def,
    BlockState.writeMemTypedR, BlockState.writeMemAsR, Option.bind, Option.map,
    Tile.bop, Tile.expandDim, Tile.ptrAdd, Tile.uop, NumericDType.add,
    NumericDType.mul, FloatDType.cast, TileShape.insertAxis,
    TileShape.dropInsertedIndex]

/-- Per-execution safety walk **under the rounding model** — the `hts`
obligation of `MaskedTileKernelIO₁.ImplementsR.intro`. -/
theorem a1Store_traceSafeR (R : RoundingModel) (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [2, 2],
      matrixOffset stride_m stride_n idx < bounds in_ptr)
    (hout : ∀ idx : TileIndex [2, 2],
      matrixOffset stride_m stride_n idx < bounds out_ptr) :
    Kernel.TraceSafeR R bounds
      ((nested3_first_a1_store in_ptr out_ptr stride_m stride_n).toAlgKernel) s := by
  simp only [matrixOffset, rowIndex, colIndex] at hin hout
  simp [Kernel.TraceSafeR, nested3_first_a1_store, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtsR, stepStmtR, evalOpR,
    evalOpR.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim,
    Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul, FloatDType.cast,
    TileShape.insertAxis, TileShape.dropInsertedIndex]
  and_intros
  all_goals try simp [Op.SafeAtR.eq_def]
  -- the four lanes of the 2×2 window, on both buffers
  all_goals try (simpa using hin (0, 0, PUnit.unit))
  all_goals try (simpa using hin (0, 1, PUnit.unit))
  all_goals try (simpa using hin (1, 0, PUnit.unit))
  all_goals try (simpa using hin (1, 1, PUnit.unit))
  all_goals try (simpa using hout (0, 0, PUnit.unit))
  all_goals try (simpa using hout (0, 1, PUnit.unit))
  all_goals try (simpa using hout (1, 0, PUnit.unit))
  all_goals try (simpa using hout (1, 1, PUnit.unit))

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for `nested_loops_processing.py`'s first `a1`
writeback: for **every** rounding model `R`, the same Hoare triple as
`nested3_first_a1_store_io_correctness`, but run under `execR R` and read back as
`.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy carrying no `.to(...)`, so the slice is cast-free and
the exact run transports verbatim. The content of the rounding face here is
exactly that: *this kernel introduces no rounding event of its own*, at any
`R`. -/
specification nested3_first_a1_store_io_correctnessR (R : RoundingModel)
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [2, 2] =>
        idx.1.val * stride_m + idx.2.1.val * stride_n)) :
    a1StoreIO in_ptr out_ptr stride_m stride_n
      ⊨[R, FloatDType.real] fun _pid xs idx => xs idx := by
  refine MaskedTileKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact a1Store_flattenOk in_ptr out_ptr stride_m stride_n
  · intro bounds s h1 h2
    exact a1Store_traceSafeR R in_ptr out_ptr stride_m stride_n bounds s
      (fun idx => h1 idx trivial) (fun idx => h2 idx trivial)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      a1Store_region_run in_ptr out_ptr stride_m stride_n s₀ hOutInj xs
        (fun idx => hx idx trivial)
    refine ⟨s1, ?_, ?_, fun r n hc => hframe r n (by
      rcases hc with hr | hn
      · exact Or.inl hr
      · exact Or.inr fun idx => hn idx trivial)⟩
    · simp only [a1StoreIO]
      rw [a1Store_castFree R in_ptr out_ptr stride_m stride_n s₀]
      exact hexec
    · intro idx _
      simp only [a1StoreIO]
      rw [BlockState.readMemAs_real]
      have := hval idx
      simp only [matrixOffset, rowIndex, colIndex] at this
      rw [this]
      simp

end IOFace


/-! The source-facing contracts below execute the complete original/candidate
kernel. Nonnegative strides exclude the pointer model's negative-offset
truncation boundary; allocation bounds exclude the wrapper's unsafe small case.
All 15 homogeneous storage profiles share the same independent ordered-write
oracle, with its explicit ideal-real floating storage normalization. -/
open TypedSource
open scoped VeriTile.Spec

/-- Complete original execution, independent output oracle, raw frame, and access safety. -/
specification original_real_correct (storage : Storage) (bounds : RegionBounds)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    Spec.Real (Contracts.RegionCopy storage Bool.false bounds s src dst sm sn) := by
  exact ⟨hinjective, PayloadReal.original_io storage s src dst sm sn hsep,
    PayloadReal.original_safe storage bounds src dst sm sn s hsrc hdst⟩

/-- The actual candidate has the same independently specified ordered output and full frame. -/
specification candidate_real_correct (storage : Storage) (bounds : RegionBounds)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < bounds src) (hdst : sm + 47 * sn < bounds dst) :
    Spec.Real (Contracts.RegionCopy storage Bool.true bounds s src dst sm sn) := by
  exact ⟨hinjective, PayloadReal.candidate_io storage s src dst sm sn hsep,
    PayloadReal.candidate_safe storage bounds src dst sm sn s hsrc hdst⟩

/-- Complete original flat-memory contract; the library bridge's zero-undef
requirement is explicit and is absent from the region-memory contract. -/
specification original_flat_real_correct (storage : Storage) (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (Contracts.FlatCopy storage Bool.false A s src dst sm sn) := by
  exact ⟨hinjective, PayloadReal.original_flat_io storage A hd hcov s src dst sm sn
    hsep hsrc hdst hu, PayloadReal.original_safe storage A.extent src dst sm sn s hsrc hdst⟩

/-- Complete candidate flat-memory contract, including untouched raw cell types. -/
specification candidate_flat_real_correct (storage : Storage) (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat)
    (hsep : src ≠ dst) (hinjective : FullReal.TileInjective sm sn)
    (hsrc : sm + 27 * sn < A.extent src) (hdst : sm + 47 * sn < A.extent dst)
    (hu : s.undef = fun _ _ ↦ 0) :
    Spec.Real (Contracts.FlatCopy storage Bool.true A s src dst sm sn) := by
  exact ⟨hinjective, PayloadReal.candidate_flat_io storage A hd hcov s src dst sm sn
    hsep hsrc hdst hu, PayloadReal.candidate_safe storage A.extent src dst sm sn s hsrc hdst⟩

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing

#guard_msgs (drop info) in
#auditModuleAxioms

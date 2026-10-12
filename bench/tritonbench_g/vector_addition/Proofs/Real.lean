import bench.tritonbench_g.vector_addition.Kernels
import VeriTile.Examples.Common

/-! Real execution, bounds and memory-frame lemmas. Public IO and mathematical
formulas are in ../Correct.lean. -/

namespace VeriTile.Bench.TritonBenchG.VectorAddition

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Algorithm-layer correctness for `add_kernel`.

Each active lane writes `x + y`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s)
          output_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then xs i + ys i
                else s.readMem output_ptr addr) := by
  intro i
  simp [observeAt, exec, add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem add_kernel_frame
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, add_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₂.Implements.intro`; the value half reuses
`add_kernel_correct` (instantiated at the tiles the state actually holds). -/
theorem add_kernel_region_run
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem y_ptr (s₀.pid * BLOCK_SIZE + j.val) = ys j) :
    ∃ s1, exec ((add_kernel x_ptr y_ptr output_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j + ys j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun j => s₀.readMem y_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s₀
      = exec ((add_kernel x_ptr y_ptr output_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((add_kernel x_ptr y_ptr output_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj, hy j hj]
      · refine add_kernel_frame x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem add_kernel_traceSafe
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hy : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds y_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [add_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hy a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_kernel_flattenOk
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [add_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

section Commuted

set_option maxHeartbeats 5000000

theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

/-- Execution equality here uses only real arithmetic; the FP layer is separate. -/
theorem commuted_exec (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) (s : BlockState) :
    exec (add_commuted_kernel x_ptr y_ptr output_ptr n B).toAlgKernel s =
      exec (add_kernel x_ptr y_ptr output_ptr n B).toAlgKernel s := by
  simp [add_commuted_kernel, add_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, add_comm, optional_add_comm]

theorem commuted_flattenOk (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) :
    (add_commuted_kernel x_ptr y_ptr output_ptr n B).toAlgKernel.FlattenOk := by
  simp [add_commuted_kernel, Kernel.FlattenOk,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem commuted_traceSafe (x_ptr y_ptr output_ptr : RegionName) (n B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds x_ptr)
    (hy : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds y_ptr)
    (hout : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds (add_commuted_kernel x_ptr y_ptr output_ptr n B).toAlgKernel s := by
  simpa [Kernel.TraceSafe, add_commuted_kernel, add_kernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, add_comm, optional_add_comm]
    using add_kernel_traceSafe x_ptr y_ptr output_ptr n B bounds s hx hy hout

end Commuted

end VeriTile.Bench.TritonBenchG.VectorAddition

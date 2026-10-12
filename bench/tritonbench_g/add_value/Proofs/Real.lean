import bench.tritonbench_g.add_value.Kernels
import VeriTile.Examples.Common

/-! Real execution, bounds and memory-frame lemmas. The public IO contract
and independent mathematical formulas are in ../Correct.lean. -/

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Algorithm-layer correctness for `puzzle1_kernel`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < N`): the output region holds `xs i + value`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store). -/
theorem puzzle1_kernel_correct
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value) s)
                output_ptr BLOCK_SIZE s.pid i
        = some (if addr < N then xs i + value
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, puzzle1_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < N
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem puzzle1_kernel_frame
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (s s1 : BlockState)
    (hExec : exec ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < N →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, puzzle1_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input window is loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₁.Implements.intro`; the value half reuses
`puzzle1_kernel_correct` (instantiated at the tile the state actually holds). -/
theorem puzzle1_kernel_region_run
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < N →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((puzzle1_kernel x_ptr output_ptr N
        BLOCK_SIZE value).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < N →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j + value)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < N →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := puzzle1_kernel_correct x_ptr output_ptr N BLOCK_SIZE value
    s₀ (fun j => s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value) s₀
      = exec ((puzzle1_kernel x_ptr output_ptr N
          BLOCK_SIZE value).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((puzzle1_kernel x_ptr output_ptr N
      BLOCK_SIZE value).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, puzzle1_kernel, ComputeKernel.toAlgKernel, stepStmts,
          stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine puzzle1_kernel_frame x_ptr output_ptr N BLOCK_SIZE value
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< N`, so the bounds
contract is **lane-wise** — every *active* lane's address is below the region
bound of the buffer it touches. -/
theorem puzzle1_kernel_traceSafe
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem puzzle1_kernel_flattenOk
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

section Commuted

set_option maxHeartbeats 5000000

/-- Real execution equality; no FP numerical rule is used in this layer. -/
theorem commuted_exec (x_ptr output_ptr : RegionName) (N B : Nat)
    (value : ℝ) (s : BlockState) :
    exec (puzzle1_commuted_kernel x_ptr output_ptr N B value).toAlgKernel s =
      exec (puzzle1_kernel x_ptr output_ptr N B value).toAlgKernel s := by
  simp [puzzle1_commuted_kernel, puzzle1_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt, add_comm]

theorem commuted_flattenOk (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) :
    (puzzle1_commuted_kernel x_ptr output_ptr N B value).toAlgKernel.FlattenOk := by
  simp [puzzle1_commuted_kernel, Kernel.FlattenOk,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem commuted_traceSafe (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N → s.pid * B + j.val < bounds x_ptr)
    (hout : ∀ j : Fin B, s.pid * B + j.val < N → s.pid * B + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds (puzzle1_commuted_kernel x_ptr output_ptr N B value).toAlgKernel s := by
  simpa [Kernel.TraceSafe, puzzle1_commuted_kernel, puzzle1_kernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt, add_comm]
    using puzzle1_kernel_traceSafe x_ptr output_ptr N B value bounds s hx hout

end Commuted

end VeriTile.Bench.TritonBenchG.AddValue

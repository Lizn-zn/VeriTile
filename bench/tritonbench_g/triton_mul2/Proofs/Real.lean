import bench.tritonbench_g.triton_mul2.Kernels
import VeriTile.Examples.Common

/-! Real execution, active-address safety and memory-frame lemmas for both
ordinary and in-place output. The public contracts are in ../Correct.lean. -/

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Algorithm-layer correctness for `mul2_kernel`. -/
theorem mul2_kernel_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s)
          out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem out_ptr addr) := by
  intro i
  simp [observeAt, exec, mul2_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `out_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem mul2_kernel_frame
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(out_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mul2_kernel, ComputeKernel.toAlgKernel,
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
`mul2_kernel_correct` (instantiated at the tile the state actually holds). -/
theorem mul2_kernel_region_run
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((mul2_kernel in_ptr0 out_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem out_ptr (s₀.pid * BLOCK_SIZE + j.val) = 2 * xs j)
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := mul2_kernel_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s₀
      = exec ((mul2_kernel in_ptr0 out_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((mul2_kernel in_ptr0 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, mul2_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine mul2_kernel_frame in_ptr0 out_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem mul2_kernel_traceSafe
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds in_ptr0)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds
      ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [mul2_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem mul2_kernel_flattenOk
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mul2_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Algorithm-layer correctness for `mul2_inplace_kernel`. -/
theorem mul2_inplace_kernel_correct
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_inplace_kernel ptr n_elements BLOCK_SIZE) s)
          ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem ptr addr) := by
  intro i
  simp [observeAt, exec, mul2_inplace_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked in-place
store is preserved by the run — in particular every cell of every region other
than `ptr`, and the *inactive* lanes of the window itself. -/
private theorem mul2_inplace_kernel_frame
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** for the in-place kernel —
termination, active-lane output values (the doubled *originally loaded*
window), and frame off the active lanes, from any launch state whose window
is loaded at the **active lanes only**. This is the `hrun` obligation of
`MaskedKernelIO₁.Implements.intro`; the value half reuses
`mul2_inplace_kernel_correct` (instantiated at the tile the state actually
holds — the kernel reads before it scatters, so the before/after reading is
sound). -/
theorem mul2_inplace_kernel_region_run
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((mul2_inplace_kernel ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem ptr (s₀.pid * BLOCK_SIZE + j.val) = 2 * xs j)
      ∧ (∀ r o,
          (r ≠ ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := mul2_inplace_kernel_correct ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (mul2_inplace_kernel ptr n_elements BLOCK_SIZE) s₀
      = exec ((mul2_inplace_kernel ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((mul2_inplace_kernel ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, mul2_inplace_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine mul2_inplace_kernel_frame ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the in-place kernel: the masked load and the
masked store both address the window `pid * BLOCK_SIZE + j` of `ptr`, active
only when `< n_elements`, so the bounds contract is **lane-wise**. -/
theorem mul2_inplace_kernel_traceSafe
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds ptr) :
    Kernel.TraceSafe bounds
      ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypothesis below.
  simp [mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  -- simp dedups the load and store obligations (same buffer, same window):
  -- one lane-wise bound remains.
  exact fun a ha => hx a ha

/-- The in-place kernel sits inside the flat-memory bridge's covered
fragment. -/
theorem mul2_inplace_kernel_flattenOk
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

section Commuted

set_option maxHeartbeats 5000000

/-- Execution equality here uses real arithmetic; FP laws live in FPEquiv.lean. -/
theorem commuted_exec (in_ptr0 out_ptr : RegionName) (n B : Nat) (s : BlockState) :
    exec (mul2_commuted_kernel in_ptr0 out_ptr n B).toAlgKernel s =
      exec (mul2_kernel in_ptr0 out_ptr n B).toAlgKernel s := by
  simp [mul2_commuted_kernel, mul2_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt, mul_comm]

theorem commuted_flattenOk (in_ptr0 out_ptr : RegionName) (n B : Nat) :
    (mul2_commuted_kernel in_ptr0 out_ptr n B).toAlgKernel.FlattenOk := by
  simp [mul2_commuted_kernel, Kernel.FlattenOk,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem commuted_traceSafe (in_ptr0 out_ptr : RegionName) (n B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds in_ptr0)
    (hout : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds (mul2_commuted_kernel in_ptr0 out_ptr n B).toAlgKernel s := by
  simpa [Kernel.TraceSafe, mul2_commuted_kernel, mul2_kernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, mul_comm]
    using mul2_kernel_traceSafe in_ptr0 out_ptr n B bounds s hx hout

/-- The two in-place sources are definitionally the ordinary sources with
both pointer arguments wired to ptr. The execution bridge permits this alias. -/
theorem inplace_commuted_exec (ptr : RegionName) (n B : Nat) (s : BlockState) :
    exec (mul2_inplace_commuted_kernel ptr n B).toAlgKernel s =
      exec (mul2_inplace_kernel ptr n B).toAlgKernel s :=
  commuted_exec ptr ptr n B s

theorem inplace_commuted_flattenOk (ptr : RegionName) (n B : Nat) :
    (mul2_inplace_commuted_kernel ptr n B).toAlgKernel.FlattenOk :=
  commuted_flattenOk ptr ptr n B

theorem inplace_commuted_traceSafe (ptr : RegionName) (n B : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds ptr) :
    Kernel.TraceSafe bounds (mul2_inplace_commuted_kernel ptr n B).toAlgKernel s :=
  commuted_traceSafe ptr ptr n B bounds s hx hx

end Commuted

end VeriTile.Bench.TritonBenchG.TritonMul2

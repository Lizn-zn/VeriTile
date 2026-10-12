import bench.tritonbench_g.vector_addition_custom.Kernels
import VeriTile.Examples.Common

/-! Real execution, active-address safety and memory-frame lemmas. The public
IO and independent mathematical formulas are in ../Correct.lean. -/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Algorithm-layer correctness for `_add_kernel`.

Each active lane writes `A + B`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    ∀ i : Fin BLOCK,
      let addr := s.pid * BLOCK + i.val
      observeAt (exec (_add_kernel A B C size BLOCK) s) C BLOCK s.pid i
        = some (if addr < size then as i + bs i else s.readMem C addr) := by
  intro i
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_a h_b
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK + i.val < size
  · simp [hi, h_a, h_b]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `C`, and the *inactive* lanes of the output window itself. -/
private theorem add_kernel_frame
    (A B C : RegionName)
    (size BLOCK : Nat) (s s1 : BlockState)
    (hExec : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK, s.pid * BLOCK + i.val < size →
      ¬(C = r ∧ s.pid * BLOCK + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, _add_kernel, ComputeKernel.toAlgKernel,
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
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK → ℝ)
    (ha : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem A (s₀.pid * BLOCK + j.val) = as j)
    (hb : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem B (s₀.pid * BLOCK + j.val) = bs j) :
    ∃ s1, exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
          s1.readMem C (s₀.pid * BLOCK + j.val) = as j + bs j)
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK,
            s₀.pid * BLOCK + j.val < size →
              o ≠ s₀.pid * BLOCK + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct A B C size BLOCK
    s₀ (fun j => s₀.readMem A (s₀.pid * BLOCK + j.val))
    (fun j => s₀.readMem B (s₀.pid * BLOCK + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (_add_kernel A B C size BLOCK) s₀
      = exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, _add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, ha j hj, hb j hj]
      · refine add_kernel_frame A B C size BLOCK
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `prog_id * BLOCK + j`, active only when `< size`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem add_kernel_traceSafe
    (A B C : RegionName)
    (size BLOCK : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (ha : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds A)
    (hb : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds B)
    (hout : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((_add_kernel A B C size BLOCK).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all five statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha' => ha a ha', fun a hb' => hb a hb', fun a ho' => hout a ho'⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_kernel_flattenOk
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ((_add_kernel A B C size BLOCK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

section Commuted

set_option maxHeartbeats 5000000

theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

/-- Only the real layer uses mathematical commutativity here. -/
theorem commuted_exec (A B C : RegionName) (size BLOCK : Nat) (s : BlockState) :
    exec (_add_commuted_kernel A B C size BLOCK).toAlgKernel s =
      exec (_add_kernel A B C size BLOCK).toAlgKernel s := by
  simp [_add_commuted_kernel, _add_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, add_comm, optional_add_comm]

theorem commuted_flattenOk (A B C : RegionName) (size BLOCK : Nat) :
    (_add_commuted_kernel A B C size BLOCK).toAlgKernel.FlattenOk := by
  simp [_add_commuted_kernel, Kernel.FlattenOk,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem commuted_traceSafe (A B C : RegionName) (size BLOCK : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (ha : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size → s.pid * BLOCK + j.val < bounds A)
    (hb : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size → s.pid * BLOCK + j.val < bounds B)
    (hc : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size → s.pid * BLOCK + j.val < bounds C) :
    Kernel.TraceSafe bounds (_add_commuted_kernel A B C size BLOCK).toAlgKernel s := by
  simpa [Kernel.TraceSafe, _add_commuted_kernel, _add_kernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, add_comm, optional_add_comm]
    using add_kernel_traceSafe A B C size BLOCK bounds s ha hb hc

end Commuted

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom

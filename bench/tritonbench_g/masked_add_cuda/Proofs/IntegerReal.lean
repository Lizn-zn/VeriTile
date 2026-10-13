import bench.tritonbench_g.masked_add_cuda.Proofs.Source
import bench.tritonbench_g.masked_add_cuda.Proofs.IntegerIO

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

def integerAddOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def integerAddActive
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  integerAddOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .int p_mask_ptr (integerAddOffset s BLOCK_SIZE i) = 0

instance integerAddActiveDecidable
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (integerAddActive s p_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold integerAddActive
  infer_instance

noncomputable def integerAddSpec
    (s : BlockState) (grad_ptr p_ptr : RegionName)
    (alpha : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem grad_ptr (integerAddOffset s BLOCK_SIZE i) +
    s.readMem p_ptr (integerAddOffset s BLOCK_SIZE i) * alpha

/-- Algorithm-layer correctness for `integer_math_kernel`. -/
theorem integer_math_kernel_correct
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (integer_math_kernel grad_ptr p_ptr p_mask_ptr
          n_elements alpha BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := integerAddOffset s BLOCK_SIZE i
      s'.readMem grad_ptr outAddr =
        if integerAddActive s p_mask_ptr n_elements BLOCK_SIZE i then
          integerAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i
        else s.readMem grad_ptr outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, integer_math_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, Bool.and_eq_true] at hExec
  subst s'
  simp only [integerAddOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hActive : integerAddActive s p_mask_ptr n_elements BLOCK_SIZE i
  · rcases hActive with ⟨hBounds, hMask⟩
    have hCond :
        s.pids 0 * BLOCK_SIZE + i.val < n_elements ∧
          (if s.pids 0 * BLOCK_SIZE + i.val < n_elements then
            s.readMemValue .int p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = 0
          else
            BlockState.defaultCarrier .int = 0) := by
      simp [integerAddOffset] at hBounds hMask
      simp [hBounds, hMask]
    have hMask' :
        s.readMemValue .int p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = 0 := by
      simpa [integerAddOffset] using hMask
    simp [integerAddActive, integerAddSpec, integerAddOffset, hMask', hCond]
  · simp [integerAddActive, integerAddOffset] at hActive ⊢
    by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
    · simp [hBounds] at hActive ⊢
      simp [hActive]
    · simp [hBounds]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked in-place
store — every cell of every region other than `grad_ptr`, and the lanes of the
output window itself that are out of bounds or vetoed by `p_mask` — is
preserved by the run. -/
theorem integer_math_kernel_frame
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((integer_math_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + i.val < n_elements →
      s.readMemValue .int p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val)
        = 0 →
      ¬(grad_ptr = r ∧ s.pids 0 * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, integer_math_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  obtain ⟨hb1, hb2⟩ := hmk
  rw [if_pos hb1] at hb2
  exact hmiss k.1 hb1 hb2 hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body, so no `0 < BLOCK_SIZE` side condition is needed). -/
theorem integer_math_kernel_exec_isSome
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((integer_math_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, integer_math_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
write-active-lane output values, and frame off the write-active output lanes,
from any launch state whose windows hold `xs`/`ys`/`bs` at the static bounds
mask. This is the `hrun` obligation of the `⊨` headline; the value half reuses
`integer_math_kernel_correct` (the kernel reads `grad` before the in-place
scatter, so the before/after reading is sound). -/
theorem integer_math_kernel_region_run
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s₀ : BlockState) (bs : Fin BLOCK_SIZE → Int) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem p_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem grad_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = ys j)
    (hb : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMemValue .int p_mask_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = bs j) :
    ∃ s1, exec ((integer_math_kernel grad_ptr p_ptr p_mask_ptr
          n_elements alpha BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE,
          s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = 0 →
          s1.readMem grad_ptr (s₀.pids 0 * BLOCK_SIZE + j.val)
            = ys j + xs j * alpha)
      ∧ (∀ r o,
          (r ≠ grad_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = 0 →
              o ≠ s₀.pids 0 * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := integer_math_kernel_exec_isSome grad_ptr p_ptr p_mask_ptr
    n_elements alpha BLOCK_SIZE s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have hActive : integerAddActive s₀ p_mask_ptr n_elements BLOCK_SIZE j :=
      ⟨hj.1, (hb j hj.1).trans hj.2⟩
    have h := integer_math_kernel_correct grad_ptr p_ptr p_mask_ptr
      n_elements alpha BLOCK_SIZE s₀ s1 hs1 j
    simp only [integerAddOffset, integerAddSpec, BlockState.pid,
      if_pos hActive] at h
    rw [h, hx j hj.1, hy j hj.1]
  · refine integer_math_kernel_frame grad_ptr p_ptr p_mask_ptr n_elements alpha
      BLOCK_SIZE s₀ s1 hs1 r o (fun i hbounds hmaskv ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i ⟨hbounds, (hb i hbounds).symm.trans hmaskv⟩ ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — six are memory-silent (`program_id`, the block-start arithmetic,
`arange`, the two mask computations, the fused register update) — and reduces
the four masked accesses (the `p_mask` integer load at the **static** bounds
mask, the `p`/`grad` loads and the `grad` store at the **narrowed**
`mask & ~p_mask`, lane `j` at `pid * BLOCK_SIZE + j`) to the lane-wise bounds
hypotheses at the static mask: narrowed-active lanes are in particular
in-bounds, so the static hypotheses cover all four (simp dedups the `grad`
load and store — same buffer, same window — into one obligation, so only
three hypotheses are needed). -/
theorem integer_math_kernel_traceSafe
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds grad_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_mask_ptr) :
    Kernel.TraceSafe bounds
      ((integer_math_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [integer_math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => h3 a ha, fun a ha _ => h1 a ha, fun a ha _ => h2 a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads — one through the `.to(tl.int1)` bool cast — and the
masked in-place store). -/
theorem integer_math_kernel_flattenOk
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ((integer_math_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [integer_math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — six are memory-silent (`program_id`, the block-start arithmetic,
`arange`, the two mask computations, the fused register update) — and reduces
the four masked accesses (the `p_mask` integer load at the **static** bounds
mask, the `p`/`grad` loads and the `grad` store at the **narrowed**
`mask & ~p_mask`, lane `j` at `pid * BLOCK_SIZE + j`) to the lane-wise bounds
hypotheses at the static mask: narrowed-active lanes are in particular
in-bounds, so the static hypotheses cover all four (simp dedups the `grad`
load and store — same buffer, same window — into one obligation, so only
three hypotheses are needed). -/
theorem integer_commuted_math_kernel_traceSafe
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds grad_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_mask_ptr) :
    Kernel.TraceSafe bounds
      ((integer_commuted_math_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [integer_commuted_math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => h3 a ha, fun a ha _ => h1 a ha, fun a ha _ => h2 a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads — one through the `.to(tl.int1)` bool cast — and the
masked in-place store). -/
theorem integer_commuted_math_kernel_flattenOk
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ((integer_commuted_math_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [integer_commuted_math_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

private theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

/-- Independent real arithmetic justification for the commuted candidate. -/
theorem integer_commuted_exec (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat)
    (s : BlockState) :
    exec (integer_commuted_math_kernel g p m n a B).toAlgKernel s =
      exec (integer_math_kernel g p m n a B).toAlgKernel s := by
  simp [integer_commuted_math_kernel, integer_math_kernel, exec, stepStmts, stepStmt,
    evalOp.eq_def, tile_elementwise, add_comm, optional_add_comm]

theorem integer_exec (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel s =
      exec (integer_math_kernel g p m n a B).toAlgKernel s := by
  rw [integer_projection]
  cases v
  · rfl
  · exact integer_commuted_exec g p m n a B s

theorem integer_flattenOk (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel.FlattenOk := by
  rw [integer_projection]
  cases v
  · exact integer_math_kernel_flattenOk g p m n a B
  · exact integer_commuted_math_kernel_flattenOk g p m n a B

theorem integer_traceSafe (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hp : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds p)
    (hg : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds g)
    (hm : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds m) :
    Kernel.TraceSafe bounds (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel s := by
  rw [integer_projection]
  cases v
  · exact integer_math_kernel_traceSafe g p m n a B bounds s hp hg hm
  · exact integer_commuted_math_kernel_traceSafe g p m n a B bounds s hp hg hm

theorem fp64_integer_exec (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgKernel s =
      exec (integer_math_kernel g p m n a B).toAlgKernel s := by
  rw [fp64_integer_projection]
  cases v
  · rfl
  · exact integer_commuted_exec g p m n a B s

theorem fp64_integer_flattenOk (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgKernel.FlattenOk := by
  rw [fp64_integer_projection]
  cases v
  · exact integer_math_kernel_flattenOk g p m n a B
  · exact integer_commuted_math_kernel_flattenOk g p m n a B

theorem fp64_integer_traceSafe (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hp : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds p)
    (hg : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds g)
    (hm : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds m) :
    Kernel.TraceSafe bounds (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgKernel s := by
  rw [fp64_integer_projection]
  cases v
  · exact integer_math_kernel_traceSafe g p m n a B bounds s hp hg hm
  · exact integer_commuted_math_kernel_traceSafe g p m n a B bounds s hp hg hm

theorem full_integer_exec (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgKernel s =
      exec (integer_math_kernel g p m n a B).toAlgKernel s := by
  rw [full_integer_projection]
  cases v
  · rfl
  · exact integer_commuted_exec g p m n a B s

theorem full_integer_flattenOk (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgKernel.FlattenOk := by
  rw [full_integer_projection]
  cases v
  · exact integer_math_kernel_flattenOk g p m n a B
  · exact integer_commuted_math_kernel_flattenOk g p m n a B

theorem full_integer_traceSafe (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hp : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds p)
    (hg : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds g)
    (hm : ∀ j : Fin B, s.pids 0 * B + j.val < n → s.pids 0 * B + j.val < bounds m) :
    Kernel.TraceSafe bounds (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgKernel s := by
  rw [full_integer_projection]
  cases v
  · exact integer_math_kernel_traceSafe g p m n a B bounds s hp hg hm
  · exact integer_commuted_math_kernel_traceSafe g p m n a B bounds s hp hg hm

end VeriTile.Bench.TritonBenchG.MaskedAddCuda

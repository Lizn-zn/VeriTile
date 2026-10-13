import bench.tritonbench_g.kldiv_compute.Kernels

namespace VeriTile.Bench.TritonBenchG.KldivCompute

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledLoss
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Algorithm-layer correctness for `kldivergence_kernel`.

For inactive lanes the masked store does not write, so the output cell is
preserved from the initial state. -/
theorem kldivergence_kernel_correct
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs)
    (h_y : InputLoadedAt s y_ptr BLOCK_SIZE ys) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s)
          output_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then klDivSpec (xs i) (ys i)
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, kldivergence_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, klDivSpec]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x, h_y]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem kldivergence_kernel_frame
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, kldivergence_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₂.Implements.intro`; the value half reuses
`kldivergence_kernel_correct` (instantiated at the tiles the state actually
holds). -/
theorem kldivergence_kernel_region_run
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem y_ptr (s₀.pid * BLOCK_SIZE + j.val) = ys j) :
    ∃ s1, exec ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = klDivSpec (xs j) (ys j))
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := kldivergence_kernel_correct x_ptr y_ptr output_ptr n_elements
    BLOCK_SIZE s₀ (fun j => s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun j => s₀.readMem y_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE) s₀
      = exec ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, kldivergence_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, NumericDType.add,
          NumericDType.mul, NumericDType.div, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj, hy j hj]
      · refine kldivergence_kernel_frame x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem kldivergence_kernel_traceSafe
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
      ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [kldivergence_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hy a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem kldivergence_kernel_flattenOk
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [kldivergence_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `kldivergence_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tiles / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
def kldivIO (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
  in1 := x_ptr
  in2 := y_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `kldivergence_kernel` implements the pointwise KL term
`klDivSpec x y = x * log (x / y)` on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose input windows hold `xs`/`ys`
at the active lanes, the translated pointer kernel terminates, every active
output lane holds `klDivSpec (xs i) (ys i)`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
theorem kldivergence_kernel_correctness_bridge
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    kldivIO x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => klDivSpec (xs i) (ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact kldivergence_kernel_flattenOk x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact kldivergence_kernel_traceSafe x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := kldivergence_kernel_region_run
      x_ptr y_ptr output_ptr n_elements BLOCK_SIZE s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.KldivCompute

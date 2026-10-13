import bench.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Lane `i`'s address in a row-strided buffer for program
`(row, col_block) = (pids 0, pids 1)`: row base `pids 0 * stride` plus column
`pids 1 * BLOCK_N + i`. -/
def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

/-- Algorithm-layer correctness for `_swiglu_fwd_kernel`: active lanes hold
`swiglu (xs i) (ys i)`, out-of-bounds lanes are preserved. -/
theorem swiglu_fwd_kernel_correct
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s s' : BlockState)
    (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (hExec : exec (swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
          ncols BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_out_row BLOCK_N i
      s'.readMem OUT outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          TiledActivation.swiglu (xs i) (ys i)
        else s.readMem OUT outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [exec, swiglu_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [swigluOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
  · have hx := h_x i
    have hy := h_y i
    simp [swigluOffset, Nat.add_assoc] at hx hy
    simp [hi, TiledActivation.swiglu, TiledActivation.silu,
          tile_elementwise, hx, hy]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `OUT`, and the *inactive* lanes
of the output window itself — is preserved by the run. -/
private theorem swiglu_fwd_kernel_frame
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s s1 : BlockState)
    (hExec : exec ((swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row
        stride_out_row ncols BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_N, s.pids 1 * BLOCK_N + i.val < ncols →
      ¬(OUT = r ∧ s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val) = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, swiglu_fwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body, so no `0 < BLOCK_N` side condition is needed). -/
private theorem swiglu_fwd_kernel_exec_isSome
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s : BlockState) :
    ∃ s1, exec ((swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row
        stride_out_row ncols BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, swiglu_fwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only**. This is the `hrun`
obligation of the `⊨` headline; the value half reuses
`swiglu_fwd_kernel_correct` (instantiated at the tiles the state actually
holds). -/
theorem swiglu_fwd_kernel_region_run
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s₀ : BlockState) (xs ys : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
      s₀.readMem X (s₀.pids 0 * stride_x_row + s₀.pids 1 * BLOCK_N + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
      s₀.readMem Y (s₀.pids 0 * stride_y_row + s₀.pids 1 * BLOCK_N + j.val) = ys j) :
    ∃ s1, exec ((swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row
          stride_out_row ncols BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
          s1.readMem OUT (s₀.pids 0 * stride_out_row + s₀.pids 1 * BLOCK_N + j.val)
            = TiledActivation.swiglu (xs j) (ys j))
      ∧ (∀ r o,
          (r ≠ OUT ∨ ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
            o ≠ s₀.pids 0 * stride_out_row + s₀.pids 1 * BLOCK_N + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_fwd_kernel_exec_isSome X Y OUT stride_x_row
    stride_y_row stride_out_row ncols BLOCK_N s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := swiglu_fwd_kernel_correct X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N s₀ s1
      (fun k => s₀.readMem X (swigluOffset s₀ stride_x_row BLOCK_N k))
      (fun k => s₀.readMem Y (swigluOffset s₀ stride_y_row BLOCK_N k))
      (fun _ => rfl) (fun _ => rfl) hs1 j
    simp only [swigluOffset, hj, if_pos] at h
    rw [h, hx j hj, hy j hj]
  · refine swiglu_fwd_kernel_frame X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi (by omega)

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, `.to(tl.float32)` casts, `sigmoid`,
masked store). -/
theorem swiglu_fwd_kernel_flattenOk
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ((swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_fwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — seven are memory-silent (`program_id`×2, the pointer staging,
`arange`, the register arithmetic) — and reduces the three masked accesses
(`X` load, `Y` load, `OUT` store, lane `j` at
`pid₀ * stride + pid₁ * BLOCK_N + j`, active iff `pid₁ * BLOCK_N + j < ncols`)
to the **lane-wise** bounds hypotheses: every *active* lane's address is below
the region bound of the buffer it touches. -/
theorem swiglu_fwd_kernel_traceSafe
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_x_row + s.pids 1 * BLOCK_N + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_y_row + s.pids 1 * BLOCK_N + j.val < bounds Y)
    (hout : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + j.val < bounds OUT) :
    Kernel.TraceSafe bounds
      ((swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [swiglu_fwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨fun a ha => ?_, fun a ha => ?_, fun a ha => ?_⟩
  · simpa [Nat.add_assoc] using hin1 a ha
  · simpa [Nat.add_assoc] using hin2 a ha
  · simpa [Nat.add_assoc] using hout a ha

/-- `_swiglu_fwd_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: gate `X`,
  value `Y`, result `OUT`);
* `B = BLOCK_N` — the column tile each program owns;
* `read1`/`read2`/`write` — program `(row, col_block) = (pid₀, pid₁)` touches
  lane `j` at `pid₀ * stride + pid₁ * BLOCK_N + j` in all three buffers (the
  host-side 2-D grid `(M, cdiv(N, BLOCK_N))` launch convention, per-buffer row
  strides);
* `mask` — the active lanes `pid₁ * BLOCK_N + j < ncols`: the column prefix
  that actually exists in the row. Inactive lanes (the overhang of the last
  column block) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def swigluIO (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    Masked2DKernelIO₂ where
  kernel := swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
    ncols BLOCK_N
  in1 := X
  in2 := Y
  out := OUT
  B := BLOCK_N
  read1 := fun pid₀ pid₁ j => pid₀ * stride_x_row + pid₁ * BLOCK_N + j.val
  read2 := fun pid₀ pid₁ j => pid₀ * stride_y_row + pid₁ * BLOCK_N + j.val
  write := fun pid₀ pid₁ j => pid₀ * stride_out_row + pid₁ * BLOCK_N + j.val
  mask := fun _ pid₁ j => pid₁ * BLOCK_N + j.val < ncols


end VeriTile.Bench.TritonBenchG.SwigluFwd

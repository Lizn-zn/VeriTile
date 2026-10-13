import bench.tritonbench_g.swiglu_backward.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluBackward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val

/-- Algorithm-layer correctness for `_swiglu_bwd_kernel`.

The kernel stores `DX`, then `DY`, then optionally `OUT`. We assume these output
regions are distinct so the later stores cannot overwrite earlier channels. -/
theorem swiglu_bwd_kernel_correct
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (s s' : BlockState)
    (xs ys douts : Fin BLOCK_N → ℝ)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (h_dout : ∀ i : Fin BLOCK_N, s.readMem DOUT (swigluOffset s stride_dout_row BLOCK_N i) = douts i)
    (hExec : exec (swiglu_bwd_kernel X Y DOUT OUT DX DY
          stride_x_row stride_y_row stride_dout_row stride_out_row
          stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT) s = some s') :
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_dx_row BLOCK_N i
      s'.readMem DX outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          TiledActivation.swigluBwdA (douts i) (xs i) (ys i)
        else s.readMem DX outAddr) ∧
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_dy_row BLOCK_N i
      s'.readMem DY outAddr =
        if s.pids 1 * BLOCK_N + i.val < ncols then
          TiledActivation.swigluBwdB (douts i) (xs i)
        else s.readMem DY outAddr) ∧
    (∀ i : Fin BLOCK_N,
      let outAddr := swigluOffset s stride_out_row BLOCK_N i
      s'.readMem OUT outAddr =
        if RECOMPUTE_OUTPUT then
          if s.pids 1 * BLOCK_N + i.val < ncols then
            TiledActivation.swiglu (xs i) (ys i)
          else s.readMem OUT outAddr
        else s.readMem OUT outAddr) := by
  have h_inj_dx : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_dx_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_dx_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  have h_inj_dy : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_dy_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_dy_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  have h_inj_out : Function.Injective
      (fun idx : TileIndex [BLOCK_N] =>
        s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + a.val =
        s.pids 0 * stride_out_row + s.pids 1 * BLOCK_N + b.val := by
      simpa [Nat.add_assoc] using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  cases RECOMPUTE_OUTPUT <;>
    simp [exec, swiglu_bwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  · subst s'
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := DX) (h_ne := hDXDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dx (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hy hdout
        simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu,
              hx, hy, hdout]
        ring
      · simp [hi]
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dy (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hdout
        simp [hi, TiledActivation.swigluBwdB, TiledActivation.silu,
              hx, hdout]
        ring
      · rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := DY) (h_ne := Ne.symm hDXDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp [hi]
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := OUT) (h_ne := hOUTDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DX) (R := OUT) (h_ne := hOUTDX)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
  · subst s'
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := OUT) (R := DX) (h_ne := Ne.symm hOUTDX)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := DY) (R := DX) (h_ne := hDXDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dx (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hy hdout
        simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu,
              hx, hy, hdout]
        ring
      · simp [hi]
    constructor
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := OUT) (R := DY) (h_ne := Ne.symm hOUTDY)
        (P := fun idx : TileIndex [BLOCK_N] =>
          s.pids 1 * BLOCK_N + idx.1.val < ncols)
        (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
        (l := TileShape.allIndices [BLOCK_N])]
      simp
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_dy (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hdout := h_dout i
        simp [swigluOffset, Nat.add_assoc] at hx hdout
        simp [hi, TiledActivation.swigluBwdB, TiledActivation.silu,
              hx, hdout]
        ring
      · rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := DY) (h_ne := Ne.symm hDXDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp [hi]
    · intro i
      simp only [swigluOffset, Nat.add_assoc]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj_out (i, PUnit.unit)]
      by_cases hi : s.pids 1 * BLOCK_N + i.val < ncols
      · have hx := h_x i
        have hy := h_y i
        simp [swigluOffset, Nat.add_assoc] at hx hy
        simp [hi, TiledActivation.swiglu, TiledActivation.silu,
              hx, hy]
      · simp [hi]
        rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DY) (R := OUT) (h_ne := hOUTDY)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        rw [BlockState.scatter_prop_masked_preserves_other_region
          (region := DX) (R := OUT) (h_ne := hOUTDX)
          (P := fun idx : TileIndex [BLOCK_N] =>
            s.pids 1 * BLOCK_N + idx.1.val < ncols)
          (off := s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val))
          (l := TileShape.allIndices [BLOCK_N])]
        simp

/-! ### The `⊨` specification -/

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the (up to three)
masked stores — every cell of every region other than `DX`/`DY`/`OUT`, the
*inactive* lanes of the output windows themselves, and the whole `OUT` window
when `RECOMPUTE_OUTPUT` is off — is preserved by the run. -/
private theorem swiglu_bwd_kernel_frame
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (s s1 : BlockState)
    (hExec : exec ((swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N
        RECOMPUTE_OUTPUT).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissDX : ∀ i : Fin BLOCK_N, s.pids 1 * BLOCK_N + i.val < ncols →
      ¬(DX = r ∧ s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + i.val) = o))
    (hmissDY : ∀ i : Fin BLOCK_N, s.pids 1 * BLOCK_N + i.val < ncols →
      ¬(DY = r ∧ s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + i.val) = o))
    (hmissOUT : ∀ i : Fin BLOCK_N, RECOMPUTE_OUTPUT = Bool.true →
      s.pids 1 * BLOCK_N + i.val < ncols →
      ¬(OUT = r ∧ s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + i.val) = o)) :
    s1.mem r o = s.mem r o := by
  cases RECOMPUTE_OUTPUT with
  | «false» =>
      simp [exec, swiglu_bwd_kernel, ComputeKernel.toAlgKernel, stepStmts,
            stepStmt, evalOp.eq_def, tile_elementwise,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
      subst hExec
      refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
        (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
      · intro k _ hmk hc
        exact hmissDY k.1 (by simpa using hmk) hc
      · intro k _ hmk hc
        exact hmissDX k.1 (by simpa using hmk) hc
  | «true» =>
      simp [exec, swiglu_bwd_kernel, ComputeKernel.toAlgKernel, stepStmts,
            stepStmt, evalOp.eq_def, tile_elementwise,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
      subst hExec
      refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
        (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
          (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl))
      · intro k _ hmk hc
        exact hmissOUT k.1 rfl (by simpa using hmk) hc
      · intro k _ hmk hc
        exact hmissDY k.1 (by simpa using hmk) hc
      · intro k _ hmk hc
        exact hmissDX k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body in both `RECOMPUTE_OUTPUT` branches, so no `0 < BLOCK_N` side
condition is needed). -/
private theorem swiglu_bwd_kernel_exec_isSome
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) (s : BlockState) :
    ∃ s1, exec ((swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N
        RECOMPUTE_OUTPUT).toAlgKernel) s = some s1 := by
  cases RECOMPUTE_OUTPUT <;>
    simp [exec, swiglu_bwd_kernel, ComputeKernel.toAlgKernel, stepStmts,
          stepStmt, evalOp.eq_def, tile_elementwise,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, write-active-lane
values of all three output channels, and frame off the write-active output
lanes, from any launch state whose input windows are loaded at the **active
lanes only** (`pid₁ · BLOCK_N + j < ncols`). This is the `hrun` obligation of
the `⊨` headline; the value half reuses `swiglu_bwd_kernel_correct`
(instantiated at the tiles the state actually holds). -/
theorem swiglu_bwd_kernel_region_run
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY)
    (s₀ : BlockState) (xs ys douts : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
      s₀.readMem X (s₀.pids 0 * stride_x_row + (s₀.pids 1 * BLOCK_N + j.val)) = xs j)
    (hy : ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
      s₀.readMem Y (s₀.pids 0 * stride_y_row + (s₀.pids 1 * BLOCK_N + j.val)) = ys j)
    (hdout : ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
      s₀.readMem DOUT (s₀.pids 0 * stride_dout_row + (s₀.pids 1 * BLOCK_N + j.val)) = douts j) :
    ∃ s1, exec ((swiglu_bwd_kernel X Y DOUT OUT DX DY
          stride_x_row stride_y_row stride_dout_row stride_out_row
          stride_dx_row stride_dy_row ncols BLOCK_N
          RECOMPUTE_OUTPUT).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
          s1.readMem DX (s₀.pids 0 * stride_dx_row + (s₀.pids 1 * BLOCK_N + j.val))
            = TiledActivation.swigluBwdA (douts j) (xs j) (ys j))
      ∧ (∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
          s1.readMem DY (s₀.pids 0 * stride_dy_row + (s₀.pids 1 * BLOCK_N + j.val))
            = TiledActivation.swigluBwdB (douts j) (xs j))
      ∧ (∀ j : Fin BLOCK_N,
          RECOMPUTE_OUTPUT = Bool.true ∧ s₀.pids 1 * BLOCK_N + j.val < ncols →
          s1.readMem OUT (s₀.pids 0 * stride_out_row + (s₀.pids 1 * BLOCK_N + j.val))
            = TiledActivation.swiglu (xs j) (ys j))
      ∧ (∀ r o,
          (r ≠ DX ∨ ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
            o ≠ s₀.pids 0 * stride_dx_row + (s₀.pids 1 * BLOCK_N + j.val)) →
          (r ≠ DY ∨ ∀ j : Fin BLOCK_N, s₀.pids 1 * BLOCK_N + j.val < ncols →
            o ≠ s₀.pids 0 * stride_dy_row + (s₀.pids 1 * BLOCK_N + j.val)) →
          (r ≠ OUT ∨ ∀ j : Fin BLOCK_N,
            RECOMPUTE_OUTPUT = Bool.true ∧ s₀.pids 1 * BLOCK_N + j.val < ncols →
            o ≠ s₀.pids 0 * stride_out_row + (s₀.pids 1 * BLOCK_N + j.val)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_bwd_kernel_exec_isSome X Y DOUT OUT DX DY
    stride_x_row stride_y_row stride_dout_row stride_out_row
    stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT s₀
  have hcorr := swiglu_bwd_kernel_correct X Y DOUT OUT DX DY
    stride_x_row stride_y_row stride_dout_row stride_out_row
    stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT s₀ s1
    (fun i => s₀.readMem X (swigluOffset s₀ stride_x_row BLOCK_N i))
    (fun i => s₀.readMem Y (swigluOffset s₀ stride_y_row BLOCK_N i))
    (fun i => s₀.readMem DOUT (swigluOffset s₀ stride_dout_row BLOCK_N i))
    hDXDY hOUTDX hOUTDY (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun j hj => ?_, fun j hj => ?_,
    fun r o h1 h2 h3 => ?_⟩
  · -- `DX` channel realizes `swigluBwdA`.
    have h := hcorr.1 j
    simp only [swigluOffset, Nat.add_assoc, hj, if_pos] at h
    rw [h, hdout j hj, hx j hj, hy j hj]
  · -- `DY` channel realizes `swigluBwdB`.
    have h := hcorr.2.1 j
    simp only [swigluOffset, Nat.add_assoc, hj, if_pos] at h
    rw [h, hdout j hj, hx j hj]
  · -- `OUT` channel realizes the recomputed forward `swiglu`.
    obtain ⟨hR, hj⟩ := hj
    subst hR
    have h := hcorr.2.2 j
    simp only [swigluOffset, Nat.add_assoc, hj, if_pos] at h
    rw [h, hx j hj, hy j hj]
  · -- frame off the three write-active output windows.
    refine swiglu_bwd_kernel_frame X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT s₀ s1 hs1 r o
      (fun i hi ⟨hr, ho⟩ => ?_) (fun i hi ⟨hr, ho⟩ => ?_)
      (fun i hR hi ⟨hr, ho⟩ => ?_)
    · rcases h1 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rcases h2 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rcases h3 with hne | hno
      · exact hne hr.symm
      · exact hno i ⟨hR, hi⟩ ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, `.to(tl.float32)` casts, elementwise
math including `tl.sigmoid`, masked stores) — in both `RECOMPUTE_OUTPUT`
branches. -/
theorem swiglu_bwd_kernel_flattenOk
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) :
    ((swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N
        RECOMPUTE_OUTPUT).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases RECOMPUTE_OUTPUT <;>
    simp [swiglu_bwd_kernel, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: all three masked loads and the (up to three)
masked stores address per-buffer windows `pid₀ · stride + pid₁ · BLOCK_N + j`,
active only when `pid₁ · BLOCK_N + j < ncols`, so the bounds contract is
**lane-wise** — every *active* lane's address is below the region bound of the
buffer it touches; the `OUT` obligation only arises when `RECOMPUTE_OUTPUT`. -/
theorem swiglu_bwd_kernel_traceSafe
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hinx : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_x_row + (s.pids 1 * BLOCK_N + j.val) < bounds X)
    (hiny : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_y_row + (s.pids 1 * BLOCK_N + j.val) < bounds Y)
    (hindout : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_dout_row + (s.pids 1 * BLOCK_N + j.val) < bounds DOUT)
    (houtdx : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_dx_row + (s.pids 1 * BLOCK_N + j.val) < bounds DX)
    (houtdy : ∀ j : Fin BLOCK_N, s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_dy_row + (s.pids 1 * BLOCK_N + j.val) < bounds DY)
    (houtout : ∀ j : Fin BLOCK_N,
      RECOMPUTE_OUTPUT = Bool.true ∧ s.pids 1 * BLOCK_N + j.val < ncols →
      s.pids 0 * stride_out_row + (s.pids 1 * BLOCK_N + j.val) < bounds OUT) :
    Kernel.TraceSafe bounds
      ((swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N
        RECOMPUTE_OUTPUT).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  cases RECOMPUTE_OUTPUT with
  | «false» =>
      simp [swiglu_bwd_kernel, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        BlockState.setReg,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]
      exact ⟨fun a ha => hinx a ha, fun a ha => hiny a ha,
        fun a ha => hindout a ha, fun a ha => houtdx a ha,
        fun a ha => houtdy a ha⟩
  | «true» =>
      simp [swiglu_bwd_kernel, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        BlockState.setReg,
        Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt]
      exact ⟨fun a ha => hinx a ha, fun a ha => hiny a ha,
        fun a ha => hindout a ha, fun a ha => houtdx a ha,
        fun a ha => houtdy a ha, fun a ha => houtout a ⟨rfl, ha⟩⟩

/-- `_swiglu_bwd_kernel`'s masked three-input / three-output **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`in3` — the gate `X`, value `Y`, upstream gradient `DOUT`;
* `out1`/`out2`/`out3` — the input gradients `DX`, `DY`, and the optionally
  recomputed forward output `OUT`;
* `B = BLOCK_N` — the column window each program owns;
* `read1..3`/`write1..3` — **per-lane 2-D windows**: program `(pid₀, pid₁)`
  touches buffer `P` at `pid₀ · stride_p_row + pid₁ · BLOCK_N + j` (the
  host-side `(M, cdiv(N, BLOCK_N))` launch convention: `pid₀` picks the row,
  `pid₁` the column block, each buffer with its own row stride);
* `mask` — the active lanes `pid₁ · BLOCK_N + j < ncols`, shared by all three
  loads and the `DX`/`DY` stores (`read2Mask`/`read3Mask`/`writeMask1`/
  `writeMask2` keep their defaults);
* `writeMask3` — the **constexpr-gated** `OUT` store:
  `RECOMPUTE_OUTPUT = Bool.true ∧ pid₁ · BLOCK_N + j < ncols`. When the heuristic
  flag is off, the `OUT` channel has no write-active lanes, so it carries no
  value, bounds, or frame obligations — exactly the kernel's dead
  `if RECOMPUTE_OUTPUT` branch.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and constexpr gating match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the write-active lanes. -/
def swigluBackwardIO (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) : Masked2DKernelIO₃ₓ₃ where
  kernel := swiglu_bwd_kernel X Y DOUT OUT DX DY
    stride_x_row stride_y_row stride_dout_row stride_out_row
    stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
  in1 := X
  in2 := Y
  in3 := DOUT
  out1 := DX
  out2 := DY
  out3 := OUT
  B := BLOCK_N
  read1 := fun pid₀ pid₁ j => pid₀ * stride_x_row + (pid₁ * BLOCK_N + j.val)
  read2 := fun pid₀ pid₁ j => pid₀ * stride_y_row + (pid₁ * BLOCK_N + j.val)
  read3 := fun pid₀ pid₁ j => pid₀ * stride_dout_row + (pid₁ * BLOCK_N + j.val)
  write1 := fun pid₀ pid₁ j => pid₀ * stride_dx_row + (pid₁ * BLOCK_N + j.val)
  write2 := fun pid₀ pid₁ j => pid₀ * stride_dy_row + (pid₁ * BLOCK_N + j.val)
  write3 := fun pid₀ pid₁ j => pid₀ * stride_out_row + (pid₁ * BLOCK_N + j.val)
  mask := fun _ pid₁ j => pid₁ * BLOCK_N + j.val < ncols
  writeMask3 := fun _ pid₁ j =>
    RECOMPUTE_OUTPUT = Bool.true ∧ pid₁ * BLOCK_N + j.val < ncols


end VeriTile.Bench.TritonBenchG.SwigluBackward

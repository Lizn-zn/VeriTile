import VeriTile.Triton

/-!
# `swiglu_backward` — strict per-kernel correctness

`_swiglu_bwd_kernel` is the backward pass of SwiGLU: program `(row, col_block)`
loads the gate `X`, value `Y`, and upstream gradient `DOUT`, recomputes
`x_sigmoid = σ(x)`, and writes the two input gradients `dx`→`DX`, `dy`→`DY`, plus
optionally (when `RECOMPUTE_OUTPUT`) the forward output `out = x·σ(x)·y`→`OUT`.
All stores are masked by `cols < ncols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_swiglu_bwd_kernel[grid](...)`, the 2-D grid
`(M, cdiv(N, BLOCK_N))`, the `@triton.autotune` choice of `BLOCK_N`, the
`@triton.heuristics` setting of `RECOMPUTE_OUTPUT`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because both program ids (`pid₀`, `pid₁`) are universally
quantified, the per-program statement covers every program of the grid; the
proofs case-split on the `RECOMPUTE_OUTPUT` flag, covering both heuristic
outcomes.

## Proof architecture

```
swiglu_bwd_kernel_correctness                 ← TOP SPECIFICATION (swigluBackwardIO ⊨ backward triple)
  ├─ swiglu_bwd_kernel_flattenOk              bridge fragment membership
  ├─ swiglu_bwd_kernel_traceSafe              per-execution lane-wise safety walk
  └─ swiglu_bwd_kernel_region_run             region-model masked Hoare triple
       ├─ swiglu_bwd_kernel_exec_isSome       termination
       ├─ swiglu_bwd_kernel_correct           algorithm-layer readback per lane
       └─ swiglu_bwd_kernel_frame             three-scatter cell frame
```

The headline is stated on the kernel's masked three-input / three-output **IO
signature** `swigluBackwardIO` (`Masked2DKernelIO₃ₓ₃`). `⊨`
(`Masked2DKernelIO₃ₓ₃.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the six buffers, **every**
program id whose active lanes are in bounds, and **every** launch state whose
active input lanes hold `xs`/`ys`/`douts`, the translated pointer kernel
terminates, every write-active output lane holds its spec value, and every
other memory cell is unchanged.

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swigluBwdA/B`, `TiledActivation.swiglu`, built on
`TiledActivation.silu`) lives once in `Math.Activation` and is reused here, so
this file only checks that the kernel realizes those oracles lane-wise.
Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)` load
casts reduce to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`); `@triton.autotune` is not modeled. The kernel writes up to three
channels (`DX`, `DY`, and conditionally `OUT`); the headline assumes the
three output regions are pairwise distinct (`DX ≠ DY`, `OUT ≠ DX`, `OUT ≠ DY`) so
later stores cannot clobber earlier channels. The `RECOMPUTE_OUTPUT`
control-flow branch is **not** case-narrowed away: the constexpr gate lives in
the signature's `writeMask3` (`RECOMPUTE_OUTPUT = Bool.true ∧ cols < ncols`), so one
headline covers both heuristic outcomes — when the flag is off, the `OUT`
channel simply has no write-active lanes and carries no obligations.
-/

namespace VeriTile.Bench.TritonBenchG.SwigluBackward

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₃ₓ₃

/-- Faithful transcription of `swiglu_backward.py`'s `_swiglu_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` / `RECOMPUTE_OUTPUT: tl.constexpr` -> Lean
  parameters. -/
def swiglu_bwd_kernel
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  DOUT += row * $(stride_dout_row)
  if RECOMPUTE_OUTPUT {
    OUT += row * $(stride_out_row)
  }
  DX += row * $(stride_dx_row)
  DY += row * $(stride_dy_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  dout = tl.load(DOUT + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  x_sigmoid = tl.sigmoid(x)
  dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
  dy = x * x_sigmoid * dout
  tl.store(DX + cols, dx, mask=cols < $(ncols))
  tl.store(DY + cols, dy, mask=cols < $(ncols))
  if RECOMPUTE_OUTPUT {
    out = x * x_sigmoid * y
    tl.store(OUT + cols, out, mask=cols < $(ncols))
  }
}

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

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked stores). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

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

/-- **The headline**: `_swiglu_bwd_kernel` implements the SwiGLU backward
oracles on its masked three-input / three-output IO signature — for every
disjoint flat placement of the six buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold
`xs` (the gate), `ys` (the value) and `douts` (the upstream gradient), the
translated pointer kernel terminates, every active `DX` lane `j` holds
`TiledActivation.swigluBwdA (douts j) (xs j) (ys j)`, every active `DY` lane
holds `TiledActivation.swigluBwdB (douts j) (xs j)`, every `OUT` lane that is
active **and** `RECOMPUTE_OUTPUT`-gated holds the recomputed forward
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged — one statement covering both heuristic outcomes of the constexpr
flag. The output buffers must be pairwise distinct (`DX ≠ DY`, `OUT ≠ DX`,
`OUT ≠ DY`) so each readback sees through the later stores. Proof:
`Masked2DKernelIO₃ₓ₃.Implements.intro` assembles the region-model masked
triple with the flat-memory bridge side conditions. -/
specification swiglu_bwd_kernel_correctness
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY) :
    swigluBackwardIO X Y DOUT OUT DX DY stride_x_row stride_y_row
        stride_dout_row stride_out_row stride_dx_row stride_dy_row ncols
        BLOCK_N RECOMPUTE_OUTPUT ⊨
      fun _ _ xs ys douts =>
        (fun j => TiledActivation.swigluBwdA (douts j) (xs j) (ys j),
         fun j => TiledActivation.swigluBwdB (douts j) (xs j),
         fun j => TiledActivation.swiglu (xs j) (ys j)) := by
  refine Masked2DKernelIO₃ₓ₃.Implements.intro _ ?_ ?_ ?_
  · exact swiglu_bwd_kernel_flattenOk X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
  · intro bounds s h1 h2 h3 h4 h5 h6
    exact swiglu_bwd_kernel_traceSafe X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
      bounds s h1 h2 h3 h4 h5 h6
  · intro s₀ xs ys douts hx hy hdout
    exact swiglu_bwd_kernel_region_run X Y DOUT OUT DX DY
      stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
      hDXDY hOUTDX hOUTDY s₀ xs ys douts hx hy hdout

end VeriTile.Bench.TritonBenchG.SwigluBackward

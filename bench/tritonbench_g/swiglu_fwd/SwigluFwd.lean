import VeriTile.Triton

/-!
# `swiglu_fwd` — strict per-kernel correctness

`_swiglu_fwd_kernel` is the forward SwiGLU: program `(row, col_block)` loads a
tile of gate `X` and value `Y` for its row, computes `swiglu(x, y) = x · σ(x) · y`
lane-wise, and stores the result to `OUT`, masked by `cols < ncols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_swiglu_fwd_kernel[grid](...)`, the 2-D grid
`(M, cdiv(N, BLOCK_N))`, the `@triton.autotune` choice of `BLOCK_N`, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because both program ids (`pids 0`, `pids 1`) are
universally quantified, the per-program statement covers every program of the
grid.

## Proof architecture

```
swiglu_fwd_kernel_correctness                 ← TOP THEOREM (swigluIO ⊨ lane-wise swiglu)
  ├─ swiglu_fwd_kernel_flattenOk              bridge fragment membership
  ├─ swiglu_fwd_kernel_traceSafe              per-execution lane-wise safety walk
  └─ swiglu_fwd_kernel_region_run             region-model masked Hoare triple
       ├─ swiglu_fwd_kernel_exec_isSome       termination
       ├─ swiglu_fwd_kernel_correct           algorithm-layer readback per lane
       └─ swiglu_fwd_kernel_frame             masked scatter frame
```

The headline is stated on the kernel's masked **IO signature** `swigluIO`
(`Masked2DKernelIO₂`, the two-axis family: this kernel reads
`tl.program_id(1)`): which buffer is which argument, where program
`(row, col_block) = (pid₀, pid₁)` reads its `X`/`Y` tiles and writes its `OUT`
tile (lane `j` at `pid₀ * stride + pid₁ * BLOCK_N + j`), and the active-lane
predicate `pid₁ * BLOCK_N + j < ncols`. `⊨` is the audit-once masked
Hoare-triple combinator (`Masked2DKernelIO₂.Implements`): for **every** disjoint
placement of the three buffers in flat memory, **every** program-id pair all of
whose *active* lanes are in bounds (partial blocks may overhang the buffer on
their inactive lanes), and **every** launch state whose input windows hold
`xs`/`ys` at the active lanes, the translated pointer kernel terminates, every
active output lane holds `TiledActivation.swiglu (xs j) (ys j)`, and every
other memory cell is unchanged.

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swiglu`, built on `TiledActivation.silu`) lives
once in `Math.Activation` and is reused here, so this file only checks that the
kernel realizes that oracle lane-wise. Arithmetic is over `ℝ` (not bit-accurate
IEEE float); the `.to(tl.float32)` casts reduce to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`); `@triton.autotune` is not modeled.
No output/input disjointness is assumed in the region model: both inputs are
read into registers before the scatter, so the result is correct even if `OUT`
aliases `X` or `Y`.
-/

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Faithful transcription of `swiglu_fwd.py`'s `_swiglu_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
def swiglu_fwd_kernel
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  OUT += row * $(stride_out_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}

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

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
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

/-- **The headline**: `_swiglu_fwd_kernel` implements the oracle SwiGLU
`TiledActivation.swiglu` (i.e. `x · σ(x) · y`) lane-wise on its masked 2-D IO
signature — for every disjoint flat placement of the three buffers, every
program-id pair `(row, col_block)` whose active lanes are in bounds, and every
launch state whose active input-window lanes hold `xs`/`ys`, the translated
pointer kernel terminates, every active output lane `j` holds
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification swiglu_fwd_kernel_correctness
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    swigluIO X Y OUT stride_x_row stride_y_row stride_out_row ncols BLOCK_N
      ⊨ fun _ _ xs ys i => TiledActivation.swiglu (xs i) (ys i) := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact swiglu_fwd_kernel_flattenOk X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact swiglu_fwd_kernel_traceSafe X Y OUT stride_x_row stride_y_row
      stride_out_row ncols BLOCK_N bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := swiglu_fwd_kernel_region_run
      X Y OUT stride_x_row stride_y_row stride_out_row ncols BLOCK_N s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.SwigluFwd

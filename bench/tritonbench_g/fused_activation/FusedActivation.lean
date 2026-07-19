import VeriTile.Triton

/-!
# `fused_activation` — strict per-kernel correctness

`fused_add_mul_activation_kernel` fuses a biased linear combine with an
activation: program `pid` loads block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of
`x_ptr` and `in_ptr`, plus a broadcast bias (`bias_ptr` indexed by
`index % num_weights`), forms `multiplier · in + x + bias`, applies either
`tl.sigmoid` or `tl.maximum(0, ·)` (ReLU) by the `activation` selector, and
stores **in place** to `x_ptr`, masked by `index < xnumel`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`fused_add_mul_activation_kernel[grid](...)`, the grid
size `cdiv(numel, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid. The Python `if activation == "sigmoid" / elif "relu"` string branch
is modeled by a Boolean `activation` (`true` = sigmoid, `false` = ReLU); both
branches are verified.

## Proof architecture

```
fused_add_mul_activation_kernel_correctness   ← TOP SPECIFICATION
  · fusedActivationIO ⊨ fusedActivationSpec: the grouped masked in-place triple
    (GroupedMasked2DKernelIO — 3 read channels, 1 in-place write channel)
  ├─ fused_add_mul_activation_kernel_flattenOk    bridge fragment membership
  ├─ fused_add_mul_activation_kernel_traceSafe    per-execution lane-wise safety walk
  └─ fused_add_mul_activation_kernel_region_run   region-model masked in-place triple
       ├─ fused_add_mul_activation_kernel_exec_isSome  termination
       ├─ fused_add_mul_activation_kernel_correct      algorithm-layer readback per lane
       └─ fused_add_mul_activation_kernel_frame        masked-store cell-level frame
```

The spec `fusedActivationSpec` applies the selected activation to
`fusedActivationInput = multiplier · input + x + bias`. The three read channels
of the IO signature are the `x_ptr` window, the **broadcast** `bias_ptr` window
at `index % num_weights`, and the `in_ptr` window; the single write channel is
`x_ptr` again (in-place, duplicate-region wiring via the skin's decoupled
`bufs`). The bias window is exactly why this kernel needs a *general-window*
skin: `(pid·BLOCK_SIZE + j) % num_weights` is not of the contiguous
`base + j` form the `MaskedKernelIO*` family assumes.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `tl.sigmoid` is modeled by
`Real.sigmoid` and the ReLU branch by the `tl.maximum` algorithm-layer form;
`@triton.autotune` is not modeled. The kernel writes in place to `x_ptr`; the
loaded tiles are read into registers before the scatter, so correctness holds
for the in-place store — no output/input disjointness side condition is needed.
-/

namespace VeriTile.Bench.TritonBenchG.FusedActivation

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `elif activation == "relu"` -> Lean
  Boolean `activation`; `true` selects sigmoid and `false` selects ReLU. -/
def fused_add_mul_activation_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = $(multiplier) * tmp3 + tmp0 + tmp1
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}

noncomputable def fusedActivationInput
    (x bias input multiplier : ℝ) : ℝ :=
  multiplier * input + x + bias

/-- Algorithm-layer branch form of the activation selector. `false` is the
`tl.maximum(0, x)` ReLU branch. -/
noncomputable def fusedActivationSpec
    (ACTIVATION_SIGMOID : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := fusedActivationInput x bias input multiplier
  if ACTIVATION_SIGMOID then
    Real.sigmoid z
  else
    WithBot.unbotD 0
      (if ComparableDType.real.gt (some 0) (some z) then
        (some 0 : WithBot ℝ)
      else
        (some z : WithBot ℝ))

def fusedActivationOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Algorithm-layer correctness for `fused_add_mul_activation_kernel`: at every
in-bounds lane the in-place store leaves `fusedActivationSpec` of the loaded
values, and every out-of-bounds lane of the window is preserved. The read
hypotheses are needed only at the active (in-bounds) lanes — exactly the lanes
the kernel's masked loads touch. -/
theorem fused_add_mul_activation_kernel_correct
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s s' : BlockState)
    (xs inputs : Fin BLOCK_SIZE → ℝ)
    (biases : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem x_ptr (fusedActivationOffset s BLOCK_SIZE i) = xs i)
    (h_in : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem in_ptr (fusedActivationOffset s BLOCK_SIZE i) = inputs i)
    (h_bias : ∀ i : Fin BLOCK_SIZE, fusedActivationOffset s BLOCK_SIZE i < xnumel →
      s.readMem bias_ptr ((fusedActivationOffset s BLOCK_SIZE i) % num_weights) = biases i)
    (hExec : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := fusedActivationOffset s BLOCK_SIZE i
      s'.readMem x_ptr outAddr =
        if outAddr < xnumel then
          fusedActivationSpec ACTIVATION_SIGMOID (xs i) (biases i) (inputs i) multiplier
        else s.readMem x_ptr outAddr := by
  intro i
  cases ACTIVATION_SIGMOID
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i (by simpa [fusedActivationOffset] using hi)
      have hin := h_in i (by simpa [fusedActivationOffset] using hi)
      have hb := h_bias i (by simpa [fusedActivationOffset] using hi)
      simp [fusedActivationOffset] at hx hin hb
      by_cases hlt : multiplier * inputs i + xs i + biases i < 0
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
      · simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb,
          ComparableDType.gt]
    · simp [hi]
  · simp [exec, fused_add_mul_activation_kernel, stepStmts, stepStmt, evalOp.eq_def,
          tile_elementwise] at hExec
    subst s'
    simp only [fusedActivationOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pid * BLOCK_SIZE + i.val < xnumel
    · have hx := h_x i (by simpa [fusedActivationOffset] using hi)
      have hin := h_in i (by simpa [fusedActivationOffset] using hi)
      have hb := h_bias i (by simpa [fusedActivationOffset] using hi)
      simp [fusedActivationOffset] at hx hin hb
      simp [hi, fusedActivationSpec, fusedActivationInput, hx, hin, hb]
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

/-- Frame half: every memory cell not actively written by the masked in-place
store — every cell of every region other than `x_ptr`, and the out-of-bounds
lanes of the output window itself — is preserved by the run. -/
private theorem fused_add_mul_activation_kernel_frame
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s s1 : BlockState)
    (hExec : exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + i.val < xnumel →
      ¬(x_ptr = r ∧ s.pids 0 * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  cases ACTIVATION_SIGMOID <;>
  · simp [exec, fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel, stepStmts,
      stepStmt, evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    exact hmiss k.1 hmk hc

/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body in both activation branches). -/
private theorem fused_add_mul_activation_kernel_exec_isSome
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s : BlockState) :
    ∃ s1, exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
        num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s
      = some s1 := by
  cases ACTIVATION_SIGMOID <;>
  simp [exec, fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel, stepStmts,
    stepStmt, evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose three read windows hold `xs`/`biases`/`inputs` at the
in-bounds lanes. This is the `hrun` obligation of the `⊨` headline; the value
half reuses `fused_add_mul_activation_kernel_correct` (the kernel reads
`x_ptr` before the in-place scatter, so the before/after reading is sound). -/
theorem fused_add_mul_activation_kernel_region_run
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (s₀ : BlockState) (xs biases inputs : Fin BLOCK_SIZE → ℝ)
    (h_x : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem x_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = xs j)
    (h_bias : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem bias_ptr ((s₀.pids 0 * BLOCK_SIZE + j.val) % num_weights) = biases j)
    (h_in : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s₀.readMem in_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = inputs j) :
    ∃ s1, exec ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
          num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
          s1.readMem x_ptr (s₀.pids 0 * BLOCK_SIZE + j.val)
            = fusedActivationSpec ACTIVATION_SIGMOID (xs j) (biases j) (inputs j)
                multiplier)
      ∧ (∀ r o,
          (∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < xnumel →
            r ≠ x_ptr ∨ o ≠ s₀.pids 0 * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := fused_add_mul_activation_kernel_exec_isSome x_ptr bias_ptr
    in_ptr num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀
  have hs1' : exec (fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID) s₀ = some s1 := by
    simpa [ComputeKernel.toAlgKernel, fused_add_mul_activation_kernel,
      ComputeExpr.toAlgorithm?] using hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := fused_add_mul_activation_kernel_correct x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀ s1 xs inputs
      biases (by simpa [fusedActivationOffset, BlockState.pid] using h_x)
      (by simpa [fusedActivationOffset, BlockState.pid] using h_in)
      (by simpa [fusedActivationOffset, BlockState.pid] using h_bias) hs1' j
    simp only [fusedActivationOffset, BlockState.pid] at h
    rw [h, if_pos hj]
  · refine fused_add_mul_activation_kernel_frame x_ptr bias_ptr in_ptr num_weights
      xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀ s1 hs1 r o
      (fun i hbounds ⟨hr, ho⟩ => ?_)
    rcases hcond i hbounds with hne | hno
    · exact hne hr.symm
    · exact hno ho.symm

/-- Per-execution safety walk: one computational unfold walks the whole body —
the pid/arange/mask/bias-index arithmetic is memory-silent, and the three
masked loads (`x_ptr` and `in_ptr` at `pid·BLOCK_SIZE + j`, `bias_ptr` at
`(pid·BLOCK_SIZE + j) % num_weights`) plus the masked in-place store to
`x_ptr` reduce to the lane-wise bounds hypotheses at the load mask
`index < xnumel`. Both activation branches are walked. -/
theorem fused_add_mul_activation_kernel_traceSafe
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s.pids 0 * BLOCK_SIZE + j.val < bounds x_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      (s.pids 0 * BLOCK_SIZE + j.val) % num_weights < bounds bias_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < xnumel →
      s.pids 0 * BLOCK_SIZE + j.val < bounds in_ptr) :
    Kernel.TraceSafe bounds
      ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights xnumel
        BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  cases ACTIVATION_SIGMOID <;>
  · simp [fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel,
      ComputeExpr.toAlgorithm?,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg, tile_elementwise,
      Tile.bop, Tile.cop, Tile.uop, Tile.select,
      NumericDType.add, NumericDType.mul, ComparableDType.lt]
    exact ⟨fun a ha => h1 a ha, fun a ha => h2 a ha, fun a ha => h3 a ha,
      fun a ha => h1 a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, three masked loads, the activation branch, and the masked in-place
store). -/
theorem fused_add_mul_activation_kernel_flattenOk
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) :
    ((fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights xnumel
        BLOCK_SIZE multiplier ACTIVATION_SIGMOID).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases ACTIVATION_SIGMOID <;>
  simp [fused_add_mul_activation_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `fused_add_mul_activation_kernel`'s grouped masked **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`GroupedMasked2DKernelIO`, the vector-channel genre; the general per-lane
windows are what the broadcast bias read needs):

* `bufs = [x_ptr, bias_ptr, in_ptr]` — every buffer once; the output channel
  names `x_ptr` again, i.e. the update is **in place**;
* `nIn = 3` — channel 0 = `x_ptr` at `pid₀·BLOCK_SIZE + j`, channel 1 =
  `bias_ptr` at `(pid₀·BLOCK_SIZE + j) % num_weights` (the broadcast bias),
  channel 2 = `in_ptr` at `pid₀·BLOCK_SIZE + j`;
* `nOut = 1` — `x_ptr` at `pid₀·BLOCK_SIZE + j`;
* every read/write gate is the kernel's single mask `pid₀·BLOCK_SIZE + j <
  xnumel`;
* `B = BLOCK_SIZE`; the kernel is a 1-D launch, so the family's second program
  id is ignored by every field.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the declared lanes. -/
def fusedActivationIO
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) : GroupedMasked2DKernelIO where
  kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights
    xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  nIn := 3
  nOut := 1
  bufs := [x_ptr, bias_ptr, in_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => x_ptr
    | ⟨1, _⟩ => bias_ptr
    | ⟨_ + 2, _⟩ => in_ptr
  out := fun _ => x_ptr
  B := BLOCK_SIZE
  read := fun i pid₀ _ j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_SIZE + j.val
    | ⟨1, _⟩ => (pid₀ * BLOCK_SIZE + j.val) % num_weights
    | ⟨_ + 2, _⟩ => pid₀ * BLOCK_SIZE + j.val
  readMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
  write := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  writeMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel

/-- **The headline**: `fused_add_mul_activation_kernel` implements the fused
biased combine + activation on its grouped IO signature — for every disjoint
flat placement of the three buffers, every program id whose masked lanes are in
bounds, and every launch state whose three read windows (including the
broadcast bias window at `index % num_weights`) hold `xs`, the translated
pointer kernel terminates, every active lane `j` (`pid₀·BLOCK_SIZE + j <
xnumel`) of the *same* buffer `x_ptr` ends up holding
`fusedActivationSpec` — the selected activation of
`multiplier·in + x + bias` on the originally-loaded values — and every other
memory cell, including the out-of-bounds lanes of the window, is unchanged.
Both activation branches (`true` = `tl.sigmoid`, `false` = `tl.maximum(0, ·)`)
are covered: `ACTIVATION_SIGMOID` is a free parameter. Proof:
`GroupedMasked2DKernelIO.Implements.intro` assembles the region-model grouped
triple with the flat-memory bridge side conditions. -/
specification fused_add_mul_activation_kernel_correctness
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) :
    fusedActivationIO x_ptr bias_ptr in_ptr num_weights xnumel BLOCK_SIZE
        multiplier ACTIVATION_SIGMOID
      ⊨ fun _ _ xs _ j =>
          fusedActivationSpec ACTIVATION_SIGMOID
            (xs (⟨0, by decide⟩ : Fin 3) j)
            (xs (⟨1, by decide⟩ : Fin 3) j)
            (xs (⟨2, by decide⟩ : Fin 3) j) multiplier := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [fusedActivationIO]
  · exact fused_add_mul_activation_kernel_flattenOk x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  · intro bounds s hread _hwrite
    exact fused_add_mul_activation_kernel_traceSafe x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID bounds s
      (fun j hj => hread (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hread (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hread (⟨2, by decide⟩ : Fin 3) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      fused_add_mul_activation_kernel_region_run x_ptr bias_ptr in_ptr num_weights
        xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀
        (fun j => xs (⟨0, by decide⟩ : Fin 3) j)
        (fun j => xs (⟨1, by decide⟩ : Fin 3) j)
        (fun j => xs (⟨2, by decide⟩ : Fin 3) j)
        (fun j hj => hx (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 3) j hj)
    exact ⟨s1, hexec, fun _o j hj => hval j hj,
      fun r o hcond => hframe r o (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

end VeriTile.Bench.TritonBenchG.FusedActivation

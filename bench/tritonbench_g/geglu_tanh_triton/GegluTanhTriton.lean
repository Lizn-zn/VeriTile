import VeriTile.Triton

/-!
# `geglu_tanh_triton` — strict per-kernel correctness

Two GeGLU kernels using the tanh approximation of GELU. `_geglu_tanh_forward_kernel`:
program `row` loads gate `a` and value `b` and stores
`0.5·a·(1 + tanh(√(2/π)·(a + 0.044715·a³)))·b` to `c`.
`_geglu_tanh_backward_kernel`: recomputes the activation from upstream gradient
`dc` and writes the two input gradients in place — `da` into `a`, `db` into `b`.
Both are masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_geglu_tanh_*_kernel[(n_rows,)](...)`,
the 1-D grid, the `calculate_settings` choice of `BLOCK_SIZE`/`num_warps`, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
each per-program statement covers every program of the grid.

## Proof architecture

```
geglu_tanh_forward_kernel_correctness           ← TOP THEOREM (forward:
  │                                               gegluTanhFwdIO ⊨ geluTanhFwd)
  ├─ geglu_tanh_forward_kernel_flattenOk        bridge fragment membership
  ├─ geglu_tanh_forward_kernel_traceSafe        per-execution lane-wise safety walk
  └─ geglu_tanh_forward_kernel_region_run       region-model masked Hoare triple
       ├─ geglu_tanh_forward_kernel_exec_isSome termination
       ├─ geglu_tanh_forward_kernel_correct     algorithm-layer readback per lane
       └─ geglu_tanh_forward_kernel_frame       masked scatter frame

geglu_tanh_backward_kernel_correctness          ← TOP THEOREM (backward:
  │                                               gegluTanhBwdIO ⊨ (geluTanhBwdA, geluTanhBwdB))
  ├─ geglu_tanh_backward_kernel_flattenOk       bridge fragment membership
  ├─ geglu_tanh_backward_kernel_traceSafe       per-execution lane-wise safety walk
  └─ geglu_tanh_backward_kernel_region_run      region-model masked Hoare triple
       ├─ geglu_tanh_backward_kernel_exec_isSome termination
       ├─ geglu_tanh_backward_kernel_correct    algorithm-layer readback per lane, per channel
       └─ geglu_tanh_backward_kernel_frame      two-store masked scatter frame
```

Each headline is stated on its kernel's masked **IO signature**
(`MaskedKernelIO₂` for the forward, `MaskedKernelIO₃ₓ₂` for the in-place
backward): which buffer is which argument, where program `pid` reads/writes
its `BLOCK_SIZE`-lane row window (`pid * stride`), and the active-lane
predicate `j < n_cols`. `⊨` is the audit-once masked Hoare-triple combinator:
for **every** disjoint placement of the declared buffers in flat memory,
**every** program id all of whose *active* lanes are in bounds (partial blocks
may overhang the buffer on their inactive lanes), and **every** launch state
whose input windows hold the given tiles at the active lanes, the translated
pointer kernel terminates, every active output lane holds the spec value, and
every other memory cell is unchanged.

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math.Activation`: the
tanh-GeGLU math (`TiledActivation.geluTanhFwd`, `geluTanhBwdA/B`, built on
`TiledActivation.geluTanhCore` / `geluTanhArg`) lives once in `Math.Activation`
and is reused here, so this file only checks that the kernels realize those
oracles lane-wise. The `from triton.language.extra.libdevice import tanh` is
represented by the DSL surface function `tanh`. Arithmetic is over `ℝ` (not
bit-accurate IEEE float); the `tl.program_id(0).to(tl.int64)` and `.to(tl.float32)`
casts reduce to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`); `calculate_settings` is not modeled. The float literals
(`0.7978845608028654`, `0.044715`) are interpreted as exact reals, not their
rounded float values. The backward kernel writes two channels (`da`→`a`,
`db`→`b`) and overwrites its own inputs in place; `MaskedKernelIO₃ₓ₂` models
this by decoupling the allocation list (`bufs`, each buffer exactly once) from
the argument roles (`out1 = in2 = a`, `out2 = in3 = b`), so the triple reads
the *old* window contents and asserts the *new* ones. Its headline assumes the
two output regions are distinct (`A ≠ B`), so the second store cannot clobber
the first channel.
-/

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python `from triton.language.extra.libdevice import tanh` is represented by
  the DSL surface function `tanh`. -/
def geglu_tanh_forward_kernel
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a += program_id * $(stride)
  b += program_id * $(stride)
  c += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_backward_kernel`.

The Python kernel overwrites `a` and `b` with `da` and `db`; the Lean port keeps
the same region arguments. -/
def geglu_tanh_backward_kernel
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc += program_id * $(stride)
  a += program_id * $(stride)
  b += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

def gegluTanhOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

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

/-! ## Forward kernel -/

/-- Algorithm-layer correctness for `_geglu_tanh_forward_kernel`. -/
theorem geglu_tanh_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i)
    (hExec : exec (geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhFwd (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  simp [exec, geglu_tanh_forward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [gegluTanhOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [gegluTanhOffset] at ha hb
    simp [hi, TiledActivation.geluTanhFwd, TiledActivation.geluTanhCore,
          TiledActivation.geluTanhArg, ha, hb]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `C`, and the *inactive* lanes of
the output row itself — is preserved by the run. -/
private theorem geglu_tanh_forward_kernel_frame
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((geglu_tanh_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(C = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, geglu_tanh_forward_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  rw [← Int.natCast_mul, Int.toNat_natCast]
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the forward kernel executes to completion from any state
(elementwise only — no reductions, so no `0 < BLOCK_SIZE` side condition). -/
private theorem geglu_tanh_forward_kernel_exec_isSome
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((geglu_tanh_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, geglu_tanh_forward_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only** (`j < n_cols`). This is
the `hrun` obligation of the forward `⊨` headline; the value half reuses
`geglu_tanh_forward_kernel_correct` (instantiated at the tiles the state
actually holds — the spec is elementwise, so lane-wise congruence is a
rewrite). -/
theorem geglu_tanh_forward_kernel_region_run
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK_SIZE → ℝ)
    (ha : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem A (s₀.pid * stride + j.val) = as j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem B (s₀.pid * stride + j.val) = bs j) :
    ∃ s1, exec ((geglu_tanh_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem C (s₀.pid * stride + j.val)
            = TiledActivation.geluTanhFwd (as j) (bs j))
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := geglu_tanh_forward_kernel_exec_isSome A B C
    stride n_cols BLOCK_SIZE s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := geglu_tanh_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
      s₀ s1
      (fun i => s₀.readMem A (gegluTanhOffset s₀ stride i))
      (fun i => s₀.readMem B (gegluTanhOffset s₀ stride i))
      (fun _ => rfl) (fun _ => rfl) hs1 j
    simp only [gegluTanhOffset, hj, if_pos] at h
    rw [show s₀.pid * stride + j.val = s₀.pids 0 * stride + j.val from rfl, h,
        show s₀.pids 0 * stride + j.val = s₀.pid * stride + j.val from rfl,
        ha j hj, hb j hj]
  · refine geglu_tanh_forward_kernel_frame A B C stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The forward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, the int64/float32 casts,
`tanh`, masked store). -/
theorem geglu_tanh_forward_kernel_flattenOk
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((geglu_tanh_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [geglu_tanh_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all sixteen
statements — thirteen are memory-silent (`program_id`, the pointer staging,
`arange`, the register arithmetic and `tanh`) — and reduces the three masked
accesses (`a` load, `b` load, `c` store, all at `pid * stride + j`, active iff
`j < n_cols`) to the **lane-wise** bounds hypotheses: every *active* lane's
address is below the region bound of the buffer it touches. -/
theorem geglu_tanh_forward_kernel_traceSafe
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hA : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hB : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B)
    (hC : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((geglu_tanh_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hA hB hC
  simp [geglu_tanh_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hA a ha, fun a ha => hB a ha, fun a ha => hC a ha⟩

/-- `_geglu_tanh_forward_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the forward `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (gate `a`, value `b`,
  output `c`);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` reads and writes its row at
  `pid * stride` in all three buffers (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes carry no
  obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def gegluTanhFwdIO (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := geglu_tanh_forward_kernel A B C stride n_cols BLOCK_SIZE
  in1 := A
  in2 := B
  out := C
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

/-- **The forward headline**: `_geglu_tanh_forward_kernel` implements the
tanh-GeGLU forward oracle on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `as`/`bs`,
the translated pointer kernel terminates, every active output-row lane `j`
holds `TiledActivation.geluTanhFwd (as j) (bs j)`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification geglu_tanh_forward_kernel_correctness
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    gegluTanhFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => TiledActivation.geluTanhFwd (as i) (bs i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact geglu_tanh_forward_kernel_flattenOk A B C stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact geglu_tanh_forward_kernel_traceSafe A B C stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := geglu_tanh_forward_kernel_region_run
      A B C stride n_cols BLOCK_SIZE s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## Backward kernel -/

/-- Algorithm-layer correctness for `_geglu_tanh_backward_kernel`.

The kernel stores `A` first and `B` second. We assume `A ≠ B` so the second
store cannot overwrite the first output channel. -/
theorem geglu_tanh_backward_kernel_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (gegluTanhOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (gegluTanhOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (gegluTanhOffset s stride i) = bs i)
    (hExec : exec (geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE) s = some s') :
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem A outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhBwdA (dcs i) (as i) (bs i)
        else s.readMem A outAddr) ∧
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := gegluTanhOffset s stride i
      s'.readMem B outAddr =
        if i.val < n_cols then
          TiledActivation.geluTanhBwdB (dcs i) (as i)
        else s.readMem B outAddr) := by
  simp [exec, geglu_tanh_backward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [gegluTanhOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_prop_masked_preserves_other_region
      (region := B) (R := A) (h_ne := hAB)
      (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
      (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      have hb := h_b i
      simp [gegluTanhOffset] at hdc ha hb
      norm_num [hi, TiledActivation.geluTanhBwdA, TiledActivation.geluTanhArg,
            hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [gegluTanhOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      simp [gegluTanhOffset] at hdc ha
      simp [hi, TiledActivation.geluTanhBwdB, TiledActivation.geluTanhCore,
            TiledActivation.geluTanhArg, hdc, ha]
    · rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := A) (R := B) (h_ne := Ne.symm hAB)
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
      simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by **either** masked
output store — every cell of every region other than `A`/`B`, and the
*inactive* lanes of the two output rows themselves — is preserved by the run. -/
private theorem geglu_tanh_backward_kernel_frame
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((geglu_tanh_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissA : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(A = r ∧ s.pid * stride + i.val = o))
    (hmissB : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(B = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, geglu_tanh_backward_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  rw [← Int.natCast_mul, Int.toNat_natCast]
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
  · intro k _ hmk hc
    exact hmissB k.1 (by simpa using hmk) hc
  · intro k _ hmk hc
    exact hmissA k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the backward kernel executes to completion from any state
(elementwise only — no reductions, so no `0 < BLOCK_SIZE` side condition). -/
private theorem geglu_tanh_backward_kernel_exec_isSome
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((geglu_tanh_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, geglu_tanh_backward_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane values
of both in-place outputs, and frame off the two active output windows, from
any launch state whose three input windows are loaded at the **active lanes
only** (`j < n_cols`). This is the `hrun` obligation of the backward `⊨`
headline; the value half reuses `geglu_tanh_backward_kernel_correct`
(instantiated at the tiles the state actually holds). Assumes the two output
regions are distinct (`A ≠ B`). -/
theorem geglu_tanh_backward_kernel_region_run
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B)
    (s₀ : BlockState) (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hdc : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem DC (s₀.pid * stride + j.val) = dcs j)
    (ha : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem A (s₀.pid * stride + j.val) = as j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem B (s₀.pid * stride + j.val) = bs j) :
    ∃ s1, exec ((geglu_tanh_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem A (s₀.pid * stride + j.val)
            = TiledActivation.geluTanhBwdA (dcs j) (as j) (bs j))
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem B (s₀.pid * stride + j.val)
            = TiledActivation.geluTanhBwdB (dcs j) (as j))
      ∧ (∀ r o,
          (r ≠ A ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          (r ≠ B ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := geglu_tanh_backward_kernel_exec_isSome DC A B
    stride n_cols BLOCK_SIZE s₀
  have h := geglu_tanh_backward_kernel_correct DC A B stride n_cols BLOCK_SIZE
    s₀ s1
    (fun i => s₀.readMem DC (gegluTanhOffset s₀ stride i))
    (fun i => s₀.readMem A (gegluTanhOffset s₀ stride i))
    (fun i => s₀.readMem B (gegluTanhOffset s₀ stride i))
    hAB (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun j hj => ?_, fun r o hcondA hcondB => ?_⟩
  · have hA := h.1 j
    simp only [gegluTanhOffset, hj, if_pos] at hA
    rw [show s₀.pid * stride + j.val = s₀.pids 0 * stride + j.val from rfl, hA,
        show s₀.pids 0 * stride + j.val = s₀.pid * stride + j.val from rfl,
        hdc j hj, ha j hj, hb j hj]
  · have hB := h.2 j
    simp only [gegluTanhOffset, hj, if_pos] at hB
    rw [show s₀.pid * stride + j.val = s₀.pids 0 * stride + j.val from rfl, hB,
        show s₀.pids 0 * stride + j.val = s₀.pid * stride + j.val from rfl,
        hdc j hj, ha j hj]
  · refine geglu_tanh_backward_kernel_frame DC A B stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_) (fun i hi ⟨hr, ho⟩ => ?_)
    · rcases hcondA with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rcases hcondB with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm

/-- The backward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, the int64/float32 casts,
`tanh`, two masked stores). -/
theorem geglu_tanh_backward_kernel_flattenOk
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((geglu_tanh_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [geglu_tanh_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all
twenty-two statements — seventeen are memory-silent — and reduces the five
masked accesses (`dc`/`a`/`b` loads, `a`/`b` stores, all at
`pid * stride + j`, active iff `j < n_cols`) to the **lane-wise** bounds
hypotheses: every *active* lane's address is below the region bound of the
buffer it touches (the load and store obligations of `a`/`b` coincide, so
three hypotheses suffice). -/
theorem geglu_tanh_backward_kernel_traceSafe
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hDC : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds DC)
    (hA : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hB : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B) :
    Kernel.TraceSafe bounds
      ((geglu_tanh_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hDC hA hB
  simp [geglu_tanh_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hDC a ha, fun a ha => hA a ha, fun a ha => hB a ha,
    fun a ha => hA a ha, fun a ha => hB a ha⟩

/-- `_geglu_tanh_backward_kernel`'s masked in-place **IO signature** — the
whole kernel-specific audit surface of the backward `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — upstream gradient `dc`, gate `a`, value `b` (the
  wiring);
* `out1 = in2`, `out2 = in3` — the **in-place** roles: the kernel rewrites
  the gate and value buffers it read (`da`→`a`, `db`→`b`);
* `read1..3`/`write1..2` — every window is the same row `pid * stride` (the
  host-side one-program-per-row launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**.
  Inactive lanes carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
def gegluTanhBwdIO (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := geglu_tanh_backward_kernel DC A B stride n_cols BLOCK_SIZE
  bufs := [DC, A, B]  -- a and b are updated in place
  in1 := DC
  in2 := A
  in3 := B
  out1 := A   -- = in2: in-place `da` into `a`
  out2 := B   -- = in3: in-place `db` into `b`
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  read3 := fun pid => pid * stride
  write1 := fun pid => pid * stride
  write2 := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

/-- **The backward headline**: `_geglu_tanh_backward_kernel` implements the
tanh-GeGLU backward oracles on its masked in-place IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input-row lanes hold
`dcs`/`as`/`bs`, the translated pointer kernel terminates, every active lane
of the gate buffer ends up holding `geluTanhBwdA` and of the value buffer
`geluTanhBwdB` — applied to the *originally loaded* windows (the standard
before/after reading of an in-place Hoare triple) — and every other memory
cell is unchanged. Assumes the two output regions are distinct (`A ≠ B`) so
the second store cannot clobber the first channel. Proof:
`MaskedKernelIO₃ₓ₂.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification geglu_tanh_backward_kernel_correctness
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B) :
    gegluTanhBwdIO DC A B stride n_cols BLOCK_SIZE ⊨ fun dcs as bs =>
      (fun i => TiledActivation.geluTanhBwdA (dcs i) (as i) (bs i),
       fun i => TiledActivation.geluTanhBwdB (dcs i) (as i)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [gegluTanhBwdIO]) (by simp [gegluTanhBwdIO]) ?_ ?_ ?_
  · exact geglu_tanh_backward_kernel_flattenOk DC A B stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _ _
    exact geglu_tanh_backward_kernel_traceSafe DC A B stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ dcs as bs hdc ha hb
    exact geglu_tanh_backward_kernel_region_run DC A B stride n_cols BLOCK_SIZE
      hAB s₀ dcs as bs hdc ha hb

end VeriTile.Bench.TritonBenchG.GegluTanhTriton

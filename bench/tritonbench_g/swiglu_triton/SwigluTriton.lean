import VeriTile.Triton

/-!
# `swiglu_triton` — strict per-kernel correctness

Two SwiGLU kernels. `_swiglu_forward_kernel`: program `row` loads gate `a` and
value `b` for its row and stores `silu(a) · b` to `c`. `_swiglu_backward_kernel`:
recomputes `silu(a)` from the loaded gate `a`, then combines it with the upstream
gradient `dc` and writes the two input gradients in place — `da` into `a_ptr`,
`db` into `b_ptr`. Both are masked by
`col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_swiglu_*_kernel[(n_rows,)](...)`, the
1-D grid, the `calculate_settings` choice of `BLOCK_SIZE`/`num_warps`, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
each per-program statement covers every program of the grid.

## Proof architecture

```
swiglu_forward_kernel_correctness             ← TOP THEOREM (swigluFwdIO ⊨ swiglu)
  ├─ swiglu_forward_kernel_flattenOk          bridge fragment membership
  ├─ swiglu_forward_kernel_traceSafe          per-execution lane-wise safety walk
  └─ swiglu_forward_kernel_region_run         region-model masked Hoare triple
       ├─ swiglu_forward_kernel_correct       algorithm-layer readback per lane
       └─ swiglu_forward_kernel_frame         masked scatter cell frame

swiglu_backward_kernel_correctness            ← TOP THEOREM (swigluBwdIO ⊨ (swigluBwdA, swigluBwdB))
  ├─ swiglu_backward_kernel_flattenOk         bridge fragment membership
  ├─ swiglu_backward_kernel_traceSafe         per-execution lane-wise safety walk
  └─ swiglu_backward_kernel_region_run        region-model masked in-place triple
       ├─ swiglu_backward_kernel_correct      algorithm-layer readback per lane, per channel
       └─ swiglu_backward_kernel_frame        two-scatter cell frame
```

Both headlines are stated on the kernels' masked **IO signatures**
(`MaskedKernelIO₂` for the forward, `MaskedKernelIO₃ₓ₂` for the in-place
backward): which buffer is which argument, where program `pid` reads/writes its
`BLOCK_SIZE`-lane row window (`pid * stride`), and the active-lane predicate
`j < n_cols`. `⊨` is the audit-once masked Hoare-triple combinator: for
**every** disjoint placement of the buffers in flat memory, **every** program
id all of whose *active* lanes are in bounds, and **every** launch state whose
input windows hold the input tiles at the active lanes, the translated pointer
kernel terminates, every active output lane holds the SwiGLU oracle value, and
every other memory cell is unchanged.

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swiglu`, `TiledActivation.swigluBwdA/B`, built on
`TiledActivation.silu`) lives once in `Math.Activation` and is reused here, so
this file only checks that the kernels realize those oracles lane-wise.
Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `tl.program_id(0)
.to(tl.int64)` cast and `.to(tl.float32)` load casts reduce to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`); `calculate_settings`
is not modeled. The backward kernel writes two channels (`da`→`a_ptr`,
`db`→`b_ptr`) and overwrites its own inputs in place; `MaskedKernelIO₃ₓ₂`
expresses this by decoupling the allocation list (`bufs`, each buffer exactly
once) from the argument roles (`out1 = in2 = a_ptr`, `out2 = in3 = b_ptr`), so
the triple reads the *old* window contents and asserts the *new* ones. Side
condition: `A ≠ B` (the two output buffers do not alias — the second masked
store would otherwise clobber the first output; no kernel caller aliases them).
-/

namespace VeriTile.Bench.TritonBenchG.SwigluTriton

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters. -/
def swiglu_forward_kernel
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  c_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_backward_kernel`.

Allowed mechanical Lean-syntax-only changes match `swiglu_forward_kernel`. -/
def swiglu_backward_kernel
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += program_id * $(stride)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

/-! ## Forward kernel -/

/-- Algorithm-layer correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          TiledActivation.swiglu (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  simp [exec, swiglu_forward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  simp only [swigluOffset]
  rw [← Int.natCast_mul, Int.toNat_natCast]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [swigluOffset] at ha hb
    simp [hi, TiledActivation.swiglu, TiledActivation.silu, ha, hb]
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
store — every cell of every region other than `C`, and the *inactive* lanes of
the output row itself — is preserved by the run. -/
private theorem swiglu_forward_kernel_frame
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(C = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, swiglu_forward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  rw [← Int.natCast_mul, Int.toNat_natCast] at hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the forward kernel executes to completion from any state (all
statements are elementwise, so no `0 < BLOCK_SIZE` side condition is needed). -/
private theorem swiglu_forward_kernel_exec_isSome
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, swiglu_forward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input rows are loaded at the **active lanes only** (`j < n_cols`). This is the
`hrun` obligation of the `⊨` headline; the value half reuses
`swiglu_forward_kernel_correct` (instantiated at the tiles the state actually
holds). -/
theorem swiglu_forward_kernel_region_run
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK_SIZE → ℝ)
    (ha : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem A (s₀.pid * stride + j.val) = as j)
    (hb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem B (s₀.pid * stride + j.val) = bs j) :
    ∃ s1, exec ((swiglu_forward_kernel A B C stride n_cols
          BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem C (s₀.pid * stride + j.val)
            = TiledActivation.swiglu (as j) (bs j))
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_forward_kernel_exec_isSome A B C stride n_cols
    BLOCK_SIZE s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := swiglu_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
      s₀ s1
      (fun i => s₀.readMem A (swigluOffset s₀ stride i))
      (fun i => s₀.readMem B (swigluOffset s₀ stride i))
      (fun _ => rfl) (fun _ => rfl) hs1 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, ha j hj, hb j hj]
  · refine swiglu_forward_kernel_frame A B C stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The forward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, `.to(...)` casts, elementwise
math, masked store). -/
theorem swiglu_forward_kernel_flattenOk
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same row window `pid * stride + j`, active only when `j < n_cols`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem swiglu_forward_kernel_traceSafe
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hina : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hinb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((swiglu_forward_kernel A B C stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hina hinb hout
  simp [swiglu_forward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
    NumericDType.mul, ComparableDType.lt]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hina a ha, fun a ha => hinb a ha, fun a ha => hout a ha⟩

/-- `_swiglu_forward_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: gate `A`,
  value `B`, output `C`);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row at
  `pid * stride` in all three buffers (the host-side one-program-per-row launch
  convention `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def swigluFwdIO (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE
  in1 := A
  in2 := B
  out := C
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

/-- **The headline (forward)**: `_swiglu_forward_kernel` implements the SwiGLU
oracle `TiledActivation.swiglu` (i.e. `silu(a) · b`) lane-wise on its masked IO
signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
active input-row lanes hold `as`/`bs`, the translated pointer kernel
terminates, every active output-row lane `j` holds
`TiledActivation.swiglu (as j) (bs j)`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
specification swiglu_forward_kernel_correctness
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    swigluFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => TiledActivation.swiglu (as i) (bs i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact swiglu_forward_kernel_flattenOk A B C stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact swiglu_forward_kernel_traceSafe A B C stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := swiglu_forward_kernel_region_run
      A B C stride n_cols BLOCK_SIZE s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## Backward kernel -/

/-- Algorithm-layer correctness for `_swiglu_backward_kernel`.

The two stores write `A` first and `B` second. We assume `A ≠ B` so the second
store cannot overwrite the first output channel. -/
theorem swiglu_backward_kernel_correct
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (dcs as bs : Fin BLOCK_SIZE → ℝ)
    (hAB : A ≠ B)
    (h_dc : ∀ i : Fin BLOCK_SIZE, s.readMem DC (swigluOffset s stride i) = dcs i)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE) s = some s') :
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem A outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdA (dcs i) (as i) (bs i)
        else s.readMem A outAddr) ∧
    (∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem B outAddr =
        if i.val < n_cols then
          TiledActivation.swigluBwdB (dcs i) (as i)
        else s.readMem B outAddr) := by
  simp [exec, swiglu_backward_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst s'
  constructor
  · intro i
    simp only [swigluOffset]
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
      simp [swigluOffset] at hdc ha hb
      simp [hi, TiledActivation.swigluBwdA, TiledActivation.silu, hdc, ha, hb]
    · simp [hi]
  · intro i
    simp only [swigluOffset]
    rw [← Int.natCast_mul, Int.toNat_natCast]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · have hdc := h_dc i
      have ha := h_a i
      simp [swigluOffset] at hdc ha
      simp [hi, TiledActivation.swigluBwdB, TiledActivation.silu, hdc, ha]
    · rw [BlockState.scatter_prop_masked_preserves_other_region
        (region := A) (R := B) (h_ne := Ne.symm hAB)
        (P := fun idx : TileIndex [BLOCK_SIZE] => idx.1.val < n_cols)
        (off := s.pids 0 * stride + i.val) (l := TileShape.allIndices [BLOCK_SIZE])]
      simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell outside **both** active output windows —
every cell of every region other than `A`/`B`, and the *inactive* lanes of the
two output rows themselves — is preserved by the run. -/
private theorem swiglu_backward_kernel_frame
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissA : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(A = r ∧ s.pid * stride + i.val = o))
    (hmissB : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(B = r ∧ s.pid * stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, swiglu_backward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl)
  · intro k _ hmk hc
    rw [← Int.natCast_mul, Int.toNat_natCast] at hc
    exact hmissB k.1 (by simpa using hmk) hc
  · intro k _ hmk hc
    rw [← Int.natCast_mul, Int.toNat_natCast] at hc
    exact hmissA k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the backward kernel executes to completion from any state (all
statements are elementwise, so no `0 < BLOCK_SIZE` side condition is needed). -/
private theorem swiglu_backward_kernel_exec_isSome
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, swiglu_backward_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, tile_elementwise,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
active-lane values of both in-place outputs, and frame off the two active
output windows, from any launch state whose three input rows are loaded at the
**active lanes only** (`j < n_cols`). This is the `hrun` obligation of the `⊨`
headline; the value half reuses `swiglu_backward_kernel_correct` (instantiated
at the tiles the state actually holds). -/
theorem swiglu_backward_kernel_region_run
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
    ∃ s1, exec ((swiglu_backward_kernel DC A B stride n_cols
          BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem A (s₀.pid * stride + j.val)
            = TiledActivation.swigluBwdA (dcs j) (as j) (bs j))
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem B (s₀.pid * stride + j.val)
            = TiledActivation.swigluBwdB (dcs j) (as j))
      ∧ (∀ r o,
          (r ≠ A ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          (r ≠ B ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := swiglu_backward_kernel_exec_isSome DC A B stride n_cols
    BLOCK_SIZE s₀
  have hcorr := swiglu_backward_kernel_correct DC A B stride n_cols BLOCK_SIZE
    s₀ s1
    (fun i => s₀.readMem DC (swigluOffset s₀ stride i))
    (fun i => s₀.readMem A (swigluOffset s₀ stride i))
    (fun i => s₀.readMem B (swigluOffset s₀ stride i))
    hAB (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun j hj => ?_, fun r o h1 h2 => ?_⟩
  · -- `A` window: the in-place gate gradient realizes `swigluBwdA`.
    have h := hcorr.1 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, hdc j hj, ha j hj, hb j hj]
  · -- `B` window: the in-place value gradient realizes `swigluBwdB`.
    have h := hcorr.2 j
    simp only [hj, if_pos] at h
    rw [show swigluOffset s₀ stride j = s₀.pid * stride + j.val from rfl] at h
    rw [h, hdc j hj, ha j hj]
  · -- frame off the two active output windows.
    refine swiglu_backward_kernel_frame DC A B stride n_cols BLOCK_SIZE
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_) (fun i hi ⟨hr, ho⟩ => ?_)
    · rcases h1 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm
    · rcases h2 with hne | hno
      · exact hne hr.symm
      · exact hno i hi ho.symm

/-- The backward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked loads with `other`, `.to(...)` casts, elementwise
math, two masked stores). -/
theorem swiglu_backward_kernel_flattenOk
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: all three masked loads and both masked stores
address the same row window `pid * stride + j`, active only when `j < n_cols`,
so the bounds contract is **lane-wise** — every *active* lane's address is
below the region bound of the buffer it touches. -/
theorem swiglu_backward_kernel_traceSafe
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hdc : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds DC)
    (hina : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds A)
    (hinb : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * stride + j.val < bounds B) :
    Kernel.TraceSafe bounds
      ((swiglu_backward_kernel DC A B stride n_cols
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hdc hina hinb
  simp [swiglu_backward_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, tile_elementwise,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    ComparableDType.lt]
  simp only [← Int.natCast_mul, Int.toNat_natCast]
  exact ⟨fun a ha => hdc a ha, fun a ha => hina a ha, fun a ha => hinb a ha,
    fun a ha => hina a ha, fun a ha => hinb a ha⟩

/-- `_swiglu_backward_kernel`'s masked in-place **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — upstream gradient `DC`, gate `A`, value `B` (the wiring);
* `out1 = in2`, `out2 = in3` — the **in-place** roles: the kernel overwrites
  the gate and value buffers it read with `da` and `db`;
* `read1..3`/`write1..2` — every window is the same row `pid * stride` (the
  host-side one-program-per-row launch convention
  `x_ptr += program_id * stride`);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def swigluBwdIO (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ where
  kernel := swiglu_backward_kernel DC A B stride n_cols BLOCK_SIZE
  bufs := [DC, A, B]  -- A and B are updated in place
  in1 := DC
  in2 := A
  in3 := B
  out1 := A    -- = in2: `da` overwrites the gate input in place
  out2 := B    -- = in3: `db` overwrites the value input in place
  B := BLOCK_SIZE
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  read3 := fun pid => pid * stride
  write1 := fun pid => pid * stride
  write2 := fun pid => pid * stride
  mask := fun _ j => j.val < n_cols

/-- **The headline (backward)**: `_swiglu_backward_kernel` implements the
SwiGLU backward oracles on its masked in-place IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input-row lanes hold
`dcs`/`as`/`bs`, the translated pointer kernel terminates, every active lane
of the gate buffer ends up holding `swigluBwdA` and of the value buffer
`swigluBwdB`, applied to the *originally loaded* windows; every other flat
cell is untouched. The side condition `A ≠ B` rules out aliasing between the
two output buffers (the second masked store would otherwise clobber the first
output). Proof: `MaskedKernelIO₃ₓ₂.Implements.intro` assembles the
region-model triple with the flat-memory bridge side conditions. -/
specification swiglu_backward_kernel_correctness
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B) :
    swigluBwdIO DC A B stride n_cols BLOCK_SIZE
      ⊨ fun dcs as bs =>
        (fun i => TiledActivation.swigluBwdA (dcs i) (as i) (bs i),
         fun i => TiledActivation.swigluBwdB (dcs i) (as i)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [swigluBwdIO]) (by simp [swigluBwdIO]) ?_ ?_ ?_
  · exact swiglu_backward_kernel_flattenOk DC A B stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _ _
    exact swiglu_backward_kernel_traceSafe DC A B stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ dcs as bs hdc ha hb
    exact swiglu_backward_kernel_region_run DC A B stride n_cols BLOCK_SIZE
      hAB s₀ dcs as bs hdc ha hb

end VeriTile.Bench.TritonBenchG.SwigluTriton

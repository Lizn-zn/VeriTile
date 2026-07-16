import VeriTile.Triton

/-!
# `dequantize_rowwise` — strict per-kernel correctness

`_dequantize_rowwise` dequantizes one row per program: program `pid` loads its
row from `x_ptr` (a `P2`-wide power-of-two arange masked by `arange < BLOCK_SIZE`),
loads the per-row max `state_x[pid]`, and stores `max_val * x * inv_127` to
`output_ptr`, masked by the same row mask.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_dequantize_rowwise[grid](...)`, the grid `(x.shape[0],)`
of one program per row, the host-side `P2 = 2^ceil(log2(cols))` and `BLOCK_SIZE
= cols` choices, the `inv_127 = 1/127` constant, and the runtime composition of
per-program writes) is the *trusted boundary*. `pid` is universally quantified,
so the per-program statement covers every row of the grid.

## Proof architecture

```
dequantize_rowwise_correctness                 ← TOP THEOREM (dequantizeRowwiseIO ⊨ rowwise dequantize)
  ├─ dequantize_rowwise_kernel_flattenOk       bridge fragment membership
  ├─ dequantize_rowwise_kernel_traceSafe       per-execution lane-wise safety walk
  └─ dequantize_rowwise_kernel_region_run      region-model masked Hoare triple
       ├─ dequantize_rowwise_kernel_correct    algorithm-layer readback per lane
       └─ dequantize_rowwise_kernel_frame      masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature**
`dequantizeRowwiseIO` (`Masked2DKernelIO₂`, the general-window family: the
quantized row is a contiguous `P2`-lane window of `x_ptr`, while the per-row
scale is the **single cell** `state_x[pid]` — every active lane's `read2`
address is that one scalar cell, which the 1D contiguous-window family cannot
express). The kernel's grid is 1D; the signature simply never mentions the
second program-id axis. `⊨` is the audit-once masked Hoare-triple combinator
(`Masked2DKernelIO₂.Implements`): for **every** disjoint placement of the
three buffers in flat memory, **every** program id whose *active* lanes are in
bounds (for the scale that is just `pid < extent state_x`), and **every**
launch state whose input windows hold `xs`/`ys` at the active lanes, the
translated pointer kernel terminates, every active output lane holds
`ys i * xs i * inv_127`, and every other memory cell is unchanged.

The active mask is `arange < BLOCK_SIZE` (the kernel's `n_elements` argument
is unused by the body and kept as `_n_elements`). The headline carries two
honest side conditions, `0 < BLOCK_SIZE` and `0 < P2`: the scale load
`tl.load(state_x + pid)` is **unmasked**, so its safety needs
`pid < extent state_x` even for a launch with no active lanes — the lane-wise
contract only supplies that bound when an active lane exists. Both hold for
every real launch (`BLOCK_SIZE = cols ≥ 1`, `P2 = 2^⌈log₂ cols⌉ ≥ 1`).

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The quantization semantics
here are the **dequantization direction only**: `x` is the loaded int8 value
taken as a real number and the output is the exact real product `max_val * x *
inv_127`. There is no rounding or int cast on the output side, so this kernel has
no `to(tl.int8)` / `llrint` blocker and the full surface lowers through algorithm
erasure. The output dtype is `float16` upstream, modeled as `ℝ` (post-erasure all
dtypes unify). `@triton.autotune` is not present here.
-/

namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}

/-- Exact dequantized value written at active lane `i`. -/
noncomputable def dequantizeRowwiseSpec
    (s : BlockState) (x_ptr state_x : RegionName)
    (BLOCK_SIZE : Nat) (inv_127 : ℝ) (i : Fin P2) : ℝ :=
  s.readMem state_x s.pid *
    s.readMem x_ptr (s.pid * BLOCK_SIZE + i.val) * inv_127

/-- Algorithm-layer correctness for `_dequantize_rowwise`.

The upstream kernel's active mask is `arange < BLOCK_SIZE`; `n_elements` is not
used by the kernel body. -/
theorem dequantize_rowwise_kernel_correct
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s : BlockState) :
    ∀ i : Fin P2,
      let outAddr := s.pid * BLOCK_SIZE + i.val
      (exec (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
            n_elements BLOCK_SIZE P2) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < BLOCK_SIZE then
            dequantizeRowwiseSpec s x_ptr state_x BLOCK_SIZE inv_127 i
          else s.readMem output_ptr outAddr) := by
  intro i
  simp [exec, dequantize_rowwise_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < BLOCK_SIZE
  · simp [hi, dequantizeRowwiseSpec, BlockState.pid_eq, mul_assoc]
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
store is preserved by the run — in particular every cell of every region other
than `output_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem dequantize_rowwise_kernel_frame
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) (s s1 : BlockState)
    (hExec : exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin P2, i.val < BLOCK_SIZE →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose row
window holds `xs` and whose scale cell holds `ys` at the active lanes. This is
the `hrun` obligation of `Masked2DKernelIO₂.Implements.intro`; the value half
reuses `dequantize_rowwise_kernel_correct` (instantiated at the values the
state actually holds). -/
theorem dequantize_rowwise_kernel_region_run
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (s₀ : BlockState) (xs ys : Fin P2 → ℝ)
    (hx : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s₀.readMem state_x s₀.pid = ys j) :
    ∃ s1, exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
        n_elements BLOCK_SIZE P2).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin P2, j.val < BLOCK_SIZE →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = ys j * xs j * inv_127)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin P2, j.val < BLOCK_SIZE →
            o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := dequantize_rowwise_kernel_correct x_ptr state_x output_ptr
    inv_127 n_elements BLOCK_SIZE P2 s₀
  rw [show exec (dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2) s₀
      = exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
          n_elements BLOCK_SIZE P2).toAlgKernel) s₀ from rfl] at hobs
  cases hsrc : exec ((dequantize_rowwise_kernel x_ptr state_x output_ptr
      inv_127 n_elements BLOCK_SIZE P2).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
          stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [Option.map_some, Option.some_inj, if_pos hj] at hje
        rw [hje]
        simp only [dequantizeRowwiseSpec]
        rw [hx j hj, hy j hj]
      · refine dequantize_rowwise_kernel_frame x_ptr state_x output_ptr
          inv_127 n_elements BLOCK_SIZE P2 s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked row load and the masked store
address the window `pid * BLOCK_SIZE + j`, active only when
`j < BLOCK_SIZE`; the **unmasked** scale load reads the single cell
`state_x[pid]`, whose bound is taken as the explicit hypothesis `hscale`. -/
theorem dequantize_rowwise_kernel_traceSafe
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hscale : s.pid < bounds state_x)
    (hout : ∀ j : Fin P2, j.val < BLOCK_SIZE →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
        n_elements BLOCK_SIZE P2).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all nine statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, hscale, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem dequantize_rowwise_kernel_flattenOk
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) :
    ((dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
      n_elements BLOCK_SIZE P2).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [dequantize_rowwise_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `_dequantize_rowwise`'s masked **IO signature** — the whole
kernel-specific audit surface of the headline: which buffer is which
argument (the wiring), where program `pid` reads its `P2`-lane quantized row
(`x_ptr[pid·BLOCK_SIZE + j]`), where it reads its per-row scale — the
**single scalar cell** `state_x[pid]`, the same address at every active lane
— where it writes its output row, and the active-lane predicate
`j < BLOCK_SIZE`. The kernel's grid is 1D, so no field mentions the second
program-id axis. The windows and mask are declared, not parsed from the
kernel: they formalize the host-side launch convention (`offsets =
pid * BLOCK_SIZE + arange; row_mask = arange < BLOCK_SIZE`, one scale per
row), and the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def dequantizeRowwiseIO (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) : Masked2DKernelIO₂ where
  kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
    n_elements BLOCK_SIZE P2
  in1 := x_ptr
  in2 := state_x
  out := output_ptr
  B := P2
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun pid _ _ => pid
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun _ _ j => j.val < BLOCK_SIZE

/-- **The headline**: `_dequantize_rowwise` implements per-row dequantization
on its masked IO signature — for every disjoint flat placement of the three
buffers, every program id whose active lanes are in bounds, and every launch
state whose row window holds `xs` and whose scale cell holds `ys` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `ys i * xs i * inv_127` (the scale times the quantized value times
`1/127`; `ys` is constant across the row's active lanes since every lane
reads the same cell), and every other memory cell is unchanged.

`0 < BLOCK_SIZE` / `0 < P2` are genuinely forced: the scale load is unmasked,
so a launch with no active lanes still reads `state_x[pid]`, whose bound the
lane-wise contract only supplies via an active lane. Both hold for every real
launch. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification dequantize_rowwise_correctness
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (hBS : 0 < BLOCK_SIZE) (hP2 : 0 < P2) :
    dequantizeRowwiseIO x_ptr state_x output_ptr inv_127 n_elements
      BLOCK_SIZE P2 ⊨ fun _ _ xs ys i => ys i * xs i * inv_127 := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact dequantize_rowwise_kernel_flattenOk x_ptr state_x output_ptr
      inv_127 n_elements BLOCK_SIZE P2
  · intro bounds s h1 h2 h3 _
    exact dequantize_rowwise_kernel_traceSafe x_ptr state_x output_ptr
      inv_127 n_elements BLOCK_SIZE P2 bounds s h1 (h2 ⟨0, hP2⟩ hBS) h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := dequantize_rowwise_kernel_region_run
      x_ptr state_x output_ptr inv_127 n_elements BLOCK_SIZE P2 s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.DequantizeRowwise

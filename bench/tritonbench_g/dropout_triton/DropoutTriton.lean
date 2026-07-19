import VeriTile.Triton

/-!
# `dropout_triton` — strict per-kernel correctness

`_dropout` is inverted dropout with a *given* keep-mask: program `pid` loads
block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of the input `x_ptr` and the keep
buffer `x_keep_ptr`, computes `tl.where(x_keep, x / (1 - p), 0.0)` lane-wise,
and stores to `output_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_dropout[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

The headline is stated on the kernel's masked bool-input IO signature
`dropoutIO` (`BoolMasked2DKernelIO₁`): which buffer is which argument, where
program `pid` reads its ℝ tile / its `Bool` keep tile / writes its output
tile, and the active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. `⊨`
is the audit-once masked Hoare-triple combinator
(`BoolMasked2DKernelIO₁.Implements`): for **every** disjoint placement of the
three buffers in flat memory, **every** program id all of whose *active*
lanes are in bounds, and **every** launch state whose input windows hold the
data tile `xs` (ℝ channel) and the keep tile `bs` (`.bool` channel) at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `if bs i then xs i / (1 - p) else 0`, and every other memory cell
is unchanged. The kernel's grid is 1D, so the signature's second program-id
axis is quantified but unused. The store is gated by the *static* load mask
only — the keep bit enters the stored **value**, not the write set — so
`dropoutIO` keeps the struct's default `writeMask` (= the static `mask`).

## Proof architecture

```
dropout_kernel_correctness                    ← TOP SPECIFICATION (dropoutIO ⊨ keep-gated scale)
  ├─ dropout_kernel_flattenOk                 bridge fragment membership
  ├─ dropout_kernel_traceSafe                 per-execution lane-wise safety walk
  └─ dropout_kernel_region_run                region-model masked Hoare triple
       ├─ dropout_kernel_correct              ← algorithm-layer readback per lane
       └─ dropout_kernel_frame                masked scatter-store cell frame
```

The state-coupled readback spec `dropoutSpec` (reads the keep-mask and input
*from memory* at this lane's offset) is internal to the readback lemma; the
public surface is the pure per-lane spec of the `⊨` headline.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The RNG is **not** modeled here: the upstream `_dropout` takes the
keep-mask `x_keep_ptr` as an explicit input buffer (the random keep/drop
decision is made by the caller), so the per-lane keep bit is read from memory
via `s.readMemValue .bool x_keep_ptr ...` and treated as given — the theorems
assume nothing about how it was produced. `x_keep_ptr` is a typed boolean
region. No output/input disjointness is assumed at the region layer: the
inputs are read into registers before the scatter, so the result is correct
even if `output_ptr` aliases `x_ptr` or `x_keep_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.DropoutTriton

open VeriTile.Triton
open scoped VeriTile.Triton.BoolMasked2DKernelIO₁

/-- Faithful transcription of `dropout_triton.py`'s `_dropout`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `x_keep_ptr` is a typed Lean boolean region so its `tl.load` call does not
  need an extra `dtype=` kwarg. -/
def dropout_kernel
    (x_ptr : RegionName) (x_keep_ptr : Region .bool) (output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Lane `i`'s address in all three windows of program `s.pid`. Internal to
the readback lemma `dropout_kernel_correct` (the `⊨` signature declares the
same window as `fun pid _ j => pid * BLOCK_SIZE + j.val`). -/
def dropoutOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- State-coupled readback spec, internal to `dropout_kernel_correct`: reads
the keep-mask and input *from memory* at this lane's offset and returns
`x / (1 - p)` when the mask is set, else `0`. The `⊨` headline re-expresses
it as a pure function of the loaded tiles. -/
noncomputable def dropoutSpec
    (s : BlockState) (x_ptr x_keep_ptr : RegionName)
    (p : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  if s.readMemValue .bool x_keep_ptr (dropoutOffset s BLOCK_SIZE i) then
    s.readMem x_ptr (dropoutOffset s BLOCK_SIZE i) / (1 - p)
  else
    0.0

/-- Algorithm-layer correctness for `_dropout`.

Active lanes write the scaled-or-zero dropout result; inactive tail lanes are
preserved. -/
theorem dropout_kernel_correct
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (dropout_kernel x_ptr x_keep_ptr output_ptr
          n_elements p BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := dropoutOffset s BLOCK_SIZE i
      s'.readMem output_ptr outAddr =
        if outAddr < n_elements then
          dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i
        else s.readMem output_ptr outAddr := by
  intro i
  simp [exec, dropout_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt] at hExec
  subst s'
  simp only [dropoutOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hBounds, dropoutSpec, dropoutOffset]
    cases hKeep : s.readMemValue .bool x_keep_ptr (s.pid * BLOCK_SIZE + i.val) <;>
      simp
  · simp [hBounds]

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
private theorem dropout_kernel_frame
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((dropout_kernel x_ptr x_keep_ptr output_ptr
      n_elements p BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, dropout_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows hold the data tile `xs` (ℝ channel) and the keep tile `bs`
(`.bool` channel) at the **active lanes only**. This is the `hrun` obligation
of `BoolMasked2DKernelIO₁.Implements.intro`; the value half reuses
`dropout_kernel_correct` (whose state-coupled `dropoutSpec` collapses to the
pure keep-gated scale under the two input hypotheses). -/
theorem dropout_kernel_region_run
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s₀ : BlockState) (bs : Fin BLOCK_SIZE → Bool) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j)
    (hb : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMemValue .bool x_keep_ptr (s₀.pid * BLOCK_SIZE + j.val) = bs j) :
    ∃ s1, exec ((dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val)
            = if bs j then xs j / (1 - p) else 0)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  cases hsrc : exec ((dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, dropout_kernel, ComputeKernel.toAlgKernel, stepStmts,
          stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          NumericDType.div, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have h := dropout_kernel_correct x_ptr x_keep_ptr output_ptr
          n_elements p BLOCK_SIZE s₀ s1 hsrc j
        simp only [dropoutOffset, dropoutSpec, hj, if_true, hx j hj, hb j hj]
          at h
        rw [h]
        cases bs j <;> norm_num
      · refine dropout_kernel_frame x_ptr x_keep_ptr output_ptr n_elements p
          BLOCK_SIZE s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads (the ℝ data tile and the
`.bool` keep tile) and the masked store address the same window
`pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the bounds
contract is **lane-wise** — every *active* lane's address is below the region
bound of the buffer it touches. -/
theorem dropout_kernel_traceSafe
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hb : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds x_keep_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all eight statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [dropout_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hb a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked ℝ and `.bool` loads, `where`, comparison, masked store). -/
theorem dropout_kernel_flattenOk
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ((dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p
      BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [dropout_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `_dropout`'s masked bool-input **IO signature** — the whole
kernel-specific audit surface of the headline: which buffer is which argument
(the wiring: `inp` the ℝ data, `mbuf` the `.bool` keep-mask, `out` the
output), where program `pid` reads/writes its `BLOCK_SIZE`-lane window (all
three at `pid * BLOCK_SIZE + j`), and the active-lane predicate
`pid * BLOCK_SIZE + j < n_elements`. The grid is 1D, so the second program-id
axis is unused. The store is masked by the same *static* load mask — the keep
bit gates the stored **value**, not the write set — so `writeMask` keeps the
struct's default (= `mask`). The windows and mask are declared, not parsed
from the kernel: they formalize the host-side launch convention
(`offsets = pid * BLOCK_SIZE + arange; mask = offsets < n_elements`), and the
headline **proves** the kernel's actual addressing and masking match them.
Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
def dropoutIO (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) : BoolMasked2DKernelIO₁ where
  kernel := dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p BLOCK_SIZE
  inp := x_ptr
  mbuf := x_keep_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid _ j => pid * BLOCK_SIZE + j.val
  readm := fun pid _ j => pid * BLOCK_SIZE + j.val
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun pid _ j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `_dropout` implements lane-wise keep-gated inverted
dropout on its masked bool-input IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose input windows hold the data tile `xs`
and the keep tile `bs` at the active lanes, the translated pointer kernel
terminates, every active output lane holds `if bs i then xs i / (1 - p)
else 0`, and every other memory cell is unchanged. Proof:
`BoolMasked2DKernelIO₁.Implements.intro` assembles the region-model masked
triple with the flat-memory bridge side conditions; the store is gated by the
static mask, so `hsub` is the identity. -/
specification dropout_kernel_correctness
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    dropoutIO x_ptr x_keep_ptr output_ptr n_elements p BLOCK_SIZE ⊨
      fun _ _ bs xs i => if bs i then xs i / (1 - p) else 0 := by
  refine BoolMasked2DKernelIO₁.Implements.intro _ ?_ (fun _ _ _ _ h => h) ?_ ?_
  · exact dropout_kernel_flattenOk x_ptr x_keep_ptr output_ptr n_elements p
      BLOCK_SIZE
  · intro bounds s h1 h2 h3
    exact dropout_kernel_traceSafe x_ptr x_keep_ptr output_ptr n_elements p
      BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ bs xs hx hb
    exact dropout_kernel_region_run x_ptr x_keep_ptr output_ptr n_elements p
      BLOCK_SIZE s₀ bs xs hx hb

end VeriTile.Bench.TritonBenchG.DropoutTriton

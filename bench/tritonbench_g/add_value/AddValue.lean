import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `add_value` — strict per-kernel correctness

`puzzle1_kernel` is an elementwise add-a-scalar: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of one input, adds the host-supplied
scalar `value` lane-wise, and stores to `output_ptr`, masked by `offsets < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`puzzle1_kernel[grid](...)`, the grid size
`cdiv(N, BLOCK_SIZE)`, scheduling, and how the runtime composes per-program
writes into one buffer) is the *trusted boundary*, not a proof obligation here.
Because `pid` is universally quantified, the per-program statement covers every
program of the grid.

The headline is stated on the masked KernelIO `⊨` surface
(`MaskedKernelIO₁.Implements`): a full masked Hoare triple over **flat pointer
memory** — for every disjoint base-pointer placement of the two buffers, every
program id whose active lanes are in bounds (partial blocks may overhang the
buffer on their inactive lanes), and every launch state whose input window
holds `xs` at the active lanes, the translated pointer kernel terminates,
every active output lane holds `xs i + value`, and every other flat cell is
untouched.

## Proof architecture

```
puzzle1_kernel_correctness                  ← TOP SPECIFICATION (io ⊨ · + value)
  ├─ puzzle1_kernel_flattenOk               bridge fragment membership
  ├─ puzzle1_kernel_traceSafe               per-execution lane-wise safety walk
  └─ puzzle1_kernel_region_run              region-model masked Hoare triple
       ├─ puzzle1_kernel_correct            algorithm-layer readback per lane
       └─ puzzle1_kernel_frame              masked scatter-store cell frame
```

The spec is plain elementwise `xs i + value` — no optimizer/reduction oracle
applies. `addValueIO` is the kernel's masked IO signature: `read`/`write`
windows at `pid * BLOCK_SIZE`, active lanes `pid * BLOCK_SIZE + j < N` (the
kernel's shared load/store mask).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed at the region layer: the input is read into
registers before the scatter, so the result is correct even if `output_ptr`
aliases `x_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `puzzle1_kernel`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < N`): the output region holds `xs i + value`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store). -/
theorem puzzle1_kernel_correct
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s x_ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value) s)
                output_ptr BLOCK_SIZE s.pid i
        = some (if addr < N then xs i + value
                else s.readMem output_ptr addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, puzzle1_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < N
  · simp [hi, h_x]
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
private theorem puzzle1_kernel_frame
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (s s1 : BlockState)
    (hExec : exec ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < N →
      ¬(output_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, puzzle1_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input window is loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₁.Implements.intro`; the value half reuses
`puzzle1_kernel_correct` (instantiated at the tile the state actually holds). -/
theorem puzzle1_kernel_region_run
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < N →
      s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((puzzle1_kernel x_ptr output_ptr N
        BLOCK_SIZE value).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < N →
          s1.readMem output_ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j + value)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < N →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := puzzle1_kernel_correct x_ptr output_ptr N BLOCK_SIZE value
    s₀ (fun j => s₀.readMem x_ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value) s₀
      = exec ((puzzle1_kernel x_ptr output_ptr N
          BLOCK_SIZE value).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((puzzle1_kernel x_ptr output_ptr N
      BLOCK_SIZE value).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, puzzle1_kernel, ComputeKernel.toAlgKernel, stepStmts,
          stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine puzzle1_kernel_frame x_ptr output_ptr N BLOCK_SIZE value
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< N`, so the bounds
contract is **lane-wise** — every *active* lane's address is below the region
bound of the buffer it touches. -/
theorem puzzle1_kernel_traceSafe
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem puzzle1_kernel_flattenOk
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `puzzle1_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < N`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange; mask = offsets < N`), and
the headline **proves** the kernel's actual addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the active lanes. -/
def addValueIO (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) : MaskedKernelIO₁ where
  kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N

/-- **The headline**: `puzzle1_kernel` implements lane-wise add-a-scalar
`xs i + value` on its masked IO signature — for every disjoint flat placement
of the two buffers, every program id whose active lanes are in bounds, and
every launch state whose input window holds `xs` at the active lanes, the
translated pointer kernel terminates, every active output lane holds
`xs i + value`, and every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification puzzle1_kernel_correctness
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨ fun xs i => xs i + value := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact puzzle1_kernel_flattenOk x_ptr output_ptr N BLOCK_SIZE value
  · intro bounds s h1 h2 _
    exact puzzle1_kernel_traceSafe x_ptr output_ptr N BLOCK_SIZE value
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := puzzle1_kernel_region_run
      x_ptr output_ptr N BLOCK_SIZE value s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

/-! ## The `⊨[R]` rounding face (single-shot genre)

Everything below is purely additive; the exact stack above is untouched.
`puzzle1_kernel` is **single-shot** (no loop) and entirely **cast-free**: the
five register assigns are index/mask/add arithmetic with no `castFloat`, the
load is a masked `.real` load, and the terminal store is `.real`-typed
(`stepStmtR` delegates `.real` writes to the exact write). So `execR R`
collapses verbatim onto the exact stepper for every `R`, and the proven
`puzzle1_kernel_region_run` stack is reused unchanged — the `⊨[R]` face adds
only the `TraceSafeR` walk and the typed readback.

```
puzzle1_kernel_io_correctness             ← TOP SPECIFICATION (io ⊨[R, .real] · + value)
  ├─ puzzle1_kernel_flattenOk             (shared with the exact headline)
  ├─ puzzle1_kernel_traceSafeR            per-execution lane-wise safety walk under R
  └─ puzzle1_kernel_castFree              execR R ≡ exec, then puzzle1_kernel_region_run
```
-/

/-- The kernel is **cast-free**: every statement steps identically under
`stepStmtsR R` and `stepStmts`, so `execR R` is the exact stepper. -/
private theorem puzzle1_kernel_castFree (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) (s : BlockState) :
    execR R ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s
      = exec ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  simp [execR, exec, puzzle1_kernel, ComputeKernel.toAlgKernel,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
    BlockState.writeMemTypedR,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk **under the rounding model** — the `hts`
obligation of `MaskedKernelIO₁.ImplementsR.intro`. Same lane-wise bounds
contract as `puzzle1_kernel_traceSafe`: the masked load and the masked store
address the window `pid * BLOCK_SIZE + j`, active only when `< N`. -/
theorem puzzle1_kernel_traceSafeR (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds x_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < N →
      s.pid * BLOCK_SIZE + j.val < bounds output_ptr) :
    Kernel.TraceSafeR R bounds
      ((puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAtR` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [puzzle1_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.SafeAtR,
    stepStmtR, evalOpR.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MemAccess.SafeAtR, MaskOpt.ActiveR, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- **The `⊨[R]` headline**: for every rounding model `R`, `puzzle1_kernel`
implements lane-wise add-a-scalar `xs i + value` on its masked IO signature
at the `.real` output grid. Same full masked Hoare triple as
`puzzle1_kernel_correctness` — ∀ disjoint flat placement, ∀ program id whose
active lanes are in bounds, ∀ launch state whose input window holds `xs` at
the active lanes — but the run is `execR R` and every active output lane is
read back as an `.real`-typed cell holding `R.round .real (xs i + value)`.

The store is untyped (`tl.store(output_ptr + offsets, output, mask=mask)` —
no `.to(...)`), so the honest grid is `.real` and the boundary round
degenerates (`R.round .real = id`): this kernel's rounding face carries the
exact value contract for every `R`, with the *modeling* claim that the
kernel introduces no rounding event of its own. No hypotheses: the block
geometry is universally quantified, `BLOCK_SIZE = 0` is the vacuous
zero-lane launch, and the window is injective outright. -/
specification puzzle1_kernel_io_correctness (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨[R, .real] fun xs i => xs i + value := by
  refine MaskedKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact puzzle1_kernel_flattenOk x_ptr output_ptr N BLOCK_SIZE value
  · intro bounds s h1 h2 _
    exact puzzle1_kernel_traceSafeR R x_ptr output_ptr N BLOCK_SIZE value
      bounds s h1 h2
  · intro s₀ xs hx
    simp only [addValueIO] at hx ⊢
    obtain ⟨s1, hexec, hval, hframe⟩ := puzzle1_kernel_region_run
      x_ptr output_ptr N BLOCK_SIZE value s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    refine ⟨s1, ?_, ?_, fun r o hcond _ => hframe r o hcond⟩
    · rw [puzzle1_kernel_castFree R x_ptr output_ptr N BLOCK_SIZE value s₀]
      exact hexec
    · intro j hj
      rw [BlockState.readMemAs_real, hval j hj]
      simp

end VeriTile.Bench.TritonBenchG.AddValue

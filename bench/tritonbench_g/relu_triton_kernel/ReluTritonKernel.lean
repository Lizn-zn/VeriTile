import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common

/-!
# `relu_triton_kernel` — strict per-kernel correctness

`relu_kernel` is an elementwise ReLU: program `pid` loads block
`[pid·block_size, (pid+1)·block_size)` of `x_ptr`, computes
`tl.where(x >= 0, x, 0.0)` lane-wise, and — *only when `pid == 0`* — stores the
result to `out_ptr`, masked by `offsets < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`relu_kernel[grid](...)`, the grid size
`cdiv(N, BLOCK_SIZE)`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here.

The headline is stated on the masked KernelIO `⊨` surface
(`MaskedKernelIO₁.Implements`): a full masked Hoare triple over **flat pointer
memory** — ∀ disjoint base-pointer placements of the two buffers, ∀ program
ids whose active lanes are in bounds, ∀ launch states whose input window holds
`xs` at the read-active lanes — the translated pointer kernel terminates,
every write-active output lane holds `relu (xs i)`, and every other flat cell
is untouched.

**The `pid == 0` write gate is signature content.** The benchmark kernel
stores only from program 0 (a quirk faithfully preserved in the
transcription), while its load is masked by `offsets < N` on *every* program.
The IO signature carries this asymmetry directly: `reluIO.mask` (read-active)
is the load mask `pid * block_size + j < N`, and `reluIO.writeMask`
(write-active: value postcondition, write bounds, frame) is the kernel's
actual store gate `pid = 0 ∧ pid * block_size + j < N`. Programs with
`pid ≠ 0` have no write-active lanes — the triple asserts they write nothing —
so the headline is fully general over `N`, exactly like the retired
`Realizes_without_Rounding` summary's `writeIf` predicate.

## Proof architecture

```
relu_kernel_correctness                     ← TOP SPECIFICATION (io ⊨ relu)
  ├─ relu_kernel_flattenOk                  bridge fragment membership
  ├─ relu_kernel_traceSafe                  per-execution lane-wise safety walk
  └─ relu_kernel_region_run                 region-model masked Hoare triple
       ├─ relu_kernel_correct               ← algorithm-layer readback per lane
       └─ relu_kernel_frame                 masked-store cell-level frame
```

The spec is the reusable `TiledActivation.relu` oracle applied to the values
this lane loads. `reluIO` is the kernel's masked IO signature: `read`/`write`
windows at `pid * block_size`, read-active lanes `pid * block_size + j < N`
(the load mask), write-active lanes `pid = 0 ∧ pid * block_size + j < N` (the
gated store mask).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. No output/input disjointness is assumed: the input is read into
registers before the (gated) scatter, so the result is correct even if `out_ptr`
aliases `x_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}

/-- Algorithm-layer correctness for `relu_kernel`, from a launch state whose
input window is loaded at the **active lanes only** (`pid * block_size + j < N`;
the masked load reads the undef channel elsewhere, so inactive lanes are never
consulted).

Only `pid = 0` writes. In that case, active lanes write the conditional-form
ReLU and inactive tail lanes are preserved; nonzero pids preserve the observed
output cells. -/
theorem relu_kernel_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (_hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : ∀ j : Fin block_size, s.pid * block_size + j.val < N →
      s.readMem x_ptr (s.pid * block_size + j.val) = xs j) :
    ∀ i : Fin block_size,
      let addr := s.pid * block_size + i.val
      observeAt (exec (relu_kernel x_ptr out_ptr N block_size) s)
          out_ptr block_size s.pid i
        = some (if s.pid = 0 ∧ addr < N then TiledActivation.relu (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  by_cases hpid : s.pid = 0
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          BlockState.tileIndex1d_offset_injective (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hx0 : s.readMem x_ptr i.val = xs i := by
        simpa [hpid] using h_x i (by simpa [hpid] using hi)
      simp [hi, hx0, TiledActivation.relu]
      split_ifs with hge
      · -- hge : (some 0 : WithBot ℝ) ≤ some (xs i)
        have h : (0 : ℝ) ≤ xs i :=
          WithBot.coe_le_coe.mp (hge : (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        change xs i = max 0 (xs i)
        exact (max_eq_right h).symm
      · have h : ¬ ((0 : ℝ) ≤ xs i) := fun h' =>
          hge (WithBot.coe_le_coe.mpr h' :
            (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        rw [not_le] at h
        change (0.0 : ℝ) = max 0 (xs i)
        rw [max_eq_left h.le]
        norm_num
    · simp [hi]
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]

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
/-- Frame half: every memory cell not actively written by the `pid == 0`-gated
masked output store is preserved by the run — in particular every cell of
every region other than `out_ptr`, the *inactive* lanes of the output window,
and (for `pid ≠ 0`) everything. -/
private theorem relu_kernel_frame (x_ptr out_ptr : RegionName)
    (N B : Nat) (s s1 : BlockState)
    (hExec : exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, s.pid = 0 ∧ s.pid * B + i.val < N →
      ¬(out_ptr = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  by_cases hpid : s.pid = 0
  · simp [exec, relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    refine hmiss k.1 ⟨hpid, ?_⟩ ⟨hc.1, ?_⟩
    · simpa [hpid] using hmk
    · simpa [hpid] using hc.2
  · simp [exec, relu_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
      evalOp.eq_def, Tile.bop, Tile.cop, Tile.select,
      NumericDType.add, NumericDType.mul,
      ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid] at hExec
    subst hExec
    rfl

/-- **The region-model masked Hoare triple** — termination, write-active-lane
output values, and frame off the write-active output lanes, from any launch
state whose input window is loaded at the **read-active lanes only**. The
write-active set is the kernel's actual store gate
`pid = 0 ∧ pid * B + j < N`, so the triple is fully general over `N`:
programs with `pid ≠ 0` have no write-active lanes and preserve all of
memory. This is what the `⊨` headline transports to flat memory. -/
theorem relu_kernel_region_run
    (x_ptr out_ptr : RegionName)
    (N B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < N →
      s₀.readMem x_ptr (s₀.pid * B + j.val) = xs j) :
    ∃ s1, exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
          s1.readMem out_ptr (s₀.pid * B + j.val) = TiledActivation.relu (xs j))
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin B, s₀.pid = 0 ∧ s₀.pid * B + j.val < N →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := relu_kernel_correct x_ptr out_ptr N B hB s₀ xs hx
  rw [show exec (relu_kernel x_ptr out_ptr N B) s₀
      = exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ from rfl] at hobs
  cases hsrc : exec ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · obtain ⟨hpid0, hj⟩ := hj
        have hj' : j.val < N := by simpa [hpid0] using hj
        have := hobs j
        rw [hsrc] at this
        simpa [observeAt, hpid0, hj'] using this
      · refine relu_kernel_frame x_ptr out_ptr N B s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

/-- The ReLU surface sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked load/store, `where`, comparison, and the
`if pid == 0` gate are all covered). -/
theorem relu_kernel_flattenOk (x_ptr out_ptr : RegionName) (N B : Nat) :
    ((relu_kernel x_ptr out_ptr N B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [relu_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load touches only its read-active
lanes (`pid * B + j < N`), and the store — reached only when the `pid == 0`
gate is taken — only its write-active lanes, so the bounds contract is
lane-wise: every *active* lane's address is below the region bound of the
buffer it touches. -/
theorem relu_kernel_traceSafe (x_ptr out_ptr : RegionName)
    (N B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < N →
      s.pid * B + j.val < bounds x_ptr)
    (hout : ∀ j : Fin B, s.pid = 0 ∧ s.pid * B + j.val < N →
      s.pid * B + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds ((relu_kernel x_ptr out_ptr N B).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [relu_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.select,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt, ComparableDType.eq, ComparableDType.ge,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  refine ⟨fun a ha => hx a ha, ?_⟩
  split <;> exact ⟨fun h a ha => hout a ⟨h, ha⟩, trivial⟩

/-- `relu_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the window length each program owns;
* `read`/`write` — where program `pid` reads its input tile / writes its
  output tile (`pid * block_size`, the launch convention
  `offsets = pid * block_size + arange`);
* `mask` — program `pid`'s **read-active lanes**, `pid * block_size + j < N`:
  the kernel's load mask (every program loads);
* `writeMask` — program `pid`'s **write-active lanes**,
  `pid = 0 ∧ pid * block_size + j < N`: the kernel's `pid == 0`-gated store
  mask. Programs with `pid ≠ 0` have no write-active lanes; the frame asserts
  they write nothing.

Inactive lanes carry no obligations on either side. The windows and masks are
declared, not parsed from the kernel; the headline **proves** the kernel's
actual addressing, masking, and store gate match them. -/
def reluIO (x_ptr out_ptr : RegionName) (N block_size : Nat) :
    MaskedKernelIO₁ where
  kernel := relu_kernel x_ptr out_ptr N block_size
  inp := x_ptr
  out := out_ptr
  B := block_size
  read := fun pid => pid * block_size
  write := fun pid => pid * block_size
  mask := fun pid j => pid * block_size + j.val < N
  writeMask := fun pid j => pid = 0 ∧ pid * block_size + j.val < N

/-- **The headline**: `relu_kernel` implements lane-wise
`TiledActivation.relu` (`max 0 ·`) on its masked IO signature — the full
masked Hoare triple over flat pointer memory (see the module docstring),
**general over `N`**: the kernel's `pid == 0` store gate is part of the
audited IO signature (`reluIO.writeMask`), so programs with `pid ≠ 0` carry
a pure frame obligation and no value claim. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the bridge side conditions. -/
specification relu_kernel_correctness
    (x_ptr out_ptr : RegionName) (N block_size : Nat)
    (hB : 0 < block_size) :
    reluIO x_ptr out_ptr N block_size ⊨
      fun xs i => TiledActivation.relu (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_kernel_flattenOk x_ptr out_ptr N block_size
  · intro bounds s h1 h2 _
    exact relu_kernel_traceSafe x_ptr out_ptr N block_size bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      relu_kernel_region_run x_ptr out_ptr N block_size hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.ReluTritonKernel

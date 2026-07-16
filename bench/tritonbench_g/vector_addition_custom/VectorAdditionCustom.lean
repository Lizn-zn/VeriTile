import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `vector_addition_custom` — strict per-kernel correctness

`_add_kernel` is an elementwise add: program `prog_id` loads block
`[prog_id·BLOCK, (prog_id+1)·BLOCK)` of inputs `A` and `B`, adds them lane-wise,
and stores to `C`, masked by `offs < size`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_add_kernel[grid](...)`, the grid size
`cdiv(size, BLOCK)`, the host-side `BLOCK` choice, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `prog_id` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
add_kernel_correctness                        ← TOP THEOREM (addCustomIO ⊨ pointwise add)
  ├─ add_kernel_flattenOk                     bridge fragment membership
  ├─ add_kernel_traceSafe                     per-execution lane-wise safety walk
  └─ add_kernel_region_run                    region-model masked Hoare triple
       ├─ add_kernel_correct                  algorithm-layer readback per lane
       └─ add_kernel_frame                    masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature** `addCustomIO`
(`MaskedKernelIO₂`): which buffer is which argument, where program `prog_id`
reads/writes its `BLOCK`-lane window, and the active-lane predicate
`prog_id * BLOCK + j < size`. `⊨` is the audit-once masked Hoare-triple
combinator (`MaskedKernelIO₂.Implements`): for **every** disjoint placement of
the three buffers in flat memory, **every** program id all of whose *active*
lanes are in bounds (partial blocks may overhang the buffer on their inactive
lanes), and **every** launch state whose input windows hold `as`/`bs` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `as i + bs i`, and every other memory cell is unchanged.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed at the region layer: both inputs are read into
registers before the scatter, so the result is correct even if `C` aliases `A`
or `B`.
-/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}

/-- Algorithm-layer correctness for `_add_kernel`.

Each active lane writes `A + B`; inactive tail lanes are preserved. -/
theorem add_kernel_correct
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    ∀ i : Fin BLOCK,
      let addr := s.pid * BLOCK + i.val
      observeAt (exec (_add_kernel A B C size BLOCK) s) C BLOCK s.pid i
        = some (if addr < size then as i + bs i else s.readMem C addr) := by
  intro i
  simp [observeAt, exec, _add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_a h_b
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK + i.val < size
  · simp [hi, h_a, h_b]
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
than `C`, and the *inactive* lanes of the output window itself. -/
private theorem add_kernel_frame
    (A B C : RegionName)
    (size BLOCK : Nat) (s s1 : BlockState)
    (hExec : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK, s.pid * BLOCK + i.val < size →
      ¬(C = r ∧ s.pid * BLOCK + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, _add_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input windows are loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₂.Implements.intro`; the value half reuses
`add_kernel_correct` (instantiated at the tiles the state actually holds). -/
theorem add_kernel_region_run
    (A B C : RegionName)
    (size BLOCK : Nat)
    (s₀ : BlockState) (as bs : Fin BLOCK → ℝ)
    (ha : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem A (s₀.pid * BLOCK + j.val) = as j)
    (hb : ∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
      s₀.readMem B (s₀.pid * BLOCK + j.val) = bs j) :
    ∃ s1, exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK, s₀.pid * BLOCK + j.val < size →
          s1.readMem C (s₀.pid * BLOCK + j.val) = as j + bs j)
      ∧ (∀ r o,
          (r ≠ C ∨ ∀ j : Fin BLOCK,
            s₀.pid * BLOCK + j.val < size →
              o ≠ s₀.pid * BLOCK + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct A B C size BLOCK
    s₀ (fun j => s₀.readMem A (s₀.pid * BLOCK + j.val))
    (fun j => s₀.readMem B (s₀.pid * BLOCK + j.val))
    (fun _ => rfl) (fun _ => rfl)
  rw [show exec (_add_kernel A B C size BLOCK) s₀
      = exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((_add_kernel A B C size BLOCK).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, _add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, ha j hj, hb j hj]
      · refine add_kernel_frame A B C size BLOCK
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: both masked loads and the masked store address
the same window `prog_id * BLOCK + j`, active only when `< size`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem add_kernel_traceSafe
    (A B C : RegionName)
    (size BLOCK : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (ha : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds A)
    (hb : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds B)
    (hout : ∀ j : Fin BLOCK, s.pid * BLOCK + j.val < size →
      s.pid * BLOCK + j.val < bounds C) :
    Kernel.TraceSafe bounds
      ((_add_kernel A B C size BLOCK).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all five statements, discharging every
  -- load-free `SafeAt` and reducing the three memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha' => ha a ha', fun a hb' => hb a hb', fun a ho' => hout a ho'⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem add_kernel_flattenOk
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ((_add_kernel A B C size BLOCK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [_add_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `_add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `prog_id` reads its input tiles / writes its output tile, and the
active-lane predicate `prog_id * BLOCK + j < size`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offs = prog_id * BLOCK + arange; mask = offs < size`), and the
headline **proves** the kernel's actual addressing and masking match them.
Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size

/-- **The headline**: `_add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `as`/`bs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `as i + bs i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification add_kernel_correctness
    (A B C : RegionName)
    (size BLOCK : Nat) :
    addCustomIO A B C size BLOCK
      ⊨ fun as bs i => as i + bs i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_kernel_flattenOk A B C size BLOCK
  · intro bounds s h1 h2 h3 _
    exact add_kernel_traceSafe A B C size BLOCK bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := add_kernel_region_run
      A B C size BLOCK s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom

import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `cosine_compute` — strict per-kernel correctness

`cos_func` is an elementwise cosine: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of input `a`, casts each lane to float32,
applies `tl.cos`, and stores `cos(x)` to output `b`, masked by
`offset < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`cos_func[(grid_size,1,1)](...)`, the grid size
`cdiv(n_elements, block_size)`, the host-side `block_size` choice, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because `pid` is universally quantified, the
per-program statement covers every program of the grid.

The headline is stated on the masked KernelIO `⊨` surface
(`MaskedKernelIO₁.Implements`): a full masked Hoare triple over **flat pointer
memory** — ∀ disjoint base-pointer placements of the two buffers, ∀ program
ids whose active lanes are in bounds, ∀ launch states whose input window holds
`xs` at the active lanes — the translated pointer kernel terminates, every
active output lane holds `Real.cos (xs i)`, and every other flat cell is
untouched.

## Proof architecture

```
cos_func_correctness                          ← TOP SPECIFICATION (cosIO ⊨ cos)
  ├─ cos_func_flattenOk                       bridge fragment membership
  ├─ cos_func_traceSafe                       per-execution lane-wise safety walk
  └─ cos_func_region_run                      region-model masked Hoare triple
       ├─ cos_func_correct                    algorithm-layer readback per lane
       └─ cos_func_frame                      masked scatter-store cell frame
```

The spec is plain elementwise `Real.cos (xs i)` — no optimizer/reduction oracle
applies. `cosIO` is the kernel's masked IO signature: `read`/`write` windows at
`pid * BLOCK_SIZE` and active-lane predicate `pid * BLOCK_SIZE + j < n_elements`
(the shared load/store mask).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The
`(a_value).to(tl.float32)` cast reduces to the identity at the algorithm layer
(post-erasure all dtypes unify to `ℝ`). No output/input disjointness is assumed
at the region layer: the input is read into registers before the scatter, so
the result is correct even if `b` aliases `a`.
-/

namespace VeriTile.Bench.TritonBenchG.CosineCompute

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
-/
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, b_value, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `cos_func`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < n_elements`): the output region holds
  `Real.cos (xs i)`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store).

The `(a_value).to(tl.float32)` cast at the compute layer reduces to
`Op.castFloat`, which is the identity at the algorithm layer (post-erasure
all dtypes unify to `ℝ`). -/
theorem cos_func_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (cos_func a b n_elements BLOCK_SIZE) s)
                b BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.cos (xs i)
                else s.readMem b addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, cos_func, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realCos,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
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
than `b`, and the *inactive* lanes of the output window itself. -/
private theorem cos_func_frame
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(b = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, cos_func, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, WithBot.realCos,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input window is loaded at the **active lanes only**. This is the `hrun`
obligation of `MaskedKernelIO₁.Implements.intro`; the value half reuses
`cos_func_correct` (instantiated at the tile the state actually holds). -/
theorem cos_func_region_run
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem a (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem b (s₀.pid * BLOCK_SIZE + j.val) = Real.cos (xs j))
      ∧ (∀ r o,
          (r ≠ b ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := cos_func_correct a b n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem a (s₀.pid * BLOCK_SIZE + j.val)) (fun _ => rfl)
  rw [show exec (cos_func a b n_elements BLOCK_SIZE) s₀
      = exec ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, cos_func, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt, WithBot.realCos,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine cos_func_frame a b n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so
the bounds contract is **lane-wise** — every *active* lane's address is below
the region bound of the buffer it touches. -/
theorem cos_func_traceSafe
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds a)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds b) :
    Kernel.TraceSafe bounds
      ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all five statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [cos_func, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt, WithBot.realCos,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked load/store, comparison, `tl.cos`, and the
float32 cast are all covered). -/
theorem cos_func_flattenOk
    (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    ((cos_func a b n_elements BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [cos_func, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `cos_func`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offset = pid * BLOCK_SIZE + arange;
mask = offset < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
def cosIO (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := cos_func a b n_elements BLOCK_SIZE
  inp := a
  out := b
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `cos_func` implements lane-wise `Real.cos` on its masked
IO signature — for every disjoint flat placement of the two buffers, every
program id whose active lanes are in bounds, and every launch state whose
input window holds `xs` at the active lanes, the translated pointer kernel
terminates, every active output lane holds `Real.cos (xs i)`, and every other
memory cell is unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles
the region-model masked triple with the flat-memory bridge side conditions. -/
specification cos_func_correctness
    (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    cosIO a b n_elements BLOCK_SIZE ⊨ fun xs i => Real.cos (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact cos_func_flattenOk a b n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact cos_func_traceSafe a b n_elements BLOCK_SIZE bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := cos_func_region_run
      a b n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.CosineCompute

/-
VeriTile.Triton.KernelLemmas.CarryFold

The **carry-fold driver** for recurrence kernels whose per-step correctness is
stated on a single-step *slice*.

## The gap this closes

A family of recurrence ports (`fused_recurrent_*`, `pow_scalar_tensor`,
`var_len_copy`, …) states its loop body as a standalone slice kernel that
*loads* the previous carry from a region `HPrev`, computes one step, and
*stores* the new carry into a region `HOut`.  Each such face is proved under a
hypothesis pinning `HPrev`'s contents to the closed form at step `m`.  Those
hypotheses are not chainable as stated, for two independent reasons:

1. **No threading.** The face concludes about `HOut` while the next step's
   hypothesis constrains `HPrev` — a *different* region, with nothing
   identifying them.
2. **No frame.** Even with `HOut = HPrev`, the next step's face is stated
   relative to *its own* initial state `t`, so its closed form reads `t`'s
   memory and `t`'s program ids; the chain only closes if the step provably
   left the *input* regions and the launch environment alone.

This file supplies both halves: `AgreeOutsideRegion` is the frame relation for
(2), and `execChain` / `carryFold_execChain` perform the induction for (1).

## Scope and honesty limits

* `execChain` models the slices as **separate launches**: every stage starts
  from a fresh register file (`BlockState.resetRegs`), exactly as
  `execPipelineR` does downstream in `VeriTile.Triton.Float.Pipeline`.  This
  file cannot reuse `execPipelineR` because `KernelLemmas` precedes the `Float`
  layer; `execChain` is its exact-semantics counterpart at this layer.
* Because the carry travels through a **memory region** between stages, a
  carry-fold result proved with this driver is a statement about a kernel whose
  recurrence state lives in memory.  A real Triton loop keeps the carry in a
  *register*.  Threading a region carry removes the per-step assumption, but it
  does **not** model register semantics: for that, the loop must be attacked
  directly with `forRange_inv` / `forRangeDyn_inv` (`LoopInvariant.lean`), which
  carry a register tile through the kernel's own `Stmt.forRange`.  The two
  drivers are complementary, not substitutes.
* **Four bench ports consume this driver** (`fused_rwkv6_kernel`,
  `fused_recurrent_delta`, `fused_recurrent_retention` — forward state only —
  and `chunk_gated_attention`).  The other four recurrence ports still state
  their carry invariant as a *pinned* `hPrev` hypothesis, and in every case that
  is **structural** rather than pending work: `fused_recurrent_hgrn` keeps its
  carry in registers only (no region is written back); `lightning_attention`'s
  kv step reads per-block *staged* regions, so a chain would re-stage the same
  block; `chunk_delta_fwd`'s carry address is time-indexed while `addr` here is
  fixed across steps; and `chunk_gate_recurrence`'s step *reads* its carry at a
  time-free address but *writes* a time-indexed history row of a different
  region, so read and write never meet — generalizing `addr` to move with the
  step would not close that one either.  A port's headline is unaffected by this driver
  unless it states the fold as its own theorem, so do not read a port's
  `hPrev` hypothesis as evidence either way — check for a `*_carry_fold`.

## What a port must supply to actually use `carryFold_execChain`

Deriving these from a port's existing single-step face is the remaining work,
and none of the three is free:

1. **A frame result for the step slice** — `AgreeOutsideRegion C s t'` from
   `exec (step m).toAlgKernel t.resetRegs = some t'`.  A port's
   `Realizes_without_Rounding` face pins only the *written* lanes, so this
   needs its own walk over the slice (`foldl_writeMem_pids` above plus
   `foldl_writeMem_mem_preserve_other_region`).
2. **A closed-form congruence** — the port's `val` reads the input regions and
   `pids` of the *reference* state `s`, while the step face at a mid-chain state
   `t` produces a value over `t`.  Transporting one to the other needs a
   congruence lemma for the port's own closed form under `AgreeOutsideRegion`,
   which in turn forces the side conditions `inputRegion ≠ C` for every input
   region the closed form reads.
3. **A lane restriction** — a masked step writes only its in-range lanes, so
   the invariant `∀ i, readMem C (addr i) = val m i` is *false* at out-of-range
   lanes (they keep `val 0`, not `val m`).  Instantiate `ι` at the subtype of
   write-active lanes rather than the full tile index.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-! ## Missing scatter frame primitives

`BlockState` ships the masked-scatter `pids` frame
(`foldl_writeMem_prop_masked_pids`) and both *unmasked* `mem` frames
(`foldl_writeMem_mem_preserve_other_region`,
`foldl_writeMem_mem_preserve_unhit`).  The two entries of that square this
driver needs and that were missing are the unmasked `pids` frame — for a slice
whose store carries no mask — and the **masked** `mem`-outside-the-region
frame, which is what `AgreeOutsideRegion.mem` is discharged by for the (usual)
masked store. -/

namespace BlockState

/-- An unmasked `writeMem` scatter leaves the program ids alone. The unmasked
companion of `foldl_writeMem_prop_masked_pids`. -/
theorem foldl_writeMem_pids {α : Type} (region : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).pids)
      = s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons, ih]; simp

/-- A **masked** `writeMem` scatter into `region` leaves every other region's
cells untouched.  The masked companion of
`foldl_writeMem_mem_preserve_other_region`; stated on `mem` rather than
`readMem` because that is the field `AgreeOutsideRegion` pins. -/
theorem foldl_writeMem_prop_masked_mem_preserve_other_region {α : Type}
    {region : RegionName} (offsetFn : α → Nat) (valueFn : α → ℝ)
    (P : α → Prop) [DecidablePred P] (l : List α)
    (r : RegionName) (hr : r ≠ region) (o : Nat) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem r o)
        = s.mem r o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih]
      by_cases hP : P hd
      · rw [if_pos hP, writeMem_mem, if_neg (fun hc => hr hc.1)]
      · rw [if_neg hP]

end BlockState

/-! ## The frame relation -/

/--
**Frame relation for carry chaining.** `t` agrees with the reference state `s`
on the launch environment (`pids`) and on every memory region other than the
carry region `C`.

This is exactly the agreement a per-step face needs in order to be applied at a
mid-chain state `t` while still speaking about closed forms written over the
*original* state `s`: the closed forms read input regions (≠ `C`) and `pids`,
both of which this relation pins.

Note it is deliberately **region-granular**, not address-granular: a chain does
not care which cells inside `C` moved, only that nothing outside `C` did. The
address-granular counterpart is the frame half of
`ComputeCorrect.RealizesFrame_without_Rounding`, which is strictly stronger and
implies this relation (see `RealizesFrame_without_Rounding.mem_other_region` in
`Correctness.lean`) but says nothing about `pids`.
-/
structure AgreeOutsideRegion (C : RegionName) (s t : BlockState) : Prop where
  /-- The launch environment is untouched. -/
  pids : t.pids = s.pids
  /-- Every region other than the carry region is untouched. -/
  mem : ∀ r, r ≠ C → ∀ o, t.mem r o = s.mem r o

namespace AgreeOutsideRegion

variable {C : RegionName} {s t u : BlockState}

/-- Every state agrees with itself. -/
theorem refl (C : RegionName) (s : BlockState) : AgreeOutsideRegion C s s :=
  ⟨rfl, fun _ _ _ => rfl⟩

/-- Agreement composes along a chain of stages. -/
theorem trans (h₁ : AgreeOutsideRegion C s t) (h₂ : AgreeOutsideRegion C t u) :
    AgreeOutsideRegion C s u :=
  ⟨h₂.pids.trans h₁.pids, fun r hr o => (h₂.mem r hr o).trans (h₁.mem r hr o)⟩

/-- Real-channel readback congruence outside the carry region: this is the form
in which closed-form definitions (`kValR`-style readers) consume the frame. -/
theorem readMem (h : AgreeOutsideRegion C s t) (r : RegionName) (hr : r ≠ C)
    (o : Nat) : t.readMem r o = s.readMem r o := by
  unfold BlockState.readMem
  rw [h.mem r hr o]

/-- A fresh-launch register reset preserves agreement: `resetRegs` touches only
the register file. -/
theorem resetRegs (h : AgreeOutsideRegion C s t) :
    AgreeOutsideRegion C s t.resetRegs :=
  ⟨h.pids, h.mem⟩

end AgreeOutsideRegion

/-! ## Chained launch semantics -/

/--
Run a list of stage kernels in order, resetting the register file before
**every** stage: a real launch never inherits registers, so the only thing that
crosses a stage seam is memory (plus the launch environment).

This is the exact-semantics counterpart of `execPipelineR .triv` from
`VeriTile.Triton.Float.Pipeline`; it is duplicated here only because
`KernelLemmas` precedes the `Float` layer in the dependency order.
-/
noncomputable def execChain : List ComputeKernel → BlockState → Option BlockState
  | [], s => some s
  | ck :: ks, s => (exec ck.toAlgKernel s.resetRegs).bind (execChain ks)

@[simp] theorem execChain_nil (s : BlockState) : execChain [] s = some s := rfl

@[simp] theorem execChain_cons (ck : ComputeKernel) (ks : List ComputeKernel)
    (s : BlockState) :
    execChain (ck :: ks) s
      = (exec ck.toAlgKernel s.resetRegs).bind (execChain ks) := rfl

/-- Peel the **last** stage off a chain. This is the orientation a forward
carry fold needs: the invariant's index grows with the prefix length. -/
theorem execChain_append_singleton (ks : List ComputeKernel)
    (ck : ComputeKernel) (s : BlockState) :
    execChain (ks ++ [ck]) s
      = (execChain ks s).bind (fun t => exec ck.toAlgKernel t.resetRegs) := by
  induction ks generalizing s with
  | nil => simp [execChain]
  | cons k ks ih =>
      rw [List.cons_append, execChain_cons, execChain_cons]
      cases hk : exec k.toAlgKernel s.resetRegs with
      | none => simp
      | some s1 => simp only [Option.bind_some]; exact ih s1

/-- The stage list of an `n`-step fold: `step 0, step 1, …, step (n-1)`. -/
def foldStages (step : Nat → ComputeKernel) (n : Nat) : List ComputeKernel :=
  (List.range n).map step

@[simp] theorem foldStages_zero (step : Nat → ComputeKernel) :
    foldStages step 0 = [] := rfl

theorem foldStages_succ (step : Nat → ComputeKernel) (n : Nat) :
    foldStages step (n + 1) = foldStages step n ++ [step n] := by
  simp [foldStages, List.range_succ]

/-! ## The driver -/

/--
**Master carry-fold driver.** Given an invariant that holds at the start and is
re-established by each stage, it holds at the end of the chain.

Postcondition style (`hRun` is a hypothesis, not a conclusion) so that the
driver never has to prove that a stage terminates — matching
`Kernel.Correct_without_Rounding` and the rest of the correctness layer. It is
generic over the step count `n`, the step family `step`, and the invariant
`Inv`, which is where the carry's closed form lives; the eight recurrence ports
differ only in `step` and `Inv`.
-/
theorem execChain_foldStages_inv
    {step : Nat → ComputeKernel} {Inv : Nat → BlockState → Prop}
    {n : Nat} {s sFinal : BlockState}
    (hBase : Inv 0 s)
    (hStep : ∀ m, m < n → ∀ t t', Inv m t →
      exec (step m).toAlgKernel t.resetRegs = some t' → Inv (m + 1) t')
    (hRun : execChain (foldStages step n) s = some sFinal) :
    Inv n sFinal := by
  -- Induct on a prefix length `j ≤ n`, so that `hStep`'s `m < n` bound (which
  -- mentions the fixed `n`) stays out of the induction motive.
  have key : ∀ j, j ≤ n → ∀ sF, execChain (foldStages step j) s = some sF → Inv j sF := by
    intro j
    induction j with
    | zero =>
        intro _ sF h
        rw [foldStages_zero, execChain_nil] at h
        exact (Option.some.inj h) ▸ hBase
    | succ j ih =>
        intro hjn sF h
        rw [foldStages_succ, execChain_append_singleton] at h
        cases hmid : execChain (foldStages step j) s with
        | none => rw [hmid] at h; simp at h
        | some t =>
            rw [hmid] at h
            simp only [Option.bind_some] at h
            exact hStep j (by omega) t sF (ih (by omega) t hmid) h
  exact key n le_rfl sFinal hRun

/--
**Carry-fold driver, family form.** The specialization of
`execChain_foldStages_inv` to the shape the recurrence ports actually have: a
carry region `C`, a state-independent lane-to-offset map `addr`, and a closed
form `val m` for the carry after `m` steps.

The three inputs named in the gap analysis appear explicitly:

* the **per-step face** is `hStep`'s conclusion `∀ i, t'.readMem C (addr i) = val (m+1) i`;
* the **threading identification** is that the same region `C` is read by the
  face's hypothesis and written by its conclusion — the caller supplies a step
  family whose carry-in and carry-out regions are literally the same name;
* the **frame fact** is `AgreeOutsideRegion C s`, threaded through `hStep` in
  both directions.

`val` is closed over the *reference* state `s`, so the conclusion speaks about
the original inputs, not about a mid-chain readback.
-/
theorem carryFold_execChain
    {ι : Type} {step : Nat → ComputeKernel} {C : RegionName}
    {addr : ι → Nat} {val : Nat → ι → ℝ} {n : Nat} {s sFinal : BlockState}
    (hBase : ∀ i, s.readMem C (addr i) = val 0 i)
    (hStep : ∀ m, m < n → ∀ t t', AgreeOutsideRegion C s t →
      (∀ i, t.readMem C (addr i) = val m i) →
      exec (step m).toAlgKernel t.resetRegs = some t' →
      AgreeOutsideRegion C s t' ∧ ∀ i, t'.readMem C (addr i) = val (m + 1) i)
    (hRun : execChain (foldStages step n) s = some sFinal) :
    AgreeOutsideRegion C s sFinal ∧ ∀ i, sFinal.readMem C (addr i) = val n i :=
  execChain_foldStages_inv
    (Inv := fun m t => AgreeOutsideRegion C s t ∧ ∀ i, t.readMem C (addr i) = val m i)
    ⟨AgreeOutsideRegion.refl C s, hBase⟩
    (fun m hm t t' ht hexec => hStep m hm t t' ht.1 ht.2 hexec)
    hRun

end VeriTile.Triton

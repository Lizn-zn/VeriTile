import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `triton_mul2` — strict per-kernel correctness

The Python module defines two `@triton.jit` programs that both double their
input: `mul2_kernel` (out-of-place, `output = 2 * x` stored to `out_ptr`) and
`mul2_inplace_kernel` (in-place, doubling the elements of `ptr`). Each program
`pid` handles block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`, masked by
`offsets < n_elements`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launches (`mul2_kernel[grid](...)`,
`mul2_inplace_kernel[grid](...)`, the grid size `cdiv(n_elements, BLOCK_SIZE)`,
scheduling, and how the runtime composes per-program writes into one buffer) are
the *trusted boundary*, not proof obligations here. Because `pid` is universally
quantified, each per-program statement covers every program of the grid.

Each kernel's headline is stated on the masked KernelIO `⊨` surface
(`MaskedKernelIO₁.Implements`): a full masked Hoare triple over **flat pointer
memory** — ∀ base-pointer placements of the buffers, ∀ program ids whose active
lanes are in bounds, ∀ launch states whose input window holds `xs` at the
active lanes — the translated pointer kernel terminates, every active output
lane holds `2 * xs i`, and every other flat cell is untouched.

**The in-place kernel's IO signature has `inp = out = ptr`.** `mul2_inplace_kernel`
rewrites the buffer it read, so its signature wires both argument roles to the
same buffer; the triple reads the *old* window contents into `xs` and asserts
the *new* ones — the standard before/after reading (same design as the Adam/Lion
in-place `⊨` headline).

## Proof architecture

```
mul2_kernel_correctness                 ← TOP SPECIFICATION (mul2IO ⊨ 2·x)
  ├─ mul2_kernel_flattenOk              bridge fragment membership
  ├─ mul2_kernel_traceSafe              per-execution lane-wise safety walk
  └─ mul2_kernel_region_run             region-model masked Hoare triple
       ├─ mul2_kernel_correct           algorithm-layer readback per lane
       └─ mul2_kernel_frame             masked scatter-store cell frame
mul2_inplace_kernel_correctness         ← TOP SPECIFICATION (mul2InplaceIO ⊨ 2·x, in place)
  ├─ mul2_inplace_kernel_flattenOk
  ├─ mul2_inplace_kernel_traceSafe
  └─ mul2_inplace_kernel_region_run
       ├─ mul2_inplace_kernel_correct
       └─ mul2_inplace_kernel_frame
```

The spec is plain elementwise `2 * xs i` — no optimizer/reduction oracle
applies.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). No output/input
disjointness is assumed at the region layer: each kernel reads its input
into registers before the scatter, so `mul2_kernel` is correct even if `out_ptr`
aliases `in_ptr0`, and `mul2_inplace_kernel` is correct writing back in place.
-/

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = 2 * x
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_inplace_kernel`.

Same allowed mechanical Lean-syntax-only changes as `mul2_kernel`. -/
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = 2 * x
  tl.store(ptr + offsets, output, mask=mask)
}

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

/-! ## `mul2_kernel` (out-of-place) -/

/-- Algorithm-layer correctness for `mul2_kernel`. -/
theorem mul2_kernel_correct
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s in_ptr0 BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s)
          out_ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem out_ptr addr) := by
  intro i
  simp [observeAt, exec, mul2_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store is preserved by the run — in particular every cell of every region other
than `out_ptr`, and the *inactive* lanes of the output window itself. -/
private theorem mul2_kernel_frame
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(out_ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mul2_kernel, ComputeKernel.toAlgKernel,
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
`mul2_kernel_correct` (instantiated at the tile the state actually holds). -/
theorem mul2_kernel_region_run
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((mul2_kernel in_ptr0 out_ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem out_ptr (s₀.pid * BLOCK_SIZE + j.val) = 2 * xs j)
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := mul2_kernel_correct in_ptr0 out_ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem in_ptr0 (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE) s₀
      = exec ((mul2_kernel in_ptr0 out_ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((mul2_kernel in_ptr0 out_ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, mul2_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine mul2_kernel_frame in_ptr0 out_ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked load and the masked store address
the same window `pid * BLOCK_SIZE + j`, active only when `< n_elements`, so the
bounds contract is **lane-wise** — every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem mul2_kernel_traceSafe
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds in_ptr0)
    (hout : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds out_ptr) :
    Kernel.TraceSafe bounds
      ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [mul2_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  exact ⟨fun a ha => hx a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem mul2_kernel_flattenOk
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mul2_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `mul2_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `mul2_kernel` implements lane-wise doubling on its masked
IO signature — for every flat placement of the two buffers, every program id
whose active lanes are in bounds, and every launch state whose input window
holds `xs` at the active lanes, the translated pointer kernel terminates,
every active output lane holds `2 * xs i`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_kernel_flattenOk in_ptr0 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact mul2_kernel_traceSafe in_ptr0 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_kernel_region_run
      in_ptr0 out_ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## `mul2_inplace_kernel` (in-place) -/

/-- Algorithm-layer correctness for `mul2_inplace_kernel`. -/
theorem mul2_inplace_kernel_correct
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s ptr BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (mul2_inplace_kernel ptr n_elements BLOCK_SIZE) s)
          ptr BLOCK_SIZE s.pid i
        = some (if addr < n_elements then 2 * xs i else s.readMem ptr addr) := by
  intro i
  simp [observeAt, exec, mul2_inplace_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked in-place
store is preserved by the run — in particular every cell of every region other
than `ptr`, and the *inactive* lanes of the window itself. -/
private theorem mul2_inplace_kernel_frame
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (s s1 : BlockState)
    (hExec : exec ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < n_elements →
      ¬(ptr = r ∧ s.pid * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** for the in-place kernel —
termination, active-lane output values (the doubled *originally loaded*
window), and frame off the active lanes, from any launch state whose window
is loaded at the **active lanes only**. This is the `hrun` obligation of
`MaskedKernelIO₁.Implements.intro`; the value half reuses
`mul2_inplace_kernel_correct` (instantiated at the tile the state actually
holds — the kernel reads before it scatters, so the before/after reading is
sound). -/
theorem mul2_inplace_kernel_region_run
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem ptr (s₀.pid * BLOCK_SIZE + j.val) = xs j) :
    ∃ s1, exec ((mul2_inplace_kernel ptr n_elements
        BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, s₀.pid * BLOCK_SIZE + j.val < n_elements →
          s1.readMem ptr (s₀.pid * BLOCK_SIZE + j.val) = 2 * xs j)
      ∧ (∀ r o,
          (r ≠ ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pid * BLOCK_SIZE + j.val < n_elements →
              o ≠ s₀.pid * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := mul2_inplace_kernel_correct ptr n_elements BLOCK_SIZE
    s₀ (fun j => s₀.readMem ptr (s₀.pid * BLOCK_SIZE + j.val))
    (fun _ => rfl)
  rw [show exec (mul2_inplace_kernel ptr n_elements BLOCK_SIZE) s₀
      = exec ((mul2_inplace_kernel ptr n_elements
          BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((mul2_inplace_kernel ptr n_elements
      BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, mul2_inplace_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, Tile.bop, Tile.cop, NumericDType.add,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [observeAt, Option.map_some, Option.some_inj, if_pos hj]
          at hje
        rw [hje, hx j hj]
      · refine mul2_inplace_kernel_frame ptr n_elements BLOCK_SIZE
          s₀ s1 hsrc r o (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the in-place kernel: the masked load and the
masked store both address the window `pid * BLOCK_SIZE + j` of `ptr`, active
only when `< n_elements`, so the bounds contract is **lane-wise**. -/
theorem mul2_inplace_kernel_traceSafe
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + j.val < n_elements →
      s.pid * BLOCK_SIZE + j.val < bounds ptr) :
    Kernel.TraceSafe bounds
      ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all seven statements, discharging every
  -- load-free `SafeAt` and reducing the two memory accesses' lane-wise
  -- address obligations to the bounds hypothesis below.
  simp [mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul,
    ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg]
  -- simp dedups the load and store obligations (same buffer, same window):
  -- one lane-wise bound remains.
  exact fun a ha => hx a ha

/-- The in-place kernel sits inside the flat-memory bridge's covered
fragment. -/
theorem mul2_inplace_kernel_flattenOk
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((mul2_inplace_kernel ptr n_elements BLOCK_SIZE
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mul2_inplace_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `mul2_inplace_kernel`'s masked **in-place IO signature**: both argument
roles are wired to the same buffer (`inp = out = ptr`) — the kernel rewrites
the window it read. Windows and mask are the standard launch convention
(`offsets = pid * BLOCK_SIZE + arange; mask = offsets < n_elements`); the
headline **proves** the kernel's actual addressing and masking match them. -/
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE
  inp := ptr
  out := ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- **The headline**: `mul2_inplace_kernel` implements lane-wise doubling on
its masked in-place IO signature — for every flat placement of the buffer,
every program id whose active lanes are in bounds, and every launch state
whose window holds `xs` at the active lanes, the translated pointer kernel
terminates, every active lane of the *same* buffer ends up holding
`2 * xs i` (the doubled originally-loaded value), and every other memory cell
is unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_inplace_kernel_flattenOk ptr n_elements BLOCK_SIZE
  · intro bounds s h1 _ _
    exact mul2_inplace_kernel_traceSafe ptr n_elements BLOCK_SIZE
      bounds s h1
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_inplace_kernel_region_run
      ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.TritonMul2

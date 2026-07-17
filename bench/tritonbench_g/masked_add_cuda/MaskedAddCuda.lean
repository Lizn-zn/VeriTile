import VeriTile.Triton

/-!
# `masked_add_cuda` — strict per-kernel correctness

`masked_add_kernel` is a masked fused multiply-add: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)`, narrows its bounds mask by the boolean
`p_mask` (`mask = mask & ~p_mask`), and for each surviving lane writes
`grad += p * alpha` back to `grad_ptr` (equivalent to
`grad.add_(p.data * (1 - p.mask), alpha)`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`masked_add_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
masked_add_kernel_correctness            ← TOP SPECIFICATION (maskedAddIO ⊨ masked in-place add)
  ├─ masked_add_kernel_flattenOk         bridge fragment membership
  ├─ masked_add_kernel_traceSafe         per-execution lane-wise safety walk
  └─ masked_add_kernel_region_run        region-model masked in-place triple
       ├─ masked_add_kernel_exec_isSome  termination
       ├─ masked_add_kernel_correct      algorithm-layer readback per lane
       └─ masked_add_kernel_frame        masked scatter frame
```

The headline is stated on the kernel's masked **IO signature** `maskedAddIO`
(`Masked2DKernelIO₂ᵦ`, the bool-input family: this kernel loads the boolean
tile `p_mask` via `.to(tl.int1)` and uses it to narrow its own active set).
The signature declares which buffer is which argument — data `p` (`in1`),
running buffer `grad` (`in2`), boolean mask `p_mask` (`mbuf`), and the
**in-place** output `out = in2 = grad_ptr` (duplicate-region wiring, the
`triton_mul2` / `MaskedKernelIO₃ₓ₂` precedent) — the shared window
`pid₀ * BLOCK_SIZE + j`, the **static** bounds mask `offset < n_elements`
(the trace-safety superset), and the data-dependent `writeMask`
`offset < n_elements ∧ bs j = false`: the lanes surviving
`mask = mask & ~p_mask`. The kernel is a 1-D launch, so the second program id
of the 2-D family is simply ignored by every field.

`⊨` (`Masked2DKernelIO₂ᵦ.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the buffers, **every**
program id whose static-mask lanes are in bounds, and **every** launch state
whose windows hold `xs`/`ys`/`bs` at the static-mask lanes, the translated
pointer kernel terminates, every *write-active* lane (in-bounds and
`bs j = false`) of `grad_ptr` holds `ys j + xs j * alpha`, and every other
memory cell — including the in-bounds lanes vetoed by `p_mask` — is
unchanged.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The kernel reads
`grad`/`p` into registers before the in-place scatter back to `grad_ptr`, so
the masked store is correct in place; no extra disjointness side condition is
required. The struct pins the input windows `xs`/`ys`/`bs` at the **static**
bounds mask, a superset of the lanes the kernel actually loads (the ℝ loads
are gated by the narrowed `mask & ~p_mask`); this costs no generality — the
tiles are free variables, so they can always be instantiated with the
buffers' actual contents — it only names values on lanes the conclusion
never mentions.
-/

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂ᵦ

/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}

def maskedAddOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def maskedAddActive
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedAddOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool p_mask_ptr (maskedAddOffset s BLOCK_SIZE i) = Bool.false

instance maskedAddActiveDecidable
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold maskedAddActive
  infer_instance

noncomputable def maskedAddSpec
    (s : BlockState) (grad_ptr p_ptr : RegionName)
    (alpha : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem grad_ptr (maskedAddOffset s BLOCK_SIZE i) +
    s.readMem p_ptr (maskedAddOffset s BLOCK_SIZE i) * alpha

/-- Algorithm-layer correctness for `masked_add_kernel`. -/
theorem masked_add_kernel_correct
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (masked_add_kernel grad_ptr p_ptr p_mask_ptr
          n_elements alpha BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := maskedAddOffset s BLOCK_SIZE i
      s'.readMem grad_ptr outAddr =
        if maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i then
          maskedAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i
        else s.readMem grad_ptr outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, masked_add_kernel, stepStmts, stepStmt, evalOp.eq_def,
        tile_elementwise, Bool.and_eq_true] at hExec
  subst s'
  simp only [maskedAddOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hActive : maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i
  · rcases hActive with ⟨hBounds, hMask⟩
    have hCond :
        s.pids 0 * BLOCK_SIZE + i.val < n_elements ∧
          (if s.pids 0 * BLOCK_SIZE + i.val < n_elements then
            s.readMemValue .bool p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = Bool.false
          else
            BlockState.defaultCarrier .bool = Bool.false) := by
      simp [maskedAddOffset] at hBounds hMask
      simp [hBounds, hMask]
    have hMask' :
        s.readMemValue .bool p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val) = Bool.false := by
      simpa [maskedAddOffset] using hMask
    simp [maskedAddActive, maskedAddSpec, maskedAddOffset, hMask', hCond]
  · simp [maskedAddActive, maskedAddOffset] at hActive ⊢
    by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
    · simp [hBounds] at hActive ⊢
      cases hMask : s.readMemValue .bool p_mask_ptr (s.pid * BLOCK_SIZE + i.val) <;>
        simp [hMask] at hActive ⊢
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
/-- Frame half: every memory cell not actively written by the masked in-place
store — every cell of every region other than `grad_ptr`, and the lanes of the
output window itself that are out of bounds or vetoed by `p_mask` — is
preserved by the run. -/
private theorem masked_add_kernel_frame
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((masked_add_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      s.pids 0 * BLOCK_SIZE + i.val < n_elements →
      s.readMemValue .bool p_mask_ptr (s.pids 0 * BLOCK_SIZE + i.val)
        = Bool.false →
      ¬(grad_ptr = r ∧ s.pids 0 * BLOCK_SIZE + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, masked_add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  obtain ⟨hb1, hb2⟩ := hmk
  rw [if_pos hb1] at hb2
  exact hmiss k.1 hb1 hb2 hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (straight-line
elementwise body, so no `0 < BLOCK_SIZE` side condition is needed). -/
private theorem masked_add_kernel_exec_isSome
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((masked_add_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, masked_add_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, tile_elementwise, ComputeExpr.toAlgorithm?]

/-- **The region-model masked in-place Hoare triple** — termination,
write-active-lane output values, and frame off the write-active output lanes,
from any launch state whose windows hold `xs`/`ys`/`bs` at the static bounds
mask. This is the `hrun` obligation of the `⊨` headline; the value half reuses
`masked_add_kernel_correct` (the kernel reads `grad` before the in-place
scatter, so the before/after reading is sound). -/
theorem masked_add_kernel_region_run
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s₀ : BlockState) (bs : Fin BLOCK_SIZE → Bool) (xs ys : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem p_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem grad_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = ys j)
    (hb : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMemValue .bool p_mask_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = bs j) :
    ∃ s1, exec ((masked_add_kernel grad_ptr p_ptr p_mask_ptr
          n_elements alpha BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE,
          s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.false →
          s1.readMem grad_ptr (s₀.pids 0 * BLOCK_SIZE + j.val)
            = ys j + xs j * alpha)
      ∧ (∀ r o,
          (r ≠ grad_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.false →
              o ≠ s₀.pids 0 * BLOCK_SIZE + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := masked_add_kernel_exec_isSome grad_ptr p_ptr p_mask_ptr
    n_elements alpha BLOCK_SIZE s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have hActive : maskedAddActive s₀ p_mask_ptr n_elements BLOCK_SIZE j :=
      ⟨hj.1, (hb j hj.1).trans hj.2⟩
    have h := masked_add_kernel_correct grad_ptr p_ptr p_mask_ptr
      n_elements alpha BLOCK_SIZE s₀ s1 hs1 j
    simp only [maskedAddOffset, maskedAddSpec, BlockState.pid,
      if_pos hActive] at h
    rw [h, hx j hj.1, hy j hj.1]
  · refine masked_add_kernel_frame grad_ptr p_ptr p_mask_ptr n_elements alpha
      BLOCK_SIZE s₀ s1 hs1 r o (fun i hbounds hmaskv ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i ⟨hbounds, (hb i hbounds).symm.trans hmaskv⟩ ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — six are memory-silent (`program_id`, the block-start arithmetic,
`arange`, the two mask computations, the fused register update) — and reduces
the four masked accesses (the `p_mask` bool load at the **static** bounds
mask, the `p`/`grad` loads and the `grad` store at the **narrowed**
`mask & ~p_mask`, lane `j` at `pid * BLOCK_SIZE + j`) to the lane-wise bounds
hypotheses at the static mask: narrowed-active lanes are in particular
in-bounds, so the static hypotheses cover all four (simp dedups the `grad`
load and store — same buffer, same window — into one obligation, so only
three hypotheses are needed). -/
theorem masked_add_kernel_traceSafe
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds grad_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds p_mask_ptr) :
    Kernel.TraceSafe bounds
      ((masked_add_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [masked_add_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => h3 a ha, fun a ha _ => h1 a ha, fun a ha _ => h2 a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads — one through the `.to(tl.int1)` bool cast — and the
masked in-place store). -/
theorem masked_add_kernel_flattenOk
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ((masked_add_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [masked_add_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `masked_add_kernel`'s masked **in-place IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`mbuf` — which buffer is which argument (data `p_ptr`, running
  buffer `grad_ptr`, boolean veto `p_mask_ptr`);
* `out = in2 = grad_ptr` — the update is **in-place** (duplicate-region
  wiring, the `triton_mul2` precedent): the triple reads the *old* window
  contents into `ys` and asserts the *new* ones;
* `B = BLOCK_SIZE`, all four windows at lane `j` = `pid₀ * BLOCK_SIZE + j`
  (the host-side 1-D `cdiv(n_elements, BLOCK_SIZE)` launch convention; the
  family's second program id is ignored);
* `mask` — the **static** bounds lanes `pid₀ * BLOCK_SIZE + j < n_elements`
  (the trace-safety/input superset);
* `writeMask` — the **data-dependent** store gate
  `pid₀ * BLOCK_SIZE + j < n_elements ∧ bs j = false`: exactly the kernel's
  `mask = mask & ~p_mask` narrowing.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and `p_mask` narrowing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the static-mask lanes. -/
def maskedAddIO (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    Masked2DKernelIO₂ᵦ where
  kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
    BLOCK_SIZE
  in1 := p_ptr
  in2 := grad_ptr
  mbuf := p_mask_ptr
  out := grad_ptr
  B := BLOCK_SIZE
  read1 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  read2 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readm := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  write := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  mask := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < n_elements
  writeMask := fun pid₀ _ bs j =>
    pid₀ * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.false

/-- **The headline**: `masked_add_kernel` implements the masked in-place fused
add on its bool-input IO signature — for every disjoint flat placement of the
buffers, every program id whose static-mask lanes are in bounds, and every
launch state whose `p`/`grad`/`p_mask` windows hold `xs`/`ys`/`bs` at the
static-mask lanes, the translated pointer kernel terminates, every
write-active lane `j` (in-bounds and `bs j = false`, the kernel's
`mask & ~p_mask`) of the *same* buffer `grad_ptr` ends up holding
`ys j + xs j * alpha` (the fused update of the originally-loaded window), and
every other memory cell — including the in-bounds lanes vetoed by `p_mask` —
is unchanged. Proof: `Masked2DKernelIO₂ᵦ.Implements.intro` assembles the
region-model masked in-place triple with the flat-memory bridge side
conditions. -/
specification masked_add_kernel_correctness
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    maskedAddIO grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
      ⊨ fun _ _ _ xs ys j => ys j + xs j * alpha := by
  refine Masked2DKernelIO₂ᵦ.Implements.intro _ ?_ ?_ ?_ ?_
  · exact masked_add_kernel_flattenOk grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE
  · exact fun _ _ _ j h => h.1
  · intro bounds s h1 h2 h3 _
    exact masked_add_kernel_traceSafe grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ bs xs ys hx hy hb
    exact masked_add_kernel_region_run grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE s₀ bs xs ys hx hy hb

end VeriTile.Bench.TritonBenchG.MaskedAddCuda

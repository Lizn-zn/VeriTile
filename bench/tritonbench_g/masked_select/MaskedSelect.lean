import VeriTile.Triton

/-!
# `masked_select` — strict per-kernel correctness

`masked_select_kernel` is a compaction scatter: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of the input, the boolean select mask, and
a precomputed exclusive-prefix-sum of the mask; each selected lane scatters its
input value to `out_ptr[prefix_sum − 1]`, i.e. its compacted output slot.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`masked_select_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, the host-side `prefix_sum`/output allocation, and
how the runtime composes per-program writes) is the *trusted boundary*, not a
proof obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

The headline is stated on the kernel's compaction-scatter IO signature
`maskedSelectIO` (`BoolScatterMasked2DKernelIO₁`, the scatter genre's bool-gated
skin): which buffer is which argument — data `inp_ptr` (`inp`), select gate
`select_mask_ptr` (`mbuf`, `.bool` channel), prefix-sum `prefix_sum_ptr`
(`idxbuf`, `.nat` index channel), and the scatter target `out_ptr` (`out`) —
the shared read window `pid₀ * BLOCK_SIZE + j`, the **static** bounds mask
`offset < n_elements` (the trace-safety/input superset), the data-dependent
`writeMask` `offset < n_elements ∧ bs j = Bool.true` (the kernel's
`select_mask and mask` store gate), and the data-dependent scatter destination
`write … ids j = ids j − 1` (the loaded `prefix_sum` value minus one). `⊨`
(`BoolScatterMasked2DKernelIO₁.Implements`) is the audit-once scatter Hoare-triple
combinator: for **every** disjoint flat placement of the four buffers, **every**
program id whose static-mask lanes are in bounds, and **every** launch state
whose windows hold `xs`/`bs`/`ids` at the static-mask lanes, the translated
pointer kernel terminates, the readback leg — guarded by the per-context
`WriteInj` antecedent (no two write-active lanes share a destination, the host
prefix-sum's no-duplicate-destination guarantee) — puts `xs j` at every
write-active lane's scatter cell `ids j − 1`, and every memory cell off the raw
scatter cells is unchanged (the frame holds with or without injectivity).

## Proof architecture

```
masked_select_kernel_correctness               ← TOP SPECIFICATION (maskedSelectIO ⊨ compaction scatter)
  ├─ masked_select_kernel_flattenOk             bridge fragment membership
  ├─ masked_select_kernel_traceSafe             per-execution lane-wise safety walk
  └─ masked_select_kernel_region_run            region-model scatter Hoare triple
       ├─ masked_select_kernel_exec_isSome      termination
       ├─ masked_select_kernel_correct_of_exec  executed-state readback per active lane
       │    └─ masked_select_kernel_correct     algorithm-layer readback per active lane
       └─ masked_select_kernel_frame            scatter-store cell frame
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The destination offset is data-dependent (read from `prefix_sum_ptr`),
so the readback leg of the `⊨` headline is guarded by the struct's per-context
`WriteInj` antecedent — no two **write-active** (in-bounds and selected) lanes
share a destination. This is deliberately weaker than injectivity over *all*
lanes: for a genuine prefix sum, an unselected lane's `prefix_sum − 1` equals
the previous selected lane's destination, so full-lane injectivity is false —
only the write-active restriction captures the host guarantee. The
`.to(tl.int1)` mask cast reduces to identity at the algorithm layer. The frame
and the trace-safety write bound are stated at the *ungated* write-active lanes,
so they hold with or without injectivity.
-/

namespace VeriTile.Bench.TritonBenchG.MaskedSelect

open VeriTile.Triton
open scoped VeriTile.Triton.BoolScatterMasked2DKernelIO₁

/-- Faithful transcription of `masked_select.py`'s `masked_select_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- `select_mask_ptr` and `prefix_sum_ptr` are typed Lean regions so their
  `tl.load` calls do not need extra `dtype=` kwargs. -/
def masked_select_kernel
    (inp_ptr : RegionName) (select_mask_ptr : Region .bool)
    (prefix_sum_ptr : Region .nat) (out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
  select_mask = tl.load(select_mask_ptr + offsets,
    mask=mask, other=0.0).to(tl.int1)
  out_offset = tl.load(prefix_sum_ptr + offsets,
    mask=mask, other=0.0) - $(1)
  tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
}

/-- Lane `i`'s address in all three read windows of program `s.pid`. Internal
to the readback lemmas (the `⊨` signature declares the same window as
`fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val`). -/
def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Lane `i`'s state-coupled scatter destination: the masked `prefix_sum` load
minus one (the kernel's `out_offset`). Internal to the readback lemmas; the `⊨`
signature re-expresses it as `ids j − 1` in terms of the pinned index tile. -/
def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  (if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1

/-- Lane `i` is **write-active**: in bounds and selected — the state-coupled
form of the signature's `writeMask`. -/
def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true

instance activeDecidable
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s select_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold active
  infer_instance

/-- Algorithm-layer cellwise correctness for active masked-select lanes, under
the write-active-lanes-only no-collision hypothesis `hOutInj` (the state-coupled
form of the signature's `WriteInj`; full-lane injectivity would be false for
genuine prefix sums, whose unselected lanes repeat the previous destination). -/
theorem masked_select_kernel_correct
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : ∀ i k : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      active s select_mask_ptr n_elements BLOCK_SIZE k →
      maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i
        = maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE k →
      i = k) :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      (exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
          n_elements BLOCK_SIZE) s).map
          (fun s' => s'.readMem out_ptr
            (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i))
        = some (s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  intro i hActive
  simp [exec, masked_select_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, tile_elementwise,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map]
  simp [maskedSelectOffset, maskedSelectStoreOffset, BlockState.readMemValue]
  rcases hActive with ⟨hBounds, hMask⟩
  have hBoundsRaw : s.pid * BLOCK_SIZE + i.val < n_elements := by
    simpa [maskedSelectOffset] using hBounds
  have hMaskRaw :
      s.readMemValue .bool select_mask_ptr (s.pid * BLOCK_SIZE + i.val) = Bool.true := by
    simpa [maskedSelectOffset] using hMask
  have hMaskMatch :
      (match s.readMemTyped TileDType.bool select_mask_ptr
          (s.pid * BLOCK_SIZE + i.val) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.bool) = Bool.true := by
    simpa [BlockState.readMemValue] using hMaskRaw
  have hcol : ∀ k : TileIndex [BLOCK_SIZE],
      ((s.pid * BLOCK_SIZE + k.1.val < n_elements ∧
          (match s.readMemTyped TileDType.bool select_mask_ptr
              (s.pid * BLOCK_SIZE + k.1.val) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.bool) = Bool.true) ∧
        s.pid * BLOCK_SIZE + k.1.val < n_elements) →
      (if s.pid * BLOCK_SIZE + k.1.val < n_elements then
          (match s.readMemTyped TileDType.nat prefix_sum_ptr
              (s.pid * BLOCK_SIZE + k.1.val) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat)
        else
          0) - 1
        = (if s.pid * BLOCK_SIZE + i.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + i.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1 →
      k = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]) := by
    rintro ⟨a, _⟩ ⟨⟨hbA, hmA⟩, _⟩ hoff
    have haA : active s select_mask_ptr n_elements BLOCK_SIZE a := by
      constructor
      · simpa [maskedSelectOffset] using hbA
      · simpa [maskedSelectOffset, BlockState.readMemValue] using hmA
    have hASO : maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE a
        = maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i := by
      simpa [maskedSelectStoreOffset, maskedSelectOffset,
        BlockState.readMemValue] using hoff
    have hai : a = i := hOutInj a i haA ⟨hBounds, hMask⟩ hASO
    subst hai
    rfl
  simpa [hBoundsRaw, hMaskRaw, hMaskMatch] using
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := out_ptr)
      (shape := [BLOCK_SIZE])
      (offsetFn := fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1)
      (valueFn := fun idx : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            some (s.readMem inp_ptr (s.pid * BLOCK_SIZE + idx.1.val))
          else
            some 0.0))
      (P := fun idx : TileIndex [BLOCK_SIZE] =>
        (s.pid * BLOCK_SIZE + idx.1.val < n_elements ∧
            (match s.readMemTyped TileDType.bool select_mask_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.bool) = Bool.true) ∧
          s.pid * BLOCK_SIZE + idx.1.val < n_elements)
      _ (i, PUnit.unit) ⟨⟨hBoundsRaw, hMaskMatch⟩, hBoundsRaw⟩ hcol

/-- Executed-state form of `masked_select_kernel_correct`. -/
theorem masked_select_kernel_correct_of_exec
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : ∀ i k : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      active s select_mask_ptr n_elements BLOCK_SIZE k →
      maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i
        = maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE k →
      i = k)
    (s' : BlockState)
    (hExec : exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      s'.readMem out_ptr
          (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)
        = s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i) := by
  intro i hActive
  have h := masked_select_kernel_correct inp_ptr select_mask_ptr prefix_sum_ptr
    out_ptr n_elements BLOCK_SIZE s hOutInj i hActive
  rw [hExec] at h
  simpa using h

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
/-- Frame half: every memory cell not actively hit by the compaction scatter —
every cell of every region other than `out_ptr`, and every `out_ptr` cell off
the write-active lanes' scatter destinations — is preserved by the run.
Unconditional in the injectivity: colliding raw scatter cells are still scatter
cells, so the exclusion set needs no `WriteInj`. -/
private theorem masked_select_kernel_frame
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr
        out_ptr n_elements BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      ¬(out_ptr = r ∧
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, masked_select_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, tile_elementwise,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]
    at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  obtain ⟨⟨hb1, hb2⟩, _⟩ := hmk
  refine hmiss k.1 ⟨?_, ?_⟩ ⟨hc.1, ?_⟩
  · simpa [maskedSelectOffset] using hb1
  · simpa [maskedSelectOffset, BlockState.readMemValue] using hb2
  · simpa [maskedSelectStoreOffset, maskedSelectOffset,
      BlockState.readMemValue] using hc.2

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (straight-line
masked loads plus one masked scatter store). -/
private theorem masked_select_kernel_exec_isSome
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr
        out_ptr n_elements BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, masked_select_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, tile_elementwise,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt]

/-- **The region-model scatter Hoare triple** — termination, write-active-lane
scatter values under the per-context no-collision antecedent, and frame off the
raw scatter cells, from any launch state whose windows hold `xs`/`bs`/`ids` at
the static bounds mask. This is the `hrun` obligation of
`BoolScatterMasked2DKernelIO₁.Implements.intro`; the value half reuses
`masked_select_kernel_correct_of_exec` (whose state-coupled store offset
collapses to `ids j − 1` under the index-tile pin). -/
theorem masked_select_kernel_region_run
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s₀ : BlockState) (bs : Fin BLOCK_SIZE → Bool) (ids : Fin BLOCK_SIZE → Nat)
    (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMem inp_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = xs j)
    (hb : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMemValue .bool select_mask_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = bs j)
    (hi : ∀ j : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s₀.readMemValue .nat prefix_sum_ptr (s₀.pids 0 * BLOCK_SIZE + j.val) = ids j) :
    ∃ s1, exec ((masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr
          out_ptr n_elements BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin BLOCK_SIZE,
            s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.true →
            s₀.pids 0 * BLOCK_SIZE + k.val < n_elements ∧ bs k = Bool.true →
            ids j - 1 = ids k - 1 → j = k) →
          ∀ j : Fin BLOCK_SIZE,
            s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.true →
            s1.readMem out_ptr (ids j - 1) = xs j)
      ∧ (∀ r o,
          (r ≠ out_ptr ∨ ∀ j : Fin BLOCK_SIZE,
            s₀.pids 0 * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.true →
              o ≠ ids j - 1) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := masked_select_kernel_exec_isSome inp_ptr select_mask_ptr
    prefix_sum_ptr out_ptr n_elements BLOCK_SIZE s₀
  have hso : ∀ a : Fin BLOCK_SIZE, s₀.pids 0 * BLOCK_SIZE + a.val < n_elements →
      maskedSelectStoreOffset s₀ prefix_sum_ptr n_elements BLOCK_SIZE a
        = ids a - 1 := by
    intro a ha
    simp [maskedSelectStoreOffset, maskedSelectOffset, ha, hi a ha]
  refine ⟨s1, hs1, fun hinj j hj => ?_, fun r o hcond => ?_⟩
  · have hActive : active s₀ select_mask_ptr n_elements BLOCK_SIZE j :=
      ⟨hj.1, (hb j hj.1).trans hj.2⟩
    have hOutInj : ∀ a c : Fin BLOCK_SIZE,
        active s₀ select_mask_ptr n_elements BLOCK_SIZE a →
        active s₀ select_mask_ptr n_elements BLOCK_SIZE c →
        maskedSelectStoreOffset s₀ prefix_sum_ptr n_elements BLOCK_SIZE a
          = maskedSelectStoreOffset s₀ prefix_sum_ptr n_elements BLOCK_SIZE c →
        a = c := by
      intro a c ha hc hoff
      have ha' : s₀.pids 0 * BLOCK_SIZE + a.val < n_elements := ha.1
      have hc' : s₀.pids 0 * BLOCK_SIZE + c.val < n_elements := hc.1
      refine hinj a c ⟨ha', (hb a ha').symm.trans ha.2⟩
        ⟨hc', (hb c hc').symm.trans hc.2⟩ ?_
      rw [← hso a ha', ← hso c hc']
      exact hoff
    have h := masked_select_kernel_correct_of_exec inp_ptr select_mask_ptr
      prefix_sum_ptr out_ptr n_elements BLOCK_SIZE s₀ hOutInj s1 hs1 j hActive
    rw [← hso j hj.1, h]
    exact hx j hj.1
  · refine masked_select_kernel_frame inp_ptr select_mask_ptr prefix_sum_ptr
      out_ptr n_elements BLOCK_SIZE s₀ s1 hs1 r o ?_
    intro a hActA hc
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · have ha' : s₀.pids 0 * BLOCK_SIZE + a.val < n_elements := hActA.1
      exact hno a ⟨ha', (hb a ha').symm.trans hActA.2⟩
        (hc.2.symm.trans (hso a ha'))

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all seven
statements — three are memory-silent (`program_id`, the window arithmetic, the
bounds-mask comparison) — and reduces the three masked loads (at the **static**
bounds mask) and the masked scatter store (at the loaded-`select_mask`-narrowed
gate, address = loaded `prefix_sum` value − 1) to the four lane-wise bounds
hypotheses. -/
theorem masked_select_kernel_traceSafe
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds inp_ptr)
    (h2 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds select_mask_ptr)
    (h3 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.pids 0 * BLOCK_SIZE + j.val < bounds prefix_sum_ptr)
    (h4 : ∀ j : Fin BLOCK_SIZE, s.pids 0 * BLOCK_SIZE + j.val < n_elements →
      s.readMemValue .bool select_mask_ptr (s.pids 0 * BLOCK_SIZE + j.val)
        = Bool.true →
      s.readMemValue .nat prefix_sum_ptr (s.pids 0 * BLOCK_SIZE + j.val) - 1
        < bounds out_ptr) :
    Kernel.TraceSafe bounds
      ((masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [masked_select_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
    BlockState.readMemValue]
  refine ⟨fun a ha => h1 a ha, fun a ha => h2 a ha, fun a ha => h3 a ha, ?_⟩
  intro a ha hsel _
  simpa [ha, BlockState.readMemValue, BlockState.readMemTyped] using h4 a ha hsel

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads — one through the `.to(tl.int1)` bool cast, one on the
`.nat` index channel — and the data-dependent masked scatter store). -/
theorem masked_select_kernel_flattenOk
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ((masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [masked_select_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `masked_select_kernel`'s compaction-scatter **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`mbuf`/`idxbuf`/`out` — which buffer is which argument (data
  `inp_ptr`, `.bool` select gate `select_mask_ptr`, `.nat` prefix-sum index
  channel `prefix_sum_ptr`, scatter target `out_ptr`);
* `B = BLOCK_SIZE`, all three read windows at lane `j` =
  `pid₀ * BLOCK_SIZE + j` (the host-side 1-D `cdiv(n_elements, BLOCK_SIZE)`
  launch convention; the family's second program id is ignored);
* `mask` — the **static** bounds lanes `pid₀ * BLOCK_SIZE + j < n_elements`
  (the trace-safety/input superset);
* `writeMask` — the **data-dependent** store gate
  `pid₀ * BLOCK_SIZE + j < n_elements ∧ bs j = Bool.true`: exactly the kernel's
  `select_mask and mask`;
* `write` — the **data-dependent** scatter destination `ids j − 1`: the loaded
  `prefix_sum` value minus one, i.e. lane `j`'s compacted output slot.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and prefix-sum indexing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the static-mask lanes and the
write-active scatter cells. -/
def maskedSelectIO (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : BoolScatterMasked2DKernelIO₁ where
  kernel := masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
    n_elements BLOCK_SIZE
  inp := inp_ptr
  mbuf := select_mask_ptr
  idxbuf := prefix_sum_ptr
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readm := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readx := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  write := fun _ _ ids j => ids j - 1
  mask := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < n_elements
  writeMask := fun pid₀ _ bs _ j =>
    pid₀ * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.true

/-- **The headline**: `masked_select_kernel` implements the value-preserving
compaction scatter on its scatter IO signature — for every disjoint flat
placement of the four buffers, every program id whose static-mask lanes are in
bounds (the write bound at the scatter cells `ids j − 1`), and every launch
state whose windows hold the data tile `xs`, the select tile `bs` (`.bool`
channel), and the prefix-sum tile `ids` (`.nat` channel) at the static-mask
lanes, the translated pointer kernel terminates; under the per-context
`WriteInj` antecedent (no two write-active lanes share a destination — the
host prefix-sum's no-duplicate-destination guarantee, the successor of the old
summary's `hOutInj` side condition, now demanded only at the write-active
lanes where it is actually true of a genuine prefix sum), every write-active
lane `j` (in-bounds and `bs j = Bool.true`, the kernel's `select_mask and mask`)
put `xs j` at its compacted slot `ids j − 1` of `out_ptr`; and every memory
cell off the raw scatter cells is unchanged — unconditionally. Proof:
`BoolScatterMasked2DKernelIO₁.Implements.intro` assembles the region-model
scatter triple with the flat-memory bridge side conditions. -/
specification masked_select_kernel_correctness
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    maskedSelectIO inp_ptr select_mask_ptr prefix_sum_ptr out_ptr n_elements
      BLOCK_SIZE ⊨ fun _ _ _ _ xs j => xs j := by
  refine BoolScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact masked_select_kernel_flattenOk inp_ptr select_mask_ptr prefix_sum_ptr
      out_ptr n_elements BLOCK_SIZE
  · intro bounds s bs ids hbsPin hidsPin hbr hbm hbx hbw
    simp only [maskedSelectIO] at hbsPin hidsPin hbr hbm hbx hbw ⊢
    refine masked_select_kernel_traceSafe inp_ptr select_mask_ptr prefix_sum_ptr
      out_ptr n_elements BLOCK_SIZE bounds s hbr hbm hbx ?_
    intro j hj hsel
    have hw := hbw j ⟨hj, (hbsPin j hj).symm.trans hsel⟩
    rw [hidsPin j hj]
    exact hw
  · intro s₀ bs ids xs hx hb hi
    exact masked_select_kernel_region_run inp_ptr select_mask_ptr prefix_sum_ptr
      out_ptr n_elements BLOCK_SIZE s₀ bs ids xs hx hb hi

end VeriTile.Bench.TritonBenchG.MaskedSelect

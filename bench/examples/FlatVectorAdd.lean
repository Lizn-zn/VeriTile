/-
bench/examples/FlatVectorAdd

**The canonical showcase of the masked KernelIO spec surface**: the
boundary-checked (masked) elementwise vector add, verified end to end.
`addKernelMasked` is the ordinary register-indirect DSL shape every bench
port uses (`offsets := ...; tl.load(x + offsets, mask=offsets < n)`), which
no ∀-state safety contract can cover — the flat-memory bridge (v1.2) takes
the per-execution `Kernel.TraceSafe` contract instead. Four parts:

1. **The kernel** — `addKernelMasked`, the Triton DSL transcription of the
   canonical tutorial kernel, mask lines included.
2. **Region-model masked Hoare triple** — `addKernelMasked_region_run`:
   from any launch state whose input windows are loaded **at the active
   lanes only** (`pid * B + j < n`), the kernel terminates, every active
   output-window lane holds `xs j + ys j`, and every other memory cell —
   including the inactive lanes of the output window — is untouched.
   Inactive lanes carry no obligations on either side of the triple: the
   masked loads read the undef channel there, not memory, so requiring
   inputs at inactive lanes (whose flat addresses may overhang the buffer
   or land in the next buffer) would be nonsense.
3. **Flat-memory bridge side conditions** — `TraceSafe` (the per-execution
   safety walk, discharged by walking the actual seven-statement execution
   from the **lane-wise** bounds: every *active* lane's address is below
   the region bound) and `FlattenOk` (bridge fragment membership). They
   license the transport of Part 2 to real pointer arithmetic.
4. **The spec** — the file's single `specification`:

       add_kernel_masked_correctness :
         addMaskedIO B n ⊨ fun xs ys i => xs i + ys i

   `addMaskedIO` is the kernel's masked **IO signature**: which buffer is
   which argument, where each program reads/writes its window, and the
   active-lane predicate `mask pid j := pid * B + j < n`. `⊨` is the
   audit-once masked Hoare-triple combinator (`MaskedKernelIO₂.Implements`,
   `VeriTile.Triton.Memory.KernelSpec`); spelled out, the headline says:
   for **every** disjoint placement of the three buffers in flat memory
   (∀ base pointers, ∀ buffer sizes), **every** program id all of whose
   *active* lanes are in bounds (partial blocks may overhang the buffer on
   their inactive lanes), and **every** launch state whose input windows
   hold `xs`/`ys` at the active lanes — all other buffer cells and all
   registers arbitrary — the translated pointer kernel terminates, every
   active output lane holds the pointwise sum, and every other memory cell
   is unchanged.

The aligned (unmasked) sibling `addKernel` lives in
`bench/examples/VectorAdd.lean` — each showcased kernel is self-contained
in its showcase file.

Source Triton (`.py` reference — the canonical tutorial kernel, with the
mask lines this file is about):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid     = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements
    x       = tl.load(x_ptr + offsets, mask=mask)
    y       = tl.load(y_ptr + offsets, mask=mask)
    output  = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
```
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FlatVectorAdd

open VeriTile.Triton
open VeriTile.Triton.MaskedKernelIO₂ (Implements)
open scoped VeriTile.Triton.MaskedKernelIO₂
open VeriTile.Examples (observeAt)

/-! ## Part 1 — the kernel -/

/-- Masked elementwise add. Lanes where `pid * blockSize + i < nElements` are
    loaded, summed, and stored. Lanes outside the bound get Triton's
    `other=None` undefined load value, but the store mask skips those lanes,
    so the undefined values are not observed. -/
def addKernelMasked (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask    := offsets < $(nElements)
  x       := tl.load($(xReg) + offsets, mask=mask)
  y       := tl.load($(yReg) + offsets, mask=mask)
  output  := x + y
  tl.store($(outReg) + offsets, output, mask=mask)
}

/-! ## Part 2 — the region-model masked Hoare triple

The mathematical core, proved against the region model (named buffers, no
pointer arithmetic): a value lemma (`add_kernel_masked_correct`), a frame
lemma (`addKernelMasked_frame`, via the masked scatter-store `foldl`), and
their package `addKernelMasked_region_run` — exactly the `hrun` obligation
of `MaskedKernelIO₂.Implements.intro`. The input hypotheses are
**active-lane only**: the masked load's inactive lanes read the undef
channel, not memory, so the evaluation never consults them. -/

/-- Value half: from any state whose **active** input lanes are loaded,
each lane reads back as follows —

* active (`pid * blockSize + i < nElements`): the output region holds
  `xs i + ys i` at `pid * blockSize + i`;
* inactive: the output region's value at `pid * blockSize + i` is preserved
  from the initial state (mask=false → no store).

No region-disjointness hypothesis: the kernel reads `x` and `y` into
local registers BEFORE the scatter to `outReg`, so even if `outReg`
aliases `xReg` or `yReg`, the result is correct. -/
theorem add_kernel_masked_correct
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : ∀ j : Fin blockSize, s.pid * blockSize + j.val < nElements →
      s.readMem xReg (s.pid * blockSize + j.val) = xs j)
    (h_y : ∀ j : Fin blockSize, s.pid * blockSize + j.val < nElements →
      s.readMem yReg (s.pid * blockSize + j.val) = ys j) :
    ∀ i : Fin blockSize,
      observeAt (exec (addKernelMasked xReg yReg outReg blockSize nElements) s)
                outReg blockSize s.pid i
        = some (if s.pid * blockSize + i.val < nElements then xs i + ys i
                else s.readMem outReg (s.pid * blockSize + i.val)) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernelMasked, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * blockSize + i.val < nElements
  · simp [hi, h_x i hi, h_y i hi]
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
store is preserved by the run — in particular every cell of every region
other than `outReg`, and the *inactive* lanes of the output window itself. -/
private theorem addKernelMasked_frame (xReg yReg outReg : RegionName)
    (B n : Nat) (s s1 : BlockState)
    (hExec : exec ((addKernelMasked xReg yReg outReg B n
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, s.pid * B + i.val < n →
      ¬(outReg = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, addKernelMasked, ComputeKernel.toAlgKernel,
    stepStmts, stepStmt, evalOp, Tile.bop, Tile.cop, NumericDType.add,
    NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane
output values, and frame off the active output lanes, from any launch state
whose input windows are loaded at the **active lanes only**. This is what
the `⊨` headline transports to flat memory. -/
theorem addKernelMasked_region_run (B n : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs ys : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.pid * B + j.val < n →
      s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val) = xs j)
    (hy : ∀ j : Fin B, s₀.pid * B + j.val < n →
      s₀.readMem ⟨"y"⟩ (s₀.pid * B + j.val) = ys j) :
    ∃ s1, exec ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin B, s₀.pid * B + j.val < n →
          s1.readMem ⟨"out"⟩ (s₀.pid * B + j.val) = xs j + ys j)
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, s₀.pid * B + j.val < n →
            o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_masked_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n hB s₀ xs ys
    hx hy
  rw [show exec (addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n) s₀
      = exec ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n).toAlgKernel) s₀
      with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have := hobs j
        rw [hsrc] at this
        simpa [observeAt, hj] using this
      · refine addKernelMasked_frame ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

/-! ## Part 3 — flat-memory bridge side conditions

`addKernelMasked` is register-indirect (`offsets := ...;
tl.load(x + offsets, mask=offsets < n)`), so no ∀-state contract covers it;
the flat-memory bridge (v1.2) takes the per-execution `Kernel.TraceSafe`
contract instead, discharged below by walking the actual seven-statement
execution, reading the offset/mask registers back at each load and at the
store. Because the kernel is **masked**, only the active lanes touch memory,
and the bounds obligation is **lane-wise**: each active lane's address is
below the region bound — the active lane's bound comes directly from the
lane-wise hypothesis at that lane, with no whole-window `n ≤ bounds`
contract in sight. -/

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

set_option maxHeartbeats 1600000 in
/-- The register-style masked vector add is trace-safe: each of its seven
statements is safe in the state the execution actually reaches. Masked
loads/store touch only the active lanes, so the bounds contract is
lane-wise: every active lane `pid * B + j < n` has its address below the
region bound. -/
theorem addKernelMasked_traceSafe (xReg yReg outReg : RegionName)
    (B n : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : ∀ j : Fin B, s.pid * B + j.val < n →
      s.pid * B + j.val < bounds xReg)
    (hy : ∀ j : Fin B, s.pid * B + j.val < n →
      s.pid * B + j.val < bounds yReg)
    (hout : ∀ j : Fin B, s.pid * B + j.val < n →
      s.pid * B + j.val < bounds outReg) :
    Kernel.TraceSafe bounds
      ((addKernelMasked xReg yReg outReg B n).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: offsets := pid * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
      (Op.arange B))
      (s.setReg "pid" .nat [] (Tile.scalar (dtype := .nat) (s.pids 0)))
      = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- statement 3: mask := offsets < n
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  rw [show evalOp (Op.lt .nat .scalarR (Op.ref .nat [B] "offsets")
      (Op.constNat n))
      ((s.setReg "pid" .nat []
          (Tile.scalar (dtype := .nat) (s.pids 0))).setReg "offsets" .nat [B]
        ⟨fun i => s.pids 0 * B + i.1.val⟩)
      = some ⟨fun i => ComparableDType.nat.lt (s.pids 0 * B + i.1.val) n⟩
      from by simp [Tile.cop, BlockState.setReg]] at hv3
  obtain rfl := Option.some_inj.mp hv3
  -- shared register-readback facts at the state before the loads
  set sm := ((s.setReg "pid" .nat []
      (Tile.scalar (dtype := .nat) (s.pids 0))).setReg "offsets" .nat [B]
      ⟨fun i => s.pids 0 * B + i.1.val⟩).setReg "mask" .bool [B]
      ⟨fun i => ComparableDType.nat.lt (s.pids 0 * B + i.1.val) n⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "offsets" ∨ nm = "mask" →
        t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (reg : RegionName),
      (∀ j : Fin B, s.pid * B + j.val < n → s.pid * B + j.val < bounds reg) →
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (Op.ref .nat [B] "offsets")) t
        ((MaskOpt.mask (dtype := .real) (Op.ref .bool [B] "mask")).Active t) := by
    intro t hframe reg hreg
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i hact
    rw [show evalOp (Op.ref .nat [B] "offsets") t
        = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
      simp [hframe .nat [B] "offsets" (Or.inl rfl), hsm,
        BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    obtain ⟨masks, hm, hd⟩ := hact
    rw [show evalOp (Op.ref .bool [B] "mask") t
        = some ⟨fun i =>
          ComparableDType.nat.lt (s.pids 0 * B + i.1.val) n⟩ from by
      simp [hframe .bool [B] "mask" (Or.inr rfl), hsm,
        BlockState.setReg]] at hm
    obtain rfl := Option.some_inj.mp hm
    rw [ComparableDType.nat_lt_eq_true] at hd
    simp only [Region.cast_self]
    exact hreg i.1 hd
  -- statement 4: x := load(xReg + offsets, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · refine ?_
    simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, by simp,
      hAAS sm (fun _ _ _ _ => rfl) xReg hx⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: y := load(yReg + offsets, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp, by simp,
      hAAS _ (fun dt sh nm hnm => ?_) yReg hy⟩
    rcases hnm with rfl | rfl <;> simp [BlockState.setReg]
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: output := x + y
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(outReg + offsets, output, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], by simp [Op.SafeAt],
    hAAS _ (fun dt sh nm hnm => ?_) outReg hout⟩
  rcases hnm with rfl | rfl <;> simp [BlockState.setReg]

/-- The register-style kernel sits inside the bridge's covered fragment. -/
theorem addKernelMasked_flattenOk (xReg yReg outReg : RegionName)
    (B n : Nat) :
    ((addKernelMasked xReg yReg outReg B n
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [addKernelMasked, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-! ## Part 4 — the spec: `addMaskedIO ⊨` pointwise addition -/

/-- `addKernelMasked`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring);
* `B` — the window length each program owns;
* `read1`/`read2` — where program `pid` reads its input tiles (the address
  half of the *pre*condition);
* `write` — where program `pid` writes its output tile (the address half
  of the *post*condition: values land there, frame holds everywhere else);
* `mask` — program `pid`'s **active lanes**, `pid * B + j < n`: the lanes
  that read, write, and carry spec content. Inactive lanes (the overhang of
  the last partial block) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel: they
formalize the host-side launch convention
(`offsets = pid * BLOCK_SIZE + arange; mask = offsets < n_elements`), and
the headline **proves** the kernel's actual addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the active lanes. -/
def addMaskedIO (B n : Nat) : MaskedKernelIO₂ where
  kernel := addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

/-- **The headline**: `addKernelMasked` implements pointwise addition on its
masked IO signature — see the module docstring for the full masked Hoare
triple `⊨` unfolds to. Proof: `Implements.intro` assembles the region-model
masked triple (Part 2) with the bridge side conditions (Part 3). -/
specification add_kernel_masked_correctness (B n : Nat) (hB : 0 < B) :
    addMaskedIO B n ⊨ fun xs ys i => xs i + ys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact addKernelMasked_flattenOk ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
  · intro bounds s h1 h2 h3
    exact addKernelMasked_traceSafe ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    exact addKernelMasked_region_run B n hB s₀ xs ys hx hy

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean add_kernel_masked_correct
#axiomsClean add_kernel_masked_correctness

/- The headline's statement surface is the masked IO signature plus the
audit-once Hoare-triple combinator — no other project constant. -/
#stmtSurfaceSubset add_kernel_masked_correctness ⊆
  [addMaskedIO, VeriTile.Triton.MaskedKernelIO₂.Implements,
   VeriTile.Triton.MaskedKernelIO₂.B]

end VeriTile.Bench.Examples.FlatVectorAdd

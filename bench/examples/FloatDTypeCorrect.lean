/-
bench/examples/FloatDTypeCorrect

**Float dtype erasure — the float-facing *correctness* policy, on the `⊨`
surface.**

VeriTile's float-facing theorem policy on the correctness surface: a kernel
carries explicit `tl.float32` memory annotations (`floatAddKernel`, the one
kernel of this file), and the correctness headline is stated about its
**erasure** to the Real channel — the `kernel` field of the IO signature
below is literally `(floatAddKernel …).eraseDType`, so the erased/ideal
pathway is visible in the audited statement surface itself. Erasure is
semantically a no-op at the algorithm-projection level
(`float_add_erase_projection`: erasing before or after projecting yields
the same algorithm kernel — the projection itself already rewrites every
`tl.float32` load to the Real channel), so every proof obligation runs the
kernel's own projected Real semantics directly: state float, prove Real.
The projected kernel coincides with the plain Real vector add showcased in
`bench/examples/VectorAdd.lean`; this file does not restate that kernel —
one kernel, one file. This is the complement of the rounding-model pathway
(`execR` / `≡[R]`, demonstrated in `bench/examples/FusedSwigluEquiv.lean`);
the refinement counterpart of the erasure policy lives in
`bench/examples/FloatDTypeEquiv.lean`.

Four parts:

1. **The kernel** — `floatAddKernel`, elementwise add with fp32-annotated
   loads.
2. **Erasure projection + region-model Hoare triple** —
   `float_add_erase_projection`, then the value/frame lemmas and their
   package `floatAdd_region_run`, all about the kernel's own projection.
   Also here, the honest region-typing facts: `Kernel.RespectsRegionTyping
   (fun _ => .fp32) floatAddKernel` is **false** under today's definitions
   (the compute→algorithm projection erases the fp32 annotations before
   the typing contract sees them) and is recorded as a proved negation,
   alongside the true all-`.real` typing fact.
3. **Flat-memory bridge side conditions** — `TraceSafe` and `FlattenOk`,
   discharged for the projected kernel.
4. **The spec** — the file's single `specification`:

       float_add_correctness : floatAddIO B ⊨ fun xs ys i => xs i + ys i

   `floatAddIO` is the erased fp32 kernel's **IO signature**: which buffer
   is which argument, where each program reads its input tiles, where it
   writes its output tile. `⊨` is the audit-once Hoare-triple combinator
   (`KernelIO₂.Implements`, `VeriTile.Triton.Memory.KernelSpec`); spelled
   out, the headline says: for **every** disjoint placement of the three
   buffers in flat memory (∀ base pointers, ∀ buffer sizes), **every**
   program id whose windows are in bounds, and **every** launch state whose
   input windows hold `xs`/`ys` — all other buffer cells and all registers
   arbitrary — the translated pointer kernel terminates, its output window
   holds the pointwise sum, and every other memory cell is unchanged.
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FloatDTypeCorrect

open VeriTile.Triton
open VeriTile.Triton.KernelIO₂ (Implements)
open scoped VeriTile.Triton.KernelIO₂
open VeriTile.Examples

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Part 1 — the kernel -/

/-- Elementwise add with fp32-annotated input memory: the loads carry
Triton-like `dtype=tl.float32` information. The headline below is stated
about this kernel's **erasure** (`.eraseDType`), and every proof runs the
projected Real semantics — the projection maps the annotated loads to the
Real channel (`float_add_erase_projection`), landing on the plain aligned
vector add (the kernel showcased in `bench/examples/VectorAdd.lean`). -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-! ## Part 2 — erasure projection, region-model Hoare triple, typing facts -/

/-- Erasure is a no-op at the algorithm-projection level: erasing the fp32
annotations and then projecting yields the same algorithm kernel as
projecting directly — the compute→algorithm projection
(`ComputeOp.toAlgorithm?`) already rewrites every `tl.float32` load to the
Real channel. This single equation transports all three `Implements.intro`
obligations (FlattenOk, TraceSafe, region-model run), stated about the
erased kernel, to the kernel's own projection. -/
private theorem float_add_erase_projection
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    (floatAddKernel xReg yReg outReg blockSize).eraseDType.toAlgKernel =
      (floatAddKernel xReg yReg outReg blockSize).toAlgKernel := by
  simp [floatAddKernel, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

/-- Value half: from any state with the inputs loaded, every output lane of
the projected kernel holds the elementwise sum. -/
theorem float_add_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      observeAt (exec (floatAddKernel xReg yReg outReg blockSize) s)
          outReg blockSize s.pid i
        = some (xs i + ys i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, floatAddKernel, ComputeKernel.toAlgKernel,
        ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_y
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_y]

/-- A scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the unmasked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Frame half: every memory cell other than the output window is preserved
by the projected kernel's run. -/
private theorem floatAdd_frame (xReg yReg outReg : RegionName)
    (B : Nat) (s s1 : BlockState)
    (hExec : exec ((floatAddKernel xReg yReg outReg B).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, ¬(outReg = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    stepStmts, stepStmt, Tile.bop, NumericDType.add, NumericDType.mul]
    at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k.1 hc

/-- **The region-model Hoare triple** — termination, output-window values,
and frame, from any launch state whose input windows are loaded. This is
what the `⊨` headline transports (through the erasure projection) to flat
memory. -/
theorem floatAdd_region_run (B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs ys : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val) = xs j)
    (hy : ∀ j : Fin B, s₀.readMem ⟨"y"⟩ (s₀.pid * B + j.val) = ys j) :
    ∃ s1, exec ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin B,
          s1.readMem ⟨"out"⟩ (s₀.pid * B + j.val) = xs j + ys j)
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := float_add_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B hB s₀ xs ys hx hy
  rw [show exec (floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B) s₀
      = exec ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀
      with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have := hobs j
        rw [hsrc] at this
        simpa [observeAt] using this
      · refine floatAdd_frame ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B s₀ s1 hsrc r o
          (fun i ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i ho.symm

/-- The original claim `Kernel.RespectsRegionTyping (fun _ => .fp32)
(floatAddKernel …)` is **false** under today's definitions and is therefore
recorded as a proved negation. `Kernel.RespectsRegionTyping` receives the
kernel through the `ComputeKernel → AlgKernel` coercion (`toAlgKernel`), and
the compute→algorithm projection (`ComputeOp.toAlgorithm?`) rewrites every
`tl.float32` load to `Op.load ComputeDType.fp32.eraseDType = Op.load .real`
before the typing contract ever sees it. The contract's load/store cases then
demand `Γ region = .real`, i.e. `TileDType.fp32 = TileDType.real`, which
reduces to `False`. -/
private theorem float_add_not_respects_fp32_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    ¬ Kernel.RespectsRegionTyping (fun _ => .fp32)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-- The true region-typing contract on today's projected surface: after the
compute→algorithm projection the fp32 annotations are erased, so the kernel
respects the all-`.real` region typing. -/
private theorem float_add_respects_real_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    Kernel.RespectsRegionTyping (fun _ => .real)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-! ## Part 3 — flat-memory bridge side conditions

The projected kernel is register-indirect (`offs := ...; tl.load(x + offs)`),
so no ∀-state contract covers it; the flat-memory bridge (v1.2) takes the
per-execution `Kernel.TraceSafe` contract instead, discharged below by
walking the actual six-statement execution. Because the kernel is
**unmasked**, `MaskOpt.none.Active` makes every lane active, and the bounds
obligation is the aligned contract `pid * B + B ≤ bounds reg` — the whole
program tile in bounds — for each of the three regions. -/

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
/-- The projected kernel is trace-safe: each of its six statements is safe
in the state the execution actually reaches. Unmasked loads/store touch all
`B` lanes of the program tile, so each region's bound must cover the whole
tile: `s.pid * B + B ≤ bounds reg`. -/
theorem floatAdd_traceSafe (xReg yReg outReg : RegionName)
    (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg)
    (hout : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafe bounds
      ((floatAddKernel xReg yReg outReg B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- Pin the projected statement list to its literal form (the plain Real
  -- vector add — fp32 annotations already projected away). The projection
  -- leaves stuck `ComputeDType.fp32.eraseDType` indices on the loads that
  -- block the `Op.SafeAt` reductions in the walk below; they sit in
  -- dependent (type-index) positions where `simp`/`dsimp` cannot rewrite,
  -- but `show` checks definitional equality at default transparency, which
  -- reduces them.
  show Stmt.TraceSafeList bounds
    [Stmt.assign TileDType.nat [] "pid" (Op.programId 0),
      Stmt.assign TileDType.nat [B] "offs"
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "pid")
            (Op.constNat B)) (Op.arange B)),
      Stmt.assign TileDType.real [B] "x"
        (Op.load TileDType.real
          (MemAccess.region xReg (Op.ref TileDType.nat [B] "offs")) MaskOpt.none),
      Stmt.assign TileDType.real [B] "y"
        (Op.load TileDType.real
          (MemAccess.region yReg (Op.ref TileDType.nat [B] "offs")) MaskOpt.none),
      Stmt.assign TileDType.real [B] "out"
        (Op.add NumericDType.real Broadcast.nil.consSame
          (Op.ref TileDType.real [B] "x") (Op.ref TileDType.real [B] "y")),
      Stmt.store TileDType.real [B]
        (MemAccess.region outReg (Op.ref TileDType.nat [B] "offs"))
        (Op.ref TileDType.real [B] "out") MaskOpt.none]
    s
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: offs := pid * B + arange B
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
  -- shared register-readback fact at the state before the loads
  set sm := (s.setReg "pid" .nat []
        (Tile.scalar (dtype := .nat) (s.pids 0))).setReg "offs" .nat [B]
      ⟨fun i => s.pids 0 * B + i.1.val⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "offs" → t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (reg : RegionName), s.pid * B + B ≤ bounds reg →
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (Op.ref .nat [B] "offs")) t
        ((MaskOpt.none (dtype := .real)).Active t) := by
    intro t hframe reg hreg
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i _
    rw [show evalOp (Op.ref .nat [B] "offs") t
        = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
      simp [hframe .nat [B] "offs" rfl, hsm, BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    simp only [Region.cast_self]
    exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg
  -- statement 3: x := load(xReg + offs)   (fp32 annotation projected away)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, trivial, hAAS sm (fun _ _ _ _ => rfl) xReg hx⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 4: y := load(yReg + offs)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp, trivial, hAAS _ (fun dt sh nm hnm => ?_) yReg hy⟩
    subst hnm; simp [BlockState.setReg]
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 5: out := x + y
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 6: store(outReg + offs, out)   (unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    hAAS _ (fun dt sh nm hnm => ?_) outReg hout⟩
  subst hnm; simp [BlockState.setReg]

/-- The projected kernel sits inside the bridge's covered fragment. -/
theorem floatAdd_flattenOk (xReg yReg outReg : RegionName) (B : Nat) :
    ((floatAddKernel xReg yReg outReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-! ## Part 4 — the spec: `floatAddIO ⊨` pointwise addition -/

/-- The erased fp32 add kernel's **IO signature** — the whole
kernel-specific audit surface of the headline:

* `kernel` — **`(floatAddKernel …).eraseDType`**: the dtype-erasure policy
  is part of the audited signature, not hidden in the proof;
* `in1`/`in2`/`out` — which buffer is which argument (the wiring);
* `B` — the tile length;
* `read1`/`read2` — where program `pid` reads its input tiles (the address
  half of the *pre*condition);
* `write` — where program `pid` writes its output tile (the address half
  of the *post*condition: values land there, frame holds everywhere else).

The windows are declared, not parsed from the kernel: they formalize the
host-side launch convention (`offsets = pid * BLOCK_SIZE + arange`), and
the headline **proves** the kernel's actual addressing matches them.
Buffer sizes are not signature content: the headline quantifies over every
allocation large enough for the windows. -/
def floatAddIO (B : Nat) : KernelIO₂ where
  kernel := (floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).eraseDType
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B

/-- **The headline**: the fp32-annotated add kernel, projected through
`eraseDType`, implements pointwise addition on its IO signature — see the
module docstring for the full Hoare triple `⊨` unfolds to. Proof:
`Implements.intro` assembles the region-model triple (Part 2) with the
bridge side conditions (Part 3), each transported from the erased kernel to
the kernel's own projection along `float_add_erase_projection`. -/
specification float_add_correctness (B : Nat) (hB : 0 < B) :
    floatAddIO B ⊨ fun xs ys i => xs i + ys i := by
  have hproj := float_add_erase_projection ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  refine KernelIO₂.Implements.intro _ ?_ ?_ ?_
  · show ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).eraseDType).toAlgKernel.FlattenOk
    rw [hproj]
    exact floatAdd_flattenOk ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  · intro bounds s h1 h2 h3
    show Kernel.TraceSafe bounds
      (((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).eraseDType).toAlgKernel) s
    rw [hproj]
    exact floatAdd_traceSafe ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    show ∃ s1,
      exec (((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).eraseDType).toAlgKernel) s₀
          = some s1
        ∧ (∀ j : Fin B,
            s1.readMem ⟨"out"⟩ (s₀.pid * B + j.val) = xs j + ys j)
        ∧ (∀ r o,
            (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
            s1.mem r o = s₀.mem r o)
    rw [hproj]
    exact floatAdd_region_run B hB s₀ xs ys hx hy

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean float_add_correctness

/- The headline's statement surface is the IO signature plus the audit-once
Hoare-triple combinator — no other project constant. The dtype-erasure
policy (`eraseDType`) lives inside `floatAddIO`'s definition, one
definition-click away from the headline. -/
#stmtSurfaceSubset float_add_correctness ⊆
  [floatAddIO, VeriTile.Triton.KernelIO₂.Implements,
   VeriTile.Triton.KernelIO₂.B]

end VeriTile.Bench.Examples.FloatDTypeCorrect

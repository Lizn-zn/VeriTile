/-
bench/examples/VectorAdd

The aligned (unmasked) elementwise vector add — the smallest complete
showcase of the standard trust stack, in four parts:

1. **Kernel + math spec** — the Triton DSL kernel `addKernel` and
   `addSpec : out[i] = xs[i] + ys[i]`.
2. **Region-model headline** — `add_kernel_correct_view` on the
   `ComputeCorrect.Realizes` surface.
3. **Flat-memory bridge side conditions** — `TraceSafe` / `FlattenOk`,
   discharged for this kernel. Because the kernel is unmasked, every lane
   of the program tile is active, so the safety contract is the aligned
   bound `pid * B + B ≤ extent` per region.
4. **Denotation headline** — `add_kernel_correctness`: ⟦`addKernel`⟧ maps
   the input arrays to their elementwise sum. Every pointer, region,
   layout, and start-state concept lives inside `denoteAddKernel` (a
   one-line instantiation of the generic `denoteKernel` combinator), and
   the statement mentions only `Nat` / `Fin` / `ℝ` / `Option` / `=`; the
   `#stmtSurfaceSubset` gate pins its statement surface to that single
   constant.

The masked boundary variant (`addKernelMasked`) lives in
`bench/examples/FlatVectorAdd.lean` — each showcased kernel is
self-contained in its showcase file.

Source Triton (`.py` reference, aligned single-block flavour):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid     = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask  = offsets < n_elements   -- omitted: aligned (n_elements = BLOCK_SIZE)
    x       = tl.load(x_ptr + offsets)
    y       = tl.load(y_ptr + offsets)
    output  = x + y
    tl.store(out_ptr + offsets, output)
```
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.VectorAdd

open VeriTile.Triton
open VeriTile.Examples

/-! ## Part 1 — the kernel and its math spec -/

/-- Elementwise add of two `blockSize`-element tiles, single-block.

Reads from `xReg` and `yReg`, writes to `outReg`. No aliasing assumption
required: even if `xReg = outReg` (read-modify-write of the same buffer)
the kernel reads first into local registers `x` / `y` before the final
scatter to `outReg`, so the result is still `xs + ys`. -/
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  y    := tl.load($(yReg) + offs)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-- Elementwise add: `out[i] = xs[i] + ys[i]`. -/
def addSpec {N : Nat} (xs ys : Fin N → ℝ) (i : Fin N) : ℝ :=
  xs i + ys i

/-! ## Part 2 — region-model correctness -/

/-- `addKernel` correctness against `addSpec`, at the raw `observeAt` level:
from any state with the inputs loaded, the kernel writes the elementwise sum
to `outReg`. -/
theorem add_kernel_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      observeAt (exec (addKernel xReg yReg outReg blockSize) s) outReg blockSize s.pid i
        = some (addSpec xs ys i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernel, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul, addSpec]
  unfold InputLoadedAt at _h_x _h_y
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_y]

/-- View-level middleware for `add_kernel_correct` (`TensorView.observe`
in place of raw `observeAt`). -/
theorem add_kernel_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (addKernel xReg yReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hy := inputLoadedAt_of_programTileView_loaded (s := s) (region := yReg)
    (N := blockSize) (xs := ys) h_y
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using add_kernel_correct xReg yReg outReg blockSize hBlockSize s xs ys hx hy idx.1

/-- **Region-model headline**: on the standard `ComputeCorrect.Realizes`
surface, from any state with the inputs loaded at the program tiles, the
kernel writes the view-level elementwise sum to `outReg`. -/
theorem add_kernel_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := addKernel xReg yReg outReg blockSize)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.ofTensorView
        (programTileView s outReg blockSize))
      (expected := fun idx : TileIndex [blockSize] => addSpec xs ys idx.1) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa [ComputeCorrect.WriteMap.ofTensorView, TensorView.observe,
    observeTileAt] using hview

/-! ## Part 3 — flat-memory bridge side conditions

`addKernel` is register-indirect (`offs := ...; tl.load(x + offs)`), so no
∀-state contract covers it; the flat-memory bridge (v1.2) takes the
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
/-- The aligned vector add is trace-safe: each of its six statements is safe
in the state the execution actually reaches. Unmasked loads/store touch all
`B` lanes of the program tile, so each region's bound must cover the whole
tile: `s.pid * B + B ≤ bounds reg`. -/
theorem addKernel_traceSafe (xReg yReg outReg : RegionName)
    (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg)
    (hout : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafe bounds
      ((addKernel xReg yReg outReg B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
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
  -- statement 3: x := load(xReg + offs)   (unmasked: all lanes active)
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

/-- The aligned kernel sits inside the bridge's covered fragment. -/
theorem addKernel_flattenOk (xReg yReg outReg : RegionName) (B : Nat) :
    ((addKernel xReg yReg outReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [addKernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- **Flat-memory aligned vector add**: for any disjoint allocation of the
three regions whose extents cover the whole program tile
(`pid * B + B ≤ extent`), the translated kernel — one flat region, addresses
`base + pid*B + i` — run on the flattened state is the flattening of the
region-model run. -/
theorem addKernel_exec_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (xReg yReg outReg : RegionName) (B : Nat) (s : BlockState)
    (hx : s.pid * B + B ≤ A.extent xReg)
    (hy : s.pid * B + B ≤ A.extent yReg)
    (hout : s.pid * B + B ≤ A.extent outReg)
    (hu : s.undef = (fun _ _ => 0)) :
    exec (A.flattenKernel ((addKernel xReg yReg outReg B).toAlgKernel))
        (A.flattenState s)
      = (exec ((addKernel xReg yReg outReg B).toAlgKernel) s).map
          A.flattenState :=
  A.exec_flatten hd hcov _ s
    (addKernel_traceSafe xReg yReg outReg B A.extent s hx hy hout)
    (addKernel_flattenOk xReg yReg outReg B) hu

/-! ## Part 4 — the denotation: array in, array out

`denoteAddKernel` ("⟦addKernel⟧") is a one-line instantiation of the generic
denotation combinator `denoteKernel` (audited once, in
`VeriTile.Triton.Memory.Denotation`), which packages the whole ceremony —
the canonical flat allocation of a region table, the canonical start state
loaded with a slot table, the flattening translation, the flat-memory
execution, and the read-back of one output lane. What remains per kernel —
the whole audit surface of the definition below — is the region table, the
slot table, and the output address. -/

/-- ⟦`addKernel`⟧: one instantiation of the generic `denoteKernel`. The
tables are the audit surface — regions `"x"`/`"y"`/`"out"` of extent `n`
laid end to end in one flat address space, `xs`/`ys` loaded at the program
tile `[pid*B, pid*B + B)` of `"x"`/`"y"`, and lane `i` of the output tile
read back from flat memory. All addressing lives inside `denoteKernel`. -/
noncomputable denotation denoteAddKernel (B n pid : Nat) (xs ys : Fin B → ℝ)
    (i : Fin B) : Option ℝ :=
  denoteKernel (addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B)
    ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] pid
    [.ofFin ⟨"x"⟩ (pid * B) xs, .ofFin ⟨"y"⟩ (pid * B) ys]
    ⟨"out"⟩ (pid * B + i.val)

/-- **The denotation headline**: whenever the program tile is in bounds
(`pid * B + B ≤ n` — the aligned, unmasked contract), ⟦`addKernel`⟧ maps the
input arrays to their elementwise sum on every lane. The statement mentions
only elementary mathematics — every pointer, region, layout, and start-state
concept is inside `denoteAddKernel`, audited once. -/
specification add_kernel_correctness (B n pid : Nat) (hB : 0 < B)
    (xs ys : Fin B → ℝ) (i : Fin B) (hn : pid * B + B ≤ n) :
    denoteAddKernel B n pid xs ys i = some (xs i + ys i) := by
  have hd := FlatAlloc.ofList_disjoint ⟨"flat"⟩
    [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] (by simp)
  have hsnd : (([DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
      DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]).map DenoteSlot.region).Nodup := by
    simp
  have hloadX : InputLoadedAt
      (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) ⟨"x"⟩ B xs := by
    intro j
    simpa using DenoteSlot.state_read_ofFin (pid := pid)
      (region := (⟨"x"⟩ : RegionName)) (off := pid * B) (arr := xs)
      (by simp) hsnd j
  have hloadY : InputLoadedAt
      (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) ⟨"y"⟩ B ys := by
    intro j
    simpa using DenoteSlot.state_read_ofFin (pid := pid)
      (region := (⟨"y"⟩ : RegionName)) (off := pid * B) (arr := ys)
      (by simp) hsnd j
  unfold denoteAddKernel denoteKernel
  rw [addKernel_exec_flatten _ hd
    (FlatAlloc.ofList_closed ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)])
    ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
    (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
      DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys])
    (by simpa [FlatAlloc.listExtent] using hn)
    (by simpa [FlatAlloc.listExtent] using hn)
    (by simpa [FlatAlloc.listExtent] using hn) rfl]
  have hobs := add_kernel_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩
    B hB (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
      DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) xs ys hloadX hloadY i
  rw [show exec (addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B)
      (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys])
    = exec ((addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
        ).toAlgKernel) (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) from rfl] at hobs
  cases hsrc : exec ((addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
      ).toAlgKernel) (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) with
  | none =>
      rw [hsrc] at hobs
      simp [observeAt] at hobs
  | some s1 =>
      rw [hsrc] at hobs
      simp only [observeAt, Option.map_some, Option.some_inj,
        DenoteSlot.state_pid] at hobs
      simp only [Option.map_some, Option.some_inj]
      rw [FlatAlloc.flattenState_readMem _ hd s1
        (r := ⟨"out"⟩) (by simp)
        (o := pid * B + i.val)
        (by have := i.isLt; simp only [FlatAlloc.ofList_extent]
            simp [FlatAlloc.listExtent]; omega)]
      simpa [addSpec] using hobs

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the public theorems' transitive proofs.
#axiomsClean add_kernel_correct_view
#axiomsClean addKernel_exec_flatten
#axiomsClean add_kernel_correctness

/- The whole point of the denotation headline: its statement surface is ONE
project constant — every addressing concept is inside `denoteAddKernel`. -/
#stmtSurfaceSubset add_kernel_correctness ⊆ [denoteAddKernel]

end VeriTile.Bench.Examples.VectorAdd

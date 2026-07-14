/-
bench/examples/FlatVectorAdd

**Stage 1 + stage 2 combined**: a masked, bounds-checked vector-add kernel
whose flat-memory corollary is obtained by *instantiating* the flat-memory
bridge (`FlatAlloc.exec_flatten`) and *discharging* its side conditions
concretely — the `Kernel.MemorySafe` bounds contract and the
`Kernel.FlattenOk` fragment membership are proved for this kernel, not
assumed. The result: running the translated kernel (one flat address space,
real pointer arithmetic `base + pid*B + i`) on the flattened state is the
flattening of the region-model run.

Two kernels demonstrate the two safety regimes:

- `flatAddKernel` writes its addresses and masks *inline*, so its bounds
  contract holds in **every** state (mask and address share the
  `program_id(0)` subterm) — the shape the ∀-state #48 contracts were made
  for.
- `addKernelMasked` (the ordinary DSL kernel, defined in this file — each
  showcased kernel is self-contained in its showcase file) uses the
  register-indirect `offs := ...; tl.load(x + offs, mask=offs < n)` style,
  which no ∀-state contract can cover. Bridge **v1.2** takes the
  per-execution `Kernel.TraceSafe` contract instead, and this file
  discharges it by walking the actual seven-statement execution — the
  template for putting real DSL kernels through the bridge.

The **headline** is `add_kernel_masked_flat_correct_view` (the file's last
theorem): flat-memory correctness on the frame-strengthened
`ComputeCorrect.RealizesFrame` trust surface — the flattened kernel, run
from the flattened state, stores `xs i + ys i` at every masked lane of the
flat output tile **and touches no other memory cell**. Inputs and output
speak the same `TensorView` layout language (`ViewsLoaded` slots in,
`WriteMap.viewIf` out); the layout/launch ceremony is bundled into
`FlatLayout`/`LaunchState`, so every hypothesis line is spec content. The
`exec_flatten` corollaries are the translation steps the proof composes
with the region-model correctness theorem and the store-frame lemma.
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FlatVectorAdd

open VeriTile.Triton
open VeriTile.Triton.TensorView (ViewsLoaded slot)
open VeriTile.Examples (programTileView InputLoadedAt observeAt
  inputLoadedAt_of_programTileView_loaded)

/-- Per-lane offsets, written inline: `program_id(0) * B + arange B`. -/
private def offsExpr (B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (.mul .nat .nil (.programId 0) (.constNat B)) (.arange B)

/-- Per-lane bounds mask, written inline: `offs < n`. -/
private def maskExpr (B n : Nat) : Op .bool [B] :=
  .lt .nat .scalarR (offsExpr B) (.constNat n)

/-- Masked vector add with inline addressing: the mask `pid*B + i < n` bounds
exactly the addresses `pid*B + i` that the loads and the store touch. -/
def flatAddKernel (xReg yReg outReg : RegionName) (B n : Nat) : Kernel where
  inputs := [xReg, yReg]
  outputs := [outReg]
  body := [
    .assign .real [B] "x"
      (.load .real (.region xReg (offsExpr B)) (.mask (maskExpr B n))),
    .assign .real [B] "y"
      (.load .real (.region yReg (offsExpr B)) (.mask (maskExpr B n))),
    .store .real [B] (.region outReg (offsExpr B))
      (.add .real (.consSame .nil) (.ref .real [B] "x") (.ref .real [B] "y"))
      (.mask (maskExpr B n))]

/-! ## Discharging the bridge's side conditions -/

/-- The offsets evaluate to `pid*B + i` in every state. -/
private theorem evalOp_offsExpr (B : Nat) (s : BlockState) :
    evalOp (offsExpr B) s = some ⟨fun i => s.pids 0 * B + i.1.val⟩ := by
  simp [offsExpr, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
    Tile.vec]

/-- Any active lane of the inline mask has its address below `n` — in every
state, because the mask and the address share the `program_id` subterm. -/
private theorem active_lane_bound (B n : Nat) (s : BlockState)
    (i : TileIndex [B])
    (hact : (MaskOpt.mask (dtype := .real) (maskExpr B n)).Active s i) :
    s.pids 0 * B + i.1.val < n := by
  obtain ⟨masks, hmasks, hdata⟩ := hact
  simp [maskExpr, evalOp_offsExpr, Tile.cop] at hmasks
  subst hmasks
  simpa using hdata

/-- The inline kernel is trace-safe from any start state whose extents cover
`n` (in fact each step is safe in *every* state — the inline shape). -/
theorem flatAddKernel_traceSafe (xReg yReg outReg : RegionName) (B n : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hx : n ≤ bounds xReg) (hy : n ≤ bounds yReg) (hout : n ≤ bounds outReg) :
    (flatAddKernel xReg yReg outReg B n).TraceSafe bounds s := by
  have msOffs : ∀ t : BlockState, (offsExpr B).SafeAt bounds t := by
    intro t; simp [offsExpr, Op.SafeAt]
  have msMaskE : ∀ t : BlockState, (maskExpr B n).SafeAt bounds t := by
    intro t; simp [maskExpr, offsExpr, Op.SafeAt]
  have bnd : ∀ (reg : RegionName), n ≤ bounds reg → ∀ (t : BlockState),
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (offsExpr B)) t
        ((MaskOpt.mask (dtype := .real) (maskExpr B n)).Active t) := by
    intro reg hreg t
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i hact
    rw [evalOp_offsExpr] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    have hlt := active_lane_bound B n t i hact
    simp only [Region.cast_self]
    exact lt_of_lt_of_le hlt hreg
  unfold Kernel.TraceSafe flatAddKernel
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt,
    MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨⟨msOffs s, msMaskE s, bnd xReg hx s⟩, ?_⟩
  split
  · refine ⟨⟨msOffs _, msMaskE _, bnd yReg hy _⟩, ?_⟩
    split
    · refine ⟨⟨msOffs _, by simp, msMaskE _, bnd outReg hout _⟩, ?_⟩
      split <;> trivial
    · trivial
  · trivial

/-- The kernel sits inside the bridge's covered fragment. -/
theorem flatAddKernel_flattenOk (xReg yReg outReg : RegionName) (B n : Nat) :
    (flatAddKernel xReg yReg outReg B n).FlattenOk := by
  unfold Kernel.FlattenOk flatAddKernel
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk, offsExpr, maskExpr]

/-! ## The flat-pointer corollary -/

/-- **Flat-memory vector add**: for any disjoint allocation of the three
regions whose extents cover `n`, the translated kernel — one flat region,
addresses `base + pid*B + i` — run on the flattened state is the flattening
of the region-model run. Stage 1 (explicit addressing) + stage 2 (the
bridge), with every side condition discharged above. -/
theorem flatAddKernel_exec_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (xReg yReg outReg : RegionName) (B n : Nat)
    (hx : n ≤ A.extent xReg) (hy : n ≤ A.extent yReg)
    (hout : n ≤ A.extent outReg)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) :
    exec (A.flattenKernel (flatAddKernel xReg yReg outReg B n))
        (A.flattenState s)
      = (exec (flatAddKernel xReg yReg outReg B n) s).map A.flattenState :=
  A.exec_flatten hd hcov _ s
    (flatAddKernel_traceSafe xReg yReg outReg B n A.extent s hx hy hout)
    (flatAddKernel_flattenOk xReg yReg outReg B n) hu

/-! ## Part 2 — the register-indirect DSL kernel (bridge v1.2 in action)

`addKernelMasked` (defined below) is the ordinary DSL shape every bench
port uses: offsets and mask are computed into registers and the memory
operations address through `Op.ref`. It handles arbitrary `n_elements` by
computing `mask = offsets < n_elements` and threading it through the load
and store, exactly mirroring the canonical Triton tutorial:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)
```

No ∀-state contract covers register-indirect addressing; the per-execution
`Kernel.TraceSafe` below is discharged by walking the actual seven-statement
execution, reading the offset/mask registers back at each load and at the
store. -/

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

/-- **`addKernelMasked` region-model correctness.**

For each lane `i ∈ Fin blockSize`:
* In-bounds (`pid * blockSize + i < nElements`): the output region holds
  `xs i + ys i` at `pid * blockSize + i`.
* Out-of-bounds: the output region's value at `pid * blockSize + i` is
  preserved from the initial state (mask=false → no store).

The hypothesis `InputLoadedAt` constrains memory at every lane in the
`blockSize`-length tile (including out-of-bounds lanes where the data
is irrelevant semantically). This matches Triton's actual behavior:
masked-off loads without `other=` do not read memory and produce
undefined lane values; the matching masked store prevents those values
from reaching memory.

No region-disjointness hypothesis: the kernel reads `x` and `y` into
local registers BEFORE the scatter to `outReg`, so even if `outReg`
aliases `xReg` or `yReg`, the result is correct. -/
theorem add_kernel_masked_correct
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      let addr := s.pid * blockSize + i.val
      observeAt (exec (addKernelMasked xReg yReg outReg blockSize nElements) s)
                outReg blockSize s.pid i
        = some (if addr < nElements then xs i + ys i
                else s.readMem outReg addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernelMasked, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * blockSize + i.val < nElements
  · simp [hi, h_x, h_y]
  · simp [hi]

/-- View-level surface for `add_kernel_masked_correct`. -/
theorem add_kernel_masked_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ∀ idx : TileIndex [blockSize],
      let addr := s.pid * blockSize + idx.1.val
      TensorView.observe (exec (addKernelMasked xReg yReg outReg blockSize nElements) s)
          (programTileView s outReg blockSize) idx
        = some (if addr < nElements then xs idx.1 + ys idx.1
                else s.readMem outReg addr) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hy := inputLoadedAt_of_programTileView_loaded (s := s) (region := yReg)
    (N := blockSize) (xs := ys) h_y
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using add_kernel_masked_correct xReg yReg outReg blockSize nElements
      hBlockSize s xs ys hx hy idx.1

/-- Compute-facing view-level surface for `add_kernel_masked_correct` — the
region-model `Realizes` middleware the flat headline composes with. -/
theorem add_kernel_masked_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := addKernelMasked xReg yReg outReg blockSize nElements)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [blockSize] =>
          s.pid * blockSize + idx.1.val < nElements)
        (fun idx => (outReg, s.pid * blockSize + idx.1.val)))
      (expected := fun idx => xs idx.1 + ys idx.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have hview := add_kernel_masked_correct_exec_view xReg yReg outReg blockSize nElements
    hBlockSize s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa [TensorView.observe, observeTileAt, programTileView, TensorView.offset,
    Offset.strided, hActive] using hview

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
statements is safe in the state the execution actually reaches. -/
theorem addKernelMasked_traceSafe (xReg yReg outReg : RegionName)
    (B n : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : n ≤ bounds xReg) (hy : n ≤ bounds yReg)
    (hout : n ≤ bounds outReg) :
    Kernel.TraceSafe bounds
      ((addKernelMasked xReg yReg outReg B n).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  -- statement 2: offsets := pid * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
      (Op.arange B)) (s.setReg "pid" .nat [] v1)
      = some ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- statement 3: mask := offsets < n
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  rw [show evalOp (Op.lt .nat .scalarR (Op.ref .nat [B] "offsets")
      (Op.constNat n))
      ((s.setReg "pid" .nat [] v1).setReg "offsets" .nat [B]
        ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩)
      = some ⟨fun i => ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩ from by
    simp [Tile.cop, BlockState.setReg]] at hv3
  obtain rfl := Option.some_inj.mp hv3
  -- shared register-readback facts at the state before the loads
  set sm := ((s.setReg "pid" .nat [] v1).setReg "offsets" .nat [B]
      ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩).setReg "mask" .bool [B]
      ⟨fun i => ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "offsets" ∨ nm = "mask" →
        t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (reg : RegionName), n ≤ bounds reg →
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (Op.ref .nat [B] "offsets")) t
        ((MaskOpt.mask (dtype := .real) (Op.ref .bool [B] "mask")).Active t) := by
    intro t hframe reg hreg
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i hact
    rw [show evalOp (Op.ref .nat [B] "offsets") t
        = some ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩ from by
      simp [hframe .nat [B] "offsets" (Or.inl rfl), hsm,
        BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    obtain ⟨masks, hm, hd⟩ := hact
    rw [show evalOp (Op.ref .bool [B] "mask") t
        = some ⟨fun i =>
          ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩ from by
      simp [hframe .bool [B] "mask" (Or.inr rfl), hsm,
        BlockState.setReg]] at hm
    obtain rfl := Option.some_inj.mp hm
    rw [ComparableDType.nat_lt_eq_true] at hd
    simp only [Region.cast_self]
    exact lt_of_lt_of_le hd hreg
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

/-- **Flat-memory masked vector add, register style**: the bridge applied to
an ordinary DSL kernel, with the per-execution safety discharged above. -/
theorem addKernelMasked_exec_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (xReg yReg outReg : RegionName) (B n : Nat)
    (hx : n ≤ A.extent xReg) (hy : n ≤ A.extent yReg)
    (hout : n ≤ A.extent outReg)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) :
    exec (A.flattenKernel
        ((addKernelMasked xReg yReg outReg B n).toAlgKernel))
        (A.flattenState s)
      = (exec ((addKernelMasked xReg yReg outReg B n
          ).toAlgKernel) s).map A.flattenState :=
  A.exec_flatten hd hcov _ s
    (addKernelMasked_traceSafe xReg yReg outReg B n A.extent s hx hy hout)
    (addKernelMasked_flattenOk xReg yReg outReg B n) hu

/-! ## Part 3 — the headline: flat-memory compute correctness

The `exec_flatten` corollaries above are *translation* statements. The
headline below is the actual **correctness** theorem on the standard trust
surface: the flattened masked vector add — one flat region, real pointer
arithmetic `A.base r + pid*B + i` — repackaged as a `ComputeKernel` and
stated with `ComputeCorrect.Realizes_without_Rounding`: every masked lane
of the flat output holds `xs i + ys i`. The proof composes the region-model
correctness theorem (`add_kernel_masked_correct`) with the bridge
(`exec_flatten`) and the flat read-back lemma below. -/

/-- The flattened masked vector add, repackaged on the compute surface
(`toAlgorithm?` recovers exactly `A.flattenKernel` of the original). -/
noncomputable def addKernelMaskedFlat (A : FlatAlloc)
    (xReg yReg outReg : RegionName) (B n : Nat) : ComputeKernel :=
  let fk := A.flattenKernel
    ((addKernelMasked xReg yReg outReg B n).toAlgKernel)
  ComputeKernel.fromKernelBody fk.inputs fk.outputs fk.body

/-- The flat output tile: lane `i` of the current program's output block, at
flat offset `A.addr outReg (pid*B) + i`. The output-side layout object — the
`write` contract below speaks the same `TensorView` language as the input
slots. -/
def flatOutTile (A : FlatAlloc) (outReg : RegionName) (pid B : Nat) :
    TensorView [B] :=
  { region := A.flat, base := A.addr outReg (pid * B), strides := [1] }

/-- The flat output contract: lane `i` writes the flat output tile exactly
when `pid*B + i < n` — the packaged `write` argument of the headline, the
flat sibling of `VeriTile.Examples.outWritesTo`. -/
def flatOutWrites (A : FlatAlloc) (outReg : RegionName) (pid B n : Nat) :
    ComputeCorrect.WriteMap (TileIndex [B]) :=
  ComputeCorrect.WriteMap.viewIf (flatOutTile A outReg pid B)
    (fun i : Fin B => pid * B + i.val < n)

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level companion of `foldl_store_preserve`). -/
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
/-- Frame for the register-style masked vector add: every memory cell not
actively written by the output store is preserved. -/
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

/-- **Flat-memory vector-add correctness**: for any flat layout covering the
three regions and any launch state loaded with the inputs, the translated
kernel stores `xs i + ys i` at every masked lane of the flat output region —
stated on the standard `ComputeCorrect.Realizes` trust surface. Every
hypothesis line is spec content: the layout, the launch state, the inputs. -/
specification add_kernel_masked_flat_correct_view
    (xReg yReg outReg : RegionName) (B n : Nat) (hB : 0 < B)
    (L : FlatLayout [(xReg, n), (yReg, n), (outReg, n)])
    (s : LaunchState) (xs ys : Fin B → ℝ)
    (hin : ViewsLoaded s [slot (programTileView s xReg B) xs,
                          slot (programTileView s yReg B) ys]) :
    ComputeCorrect.RealizesFrame_without_Rounding
      (kernel := addKernelMaskedFlat L.alloc xReg yReg outReg B n)
      (initialState := L.alloc.flattenState s)
      (write := flatOutWrites L.alloc outReg s.pid B n)
      (expected := fun idx => xs idx.1 + ys idx.1) := by
  unfold flatOutWrites LaunchState.pid
  obtain ⟨A, hd, hcov, hcovers⟩ := L
  obtain ⟨s, hu⟩ := s
  obtain ⟨h_x', h_y', -⟩ := hin
  have h_x := inputLoadedAt_of_programTileView_loaded h_x'
  have h_y := inputLoadedAt_of_programTileView_loaded h_y'
  have hx : n ≤ A.extent xReg := hcovers (xReg, n) (by simp)
  have hy : n ≤ A.extent yReg := hcovers (yReg, n) (by simp)
  have hout : n ≤ A.extent outReg := hcovers (outReg, n) (by simp)
  have hak : (addKernelMaskedFlat A xReg yReg outReg B n).toAlgKernel
      = A.flattenKernel
        ((addKernelMasked xReg yReg outReg B n).toAlgKernel) := by
    simp [addKernelMaskedFlat]
  have haddr : ∀ i : Fin B,
      ((flatOutTile A outReg s.pid B).region,
        (flatOutTile A outReg s.pid B).offset (i, PUnit.unit))
      = (A.flat, A.addr outReg (s.pid * B + i.val)) := by
    intro i
    simp [flatOutTile, TensorView.offset, Offset.strided, FlatAlloc.addr,
      Nat.add_assoc]
  constructor
  · -- the value half: every masked lane holds `xs i + ys i`
    rw [show ComputeCorrect.WriteMap.viewIf
          (flatOutTile A outReg s.pid B)
          (fun i : Fin B => s.pid * B + i.val < n)
        = ComputeCorrect.WriteMap.writeIf
            (fun idx : TileIndex [B] => s.pid * B + idx.1.val < n)
            (fun idx => (A.flat, A.addr outReg (s.pid * B + idx.1.val))) from by
      funext idx
      simp only [ComputeCorrect.WriteMap.viewIf, ComputeCorrect.WriteMap.writeIf]
      split
      · exact congrArg some (haddr idx.1)
      · rfl]
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    have houtr : outReg ∈ A.regions := by
      by_contra hn
      exact absurd (hcov outReg hn) (by omega)
    rw [hak, A.exec_flatten hd hcov _ s
      (addKernelMasked_traceSafe xReg yReg outReg B n A.extent s hx hy hout)
      (addKernelMasked_flattenOk xReg yReg outReg B n) hu] at hExec
    cases hsrc : exec
        ((addKernelMasked xReg yReg outReg B n).toAlgKernel) s with
    | none => rw [hsrc] at hExec; exact absurd hExec (by simp)
    | some s1 =>
        rw [hsrc] at hExec
        replace hExec : some (A.flattenState s1) = some s' := hExec
        obtain rfl := (Option.some_inj.mp hExec).symm
        have hobs := add_kernel_masked_correct
          xReg yReg outReg B n hB s xs ys h_x h_y idx.1
        rw [show exec (addKernelMasked xReg yReg outReg B n) s
            = exec ((addKernelMasked
                xReg yReg outReg B n).toAlgKernel) s from rfl, hsrc] at hobs
        simp only [observeAt, Option.map_some,
          Option.some_inj, if_pos hActive] at hobs
        simpa [A.flattenState_readMem hd s1 houtr
          (lt_of_lt_of_le hActive hout)] using hobs
  · -- the frame half: every cell outside the write image is preserved
    apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
    intro s0 s' hExec hs0
    subst s0
    intro addr hnw
    rw [hak, A.exec_flatten hd hcov _ s
      (addKernelMasked_traceSafe xReg yReg outReg B n A.extent s hx hy hout)
      (addKernelMasked_flattenOk xReg yReg outReg B n) hu] at hExec
    cases hsrc : exec
        ((addKernelMasked xReg yReg outReg B n).toAlgKernel) s with
    | none => rw [hsrc] at hExec; exact absurd hExec (by simp)
    | some s1 =>
        rw [hsrc] at hExec
        replace hExec : some (A.flattenState s1) = some s' := hExec
        obtain rfl := (Option.some_inj.mp hExec).symm
        obtain ⟨raddr, oaddr⟩ := addr
        show (if raddr = A.flat then A.readFlat s1 oaddr else MemCell.real 0)
          = (if raddr = A.flat then A.readFlat s oaddr else MemCell.real 0)
        by_cases hr : raddr = A.flat
        · rw [if_pos hr, if_pos hr]
          unfold FlatAlloc.readFlat
          cases hdec : A.decode oaddr with
          | none => rfl
          | some p =>
              obtain ⟨rr, j⟩ := p
              obtain ⟨hrr, hoeq, hjlt⟩ := A.decode_sound hdec
              simp only []
              congr 1
              refine addKernelMasked_frame xReg yReg outReg B n s s1 hsrc rr j
                (fun i hi hc => ?_)
              refine hnw (i, PUnit.unit) ?_
              simp only [ComputeCorrect.WriteMap.viewIf,
                ComputeCorrect.WriteMap.writeIf, if_pos hi, haddr i,
                Option.some_inj]
              rw [hr, hoeq, ← hc.1, ← hc.2]
        · rw [if_neg hr, if_neg hr]

/-! ## Part 4 — the denotation: array in, array out

The headline above still *mentions* layouts, launch states, and write maps.
`denoteAddKernel` ("⟦addKernelMasked⟧") is a one-line instantiation of the
generic denotation combinator `denoteKernel` (audited once, in
`VeriTile.Triton.Memory.Denotation`), which packages the whole ceremony —
the canonical flat allocation of a region table, the canonical start state
loaded with a slot table, the flattening translation, the flat-memory
execution, and the read-back of one output lane. What remains per kernel —
the whole audit surface of the definition below — is the region table, the
slot table, and the output address. The denotation headline
`add_kernel_correctness` then speaks only elementary mathematics: `Nat`,
`Fin`, `ℝ`, `Option`, `=`. The `#stmtSurfaceSubset` gate below pins its
statement surface to the single constant `denoteAddKernel`. -/

/-- ⟦`addKernelMasked`⟧: one instantiation of the generic `denoteKernel`.
The tables are the audit surface — regions `"x"`/`"y"`/`"out"` of extent `n`
laid end to end in one flat address space, `xs`/`ys` loaded at the program
tile `[pid*B, pid*B + B)` of `"x"`/`"y"`, and lane `i` of the output tile
read back from flat memory. All addressing lives inside `denoteKernel`. -/
noncomputable denotation denoteAddKernel (B n pid : Nat) (xs ys : Fin B → ℝ)
    (i : Fin B) : Option ℝ :=
  denoteKernel (addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n)
    ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] pid
    [.ofFin ⟨"x"⟩ (pid * B) xs, .ofFin ⟨"y"⟩ (pid * B) ys]
    ⟨"out"⟩ (pid * B + i.val)

/-- **The denotation headline**: on every in-bounds lane, ⟦`addKernelMasked`⟧
maps the input arrays to their elementwise sum. The statement mentions only
elementary mathematics — every pointer, region, layout, and write-map concept
is inside `denoteAddKernel`, audited once. -/
specification add_kernel_correctness (B n pid : Nat) (hB : 0 < B)
    (xs ys : Fin B → ℝ) (i : Fin B) (hi : pid * B + i.val < n) :
    denoteAddKernel B n pid xs ys i = some (xs i + ys i) := by
  have hnd : ((([(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] :
      List (RegionName × Nat))).map Prod.fst).Nodup := by simp
  have hd := FlatAlloc.ofList_disjoint ⟨"flat"⟩
    [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] hnd
  have hx : n ≤ (FlatAlloc.ofList ⟨"flat"⟩
      [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).extent ⟨"x"⟩ := by
    simp [FlatAlloc.listExtent]
  have hy : n ≤ (FlatAlloc.ofList ⟨"flat"⟩
      [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).extent ⟨"y"⟩ := by
    simp [FlatAlloc.listExtent]
  have hout : n ≤ (FlatAlloc.ofList ⟨"flat"⟩
      [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).extent ⟨"out"⟩ := by
    simp [FlatAlloc.listExtent]
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
  rw [addKernelMasked_exec_flatten _ hd
    (FlatAlloc.ofList_closed ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)])
    ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n hx hy hout
    (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
      DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) rfl]
  have hobs := add_kernel_masked_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩
    B n hB (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
      DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) xs ys hloadX hloadY i
  rw [show exec (addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n)
      (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys])
    = exec ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
        ).toAlgKernel) (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) from rfl] at hobs
  cases hsrc : exec ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
      ).toAlgKernel) (DenoteSlot.state pid [DenoteSlot.ofFin ⟨"x"⟩ (pid * B) xs,
        DenoteSlot.ofFin ⟨"y"⟩ (pid * B) ys]) with
  | none =>
      rw [hsrc] at hobs
      simp [observeAt] at hobs
  | some s1 =>
      rw [hsrc] at hobs
      simp only [observeAt, Option.map_some, Option.some_inj,
        DenoteSlot.state_pid, if_pos hi] at hobs
      simp only [Option.map_some, Option.some_inj]
      rw [FlatAlloc.flattenState_readMem _ hd s1
        (r := ⟨"out"⟩) (by simp)
        (o := pid * B + i.val) (by simpa [FlatAlloc.listExtent] using hi)]
      exact hobs

/-! ## Part 5 — junk tolerance: locality of the flat run

`add_kernel_correctness` runs the flattened kernel from the **canonical**
flat state, whose memory is `0` outside the loaded windows. Real memory
carries junk there. The execution-locality theorem
(`VeriTile.Triton.exec_agreeOn`) closes that gap: the flattened kernel is
trace-safe inside the flat window `[0, 3n)` of the canonical end-to-end
allocation, so **any** start state agreeing with the canonical one inside
that window — and holding arbitrary junk outside it — produces the same
in-bounds read-back. -/

set_option maxHeartbeats 1600000 in
/-- The **flattened** register-style masked vector add is trace-safe for any
flat bounds map covering the three relocated windows, from any start state:
every load/store lands at `A.base r + (pid*B + i)` on a masked lane
`pid*B + i < n`. Flat-side sibling of `addKernelMasked_traceSafe`. -/
theorem addKernelMaskedFlat_traceSafe (A : FlatAlloc)
    (xReg yReg outReg : RegionName) (B n : Nat) (fb : RegionBounds)
    (s : BlockState)
    (hx : A.base xReg + n ≤ fb A.flat) (hy : A.base yReg + n ≤ fb A.flat)
    (hout : A.base outReg + n ≤ fb A.flat) :
    Kernel.TraceSafe fb
      (A.flattenKernel
        ((addKernelMasked xReg yReg outReg B n
          ).toAlgKernel)) s := by
  have hbody : (A.flattenKernel
      ((addKernelMasked xReg yReg outReg B n
        ).toAlgKernel)).body
      = [Stmt.assign .nat [] "pid" (Op.programId 0),
         Stmt.assign .nat [B] "offsets"
           (Op.add .nat .scalarL
             (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
             (Op.arange B)),
         Stmt.assign .bool [B] "mask"
           (Op.lt .nat .scalarR (Op.ref .nat [B] "offsets") (Op.constNat n)),
         Stmt.assign .real [B] "x"
           (Op.load .real
             (MemAccess.region A.flat
               (Op.add .nat .scalarL (Op.constNat (A.base xReg))
                 (Op.ref .nat [B] "offsets")))
             (MaskOpt.mask (Op.ref .bool [B] "mask"))),
         Stmt.assign .real [B] "y"
           (Op.load .real
             (MemAccess.region A.flat
               (Op.add .nat .scalarL (Op.constNat (A.base yReg))
                 (Op.ref .nat [B] "offsets")))
             (MaskOpt.mask (Op.ref .bool [B] "mask"))),
         Stmt.assign .real [B] "output"
           (Op.add .real (.consSame .nil) (Op.ref .real [B] "x")
             (Op.ref .real [B] "y")),
         Stmt.store .real [B]
           (MemAccess.region A.flat
             (Op.add .nat .scalarL (Op.constNat (A.base outReg))
               (Op.ref .nat [B] "offsets")))
           (Op.ref .real [B] "output")
           (MaskOpt.mask (Op.ref .bool [B] "mask"))] := by
    simp [addKernelMasked, ComputeKernel.toAlgKernel,
      FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp, FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
      FlatAlloc.shiftBroadcast]
  unfold Kernel.TraceSafe
  rw [hbody]
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  -- statement 2: offsets := pid * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
      (Op.arange B)) (s.setReg "pid" .nat [] v1)
      = some ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- statement 3: mask := offsets < n
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  rw [show evalOp (Op.lt .nat .scalarR (Op.ref .nat [B] "offsets")
      (Op.constNat n))
      ((s.setReg "pid" .nat [] v1).setReg "offsets" .nat [B]
        ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩)
      = some ⟨fun i => ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩ from by
    simp [Tile.cop, BlockState.setReg]] at hv3
  obtain rfl := Option.some_inj.mp hv3
  -- shared register-readback facts at the state before the loads
  set sm := ((s.setReg "pid" .nat [] v1).setReg "offsets" .nat [B]
      ⟨fun i => (v1.data PUnit.unit) * B + i.1.val⟩).setReg "mask" .bool [B]
      ⟨fun i => ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "offsets" ∨ nm = "mask" →
        t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (base : Nat), base + n ≤ fb A.flat →
      MemAccess.ActiveAddressSafe fb
        (MemAccess.region A.flat
          (Op.add .nat .scalarL (Op.constNat base)
            (Op.ref .nat [B] "offsets"))) t
        ((MaskOpt.mask (dtype := .real) (Op.ref .bool [B] "mask")).Active t) := by
    intro t hframe base hbase
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i hact
    rw [show evalOp (Op.add .nat .scalarL (Op.constNat base)
        (Op.ref .nat [B] "offsets")) t
        = some ⟨fun i => base + ((v1.data PUnit.unit) * B + i.1.val)⟩ from by
      simp [hframe .nat [B] "offsets" (Or.inl rfl), hsm,
        BlockState.setReg, Tile.bop, NumericDType.nat_add]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    obtain ⟨masks, hm, hd⟩ := hact
    rw [show evalOp (Op.ref .bool [B] "mask") t
        = some ⟨fun i =>
          ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩ from by
      simp [hframe .bool [B] "mask" (Or.inr rfl), hsm,
        BlockState.setReg]] at hm
    obtain rfl := Option.some_inj.mp hm
    rw [ComparableDType.nat_lt_eq_true] at hd
    show base + ((v1.data PUnit.unit) * B + i.1.val) < fb A.flat
    omega
  -- statement 4: x := load(flat + (base x + offsets), mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, by simp,
      hAAS sm (fun _ _ _ _ => rfl) (A.base xReg) hx⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: y := load(flat + (base y + offsets), mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp, by simp,
      hAAS _ (fun dt sh nm hnm => ?_) (A.base yReg) hy⟩
    rcases hnm with rfl | rfl <;> simp [BlockState.setReg]
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: output := x + y
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(flat + (base out + offsets), output, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], by simp [Op.SafeAt],
    hAAS _ (fun dt sh nm hnm => ?_) (A.base outReg) hout⟩
  rcases hnm with rfl | rfl <;> simp [BlockState.setReg]

/-- **Junk tolerance of the denotation** (#487 step 2): running the
flattened masked vector add from **any** flat state `sF` that agrees with
the canonical denotation start state inside the allocated flat window
`[0, 3n)` — with arbitrary junk outside it and in unrelated regions —
reads back the same in-bounds answer `xs i + ys i`. The canonical
zero-outside-the-slots start state of `denoteKernel` is therefore no
idealization: out-of-window memory content cannot change the answer. -/
theorem add_kernel_correctness_junk (B n pid : Nat) (hB : 0 < B)
    (xs ys : Fin B → ℝ) (i : Fin B) (hi : pid * B + i.val < n)
    (sF : BlockState)
    (hagree : ((FlatAlloc.ofList ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).flattenState
        (DenoteSlot.state pid [.ofFin ⟨"x"⟩ (pid * B) xs,
          .ofFin ⟨"y"⟩ (pid * B) ys])).AgreeOn
      (RegionBounds.Window fun r => if r = ⟨"flat"⟩ then 3 * n else 0) sF) :
    (exec ((FlatAlloc.ofList ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).flattenKernel
        ((addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
          ).toAlgKernel)) sF).map
      (fun sF' => sF'.readMem ⟨"flat"⟩
        ((FlatAlloc.ofList ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)]).addr ⟨"out"⟩
          (pid * B + i.val)))
      = some (xs i + ys i) := by
  set A := FlatAlloc.ofList ⟨"flat"⟩ [(⟨"x"⟩, n), (⟨"y"⟩, n), (⟨"out"⟩, n)] with hA
  set k := (addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n).toAlgKernel
    with hk
  set sC := A.flattenState (DenoteSlot.state pid
    [.ofFin ⟨"x"⟩ (pid * B) xs, .ofFin ⟨"y"⟩ (pid * B) ys]) with hsC
  -- flat-side trace safety of the flattened kernel, window `[0, 3n)` of flat
  have hbx : A.base ⟨"x"⟩ = 0 := by simp [hA, FlatAlloc.listBase]
  have hby : A.base ⟨"y"⟩ = n := by simp [hA, FlatAlloc.listBase]
  have hbout : A.base ⟨"out"⟩ = n + n := by simp [hA, FlatAlloc.listBase]
  have hflat : A.flat = ⟨"flat"⟩ := rfl
  have hts := addKernelMaskedFlat_traceSafe A ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
    (fun r => if r = ⟨"flat"⟩ then 3 * n else 0) sC
    (by rw [hbx, hflat]; simp; omega)
    (by rw [hby, hflat]; simp; omega)
    (by rw [hbout, hflat]; simp; omega)
  -- locality: the junk state runs in lockstep with the canonical state
  have hrel := exec_agreeOn (fun r => if r = ⟨"flat"⟩ then 3 * n else 0)
    (A.flattenKernel k) sC sF hts hagree
  -- the canonical run's read-back is the denotation headline
  have hden := add_kernel_correctness B n pid hB xs ys i hi
  unfold denoteAddKernel denoteKernel at hden
  cases hC : exec (A.flattenKernel k) sC with
  | none => rw [hC] at hden; simp at hden
  | some sC' =>
      rw [hC] at hrel hden
      cases hF : exec (A.flattenKernel k) sF with
      | none => rw [hF] at hrel; cases hrel
      | some sF' =>
          rw [hF] at hrel
          cases hrel with
          | some hag' =>
              simp only [Option.map_some, Option.some_inj] at hden ⊢
              rw [← hden]
              -- the read-back cell sits inside the agreed window
              refine (BlockState.readMem_congr (hag'.mem ?_)).symm
              show A.addr ⟨"out"⟩ (pid * B + i.val)
                < if (⟨"flat"⟩ : RegionName) = ⟨"flat"⟩ then 3 * n else 0
              rw [if_pos rfl]
              have : A.addr ⟨"out"⟩ (pid * B + i.val)
                  = (n + n) + (pid * B + i.val) := by
                rw [FlatAlloc.addr, hbout]
              omega

/-! ## Trust gates -/

#axiomsClean add_kernel_masked_correct_view
#axiomsClean addKernelMaskedFlat_traceSafe
#axiomsClean add_kernel_correctness_junk
#axiomsClean flatAddKernel_exec_flatten
#axiomsClean addKernelMasked_exec_flatten
#axiomsClean add_kernel_masked_flat_correct_view
#axiomsClean add_kernel_correctness

/- The whole point of Part 4: the denotation headline's statement surface is
ONE project constant — every addressing concept is inside `denoteAddKernel`. -/
#stmtSurfaceSubset add_kernel_correctness ⊆ [denoteAddKernel]

end VeriTile.Bench.Examples.FlatVectorAdd

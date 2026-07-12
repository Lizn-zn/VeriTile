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
- `addKernelMasked` (the ordinary DSL kernel from `VectorAdd.lean`) uses the
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
import VeriTile.Examples.VectorAdd
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FlatVectorAdd

open VeriTile.Triton
open VeriTile.Triton.TensorView (ViewsLoaded slot)
open VeriTile.Examples (programTileView)

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

`addKernelMasked` (from `VectorAdd.lean`) is the ordinary DSL shape every
bench port uses: offsets and mask are computed into registers and the memory
operations address through `Op.ref`. No ∀-state contract covers that; the
per-execution `Kernel.TraceSafe` below is discharged by walking the actual
seven-statement execution, reading the offset/mask registers back at each
load and at the store. -/

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
      ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel)
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
    ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [VeriTile.Examples.addKernelMasked, ComputeKernel.toAlgKernel,
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
        ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel))
        (A.flattenState s)
      = (exec ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n
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
    ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel)
  ComputeKernel.fromKernelBody fk.inputs fk.outputs fk.body

/-- Scalar `readMem` at a translated in-bounds address reads the region-model
cell (the `.real` carrier translates identically). -/
private theorem flattenState_readMem (A : FlatAlloc) (hd : A.Disjoint)
    (s : BlockState) {r : RegionName} (hr : r ∈ A.regions) {o : Nat}
    (ho : o < A.extent r) :
    (A.flattenState s).readMem A.flat (A.addr r o) = s.readMem r o := by
  unfold BlockState.readMem
  rw [A.flattenState_mem_addr hd s hr ho, A.readAs_trCell]
  cases hcell : (s.mem r o).readAs .real with
  | none => rfl
  | some v => cases v <;> rfl

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
    (hExec : exec ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n
      ).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, s.pid * B + i.val < n →
      ¬(outReg = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, VeriTile.Examples.addKernelMasked, ComputeKernel.toAlgKernel,
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
  have h_x := VeriTile.Examples.inputLoadedAt_of_programTileView_loaded h_x'
  have h_y := VeriTile.Examples.inputLoadedAt_of_programTileView_loaded h_y'
  have hx : n ≤ A.extent xReg := hcovers (xReg, n) (by simp)
  have hy : n ≤ A.extent yReg := hcovers (yReg, n) (by simp)
  have hout : n ≤ A.extent outReg := hcovers (outReg, n) (by simp)
  have hak : (addKernelMaskedFlat A xReg yReg outReg B n).toAlgKernel
      = A.flattenKernel
        ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel) := by
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
        ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel) s with
    | none => rw [hsrc] at hExec; exact absurd hExec (by simp)
    | some s1 =>
        rw [hsrc] at hExec
        replace hExec : some (A.flattenState s1) = some s' := hExec
        obtain rfl := (Option.some_inj.mp hExec).symm
        have hobs := VeriTile.Examples.add_kernel_masked_correct
          xReg yReg outReg B n hB s xs ys h_x h_y idx.1
        rw [show exec (VeriTile.Examples.addKernelMasked xReg yReg outReg B n) s
            = exec ((VeriTile.Examples.addKernelMasked
                xReg yReg outReg B n).toAlgKernel) s from rfl, hsrc] at hobs
        simp only [VeriTile.Examples.observeAt, Option.map_some,
          Option.some_inj, if_pos hActive] at hobs
        simpa [flattenState_readMem A hd s1 houtr
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
        ((VeriTile.Examples.addKernelMasked xReg yReg outReg B n).toAlgKernel) s with
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
`denoteAddKernel` ("⟦addKernelMasked⟧") packages the whole ceremony — the
canonical flat allocation, the canonical loaded start state, the flattening
translation, the flat-memory execution, and the read-back of one output
lane — into **one** audited definition. The denotation headline
`add_kernel_denotation` then speaks only elementary mathematics: `Nat`,
`Fin`, `ℝ`, `Option`, `=`. The `#stmtSurfaceSubset` gate below pins its
statement surface to the single constant `denoteAddKernel`. -/

/-- Value held at lane `k` of a shifted `B`-tile: `arr k` in bounds, `0`
outside. The memory-content formula of `canonicalState`. -/
private def laneVal (B : Nat) (arr : Fin B → ℝ) (k : Nat) : ℝ :=
  if h : k < B then arr ⟨k, h⟩ else 0

/-- Canonical flat allocation for the three arrays: `"x"` at base `0`, `"y"`
at base `n`, `"out"` at base `2*n`, each of extent `n`, all inside the single
flat region `"flat"`. -/
def canonicalAlloc (n : Nat) : FlatAlloc where
  flat := ⟨"flat"⟩
  regions := [⟨"x"⟩, ⟨"y"⟩, ⟨"out"⟩]
  base := fun r =>
    if r = (⟨"x"⟩ : RegionName) then 0
    else if r = (⟨"y"⟩ : RegionName) then n
    else if r = (⟨"out"⟩ : RegionName) then 2 * n
    else 0
  extent := fun r =>
    if r = (⟨"x"⟩ : RegionName) ∨ r = (⟨"y"⟩ : RegionName)
        ∨ r = (⟨"out"⟩ : RegionName) then n
    else 0

/-- Canonical region-model start state for program `pid`: `xs`/`ys` held at
the program tile offsets `pid*B + i` of regions `"x"`/`"y"` (via the shifted
lane formula `laneVal`), every other cell `0`, empty registers, clean
`undef`. -/
noncomputable def canonicalState (pid B : Nat) (xs ys : Fin B → ℝ) :
    BlockState where
  mem := fun r o =>
    if r = (⟨"x"⟩ : RegionName) then MemCell.real (laneVal B xs (o - pid * B))
    else if r = (⟨"y"⟩ : RegionName) then
      MemCell.real (laneVal B ys (o - pid * B))
    else MemCell.real 0
  regs := fun _ _ _ => none
  pids := fun _ => pid
  undef := fun _ _ => 0
  numPids := fun _ => 1

/-- ⟦`addKernelMasked`⟧: run the **flattened** masked vector add (one flat
memory, real pointer arithmetic `base + pid*B + i`) from the canonical
flattened state holding `xs`/`ys`, and read lane `i` of the output array
back from flat memory. All addressing lives inside this one definition. -/
noncomputable def denoteAddKernel (B n pid : Nat) (xs ys : Fin B → ℝ)
    (i : Fin B) : Option ℝ :=
  (exec ((canonicalAlloc n).flattenKernel
      ((VeriTile.Examples.addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n).toAlgKernel))
    ((canonicalAlloc n).flattenState (canonicalState pid B xs ys))).map
    (fun sF => sF.readMem (canonicalAlloc n).flat
      ((canonicalAlloc n).addr ⟨"out"⟩ (pid * B + i.val)))

set_option linter.unnecessarySeqFocus false in
/-- The three canonical intervals `[0,n)`, `[n,2n)`, `[2n,3n)` are disjoint. -/
private theorem canonicalAlloc_disjoint (n : Nat) :
    (canonicalAlloc n).Disjoint := by
  intro r hr r' hr' hne
  simp only [canonicalAlloc, List.mem_cons, List.not_mem_nil, or_false] at hr hr'
  rcases hr with rfl | rfl | rfl <;> rcases hr' with rfl | rfl | rfl <;>
    first
      | exact absurd rfl hne
      | simp [canonicalAlloc] <;> omega

/-- Extent closure: regions outside the canonical allocation have extent 0. -/
private theorem canonicalAlloc_closed (n : Nat) :
    ∀ r, r ∉ (canonicalAlloc n).regions → (canonicalAlloc n).extent r = 0 := by
  intro r hr
  simp only [canonicalAlloc, List.mem_cons, List.not_mem_nil, or_false,
    not_or] at hr
  obtain ⟨h1, h2, h3⟩ := hr
  simp [canonicalAlloc, h1, h2, h3]

private theorem canonicalState_pid (pid B : Nat) (xs ys : Fin B → ℝ) :
    (canonicalState pid B xs ys).pid = pid := rfl

/-- The canonical state holds `xs` at the program tile of `"x"`. -/
private theorem canonicalState_loadedX (pid B : Nat) (xs ys : Fin B → ℝ) :
    VeriTile.Examples.InputLoadedAt (canonicalState pid B xs ys) ⟨"x"⟩ B xs := by
  intro i
  simp [canonicalState, BlockState.readMem, laneVal, i.isLt]

/-- The canonical state holds `ys` at the program tile of `"y"`. -/
private theorem canonicalState_loadedY (pid B : Nat) (xs ys : Fin B → ℝ) :
    VeriTile.Examples.InputLoadedAt (canonicalState pid B xs ys) ⟨"y"⟩ B ys := by
  intro i
  simp [canonicalState, BlockState.readMem, laneVal, i.isLt]

/-- **The denotation headline**: on every in-bounds lane, ⟦`addKernelMasked`⟧
maps the input arrays to their elementwise sum. The statement mentions only
elementary mathematics — every pointer, region, layout, and write-map concept
is inside `denoteAddKernel`, audited once. -/
specification add_kernel_denotation (B n pid : Nat) (hB : 0 < B)
    (xs ys : Fin B → ℝ) (i : Fin B) (hi : pid * B + i.val < n) :
    denoteAddKernel B n pid xs ys i = some (xs i + ys i) := by
  have hd := canonicalAlloc_disjoint n
  have hx : n ≤ (canonicalAlloc n).extent ⟨"x"⟩ := by simp [canonicalAlloc]
  have hy : n ≤ (canonicalAlloc n).extent ⟨"y"⟩ := by simp [canonicalAlloc]
  have hout : n ≤ (canonicalAlloc n).extent ⟨"out"⟩ := by simp [canonicalAlloc]
  unfold denoteAddKernel
  rw [addKernelMasked_exec_flatten (canonicalAlloc n) hd
    (canonicalAlloc_closed n) ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n hx hy hout
    (canonicalState pid B xs ys) rfl]
  have hobs := VeriTile.Examples.add_kernel_masked_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩
    B n hB (canonicalState pid B xs ys) xs ys
    (canonicalState_loadedX pid B xs ys) (canonicalState_loadedY pid B xs ys) i
  rw [show exec (VeriTile.Examples.addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n)
      (canonicalState pid B xs ys)
    = exec ((VeriTile.Examples.addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
        ).toAlgKernel) (canonicalState pid B xs ys) from rfl] at hobs
  cases hsrc : exec ((VeriTile.Examples.addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
      ).toAlgKernel) (canonicalState pid B xs ys) with
  | none =>
      rw [hsrc] at hobs
      simp [VeriTile.Examples.observeAt] at hobs
  | some s1 =>
      rw [hsrc] at hobs
      simp only [VeriTile.Examples.observeAt, Option.map_some, Option.some_inj,
        canonicalState_pid, if_pos hi] at hobs
      simp only [Option.map_some, Option.some_inj]
      rw [flattenState_readMem (canonicalAlloc n) hd s1
        (r := ⟨"out"⟩) (by simp [canonicalAlloc])
        (o := pid * B + i.val) (by simpa [canonicalAlloc] using hi)]
      exact hobs

/-! ## Trust gates -/

#axiomsClean flatAddKernel_exec_flatten
#axiomsClean addKernelMasked_exec_flatten
#axiomsClean add_kernel_masked_flat_correct_view
#axiomsClean add_kernel_denotation

/- The whole point of Part 4: the denotation headline's statement surface is
ONE project constant — every addressing concept is inside `denoteAddKernel`. -/
#stmtSurfaceSubset add_kernel_denotation ⊆ [denoteAddKernel]

end VeriTile.Bench.Examples.FlatVectorAdd

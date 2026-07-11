/-
VeriTile.Examples.FlatVectorAdd

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
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.VectorAdd

namespace VeriTile.Examples.FlatVectorAdd

open VeriTile.Triton

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
  simp [offsExpr, evalOp, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
    Tile.vec]

/-- Any active lane of the inline mask has its address below `n` — in every
state, because the mask and the address share the `program_id` subterm. -/
private theorem active_lane_bound (B n : Nat) (s : BlockState)
    (i : TileIndex [B])
    (hact : (MaskOpt.mask (dtype := .real) (maskExpr B n)).Active s i) :
    s.pids 0 * B + i.1.val < n := by
  obtain ⟨masks, hmasks, hdata⟩ := hact
  simp [maskExpr, evalOp, evalOp_offsExpr, Tile.cop] at hmasks
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
    · refine ⟨⟨msOffs _, by simp [Op.SafeAt], msMaskE _, bnd outReg hout _⟩, ?_⟩
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
    simp [evalOp, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
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
    simp [evalOp, Tile.cop, BlockState.setReg]] at hv3
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
      simp [evalOp, hframe .nat [B] "offsets" (Or.inl rfl), hsm,
        BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    obtain ⟨masks, hm, hd⟩ := hact
    rw [show evalOp (Op.ref .bool [B] "mask") t
        = some ⟨fun i =>
          ComparableDType.nat.lt ((v1.data PUnit.unit) * B + i.1.val) n⟩ from by
      simp [evalOp, hframe .bool [B] "mask" (Or.inr rfl), hsm,
        BlockState.setReg]] at hm
    obtain rfl := Option.some_inj.mp hm
    rw [ComparableDType.nat_lt_eq_true] at hd
    simp only [Region.cast_self]
    exact lt_of_lt_of_le hd hreg
  -- statement 4: x := load(xReg + offsets, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · refine ?_
    simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp [Op.SafeAt], by simp [Op.SafeAt],
      hAAS sm (fun _ _ _ _ => rfl) xReg hx⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: y := load(yReg + offsets, mask)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt],
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


end VeriTile.Examples.FlatVectorAdd

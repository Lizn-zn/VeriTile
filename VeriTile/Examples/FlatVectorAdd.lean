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

Why the addresses and masks are written *inline* (rather than the usual
`offs := ...; tl.load(x + offs, mask=offs < n)` register style): the #48
`MemorySafe` contract quantifies over **all** states, and a register-indirect
address is unconstrained in an arbitrary state — the contract is only
provable when the mask and the address are visibly coupled (here through the
shared `program_id(0)` subterm). Register-style kernels need a
per-execution (trace-level) safety variant of the bridge; that is the
documented next step on the address-layout roadmap.
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten

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

/-- The kernel is memory-safe at any extent map covering `n` on all three
regions. -/
theorem flatAddKernel_memorySafe (xReg yReg outReg : RegionName) (B n : Nat)
    (bounds : RegionBounds)
    (hx : n ≤ bounds xReg) (hy : n ≤ bounds yReg) (hout : n ≤ bounds outReg) :
    (flatAddKernel xReg yReg outReg B n).MemorySafe bounds := by
  have msOffs : (offsExpr B).MemorySafe bounds := by
    simp [offsExpr, Op.MemorySafe]
  have msMaskE : (maskExpr B n).MemorySafe bounds := by
    simp [maskExpr, offsExpr, Op.MemorySafe]
  have msVal : (Op.add .real (.consSame .nil)
      (.ref .real [B] "x") (.ref .real [B] "y")).MemorySafe bounds := by
    simp [Op.MemorySafe]
  have bnd : ∀ (reg : RegionName), n ≤ bounds reg → ∀ (s : BlockState),
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (offsExpr B)) s
        ((MaskOpt.mask (dtype := .real) (maskExpr B n)).Active s) := by
    intro reg hreg s
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i hact
    rw [evalOp_offsExpr] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    have hlt := active_lane_bound B n s i hact
    simp only [Region.cast_self]
    exact lt_of_lt_of_le hlt hreg
  unfold Kernel.MemorySafe flatAddKernel
  unfold StmtList.MemorySafe
  simp only [Stmt.MemorySafeList, Stmt.MemorySafe, Op.MemorySafe]
  repeat' apply And.intro
  all_goals first
    | trivial
    | exact msOffs
    | exact msVal
    | exact msMaskE
    | exact bnd xReg hx
    | exact bnd yReg hy
    | exact bnd outReg hout

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
    (flatAddKernel_memorySafe xReg yReg outReg B n A.extent hx hy hout)
    (flatAddKernel_flattenOk xReg yReg outReg B n) hu

end VeriTile.Examples.FlatVectorAdd

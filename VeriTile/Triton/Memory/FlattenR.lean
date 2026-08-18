/-
VeriTile.Triton.Memory.FlattenR

Rounding-model (`execR`) version of the flat-memory bridge in
`VeriTile.Triton.Memory.Flatten`.

The bridge theorems there are stated for the exact stepper `exec`. This file
proves the isomorphic statements for the rounding-model-parametric stepper
`execR R` (#447): running the *translated* kernel under `execR R` on the
*flattened* state is the flattening of the source `execR R` run.

Everything syntactic is reused verbatim from `Flatten` (`FlatAlloc`,
`trCarrier`/`trTile`, `flattenState`, `flattenOp`/`flattenStmt`/
`flattenKernel`, `FlattenOk`, the decode/read/write lemmas). The rounding
model only ever acts on the four float channels (`R.cast` at `Op.castFloat`,
`R.storeValue` inside `writeMemTypedR`), and the value translation
`trCarrier` is the identity on every float channel — so `R` commutes with the
translation by congruence: both sides apply the *same* `R.cast`/`R.storeValue`
to equal real values.

Safety contracts mirror bridge v1.2's per-execution (trace-level) story with
`evalOpR R`/`stepStmtR R` in place of `evalOp`/`stepStmt`:

- `Op.SafeAtR R` mirrors `Op.SafeAt` (single-state active-address bounds).
  Its `ptrSub` clause additionally carries the per-state no-underflow side
  condition on the `R`-evaluated pointers/offsets — under `R`, offset
  expressions can round differently than under the exact evaluator, so the
  `∀`-state `evalOp`-based clause inside `Op.FlattenOk` does not cover the
  `R`-run; the per-execution home for it is the safety predicate.
- `MaskOpt.ActiveR R` / `MemAccess.ActiveAddressSafeR R` mirror
  `MaskOpt.Active` / `MemAccess.ActiveAddressSafe` (the redundant embedded
  `Op.MemorySafe` conjunct of the exact `.ptr`/`.blockPtr` forms is dropped —
  the bridge proofs never used it).
- `Stmt.TraceSafeR R` / `Stmt.TraceSafeListR R` / `Kernel.TraceSafeR R`
  thread safety along the actual `stepStmtR R` execution.

The syntactic fragment is `FlattenOkR`: `FlattenOk` with the `∀`-state
no-underflow `ptrSub` conjunct dropped — the `R` bridge discharges underflow
from the per-trace `Op.SafeAtR` clause instead, which is what admits
reverse-pointer-decrement loop kernels. `FlattenOk` still implies it
(`Kernel.FlattenOk.toR`), and `execR_flatten` keeps the exact-fragment
statement as a corollary of `execR_flattenR`. Pointer-dtype element
loads/reshape stay excluded; `atomicRMW`'s clause is the exact-side one
verbatim (it delegates to the exact semantics).
-/

import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Float.StepR

namespace VeriTile.Triton

/-! ## Address-shift evaluation under `R` -/

/-- `R`-version of `evalOp_shiftAdd`: evaluating the translated offset
(`base + off`, as built by `flattenAccess`) under `evalOpR R` shifts every
lane of the source offsets by `b` (the shift is pure `Nat` arithmetic; `R`
never fires). -/
theorem evalOpR_shiftAdd (R : RoundingModel) {sh : TileShape} (b : Nat)
    (off : Op .nat sh) (s : BlockState) :
    evalOpR R (.add .nat (FlatAlloc.shiftBroadcast sh) (.constNat b) off) s
      = (evalOpR R off s).map fun t => ⟨fun i => b + t.data i⟩ := by
  cases hoff : evalOpR R off s with
  | none => cases sh <;> simp [evalOpR, hoff, FlatAlloc.shiftBroadcast]
  | some t =>
      cases sh with
      | nil =>
          simp [evalOpR, hoff, FlatAlloc.shiftBroadcast, Tile.bop,
            NumericDType.nat_add]
          rfl
      | cons n r =>
          simp [evalOpR, hoff, FlatAlloc.shiftBroadcast, Tile.bop,
            NumericDType.nat_add]

/-! ## Per-execution safety under `R` (mirrors bridge v1.2)

The `evalOpR R`-side mirrors of `MaskOpt.Active`,
`MemAccess.ActiveAddressSafe` and `Op.SafeAt` from
`VeriTile.Triton.Memory.Bounds` / `Flatten`. -/

namespace MaskOpt

/-- Lanes active for a memory operation under a mask in state `s`, with the
mask evaluated by `evalOpR R` (mirror of `MaskOpt.Active`). -/
def ActiveR (R : RoundingModel) (s : BlockState) :
    (mask : MaskOpt dtype shape) → TileIndex shape → Prop
  | .none => fun _ => True
  | .mask m => fun i =>
      ∃ masks, evalOpR R m s = some masks ∧ masks.data i = true
  | .maskOther m _ => fun i =>
      ∃ masks, evalOpR R m s = some masks ∧ masks.data i = true

end MaskOpt

/-- Active `R`-evaluated addresses of a memory access form are in bounds
(mirror of `memAccessActiveAddressSafe`; the redundant `Op.MemorySafe`
conjunct of the `.ptr`/`.blockPtr` forms is dropped — the bridge never
uses it). -/
def memAccessActiveAddressSafeR {dtype : TileDType} (R : RoundingModel)
    (bounds : RegionBounds) (mem : MemAccess dtype shape) (s : BlockState)
    (active : TileIndex shape → Prop) : Prop :=
  match mem with
  | .region region off =>
      ∀ offsets, evalOpR R off s = some offsets →
        ∀ i : TileIndex shape, active i →
          offsets.data i < bounds (Region.cast region)
  | .ptr ptr =>
      ∀ ptrs, evalOpR R ptr s = some ptrs →
        ∀ i : TileIndex shape,
          active i → (ptrs.data i).2 < bounds (ptrs.data i).1
  | .blockPtr ptr boundaryCheck =>
      ∀ ptrs, evalOpR R ptr s = some ptrs →
        ∀ i : TileIndex shape,
          active i →
          let bp := ptrs.data i
          let idx := TileShape.indexToList shape i
          bp.inBounds idx boundaryCheck = true →
            bp.address idx < bounds bp.region

/-- Namespace-facing wrapper for `R`-active address bounds. -/
abbrev MemAccess.ActiveAddressSafeR {dtype : TileDType} (R : RoundingModel)
    (bounds : RegionBounds) (mem : MemAccess dtype shape) (s : BlockState)
    (active : TileIndex shape → Prop) : Prop :=
  memAccessActiveAddressSafeR R bounds mem s active

set_option maxHeartbeats 1600000 in
/-- `evalOpR R` mirror of `Op.SafeAt`: every load's active lanes are in
bounds **in the given state**, with masks/addresses evaluated under `R`.
The `ptrSub` clause additionally carries the per-state no-underflow side
condition (see the module docstring). -/
def Op.SafeAtR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .numPrograms _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.SafeAtR R bounds s
  | .full _ e => e.SafeAtR R bounds s
  | .castFloat _ _ e => e.SafeAtR R bounds s
  | .castNatToInt e => e.SafeAtR R bounds s
  | .castIntToNat e => e.SafeAtR R bounds s
  | .castRealToInt8 e => e.SafeAtR R bounds s
  | .add _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .sub _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .mul _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .div _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .floorDiv _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .mod _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .bitAnd _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .bitOr _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .bitXor _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .shiftLeft _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .shiftRight _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .exp a => a.SafeAtR R bounds s
  | .exp2 a => a.SafeAtR R bounds s
  | .log a => a.SafeAtR R bounds s
  | .log2 a => a.SafeAtR R bounds s
  | .sigmoid a => a.SafeAtR R bounds s
  | .sqrt a => a.SafeAtR R bounds s
  | .rsqrt a => a.SafeAtR R bounds s
  | .tanh a => a.SafeAtR R bounds s
  | .sin a => a.SafeAtR R bounds s
  | .cos a => a.SafeAtR R bounds s
  | .tan a => a.SafeAtR R bounds s
  | .atan a => a.SafeAtR R bounds s
  | .cosh a => a.SafeAtR R bounds s
  | .sinh a => a.SafeAtR R bounds s
  | .erf a => a.SafeAtR R bounds s
  | .lt _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .le _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .eq _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .gt _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .ge _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .ne _ _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .boolAnd _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .boolOr _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .boolNot a => a.SafeAtR R bounds s
  | .max2 _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .pow _ a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .where c a b => c.SafeAtR R bounds s ∧ a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .ite c a b => c.SafeAtR R bounds s ∧ a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .reduceMax _ _ a => a.SafeAtR R bounds s
  | .reduceMaxNat _ _ a => a.SafeAtR R bounds s
  | .reduceSum _ _ a => a.SafeAtR R bounds s
  | .scan _ _ _ a => a.SafeAtR R bounds s
  | .argMax _ a => a.SafeAtR R bounds s
  | .argMin _ a => a.SafeAtR R bounds s
  | .sort _ a => a.SafeAtR R bounds s
  | .dot a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .dotInt a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .transpose a => a.SafeAtR R bounds s
  | .reshape _ a => a.SafeAtR R bounds s
  | .remap _ _ a => a.SafeAtR R bounds s
  | .join a b => a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s
  | .split _ a => a.SafeAtR R bounds s
  | .expandDim _ a => a.SafeAtR R bounds s
  | .ptrBase _ => True
  | .ptrAdd _ ptr off => ptr.SafeAtR R bounds s ∧ off.SafeAtR R bounds s
  | .ptrSub bc ptr off => ptr.SafeAtR R bounds s ∧ off.SafeAtR R bounds s ∧
      -- Per-state no-underflow on the *R-evaluated* pointers/offsets: the
      -- `evalOp`-based `∀`-state clause of `Op.FlattenOk` does not cover the
      -- `R`-run (offset expressions may round differently under `R`).
      ∀ ptrs offs, evalOpR R ptr s = some ptrs → evalOpR R off s = some offs →
        ∀ i, offs.data (bc.rightIndex i) ≤ (ptrs.data (bc.leftIndex i)).2
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => base.SafeAtR R bounds s
  | .makeBlockPtrDynOffsets _ base _ _ _ offsets =>
      base.SafeAtR R bounds s ∧ ∀ off ∈ offsets, off.SafeAtR R bounds s
  | .advanceBlockPtr ptr _ => ptr.SafeAtR R bounds s
  | .load _ mem mask =>
      (match mem with
        | .region _ off => off.SafeAtR R bounds s
        | .ptr ptr => ptr.SafeAtR R bounds s
        | .blockPtr ptr _ => ptr.SafeAtR R bounds s) ∧
      (match mask with
        | .none => True
        | .mask m => m.SafeAtR R bounds s
        | .maskOther m other => m.SafeAtR R bounds s ∧ other.SafeAtR R bounds s) ∧
      mem.ActiveAddressSafeR R bounds s (mask.ActiveR R s)
  | .natToReal a => a.SafeAtR R bounds s
  | .intToReal a => a.SafeAtR R bounds s

/-- `mapM`-level commutation for dynamic block-pointer offset lists: if every
member op evaluates identically after translation, so does the scalar-offset
`mapM`. -/
private theorem mapM_data_flattenR (R : RoundingModel) (A : FlatAlloc) (s' s : BlockState) :
    ∀ (l : List (Op TileDType.nat [])),
      (∀ off ∈ l, evalOpR R (A.flattenOp off) s' = evalOpR R off s) →
      (l.map A.flattenOp).mapM (fun off => do
          let v ← evalOpR R off s'
          some (v.data PUnit.unit))
        = l.mapM (fun off => do
            let v ← evalOpR R off s
            some (v.data PUnit.unit))
  | [], _ => rfl
  | o :: rest, h => by
      simp only [List.map_cons, List.mapM_cons]
      rw [h o (List.mem_cons_self ..)]
      cases evalOpR R o s with
      | none => rfl
      | some v =>
          rw [mapM_data_flattenR R A s' s rest
            (fun off hoff => h off (List.mem_cons_of_mem _ hoff))]

set_option maxHeartbeats 3200000 in
/-- **Op-level bridge under `R`**: on the `FlattenOkR` fragment, evaluating
the translated op with `evalOpR R` in the flattened state returns the
translated value. The in-bounds obligations come from the per-execution
`Op.SafeAtR R` contract at the allocation's extents; `hcov` turns "active
address below extent" into membership in the allocation's region list
(regions outside it have extent `0`, so no active access can touch them). -/
theorem FlatAlloc.evalOpR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) :
    ∀ {d : TileDType} {sh : TileShape} (e : Op d sh) (s : BlockState),
      e.SafeAtR R A.extent s → e.FlattenOkR → s.undef = (fun _ _ => 0) →
      evalOpR R (A.flattenOp e) (A.flattenState s)
        = (evalOpR R e s).map A.trTile
  | _, _, .const c, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .constFloat fd c, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]
      rw [A.trTileFun_data fd.toTileDType_ne_ptr fd.toTileDType_ne_blockPtr]; rfl
  | _, _, .constNat n, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .constInt n, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .constBool b, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .negInf, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .programId ax, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .numPrograms ax, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .ref d sh n, s, _, _, _ => by
      simp only [flattenOp, evalOpR]; rfl
  | _, _, .arange n, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .broadcast e sh, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R e s hms hok hu]
      cases evalOpR R e s <;> rfl
  | _, _, .full sh e, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R e s hms hok hu]
      cases evalOpR R e s <;> rfl
  | _, _, .castFloat src dst a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data src.toTileDType_ne_ptr src.toTileDType_ne_blockPtr,
        A.trTileFun_data dst.toTileDType_ne_ptr dst.toTileDType_ne_blockPtr,
        Option.map_id, id_eq]
  | _, _, .add nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .sub nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .mul nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .div nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .floorDiv nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .mod nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .lt nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .le nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .eq nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .gt nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .ge nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .ne nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitAnd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitOr bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitXor bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .shiftLeft bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .shiftRight bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolAnd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolOr bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .max2 bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .pow bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .dot a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .dotInt a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .int) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .exp a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .exp2 a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .log a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .log2 a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sigmoid a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sqrt a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .rsqrt a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .tanh a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sin a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .cos a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .tan a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .atan a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .cosh a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sinh a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .erf a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolNot a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .natToReal a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .intToReal a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .int) (by decide) (by decide),
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castNatToInt a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        A.trTileFun_data (d := .int) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castIntToNat a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .int) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castRealToInt8 a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .int) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sort ax a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .scan op ax dir a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .argMax ax a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .argMin ax a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceMax ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceMaxNat ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceSum ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .where c a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hmc, hma, hmb⟩ := hms
      obtain ⟨hkc, hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R c s hmc hkc hu,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide), Option.map_id, id_eq]
      cases evalOpR R c s <;> cases evalOpR R a s <;> cases evalOpR R b s <;>
        simp [A.trTile_select]
  | _, _, .ite c a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hmc, hma, hmb⟩ := hms
      obtain ⟨hkc, hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R c s hmc hkc hu,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide), Option.map_id, id_eq]
      cases hvc : evalOpR R c s with
      | none => rfl
      | some vc => by_cases hb : vc.data PUnit.unit = true <;> simp [hb]
  | _, _, .transpose a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu]
      cases evalOpR R a s <;> simp [A.trTile_transpose]
  | _, _, .reshape outSh a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨⟨h1, h2⟩, hka⟩ := hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hka hu,
        A.trTileFun_data h1 h2, Option.map_id, id_eq]
  | _, _, .remap outSh f a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu]
      cases evalOpR R a s <;> rfl
  | _, _, .join a b, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R a s hma hka hu,
        evalOpR_flatten A hd hcov R b s hmb hkb hu]
      cases evalOpR R a s <;> cases evalOpR R b s <;> simp [A.trTile_join]
  | _, _, .split side a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu]
      cases evalOpR R a s <;> rfl
  | _, _, .expandDim ax a, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R a s hms hok hu]
      cases evalOpR R a s <;> rfl
  | _, _, .ptrBase r, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]
      exact congrArg some (Tile.ext fun i => by
        show ((Region.cast A.flat : RegionName), 0 + A.base (Region.cast r))
            = (A.flat, A.addr (Region.cast r) 0)
        simp [FlatAlloc.addr])
  | _, _, .ptrAdd bc p o, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hmp, hmo⟩ := hms
      obtain ⟨hkp, hko⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R p s hmp hkp hu,
        evalOpR_flatten A hd hcov R o s hmo hko hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide), Option.map_id, id_eq]
      cases evalOpR R p s <;> cases evalOpR R o s <;> simp [A.trTile_ptrAdd]
  | _, _, .ptrSub bc p o, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hkp, hko⟩ := hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R p s hms.1 hkp hu,
        evalOpR_flatten A hd hcov R o s hms.2.1 hko hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
      cases hps : evalOpR R p s with
      | none => rfl
      | some ps =>
          cases hos : evalOpR R o s with
          | none => rfl
          | some os =>
              refine congrArg some (Tile.ext fun i => ?_)
              have hle := hms.2.2 ps os hps hos i
              show (A.flat,
                  A.addr (ps.data (bc.leftIndex i)).1
                      (ps.data (bc.leftIndex i)).2
                    - os.data (bc.rightIndex i))
                = (A.flat, A.addr (ps.data (bc.leftIndex i)).1
                    ((ps.data (bc.leftIndex i)).2 - os.data (bc.rightIndex i)))
              have harith :
                  A.addr (ps.data (bc.leftIndex i)).1
                      (ps.data (bc.leftIndex i)).2
                    - os.data (bc.rightIndex i)
                  = A.addr (ps.data (bc.leftIndex i)).1
                      ((ps.data (bc.leftIndex i)).2
                        - os.data (bc.rightIndex i)) := by
                simp only [FlatAlloc.addr]
                exact Nat.add_sub_assoc hle _
              rw [harith]
  | _, _, .makeBlockPtr r bo ps bs strides offs, s, _, _, _ => by
      simp only [flattenOp, evalOpR, Option.map_some]; rfl
  | _, _, .makeBlockPtrDyn r bo ps bs strides offs, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R bo s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide), Option.map_id, id_eq]
      cases hbo : evalOpR R bo s with
      | none => rfl
      | some b =>
          simp [Tile.bop, NumericDType.nat_add, FlatAlloc.trTile,
            FlatAlloc.trCarrier, FlatAlloc.addr]
  | _, _, .makeBlockPtrDynOffsets r bo ps bs strides offs, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      obtain ⟨hkb, hkoffs⟩ := hok
      obtain ⟨hmb, hmoffs⟩ := hms
      have hmem : ∀ off ∈ offs,
          evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s :=
        fun off hoff => by
          rw [evalOpR_flatten A hd hcov R off s (hmoffs off hoff)
              (hkoffs off hoff) hu,
            A.trTileFun_data (d := .nat) (by decide) (by decide),
            Option.map_id, id_eq]
      have hattach : (offs.attach.map fun o => A.flattenOp o.1)
          = offs.map A.flattenOp := by
        simp
      simp only [flattenOp, evalOpR,
        evalOpR_flatten A hd hcov R bo s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
      cases hbo : evalOpR R bo s with
      | none => rfl
      | some b =>
          rw [show (offs.attach.map fun o => A.flattenOp o.1)
              = offs.map A.flattenOp from hattach,
            mapM_data_flattenR R A (A.flattenState s) s offs hmem]
          cases hoffl : offs.mapM (fun off => do
              let v ← evalOpR R off s
              some (v.data PUnit.unit)) with
          | none => rfl
          | some offlist =>
              simp [Tile.bop, NumericDType.nat_add, FlatAlloc.trTile,
                FlatAlloc.trCarrier, FlatAlloc.addr]
  | _, _, .advanceBlockPtr p ds, s, hms, hok, hu => by
      simp only [Op.SafeAtR] at hms
      simp only [Op.FlattenOkR] at hok
      simp only [flattenOp, evalOpR, evalOpR_flatten A hd hcov R p s hms hok hu]
      cases evalOpR R p s <;>
        simp [FlatAlloc.trTile, FlatAlloc.trCarrier_advance]
  | _, _, .load d mem mask, s, hms, hok, hu => by
      rw [FlatAlloc.flattenOp_load]
      cases mem with
      | region r off =>
          cases mask with
          | none =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHoff : evalOpR R (A.flattenOp off) (A.flattenState s)
                  = evalOpR R off s := by
                rw [evalOpR_flatten A hd hcov R off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hoffs : evalOpR R off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    evalOpR_shiftAdd, IHoff, hoffs]
              | some offs =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOpR, evalOpR_shiftAdd, IHoff, hoffs, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  have hbound : offs.data i < A.extent (Region.cast r) :=
                    haddr offs hoffs i trivial
                  have hreg : (Region.cast r : RegionName) ∈ A.regions := by
                    by_contra hnr
                    rw [hcov _ hnr] at hbound
                    exact absurd hbound (Nat.not_lt_zero _)
                  rw [show A.base (Region.cast r) + offs.data i
                      = A.addr (Region.cast r) (offs.data i) from rfl]
                  rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                    A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHoff : evalOpR R (A.flattenOp off) (A.flattenState s)
                  = evalOpR R off s := by
                rw [evalOpR_flatten A hd hcov R off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hoffs : evalOpR R off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    evalOpR_shiftAdd, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, evalOpR_shiftAdd, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, evalOpR_shiftAdd, IHoff, hoffs, IHm, hm,
                        Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      by_cases hact : ms.data i = true
                      · have hbound : offs.data i < A.extent (Region.cast r) :=
                          haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hreg : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hbound
                          exact absurd hbound (Nat.not_lt_zero _)
                        simp only [hact, if_true]
                        rw [show A.base (Region.cast r) + offs.data i
                            = A.addr (Region.cast r) (offs.data i) from rfl]
                        rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                          A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHoff : evalOpR R (A.flattenOp off) (A.flattenState s)
                  = evalOpR R off s := by
                rw [evalOpR_flatten A hd hcov R off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOpR R (A.flattenOp other) (A.flattenState s)
                  = evalOpR R other s := by
                rw [evalOpR_flatten A hd hcov R other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hoffs : evalOpR R off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    evalOpR_shiftAdd, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, evalOpR_shiftAdd, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      cases ho : evalOpR R other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOpR, evalOpR_shiftAdd, IHoff, hoffs, IHm, hm,
                            IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOpR, evalOpR_shiftAdd,
                            IHoff, hoffs, IHm, hm, IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          by_cases hact : ms.data i = true
                          · have hbound : offs.data i
                                < A.extent (Region.cast r) :=
                              haddr offs hoffs i ⟨ms, hm, hact⟩
                            have hreg : (Region.cast r : RegionName)
                                ∈ A.regions := by
                              by_contra hnr
                              rw [hcov _ hnr] at hbound
                              exact absurd hbound (Nat.not_lt_zero _)
                            simp only [hact, if_true]
                            rw [show A.base (Region.cast r) + offs.data i
                                = A.addr (Region.cast r) (offs.data i) from rfl]
                            rw [A.flattenState_readMemValue hd s hreg hbound
                                d hdt,
                              A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]
      | ptr p =>
          cases mask with
          | none =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOpR, IHp, hps, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  have hbound : (ps.data i).2 < A.extent (ps.data i).1 :=
                    haddr ps hps i trivial
                  have hreg : (ps.data i).1 ∈ A.regions := by
                    by_contra hnr
                    rw [hcov _ hnr] at hbound
                    exact absurd hbound (Nat.not_lt_zero _)
                  show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                  rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                    A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm, Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      by_cases hact : ms.data i = true
                      · have hbound : (ps.data i).2 < A.extent (ps.data i).1 :=
                          haddr ps hps i ⟨ms, hm, hact⟩
                        have hreg : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hbound
                          exact absurd hbound (Nat.not_lt_zero _)
                        simp only [hact, if_true]
                        show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                        rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                          A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOpR R (A.flattenOp other) (A.flattenState s)
                  = evalOpR R other s := by
                rw [evalOpR_flatten A hd hcov R other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOpR R other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOpR, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOpR, IHp, hps, IHm, hm,
                            IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          by_cases hact : ms.data i = true
                          · have hbound : (ps.data i).2
                                < A.extent (ps.data i).1 :=
                              haddr ps hps i ⟨ms, hm, hact⟩
                            have hreg : (ps.data i).1 ∈ A.regions := by
                              by_contra hnr
                              rw [hcov _ hnr] at hbound
                              exact absurd hbound (Nat.not_lt_zero _)
                            simp only [hact, if_true]
                            show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                            rw [A.flattenState_readMemValue hd s hreg hbound
                                d hdt,
                              A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]
      | blockPtr p bc =>
          cases mask with
          | none =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOpR, IHp, hps, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  rw [A.trCarrier_blockPtr_region, A.trCarrier_blockPtr_address,
                    A.trCarrier_blockPtr_inBounds]
                  by_cases hib : (ps.data i).inBounds
                      (TileShape.indexToList _ i) bc = true
                  · have hbound := haddr ps hps i trivial hib
                    have hreg : (ps.data i).region ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hbound
                      exact absurd hbound (Nat.not_lt_zero _)
                    simp only [hib, if_true]
                    rw [show A.base (ps.data i).region
                          + (ps.data i).address (TileShape.indexToList _ i)
                        = A.addr (ps.data i).region
                          ((ps.data i).address (TileShape.indexToList _ i))
                        from rfl]
                    rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                      A.trCarrier_data hdt.1 hdt.2]
                  · simp only [Bool.not_eq_true] at hib
                    simp only [hib, Bool.false_eq_true, if_false]
                    rw [A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm, Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      rw [A.trCarrier_blockPtr_region,
                        A.trCarrier_blockPtr_address,
                        A.trCarrier_blockPtr_inBounds]
                      by_cases hact : ms.data i = true
                      · simp only [hact, if_true]
                        by_cases hib : (ps.data i).inBounds
                            (TileShape.indexToList _ i) bc = true
                        · have hbound := haddr ps hps i ⟨ms, hm, hact⟩ hib
                          have hreg : (ps.data i).region ∈ A.regions := by
                            by_contra hnr
                            rw [hcov _ hnr] at hbound
                            exact absurd hbound (Nat.not_lt_zero _)
                          simp only [hib, if_true]
                          rw [show A.base (ps.data i).region
                                + (ps.data i).address (TileShape.indexToList _ i)
                              = A.addr (ps.data i).region
                                ((ps.data i).address (TileShape.indexToList _ i))
                              from rfl]
                          rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                            A.trCarrier_data hdt.1 hdt.2]
                        · simp only [Bool.not_eq_true] at hib
                          simp only [hib, Bool.false_eq_true, if_false]
                          rw [A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAtR] at hms
              simp only [Op.FlattenOkR] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafeR,
                memAccessActiveAddressSafeR] at haddr
              have IHp := evalOpR_flatten A hd hcov R p s hmsmem hokmem hu
              have IHm : evalOpR R (A.flattenOp m) (A.flattenState s)
                  = evalOpR R m s := by
                rw [evalOpR_flatten A hd hcov R m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOpR R (A.flattenOp other) (A.flattenState s)
                  = evalOpR R other s := by
                rw [evalOpR_flatten A hd hcov R other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hps : evalOpR R p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOpR,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOpR R m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOpR, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOpR R other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOpR, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOpR, IHp, hps, IHm, hm,
                            IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          rw [A.trCarrier_blockPtr_region,
                            A.trCarrier_blockPtr_address,
                            A.trCarrier_blockPtr_inBounds]
                          by_cases hact : ms.data i = true
                          · simp only [hact, if_true]
                            by_cases hib : (ps.data i).inBounds
                                (TileShape.indexToList _ i) bc = true
                            · have hbound := haddr ps hps i ⟨ms, hm, hact⟩ hib
                              have hreg : (ps.data i).region ∈ A.regions := by
                                by_contra hnr
                                rw [hcov _ hnr] at hbound
                                exact absurd hbound (Nat.not_lt_zero _)
                              simp only [hib, if_true]
                              rw [show A.base (ps.data i).region
                                    + (ps.data i).address
                                      (TileShape.indexToList _ i)
                                  = A.addr (ps.data i).region
                                    ((ps.data i).address
                                      (TileShape.indexToList _ i))
                                  from rfl]
                              rw [A.flattenState_readMemValue hd s hreg hbound
                                  d hdt,
                                A.trCarrier_data hdt.1 hdt.2]
                            · simp only [Bool.not_eq_true] at hib
                              simp only [hib, Bool.false_eq_true, if_false]
                              rw [A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]

/-- Address-form children are safe at `s`. -/
def MemAccess.SafeAtR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState) :
    {d : TileDType} → {sh : TileShape} → MemAccess d sh → Prop
  | _, _, .region _ off => off.SafeAtR R bounds s
  | _, _, .ptr p => p.SafeAtR R bounds s
  | _, _, .blockPtr p _ => p.SafeAtR R bounds s

/-- Mask children are safe at `s`. -/
def MaskOpt.SafeAtR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState) :
    {d : TileDType} → {sh : TileShape} → MaskOpt d sh → Prop
  | _, _, .none => True
  | _, _, .mask m => m.SafeAtR R bounds s
  | _, _, .maskOther m o => m.SafeAtR R bounds s ∧ o.SafeAtR R bounds s

/-! ## Per-execution statement safety under `R` -/

mutual

/-- Per-execution safety: the statement's own accesses are in bounds in the
state it runs in, and every successor state reached by the actual execution
is recursively safe. A failing step has no successors to constrain. -/
def Stmt.TraceSafeR (R : RoundingModel) (bounds : RegionBounds) : Stmt → BlockState → Prop
  | .assign _ _ _ e, s => e.SafeAtR R bounds s
  | .store _ _ mem val mask, s =>
      mem.SafeAtR R bounds s ∧ val.SafeAtR R bounds s ∧ mask.SafeAtR R bounds s ∧
      mem.ActiveAddressSafeR R bounds s (mask.ActiveR R s)
  | .atomicAdd _ _ mem val mask, s =>
      mem.SafeAtR R bounds s ∧ val.SafeAtR R bounds s ∧ mask.SafeAtR R bounds s ∧
      mem.ActiveAddressSafeR R bounds s (mask.ActiveR R s)
  | .atomicRMW _ _ _ mem input extra mask _, s =>
      -- RMW delegates to the exact semantics, so its per-execution contract
      -- is the exact-side one (see `stepStmtR`'s deliberate delegation).
      mem.SafeAt bounds s ∧ input.SafeAt bounds s ∧
      extra.elim True (·.SafeAt bounds s) ∧ mask.SafeAt bounds s ∧
      mem.ActiveAddressSafe bounds s (mask.Active s)
  | .forLoop idx n body, s => Stmt.forLoopTraceSafeR R bounds idx 0 n body s
  | .forRange idx start stop step body, s =>
      Stmt.forRangeTraceSafeR R bounds idx start stop step body s
  | .forRangeDyn idx start stop step body, s =>
      start.SafeAtR R bounds s ∧ stop.SafeAtR R bounds s ∧ step.SafeAtR R bounds s ∧
      (match evalOpR R start s, evalOpR R stop s, evalOpR R step s with
        | some a, some b, some c =>
            Stmt.forRangeTraceSafeR R bounds idx (a.data PUnit.unit)
              (b.data PUnit.unit) (c.data PUnit.unit) body s
        | _, _, _ => True)
  | .ifThen c body, s =>
      c.SafeAtR R bounds s ∧
      (match evalOpR R c s with
        | some vc => if vc.data PUnit.unit then Stmt.TraceSafeListR R bounds body s
                     else True
        | none => True)
  | .ifThenElse c tb eb, s =>
      c.SafeAtR R bounds s ∧
      (match evalOpR R c s with
        | some vc => if vc.data PUnit.unit then Stmt.TraceSafeListR R bounds tb s
                     else Stmt.TraceSafeListR R bounds eb s
        | none => True)
  termination_by st _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- Per-execution safety for a statement list. -/
def Stmt.TraceSafeListR (R : RoundingModel) (bounds : RegionBounds) : List Stmt → BlockState → Prop
  | [], _ => True
  | st :: rest, s =>
      st.TraceSafeR R bounds s ∧
      (match stepStmtR R st s with
        | some s' => Stmt.TraceSafeListR R bounds rest s'
        | none => True)
  termination_by l _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- Per-execution safety through a `forLoop` unrolling. -/
def Stmt.forLoopTraceSafeR (R : RoundingModel) (bounds : RegionBounds) (idx : RegName)
    (start n : Nat) (body : List Stmt) (s : BlockState) : Prop :=
  if start < n then
    Stmt.TraceSafeListR R bounds body (s.setReg idx .nat [] (Tile.scalar start)) ∧
    (match stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar start)) with
      | some s' => Stmt.forLoopTraceSafeR R bounds idx (start + 1) n body s'
      | none => True)
  else True
  termination_by (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- Per-execution safety through a `forRange` unrolling. -/
def Stmt.forRangeTraceSafeR (R : RoundingModel) (bounds : RegionBounds) (idx : RegName)
    (cur stop step : Nat) (body : List Stmt) (s : BlockState) : Prop :=
  if step = 0 then True
  else if cur < stop then
    Stmt.TraceSafeListR R bounds body (s.setReg idx .nat [] (Tile.scalar cur)) ∧
    (match stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur)) with
      | some s' =>
          Stmt.forRangeTraceSafeR R bounds idx (cur + step) stop step body s'
      | none => True)
  else True
  termination_by (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end


/-- Introduction form for `TraceSafeList` at a cons: prove the head safe and
the tail safe for whatever successor the step actually produces. -/
theorem Stmt.TraceSafeListR.cons_intro {R : RoundingModel}
    {bounds : RegionBounds} {st : Stmt}
    {rest : List Stmt} {s : BlockState}
    (h1 : st.TraceSafeR R bounds s)
    (h2 : ∀ s', stepStmtR R st s = some s' →
      Stmt.TraceSafeListR R bounds rest s') :
    Stmt.TraceSafeListR R bounds (st :: rest) s := by
  rw [Stmt.TraceSafeListR]
  refine ⟨h1, ?_⟩
  cases h : stepStmtR R st s with
  | none => trivial
  | some s' => exact h2 s' h

/-- `TraceSafeList` of `[]` is trivial (registered for terminal steps). -/
theorem Stmt.TraceSafeListR.nil_intro {R : RoundingModel}
    {bounds : RegionBounds}
    {s : BlockState} : Stmt.TraceSafeListR R bounds [] s := by
  rw [Stmt.TraceSafeListR]
  trivial

/-- `TraceSafeListR` append principle: a concatenation is trace-safe when the
first part is and every successor it actually reaches makes the second part
trace-safe. Pairs with `stepStmtsR_cons_some` to walk a decomposed kernel body
(prefix ++ loop ++ tail) one segment at a time. -/
theorem Stmt.TraceSafeListR.append_intro {R : RoundingModel} {bounds : RegionBounds} :
    ∀ (l1 : List Stmt) {l2 : List Stmt} (s : BlockState),
      Stmt.TraceSafeListR R bounds l1 s →
      (∀ s', stepStmtsR R l1 s = some s' → Stmt.TraceSafeListR R bounds l2 s') →
      Stmt.TraceSafeListR R bounds (l1 ++ l2) s
  | [], _, s, _, h2 => h2 s (by simp only [stepStmtsR])
  | st :: rest, l2, s, h1, h2 => by
      rw [Stmt.TraceSafeListR] at h1
      refine Stmt.TraceSafeListR.cons_intro h1.1 (fun s' hs' => ?_)
      have htl := h1.2
      rw [hs'] at htl
      exact Stmt.TraceSafeListR.append_intro rest s' htl
        (fun s'' hs'' => h2 s'' ((stepStmtsR_cons_some hs').trans hs''))

/-- Statements safe at *every* state are trace-safe as a list from any state
(covers register-only assign runs, where no successor computation is
needed). -/
theorem Stmt.TraceSafeListR.of_forall {R : RoundingModel} {bounds : RegionBounds} :
    ∀ (l : List Stmt) (s : BlockState),
      (∀ st ∈ l, ∀ s', Stmt.TraceSafeR R bounds st s') →
      Stmt.TraceSafeListR R bounds l s
  | [], _, _ => Stmt.TraceSafeListR.nil_intro
  | st :: rest, s, h => by
      refine Stmt.TraceSafeListR.cons_intro (h st List.mem_cons_self s) (fun s' _ => ?_)
      exact Stmt.TraceSafeListR.of_forall rest s'
        (fun st' hst' => h st' (List.mem_cons_of_mem st hst'))

/-- **Invariant principle for `forRangeTraceSafeR`.** To discharge trace
safety of a whole strided loop, supply an invariant `P` (indexed by the loop
counter) such that every in-range iteration started from a `P`-state (with
the counter register set) has a trace-safe body, steps successfully, and
re-establishes `P` at the advanced counter. Consumers (e.g. the streaming
matmul family) pin the loop-carried pointer registers' in-bounds addresses
with `P` and prove the per-iteration obligation once. The `step = 0` and
`stop ≤ cur` cases are vacuously safe, so no positivity hypothesis on `step`
is needed. -/
theorem Stmt.forRangeTraceSafeR_inv (R : RoundingModel) (bounds : RegionBounds)
    (idx : RegName) (stop step : Nat) (body : List Stmt)
    (P : Nat → BlockState → Prop)
    (hstep : ∀ c s, c < stop → P c s →
      Stmt.TraceSafeListR R bounds body (s.setReg idx .nat [] (Tile.scalar c)) ∧
      ∃ s', stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar c)) = some s' ∧
        P (c + step) s') :
    ∀ cur s, P cur s → Stmt.forRangeTraceSafeR R bounds idx cur stop step body s
  | cur, s, hP => by
      rw [Stmt.forRangeTraceSafeR]
      split
      · trivial
      · split
        · obtain ⟨hsafe, s', hrun, hP'⟩ := hstep cur s ‹cur < stop› hP
          refine ⟨hsafe, ?_⟩
          rw [hrun]
          exact Stmt.forRangeTraceSafeR_inv R bounds idx stop step body P hstep
            (cur + step) s' hP'
        · trivial
  termination_by cur _ _ => stop - cur
  decreasing_by omega

/-- Kernel-level per-execution safety. -/
def Kernel.TraceSafeR (R : RoundingModel) (bounds : RegionBounds) (k : Kernel)
    (s : BlockState) : Prop :=
  Stmt.TraceSafeListR R bounds k.body s

/-! ## Write-side commutation under `R` -/

namespace FlatAlloc

variable (A : FlatAlloc)

/-- Flattening commutes with a single in-bounds `R`-rounded float write: the
stored cell is `MemCell.of _ (fd.ofReal (R.storeValue fd v))` on both sides
(float carriers translate identically), so this is `flattenState_writeMemTyped`
with the rounded cell. -/
theorem flattenState_writeMemAsR (hd : A.Disjoint) (R : RoundingModel)
    (s : BlockState) {r : RegionName} (hr : r ∈ A.regions) {o : Nat}
    (ho : o < A.extent r) (fd : FloatDType) (v : TileCarrier fd.toTileDType) :
    A.flattenState (s.writeMemAsR R fd r o v)
      = (A.flattenState s).writeMemAsR R fd A.flat (A.addr r o) v := by
  cases fd <;>
  · refine BlockState.ext (fun r' o' => ?_) (fun _ _ _ => rfl) (fun _ => rfl)
      (fun _ _ => rfl) (fun _ => rfl)
    show (if r' = A.flat then A.readFlat _ o' else MemCell.real 0) = _
    simp only [BlockState.writeMemAsR]
    by_cases hrf : r' = A.flat
    · subst hrf
      rw [if_pos rfl, A.readFlat_update hd s hr ho]
      by_cases heq : o' = A.addr r o
      · subst heq
        rw [if_pos rfl, if_pos ⟨rfl, rfl⟩]
        rfl
      · rw [if_neg heq, if_neg (fun h => heq h.2)]
        show A.readFlat s o'
            = if A.flat = A.flat then A.readFlat s o' else MemCell.real 0
        rw [if_pos rfl]
    · rw [if_neg hrf, if_neg (fun h => hrf h.1)]
      show MemCell.real 0
          = if r' = A.flat then A.readFlat s o' else MemCell.real 0
      rw [if_neg hrf]

/-- Flattening commutes with a single in-bounds `writeMemTypedR`: narrowing
float channels route through `flattenState_writeMemAsR`, every other channel
is the exact `flattenState_writeMemTyped`. -/
theorem flattenState_writeMemTypedR (hd : A.Disjoint) (R : RoundingModel)
    (s : BlockState) {r : RegionName} (hr : r ∈ A.regions) {o : Nat}
    (ho : o < A.extent r) (d : TileDType) (v : TileCarrier d) :
    A.flattenState (s.writeMemTypedR R d r o v)
      = (A.flattenState s).writeMemTypedR R d A.flat (A.addr r o)
          (A.trCarrier d v) := by
  cases d
  case fp32 => exact A.flattenState_writeMemAsR hd R s hr ho .fp32 v
  case fp16 => exact A.flattenState_writeMemAsR hd R s hr ho .fp16 v
  case bf16 => exact A.flattenState_writeMemAsR hd R s hr ho .bf16 v
  case f8e4 => exact A.flattenState_writeMemAsR hd R s hr ho .f8e4 v
  case f8e5 => exact A.flattenState_writeMemAsR hd R s hr ho .f8e5 v
  all_goals exact A.flattenState_writeMemTyped hd s hr ho _ v

end FlatAlloc

/-! ## Fold-level write commutation under `R` -/

namespace FlatAlloc

variable (A : FlatAlloc)

/-- A guarded `writeMemTyped` fold (the shape of every store) commutes with
flattening, provided every active write is in bounds. The per-lane
`reg`/`off`/`v`/`act` functions cover all three access forms at once. -/
theorem flattenState_foldl_storeR (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel)
    {ι : Type _} {d : TileDType} (l : List ι) (s : BlockState)
    (act : ι → Bool) (reg : ι → RegionName) (off : ι → Nat)
    (v : ι → TileCarrier d)
    (hbound : ∀ i ∈ l, act i = true → off i < A.extent (reg i)) :
    A.flattenState (l.foldl
      (fun acc i => if act i
        then acc.writeMemTypedR R d (reg i) (off i) (v i) else acc) s)
      = l.foldl
        (fun acc i => if act i
          then acc.writeMemTypedR R d A.flat (A.addr (reg i) (off i))
            (A.trCarrier d (v i)) else acc) (A.flattenState s) := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · have hb := hbound j (List.mem_cons_self ..) hj
        have hrj : reg j ∈ A.regions := by
          by_contra hnr
          rw [hcov _ hnr] at hb
          exact absurd hb (Nat.not_lt_zero _)
        rw [if_pos hj, if_pos hj,
          ← A.flattenState_writeMemTypedR hd R s hrj hb d (v j)]
        exact ih _ fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact
      · rw [if_neg hj, if_neg hj]
        exact ih _ fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact

/-- Read-modify-write variant of `flattenState_foldl_store`: the written
value may depend on the running accumulator (the `atomicAdd` shape). The
`hv` hypothesis relates the two per-lane value functions across the
flattening at every intermediate state. -/
theorem flattenState_foldl_rmwR (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel)
    {ι : Type _} {d : TileDType} (l : List ι) (s : BlockState)
    (act : ι → Bool) (reg : ι → RegionName) (off : ι → Nat)
    (v vf : BlockState → ι → TileCarrier d)
    (hbound : ∀ i ∈ l, act i = true → off i < A.extent (reg i))
    (hv : ∀ acc i, i ∈ l → act i = true →
      vf (A.flattenState acc) i = A.trCarrier d (v acc i)) :
    A.flattenState (l.foldl
      (fun acc i => if act i
        then acc.writeMemTypedR R d (reg i) (off i) (v acc i) else acc) s)
      = l.foldl
        (fun acc i => if act i
          then acc.writeMemTypedR R d A.flat (A.addr (reg i) (off i))
            (vf acc i) else acc) (A.flattenState s) := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · have hb := hbound j (List.mem_cons_self ..) hj
        have hrj : reg j ∈ A.regions := by
          by_contra hnr
          rw [hcov _ hnr] at hb
          exact absurd hb (Nat.not_lt_zero _)
        rw [if_pos hj, if_pos hj, hv s j (List.mem_cons_self ..) hj,
          ← A.flattenState_writeMemTypedR hd R s hrj hb d (v s j)]
        exact ih _ (fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact)
          (fun acc i hi hact => hv acc i (List.mem_cons_of_mem _ hi) hact)
      · rw [if_neg hj, if_neg hj]
        exact ih _ (fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact)
          (fun acc i hi hact => hv acc i (List.mem_cons_of_mem _ hi) hact)

end FlatAlloc

/-! ## `stepStmtR` preserves `undef` on the covered fragment -/

@[simp] theorem BlockState.writeMemTypedR_undef (R : RoundingModel)
    (s : BlockState) (dtype : TileDType) (region : RegionName) (offset : Nat)
    (v : TileCarrier dtype) (r : RegionName) (o : Nat) :
    (s.writeMemTypedR R dtype region offset v).undef r o = s.undef r o := by
  cases dtype <;> rfl

private theorem foldl_guarded_writeR_undef (R : RoundingModel)
    {ι : Type _} {d : TileDType}
    (l : List ι) (s : BlockState) (act : ι → Bool) (reg : ι → RegionName)
    (off : ι → Nat) (v : BlockState → ι → TileCarrier d) :
    (l.foldl (fun acc i => if act i
      then acc.writeMemTypedR R d (reg i) (off i) (v acc i) else acc) s).undef
      = s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · rw [if_pos hj, ih]
        funext rr oo
        exact BlockState.writeMemTypedR_undef ..
      · rw [if_neg hj]
        exact ih _

mutual

/-- On the covered fragment, stepping a statement never changes `undef`. -/
theorem stepStmtR_undef (R : RoundingModel) : ∀ (st : Stmt) (s s' : BlockState),
    st.FlattenOkR → stepStmtR R st s = some s' → s'.undef = s.undef
  | .assign d sh n e, s, s', _, h => by
      simp only [stepStmtR] at h
      cases hv : evalOpR R e s with
      | none => rw [hv] at h; exact absurd h (by simp)
      | some v =>
          rw [hv] at h
          replace h : some (s.setReg n d sh v) = some s' := h
          obtain rfl := Option.some_inj.mp h
          rfl
  | .store d sh mem val mask, s, s', _, h => by
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOpR R off s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOpR R p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOpR R p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hv : evalOpR R val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
  | .atomicAdd nd sh mem val mask, s, s', _, h => by
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOpR R off s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOpR R p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOpR R p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
      · simp only [stepStmtR] at h
        cases hvv : evalOpR R val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOpR R m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOpR R p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_writeR_undef ..
  | .atomicRMW op d sh mem input extra mask dest, s, s', hok, h => by
      simp only [stepStmtR] at h
      -- RMW delegates to the exact semantics; the `FlattenOkR` clause for it
      -- is the exact-side one verbatim, so just respell it.
      exact stepStmt_undef _ s s'
        (by simp only [Stmt.FlattenOkR] at hok
            simp only [Stmt.FlattenOk]; exact hok) h
  | .forLoop idx n body, s, s', hok, h => by
      simp only [Stmt.FlattenOkR] at hok
      simp only [stepStmtR] at h
      exact stepForLoopAuxR_undef R idx 0 n body s s' hok h
  | .forRange idx start stop step body, s, s', hok, h => by
      simp only [Stmt.FlattenOkR] at hok
      simp only [stepStmtR] at h
      exact stepForRangeAuxR_undef R idx start stop step body s s' hok h
  | .forRangeDyn idx start stop step body, s, s', hok, h => by
      simp only [Stmt.FlattenOkR] at hok
      simp only [stepStmtR] at h
      cases h1 : evalOpR R start s with
      | none => rw [h1] at h; exact absurd h (by simp)
      | some a =>
          rw [h1] at h
          cases h2 : evalOpR R stop s with
          | none => rw [h2] at h; exact absurd h (by simp)
          | some b =>
              rw [h2] at h
              cases h3 : evalOpR R step s with
              | none => rw [h3] at h; exact absurd h (by simp)
              | some c =>
                  rw [h3] at h
                  replace h : stepForRangeAuxR R idx (a.data PUnit.unit)
                      (b.data PUnit.unit) (c.data PUnit.unit) body s
                      = some s' := h
                  exact stepForRangeAuxR_undef R idx _ _ _ body s s' hok.2.2.2 h
  | .ifThen c body, s, s', hok, h => by
      simp only [Stmt.FlattenOkR] at hok
      simp only [stepStmtR] at h
      cases hc : evalOpR R c s with
      | none => rw [hc] at h; exact absurd h (by simp)
      | some vc =>
          rw [hc] at h
          replace h : (if vc.data PUnit.unit = true then stepStmtsR R body s
              else some s) = some s' := h
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb] at h
            exact stepStmtsR_undef R body s s' hok.2 h
          · rw [if_neg hb] at h
            obtain rfl := Option.some_inj.mp h
            rfl
  | .ifThenElse c tb eb, s, s', hok, h => by
      simp only [Stmt.FlattenOkR] at hok
      simp only [stepStmtR] at h
      cases hc : evalOpR R c s with
      | none => rw [hc] at h; exact absurd h (by simp)
      | some vc =>
          rw [hc] at h
          replace h : (if vc.data PUnit.unit = true then stepStmtsR R tb s
              else stepStmtsR R eb s) = some s' := h
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb] at h
            exact stepStmtsR_undef R tb s s' hok.2.1 h
          · rw [if_neg hb] at h
            exact stepStmtsR_undef R eb s s' hok.2.2 h
  termination_by st _ _ _ _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- List version of `stepStmtR_undef R`. -/
theorem stepStmtsR_undef (R : RoundingModel) : ∀ (l : List Stmt) (s s' : BlockState),
    StmtList.FlattenOkR l → stepStmtsR R l s = some s' → s'.undef = s.undef
  | [], s, s', _, h => by
      simp only [stepStmtsR] at h
      obtain rfl := Option.some_inj.mp h
      rfl
  | st :: rest, s, s', hok, h => by
      simp only [StmtList.FlattenOkR] at hok
      simp only [stepStmtsR] at h
      cases h1 : stepStmtR R st s with
      | none => rw [h1] at h; exact absurd h (by simp)
      | some s1 =>
          rw [h1] at h
          replace h : stepStmtsR R rest s1 = some s' := h
          rw [stepStmtsR_undef R rest s1 s' hok.2 h,
            stepStmtR_undef R st s s1 hok.1 h1]
  termination_by l _ _ _ _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary version of `stepStmtR_undef R`. -/
theorem stepForLoopAuxR_undef (R : RoundingModel) : ∀ (idx : RegName) (start n : Nat)
    (body : List Stmt) (s s' : BlockState),
    StmtList.FlattenOkR body →
    stepForLoopAuxR R idx start n body s = some s' → s'.undef = s.undef
  | idx, start, n, body, s, s', hok, h => by
      rw [stepForLoopAuxR] at h
      split at h
      · cases h1 : stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rw [h1] at h; exact absurd h (by simp)
        | some s1 =>
            rw [h1] at h
            replace h : stepForLoopAuxR R idx (start + 1) n body s1 = some s' := h
            rw [stepForLoopAuxR_undef R idx (start + 1) n body s1 s' hok h,
              stepStmtsR_undef R body _ s1 hok h1]
            rfl
      · obtain rfl := Option.some_inj.mp h
        rfl
  termination_by _ start n body _ _ _ _ => (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- `forRange` auxiliary version of `stepStmtR_undef R`. -/
theorem stepForRangeAuxR_undef (R : RoundingModel) : ∀ (idx : RegName) (cur stop step : Nat)
    (body : List Stmt) (s s' : BlockState),
    StmtList.FlattenOkR body →
    stepForRangeAuxR R idx cur stop step body s = some s' → s'.undef = s.undef
  | idx, cur, stop, step, body, s, s', hok, h => by
      rw [stepForRangeAuxR] at h
      split at h
      · obtain rfl := Option.some_inj.mp h
        rfl
      · split at h
        · cases h1 : stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rw [h1] at h; exact absurd h (by simp)
          | some s1 =>
              rw [h1] at h
              replace h : stepForRangeAuxR R idx (cur + step) stop step body s1
                  = some s' := h
              rw [stepForRangeAuxR_undef R idx (cur + step) stop step body s1 s'
                  hok h,
                stepStmtsR_undef R body _ s1 hok h1]
              rfl
        · obtain rfl := Option.some_inj.mp h
          rfl
  termination_by _ cur stop step body _ _ _ _ => (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end

/-! ## Statement-level bridge under `R` -/

mutual

/-- **Statement-level bridge under `R`**: on the covered fragment, stepping
the translated statement with `stepStmtR R` in the flattened state is the
flattening of the source `R`-step. -/
theorem FlatAlloc.stepStmtR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) :
    ∀ (st : Stmt) (s : BlockState),
      st.TraceSafeR R A.extent s → st.FlattenOkR → s.undef = (fun _ _ => 0) →
      stepStmtR R (A.flattenStmt st) (A.flattenState s)
        = (stepStmtR R st s).map A.flattenState
  | .assign d sh n e, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      rw [A.evalOpR_flatten hd hcov R e s hms hok hu]
      cases hv : evalOpR R e s with
      | none => rfl
      | some v =>
          refine congrArg some ?_
          exact (A.flattenState_setReg s n d sh v).symm
  | .store d sh mem val mask, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt]
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩ <;>
        simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask]
      -- none.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [evalOpR_shiftAdd,
              show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hoffs : evalOpR R off s with
            | none => rfl
            | some offs =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                  (fun i _ _ => haddr offs hoffs i trivial)]
                rfl
      -- none.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
            cases hps : evalOpR R p s with
            | none => rfl
            | some ps =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                  (fun i _ _ => haddr ps hps i trivial)]
                rfl
      -- none.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
            cases hps : evalOpR R p s with
            | none => rfl
            | some ps =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                  (fun i _ hact => haddr ps hps i trivial (by simpa using hact))]
                congr 1
                funext acc i
                simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                  FlatAlloc.trCarrier_blockPtr_address,
                  FlatAlloc.trCarrier_blockPtr_inBounds]
                rfl
      -- mask.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [evalOpR_shiftAdd,
                  show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                    rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOpR R off s with
                | none => rfl
                | some offs =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => haddr ps hps i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2)]
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile,
                      FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
      -- mask.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [evalOpR_shiftAdd,
                  show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                    rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOpR R off s with
                | none => rfl
                | some offs =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => haddr ps hps i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu]
        cases hv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_storeR hd hcov R _ s _ _ _ _
                      (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2)]
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile,
                      FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
  | .atomicAdd nd sh mem val mask, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt]
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩ <;>
        simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask]
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [evalOpR_shiftAdd,
              show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hoffs : evalOpR R off s with
            | none => rfl
            | some offs =>
                have hfold := A.flattenState_foldl_rmwR hd hcov R
                  (TileShape.allIndices sh) s
                  (act := fun _ => true)
                  (reg := fun _ => (Region.cast r : RegionName))
                  (off := fun i => offs.data i)
                  (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                  (fun i _ _ => haddr offs hoffs i trivial)
                  (fun acc i _ _ => by
                    simp only []
                    have hb2 := haddr offs hoffs i trivial
                    have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
            cases hps : evalOpR R p s with
            | none => rfl
            | some ps =>
                have hfold := A.flattenState_foldl_rmwR hd hcov R
                  (TileShape.allIndices sh) s
                  (act := fun _ => true)
                  (reg := fun i => (ps.data i).1)
                  (off := fun i => (ps.data i).2)
                  (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                  (fun i _ _ => haddr ps hps i trivial)
                  (fun acc i _ _ => by
                    simp only []
                    have hb2 := haddr ps hps i trivial
                    have hr2 : (ps.data i).1 ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
            cases hps : evalOpR R p s with
            | none => rfl
            | some ps =>
                have hfold := A.flattenState_foldl_rmwR hd hcov R
                  (TileShape.allIndices sh) s
                  (act := fun i => true && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                  (reg := fun i => (ps.data i).region)
                  (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                  (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                  (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr ps hps i trivial hact.2)
                  (fun acc i _ hact => by
                    simp only []
                    simp only [Bool.and_eq_true] at hact
                    have hb2 := haddr ps hps i trivial hact.2
                    have hr2 : (ps.data i).region ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                refine Eq.trans ?_ hfold.symm
                congr 1
                funext acc i
                simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                  FlatAlloc.trCarrier_blockPtr_address,
                  FlatAlloc.trCarrier_blockPtr_inBounds]
                rfl
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [evalOpR_shiftAdd,
                  show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                    rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOpR R off s with
                | none => rfl
                | some offs =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun _ => (Region.cast r : RegionName))
                      (off := fun i => offs.data i)
                      (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun i => (ps.data i).1)
                      (off := fun i => (ps.data i).2)
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                      (fun i _ hact => haddr ps hps i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr ps hps i ⟨ms, hm, hact⟩
                        have hr2 : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      (reg := fun i => (ps.data i).region)
                      (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2)
                      (fun acc i _ hact => by
                        simp only []
                        simp only [Bool.and_eq_true] at hact
                        have hb2 := haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2
                        have hr2 : (ps.data i).region ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    refine Eq.trans ?_ hfold.symm
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [evalOpR_shiftAdd,
                  show evalOpR R (A.flattenOp off) (A.flattenState s) = evalOpR R off s from by
                    rw [A.evalOpR_flatten hd hcov R off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOpR R off s with
                | none => rfl
                | some offs =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun _ => (Region.cast r : RegionName))
                      (off := fun i => offs.data i)
                      (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun i => (ps.data i).1)
                      (off := fun i => (ps.data i).2)
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                      (fun i _ hact => haddr ps hps i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr ps hps i ⟨ms, hm, hact⟩
                        have hr2 : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR] at haddr
        simp only [stepStmtR]
        rw [A.evalOpR_flatten hd hcov R val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOpR R val s with
        | none => rfl
        | some values =>
            rw [show evalOpR R (A.flattenOp m) (A.flattenState s) = evalOpR R m s from by
                rw [A.evalOpR_flatten hd hcov R m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOpR R m s with
            | none => rfl
            | some ms =>
                rw [A.evalOpR_flatten hd hcov R p s hmsmem hokmem hu]
                cases hps : evalOpR R p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmwR hd hcov R
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      (reg := fun i => (ps.data i).region)
                      (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2)
                      (fun acc i _ hact => by
                        simp only []
                        simp only [Bool.and_eq_true] at hact
                        have hb2 := haddr ps hps i ⟨ms, hm, hact.1⟩ hact.2
                        have hr2 : (ps.data i).region ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    refine Eq.trans ?_ hfold.symm
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
  | .atomicRMW op d sh mem input extra mask dest, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      have h := A.stepStmt_flatten hd hcov
        (.atomicRMW op d sh mem input extra mask dest) s
        (by simp only [Stmt.TraceSafe]; exact hms)
        (by simp only [Stmt.FlattenOkR] at hok
            simp only [Stmt.FlattenOk]; exact hok) hu
      simp only [FlatAlloc.flattenStmt] at h ⊢
      simp only [stepStmtR]
      exact h
  | .forLoop idx n body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      exact A.stepForLoopAuxR_flatten hd hcov R idx 0 n body s hms hok hu
  | .forRange idx start stop step body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      exact A.stepForRangeAuxR_flatten hd hcov R idx start stop step body s hms hok hu
  | .forRangeDyn idx start stop step body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      obtain ⟨hms1, hms2, hms3, hmsb⟩ := hms
      obtain ⟨hok1, hok2, hok3, hokb⟩ := hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      rw [show evalOpR R (A.flattenOp start) (A.flattenState s) = evalOpR R start s from by
          rw [A.evalOpR_flatten hd hcov R start s hms1 hok1 hu,
            A.trTileFun_data (d := .nat) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases h1 : evalOpR R start s with
      | none => rfl
      | some a =>
          rw [show evalOpR R (A.flattenOp stop) (A.flattenState s) = evalOpR R stop s from by
              rw [A.evalOpR_flatten hd hcov R stop s hms2 hok2 hu,
                A.trTileFun_data (d := .nat) (by decide) (by decide),
                Option.map_id, id_eq]]
          cases h2 : evalOpR R stop s with
          | none => rfl
          | some b =>
              rw [show evalOpR R (A.flattenOp step) (A.flattenState s)
                  = evalOpR R step s from by
                  rw [A.evalOpR_flatten hd hcov R step s hms3 hok3 hu,
                    A.trTileFun_data (d := .nat) (by decide) (by decide),
                    Option.map_id, id_eq]]
              cases h3 : evalOpR R step s with
              | none => rfl
              | some c =>
                  rw [h1, h2, h3] at hmsb
                  replace hmsb : Stmt.forRangeTraceSafeR R A.extent idx
                      (a.data PUnit.unit) (b.data PUnit.unit)
                      (c.data PUnit.unit) body s := hmsb
                  exact A.stepForRangeAuxR_flatten hd hcov R idx _ _ _ body s
                    hmsb hokb hu
  | .ifThen c body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      rw [show evalOpR R (A.flattenOp c) (A.flattenState s) = evalOpR R c s from by
          rw [A.evalOpR_flatten hd hcov R c s hms.1 hok.1 hu,
            A.trTileFun_data (d := .bool) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases hc : evalOpR R c s with
      | none => rfl
      | some vc =>
          show (if vc.data PUnit.unit = true
              then stepStmtsR R (A.flattenStmts body) (A.flattenState s)
              else some (A.flattenState s))
            = (if vc.data PUnit.unit = true then stepStmtsR R body s
              else some s).map A.flattenState
          have hbody := hms.2
          rw [hc] at hbody
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            replace hbody : (if vc.data PUnit.unit = true
                then Stmt.TraceSafeListR R A.extent body s else True) := hbody
            rw [if_pos hb] at hbody
            exact A.stepStmtsR_flatten hd hcov R body s hbody hok.2 hu
          · rw [if_neg hb, if_neg hb]
            rfl
  | .ifThenElse c tb eb, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeR] at hms
      simp only [Stmt.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmt, stepStmtR]
      rw [show evalOpR R (A.flattenOp c) (A.flattenState s) = evalOpR R c s from by
          rw [A.evalOpR_flatten hd hcov R c s hms.1 hok.1 hu,
            A.trTileFun_data (d := .bool) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases hc : evalOpR R c s with
      | none => rfl
      | some vc =>
          show (if vc.data PUnit.unit = true
              then stepStmtsR R (A.flattenStmts tb) (A.flattenState s)
              else stepStmtsR R (A.flattenStmts eb) (A.flattenState s))
            = (if vc.data PUnit.unit = true then stepStmtsR R tb s
              else stepStmtsR R eb s).map A.flattenState
          have hbody := hms.2
          rw [hc] at hbody
          replace hbody : (if vc.data PUnit.unit = true
              then Stmt.TraceSafeListR R A.extent tb s
              else Stmt.TraceSafeListR R A.extent eb s) := hbody
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            rw [if_pos hb] at hbody
            exact A.stepStmtsR_flatten hd hcov R tb s hbody hok.2.1 hu
          · rw [if_neg hb, if_neg hb]
            rw [if_neg hb] at hbody
            exact A.stepStmtsR_flatten hd hcov R eb s hbody hok.2.2 hu
  termination_by st _ _ _ _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- List version of the statement bridge. -/
theorem FlatAlloc.stepStmtsR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) :
    ∀ (l : List Stmt) (s : BlockState),
      Stmt.TraceSafeListR R A.extent l s → StmtList.FlattenOkR l →
      s.undef = (fun _ _ => 0) →
      stepStmtsR R (A.flattenStmts l) (A.flattenState s)
        = (stepStmtsR R l s).map A.flattenState
  | [], s, _, _, _ => by
      simp only [FlatAlloc.flattenStmts, stepStmtsR, Option.map_some]
  | st :: rest, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeListR] at hms
      simp only [StmtList.FlattenOkR] at hok
      simp only [FlatAlloc.flattenStmts, stepStmtsR]
      rw [A.stepStmtR_flatten hd hcov R st s hms.1 hok.1 hu]
      cases h1 : stepStmtR R st s with
      | none => rfl
      | some s1 =>
          have hu1 : s1.undef = (fun _ _ => 0) := by
            rw [stepStmtR_undef R st s s1 hok.1 h1, hu]
          have hrest := hms.2
          rw [h1] at hrest
          replace hrest : Stmt.TraceSafeListR R A.extent rest s1 := hrest
          exact A.stepStmtsR_flatten hd hcov R rest s1 hrest hok.2 hu1
  termination_by l _ _ _ _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary bridge. -/
theorem FlatAlloc.stepForLoopAuxR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) :
    ∀ (idx : RegName) (start n : Nat) (body : List Stmt) (s : BlockState),
      Stmt.forLoopTraceSafeR R A.extent idx start n body s →
      StmtList.FlattenOkR body → s.undef = (fun _ _ => 0) →
      stepForLoopAuxR R idx start n (A.flattenStmts body) (A.flattenState s)
        = (stepForLoopAuxR R idx start n body s).map A.flattenState
  | idx, start, n, body, s, hms, hok, hu => by
      rw [stepForLoopAuxR, stepForLoopAuxR]
      rw [Stmt.forLoopTraceSafeR] at hms
      split
      next hlt =>
        rw [if_pos hlt] at hms
        obtain ⟨hbody, hnext⟩ := hms
        have hset : (A.flattenState s).setReg idx .nat [] (Tile.scalar start)
            = A.flattenState (s.setReg idx .nat [] (Tile.scalar start)) :=
          (A.flattenState_setReg s idx .nat [] (Tile.scalar start)).symm
        rw [hset, A.stepStmtsR_flatten hd hcov R body
          (s.setReg idx .nat [] (Tile.scalar start)) hbody hok hu]
        cases h1 : stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rfl
        | some s1 =>
            have hu1 : s1.undef = (fun _ _ => 0) := by
              rw [stepStmtsR_undef R body _ s1 hok h1]
              exact hu
            rw [h1] at hnext
            replace hnext : Stmt.forLoopTraceSafeR R A.extent idx (start + 1) n
                body s1 := hnext
            exact A.stepForLoopAuxR_flatten hd hcov R idx (start + 1) n body s1
              hnext hok hu1
      next => rfl
  termination_by _ start n body _ _ _ _ => (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- `forRange` auxiliary bridge. -/
theorem FlatAlloc.stepForRangeAuxR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) :
    ∀ (idx : RegName) (cur stop step : Nat) (body : List Stmt)
      (s : BlockState),
      Stmt.forRangeTraceSafeR R A.extent idx cur stop step body s →
      StmtList.FlattenOkR body → s.undef = (fun _ _ => 0) →
      stepForRangeAuxR R idx cur stop step (A.flattenStmts body)
          (A.flattenState s)
        = (stepForRangeAuxR R idx cur stop step body s).map A.flattenState
  | idx, cur, stop, step, body, s, hms, hok, hu => by
      rw [stepForRangeAuxR, stepForRangeAuxR]
      rw [Stmt.forRangeTraceSafeR] at hms
      split
      next => rfl
      next hz =>
        rw [if_neg hz] at hms
        split
        next hlt =>
          rw [if_pos hlt] at hms
          obtain ⟨hbody, hnext⟩ := hms
          have hset : (A.flattenState s).setReg idx .nat [] (Tile.scalar cur)
              = A.flattenState (s.setReg idx .nat [] (Tile.scalar cur)) :=
            (A.flattenState_setReg s idx .nat [] (Tile.scalar cur)).symm
          rw [hset, A.stepStmtsR_flatten hd hcov R body
            (s.setReg idx .nat [] (Tile.scalar cur)) hbody hok hu]
          cases h1 : stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rfl
          | some s1 =>
              have hu1 : s1.undef = (fun _ _ => 0) := by
                rw [stepStmtsR_undef R body _ s1 hok h1]
                exact hu
              rw [h1] at hnext
              replace hnext : Stmt.forRangeTraceSafeR R A.extent idx (cur + step)
                  stop step body s1 := hnext
              exact A.stepForRangeAuxR_flatten hd hcov R idx (cur + step) stop step
                body s1 hnext hok hu1
        next => rfl
  termination_by _ cur stop step body _ _ _ _ => (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end

/-- **The flat-memory bridge under `R`**, on the `R` fragment `FlattenOkR`
(`FlattenOk` minus the `∀`-state no-underflow `ptrSub` conjunct — underflow
is discharged per trace by `Op.SafeAtR`): running the translated kernel under
`execR R` on the flattened state is the flattening of the source `execR R`
run. Every region-model rounding-surface theorem transports along this
equation to a single flat address space. -/
theorem FlatAlloc.execR_flattenR (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) (k : Kernel)
    (s : BlockState) (hms : k.TraceSafeR R A.extent s) (hok : k.FlattenOkR)
    (hu : s.undef = (fun _ _ => 0)) :
    execR R (A.flattenKernel k) (A.flattenState s)
      = (execR R k s).map A.flattenState := by
  simp only [execR, FlatAlloc.flattenKernel]
  exact A.stepStmtsR_flatten hd hcov R k.body s hms hok hu

/-- **The flat-memory bridge under `R`** on the exact fragment `FlattenOk`
(a corollary of `execR_flattenR` via `Kernel.FlattenOk.toR`): running the
translated kernel under `execR R` on the flattened state is the flattening
of the source `execR R` run. Every region-model rounding-surface theorem
transports along this equation to a single flat address space. -/
theorem FlatAlloc.execR_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel) (k : Kernel)
    (s : BlockState) (hms : k.TraceSafeR R A.extent s) (hok : k.FlattenOk)
    (hu : s.undef = (fun _ _ => 0)) :
    execR R (A.flattenKernel k) (A.flattenState s)
      = (execR R k s).map A.flattenState :=
  A.execR_flattenR hd hcov R k s hms hok.toR hu

/-- The flat image's memory is determined by the source memory: states that
agree cell-for-cell flatten to states that agree cell-for-cell. -/
theorem FlatAlloc.flattenState_mem_congr (A : FlatAlloc) {s t : BlockState}
    (h : ∀ r o, s.mem r o = t.mem r o) (r' : RegionName) (o' : Nat) :
    (A.flattenState s).mem r' o' = (A.flattenState t).mem r' o' := by
  show (if r' = A.flat then A.readFlat s o' else MemCell.real 0)
    = (if r' = A.flat then A.readFlat t o' else MemCell.real 0)
  by_cases hr : r' = A.flat
  · rw [if_pos hr, if_pos hr]
    unfold FlatAlloc.readFlat
    cases A.decode o' with
    | none => rfl
    | some p => exact congrArg A.trCell (h p.1 p.2)
  · rw [if_neg hr, if_neg hr]

/-- **Refinement transports across the bridge**: if the region-model runs of
`lhs` and `rhs` from `s` perform the same writes (the `Refines R … []`
conclusion shape — final memories agree at every cell), then the flat-model
runs of their translations from the flattened state do too. The safety and
fragment side conditions are per kernel, exactly as in `execR_flatten`. -/
theorem FlatAlloc.refineR_mem_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (R : RoundingModel)
    (lhs rhs : Kernel) (s : BlockState)
    (hmsL : lhs.TraceSafeR R A.extent s) (hokL : lhs.FlattenOk)
    (hmsR : rhs.TraceSafeR R A.extent s) (hokR : rhs.FlattenOk)
    (hu : s.undef = (fun _ _ => 0))
    (h : ∀ lhs' rhs', execR R lhs s = some lhs' →
      execR R rhs s = some rhs' → ∀ r o, lhs'.mem r o = rhs'.mem r o) :
    ∀ lhs' rhs',
      execR R (A.flattenKernel lhs) (A.flattenState s) = some lhs' →
      execR R (A.flattenKernel rhs) (A.flattenState s) = some rhs' →
      ∀ r o, lhs'.mem r o = rhs'.mem r o := by
  intro lhs' rhs' hL hR
  rw [A.execR_flatten hd hcov R lhs s hmsL hokL hu] at hL
  rw [A.execR_flatten hd hcov R rhs s hmsR hokR hu] at hR
  cases hLs : execR R lhs s with
  | none => rw [hLs] at hL; exact absurd hL (by simp)
  | some sL =>
      cases hRs : execR R rhs s with
      | none => rw [hRs] at hR; exact absurd hR (by simp)
      | some sR =>
          rw [hLs, Option.map_some] at hL
          rw [hRs, Option.map_some] at hR
          obtain rfl := Option.some_inj.mp hL
          obtain rfl := Option.some_inj.mp hR
          exact fun r o => A.flattenState_mem_congr (h sL sR hLs hRs) r o

end VeriTile.Triton

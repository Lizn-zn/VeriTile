/-
VeriTile.Triton.Memory.Bounds

Layer-1 bounds safety for the VeriTile memory model.
-/

import VeriTile.Triton.Semantics.EvalOp

namespace VeriTile.Triton

/-!
This module implements #48's lightweight memory-safety layer.

The contract is intentionally state-independent at the public API level:
`Op.MemorySafe bounds e`, `Stmt.MemorySafe bounds st`, and
`Kernel.MemorySafe bounds k` are plain propositions over syntax. Memory
operations internally quantify over all `BlockState`s so masks, pointer values,
and block-pointer values are interpreted with the same evaluator as execution.

Relationship to adjacent work:

* Parent: #56 Memory modeling roadmap, Layer 1 bounds safety.
* Foundation for: #49 disjoint-frame / composition work.
* Composable with: #46 checker/provenance, which can later imply dynamic
  pointer safety predicates.
* Independent of: #59 differential testing.
* Deferred to: #12 atomics/concurrency/barriers/async.

Non-goals: race freedom, frame theorems, fractional permissions, atomics,
barriers, async copy, shared memory, and operational semantic changes.
-/

/-- Runtime region sizes used by the Layer-1 bounds-safety contract. -/
abbrev RegionBounds := RegionName → Nat

namespace MaskOpt

/-- Lanes that are active for a memory operation under a mask in state `s`.

If the mask expression does not evaluate, no lane is considered active. This
matches the operational behavior: the memory operation itself fails before any
lane-specific memory access occurs. -/
def Active (s : BlockState) : (mask : MaskOpt dtype shape) → TileIndex shape → Prop
  | .none => fun _ => True
  | .mask m => fun i =>
      ∃ masks, evalOp m s = some masks ∧ masks.data i = true
  | .maskOther m _ => fun i =>
      ∃ masks, evalOp m s = some masks ∧ masks.data i = true

end MaskOpt

/-- State-independent memory-safety contract for expressions.

For `Op.load`, the predicate recursively checks address/mask/other
subexpressions and then quantifies over all states to require every active
lane address to stay within `bounds`. -/
def Op.MemorySafe (bounds : RegionBounds) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.MemorySafe bounds
  | .full _ e => e.MemorySafe bounds
  | .castFloat _ _ e => e.MemorySafe bounds
  | .castNatToInt e => e.MemorySafe bounds
  | .castIntToNat e => e.MemorySafe bounds
  | .castRealToInt8 e => e.MemorySafe bounds
  | .add _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .sub _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .mul _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .div _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .floorDiv _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .mod _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .bitAnd _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .bitOr _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .bitXor _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .shiftLeft _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .shiftRight _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .exp a => a.MemorySafe bounds
  | .exp2 a => a.MemorySafe bounds
  | .log a => a.MemorySafe bounds
  | .log2 a => a.MemorySafe bounds
  | .sigmoid a => a.MemorySafe bounds
  | .sqrt a => a.MemorySafe bounds
  | .rsqrt a => a.MemorySafe bounds
  | .tanh a => a.MemorySafe bounds
  | .sin a => a.MemorySafe bounds
  | .cos a => a.MemorySafe bounds
  | .tan a => a.MemorySafe bounds
  | .atan a => a.MemorySafe bounds
  | .cosh a => a.MemorySafe bounds
  | .sinh a => a.MemorySafe bounds
  | .erf a => a.MemorySafe bounds
  | .lt _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .le _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .eq _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .gt _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .ge _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .ne _ _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .boolAnd _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .boolOr _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .boolNot a => a.MemorySafe bounds
  | .max2 _ a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .where c a b => c.MemorySafe bounds ∧ a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .ite c a b => c.MemorySafe bounds ∧ a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .reduceMax _ _ a => a.MemorySafe bounds
  | .reduceMaxNat _ _ a => a.MemorySafe bounds
  | .reduceSum _ _ a => a.MemorySafe bounds
  | .scan _ _ a => a.MemorySafe bounds
  | .argMax _ a => a.MemorySafe bounds
  | .argMin _ a => a.MemorySafe bounds
  | .sort _ a => a.MemorySafe bounds
  | .dot a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .transpose a => a.MemorySafe bounds
  | .reshape _ a => a.MemorySafe bounds
  | .remap _ _ a => a.MemorySafe bounds
  | .join a b => a.MemorySafe bounds ∧ b.MemorySafe bounds
  | .split _ a => a.MemorySafe bounds
  | .expandDim _ a => a.MemorySafe bounds
  | .ptrBase _ => True
  | .ptrAdd _ ptr off => ptr.MemorySafe bounds ∧ off.MemorySafe bounds
  | .ptrSub _ ptr off => ptr.MemorySafe bounds ∧ off.MemorySafe bounds
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => base.MemorySafe bounds
  | .makeBlockPtrDynOffsets _ base _ _ _ offsets =>
      base.MemorySafe bounds ∧ ∀ off ∈ offsets, off.MemorySafe bounds
  | .advanceBlockPtr ptr _ => ptr.MemorySafe bounds
  | .load _ mem mask =>
      (match mem with
        | .region _ off => off.MemorySafe bounds
        | .ptr ptr => ptr.MemorySafe bounds
        | .blockPtr ptr _ => ptr.MemorySafe bounds) ∧
      (match mask with
        | .none => True
        | .mask m => m.MemorySafe bounds
        | .maskOther m other => m.MemorySafe bounds ∧ other.MemorySafe bounds) ∧
      ∀ s,
        match mem with
        | .region region off =>
            ∀ offsets, evalOp off s = some offsets →
              ∀ i : TileIndex shape, mask.Active s i → offsets.data i < bounds (Region.cast region)
        | .ptr ptr =>
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                mask.Active s i → (ptrs.data i).2 < bounds (ptrs.data i).1
        | .blockPtr ptr boundaryCheck =>
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                mask.Active s i →
                let bp := ptrs.data i
                let idx := TileShape.indexToList shape i
                bp.inBounds idx boundaryCheck = true →
                  bp.address idx < bounds bp.region
  | .natToReal a => a.MemorySafe bounds

/-- Dynamic pointer addresses are safe for active lanes in state `s`. -/
def Op.PointerAddressesSafeOn (bounds : RegionBounds) (ptr : Op .ptr shape)
    (s : BlockState) (active : TileIndex shape → Prop) : Prop :=
  ptr.MemorySafe bounds ∧
    ∀ ptrs, evalOp ptr s = some ptrs →
      ∀ i : TileIndex shape,
        active i → (ptrs.data i).2 < bounds (ptrs.data i).1

/-- Dynamic pointer addresses are safe for every lane in every state. -/
def Op.PointerAddressesSafe (bounds : RegionBounds) (ptr : Op .ptr shape) : Prop :=
  ptr.MemorySafe bounds ∧
    ∀ s ptrs, evalOp ptr s = some ptrs →
      ∀ i : TileIndex shape, (ptrs.data i).2 < bounds (ptrs.data i).1

/-- Child expressions inside a memory address form are memory-safe. -/
def memAccessMemorySafe {dtype : TileDType} (bounds : RegionBounds) :
    MemAccess dtype shape → Prop
  | .region _ off => off.MemorySafe bounds
  | .ptr ptr => ptr.MemorySafe bounds
  | .blockPtr ptr _ => ptr.MemorySafe bounds

/-- Namespace-facing wrapper for memory-address safety. -/
abbrev MemAccess.MemorySafe {dtype : TileDType} (bounds : RegionBounds)
    (mem : MemAccess dtype shape) : Prop :=
  memAccessMemorySafe bounds mem

/-- Active memory addresses produced by a memory address form are in bounds.

For checked block pointers, only lanes whose `BlockPtr.inBounds` predicate is
true need an address bound: checked OOB lanes are inactive at the memory layer
and are handled by padding/no-op semantics. -/
def memAccessActiveAddressSafe {dtype : TileDType} (bounds : RegionBounds)
    (mem : MemAccess dtype shape) (s : BlockState)
    (active : TileIndex shape → Prop) : Prop :=
  match mem with
  | .region region off =>
      ∀ offsets, evalOp off s = some offsets →
        ∀ i : TileIndex shape, active i → offsets.data i < bounds (Region.cast region)
  | .ptr ptr =>
      Op.PointerAddressesSafeOn bounds ptr s active
  | .blockPtr ptr boundaryCheck =>
      ptr.MemorySafe bounds ∧
        ∀ ptrs, evalOp ptr s = some ptrs →
          ∀ i : TileIndex shape,
            active i →
            let bp := ptrs.data i
            let idx := TileShape.indexToList shape i
            bp.inBounds idx boundaryCheck = true →
              bp.address idx < bounds bp.region

/-- Namespace-facing wrapper for active address bounds. -/
abbrev MemAccess.ActiveAddressSafe {dtype : TileDType} (bounds : RegionBounds)
    (mem : MemAccess dtype shape) (s : BlockState)
    (active : TileIndex shape → Prop) : Prop :=
  memAccessActiveAddressSafe bounds mem s active
namespace MaskOpt

/-- Child expressions inside a mask are memory-safe. -/
def MemorySafe (bounds : RegionBounds) : MaskOpt dtype shape → Prop
  | .none => True
  | .mask m => m.MemorySafe bounds
  | .maskOther m other => m.MemorySafe bounds ∧ other.MemorySafe bounds

end MaskOpt

namespace Stmt

mutual

/-- State-independent memory-safety contract for statements. -/
def MemorySafe (bounds : RegionBounds) : Stmt → Prop
  | .assign _ _ _ e => e.MemorySafe bounds
  | .store _ _ mem value mask =>
      mem.MemorySafe bounds ∧
      value.MemorySafe bounds ∧
      mask.MemorySafe bounds ∧
      ∀ s, mem.ActiveAddressSafe bounds s (mask.Active s)
  | .atomicAdd _ _ mem value mask =>
      mem.MemorySafe bounds ∧
      value.MemorySafe bounds ∧
      mask.MemorySafe bounds ∧
      ∀ s, mem.ActiveAddressSafe bounds s (mask.Active s)
  | .atomicRMW _ _ _ mem input extraInput mask _ =>
      mem.MemorySafe bounds ∧
      input.MemorySafe bounds ∧
      (match extraInput with
        | none => True
        | some extra => extra.MemorySafe bounds) ∧
      mask.MemorySafe bounds ∧
      ∀ s, mem.ActiveAddressSafe bounds s (mask.Active s)
  | .forLoop _ _ body => MemorySafeList bounds body
  | .forRange _ _ _ _ body => MemorySafeList bounds body
  | .forRangeDyn _ start stop step body =>
      start.MemorySafe bounds ∧ stop.MemorySafe bounds ∧ step.MemorySafe bounds ∧
        MemorySafeList bounds body
  | .ifThen cond body => cond.MemorySafe bounds ∧ MemorySafeList bounds body
  | .ifThenElse cond thenBody elseBody =>
      cond.MemorySafe bounds ∧
      MemorySafeList bounds thenBody ∧
      MemorySafeList bounds elseBody

/-- Memory-safety contract for statement lists. -/
def MemorySafeList (bounds : RegionBounds) : List Stmt → Prop
  | [] => True
  | st :: rest => st.MemorySafe bounds ∧ MemorySafeList bounds rest

end

end Stmt

namespace StmtList

/-- Namespace wrapper for statement-list memory safety. -/
def MemorySafe (bounds : RegionBounds) (body : List Stmt) : Prop :=
  Stmt.MemorySafeList bounds body

end StmtList

namespace Kernel

/-- Kernel-level Layer-1 bounds-safety contract. -/
def MemorySafe (bounds : RegionBounds) (k : Kernel) : Prop :=
  StmtList.MemorySafe bounds k.body

@[simp] theorem memorySafe_mk (bounds : RegionBounds)
    (inputs outputs body) :
    (Kernel.mk inputs outputs body).MemorySafe bounds = StmtList.MemorySafe bounds body := rfl

end Kernel

namespace ComputeKernel

/-- Compute-kernel memory safety through the algorithm projection. -/
def MemorySafe (bounds : RegionBounds) (ck : ComputeKernel) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok k => k.MemorySafe bounds
  | Except.error _ => False

end ComputeKernel

end VeriTile.Triton

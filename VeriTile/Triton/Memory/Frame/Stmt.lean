/-
VeriTile.Triton.Memory.Frame.Stmt

Per-statement frame contracts for stores and atomics.
-/

import VeriTile.Triton.Memory.Frame.Basic

namespace VeriTile.Triton

namespace Stmt

/-- Active store addresses are included in the supplied footprint.

This predicate mirrors `stepStmt`'s store cases: masked-off lanes and checked
block-pointer OOB lanes do not write and therefore need no footprint membership. -/
def StoreAddressesWithin (P : WriteFootprint) (s : BlockState)
    (mem : MemAccess shape) (mask : MaskOpt dtype shape) : Prop :=
  match mask with
  | .none =>
      match mem with
      | .region region off =>
          ∀ offsets, evalOp off s = some offsets →
            ∀ i : TileIndex shape, P (region, offsets.data i)
      | .ptr ptr =>
          ∀ ptrs, evalOp ptr s = some ptrs →
            ∀ i : TileIndex shape, P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ ptrs, evalOp ptr s = some ptrs →
            ∀ i : TileIndex shape,
              (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                P ((ptrs.data i).region,
                  (ptrs.data i).address (TileShape.indexToList shape i))
  | .mask maskOp =>
      match mem with
      | .region region off =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ offsets, evalOp off s = some offsets →
              ∀ i : TileIndex shape, masks.data i = true → P (region, offsets.data i)
      | .ptr ptr =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape, masks.data i = true → P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                masks.data i = true →
                (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                  P ((ptrs.data i).region,
                    (ptrs.data i).address (TileShape.indexToList shape i))
  | .maskOther maskOp _ =>
      match mem with
      | .region region off =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ offsets, evalOp off s = some offsets →
              ∀ i : TileIndex shape, masks.data i = true → P (region, offsets.data i)
      | .ptr ptr =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape, masks.data i = true → P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                masks.data i = true →
                (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                  P ((ptrs.data i).region,
                    (ptrs.data i).address (TileShape.indexToList shape i))

end Stmt

theorem stepStmt_store_writeWithin {P : WriteFootprint} {s s' : BlockState}
    {dtype : TileDType} {shape : TileShape}
    {mem : MemAccess shape} {value : Op dtype shape} {mask : MaskOpt dtype shape}
    (hStep : stepStmt (Stmt.store dtype shape mem value mask) s = some s')
    (hAddr : Stmt.StoreAddressesWithin P s mem mask) :
    BlockState.WriteWithin P s s' := by
  cases hvalue : evalOp value s with
  | none =>
      unfold stepStmt at hStep
      simp [hvalue] at hStep
  | some values =>
      cases mask with
      | none =>
          cases mem with
          | region region off =>
              cases hoff : evalOp off s with
              | none => simp [stepStmt, hvalue, hoff] at hStep
              | some offsets =>
                  simp [stepStmt, hvalue, hoff] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun _ : TileIndex shape => region)
                    (fun i : TileIndex shape => offsets.data i)
                    (fun i : TileIndex shape => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr offsets hoff i)
          | ptr ptr =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun i : TileIndex shape => (ptrs.data i).1)
                    (fun i : TileIndex shape => (ptrs.data i).2)
                    (fun i : TileIndex shape => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr ptrs hptr i)
          | blockPtr ptr boundaryCheck =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun i : TileIndex shape => (ptrs.data i).region)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).address (TileShape.indexToList shape i))
                    (fun i : TileIndex shape => values.data i)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                    (TileShape.allIndices shape) s
                    (fun i hi hactive => hAddr ptrs hptr i hactive)
      | mask maskOp =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_prop_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)
      | maskOther maskOp other =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_prop_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)

theorem stepStmt_atomicAdd_writeWithin {P : WriteFootprint} {s s' : BlockState}
    {dtype : TileDType} {hnum : NumericDType dtype} {shape : TileShape}
    {mem : MemAccess shape} {value : Op dtype shape} {mask : MaskOpt dtype shape}
    (hStep : stepStmt (Stmt.atomicAdd hnum shape mem value mask) s = some s')
    (hAddr : Stmt.StoreAddressesWithin P s mem mask) :
    BlockState.WriteWithin P s s' := by
  cases hvalue : evalOp value s with
  | none =>
      unfold stepStmt at hStep
      simp [hvalue] at hStep
  | some values =>
      cases mask with
      | none =>
          cases mem with
          | region region off =>
              cases hoff : evalOp off s with
              | none => simp [stepStmt, hvalue, hoff] at hStep
              | some offsets =>
                  simp [stepStmt, hvalue, hoff] at hStep
                  cases hStep
                  exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                    (fun _ : TileIndex shape => region)
                    (fun i : TileIndex shape => offsets.data i)
                    (fun _ i => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr offsets hoff i)
          | ptr ptr =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                    (fun i : TileIndex shape => (ptrs.data i).1)
                    (fun i : TileIndex shape => (ptrs.data i).2)
                    (fun _ i => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr ptrs hptr i)
          | blockPtr ptr boundaryCheck =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                    (fun i : TileIndex shape => (ptrs.data i).region)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).address (TileShape.indexToList shape i))
                    (fun _ i => values.data i)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                    (TileShape.allIndices shape) s
                    (fun i hi hactive => hAddr ptrs hptr i hactive)
      | mask maskOp =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_prop_writeWithin dtype hnum
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)
      | maskOther maskOp other =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_masked_writeWithin dtype hnum
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_atomicAddAt_prop_writeWithin dtype hnum
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun _ i => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)

end VeriTile.Triton

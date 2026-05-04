/-
VeriTile.Triton.Memory.Footprint

Proof-ergonomic write-footprint constructors and extraction lemmas.
-/

import VeriTile.Triton.Memory.Frame

namespace VeriTile.Triton

namespace WriteFootprint

/-- Image of a tile of offsets into one memory region. -/
def tileImage {shape : TileShape}
    (region : RegionName) (offset : TileIndex shape → Nat) : WriteFootprint :=
  fun addr => addr.1 = region ∧ ∃ i : TileIndex shape, offset i = addr.2

/-- Image of a tile of offsets into one memory region, restricted to active lanes. -/
def activeTileImage {shape : TileShape}
    (region : RegionName) (offset : TileIndex shape → Nat)
    (active : TileIndex shape → Prop) : WriteFootprint :=
  fun addr => addr.1 = region ∧
    ∃ i : TileIndex shape, active i ∧ offset i = addr.2

/-- Image of a tile of full `(region, offset)` addresses. Useful for block pointers. -/
def addrTileImage {shape : TileShape}
    (addr : TileIndex shape → MemCellAddr) : WriteFootprint :=
  fun cell => ∃ i : TileIndex shape, addr i = cell

/-- Active-lane image of full `(region, offset)` addresses. -/
def activeAddrTileImage {shape : TileShape}
    (addr : TileIndex shape → MemCellAddr)
    (active : TileIndex shape → Prop) : WriteFootprint :=
  fun cell => ∃ i : TileIndex shape, active i ∧ addr i = cell

@[simp] theorem mem_tileImage {shape : TileShape}
    {region region' : RegionName} {offset : TileIndex shape → Nat} {n : Nat} :
    tileImage region offset (region', n) ↔
      region' = region ∧ ∃ i : TileIndex shape, offset i = n := Iff.rfl

@[simp] theorem mem_activeTileImage {shape : TileShape}
    {region region' : RegionName} {offset : TileIndex shape → Nat}
    {active : TileIndex shape → Prop} {n : Nat} :
    activeTileImage region offset active (region', n) ↔
      region' = region ∧
        ∃ i : TileIndex shape, active i ∧ offset i = n := Iff.rfl

@[simp] theorem mem_addrTileImage {shape : TileShape}
    {addrFn : TileIndex shape → MemCellAddr} {addr : MemCellAddr} :
    addrTileImage addrFn addr ↔ ∃ i : TileIndex shape, addrFn i = addr := Iff.rfl

@[simp] theorem mem_activeAddrTileImage {shape : TileShape}
    {addrFn : TileIndex shape → MemCellAddr}
    {active : TileIndex shape → Prop} {addr : MemCellAddr} :
    activeAddrTileImage addrFn active addr ↔
      ∃ i : TileIndex shape, active i ∧ addrFn i = addr := Iff.rfl

theorem activeTileImage_subset_tileImage {shape : TileShape}
    {region : RegionName} {offset : TileIndex shape → Nat}
    {active : TileIndex shape → Prop} :
    ∀ addr, activeTileImage region offset active addr → tileImage region offset addr := by
  intro addr h
  rcases addr with ⟨region', n⟩
  rcases h with ⟨hr, i, _, hoff⟩
  exact ⟨hr, i, hoff⟩

theorem activeAddrTileImage_subset_addrTileImage {shape : TileShape}
    {addrFn : TileIndex shape → MemCellAddr}
    {active : TileIndex shape → Prop} :
    ∀ addr, activeAddrTileImage addrFn active addr → addrTileImage addrFn addr := by
  intro addr h
  rcases h with ⟨i, _, hi⟩
  exact ⟨i, hi⟩

theorem disjoint_tileImage_of_region_ne
    {r₁ r₂ : RegionName} (h : r₁ ≠ r₂)
    {shape₁ shape₂ : TileShape}
    (off₁ : TileIndex shape₁ → Nat)
    (off₂ : TileIndex shape₂ → Nat) :
    WriteFootprint.disjoint
      (WriteFootprint.tileImage r₁ off₁)
      (WriteFootprint.tileImage r₂ off₂) := by
  intro addr h₁ h₂
  rcases addr with ⟨region, offset⟩
  exact h (h₁.1.symm.trans h₂.1)

theorem disjoint_activeTileImage_of_region_ne
    {r₁ r₂ : RegionName} (h : r₁ ≠ r₂)
    {shape₁ shape₂ : TileShape}
    (off₁ : TileIndex shape₁ → Nat) (active₁ : TileIndex shape₁ → Prop)
    (off₂ : TileIndex shape₂ → Nat) (active₂ : TileIndex shape₂ → Prop) :
    WriteFootprint.disjoint
      (WriteFootprint.activeTileImage r₁ off₁ active₁)
      (WriteFootprint.activeTileImage r₂ off₂ active₂) := by
  intro addr h₁ h₂
  rcases addr with ⟨region, offset⟩
  exact h (h₁.1.symm.trans h₂.1)

theorem disjoint_tileImage_of_image_disjoint
    {r : RegionName} {shape₁ shape₂ : TileShape}
    {off₁ : TileIndex shape₁ → Nat}
    {off₂ : TileIndex shape₂ → Nat}
    (h : ∀ i j, off₁ i ≠ off₂ j) :
    WriteFootprint.disjoint
      (WriteFootprint.tileImage r off₁)
      (WriteFootprint.tileImage r off₂) := by
  intro addr h₁ h₂
  rcases addr with ⟨region, offset⟩
  rcases h₁ with ⟨_, i, hi⟩
  rcases h₂ with ⟨_, j, hj⟩
  exact h i j (hi.trans hj.symm)

theorem disjoint_activeTileImage_of_image_disjoint
    {r : RegionName} {shape₁ shape₂ : TileShape}
    {off₁ : TileIndex shape₁ → Nat} {active₁ : TileIndex shape₁ → Prop}
    {off₂ : TileIndex shape₂ → Nat} {active₂ : TileIndex shape₂ → Prop}
    (h : ∀ i j, active₁ i → active₂ j → off₁ i ≠ off₂ j) :
    WriteFootprint.disjoint
      (WriteFootprint.activeTileImage r off₁ active₁)
      (WriteFootprint.activeTileImage r off₂ active₂) := by
  intro addr h₁ h₂
  rcases addr with ⟨region, offset⟩
  rcases h₁ with ⟨_, i, hai, hi⟩
  rcases h₂ with ⟨_, j, haj, hj⟩
  exact h i j hai haj (hi.trans hj.symm)

theorem disjoint_addrTileImage_of_image_disjoint
    {shape₁ shape₂ : TileShape}
    {addr₁ : TileIndex shape₁ → MemCellAddr}
    {addr₂ : TileIndex shape₂ → MemCellAddr}
    (h : ∀ i j, addr₁ i ≠ addr₂ j) :
    WriteFootprint.disjoint
      (WriteFootprint.addrTileImage addr₁)
      (WriteFootprint.addrTileImage addr₂) := by
  intro addr h₁ h₂
  rcases h₁ with ⟨i, hi⟩
  rcases h₂ with ⟨j, hj⟩
  exact h i j (hi.trans hj.symm)

theorem disjoint_activeAddrTileImage_of_image_disjoint
    {shape₁ shape₂ : TileShape}
    {addr₁ : TileIndex shape₁ → MemCellAddr} {active₁ : TileIndex shape₁ → Prop}
    {addr₂ : TileIndex shape₂ → MemCellAddr} {active₂ : TileIndex shape₂ → Prop}
    (h : ∀ i j, active₁ i → active₂ j → addr₁ i ≠ addr₂ j) :
    WriteFootprint.disjoint
      (WriteFootprint.activeAddrTileImage addr₁ active₁)
      (WriteFootprint.activeAddrTileImage addr₂ active₂) := by
  intro addr h₁ h₂
  rcases h₁ with ⟨i, hai, hi⟩
  rcases h₂ with ⟨j, haj, hj⟩
  exact h i j hai haj (hi.trans hj.symm)

theorem not_tileImage_of_region_ne {shape : TileShape}
    {region other : RegionName} {off : TileIndex shape → Nat} {offset : Nat}
    (h : other ≠ region) :
    ¬ WriteFootprint.tileImage region off (other, offset) := by
  intro hmem
  exact h hmem.1

theorem not_activeTileImage_of_region_ne {shape : TileShape}
    {region other : RegionName} {off : TileIndex shape → Nat}
    {active : TileIndex shape → Prop} {offset : Nat}
    (h : other ≠ region) :
    ¬ WriteFootprint.activeTileImage region off active (other, offset) := by
  intro hmem
  exact h hmem.1

theorem not_addrTileImage_of_forall_ne {shape : TileShape}
    {addr : TileIndex shape → MemCellAddr} {cell : MemCellAddr}
    (h : ∀ i, addr i ≠ cell) :
    ¬ WriteFootprint.addrTileImage addr cell := by
  intro hmem
  rcases hmem with ⟨i, hi⟩
  exact h i hi

theorem not_activeAddrTileImage_of_forall_ne {shape : TileShape}
    {addr : TileIndex shape → MemCellAddr}
    {active : TileIndex shape → Prop} {cell : MemCellAddr}
    (h : ∀ i, active i → addr i ≠ cell) :
    ¬ WriteFootprint.activeAddrTileImage addr active cell := by
  intro hmem
  rcases hmem with ⟨i, hactive, hi⟩
  exact h i hactive hi

end WriteFootprint

namespace Stmt

theorem storeAddressesWithin_region_unmasked {shape : TileShape} {dtype : TileDType}
    (region : RegionName) (off : Op .nat shape) (s : BlockState)
    (offsets : Tile .nat shape) (hOffsets : evalOp off s = some offsets) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.tileImage region (fun i => offsets.data i))
      s (.region region off) (MaskOpt.none : MaskOpt dtype shape) := by
  intro offsets' hOffsets' i
  have hEq : offsets' = offsets := Option.some.inj (hOffsets'.symm.trans hOffsets)
  subst hEq
  exact ⟨rfl, i, rfl⟩

theorem storeAddressesWithin_region_mask {shape : TileShape}
    {dtype : TileDType}
    (region : RegionName) (off : Op .nat shape) (maskOp : Op .bool shape)
    (s : BlockState)
    (offsets : Tile .nat shape) (hOffsets : evalOp off s = some offsets)
    (masks : Tile .bool shape) (hMasks : evalOp maskOp s = some masks) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.activeTileImage region
        (fun i => offsets.data i) (fun i => masks.data i = true))
      s (.region region off) (.mask maskOp : MaskOpt dtype shape) := by
  intro masks' hMasks' offsets' hOffsets' i hActive
  have hm : masks' = masks := Option.some.inj (hMasks'.symm.trans hMasks)
  have ho : offsets' = offsets := Option.some.inj (hOffsets'.symm.trans hOffsets)
  subst hm
  subst ho
  exact ⟨rfl, i, hActive, rfl⟩

theorem storeAddressesWithin_region_maskOther {shape : TileShape}
    {dtype : TileDType}
    (region : RegionName) (off : Op .nat shape) (maskOp : Op .bool shape)
    (other : Op dtype shape) (s : BlockState)
    (offsets : Tile .nat shape) (hOffsets : evalOp off s = some offsets)
    (masks : Tile .bool shape) (hMasks : evalOp maskOp s = some masks) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.activeTileImage region
        (fun i => offsets.data i) (fun i => masks.data i = true))
      s (.region region off) (.maskOther maskOp other : MaskOpt dtype shape) := by
  intro masks' hMasks' offsets' hOffsets' i hActive
  have hm : masks' = masks := Option.some.inj (hMasks'.symm.trans hMasks)
  have ho : offsets' = offsets := Option.some.inj (hOffsets'.symm.trans hOffsets)
  subst hm
  subst ho
  exact ⟨rfl, i, hActive, rfl⟩

theorem storeAddressesWithin_blockPtr_unmasked {shape : TileShape} {dtype : TileDType}
    (ptr : Op .blockPtr shape) (boundaryCheck : List Nat) (s : BlockState)
    (ptrs : Tile .blockPtr shape) (hPtrs : evalOp ptr s = some ptrs) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.activeAddrTileImage
        (fun i : TileIndex shape =>
          let bp := ptrs.data i
          (bp.region, bp.address (TileShape.indexToList shape i)))
      (fun i : TileIndex shape =>
          (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true))
      s (.blockPtr ptr boundaryCheck) (MaskOpt.none : MaskOpt dtype shape) := by
  intro ptrs' hPtrs' i hInBounds
  have hp : ptrs' = ptrs := Option.some.inj (hPtrs'.symm.trans hPtrs)
  subst hp
  exact ⟨i, hInBounds, rfl⟩

theorem storeAddressesWithin_blockPtr_mask {shape : TileShape}
    {dtype : TileDType}
    (ptr : Op .blockPtr shape) (boundaryCheck : List Nat)
    (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .blockPtr shape) (hPtrs : evalOp ptr s = some ptrs)
    (masks : Tile .bool shape) (hMasks : evalOp maskOp s = some masks) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.activeAddrTileImage
        (fun i : TileIndex shape =>
          let bp := ptrs.data i
          (bp.region, bp.address (TileShape.indexToList shape i)))
        (fun i : TileIndex shape =>
          masks.data i = true ∧
            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true))
      s (.blockPtr ptr boundaryCheck) (.mask maskOp : MaskOpt dtype shape) := by
  intro masks' hMasks' ptrs' hPtrs' i hMask hInBounds
  have hm : masks' = masks := Option.some.inj (hMasks'.symm.trans hMasks)
  have hp : ptrs' = ptrs := Option.some.inj (hPtrs'.symm.trans hPtrs)
  subst hm
  subst hp
  exact ⟨i, ⟨hMask, hInBounds⟩, rfl⟩

theorem storeAddressesWithin_blockPtr_maskOther {shape : TileShape}
    {dtype : TileDType}
    (ptr : Op .blockPtr shape) (boundaryCheck : List Nat)
    (maskOp : Op .bool shape) (other : Op dtype shape) (s : BlockState)
    (ptrs : Tile .blockPtr shape) (hPtrs : evalOp ptr s = some ptrs)
    (masks : Tile .bool shape) (hMasks : evalOp maskOp s = some masks) :
    Stmt.StoreAddressesWithin
      (WriteFootprint.activeAddrTileImage
        (fun i : TileIndex shape =>
          let bp := ptrs.data i
          (bp.region, bp.address (TileShape.indexToList shape i)))
        (fun i : TileIndex shape =>
          masks.data i = true ∧
            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true))
      s (.blockPtr ptr boundaryCheck) (.maskOther maskOp other : MaskOpt dtype shape) := by
  intro masks' hMasks' ptrs' hPtrs' i hMask hInBounds
  have hm : masks' = masks := Option.some.inj (hMasks'.symm.trans hMasks)
  have hp : ptrs' = ptrs := Option.some.inj (hPtrs'.symm.trans hPtrs)
  subst hm
  subst hp
  exact ⟨i, ⟨hMask, hInBounds⟩, rfl⟩

end Stmt

namespace BlockState

theorem writeWithin_mono {P Q : WriteFootprint} {s₀ s₁ : BlockState}
    (hSub : ∀ addr, P addr → Q addr)
    (h : BlockState.WriteWithin P s₀ s₁) :
    BlockState.WriteWithin Q s₀ s₁ := by
  intro region offset hnot
  exact h region offset (fun hp => hnot (hSub (region, offset) hp))

theorem writeWithin_union_of_steps {P Q : WriteFootprint}
    {s₀ s₁ s₂ : BlockState}
    (h₀₁ : BlockState.WriteWithin P s₀ s₁)
    (h₁₂ : BlockState.WriteWithin Q s₁ s₂) :
    BlockState.WriteWithin (WriteFootprint.union P Q) s₀ s₂ := by
  intro region offset hnot
  have hp : ¬ P (region, offset) := fun h => hnot (Or.inl h)
  have hq : ¬ Q (region, offset) := fun h => hnot (Or.inr h)
  exact (h₀₁ region offset hp).trans (h₁₂ region offset hq)

end BlockState

namespace Kernel

private theorem stepStmts_writeWithin_of_perStmt
    (body : List Stmt)
    (writes : Stmt → WriteFootprint)
    (h_each : ∀ stmt ∈ body, ∀ s s',
      stepStmt stmt s = some s' →
      BlockState.WriteWithin (writes stmt) s s') :
    ∀ {s s'}, stepStmts body s = some s' →
      BlockState.WriteWithin
        (body.foldr (fun st acc => WriteFootprint.union (writes st) acc)
          WriteFootprint.empty)
        s s' := by
  induction body with
  | nil =>
      intro s s' h
      simp at h
      cases h
      exact BlockState.writeWithin_refl WriteFootprint.empty s
  | cons st rest ih =>
      intro s s' h
      unfold stepStmts at h
      cases hStep : stepStmt st s with
      | none =>
          simp [hStep] at h
      | some mid =>
          have hRest : stepStmts rest mid = some s' := by
            simpa [hStep] using h
          have hFirst : BlockState.WriteWithin (writes st) s mid :=
            h_each st List.mem_cons_self s mid hStep
          have hTail : BlockState.WriteWithin
              (rest.foldr (fun st acc => WriteFootprint.union (writes st) acc)
                WriteFootprint.empty)
              mid s' :=
            ih (fun stmt hmem => h_each stmt (List.mem_cons_of_mem st hmem)) hRest
          exact BlockState.writeWithin_union_of_steps hFirst hTail

theorem execWritesWithin_of_perStmt
    (k : Kernel) (writes : Stmt → WriteFootprint)
    (h_each : ∀ stmt ∈ k.body, ∀ s s',
      stepStmt stmt s = some s' →
      BlockState.WriteWithin (writes stmt) s s') :
    ∀ s, Kernel.ExecWritesWithin k s
      (k.body.foldr (fun st acc =>
        WriteFootprint.union (writes st) acc) WriteFootprint.empty) := by
  intro s s' hExec
  exact stepStmts_writeWithin_of_perStmt k.body writes h_each hExec

end Kernel

end VeriTile.Triton

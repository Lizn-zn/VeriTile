/-
VeriTile.Examples.MemoryFrame

Representative Layer-2a write-footprint / frame examples.
-/

import VeriTile.Triton.Memory.Footprint

namespace VeriTile.Examples.MemoryFrame

open VeriTile.Triton

/-- Scalar direct store to one known cell. -/
def scalarDirectStoreKernel (outReg : RegionName) : Kernel :=
  { inputs := []
    outputs := [outReg]
    body :=
      [ Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
          (Op.const 1) MaskOpt.none
      ] }

/-- Masked-false store; no lane writes. -/
def maskedFalseStoreKernel (outReg : RegionName) : Kernel :=
  { inputs := []
    outputs := [outReg]
    body :=
      [ Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
          (Op.const 1) (MaskOpt.mask (Op.constBool false))
      ] }

/-- Checked-OOB block-pointer store; boundary check skips the only lane. -/
def blockPtrOOBStoreKernel (outReg : RegionName) : Kernel :=
  let ptr : Op .blockPtr [1] :=
    Op.makeBlockPtr outReg 0 [0] [1] [1] [0]
  { inputs := []
    outputs := [outReg]
    body :=
      [ Stmt.store .real [1] (MemAccess.blockPtr ptr [0])
          (Op.full [1] (Op.const 1)) MaskOpt.none
      ] }

private def oobPtr (outReg : RegionName) : BlockPtr :=
  { region := outReg, baseOffset := 0, parentShape := [0],
    blockShape := [1], strides := [1], offsets := [0] }

private def oobPtrTile (outReg : RegionName) : Tile .blockPtr [1] :=
  ⟨fun _ => oobPtr outReg⟩

private def scalarDirectStoreWrites (outReg : RegionName) (_ : Stmt) :
    WriteFootprint :=
  WriteFootprint.tileImage outReg (fun _ : TileIndex [] => 0)

theorem scalarDirectStore_writesWithin_tileImage
    (outReg : RegionName) (s : BlockState) :
    (scalarDirectStoreKernel outReg).ExecWritesWithin s
      (WriteFootprint.tileImage outReg (fun _ : TileIndex [] => 0)) := by
  intro s' hExec
  let st : Stmt :=
    Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
      (Op.const 1) MaskOpt.none
  have hStep : stepStmt st s = some s' := by
    cases hst : stepStmt st s with
    | none =>
        simp [exec, scalarDirectStoreKernel, st,
          stepStmts, hst] at hExec
    | some mid =>
        simp [exec, scalarDirectStoreKernel, st,
          stepStmts, hst] at hExec
        cases hExec
        rfl
  exact stepStmt_store_writeWithin
      (P := WriteFootprint.tileImage outReg (fun _ : TileIndex [] => 0))
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      exact Stmt.storeAddressesWithin_region_unmasked outReg (Op.constNat 0)
        s (Tile.scalar 0) (by simp [evalOp]))

theorem scalarDirectStore_preserves_unrelated_region
    (outReg otherReg : RegionName) (s s' : BlockState)
    (hOther : otherReg ≠ outReg)
    (hExec : exec (scalarDirectStoreKernel outReg) s = some s') :
    s'.mem otherReg = s.mem otherReg := by
  exact Kernel.ExecWritesWithin.mem_eq_of_region_not_written
    (scalarDirectStore_writesWithin_tileImage outReg s) hExec
    (fun offset => WriteFootprint.not_tileImage_of_region_ne hOther)

theorem scalarDirectStore_writesWithin_perStmt
    (outReg : RegionName) (s : BlockState) :
    (scalarDirectStoreKernel outReg).ExecWritesWithin s
      ((scalarDirectStoreKernel outReg).body.foldr
        (fun st acc =>
          WriteFootprint.union (scalarDirectStoreWrites outReg st) acc)
        WriteFootprint.empty) := by
  apply Kernel.execWritesWithin_of_perStmt (writes := scalarDirectStoreWrites outReg)
  intro stmt hmem s₀ s' hStep
  simp [scalarDirectStoreKernel] at hmem
  rcases hmem with rfl
  exact stepStmt_store_writeWithin (s := s₀) (s' := s') (hStep := hStep)
    (hAddr := by
      exact Stmt.storeAddressesWithin_region_unmasked outReg (Op.constNat 0)
        s₀ (Tile.scalar 0) (by simp [evalOp]))

theorem maskedFalseStore_writesWithin_activeTileImage
    (outReg : RegionName) (s : BlockState) :
    (maskedFalseStoreKernel outReg).ExecWritesWithin s
      (WriteFootprint.activeTileImage outReg
        (fun _ : TileIndex [] => 0) (fun _ : TileIndex [] => False)) := by
  intro s' hExec
  let st : Stmt :=
    Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
      (Op.const 1) (MaskOpt.mask (Op.constBool false))
  have hStep : stepStmt st s = some s' := by
    cases hst : stepStmt st s with
    | none =>
        simp [exec, maskedFalseStoreKernel, st,
          stepStmts, hst] at hExec
    | some mid =>
        simp [exec, maskedFalseStoreKernel, st,
          stepStmts, hst] at hExec
        cases hExec
        rfl
  exact stepStmt_store_writeWithin
    (P := WriteFootprint.activeTileImage outReg
      (fun _ : TileIndex [] => 0) (fun _ : TileIndex [] => False))
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      convert Stmt.storeAddressesWithin_region_mask outReg (Op.constNat 0)
        (Op.constBool false) s (Tile.scalar 0) (by simp [evalOp])
        (Tile.scalar false) (by simp [evalOp]) using 1
      funext i
      simp)

specification blockPtrOOBStore_writesWithin_checkedFootprint
    (outReg : RegionName) (s : BlockState) :
    (blockPtrOOBStoreKernel outReg).ExecWritesWithin s
      (WriteFootprint.activeAddrTileImage
        (fun i : TileIndex [1] =>
          let bp := (oobPtrTile outReg).data i
          (bp.region, bp.address (TileShape.indexToList [1] i)))
        (fun i : TileIndex [1] =>
          let bp := (oobPtrTile outReg).data i
          bp.inBounds (TileShape.indexToList [1] i) [0] = true)) := by
  intro s' hExec
  let ptr : Op .blockPtr [1] :=
    Op.makeBlockPtr outReg 0 [0] [1] [1] [0]
  let st : Stmt :=
    Stmt.store .real [1] (MemAccess.blockPtr ptr [0])
      (Op.full [1] (Op.const 1)) MaskOpt.none
  have hStep : stepStmt st s = some s' := by
    cases hst : stepStmt st s with
    | none =>
        simp [exec, blockPtrOOBStoreKernel, st, ptr,
          stepStmts, hst] at hExec
    | some mid =>
        simp [exec, blockPtrOOBStoreKernel, st, ptr,
          stepStmts, hst] at hExec
        cases hExec
        rfl
  exact stepStmt_store_writeWithin
    (P := WriteFootprint.activeAddrTileImage
        (fun i : TileIndex [1] =>
          let bp := (oobPtrTile outReg).data i
          (bp.region, bp.address (TileShape.indexToList [1] i)))
        (fun i : TileIndex [1] =>
          let bp := (oobPtrTile outReg).data i
          bp.inBounds (TileShape.indexToList [1] i) [0] = true))
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      exact Stmt.storeAddressesWithin_blockPtr_unmasked ptr [0] s
        (oobPtrTile outReg) (by simp [ptr, oobPtrTile, oobPtr, evalOp]))

end VeriTile.Examples.MemoryFrame

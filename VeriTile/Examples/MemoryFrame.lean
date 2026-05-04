/-
VeriTile.Examples.MemoryFrame

Representative Layer-2a write-footprint / frame examples.
-/

import VeriTile.Triton.MemoryFrame

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

theorem scalarDirectStore_writesWithin_singleton
    (outReg : RegionName) (s : BlockState) :
    (scalarDirectStoreKernel outReg).ExecWritesWithin s
      (WriteFootprint.singleton outReg 0) := by
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
  exact stepStmt_store_writeWithin (P := WriteFootprint.singleton outReg 0)
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      intro offsets hOffsets i
      simp [evalOp] at hOffsets
      cases hOffsets
      simp [WriteFootprint.singleton])

theorem maskedFalseStore_writesWithin_empty
    (outReg : RegionName) (s : BlockState) :
    (maskedFalseStoreKernel outReg).ExecWritesWithin s WriteFootprint.empty := by
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
  exact stepStmt_store_writeWithin (P := WriteFootprint.empty)
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      intro masks hMasks offsets hOffsets i hActive
      simp [evalOp] at hMasks
      cases hMasks
      simp at hActive)

theorem blockPtrOOBStore_writesWithin_empty
    (outReg : RegionName) (s : BlockState) :
    (blockPtrOOBStoreKernel outReg).ExecWritesWithin s WriteFootprint.empty := by
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
  exact stepStmt_store_writeWithin (P := WriteFootprint.empty)
    (s := s) (s' := s') (hStep := hStep) (hAddr := by
      intro ptrs hPtrs i hInBounds
      simp [ptr, evalOp] at hPtrs
      cases hPtrs
      simp [BlockPtr.inBounds] at hInBounds)

end VeriTile.Examples.MemoryFrame

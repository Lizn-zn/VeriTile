/-
VeriTile.Examples.MemorySafety

Representative Layer-1 memory-bounds safety examples.
-/

import VeriTile.Triton.Memory.Bounds
import Mathlib.Tactic
import VeriTile.Meta.Specification

namespace VeriTile.Examples.MemorySafety

open VeriTile.Triton

/-- Baseline scalar copy with direct region+offset addressing. -/
def straightLineCopyKernel (xReg outReg : RegionName) : Kernel :=
  { inputs := [xReg]
    outputs := [outReg]
    body :=
      [ Stmt.assign .real [] "x"
          (Op.load .real (MemAccess.region xReg (Op.constNat 0)) MaskOpt.none)
      , Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
          (Op.ref .real [] "x") MaskOpt.none
      ] }

/-- A masked scalar add that models a single tail lane guarded by a runtime mask. -/
def maskedTailAddKernel (xReg yReg outReg : RegionName) (tailActive : Bool) : Kernel :=
  let activeMask : Op .bool [] := Op.constBool tailActive
  { inputs := [xReg, yReg]
    outputs := [outReg]
    body :=
      [ Stmt.assign .real [] "x"
          (Op.load .real (MemAccess.region xReg (Op.constNat 0)) (MaskOpt.mask activeMask))
      , Stmt.assign .real [] "y"
          (Op.load .real (MemAccess.region yReg (Op.constNat 0)) (MaskOpt.mask activeMask))
      , Stmt.assign .real [] "z"
          (Op.add NumericDType.real Broadcast.nil
            (Op.ref .real [] "x") (Op.ref .real [] "y"))
      , Stmt.store .real [] (MemAccess.region outReg (Op.constNat 0))
          (Op.ref .real [] "z") (MaskOpt.mask activeMask)
      ] }

/-- A checked block-pointer load whose only local lane is boundary-masked out. -/
def blockPtrBoundaryKernel (xReg outReg : RegionName) : Kernel :=
  let ptr : Op .blockPtr [1] :=
    Op.makeBlockPtr xReg 0 [0] [1] [1] [0]
  { inputs := [xReg]
    outputs := [outReg]
    body :=
      [ Stmt.assign .real [1] "x"
          (Op.load .real (MemAccess.blockPtr ptr [0]) MaskOpt.none)
      ] }

theorem straightLineCopy_memorySafe
    (bounds : RegionBounds) (xReg outReg : RegionName)
    (hx : 0 < bounds xReg) (hout : 0 < bounds outReg) :
    (straightLineCopyKernel xReg outReg).MemorySafe bounds := by
  simp [straightLineCopyKernel, Kernel.MemorySafe, StmtList.MemorySafe,
    Stmt.MemorySafeList, Stmt.MemorySafe, Op.MemorySafe, memAccessMemorySafe,
    memAccessActiveAddressSafe, MaskOpt.Active, MaskOpt.MemorySafe, evalOp, hx, hout]

theorem maskedTailAdd_memorySafe
    (bounds : RegionBounds) (xReg yReg outReg : RegionName) (tailActive : Bool)
    (hx : tailActive = true → 0 < bounds xReg)
    (hy : tailActive = true → 0 < bounds yReg)
    (hout : tailActive = true → 0 < bounds outReg) :
    (maskedTailAddKernel xReg yReg outReg tailActive).MemorySafe bounds := by
  simp [maskedTailAddKernel, Kernel.MemorySafe, StmtList.MemorySafe,
    Stmt.MemorySafeList, Stmt.MemorySafe, Op.MemorySafe, memAccessMemorySafe,
    memAccessActiveAddressSafe, MaskOpt.Active, MaskOpt.MemorySafe, evalOp]
  constructor
  · exact hx
  constructor
  · exact hy
  · exact hout

specification blockPtrBoundary_memorySafe
    (bounds : RegionBounds) (xReg outReg : RegionName) :
    (blockPtrBoundaryKernel xReg outReg).MemorySafe bounds := by
  simp [blockPtrBoundaryKernel, Kernel.MemorySafe, StmtList.MemorySafe,
    Stmt.MemorySafeList, Stmt.MemorySafe, Op.MemorySafe, MaskOpt.Active, evalOp]

end VeriTile.Examples.MemorySafety

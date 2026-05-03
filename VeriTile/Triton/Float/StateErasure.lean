/-
VeriTile.Triton.Float.StateErasure

State-level erasure for floating dtype annotations.
-/

import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-! ## State-level floating erasure -/

def eraseDTypeCarrier : (dtype : TileDType) →
    TileCarrier dtype → TileCarrier (eraseDType dtype)
  | .real, value => value
  | .fp32, value => FloatDType.fp32.toWithBot value
  | .fp16, value => FloatDType.fp16.toWithBot value
  | .bf16, value => FloatDType.bf16.toWithBot value
  | .int, value => value
  | .nat, value => value
  | .bool, value => value
  | .ptr, value => value
  | .blockPtr, value => value

def Tile.eraseDType {dtype : TileDType} {shape : TileShape}
    (tile : Tile dtype shape) : Tile (eraseDType dtype) shape :=
  ⟨fun idx => eraseDTypeCarrier dtype (tile.data idx)⟩

namespace BlockState

private def eraseDTypeRealReg (regs : RegFile) (shape : TileShape)
    (name : RegName) : Option (Tile .real shape) :=
  match regs .real shape name with
  | some value => some value
  | none =>
      match regs .fp32 shape name with
      | some value => some value.eraseDType
      | none =>
          match regs .fp16 shape name with
          | some value => some value.eraseDType
          | none =>
              match regs .bf16 shape name with
              | some value => some value.eraseDType
              | none => none

def eraseDTypeRegs (regs : RegFile) : RegFile :=
  fun dtype shape name =>
    match dtype with
    | .real => eraseDTypeRealReg regs shape name
    | .fp32 => none
    | .fp16 => none
    | .bf16 => none
    | .int => regs .int shape name
    | .nat => regs .nat shape name
    | .bool => regs .bool shape name
    | .ptr => regs .ptr shape name
    | .blockPtr => regs .blockPtr shape name

/-- Erase floating dtype tags in memory and registers, preserving launch ids and
the masked-load undef oracle. -/
def eraseDType (s : BlockState) : BlockState :=
  { mem := fun region offset => (s.mem region offset).eraseDType
  , regs := eraseDTypeRegs s.regs
  , pids := s.pids
  , undef := s.undef }

@[simp] theorem eraseDType_pids (s : BlockState) :
    s.eraseDType.pids = s.pids := rfl

@[simp] theorem eraseDType_pid (s : BlockState) :
    s.eraseDType.pid = s.pid := rfl

@[simp] theorem eraseDType_undef (s : BlockState) :
    s.eraseDType.undef = s.undef := rfl

end BlockState

end VeriTile.Triton

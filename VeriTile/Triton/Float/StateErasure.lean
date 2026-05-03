/-
VeriTile.Triton.Float.StateErasure

State-level erasure for floating dtype annotations.
-/

import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-! ## State-level floating erasure -/

def eraseFloatCarrier : (dtype : TileDType) →
    TileCarrier dtype → TileCarrier (eraseFloatDType dtype)
  | .real, value => value
  | .fp32, value => FloatDType.fp32.toWithBot value
  | .fp16, value => FloatDType.fp16.toWithBot value
  | .bf16, value => FloatDType.bf16.toWithBot value
  | .int32, value => value
  | .nat, value => value
  | .bool, value => value
  | .ptr, value => value
  | .blockPtr, value => value

def Tile.eraseFloat {dtype : TileDType} {shape : TileShape}
    (tile : Tile dtype shape) : Tile (eraseFloatDType dtype) shape :=
  ⟨fun idx => eraseFloatCarrier dtype (tile.data idx)⟩

namespace BlockState

private def eraseFloatRealReg (regs : RegFile) (shape : TileShape)
    (name : RegName) : Option (Tile .real shape) :=
  match regs .real shape name with
  | some value => some value
  | none =>
      match regs .fp32 shape name with
      | some value => some value.eraseFloat
      | none =>
          match regs .fp16 shape name with
          | some value => some value.eraseFloat
          | none =>
              match regs .bf16 shape name with
              | some value => some value.eraseFloat
              | none => none

def eraseFloatRegs (regs : RegFile) : RegFile :=
  fun dtype shape name =>
    match dtype with
    | .real => eraseFloatRealReg regs shape name
    | .fp32 => none
    | .fp16 => none
    | .bf16 => none
    | .int32 => regs .int32 shape name
    | .nat => regs .nat shape name
    | .bool => regs .bool shape name
    | .ptr => regs .ptr shape name
    | .blockPtr => regs .blockPtr shape name

/-- Erase floating dtype tags in memory and registers, preserving launch ids and
the masked-load undef oracle. -/
def eraseFloat (s : BlockState) : BlockState :=
  { mem := fun region offset => (s.mem region offset).eraseFloat
  , regs := eraseFloatRegs s.regs
  , pids := s.pids
  , undef := s.undef }

@[simp] theorem eraseFloat_pids (s : BlockState) :
    s.eraseFloat.pids = s.pids := rfl

@[simp] theorem eraseFloat_pid (s : BlockState) :
    s.eraseFloat.pid = s.pid := rfl

@[simp] theorem eraseFloat_undef (s : BlockState) :
    s.eraseFloat.undef = s.undef := rfl

end BlockState

end VeriTile.Triton

/-
VeriTile.Triton.Compute

Compute-facing dtype and bit-payload model.
-/

import Mathlib.Data.Rat.Defs
import VeriTile.Triton.Float

namespace VeriTile.Triton

/-! ## Bit payload wrappers -/

structure UInt32Bits where
  bits : BitVec 32
  deriving Repr, BEq, DecidableEq

structure Float32Bits where
  bits : BitVec 32
  deriving Repr, BEq, DecidableEq

namespace UInt32Bits

def toNat (x : UInt32Bits) : Nat :=
  x.bits.toNat

end UInt32Bits

namespace Float32Bits

private def pow2 (n : Nat) : Rat :=
  (2 : Rat) ^ n

private def scalePow2 (q : Rat) (e : Int) : Rat :=
  if _h : 0 ≤ e then
    q * pow2 e.toNat
  else
    q / pow2 (-e).toNat

/--
Computable finite-normal IEEE-754 binary32 decode to `Rat`.

This initial algorithm bridge intentionally supports only finite normal values.
Zeros, subnormals, infinities, and NaNs return `none` until the compute-correct
IEEE model is widened.
-/
def decodeRat (x : Float32Bits) : Option Rat :=
  let n := x.bits.toNat
  let sign := n / 2^31
  let exp := (n / 2^23) % 256
  let frac := n % 2^23
  if exp = 0 ∨ exp = 255 then
    none
  else
    let significand : Rat := (2^23 + frac : Nat)
    let signed := if sign = 0 then significand else -significand
    some (scalePow2 signed (Int.ofNat exp - 150))

noncomputable def decodeReal (x : Float32Bits) : WithBot ℝ :=
  match decodeRat x with
  | some q => some (q : ℝ)
  | none => none

example :
    decodeRat { bits := 0x3f800000#32 } = some 1 := by
  native_decide

end Float32Bits

/-! ## Compute dtype/op skeleton -/

inductive ComputeDType where
  | uint32
  | fp32
  deriving Repr, BEq, DecidableEq

namespace ComputeDType

def eraseDType : ComputeDType → AlgDType
  | .uint32 => .nat
  | .fp32 => .real

def width : ComputeDType → Nat
  | .uint32 => 32
  | .fp32 => 32

end ComputeDType

def ComputeCarrier : ComputeDType → Type
  | .uint32 => UInt32Bits
  | .fp32 => Float32Bits

inductive ComputeOp : ComputeDType → TileShape → Type where
  | const : ComputeCarrier dtype → ComputeOp dtype []
  | bitcast :
      (src dst : ComputeDType) →
      src.width = dst.width →
      ComputeOp src shape →
      ComputeOp dst shape

namespace ComputeOp

def bitcastPayload (src dst : ComputeDType) :
    ComputeCarrier src → Option (ComputeCarrier dst) :=
  match src, dst with
  | .uint32, .fp32 => fun x => some ({ bits := x.bits } : Float32Bits)
  | .fp32, .uint32 => fun x => some ({ bits := x.bits } : UInt32Bits)
  | .uint32, .uint32 => fun x => some x
  | .fp32, .fp32 => fun x => some x

def constPayload? : ComputeOp dtype [] → Option (ComputeCarrier dtype)
  | .const value => some value
  | .bitcast src dst _ e =>
      match constPayload? e with
      | some value => bitcastPayload src dst value
      | none => none

def constToAlgorithm? :
    (dtype : ComputeDType) → ComputeCarrier dtype →
      Except ComputeKernel.EraseDTypeError (Op dtype.eraseDType [])
  | .uint32, value => Except.ok (Op.constNat value.toNat)
  | .fp32, value =>
      match Float32Bits.decodeRat value with
      | some q => Except.ok (Op.const (q : ℝ))
      | none => Except.error (.unsupportedBitcast "unsupported fp32 decode")

def constOpToAlgorithm? (op : ComputeOp dtype []) :
    Except ComputeKernel.EraseDTypeError (Op dtype.eraseDType []) := do
  let value ←
    match constPayload? op with
    | some value => Except.ok value
    | none => Except.error (.requiresComputeSemantics "runtime bitcast")
  constToAlgorithm? dtype value

def oneBits : UInt32Bits :=
  { bits := 0x3f800000#32 }

def oneBitcast : ComputeOp .fp32 [] :=
  ComputeOp.bitcast .uint32 .fp32 rfl (ComputeOp.const oneBits)

example :
    constOpToAlgorithm? oneBitcast = Except.ok (Op.const 1) := by
  have hdecode :
      Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) = some 1 := by
    native_decide
  simp [constOpToAlgorithm?, oneBitcast, oneBits, constPayload?, bitcastPayload,
    constToAlgorithm?]
  change
    (match Float32Bits.decodeRat ({ bits := 0x3f800000#32 } : Float32Bits) with
      | some q => Except.ok (Op.const (q : ℝ))
      | none => Except.error (.unsupportedBitcast "unsupported fp32 decode")) =
    Except.ok (Op.const 1)
  rw [hdecode]
  norm_num

end ComputeOp

end VeriTile.Triton

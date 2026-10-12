import Mathlib.Data.Rat.Defs
import Mathlib.Data.Int.DivMod

/-! E5M2 byte representation used by `tl.float8e5`. The format has a sign
bit, five exponent bits (bias 15), and two fraction bits. Exponent 31 encodes
infinities and NaNs; exponent zero encodes signed zeros and subnormals.
See https://arxiv.org/html/2209.05433v2#S3, Table 1.

The representation keeps every bit, including the sign of zero and NaN
payloads. `decodeRat` is a separate, partial projection to finite numbers.
It does not assign a real value to infinity or NaN. -/
namespace VeriTile.Triton

abbrev Float8E5Bits := Fin 256

namespace Float8E5Bits

def sign (b : Float8E5Bits) : Bool := decide (128 ≤ b.val)
def exponent (b : Float8E5Bits) : Nat := b.val / 4 % 32
def fraction (b : Float8E5Bits) : Nat := b.val % 4

inductive Class where
  | zero | subnormal | normal | infinity | nan
  deriving DecidableEq, Repr

def classify (b : Float8E5Bits) : Class :=
  if b.exponent = 31 then
    if b.fraction = 0 then .infinity else .nan
  else if b.exponent = 0 then
    if b.fraction = 0 then .zero else .subnormal
  else .normal

/-- Extract the low byte. For a signed int8 value this is its two's-complement
representation. Callers interpreting an int8 channel check `isInt8` first. -/
def ofInt (n : Int) : Float8E5Bits :=
  ⟨(n % 256).toNat, by omega⟩

def isInt8 (n : Int) : Bool := decide (-128 ≤ n ∧ n < 128)

/-- Finite E5M2 values are dyadic rationals. Subnormals are `fraction * 2^-16`;
normal values are `(4 + fraction) * 2^(exponent - 17)`. -/
def decodeRat (b : Float8E5Bits) : Option ℚ :=
  if b.exponent = 31 then none else
    let magnitude : ℚ :=
      if b.exponent = 0 then (b.fraction : ℚ) / 65536
      else if b.exponent < 17 then (4 + b.fraction : ℚ) / (2 ^ (17 - b.exponent))
      else (4 + b.fraction : ℚ) * (2 ^ (b.exponent - 17))
    some (if b.sign then -magnitude else magnitude)

@[simp] theorem decodeRat_isSome (b : Float8E5Bits) :
    b.decodeRat.isSome = decide (b.exponent ≠ 31) := by
  simp only [decodeRat]
  split <;> simp_all

end Float8E5Bits
end VeriTile.Triton

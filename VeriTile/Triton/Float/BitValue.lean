/-
Concrete bf16/fp32 values. This layer is independent of the existing Real-valued
TileCarrier and abstract RoundingModel: importing it does not change old proofs.
-/
import Init.Data.Rat.Basic
import Init.Data.Nat.Log2
import Init.Data.BitVec.Basic
import VeriTile.Meta.Specification

namespace VeriTile.Triton.FP

inductive Format where
  | bf16 | fp32
  deriving Repr, BEq, DecidableEq

def Format.fractionBits : Format → Nat
  | .bf16 => 7
  | .fp32 => 23

def Format.width (f : Format) : Nat := f.fractionBits + 9

/-- A format-indexed bit pattern, including every NaN payload and signed zero. -/
structure Value (f : Format) where
  bits : BitVec f.width
  deriving Repr, BEq, DecidableEq

def pow2 (e : Int) : Rat :=
  if e < 0 then mkRat 1 (2 ^ e.natAbs) else Rat.ofInt (2 ^ e.toNat)

/-- The sign is kept separately so that decoding does not erase negative zero.
`magnitude` is nonnegative for values produced by `Value.decode`. -/
inductive Decoded where
  | finite (negative : Bool) (magnitude : Rat)
  | infinity (negative : Bool)
  | nan (negative : Bool) (signaling : Bool) (payload : Nat)
  deriving Repr, DecidableEq

namespace Value

def ofBits (f : Format) (n : Nat) : Value f := ⟨BitVec.ofNat f.width n⟩

def raw {f : Format} (x : Value f) : Nat := x.bits.toNat

def sign {f : Format} (x : Value f) : Bool :=
  x.raw / 2 ^ (f.fractionBits + 8) != 0

def exponent {f : Format} (x : Value f) : Nat :=
  (x.raw / 2 ^ f.fractionBits) % 256

def fraction {f : Format} (x : Value f) : Nat := x.raw % 2 ^ f.fractionBits

private def pack (f : Format) (negative : Bool) (exp frac : Nat) : Value f :=
  ofBits f ((if negative then 2 ^ (f.fractionBits + 8) else 0) +
    exp * 2 ^ f.fractionBits + frac)

def zero (f : Format) (negative : Bool := false) : Value f := pack f negative 0 0

def infinity (f : Format) (negative : Bool := false) : Value f :=
  pack f negative 255 0

/-- Canonical positive quiet NaN used by the software arithmetic profile. -/
def qNaN (f : Format) : Value f := pack f false 255 (2 ^ (f.fractionBits - 1))

def maxFinite (f : Format) (negative : Bool := false) : Value f :=
  pack f negative 254 (2 ^ f.fractionBits - 1)

def isNaN {f : Format} (x : Value f) : Bool := x.exponent == 255 && x.fraction != 0
def isInf {f : Format} (x : Value f) : Bool := x.exponent == 255 && x.fraction == 0
def isZero {f : Format} (x : Value f) : Bool := x.exponent == 0 && x.fraction == 0
def isSubnormal {f : Format} (x : Value f) : Bool :=
  x.exponent == 0 && x.fraction != 0

def decode {f : Format} (x : Value f) : Decoded :=
  let exp := x.exponent
  let frac := x.fraction
  if exp == 255 then
    if frac == 0 then .infinity x.sign
    else .nan x.sign (frac < 2 ^ (f.fractionBits - 1)) frac
  else
    let significand := if exp == 0 then frac else 2 ^ f.fractionBits + frac
    let scale : Int := (if exp == 0 then -126 else (exp : Int) - 127) - f.fractionBits
    .finite x.sign (Rat.ofInt significand * pow2 scale)

/-- Mathematical finite projection. NaN/Inf have no rational projection. -/
def toRat? {f : Format} (x : Value f) : Option Rat :=
  match x.decode with
  | .finite negative magnitude => some (if negative then -magnitude else magnitude)
  | _ => none

/-- Sign-bit operation; preserves NaN payloads and does not perform arithmetic. -/
def neg {f : Format} (x : Value f) : Value f :=
  ofBits f ((if x.sign then 0 else 2 ^ (f.fractionBits + 8)) +
    x.raw % 2 ^ (f.fractionBits + 8))

def abs {f : Format} (x : Value f) : Value f :=
  ofBits f (x.raw % 2 ^ (f.fractionBits + 8))

end Value

inductive Rounding where
  | nearestEven | towardZero | towardPositive | towardNegative
  deriving Repr, BEq, DecidableEq

/-- Software value profile. Arithmetic canonicalizes NaNs and does not expose
exception flags. Input flushing and result flushing are separate choices;
neither setting asserts conformance with a GPU instruction. -/
structure Config where
  rounding : Rounding
  flushInputs : Bool
  flushResults : Bool
  deriving Repr, BEq, DecidableEq

def Config.ieeeValue : Config := ⟨.nearestEven, false, false⟩

private def roundMagnitude (mode : Rounding) (negative : Bool) (q : Rat) : Nat :=
  let n := q.num.natAbs / q.den
  let rem := q.num.natAbs % q.den
  let up := match mode with
    | .nearestEven => 2 * rem > q.den || (2 * rem == q.den && n % 2 == 1)
    | .towardZero => false
    | .towardPositive => !negative && rem != 0
    | .towardNegative => negative && rem != 0
  if up then n + 1 else n

private def overflow (f : Format) (mode : Rounding) (negative : Bool) : Value f :=
  let toInfinity := match mode with
    | .nearestEven => true
    | .towardZero => false
    | .towardPositive => !negative
    | .towardNegative => negative
  if toInfinity then Value.infinity f negative else Value.maxFinite f negative

/-- Round an exact rational directly to the target format, including subnormals
and overflow. `negativeZero` is consulted only for an exactly zero argument.
No host float, intermediate fp32, or real-number oracle is used here. -/
def round (f : Format) (cfg : Config) (q : Rat) (negativeZero : Bool := false) : Value f :=
  if q == 0 then Value.zero f negativeZero
  else
    let negative := q < 0
    let magnitude := q.abs
    let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
    let exponent := if magnitude < pow2 estimate then estimate - 1 else estimate
    let step : Int := max exponent (-126) - f.fractionBits
    let significand := roundMagnitude cfg.rounding negative (magnitude / pow2 step)
    let carry := significand >= 2 ^ (f.fractionBits + 1)
    let significand := if carry then significand / 2 else significand
    let exponent := step + f.fractionBits + (if carry then 1 else 0)
    if exponent > 127 then overflow f cfg.rounding negative
    else if significand < 2 ^ f.fractionBits then
      if cfg.flushResults then Value.zero f negative
      else Value.ofBits f ((if negative then 2 ^ (f.fractionBits + 8) else 0) + significand)
    else
      Value.ofBits f ((if negative then 2 ^ (f.fractionBits + 8) else 0) +
        (exponent + 127).toNat * 2 ^ f.fractionBits + significand - 2 ^ f.fractionBits)

/-- Apply the explicitly selected input-subnormal policy before arithmetic. -/
def prepare {f : Format} (cfg : Config) (x : Value f) : Value f :=
  if cfg.flushInputs && x.isSubnormal then Value.zero f x.sign else x

def cast {src : Format} (dst : Format) (cfg : Config) (x : Value src) : Value dst :=
  match (prepare cfg x).decode with
  | .nan .. => Value.qNaN dst
  | .infinity negative => Value.infinity dst negative
  | .finite negative magnitude => round dst cfg (if negative then -magnitude else magnitude) negative

attribute [spec_primitive] round cast Value.neg Value.abs

end VeriTile.Triton.FP

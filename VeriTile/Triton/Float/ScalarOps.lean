/-
Executable scalar operations for the concrete bf16/fp32 software value profile.
Each operation rounds once at its result; FMA rounds the exact product-plus-add.
Operands must have the operation's format; mixed precision requires explicit
casts. This is not yet a Triton lowering or a hardware conformance claim.
-/
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Triton.FP

private def signed (negative : Bool) (magnitude : Rat) : Rat :=
  if negative then -magnitude else magnitude

private def finiteAdd (f : Format) (cfg : Config)
    (sx : Bool) (x : Rat) (sy : Bool) (y : Rat) : Value f :=
  let negativeZero :=
    if x == 0 && y == 0 && sx == sy then sx else cfg.rounding == .towardNegative
  round f cfg (signed sx x + signed sy y) negativeZero

def add {f : Format} (cfg : Config) (x y : Value f) : Value f :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. => Value.qNaN f
  | .infinity sx, .infinity sy => if sx == sy then Value.infinity f sx else Value.qNaN f
  | .infinity sx, _ => Value.infinity f sx
  | _, .infinity sy => Value.infinity f sy
  | .finite sx x, .finite sy y => finiteAdd f cfg sx x sy y

def sub {f : Format} (cfg : Config) (x y : Value f) : Value f := add cfg x y.neg

def mul {f : Format} (cfg : Config) (x y : Value f) : Value f :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. => Value.qNaN f
  | .infinity sx, .infinity sy => Value.infinity f (sx != sy)
  | .infinity sx, .finite sy y =>
    if y == 0 then Value.qNaN f else Value.infinity f (sx != sy)
  | .finite sx x, .infinity sy =>
    if x == 0 then Value.qNaN f else Value.infinity f (sx != sy)
  | .finite sx x, .finite sy y =>
    round f cfg (signed (sx != sy) (x * y)) (sx != sy)

def div {f : Format} (cfg : Config) (x y : Value f) : Value f :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. | .infinity _, .infinity _ => Value.qNaN f
  | .infinity sx, .finite sy _ => Value.infinity f (sx != sy)
  | .finite sx _, .infinity sy => Value.zero f (sx != sy)
  | .finite sx x, .finite sy y =>
    if y == 0 then
      if x == 0 then Value.qNaN f else Value.infinity f (sx != sy)
    else round f cfg (signed (sx != sy) (x / y)) (sx != sy)

def fma {f : Format} (cfg : Config) (x y z : Value f) : Value f :=
  let x := prepare cfg x
  let y := prepare cfg y
  let z := prepare cfg z
  if x.isNaN || y.isNaN || z.isNaN then Value.qNaN f
  else if (x.isInf && y.isZero) || (x.isZero && y.isInf) then Value.qNaN f
  else if x.isInf || y.isInf then
    let negative := x.sign != y.sign
    if z.isInf && z.sign != negative then Value.qNaN f else Value.infinity f negative
  else
    match x.decode, y.decode, z.decode with
    | .finite sx x, .finite sy y, .finite sz z => finiteAdd f cfg (sx != sy) (x * y) sz z
    | _, _, .infinity sz => Value.infinity f sz
    | _, _, _ => Value.qNaN f

/-- Numeric equality, distinct from `Value`'s bit-pattern equality. -/
def eq {f : Format} (cfg : Config) (x y : Value f) : Bool :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .finite sx x, .finite sy y => signed sx x == signed sy y
  | .infinity sx, .infinity sy => sx == sy
  | _, _ => false

/-- Ordered less-than; NaNs compare false. -/
def lt {f : Format} (cfg : Config) (x y : Value f) : Bool :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. => false
  | .infinity sx, .infinity sy => sx && !sy
  | .infinity sx, .finite .. => sx
  | .finite .., .infinity sy => !sy
  | .finite sx x, .finite sy y => signed sx x < signed sy y

attribute [spec_primitive] add sub mul div fma eq lt

end VeriTile.Triton.FP

/- Batch interface used by test_fp_scalar.py; not a Triton execution backend. -/
import VeriTile.Triton.Float.ScalarOps

open VeriTile.Triton.FP

private def parseFormat : String → Except String Format
  | "bf16" => .ok .bf16
  | "fp32" => .ok .fp32
  | _ => .error "unknown format"

private def parseMode : String → Except String Rounding
  | "rne" => .ok .nearestEven
  | "rtz" => .ok .towardZero
  | "rup" => .ok .towardPositive
  | "rdn" => .ok .towardNegative
  | _ => .error "unknown rounding mode"

private def parseBool : String → Except String Bool
  | "0" => .ok false
  | "1" => .ok true
  | _ => .error "expected 0 or 1"

private def parseBits (f : Format) (s : String) : Except String (Value f) := do
  let some n := s.toNat? | throw "expected unsigned bits"
  if n >= 2 ^ f.width then throw "bit pattern exceeds format width"
  return Value.ofBits f n

private def runLine (line : String) : Except String Nat := do
  let [fmt, mode, flushIn, flushOut, op, a, b, c] := line.splitOn " "
    | throw "expected: format mode flushIn flushOut op a b c"
  let f ← parseFormat fmt
  let cfg : Config := ⟨← parseMode mode, ← parseBool flushIn, ← parseBool flushOut⟩
  if op == "round" then
    let some num := a.toInt? | throw "invalid numerator"
    let some den := b.toNat? | throw "invalid denominator"
    if den == 0 then throw "zero denominator"
    return (round f cfg (mkRat num den) (← parseBool c)).raw
  let x ← parseBits f a
  if op == "to_bf16" then return (cast .bf16 cfg x).raw
  if op == "to_fp32" then return (cast .fp32 cfg x).raw
  if op == "neg" then return x.neg.raw
  if op == "abs" then return x.abs.raw
  let y ← parseBits f b
  match op with
  | "add" => return (add cfg x y).raw
  | "sub" => return (sub cfg x y).raw
  | "mul" => return (mul cfg x y).raw
  | "div" => return (div cfg x y).raw
  | "eq" => return if eq cfg x y then 1 else 0
  | "lt" => return if lt cfg x y then 1 else 0
  | "fma" => return (fma cfg x y (← parseBits f c)).raw
  | _ => throw "unknown operation"

def main : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  repeat
    let line ← input.getLine
    if line.isEmpty then break
    match runLine line.trimAscii.toString with
    | .ok n => output.putStrLn (toString n)
    | .error e => throw (IO.userError e)

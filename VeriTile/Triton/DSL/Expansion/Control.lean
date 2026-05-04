/-
VeriTile.Triton.DSL.Expansion.Control

Control/range-surface expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

partial def expandProgramId (e : TSyntax `tritonExpr) : MacroM EOut := do
  let axisTerm : TSyntax `term ← match e with
    | `(tritonExpr| $($t:term)) => `(($t : Nat))
    | `(tritonExpr| $n:num)     => `(($n : Nat))
    | _ => Macro.throwError
            "tl.program_id(axis): axis must be a numeric literal or $(N)"
  pure ⟨← `(Op.programId $axisTerm), .nat, SInfo.scalar, none⟩

partial def expandArange (e : TSyntax `tritonExpr) : MacroM EOut := do
  match e with
  | `(tritonExpr| $($t:term)) => pure ⟨← `(Op.arange $t), .nat, SInfo.vec t, none⟩
  | `(tritonExpr| $n:num)     => pure ⟨← `(Op.arange $n), .nat, SInfo.vec (← `(($n : Nat))), none⟩
  | _ => Macro.throwError
          "tl.arange(...) expects a Lean Nat: either a numeric literal or $(N)"

partial def expandArangeRange (s e : TSyntax `tritonExpr) : MacroM EOut := do
  let sTerm : TSyntax `term ← match s with
    | `(tritonExpr| $($t:term)) => `(($t : Nat))
    | `(tritonExpr| $n:num)     => `(($n : Nat))
    | _ => Macro.throwError
            "tl.arange(start, end): start must be a numeric literal or $(N)"
  let eTerm : TSyntax `term ← match e with
    | `(tritonExpr| $($t:term)) => `(($t : Nat))
    | `(tritonExpr| $n:num)     => `(($n : Nat))
    | _ => Macro.throwError
            "tl.arange(start, end): end must be a numeric literal or $(N)"
  match s with
  | `(tritonExpr| $n:num) =>
      if n.getNat = 0 then
        pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none⟩
      else
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
          .nat, SInfo.vec (← `($eTerm - $sTerm)), none⟩
  | _ =>
      pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
            (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
        .nat, SInfo.vec (← `($eTerm - $sTerm)), none⟩

end VeriTile.Triton.DSL

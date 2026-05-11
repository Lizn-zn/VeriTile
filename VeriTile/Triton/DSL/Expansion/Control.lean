/-
VeriTile.Triton.DSL.Expansion.Control

Control/range-surface expansion helpers.
-/

import VeriTile.Triton.DSL.Expansion.Common

open Lean

namespace VeriTile.Triton.DSL

partial def expandProgramId (e : TSyntax `tritonExpr) : MacroM EOut := do
  let axisTerm ← natDimTerm "tl.program_id(axis)" e
  pure ⟨← `(Op.programId $axisTerm), .nat, SInfo.scalar, none, none⟩

partial def expandArange (e : TSyntax `tritonExpr) : MacroM EOut := do
  let eTerm ← natDimTerm "tl.arange(...)" e
  pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none, none⟩

partial def expandArangeRange (s e : TSyntax `tritonExpr) : MacroM EOut := do
  let sTerm ← natDimTerm "tl.arange(start, end) start" s
  let eTerm ← natDimTerm "tl.arange(start, end) end" e
  match s with
  | `(tritonExpr| $n:num) =>
      if n.getNat = 0 then
        pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none, none⟩
      else
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
          .nat, SInfo.vec (← `($eTerm - $sTerm)), none, none⟩
  | _ =>
      pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
            (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
        .nat, SInfo.vec (← `($eTerm - $sTerm)), none, none⟩

end VeriTile.Triton.DSL

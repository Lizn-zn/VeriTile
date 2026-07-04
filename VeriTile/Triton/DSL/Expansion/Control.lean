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

partial def expandNumPrograms (e : TSyntax `tritonExpr) : MacroM EOut := do
  let axisTerm ← natDimTerm "tl.num_programs(axis)" e
  pure ⟨← `(Op.numPrograms $axisTerm), .nat, SInfo.scalar, none, none⟩

partial def expandArange (e : TSyntax `tritonExpr) : MacroM EOut := do
  let eTerm ← natDimTerm "tl.arange(...)" e
  pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none, none⟩

private def sameRaw (a b : Syntax) : Bool :=
  toString a == toString b

partial def expandArangeRange (s e : TSyntax `tritonExpr) : MacroM EOut := do
  let sTerm ← natDimTerm "tl.arange(start, end) start" s
  let eTerm ← natDimTerm "tl.arange(start, end) end" e
  match s, e with
  | `(tritonExpr| $($base:term) // $den:tritonExpr), `(tritonExpr| $($endBase:term)) =>
      let denIsTwo :=
        toString den.raw ==
          "(VeriTile.Triton.DSL.«tritonExpr$(_)» \"$(\" (num \"2\") \")\")" ||
        toString den.raw == "(num \"2\")"
      if sameRaw base.raw endBase.raw && denIsTwo then
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              (Op.constNat $sTerm) (Op.arange $sTerm)),
          .nat, SInfo.vec sTerm, none, none⟩
      else
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
          .nat, SInfo.vec (← `($eTerm - $sTerm)), none, none⟩
  | `(tritonExpr| $n:num), _ =>
      if n.getNat = 0 then
        pure ⟨← `(Op.arange $eTerm), .nat, SInfo.vec eTerm, none, none⟩
      else
        pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
              (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
          .nat, SInfo.vec (← `($eTerm - $sTerm)), none, none⟩
  | _, _ =>
      pure ⟨← `(Op.add NumericDType.nat Broadcast.scalarL
            (Op.constNat $sTerm) (Op.arange ($eTerm - $sTerm))),
        .nat, SInfo.vec (← `($eTerm - $sTerm)), none, none⟩

end VeriTile.Triton.DSL

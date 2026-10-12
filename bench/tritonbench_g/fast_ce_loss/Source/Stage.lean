import VeriTile.Triton.DSL
import VeriTile.Triton.Source
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.TypedSource
open Lean VeriTile.Triton
/-- Triton specializes constexpr branches before assigning register types.
The full template below stays visible; all four flag combinations are exposed. -/
private partial def stageFlags (cap scale : Bool) (ss : Array (TSyntax `tritonStmt)) :
    MacroM (Array (TSyntax `tritonStmt)) := do
  let mut out := #[]
  for st in ss do
    match st with
    | `(tritonStmt| if $flag:ident { $body:tritonStmt* }) =>
        if flag.getId == `DO_SOFTCAPPING || flag.getId == `DO_LOGIT_SCALING then
          let enabled := if flag.getId == `DO_SOFTCAPPING then cap else scale
          if enabled then out := out ++ (← stageFlags cap scale body)
        else
          let body ← stageFlags cap scale body
          let cond ← `(tritonExpr| $flag:ident)
          out := out.push (← `(tritonStmt| if $cond { $body:tritonStmt* }))
    | `(tritonStmt| if $cond:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
        let yes ← stageFlags cap scale yes
        let no ← stageFlags cap scale no
        out := out.push (← `(tritonStmt| if $cond { $yes:tritonStmt* } else { $no:tritonStmt* }))
    | `(tritonStmt| if $cond:tritonExpr { $body:tritonStmt* }) =>
        let body ← stageFlags cap scale body
        out := out.push (← `(tritonStmt| if $cond { $body:tritonStmt* }))
    | _ => out := out.push st
  return out

syntax "fastCeStage%" "[" ident "," ident "]" "triton" "{" tritonStmt* "}" : term
macro_rules
  | `(fastCeStage%[$cap:ident,$scale:ident] triton { $ss:tritonStmt* }) => do
      unless cap.getId == `cap || cap.getId == `noCap do Macro.throwError "expected cap or noCap"
      unless scale.getId == `scale || scale.getId == `noScale do Macro.throwError "expected scale or noScale"
      let staged ← stageFlags (cap.getId == `cap) (scale.getId == `scale) ss
      `(triton { $staged:tritonStmt* })


end VeriTile.Bench.TritonBenchG.FastCeLoss.TypedSource

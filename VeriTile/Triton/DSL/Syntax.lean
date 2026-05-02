/-
VeriTile.Triton.DSL.Syntax

Surface syntax declarations for the Lean-embedded Triton subset.
-/

namespace VeriTile.Triton.DSL

/-! ## Syntax declarations -/

declare_syntax_cat tritonExpr
declare_syntax_cat tritonStmt
declare_syntax_cat tritonKwarg
declare_syntax_cat tritonReduceKwarg
declare_syntax_cat tritonDType

-- Expressions
syntax num : tritonExpr
-- `:max` so trailing parsers (e.g. the slicer postfix `e[:, None]`) chain
-- on bare register identifiers without the user wrapping them in parens.
syntax:max ident : tritonExpr
syntax "$(" term ")" : tritonExpr
syntax "$ℝ(" term ")" : tritonExpr
syntax "(" tritonExpr ")" : tritonExpr
syntax "tl.program_id(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.exp(" tritonExpr ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.sigmoid(" tritonExpr ")" : tritonExpr
syntax "tl.sqrt(" tritonExpr ")" : tritonExpr
syntax "tl.tanh(" tritonExpr ")" : tritonExpr
syntax "tl.logical_and(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
-- Element-wise select. All three operands must broadcast to a common
-- shape; the macro lifts scalars via `Op.broadcast`.
syntax "tl.where(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
syntax "tl.cast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax "-inf" : tritonExpr

syntax "tl.float64" : tritonDType
syntax "tl.float32" : tritonDType
syntax "tl.float16" : tritonDType
syntax "tl.bfloat16" : tritonDType

-- Block-level matrix multiply.
syntax "tl.dot(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.dot(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr

-- kwarg: `name = expr`. Used for `mask=` / `other=` in tl.load / tl.store.
syntax ident " = " tritonExpr : tritonKwarg

syntax "axis" "=" num : tritonReduceKwarg
syntax "keep_dims" "=" "false" : tritonReduceKwarg
syntax "keep_dims" "=" "true" : tritonReduceKwarg
syntax ident "=" tritonExpr : tritonReduceKwarg

syntax "tl.sum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr

syntax "tl.load(" tritonExpr ("," tritonKwarg)* ")" : tritonExpr

syntax:max tritonExpr:max noWs "[" ":" "," "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," ":" "]" : tritonExpr

syntax "tl.trans(" tritonExpr ")" : tritonExpr

syntax "tl.full(" "[" tritonExpr,* "]" ", " tritonExpr ")" : tritonExpr
syntax "tl.zeros(" "[" tritonExpr,* "]" ")" : tritonExpr

syntax:50 tritonExpr:51 " < "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " <= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " == " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " > "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " >= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " != " tritonExpr:51 : tritonExpr

syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ("," tritonKwarg)* ")" : tritonStmt
syntax "tl.for " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.for " ident " in " num " { " tritonStmt* " }" : tritonStmt
syntax "tl.if " tritonExpr " { " tritonStmt* " }" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

end VeriTile.Triton.DSL

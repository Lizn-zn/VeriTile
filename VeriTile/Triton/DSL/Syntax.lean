/-
VeriTile.Triton.DSL.Syntax

Surface syntax declarations for the Lean-embedded Triton subset.
-/

namespace VeriTile.Triton.DSL

/-! ## Syntax declarations -/

declare_syntax_cat tritonExpr
declare_syntax_cat tritonStmt
declare_syntax_cat tritonKwarg
declare_syntax_cat tritonMemKwarg
declare_syntax_cat tritonReduceKwarg
declare_syntax_cat tritonScanOp
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
syntax "tl.exp2(" tritonExpr ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.log2(" tritonExpr ")" : tritonExpr
syntax "tl.sigmoid(" tritonExpr ")" : tritonExpr
syntax "tl.sqrt(" tritonExpr ")" : tritonExpr
syntax "tl.tanh(" tritonExpr ")" : tritonExpr
syntax "tl.sin(" tritonExpr ")" : tritonExpr
syntax "tl.cos(" tritonExpr ")" : tritonExpr
syntax "tl.tan(" tritonExpr ")" : tritonExpr
syntax "tl.atan(" tritonExpr ")" : tritonExpr
syntax "tl.cosh(" tritonExpr ")" : tritonExpr
syntax "tl.sinh(" tritonExpr ")" : tritonExpr
syntax "tl.abs(" tritonExpr ")" : tritonExpr
syntax "tl.logical_and(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.logical_or(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.logical_not(" tritonExpr ")" : tritonExpr
syntax "tl.cdiv(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.maximum(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.minimum(" tritonExpr ", " tritonExpr ")" : tritonExpr
-- Element-wise select. All three operands must broadcast to a common
-- shape; the macro lifts scalars via `Op.broadcast`.
syntax "tl.where(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
syntax "tl.cast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax "tl.bitcast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax:max (name := tritonMethodCast) tritonExpr:max noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax "-inf" : tritonExpr

syntax "tl.float64" : tritonDType
syntax "tl.float32" : tritonDType
syntax "tl.float16" : tritonDType
syntax "tl.bfloat16" : tritonDType
syntax "tl.int8" : tritonDType
syntax "tl.int16" : tritonDType
syntax "tl.int32" : tritonDType
syntax "tl.int64" : tritonDType
syntax "tl.uint8" : tritonDType
syntax "tl.uint16" : tritonDType
syntax "tl.uint32" : tritonDType
syntax "tl.uint64" : tritonDType

-- Block-level matrix multiply.
syntax "tl.dot(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.dot(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr

-- kwarg: `name = expr`. Used for `mask=` / `other=` in tl.load / tl.store.
syntax ident " = " tritonExpr : tritonKwarg
syntax ident " = " tritonExpr : tritonMemKwarg
syntax ident "=" tritonDType : tritonMemKwarg
syntax "boundary_check=" term : tritonMemKwarg
syntax "padding_option=\"zero\"" : tritonMemKwarg

syntax "axis" "=" num : tritonReduceKwarg
syntax "keep_dims" "=" "false" : tritonReduceKwarg
syntax "keep_dims" "=" "true" : tritonReduceKwarg
syntax ident "=" tritonExpr : tritonReduceKwarg

syntax "tl.sum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax ident : tritonScanOp
syntax "tl.cumsum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.cumprod(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.associative_scan(" tritonExpr ", " tritonScanOp ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.argmax(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.argmin(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.sort(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr

syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax "tl.make_block_ptr(" tritonExpr ", " ident "=" tritonExpr ", " ident "=" "[" tritonExpr,*
  "]" ", " ident "=" "[" tritonExpr,* "]" ", " ident "=" "[" tritonExpr,* "]"
  ", " ident "=" "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.advance(" tritonExpr ", " "[" tritonExpr,* "]" ")" : tritonExpr

syntax:max tritonExpr:max noWs "[" ":" "," "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," ":" "]" : tritonExpr
syntax "tl.expand_dims(" tritonExpr ", " tritonReduceKwarg ")" : tritonExpr
syntax "tl.expand_dims(" tritonExpr ", " num ")" : tritonExpr

syntax "tl.trans(" tritonExpr ")" : tritonExpr
syntax "tl.permute(" tritonExpr ", " "[" num,* "]" ")" : tritonExpr
syntax "tl.reshape(" tritonExpr ", " "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.view(" tritonExpr ", " "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.ravel(" tritonExpr ")" : tritonExpr
syntax "tl.flip(" tritonExpr ", " num ")" : tritonExpr
syntax "tl.flip(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.join(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.split(" tritonExpr ", " num ")" : tritonExpr

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
syntax:55 tritonExpr:55 " & " tritonExpr:56 : tritonExpr
syntax:55 tritonExpr:55 " ^ " tritonExpr:56 : tritonExpr
syntax:54 tritonExpr:54 " | " tritonExpr:55 : tritonExpr
syntax:75 "~" tritonExpr:76 : tritonExpr
syntax:65 tritonExpr:65 " << " tritonExpr:66 : tritonExpr
syntax:65 tritonExpr:65 " >> " tritonExpr:66 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " // " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " % " tritonExpr:71 : tritonExpr
syntax "tl.atomic_xchg(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax "tl.atomic_cas(" tritonExpr ", " tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr

-- Statements
syntax ident " := " tritonExpr : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_add(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_max(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_min(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_and(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_or(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_xor(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_xchg(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_cas(" tritonExpr ", " tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.async_copy(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.async_wait()" : tritonStmt
syntax "tl.debug_barrier()" : tritonStmt
syntax "tl.for " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.for " ident " in " num " { " tritonStmt* " }" : tritonStmt
syntax "tl.static_range " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.static_range " ident " in " num " { " tritonStmt* " }" : tritonStmt
syntax "tl.if " tritonExpr " { " tritonStmt* " }" : tritonStmt
syntax "tl.if " tritonExpr " { " tritonStmt* " }" " else " " { " tritonStmt* " }" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

end VeriTile.Triton.DSL

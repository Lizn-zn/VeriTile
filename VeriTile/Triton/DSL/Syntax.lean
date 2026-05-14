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
syntax scientific : tritonExpr
syntax "true" : tritonExpr
syntax "false" : tritonExpr
syntax "None" : tritonExpr
-- `:max` so trailing parsers (e.g. the slicer postfix `e[:, None]`) chain
-- on bare register identifiers without the user wrapping them in parens.
syntax:max ident : tritonExpr
syntax "$(" term ")" : tritonExpr
syntax "(" tritonExpr ")" : tritonExpr
syntax "tl.program_id(" tritonExpr ")" : tritonExpr
syntax "tl.program_id(axis=" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ")" : tritonExpr
syntax "tl.arange(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.exp(" tritonExpr ")" : tritonExpr
syntax "tl.exp2(" tritonExpr ")" : tritonExpr
syntax "tl.math.exp2(" tritonExpr ")" : tritonExpr
syntax "tl.extra.cuda.libdevice.pow(" tritonExpr ", " num ")" : tritonExpr
syntax "tl.log(" tritonExpr ")" : tritonExpr
syntax "tl.log2(" tritonExpr ")" : tritonExpr
syntax "tl.sigmoid(" tritonExpr ")" : tritonExpr
syntax "silu(" tritonExpr ")" : tritonExpr
syntax "leaky_relu" "(" tritonExpr ")" : tritonExpr
syntax "tl.sqrt(" tritonExpr ")" : tritonExpr
syntax "tl.math.rsqrt(" tritonExpr ")" : tritonExpr
syntax "tl.rsqrt(" tritonExpr ")" : tritonExpr
syntax "tl.tanh(" tritonExpr ")" : tritonExpr
syntax "triton_tanh(" tritonExpr ")" : tritonExpr
syntax "tanh(" tritonExpr ")" : tritonExpr
syntax "tl.sin(" tritonExpr ")" : tritonExpr
syntax "tl.math.sin(" tritonExpr ")" : tritonExpr
syntax "tl.cos(" tritonExpr ")" : tritonExpr
syntax "tl.tan(" tritonExpr ")" : tritonExpr
syntax "tl.atan(" tritonExpr ")" : tritonExpr
syntax "tl.cosh(" tritonExpr ")" : tritonExpr
syntax "tl.sinh(" tritonExpr ")" : tritonExpr
syntax "tl.erf(" tritonExpr ")" : tritonExpr
syntax "tl.extra.cuda.libdevice.erf(" tritonExpr ")" : tritonExpr
syntax "tl.abs(" tritonExpr ")" : tritonExpr
syntax "tl.logical_and(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.logical_or(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.logical_not(" tritonExpr ")" : tritonExpr
syntax "tl.cdiv(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "_div_up(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "min(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.maximum(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.minimum(" tritonExpr ", " tritonExpr ")" : tritonExpr
-- Element-wise select. All three operands must broadcast to a common
-- shape; the macro lifts scalars via `Op.broadcast`.
syntax "tl.where(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "(" tritonExpr " if " tritonExpr " else " tritonExpr ")" : tritonExpr
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
syntax "tl.cast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax "tl.bitcast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax "tl.extra.cuda.libdevice.llrint(" tritonExpr ")" : tritonExpr
syntax:max (name := tritonMethodCastDTypeIdent) tritonExpr:max noWs "." "to" "(" ident "." ident ")" : tritonExpr
syntax:max (name := tritonIdentMethodCastDTypeIdent) ident noWs "." "to" "(" ident "." ident ")" : tritonExpr
syntax (name := tritonIdentMethodCastDTypeIdentSpaced) ident "." "to" "(" ident "." ident ")" : tritonExpr
syntax:max (name := tritonMethodCast) tritonExpr:max noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax:max (name := tritonIdentMethodCast) ident noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax (name := tritonIdentMethodCastSpaced) ident "." "to" "(" tritonDType ")" : tritonExpr
syntax:max (name := tritonDottedIdentMethodCast) ident noWs "(" tritonDType ")" : tritonExpr
syntax:max (name := tritonMethodCastElementTy) tritonExpr:max noWs "." "to" "(" term ")" : tritonExpr
syntax:max (name := tritonMethodCastElementTyIdent) tritonExpr:max noWs "." "to" "(" ident "." ident "." ident ")" : tritonExpr
syntax:max (name := tritonIdentMethodCastElementTyIdent) ident noWs "." "to" "(" ident "." ident "." ident ")" : tritonExpr
syntax (name := tritonIdentMethodCastElementTyIdentSpaced) ident "." "to" "(" ident "." ident "." ident ")" : tritonExpr
syntax "tl.multiple_of(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "tl.max_contiguous" "(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax "-inf" : tritonExpr
syntax "-" "float" "(" term ")" : tritonExpr
syntax "float" "(" term ")" : tritonExpr

syntax "tl.float64" : tritonDType
syntax "tl.float32" : tritonDType
syntax "tl.float16" : tritonDType
syntax "tl.bfloat16" : tritonDType
syntax "tl.int1" : tritonDType
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
syntax "tl.dot(" tritonExpr ", " tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.dot(" tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonExpr

syntax ident "=" tritonDType : tritonMemKwarg
syntax ident " = " tritonDType : tritonMemKwarg
-- kwarg: `name = expr`. Used for `mask=` / `other=` in tl.load / tl.store.
syntax ident " = " tritonExpr : tritonKwarg
syntax ident " = " tritonExpr : tritonMemKwarg
syntax "boundary_check=" term : tritonMemKwarg
syntax "boundary_check=(" term ")" : tritonMemKwarg
syntax "boundary_check=([" num,* "]" ":" term ")" : tritonMemKwarg
syntax "boundary_check=(" num,* ")" : tritonMemKwarg
syntax "padding_option=\"zero\"" : tritonMemKwarg
syntax "eviction_policy=\"evict_last\"" : tritonMemKwarg

syntax "axis" "=" num : tritonReduceKwarg
syntax num : tritonReduceKwarg
syntax "keep_dims" "=" "false" : tritonReduceKwarg
syntax "keep_dims" "=" "true" : tritonReduceKwarg
syntax "return_indices=True" : tritonReduceKwarg
syntax "return_indices=False" : tritonReduceKwarg
syntax "out_dtype=" tritonDType : tritonReduceKwarg
syntax ident "=" tritonExpr : tritonReduceKwarg

syntax "tl.sum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
syntax ident : tritonScanOp
syntax "tl.cumsum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.cumprod(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.associative_scan(" tritonExpr ", " tritonScanOp ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.argmax(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.argmin(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.sort(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr

syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax "tl.load(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax "tl.load(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" ident ")" : tritonExpr
syntax "tl.load(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" ident ")" : tritonExpr
syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" ident "." ident ")" : tritonExpr
syntax "tl.load(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" noWs "." "to" "(" ident "." ident ")" : tritonExpr
syntax "tl.make_block_ptr(" tritonExpr ", " ident "=" tritonExpr ", " ident "=" "[" tritonExpr,*
  "]" ", " ident "=" "[" tritonExpr,* "]" ", " ident "=" "[" tritonExpr,* "]"
  ", " ident "=" "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.make_block_ptr(" ident "=" tritonExpr ", " ident "=" "(" tritonExpr,*
  ")" ", " ident "=" "(" tritonExpr,* ")" ", " ident "=" "(" tritonExpr,* ")"
  ", " ident "=" "(" tritonExpr,* ")" ", " ident "=" "(" num,* ")" ")" : tritonExpr
syntax "tl.make_block_ptr(" ident "=" tritonExpr ", " ident "=" "(" tritonExpr,*
  ")" ", " ident "=" "(" tritonExpr,* ")" ", " ident "=" "(" tritonExpr,* ")"
  ", " ident "=" "(" tritonExpr,* ")" ")" : tritonExpr
syntax "tl.advance(" tritonExpr ", " "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.advance(" tritonExpr ", " ident "=" "(" tritonExpr,* ")" ")" : tritonExpr

syntax:max tritonExpr:max noWs "[" ":" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" ":" "," "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," ":" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" ":" "," "None" "," "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," ":" "," "None" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," "None" "," ":" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" ":" "," "None" "," ":" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" "None" "," ":" "," ":" "]" : tritonExpr
syntax:max tritonExpr:max noWs "[" ":" "," ":" "," "None" "]" : tritonExpr
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
syntax "tl.broadcast(" tritonExpr ", " tritonExpr ")" : tritonExpr

syntax "tl.full(" "[" tritonExpr,* "]" ", " tritonExpr ")" : tritonExpr
syntax "tl.full(" "[" tritonExpr,* "]" ", " tritonExpr ", " ident "=" tritonDType ")" : tritonExpr
syntax "tl.full(" "[" tritonExpr,* "]" ", " ident "=" tritonDType ", " ident "=" tritonExpr ")" : tritonExpr
syntax "tl.full(" "[" tritonExpr,* "]" ", " ident "=" tritonExpr ", " ident "=" tritonDType ")" : tritonExpr
syntax "tl.zeros(" "[" tritonExpr,* "]" ")" : tritonExpr
syntax "tl.zeros(" "[" tritonExpr,* "]" ", " tritonDType ")" : tritonExpr
syntax (priority := high) "tl.zeros(" "[" tritonExpr,* "]" ", " ident "=" term ")" : tritonExpr
syntax "tl.zeros(" "[" tritonExpr,* "]" ", " ident "=" tritonDType ")" : tritonExpr
syntax "tl.zeros_like(" tritonExpr ")" : tritonExpr

syntax:50 tritonExpr:51 " < "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " <= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " == " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " > "  tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " >= " tritonExpr:51 : tritonExpr
syntax:50 tritonExpr:51 " != " tritonExpr:51 : tritonExpr

syntax:60 tritonExpr:60 " + " tritonExpr:61 : tritonExpr
syntax:60 tritonExpr:60 " - " tritonExpr:61 : tritonExpr
syntax:55 tritonExpr:55 " & " tritonExpr:56 : tritonExpr
syntax:45 tritonExpr:46 " and " tritonExpr:46 : tritonExpr
syntax:55 tritonExpr:55 " ^ " tritonExpr:56 : tritonExpr
syntax:54 tritonExpr:54 " | " tritonExpr:55 : tritonExpr
syntax:75 "~" tritonExpr:76 : tritonExpr
syntax:75 "not " tritonExpr:76 : tritonExpr
syntax:75 "-" tritonExpr:76 : tritonExpr
syntax:65 tritonExpr:65 " << " tritonExpr:66 : tritonExpr
syntax:65 tritonExpr:65 " >> " tritonExpr:66 : tritonExpr
syntax:70 tritonExpr:70 " * " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " / " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " // " tritonExpr:71 : tritonExpr
syntax:70 tritonExpr:70 " % " tritonExpr:71 : tritonExpr
syntax "tl.atomic_xchg(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax "tl.atomic_cas(" tritonExpr ", " tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr

-- Statements

/-- In-body region dtype declaration. Lets the macro know the element
dtype of a kernel parameter region so `tl.load(R + offs)` /
`tl.store(R + offs, v)` can recover the dtype without an explicit
`dtype=` kwarg. The directive itself is stripped from the emitted
kernel body — it is metadata, not a runtime statement. Multiple
declarations may be combined with `,`:

  tl.region mid_index = tl.int64, out = tl.int64
-/
syntax "tl.region " ident " = " tritonDType ("," ident " = " tritonDType)* : tritonStmt

syntax ident ", " ident " = " "tl.broadcast(" tritonExpr ", " tritonExpr ")" : tritonStmt
syntax ident ", " ident " := " "tl.broadcast(" tritonExpr ", " tritonExpr ")" : tritonStmt
syntax ident ", " "_" " = " "tl.broadcast(" tritonExpr ", " tritonExpr ")" : tritonStmt
syntax ident ", " "_" " := " "tl.broadcast(" tritonExpr ", " tritonExpr ")" : tritonStmt
syntax ident ", " ident (", " ident)* " := "
  tritonExpr ", " tritonExpr (", " tritonExpr)* : tritonStmt
syntax ident ", " ident (", " ident)* " = "
  tritonExpr ", " tritonExpr (", " tritonExpr)* : tritonStmt
syntax ident " % " tritonExpr : tritonStmt
syntax ident " := " "tl.max(" tritonExpr ", " num ")" : tritonStmt
syntax ident " = " "tl.max(" tritonExpr ", " num ")" : tritonStmt
syntax ident " := " tritonExpr : tritonStmt
syntax ident " = " tritonExpr : tritonStmt
syntax ident " += " tritonExpr : tritonStmt
syntax ident " -= " tritonExpr : tritonStmt
syntax ident " *= " tritonExpr : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ")" : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ", " tritonMemKwarg ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_add(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_max(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_min(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_and(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_or(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_xor(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_xchg(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.atomic_cas(" tritonExpr ", " tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax ident ", " ident " = " "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonStmt
syntax ident ", " ident " := " "tl.max(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonStmt
syntax "tl.async_copy(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.async_wait()" : tritonStmt
syntax "tl.debug_barrier()" : tritonStmt
syntax "tl.static_print(" term ")" : tritonStmt
syntax "tl.for " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.for " ident " in " num " { " tritonStmt* " }" : tritonStmt
syntax "for " ident " in " "range(0, " "$(" term ")" ", " "$(" term ")" ")" " { " tritonStmt* " }" : tritonStmt
syntax "for " ident " in " "range(" tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax "for " ident " in " "range(" "$(" term ")" ", " "$(" term ")" ", " "$(" term ")" ")" " { " tritonStmt* " }" : tritonStmt
syntax "for " ident " in " "range(" tritonExpr ", " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax "for " ident " in " "range(" tritonExpr ", " tritonExpr ", " tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.static_range " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
syntax "tl.static_range " ident " in " num " { " tritonStmt* " }" : tritonStmt
syntax "tl.if " tritonExpr " { " tritonStmt* " }" : tritonStmt
syntax "tl.if " tritonExpr " { " tritonStmt* " }" " else " " { " tritonStmt* " }" : tritonStmt
-- Python-style alias: `if cond { body }` (and `if-else`) without the `tl.` prefix.
-- Within `triton { ... }` the parser is in `tritonStmt` mode, so `if` here is a
-- DSL token, distinct from Lean's term-level `if`.
syntax "if " tritonExpr " { " tritonStmt* " }" : tritonStmt
syntax "if " tritonExpr " { " tritonStmt* " }" " else " " { " tritonStmt* " }" : tritonStmt

-- Block (the user-facing entry point)
syntax (name := tritonBlock) "triton " "{" tritonStmt* "}" : term

end VeriTile.Triton.DSL

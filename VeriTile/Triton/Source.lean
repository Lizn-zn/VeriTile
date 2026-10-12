import VeriTile.Triton.Core.Ast

namespace VeriTile.Triton.Source

/-- Locate a top-level register assignment, counting repeated writes from zero.
The prefix includes all preceding statements. If no such assignment exists,
the entire list is returned; a rewrite proof must still match the actual site. -/
def prefixBefore (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :
    List ComputeStmt :=
  match code with
  | [] => []
  | st :: rest =>
    match st with
    | .assign _ _ out _ =>
      if out = name then
        match occurrence with
        | 0 => []
        | n + 1 => st :: prefixBefore rest name n
      else st :: prefixBefore rest name occurrence
    | _ => st :: prefixBefore rest name occurrence

/-- Locate a top-level store, counting stores from zero. Non-store statements
do not affect the occurrence index. The whole list is returned when absent;
a contextual proof must still match the selected prefix and unchanged memory
access. This also recognizes stores wrapped as algorithm statements. -/
def prefixBeforeStore (code : List ComputeStmt) (occurrence : Nat := 0) :
    List ComputeStmt :=
  match code with
  | [] => []
  | st :: rest =>
    match st with
    | .store .. | .alg (.store ..) =>
      match occurrence with
      | 0 => []
      | n + 1 => st :: prefixBeforeStore rest n
    | _ => st :: prefixBeforeStore rest occurrence

/-- The actual context and bounds of a named dynamic range. Locating a loop
does not assert any execution property or permit changing its bounds. -/
structure DynamicRange where
  before : List ComputeStmt
  start : Op .nat []
  stop : Op .nat []
  stride : Op .nat []
  body : List ComputeStmt
  after : List ComputeStmt

/-- Locate a top-level natural range by its index register. Static bounds are
represented by constant operations; the original statement is not rewritten.
Other statements
remain in the prefix, so common insertions need no numeric statement offset.
A missing loop is reported as `none`; occurrences count from zero. -/
def dynamicRange? (code : List ComputeStmt) (index : RegName)
    (occurrence : Nat := 0) : Option DynamicRange :=
  match code with
  | [] => none
  | st :: rest =>
    match st with
    | .forRangeDyn name start stop stride body =>
      if name = index then
        match occurrence with
        | 0 => some ⟨[], start, stop, stride, body, rest⟩
        | n + 1 => (dynamicRange? rest index n).map fun r =>
            { r with before := st :: r.before }
      else (dynamicRange? rest index occurrence).map fun r =>
        { r with before := st :: r.before }
    | .forRange name start stop stride body =>
      if name = index then
        match occurrence with
        | 0 => some ⟨[], .constNat start, .constNat stop, .constNat stride, body, rest⟩
        | n + 1 => (dynamicRange? rest index n).map fun r =>
            { r with before := st :: r.before }
      else (dynamicRange? rest index occurrence).map fun r =>
        { r with before := st :: r.before }
    | _ => (dynamicRange? rest index occurrence).map fun r =>
        { r with before := st :: r.before }

/-- A signed dynamic range. The stride is natural, as in the source AST;
start/stop and the bound index retain their signed integer type. -/
structure IntRange where
  before : List ComputeStmt
  start : Op .int []
  stop : Op .int []
  stride : Op .nat []
  body : List ComputeStmt
  after : List ComputeStmt

/-- Locate an actual signed range without clipping its bounds or replacing
it by a natural range. Occurrences of the same named signed loop count from
zero; all preceding statements remain in its execution prefix. -/
def intRange? (code : List ComputeStmt) (index : RegName)
    (occurrence : Nat := 0) : Option IntRange :=
  match code with
  | [] => none
  | st :: rest =>
    match st with
    | .forRangeInt name start stop stride body =>
      if name = index then
        match occurrence with
        | 0 => some ⟨[], start, stop, stride, body, rest⟩
        | n + 1 => (intRange? rest index n).map fun r =>
            { r with before := st :: r.before }
      else (intRange? rest index occurrence).map fun r =>
        { r with before := st :: r.before }
    | _ => (intRange? rest index occurrence).map fun r =>
        { r with before := st :: r.before }

/-- A conditional with its unchanged surrounding statements. -/
structure Conditional where
  before : List ComputeStmt
  condition : Op .bool []
  yes : List ComputeStmt
  no : List ComputeStmt
  after : List ComputeStmt

/-- Locate a top-level conditional by a property of its then arm. A one-arm
conditional has an empty `no` list. A missing site stays `none`; a rewrite still
has to match the actual source context and conditional constructor. -/
def conditional? (code : List ComputeStmt) (select : List ComputeStmt → Bool) :
    Option Conditional :=
  match code with
  | [] => none
  | st :: rest =>
    match st with
    | .ifThenElse condition yes no =>
      if select yes then some ⟨[], condition, yes, no, rest⟩
      else (conditional? rest select).map fun r => { r with before := st :: r.before }
    | .ifThen condition yes =>
      if select yes then some ⟨[], condition, yes, [], rest⟩
      else (conditional? rest select).map fun r => { r with before := st :: r.before }
    | _ => (conditional? rest select).map fun r => { r with before := st :: r.before }

end VeriTile.Triton.Source

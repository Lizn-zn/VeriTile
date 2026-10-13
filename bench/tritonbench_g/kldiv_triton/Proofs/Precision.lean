import bench.tritonbench_g.kldiv_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.Precision
open VeriTile.Triton TypedSource
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeInt _ _ _ _ body => body
  | _ => []

def selectedBranch (yes : Bool) : ComputeStmt → List ComputeStmt
  | .ifThenElse _ lhs rhs => if yes then lhs else rhs
  | _ => []

def targetStorage : MathStorage → Storage
  | .fp32 => .fp32
  | .fp64 => .fp64

def expectedLoad (storage : Storage) (name pointer : RegName) (B : Nat) : ComputeStmt :=
  let address : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] pointer)
    (.castIntToNat (.ref .int [B] "offsets"))
  let mask : Op .bool [B] := .ref .bool [B] "mask"
  match storage with
  | .fp32 => .assign .real [B] name (.compute (.load .fp32 (.ptr address)
      (.maskOther mask ((Op.const 0.0).broadcast [B]))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp64 (.load .real (.ptr address)
      (.maskOther mask ((Op.const 0.0).broadcast [B])))))
  | .fp16 => .assign .fp16 [B] name (.alg (.load .fp16 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))
  | .bf16 => .assign .bf16 [B] name (.alg (.load .bf16 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))
  | .f8e4 => .assign .f8e4 [B] name (.alg (.load .f8e4 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B]))))
  | .f8e5 => .assign .f8e5 [B] name (.alg (.load .f8e5 (.ptr address)
      (.maskOther mask ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B]))))

/-- Every load retains signed offsets, source storage and the explicit zero other value. -/
theorem forward_loads (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    let body := loopBody (forward pred target output v Y T O sy st so N B flag reduction).surfaceBody[5]!
    body[2]! = expectedLoad pred "y" "Y" B ∧
    body[3]! = expectedLoad (targetStorage target) "y_true" "T" B := by
  cases pred <;> cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

theorem backward_default_load (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardDefault target output v T O st so N B).surfaceBody[5]!)[2]! =
      expectedLoad target "target" "T" B := by
  cases target <;> cases output <;> cases v <;> rfl

theorem backward_log_load (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardLog target output v T O st so N B).surfaceBody[5]!)[2]! =
      expectedLoad (targetStorage target) "target" "T" B := by
  cases target <;> cases output <;> cases v <;> rfl

def isWide (pred : Storage) (target : MathStorage) : Bool :=
  pred == .fp64 || target == .fp64

def promotedOperand (storage : Storage) (wide : Bool) (name : RegName)
    (B : Nat) : Op .real [B] :=
  match storage with
  | .fp64 => .ref .real [B] name
  | .fp32 => if wide then .withPrecision .real .fp32 (.ref .real [B] name)
      else .ref .real [B] name
  | .fp16 => .castFloat .fp16 .real (if wide then
      .withPrecision .fp16 .fp32 (.ref .fp16 [B] name) else .ref .fp16 [B] name)
  | .bf16 => .castFloat .bf16 .real (if wide then
      .withPrecision .bf16 .fp32 (.ref .bf16 [B] name) else .ref .bf16 [B] name)
  | .f8e4 => .castFloat .f8e4 .real (if wide then
      .withPrecision .f8e4 .fp32 (.ref .f8e4 [B] name) else .ref .f8e4 [B] name)
  | .f8e5 => .castFloat .f8e5 .real (if wide then
      .withPrecision .f8e5 .fp32 (.ref .f8e5 [B] name) else .ref .f8e5 [B] name)

def expectedFormula (pred : Storage) (target : MathStorage) (flag : Bool) (B : Nat) : ComputeStmt :=
  let wide := isWide pred target
  let y := promotedOperand pred wide "y" B
  let t := promotedOperand (targetStorage target) wide "y_true" B
  let raw : Op .real [B] := .ref .real [B] "y_true"
  let intrinsic := if flag then Op.exp raw else Op.log raw
  let result := if target == .fp32 && wide then .withPrecision .real .fp32 intrinsic else intrinsic
  let expr := if flag then Op.mul .real (.consSame .nil) result (.sub .real (.consSame .nil) t y)
    else Op.mul .real (.consSame .nil) t (.sub .real (.consSame .nil) result y)
  .assign .real [B] "loss" (if wide then .compute (.alg .fp64 expr)
    else .compute (.alg .fp32 expr))

/-- Direct log/exp use target precision before promotion with the prediction tensor. -/
theorem forward_formula (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    let body := loopBody (forward pred target output v Y T O sy st so N B flag reduction).surfaceBody[5]!
    (selectedBranch Bool.true body[4]!)[0]! = expectedFormula pred target Bool.false B ∧
    (selectedBranch Bool.false body[4]!)[0]! = expectedFormula pred target Bool.true B := by
  cases pred <;> cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

/-- Both output branches are present in every source profile; reduction is not accumulated. -/
theorem forward_shape (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    let source := forward pred target output v Y T O sy st so N B flag reduction
    let body := loopBody source.surfaceBody[5]!
    source.surfaceBody.length = 6 ∧ body.length = 6 ∧
    (selectedBranch Bool.true body[5]!).length = 1 ∧
    (selectedBranch Bool.false body[5]!).length = 3 ∧
    (selectedBranch Bool.false body[5]!)[2]! =
      .assign .ptr [] "O" (.alg (.ptrAdd .nil (.ref .ptr [] "O") (.constNat 1))) := by
  cases pred <;> cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩


def outputMask (dtype : TileDType) (shape : TileShape) (mask : Option (Op .bool shape)) :
    MaskOpt dtype shape := match mask with
  | none => .none
  | some m => .mask m

def expectedStore (wide : Bool) (output : Storage) (shape : TileShape)
    (address : Op .ptr shape) (mask : Option (Op .bool shape)) : ComputeStmt :=
  let raw : Op .real shape := .ref .real shape "loss"
  let value : Op .real shape := if wide then .withPrecision .real .fp64 raw else raw
  match output with
  | .fp32 => .store .real shape (.ptr address)
      (.compute (.alg .fp32 (if wide then .castFloat .real .real value else raw)))
      (outputMask .real shape mask)
  | .fp64 => .store .real shape (.ptr address)
      (.compute (.alg .fp64 (if wide then raw else .castFloat .real .real
        (.withPrecision .real .fp32 raw)))) (outputMask .real shape mask)
  | .fp16 => .store .fp16 shape (.ptr address) (.alg (.castFloat .real .fp16 value))
      (outputMask .fp16 shape mask)
  | .bf16 => .store .bf16 shape (.ptr address) (.alg (.castFloat .real .bf16 value))
      (outputMask .bf16 shape mask)
  | .f8e4 => .store .f8e4 shape (.ptr address) (.alg (.castFloat .real .f8e4 value))
      (outputMask .f8e4 shape mask)
  | .f8e5 => .store .f8e5 shape (.ptr address) (.alg (.castFloat .real .f8e5 value))
      (outputMask .f8e5 shape mask)

/-- Exact destination conversion for both elementwise and reduced stores, plus the
actual block-sum precision. All 72 independent storage combinations are checked. -/
theorem forward_stores (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    let body := loopBody (forward pred target output v Y T O sy st so N B flag reduction).surfaceBody[5]!
    let wide := isWide pred target
    let sum : Op .real [] := .reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false
      (.ref .real [B] "loss")
    (selectedBranch Bool.true body[5]!)[0]! =
      expectedStore wide output [B]
        (.ptrAdd .scalarL (.ref .ptr [] "O") (.castIntToNat (.ref .int [B] "offsets")))
        (some (.ref .bool [B] "mask")) ∧
    (selectedBranch Bool.false body[5]!)[0]! =
      .assign .real [] "loss" (if wide then .compute (.alg .fp64 sum)
        else .compute (.alg .fp32 sum)) ∧
    (selectedBranch Bool.false body[5]!)[1]! =
      expectedStore wide output [] (.ref .ptr [] "O") none := by
  cases pred <;> cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

def expectedSameBackwardStore (target : Storage) (B : Nat) : ComputeStmt :=
  let address : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] "O")
    (.castIntToNat (.ref .int [B] "offsets"))
  match target with
  | .fp32 => .store .real [B] (.ptr address)
      (.compute (.alg .fp32 (.ref .real [B] "res"))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (.ptr address)
      (.compute (.alg .fp64 (.ref .real [B] "res"))) (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (.ptr address) (.alg (.ref .fp16 [B] "res"))
      (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (.ptr address) (.alg (.ref .bf16 [B] "res"))
      (.mask (.ref .bool [B] "mask"))
  | .f8e4 => .store .f8e4 [B] (.ptr address) (.alg (.ref .f8e4 [B] "res"))
      (.mask (.ref .bool [B] "mask"))
  | .f8e5 => .store .f8e5 [B] (.ptr address) (.alg (.ref .f8e5 [B] "res"))
      (.mask (.ref .bool [B] "mask"))

/-- The weak scalar -1 must not silently widen narrow tensor arithmetic or its store. -/
theorem backward_same_storage (target : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardDefault target target v T O st so N B).surfaceBody[5]!)[4]! =
      expectedSameBackwardStore target B := by
  cases target <;> cases v <;> rfl

def storageFloat : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

def backwardValue (target output : Storage) (B : Nat) :
    Op (storageFloat target).toTileDType [B] :=
  match target with
  | .fp32 => if output = .fp64 then .withPrecision .real .fp32 (.ref .real [B] "res")
      else .ref .real [B] "res"
  | .fp64 => if output = .fp64 then .ref .real [B] "res"
      else .withPrecision .real .fp64 (.ref .real [B] "res")
  | other => if output = .fp64 then
      .withPrecision (storageFloat other) .fp32 (.ref (storageFloat other).toTileDType [B] "res")
      else .ref (storageFloat other).toTileDType [B] "res"

def expectedBackwardStore (target output : Storage) (B : Nat) : ComputeStmt :=
  let address : MemAccess .real [B] := .ptr (.ptrAdd .scalarL (.ref .ptr [] "O")
    (.castIntToNat (.ref .int [B] "offsets")))
  let ptr : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] "O")
    (.castIntToNat (.ref .int [B] "offsets"))
  let mask : Op .bool [B] := .ref .bool [B] "mask"
  if target = output then expectedSameBackwardStore target B else
    match output with
    | .fp32 => .store .real [B] address (.compute (.alg .fp32
        (.castFloat (storageFloat target) .real (backwardValue target output B)))) (.mask mask)
    | .fp64 => .store .real [B] address (.compute (.alg .fp64
        (.castFloat (storageFloat target) .real (backwardValue target output B)))) (.mask mask)
    | .fp16 => .store .fp16 [B] (.ptr ptr) (.alg
        (.castFloat (storageFloat target) .fp16 (backwardValue target output B))) (.mask mask)
    | .bf16 => .store .bf16 [B] (.ptr ptr) (.alg
        (.castFloat (storageFloat target) .bf16 (backwardValue target output B))) (.mask mask)
    | .f8e4 => .store .f8e4 [B] (.ptr ptr) (.alg
        (.castFloat (storageFloat target) .f8e4 (backwardValue target output B))) (.mask mask)
    | .f8e5 => .store .f8e5 [B] (.ptr ptr) (.alg
        (.castFloat (storageFloat target) .f8e5 (backwardValue target output B))) (.mask mask)

/-- All independent default-backward output conversions; fp8 inputs are an abstract extension. -/
theorem backward_default_stores (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardDefault target output v T O st so N B).surfaceBody[5]!)[4]! =
      expectedBackwardStore target output B := by
  cases target <;> cases output <;> cases v <;> rfl

/-- All logarithmic-backward output conversions retain the intrinsic's original precision. -/
theorem backward_log_stores (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardLog target output v T O st so N B).surfaceBody[5]!)[4]! =
      expectedBackwardStore (targetStorage target) output B := by
  cases target <;> cases output <;> cases v <;> rfl

def expectedDefaultFormula (target : Storage) (B : Nat) : ComputeStmt :=
  match target with
  | .fp32 => .assign .real [B] "res" (.compute (.alg .fp32
      (.mul .real .scalarR (.ref .real [B] "target") (.const (-1)))))
  | .fp64 => .assign .real [B] "res" (.compute (.alg .fp64
      (.mul .real .scalarR (.ref .real [B] "target") (.const (-1)))))
  | .fp16 => .assign .fp16 [B] "res" (.alg
      (.mul .fp16 .scalarR (.ref .fp16 [B] "target") (.constFloat .fp16 (-1))))
  | .bf16 => .assign .bf16 [B] "res" (.alg
      (.mul .bf16 .scalarR (.ref .bf16 [B] "target") (.constFloat .bf16 (-1))))
  | .f8e4 => .assign .f8e4 [B] "res" (.alg
      (.mul .f8e4 .scalarR (.ref .f8e4 [B] "target") (.constFloat .f8e4 (-1))))
  | .f8e5 => .assign .f8e5 [B] "res" (.alg
      (.mul .f8e5 .scalarR (.ref .f8e5 [B] "target") (.constFloat .f8e5 (-1))))

/-- The constexpr `-1` is one literal; no extra runtime subtraction is introduced. -/
theorem backward_default_formula (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (loopBody (backwardDefault target output v T O st so N B).surfaceBody[5]!)[3]! =
      expectedDefaultFormula target B := by
  cases target <;> cases output <;> cases v <;> rfl

/-- Runtime negation of the exponential remains a subtraction from typed zero. -/
theorem backward_log_formula (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    let expr : Op .real [B] := .sub .real .scalarL (.const 0.0)
      (.exp (.ref .real [B] "target"))
    (loopBody (backwardLog target output v T O st so N B).surfaceBody[5]!)[3]! =
      .assign .real [B] "res" (match target with
        | .fp32 => .compute (.alg .fp32 expr)
        | .fp64 => .compute (.alg .fp64 expr)) := by
  cases target <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.Precision

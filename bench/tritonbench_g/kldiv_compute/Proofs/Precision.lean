import bench.tritonbench_g.kldiv_compute.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivCompute.Precision
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def expectedLoad (storage : Storage) (name : RegName) (R : RegionName) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .assign .real [B] name
      (.compute (.load .fp32 (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask"))))
  | .fp64 => .assign .real [B] name
      (.compute (.alg .fp64 (.load .real (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask")))))
  | .fp16 => .assign .fp16 [B] name
      (.alg (.load .fp16 (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask"))))
  | .bf16 => .assign .bf16 [B] name
      (.alg (.load .bf16 (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask"))))
  | .f8e4 => .assign .f8e4 [B] name
      (.alg (.load .f8e4 (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask"))))
  | .f8e5 => .assign .f8e5 [B] name
      (.alg (.load .f8e5 (.region R (.ref .nat [B] "offsets"))
        (.mask (.ref .bool [B] "mask"))))

theorem x_load (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody[4]? =
      some (expectedLoad x "x" X B) := by
  cases x <;> cases y <;> cases output <;> cases v <;> rfl

theorem y_load (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody[5]? =
      some (expectedLoad y "y" Y B) := by
  cases x <;> cases y <;> cases output <;> cases v <;> rfl

def isWide (x y : Storage) : Bool := x == .fp64 || y == .fp64

def expectedStore (wide : Bool) (output : Storage) (O : RegionName) (B : Nat) : ComputeStmt :=
  let value : Op .real [B] := if wide then
    .withPrecision .real .fp64 (.ref .real [B] "output")
    else .ref .real [B] "output"
  match output with
  | .fp32 => .store .real [B] (.region O (.ref .nat [B] "offsets"))
      (.compute (.alg .fp32 (if wide then .castFloat .real .real value
        else .ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (.region O (.ref .nat [B] "offsets"))
      (.compute (.alg .fp64 (if wide then .ref .real [B] "output"
        else .castFloat .real .real
          (.withPrecision .real .fp32 (.ref .real [B] "output")))))
      (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (.region O (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .fp16 value)) (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (.region O (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .bf16 value)) (.mask (.ref .bool [B] "mask"))
  | .f8e4 => .store .f8e4 [B] (.region O (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e4 value)) (.mask (.ref .bool [B] "mask"))
  | .f8e5 => .store .f8e5 [B] (.region O (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e5 value)) (.mask (.ref .bool [B] "mask"))

theorem source_store (x y output : Storage) (v : Bool)
    (hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody[7]? =
      some (expectedStore (isWide x y) output O B) := by
  cases x <;> cases y <;> cases output <;> cases v <;>
    first | contradiction | rfl

def expectedFormula (wide : Bool) (B : Nat) : ComputeStmt :=
  let expr : Op .real [B] :=
    .mul .real (.consSame .nil) (.ref .real [B] "x")
      (.log (.div .real (.consSame .nil)
        (.ref .real [B] "x") (.ref .real [B] "y")))
  .assign .real [B] "output"
    (if wide then .compute (.alg .fp64 expr) else .compute (.alg .fp32 expr))

/-- Independently specified promotion of each loaded operand into the arithmetic expression. -/
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

def expectedMixedFormula (x y : Storage) (B : Nat) : ComputeStmt :=
  let wide := isWide x y
  let a := promotedOperand x wide "x" B
  let b := promotedOperand y wide "y" B
  let expr := Op.mul .real (.consSame .nil) a
    (.log (.div .real (.consSame .nil) a b))
  .assign .real [B] "output"
    (if wide then .compute (.alg .fp64 expr) else .compute (.alg .fp32 expr))

/-- Pins division, logarithm and multiplication together for every supported source profile. -/
theorem source_formula (x y output : Storage) (v : Bool)
    (hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody[6]? =
      some (expectedMixedFormula x y B) := by
  cases x <;> cases y <;> cases output <;> cases v <;>
    first | contradiction | rfl

theorem fp32_formula (output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel .fp32 .fp32 output v X Y O N B).surfaceBody[6]? =
      some (expectedFormula Bool.false B) := by
  cases output <;> cases v <;> rfl

theorem fp64_formula (output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel .fp64 .fp64 output v X Y O N B).surfaceBody[6]? =
      some (expectedFormula Bool.true B) := by
  cases output <;> cases v <;> rfl

theorem source_shape (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).surfaceBody.length = 8 ∧
    (kernel x y output v X Y O N B).inputs = [X,Y] ∧
    (kernel x y output v X Y O N B).outputs = [O] := by
  cases x <;> cases y <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute.Precision

import bench.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRange _ _ _ _ body => body
  | _ => []

def expectedLoad (input : Storage) (name : RegName) (X : RegionName)
    (B : Nat) (offset : Op .nat [B]) (zero : Bool) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] name (.compute (.load .fp32 (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") ((Op.const 0.0).broadcast [B]) else .mask (.ref .bool [B] "x_ptr_mask"))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp64 (.load .real (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") ((Op.const 0.0).broadcast [B]) else .mask (.ref .bool [B] "x_ptr_mask")))))
  | .fp16 => .assign .fp16 [B] name (.alg (.load .fp16 (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") (((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B])) else .mask (.ref .bool [B] "x_ptr_mask"))))
  | .bf16 => .assign .bf16 [B] name (.alg (.load .bf16 (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") (((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B])) else .mask (.ref .bool [B] "x_ptr_mask"))))
  | .f8e4 => .assign .f8e4 [B] name (.alg (.load .f8e4 (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") (((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B])) else .mask (.ref .bool [B] "x_ptr_mask"))))
  | .f8e5 => .assign .f8e5 [B] name (.alg (.load .f8e5 (.region X offset) (if zero then .maskOther (.ref .bool [B] "x_ptr_mask") (((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B])) else .mask (.ref .bool [B] "x_ptr_mask"))))

def expectedXF (input : Storage) (B : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .real .real (.ref .real [B] "x"))))
  | .fp64 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .real .real (.withPrecision .real .fp64 (.ref .real [B] "x")))))
  | .fp16 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .fp16 .real (.ref .fp16 [B] "x"))))
  | .bf16 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .bf16 .real (.ref .bf16 [B] "x"))))
  | .f8e4 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .f8e4 .real (.ref .f8e4 [B] "x"))))
  | .f8e5 => .assign .real [B] "xf" (.compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 [B] "x"))))

def xAddress (sxk B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (.ref .nat [] "offset_m")
    (.mul .nat .scalarR (.ref .nat [B] "offset_n") (.constNat sxk))

def sourceBody (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :=
  (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).surfaceBody

/-- Both source loops and both masks remain present, with all IO regions. -/
theorem source_shape (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps).length = 9 ∧
    (loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[5]!)).length = 5 ∧
    (loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[8]!)).length = 8 := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Pass one loads actual storage, then casts to fp32; pass two reloads the raw
input and raw weight. Only input loads specify zero padding. -/
theorem source_loads (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    ((loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[5]!)).drop 2).take 2 =
      [expectedLoad input "x" X B (xAddress sxk B) Bool.true, expectedXF input B] ∧
    ((loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[8]!)).drop 2).take 2 =
      [expectedLoad weight "rms_w_offset" W B
        (.mul .nat .scalarR (.ref .nat [B] "offset_n") (.constNat sw)) Bool.false,
       expectedLoad input "x" X B (xAddress sxk B) Bool.true] := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

def expectedDivision (input : Storage) (B : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] "x_new" (.compute (.alg .fp32 (.div .real .scalarR (.ref .real [B] "x") (.ref .real [] "std"))))
  | .fp64 => .assign .real [B] "x_new" (.compute (.alg .fp64 (.div .real .scalarR (.ref .real [B] "x") (.withPrecision .real .fp32 (.ref .real [] "std")))))
  | .fp16 => .assign .real [B] "x_new" (.compute (.alg .fp32 (.div .real .scalarR (.castFloat .fp16 .real (.ref .fp16 [B] "x")) (.ref .real [] "std"))))
  | .bf16 => .assign .real [B] "x_new" (.compute (.alg .fp32 (.div .real .scalarR (.castFloat .bf16 .real (.ref .bf16 [B] "x")) (.ref .real [] "std"))))
  | .f8e4 => .assign .real [B] "x_new" (.compute (.alg .fp32 (.div .real .scalarR (.castFloat .f8e4 .real (.ref .f8e4 [B] "x")) (.ref .real [] "std"))))
  | .f8e5 => .assign .real [B] "x_new" (.compute (.alg .fp32 (.div .real .scalarR (.castFloat .f8e5 .real (.ref .f8e5 [B] "x")) (.ref .real [] "std"))))

def expectedMultiply (input weight : Storage) (B : Nat) : ComputeStmt :=
  let wide := decide (input = .fp64) || decide (weight = .fp64)
  let x : Op .real [B] := if decide (input ≠ .fp64) && wide then
    .withPrecision .real .fp32 (.ref .real [B] "x_new") else .ref .real [B] "x_new"
  let w : Op .real [B] := match weight with
    | .fp32 => if wide then .withPrecision .real .fp32 (.ref .real [B] "rms_w_offset")
      else .ref .real [B] "rms_w_offset"
    | .fp64 => .ref .real [B] "rms_w_offset"
    | .fp16 => .castFloat .fp16 .real (if wide then
        .withPrecision .fp16 .fp32 (.ref .fp16 [B] "rms_w_offset") else .ref .fp16 [B] "rms_w_offset")
    | .bf16 => .castFloat .bf16 .real (if wide then
        .withPrecision .bf16 .fp32 (.ref .bf16 [B] "rms_w_offset") else .ref .bf16 [B] "rms_w_offset")
    | .f8e4 => .castFloat .f8e4 .real (if wide then
        .withPrecision .f8e4 .fp32 (.ref .f8e4 [B] "rms_w_offset") else .ref .f8e4 [B] "rms_w_offset")
    | .f8e5 => .castFloat .f8e5 .real (if wide then
        .withPrecision .f8e5 .fp32 (.ref .f8e5 [B] "rms_w_offset") else .ref .f8e5 [B] "rms_w_offset")
  .assign .real [B] "out" (if wide then
    .compute (.alg .fp64 (.mul .real Broadcast.nil.consSame x w)) else
    .compute (.alg .fp32 (.mul .real Broadcast.nil.consSame x w)))

def expectedStore (input weight output : Storage) (O : RegionName) (B : Nat) : ComputeStmt :=
  let wide := decide (input = .fp64) || decide (weight = .fp64)
  let raw : Op .real [B] := .ref .real [B] "out"
  let wideScoped : Op .real [B] := if wide then .withPrecision .real .fp64 raw else raw
  match output with
  | .fp32 => .store .real [B] (.region O (.ref .nat [B] "out_offset"))
      (.compute (.alg .fp32 (if wide then .castFloat .real .real wideScoped else raw)))
      (.mask (.ref .bool [B] "x_ptr_mask"))
  | .fp64 => .store .real [B] (.region O (.ref .nat [B] "out_offset"))
      (.compute (.alg .fp64 (if wide then raw else
        .castFloat .real .real (.withPrecision .real .fp32 raw))))
      (.mask (.ref .bool [B] "x_ptr_mask"))
  | .fp16 => .store .fp16 [B] (.region O (.ref .nat [B] "out_offset"))
      (.alg (.castFloat .real .fp16 wideScoped)) (.mask (.ref .bool [B] "x_ptr_mask"))
  | .bf16 => .store .bf16 [B] (.region O (.ref .nat [B] "out_offset"))
      (.alg (.castFloat .real .bf16 wideScoped)) (.mask (.ref .bool [B] "x_ptr_mask"))
  | .f8e4 => .store .f8e4 [B] (.region O (.ref .nat [B] "out_offset"))
      (.alg (.castFloat .real .f8e4 wideScoped)) (.mask (.ref .bool [B] "x_ptr_mask"))
  | .f8e5 => .store .f8e5 [B] (.region O (.ref .nat [B] "out_offset"))
      (.alg (.castFloat .real .f8e5 wideScoped)) (.mask (.ref .bool [B] "x_ptr_mask"))

/-- The denominator stays fp32, and each arithmetic operation uses its actual
promoted dtype before exactly one destination conversion. -/
theorem source_arithmetic (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[7]! =
      .assign .real [] "std" (.compute (.alg .fp32
        (.sqrt (.add .real .nil (.ref .real [] "var") (.const eps))))) ∧
    ((loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[8]!)).drop 4).take 2 =
      [expectedDivision input B, expectedMultiply input weight B] ∧
    (loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[8]!))[7]! =
      expectedStore input weight output O B := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- The complete variance recurrence and final division execute in fp32. -/
theorem variance_precision (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (loopBody ((sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[5]!))[4]! =
      .assign .real [B] "var" (.compute (.alg .fp32
        (.add .real Broadcast.nil.consSame (.ref .real [B] "var")
          (.mul .real Broadcast.nil.consSame (.ref .real [B] "xf") (.ref .real [B] "xf"))))) ∧
    (sourceBody input weight output v X W O sxb sxm sxk sw sob som sok N B eps)[6]! =
      .assign .real [] "var" (.compute (.alg .fp32
        (.div .real .nil (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [B] "var"))
          (.const (N : ℝ))))) := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

theorem io_roles (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).inputs = [X, W] ∧
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).outputs = [O] := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision

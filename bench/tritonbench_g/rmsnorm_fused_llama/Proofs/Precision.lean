import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option maxErrors 5
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRange _ _ _ _ body => body
  | _ => []

def body (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :=
  (typedKernel input weight output v X Y W stride N B eps).surfaceBody

def active (first : Bool) (N B : Nat) : Op .bool [B] :=
  if first then .lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)
  else .ref .bool [B] "mask"

def inputCast (input : Storage) (first : Bool) (N B : Nat) : ComputeStmt :=
  let p : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] "X") (.ref .nat [B] "cols")
  let mask := active first N B
  let zero : Op .real [B] := (Op.const 0.0).broadcast [B]
  let raw : Op .real [B] := .load .real (.ptr p) (.maskOther mask zero)
  let value : Op .real [B] := match input with
    | .fp32 => .castFloat .real .real raw
    | .fp64 => .castFloat .real .real (.withPrecision .real .fp64 raw)
    | .fp16 => .castFloat .fp16 .real (.load .fp16 (.ptr p) (.maskOther mask ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B])))
    | .bf16 => .castFloat .bf16 .real (.load .bf16 (.ptr p) (.maskOther mask ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B])))
    | .f8e4 => .castFloat .f8e4 .real (.load .f8e4 (.ptr p) (.maskOther mask ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B])))
    | .f8e5 => .castFloat .f8e5 .real (.load .f8e5 (.ptr p) (.maskOther mask ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B])))
  .assign .real [B] "x" (.compute (.alg .fp32 value))

def weightLoad (weight : Storage) (W : RegionName) (B : Nat) : ComputeStmt :=
  let p (d : TileDType) : MemAccess d [B] := .region W.cast (.ref .nat [B] "cols")
  let mask : MaskOpt _ [B] := .mask (.ref .bool [B] "mask")
  let raw : Op .real [B] := .load .real (p .real) mask
  let value : Op .real [B] := match weight with
    | .fp32 => .castFloat .real .real raw
    | .fp64 => .castFloat .real .real (.withPrecision .real .fp64 raw)
    | .fp16 => .castFloat .fp16 .real (.load .fp16 (p .fp16) (.mask (.ref .bool [B] "mask")))
    | .bf16 => .castFloat .bf16 .real (.load .bf16 (p .bf16) (.mask (.ref .bool [B] "mask")))
    | .f8e4 => .castFloat .f8e4 .real (.load .f8e4 (p .f8e4) (.mask (.ref .bool [B] "mask")))
    | .f8e5 => .castFloat .f8e5 .real (.load .f8e5 (p .f8e5) (.mask (.ref .bool [B] "mask")))
  .assign .real [B] "w" (.compute (.alg .fp32 value))

def weighted (B : Nat) : ComputeStmt :=
  .assign .real [B] "y" (.compute (.alg .fp32
    (.mul .real Broadcast.nil.consSame (.ref .real [B] "x_hat") (.ref .real [B] "w"))))

def store (output : Storage) (B : Nat) : ComputeStmt :=
  let half : Op .fp16 [B] := .castFloat .real .fp16 (.ref .real [B] "y")
  let p : Op .ptr [B] := .ptrAdd .scalarL (.ref .ptr [] "Y") (.ref .nat [B] "cols")
  match output with
  | .fp32 => .store .real [B] (.ptr p) (.compute (.alg .fp32 (.castFloat .fp16 .real half))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (.ptr p) (.compute (.alg .fp64 (.castFloat .fp16 .real (.withPrecision .fp16 .fp32 half)))) (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (.ptr p) (.alg half) (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (.ptr p) (.alg (.castFloat .fp16 .bf16 half)) (.mask (.ref .bool [B] "mask"))
  | .f8e4 => .store .f8e4 [B] (.ptr p) (.alg (.castFloat .fp16 .f8e4 half)) (.mask (.ref .bool [B] "mask"))
  | .f8e5 => .store .f8e5 [B] (.ptr p) (.alg (.castFloat .fp16 .f8e5 half)) (.mask (.ref .bool [B] "mask"))

/-- Exact load and store carriers and casts for all 216 floating combinations. -/
theorem memory_profiles (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (loopBody (body input weight output v X Y W stride N B eps)[4]!)[1]! = inputCast input Bool.true N B ∧
    (loopBody (body input weight output v X Y W stride N B eps)[7]!)[2]! = weightLoad weight W B ∧
    (loopBody (body input weight output v X Y W stride N B eps)[7]!)[3]! = inputCast input Bool.false N B ∧
    (loopBody (body input weight output v X Y W stride N B eps)[7]!)[6]! = store output B := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl⟩

/-- Fp32 normalization and weight multiplication, independent of storage formats. -/
theorem computation_profiles (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (body input weight output v X Y W stride N B eps)[6]! =
      .assign .real [] "rstd" (.compute (.alg .fp32
        (.div .real .nil (.const 1) (.sqrt (.add .real .nil (.ref .real [] "var") (.const eps)))))) ∧
    (loopBody (body input weight output v X Y W stride N B eps)[7]!)[4]! =
      .assign .real [B] "x_hat" (.compute (.alg .fp32
        (.mul .real .scalarR (.ref .real [B] "x") (.ref .real [] "rstd")))) ∧
    (loopBody (body input weight output v X Y W stride N B eps)[7]!)[5]! = weighted B := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Both variance accumulation and its final reduction/division retain fp32. -/
theorem variance_precision (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (loopBody (body input weight output v X Y W stride N B eps)[4]!)[2]! =
      .assign .real [B] "_var" (.compute (.alg .fp32
        (.add .real Broadcast.nil.consSame (.ref .real [B] "_var")
          (.mul .real Broadcast.nil.consSame (.ref .real [B] "x") (.ref .real [B] "x"))))) ∧
    (body input weight output v X Y W stride N B eps)[5]! =
      .assign .real [] "var" (.compute (.alg .fp32
        (.div .real .nil (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [B] "_var"))
          (.const (N : ℝ))))) := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

theorem io_roles (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X Y W stride N B eps).inputs = [X, W] ∧
    (typedKernel input weight output v X Y W stride N B eps).outputs = [Y] := by
  cases input <;> cases weight <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.Precision

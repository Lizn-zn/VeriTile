import bench.tritonbench_g.sin_computation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SinComputation.Precision
open VeriTile.Triton
set_option maxHeartbeats 2000000

def expectedLoad (input : InputStorage) (X : RegionName) (B : Nat) : ComputeStmt :=
  let address : MemAccess .real [B] := .region X (.ref .nat [B] "offsets")
  let mask : MaskOpt .real [B] := .mask (.ref .bool [B] "mask")
  match input with
  | .fp32 => .assign .real [B] "x" (.compute (.load .fp32 address mask))
  | .fp64 => .assign .real [B] "x" (.compute (.alg .fp64 (.load .real address mask)))

def expectedSin (input : InputStorage) (B : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] "output"
      (.compute (.alg .fp32 (.sin (.ref .real [B] "x"))))
  | .fp64 => .assign .real [B] "output"
      (.compute (.alg .fp64 (.sin (.ref .real [B] "x"))))

def expectedStore (input : InputStorage) (output : OutputStorage) (Y : RegionName) (B : Nat) : ComputeStmt :=
  match input, output with
  | .fp32, .fp32 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp32 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp32, .fp64 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp64 (.castFloat .real .real (.withPrecision .real .fp32 (.ref .real [B] "output"))))) (.mask (.ref .bool [B] "mask"))
  | .fp32, .fp16 => .store .fp16 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .fp16 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp32, .bf16 => .store .bf16 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .bf16 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp32, .f8e4 => .store .f8e4 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e4 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp32, .f8e5 => .store .f8e5 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e5 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .fp32 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp32 (.castFloat .real .real (.withPrecision .real .fp64 (.ref .real [B] "output"))))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .fp64 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp64 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .fp16 => .store .fp16 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp64 (.ref .real [B] "output")))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .bf16 => .store .bf16 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .bf16 (.withPrecision .real .fp64 (.ref .real [B] "output")))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .f8e4 => .store .f8e4 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e4 (.withPrecision .real .fp64 (.ref .real [B] "output")))) (.mask (.ref .bool [B] "mask"))
  | .fp64, .f8e5 => .store .f8e5 [B] (.region Y (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .f8e5 (.withPrecision .real .fp64 (.ref .real [B] "output")))) (.mask (.ref .bool [B] "mask"))

/-- Exact complete floating suffix: raw load without `other`, the source sine
at its preserved precision, and one output store with its precise conversion. -/
theorem source_precision (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input output v X Y N B).surfaceBody.drop 4 =
      [expectedLoad input X B, expectedSin input B, expectedStore input output Y B] := by
  cases input <;> cases output <;> cases v <;> rfl

/-- No extra padding, computations, statements or IO roles are introduced. -/
theorem source_shape (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input output v X Y N B).surfaceBody.length = 7 ∧
    (typedKernel input output v X Y N B).inputs = [X] ∧
    (typedKernel input output v X Y N B).outputs = [Y] := by
  cases input <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinComputation.Precision

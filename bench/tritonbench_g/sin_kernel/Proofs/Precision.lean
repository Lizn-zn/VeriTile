import bench.tritonbench_g.sin_kernel.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SinKernel.Precision
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

def expectedStore (input : InputStorage) (Y : RegionName) (B : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp32 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (.region Y (.ref .nat [B] "offsets"))
      (.compute (.alg .fp64 (.ref .real [B] "output"))) (.mask (.ref .bool [B] "mask"))

/-- Exact complete floating suffix: raw load without `other`, the source sine
at its preserved precision, and one output store with its precise conversion. -/
theorem source_precision (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).surfaceBody.drop 4 =
      [expectedLoad input X B, expectedSin input B, expectedStore input Y B] := by
  cases input <;> cases v <;> rfl

/-- No extra padding, computations, statements or IO roles are introduced. -/
theorem source_shape (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).surfaceBody.length = 7 ∧
    (typedKernel input v X Y N B).inputs = [X] ∧
    (typedKernel input v X Y N B).outputs = [Y] := by
  cases input <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel.Precision

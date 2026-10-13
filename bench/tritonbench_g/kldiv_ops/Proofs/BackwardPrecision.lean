import bench.tritonbench_g.kldiv_ops.Proofs.SourceDomain

namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 1000000

def backwardBody (source : ComputeKernel) : List ComputeStmt :=
  match (source.surfaceBody[5]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ statements) => statements
  | _ => []

def negativeAssignment (target : Storage) (B : Nat) : ComputeStmt :=
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

/-- Every formal default profile uses one weak scalar -1 and the target's arithmetic type.
The fp8 cases describe the formal extension; SourceDomain excludes their nonzero
scalar construction from pinned Triton compilation coverage. -/
theorem backward_default_negative_precision (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardBody (backwardDefault target output v T O st so N B))[3]? =
      some (negativeAssignment target B) := by
  cases target <;> cases output <;> cases v <;> rfl

/-- The signed loop is retained, with its actual initialization prefix and five body statements. -/
theorem backward_default_control (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    let source := backwardDefault target output v T O st so N B
    source.surfaceBody.length = 6 ∧
      source.surfaceBody[5]? = some (.forRangeInt "i" (.constInt 0) (.constInt N)
        (.constNat B) (backwardBody source)) ∧
      (backwardBody source).length = 5 := by
  cases target <;> cases output <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource

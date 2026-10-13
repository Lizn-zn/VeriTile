import bench.tritonbench_g.index_select_bwd.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def expectedInputLoad (input : Storage) (BI BC : Nat) : ComputeExpr .real [BI,BC] :=
  match input with
  | .fp32 => .compute (.alg .fp32 (.castFloat .real .real
      (.load .real (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask")))))
  | .fp16 => .compute (.alg .fp32 (.castFloat .fp16 .real
      (.load .fp16 (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask")))))
  | .bf16 => .compute (.alg .fp32 (.castFloat .bf16 .real
      (.load .bf16 (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask")))))
  | .f8e4 => .compute (.alg .fp32 (.castFloat .f8e4 .real
      (.load .f8e4 (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask")))))
  | .f8e5 => .compute (.alg .fp32 (.castFloat .f8e5 .real
      (.load .f8e5 (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask")))))
  | .fp64 => .compute (.alg .fp32 (.castFloat .real .real (.withPrecision .real .fp64
      (.load .real (.ptr (.ref .ptr [BI,BC] "grad_output_offsets"))
        (.mask (.ref .bool [BI,BC] "grad_output_mask"))))))

def expectedOutputStore (output : Storage) (BI BC : Nat) : ComputeStmt :=
  match output with
  | .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.compute (.alg .fp32 (.ref .real [BI,BC] "grad_output")))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))
  | .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [BI,BC] "grad_output")))))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))
  | .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.alg (.castFloat .real .fp16 (.ref .real [BI,BC] "grad_output")))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))
  | .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.alg (.castFloat .real .bf16 (.ref .real [BI,BC] "grad_output")))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))
  | .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.alg (.castFloat .real .f8e4 (.ref .real [BI,BC] "grad_output")))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))
  | .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "grad_source_offsets"))
      (.alg (.castFloat .real .f8e5 (.ref .real [BI,BC] "grad_output")))
      (.mask (.ref .bool [BI,BC] "grad_output_mask"))

/-- All storage pairs retain the actual source carrier, mandatory fp32 result,
and exactly the output storage conversion. Missing `other` remains missing. -/
theorem source_precision (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp rows n c s0 s1 BI BC).surfaceBody[6]? =
      some (.assign .real [BI,BC] "grad_output" (expectedInputLoad input BI BC)) ∧
    (source input output index v out idx inp rows n c s0 s1 BI BC).surfaceBody[9]? =
      some (expectedOutputStore output BI BC) := by
  cases input <;> cases output <;> cases index <;> cases v <;> exact ⟨rfl,rfl⟩

/-- Both signed storage declarations read the integer channel, never a natural
index buffer. Only the source's row mask guards this independent index load. -/
theorem signed_index_load (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp rows n c s0 s1 BI BC).surfaceBody[7]? =
      some (.assign .int [BI] "grad_source_indices" (.alg (.load .int
        (.region (Region.cast idx) (.ref .nat [BI] "grad_output_indices"))
        (.mask (.lt .nat .scalarR (.ref .nat [BI] "grad_output_indices") (.constNat n)))))) := by
  cases input <;> cases output <;> cases index <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectBwd

import bench.tritonbench_g.index_select_cat.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def expectedInputLoad (input : Storage) (BI BC : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [BI,BC] "output" (.compute (.load .fp32 (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask"))))
  | .fp16 => .assign .fp16 [BI,BC] "output" (.alg (.load .fp16 (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask"))))
  | .bf16 => .assign .bf16 [BI,BC] "output" (.alg (.load .bf16 (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask"))))
  | .f8e4 => .assign .f8e4 [BI,BC] "output" (.alg (.load .f8e4 (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask"))))
  | .f8e5 => .assign .f8e5 [BI,BC] "output" (.alg (.load .f8e5 (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask"))))
  | .fp64 => .assign .real [BI,BC] "output" (.compute (.alg .fp64 (.load .real (.ptr (.ref .ptr [BI,BC] "source_offsets")) (.mask (.ref .bool [BI,BC] "mask")))))

def expectedOutputStore (input output : Storage) (BI BC : Nat) : ComputeStmt :=
  match input, output with
  | .fp32, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp32, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .fp16 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp32, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .bf16 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp32, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .f8e4 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp32, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .f8e5 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp32, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.castFloat .real .real (.withPrecision .real .fp32 (.ref .real [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.castFloat .fp16 .real (.ref .fp16 [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.ref .fp16 [BI,BC] "output")) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .fp16 .bf16 (.ref .fp16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .fp16 .f8e4 (.ref .fp16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .fp16 .f8e5 (.ref .fp16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp16, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.castFloat .fp16 .real (.withPrecision .fp16 .fp32 (.ref .fp16 [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.castFloat .bf16 .real (.ref .bf16 [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .bf16 .fp16 (.ref .bf16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.ref .bf16 [BI,BC] "output")) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .bf16 .f8e4 (.ref .bf16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .bf16 .f8e5 (.ref .bf16 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .bf16, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.castFloat .bf16 .real (.withPrecision .bf16 .fp32 (.ref .bf16 [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.castFloat .f8e4 .real (.ref .f8e4 [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e4 .fp16 (.ref .f8e4 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e4 .bf16 (.ref .f8e4 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.ref .f8e4 [BI,BC] "output")) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e4 .f8e5 (.ref .f8e4 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e4, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.castFloat .f8e4 .real (.withPrecision .f8e4 .fp32 (.ref .f8e4 [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e5 .fp16 (.ref .f8e5 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e5 .bf16 (.ref .f8e5 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .f8e5 .f8e4 (.ref .f8e5 [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.ref .f8e5 [BI,BC] "output")) (.mask (.ref .bool [BI,BC] "mask"))
  | .f8e5, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.castFloat .f8e5 .real (.withPrecision .f8e5 .fp32 (.ref .f8e5 [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .fp32 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp32 (.castFloat .real .real (.withPrecision .real .fp64 (.ref .real [BI,BC] "output"))))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .fp16 => .store .fp16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp64 (.ref .real [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .bf16 => .store .bf16 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .bf16 (.withPrecision .real .fp64 (.ref .real [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .f8e4 => .store .f8e4 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .f8e4 (.withPrecision .real .fp64 (.ref .real [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .f8e5 => .store .f8e5 [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.alg (.castFloat .real .f8e5 (.withPrecision .real .fp64 (.ref .real [BI,BC] "output")))) (.mask (.ref .bool [BI,BC] "mask"))
  | .fp64, .fp64 => .store .real [BI,BC] (.ptr (.ref .ptr [BI,BC] "output_offsets"))
      (.compute (.alg .fp64 (.ref .real [BI,BC] "output"))) (.mask (.ref .bool [BI,BC] "mask"))

/-- The gather load preserves its actual storage carrier and has no implicit
fp32 promotion or padding value. Stores convert exactly when storage differs. -/
theorem source_precision (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp n c s0 s1 BI BC).surfaceBody[7]? =
      some (expectedInputLoad input BI BC) ∧
    (source input output index v out idx inp n c s0 s1 BI BC).surfaceBody[9]? =
      some (expectedOutputStore input output BI BC) := by
  cases input <;> cases output <;> cases index <;> cases v <;> exact ⟨rfl,rfl⟩

/-- Both signed declarations retain the integer index load and the source's
independent row mask, with no natural-index substitution or added row guard. -/
theorem signed_index_load (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    (source input output index v out idx inp n c s0 s1 BI BC).surfaceBody[3]? =
      some (.assign .int [BI] "rows" (.alg (.load .int
        (.region (Region.cast idx) (.ref .nat [BI] "indices"))
        (.mask (.lt .nat .scalarR (.ref .nat [BI] "indices") (.constNat n)))))) := by
  cases input <;> cases output <;> cases index <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectCat

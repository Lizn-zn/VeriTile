import bench.tritonbench_g.var_len_copy.Proofs.FPBody

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.SourceChecks
open VeriTile Triton TypedSource FPProof
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Signed addition is performed before converting the complete offset to Nat. -/
theorem fp64_load_profile (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    loadStatement .fp64 Bool.false os ol ns src dst B =
      .assign .real [B] "v" (.compute (.alg .fp64
        (.load .real (.region src (offsets Bool.false B))
          (.mask (.lt .int .scalarR (.castNatToInt (.ref .nat [B] "old_offset"))
            (.ref .int [] "length")))))) := by rfl

/-- A raw fp32 load retains its distinct fp32 backend callback. -/
theorem fp32_load_profile (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    loadStatement .fp32 Bool.false os ol ns src dst B =
      .assign .real [B] "v" (.compute (.load .fp32
        (.region src (offsets Bool.false B))
          (.mask (.lt .int .scalarR (.castNatToInt (.ref .nat [B] "old_offset"))
            (.ref .int [] "length"))))) := by rfl

/-- A source storage tag is preserved on the raw half load. -/
theorem fp16_load_profile (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    loadStatement .fp16 Bool.false os ol ns src dst B =
      .assign .fp16 [B] "v" (.alg
        (.load .fp16 (.region src.cast (offsets Bool.false B))
          (.mask (.lt .int .scalarR (.castNatToInt (.ref .nat [B] "old_offset"))
            (.ref .int [] "length"))))) := by rfl

/-- Source storage channels; fp32/fp64 working precision is pinned separately below. -/
def storageFloat : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

/-- Independent golden matrix for the source store's destination conversion.
The fp64 boundary retains the child's original precision in both directions. -/
def expectedStoreValue (input output : Storage) (B : Nat) :
    ComputeExpr (storageFloat output).toTileDType [B] :=
  match input, output with
  | .fp32, .fp32 => .compute (.alg .fp32 (.ref .real [B] "v"))
  | .fp32, .fp16 => .alg (.castFloat .real .fp16 (.ref .real [B] "v"))
  | .fp32, .bf16 => .alg (.castFloat .real .bf16 (.ref .real [B] "v"))
  | .fp32, .fp64 => .compute (.alg .fp64 (.castFloat .real .real (.withPrecision .real .fp32 (.ref .real [B] "v"))))
  | .fp32, .f8e4 => .alg (.castFloat .real .f8e4 (.ref .real [B] "v"))
  | .fp32, .f8e5 => .alg (.castFloat .real .f8e5 (.ref .real [B] "v"))
  | .fp16, .fp32 => .compute (.alg .fp32 (.castFloat .fp16 .real (.ref .fp16 [B] "v")))
  | .fp16, .fp16 => .alg (.ref .fp16 [B] "v")
  | .fp16, .bf16 => .alg (.castFloat .fp16 .bf16 (.ref .fp16 [B] "v"))
  | .fp16, .fp64 => .compute (.alg .fp64 (.castFloat .fp16 .real (.withPrecision .fp16 .fp32 (.ref .fp16 [B] "v"))))
  | .fp16, .f8e4 => .alg (.castFloat .fp16 .f8e4 (.ref .fp16 [B] "v"))
  | .fp16, .f8e5 => .alg (.castFloat .fp16 .f8e5 (.ref .fp16 [B] "v"))
  | .bf16, .fp32 => .compute (.alg .fp32 (.castFloat .bf16 .real (.ref .bf16 [B] "v")))
  | .bf16, .fp16 => .alg (.castFloat .bf16 .fp16 (.ref .bf16 [B] "v"))
  | .bf16, .bf16 => .alg (.ref .bf16 [B] "v")
  | .bf16, .fp64 => .compute (.alg .fp64 (.castFloat .bf16 .real (.withPrecision .bf16 .fp32 (.ref .bf16 [B] "v"))))
  | .bf16, .f8e4 => .alg (.castFloat .bf16 .f8e4 (.ref .bf16 [B] "v"))
  | .bf16, .f8e5 => .alg (.castFloat .bf16 .f8e5 (.ref .bf16 [B] "v"))
  | .fp64, .fp32 => .compute (.alg .fp32 (.castFloat .real .real (.withPrecision .real .fp64 (.ref .real [B] "v"))))
  | .fp64, .fp16 => .alg (.castFloat .real .fp16 (.withPrecision .real .fp64 (.ref .real [B] "v")))
  | .fp64, .bf16 => .alg (.castFloat .real .bf16 (.withPrecision .real .fp64 (.ref .real [B] "v")))
  | .fp64, .fp64 => .compute (.alg .fp64 (.ref .real [B] "v"))
  | .fp64, .f8e4 => .alg (.castFloat .real .f8e4 (.withPrecision .real .fp64 (.ref .real [B] "v")))
  | .fp64, .f8e5 => .alg (.castFloat .real .f8e5 (.withPrecision .real .fp64 (.ref .real [B] "v")))
  | .f8e4, .fp32 => .compute (.alg .fp32 (.castFloat .f8e4 .real (.ref .f8e4 [B] "v")))
  | .f8e4, .fp16 => .alg (.castFloat .f8e4 .fp16 (.ref .f8e4 [B] "v"))
  | .f8e4, .bf16 => .alg (.castFloat .f8e4 .bf16 (.ref .f8e4 [B] "v"))
  | .f8e4, .fp64 => .compute (.alg .fp64 (.castFloat .f8e4 .real (.withPrecision .f8e4 .fp32 (.ref .f8e4 [B] "v"))))
  | .f8e4, .f8e4 => .alg (.ref .f8e4 [B] "v")
  | .f8e4, .f8e5 => .alg (.castFloat .f8e4 .f8e5 (.ref .f8e4 [B] "v"))
  | .f8e5, .fp32 => .compute (.alg .fp32 (.castFloat .f8e5 .real (.ref .f8e5 [B] "v")))
  | .f8e5, .fp16 => .alg (.castFloat .f8e5 .fp16 (.ref .f8e5 [B] "v"))
  | .f8e5, .bf16 => .alg (.castFloat .f8e5 .bf16 (.ref .f8e5 [B] "v"))
  | .f8e5, .fp64 => .compute (.alg .fp64 (.castFloat .f8e5 .real (.withPrecision .f8e5 .fp32 (.ref .f8e5 [B] "v"))))
  | .f8e5, .f8e4 => .alg (.castFloat .f8e5 .f8e4 (.ref .f8e5 [B] "v"))
  | .f8e5, .f8e5 => .alg (.ref .f8e5 [B] "v")

def expectedStore (input output : Storage) (dst : RegionName) (B : Nat) : ComputeStmt :=
  .store (storageFloat output).toTileDType [B]
    (.region dst.cast (.castIntToNat (.add .int .scalarL
      (.add .int .nil (.ref .int [] "new_start") (.ref .int [] "i"))
      (.castNatToInt (.ref .nat [B] "new_offset")))))
    (expectedStoreValue input output B)
    (.mask (.lt .int .scalarR (.castNatToInt (.ref .nat [B] "new_offset"))
      (.ref .int [] "length")))

/-- All 36 floating source/destination pairs and both source versions have the
exact typed destination store, cast boundary, precision scopes, and lane-only mask. -/
theorem all_store_profiles (input output : Storage) (v : Bool)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (body input output v os ol ns src dst B).tail = [expectedStore input output dst B] := by
  cases input <;> cases output <;> cases v <;> rfl

end VeriTile.Bench.TritonBenchG.VarLenCopy.SourceChecks

#guard_msgs (drop info) in
#auditModuleAxioms

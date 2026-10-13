import bench.tritonbench_g.nested_loops_processing.Kernels

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.SourceChecks
open VeriTile.Triton TypedSource

theorem scatter_lane_order : TileShape.allIndices [2,2] =
    [(0, 0, PUnit.unit), (0, 1, PUnit.unit),
     (1, 0, PUnit.unit), (1, 1, PUnit.unit)] := by
  decide

def outer (storage : Storage) (v : Bool) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  match ((kernel storage v src dst sm sn).surfaceBody[6]? : Option ComputeStmt) with
  | some (.forRange "i" 0 2 1 body) => body
  | _ => []

def middle (storage : Storage) (v : Bool) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  match ((outer storage v src dst sm sn)[1]? : Option ComputeStmt) with
  | some (.forRange "j" 0 2 1 body) => body
  | _ => []

def inner (storage : Storage) (v : Bool) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  match ((middle storage v src dst sm sn)[2]? : Option ComputeStmt) with
  | some (.forRange "k" 0 2 1 body) => body
  | _ => []

theorem full_loop_shape (storage : Storage) (v : Bool)
    (src dst : RegionName) (sm sn : Int) :
    (kernel storage v src dst sm sn).surfaceBody.length = 7 ∧
    (outer storage v src dst sm sn).length = 3 ∧
    (middle storage v src dst sm sn).length = 3 ∧
    (inner storage v src dst sm sn).length = 8 := by
  cases storage <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl⟩

theorem int32_inner_store (v : Bool) (src dst : RegionName) (sm sn : Int) :
    (inner .int32 v src dst sm sn)[2]? = some
      (.store .int [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (.alg (.ref .int [2,2] "a1")) .none) ∧
    (inner .int32 v src dst sm sn)[4]? = some
      (.store .int [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (.alg (.ref .int [2,2] "a2")) .none) ∧
    (inner .int32 v src dst sm sn)[6]? = some
      (.store .int [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (.alg (.ref .int [2,2] "a3")) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

def dtype : Storage → TileDType
  | .int8 | .int16 | .int32 | .int64 => .int
  | .uint8 | .uint16 | .uint32 | .uint64 => .nat
  | .boolean => .bool
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .f8e4 => .f8e4
  | .f8e5 => .f8e5

def expectedLoad : (storage : Storage) → ComputeExpr (dtype storage) [2,2]
  | .int8 | .int16 | .int32 | .int64 => .alg (.load .int (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .uint8 | .uint16 | .uint32 | .uint64 => .alg (.load .nat (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .boolean => .alg (.load .bool (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .fp16 => .alg (.load .fp16 (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .bf16 => .alg (.load .bf16 (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .f8e4 => .alg (.load .f8e4 (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .f8e5 => .alg (.load .f8e5 (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .fp32 => .compute (.load .fp32 (.ptr (.ref .ptr [2,2] "a_ptrs")) .none)
  | .fp64 => .compute (.alg .fp64 (.load .real (.ptr (.ref .ptr [2,2] "a_ptrs")) .none))

def expectedValue : (storage : Storage) → RegName → ComputeExpr (dtype storage) [2,2]
  | .int8, name => .alg (.ref .int [2,2] name)
  | .int16, name => .alg (.ref .int [2,2] name)
  | .int32, name => .alg (.ref .int [2,2] name)
  | .int64, name => .alg (.ref .int [2,2] name)
  | .uint8, name => .alg (.ref .nat [2,2] name)
  | .uint16, name => .alg (.ref .nat [2,2] name)
  | .uint32, name => .alg (.ref .nat [2,2] name)
  | .uint64, name => .alg (.ref .nat [2,2] name)
  | .boolean, name => .alg (.ref .bool [2,2] name)
  | .fp16, name => .alg (.ref .fp16 [2,2] name)
  | .bf16, name => .alg (.ref .bf16 [2,2] name)
  | .f8e4, name => .alg (.ref .f8e4 [2,2] name)
  | .f8e5, name => .alg (.ref .f8e5 [2,2] name)
  | .fp32, name => .compute (.alg .fp32 (.ref .real [2,2] name))
  | .fp64, name => .compute (.alg .fp64 (.ref .real [2,2] name))

theorem all_payload_profiles (storage : Storage) (v : Bool)
    (src dst : RegionName) (sm sn : Int) :
    (outer storage v src dst sm sn)[0]? =
      some (.assign (dtype storage) [2,2] "a1" (expectedLoad storage)) ∧
    (middle storage v src dst sm sn)[1]? =
      some (.assign (dtype storage) [2,2] "a2" (expectedLoad storage)) ∧
    (inner storage v src dst sm sn)[1]? =
      some (.assign (dtype storage) [2,2] "a3" (expectedLoad storage)) ∧
    (inner storage v src dst sm sn)[2]? =
      some (.store (dtype storage) [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (expectedValue storage "a1") .none) ∧
    (inner storage v src dst sm sn)[4]? =
      some (.store (dtype storage) [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (expectedValue storage "a2") .none) ∧
    (inner storage v src dst sm sn)[6]? =
      some (.store (dtype storage) [2,2] (.ptr (.ref .ptr [2,2] "c_ptrs"))
        (expectedValue storage "a3") .none) := by
  cases storage <;> cases v <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.SourceChecks

#guard_msgs (drop info) in
#auditModuleAxioms

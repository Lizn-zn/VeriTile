import bench.tritonbench_g.fast_rope_embedding.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

def storeTypes (body : List ComputeStmt) : List TileDType :=
  body.filterMap fun st => match st with
    | .store d _ _ _ _ => some d
    | _ => none

def body (kernel : ComputeKernel) : List ComputeStmt :=
  ((Source.dynamicRange? kernel.surfaceBody "k").map (·.body)).getD []

def flip (kernel : ComputeKernel) : List ComputeStmt :=
  ((Source.conditional? kernel.surfaceBody (fun _ => Bool.true)).map (·.yes)).getD []

variable (input cosine sine : Storage) (swapped backwards : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat)

theorem source_precision :
    (types (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards).surfaceBody "sin1",
     types (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards).surfaceBody "cos1",
     types (flip (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards)) "sin1",
     types (body (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards)) "Q1",
     types (body (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards)) "Q2",
     storeTypes (body (rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards)))=
      ([storageType sine],[storageType cosine],[storageType sine],[storageType sine],[storageType sine],
        [storageType input,storageType input]) := by
  cases input <;> cases cosine <;> cases sine <;> cases swapped <;> cases backwards <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.SourceChecks

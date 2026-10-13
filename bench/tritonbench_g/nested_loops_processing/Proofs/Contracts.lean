import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadBoth

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.Contracts
open VeriTile Triton TypedSource

/-- Complete per-program execution and ordered mathematical output, a full raw
memory frame, and safety of every actual load/store. Tile injectivity separates
native concurrent lane stores from the retained cross-tile overwrite order. -/
def RegionCopy (storage : Storage) (version : Bool) (bounds : RegionBounds)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat) : Prop :=
  FullReal.TileInjective sm sn ∧
  PayloadReal.RegionCopyContract (PayloadReal.carrier storage)
    (RealCandidate.algorithm storage version src dst sm sn) s src dst sm sn ∧
  (RealCandidate.algorithm storage version src dst sm sn).TraceSafe bounds s

/-- The complete flat-memory version retains translated cell types and the full
outside-footprint frame, with source access safety and native lane injectivity. -/
def FlatCopy (storage : Storage) (version : Bool) (A : FlatAlloc)
    (s : BlockState) (src dst : RegionName) (sm sn : Nat) : Prop :=
  FullReal.TileInjective sm sn ∧
  PayloadReal.FlatCopyContract (PayloadReal.carrier storage) A
    (RealCandidate.algorithm storage version src dst sm sn) s src dst sm sn ∧
  (RealCandidate.algorithm storage version src dst sm sn).TraceSafe A.extent s

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.Contracts

#guard_msgs (drop info) in
#auditModuleAxioms

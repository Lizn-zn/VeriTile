import bench.tritonbench_g.chunk_retention_ops.Kernels

/-! These three complete source pairs have identical typed bodies in the two
corpus files. Kernel equality, including signatures, connects reused proofs
to this directory's own complete source. Backward state is different. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.SourceComparison
set_option maxHeartbeats 4000000

theorem forwardH (s : Storage) (swapped : Bool) :
    forwardHSource s swapped = VeriTile.Bench.TritonBenchG.ChunkRetention.forwardHSource s swapped := by
  cases s <;> cases swapped <;> rfl

theorem forwardO (s v : Storage) (swapped : Bool) :
    forwardOSource s v swapped = VeriTile.Bench.TritonBenchG.ChunkRetention.forwardOSource s v swapped := by
  cases s <;> cases v <;> cases swapped <;> rfl

theorem backwardQKV (s : Storage) (swapped : Bool) :
    backwardQKVSource s swapped = VeriTile.Bench.TritonBenchG.ChunkRetention.backwardQKVSource s swapped := by
  cases s <;> cases swapped <;> rfl
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.SourceComparison

#guard_msgs (drop info) in
#auditModuleAxioms

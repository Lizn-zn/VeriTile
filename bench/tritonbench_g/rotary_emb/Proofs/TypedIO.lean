import bench.tritonbench_g.rotary_emb.Proofs.SourceRun
import bench.tritonbench_g.rotary_emb.Proofs.SafetyTransport

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Complete typed source, projected to exact real arithmetic, with independent
initial-memory formulas for both output regions and an actual-cell frame. -/
theorem typed_io (q k cos sin : FullSource.Storage) (v : Bool)
    (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState)
    (hsep : RegionsSeparate Q K Cos Sin) (hGeometry : p.Geometry s) :
    ∃ t, exec
      (FullSource.kernel q k cos sin v Q K Cos Sin p.qBatch p.qHead p.qDim
        p.kBatch p.kHead p.kDim p.cosBatch p.cosDim p.sinBatch p.sinDim p.length
        p.headsQ p.headsK p.blockHead p.blockSeq p.dimension).eraseDType.toAlgKernel s = some t ∧
      (∀ odd i, p.activeQ s i →
        t.readMem Q (p.qAddress s odd i) = p.expectedQ s Q Cos Sin odd i) ∧
      (∀ odd i, p.activeK s i →
        t.readMem K (p.kAddress s odd i) = p.expectedK s K Cos Sin odd i) ∧
      BlockState.WriteWithin (p.footprint s Q K) s t := by
  rw [FullSource.typed_run]
  exact source_io p Q K Cos Sin s hsep hGeometry

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath

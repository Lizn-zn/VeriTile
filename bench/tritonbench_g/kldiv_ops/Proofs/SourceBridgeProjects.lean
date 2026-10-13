import bench.tritonbench_g.kldiv_ops.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.Source
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000

/-- Every formal typed profile has a complete algorithm extraction, including both variants.
Frontend admissibility is separately stated by SourceDomain; extraction alone is not admission. -/
theorem forward_default_projects (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardDefault pred target output v Y T O sy st so N B eps reduction).toAlgorithm? =
      .ok (forwardDefault pred target output v Y T O sy st so N B eps reduction).toAlgKernel := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

theorem forward_log_projects (pred : Storage) (target : MathStorage) (output : Storage)
    (v : Bool) (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardLog pred target output v Y T O sy st so N B eps reduction).toAlgorithm? =
      .ok (forwardLog pred target output v Y T O sy st so N B eps reduction).toAlgKernel := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

theorem backward_default_projects (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).toAlgorithm? =
      .ok (backwardDefault target output v T O st so N B).toAlgKernel := by
  cases target <;> cases output <;> cases v <;> rfl

theorem backward_log_projects (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).toAlgorithm? =
      .ok (backwardLog target output v T O st so N B).toAlgKernel := by
  cases target <;> cases output <;> cases v <;> rfl

end VeriTile.Bench.TritonBenchG.KldivOps.Source
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.Source.forward_default_projects
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.Source.forward_log_projects
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.Source.backward_default_projects
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.Source.backward_log_projects

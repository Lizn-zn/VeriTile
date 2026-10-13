import bench.tritonbench_g.rms_norm_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 1000000

@[spec_rule] theorem ptr_index_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (r : RegionName) (a b : Op .nat []) :
    OpEquivalent M s before
      (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil a b))
      (.ptrAdd .nil (.ptrBase r) (.mul .nat .nil b a)) := by
  intro t ht
  have h := CacheTransformTriton.IndexRewrite.index_product_commutes M s before a b t ht
  change (evalOp (engine M) none (.mul .nat .nil a b) t >>= fun y =>
    some (bop (fun (p : RegionName × Nat) (n : Nat) => (p.1,p.2+n))
      .nil (fun _ => (r,0)) y)) = _
  rw [h]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.FPProof

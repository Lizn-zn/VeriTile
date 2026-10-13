import bench.tritonbench_g.relu_strided_buffer.Kernels
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.FPProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite

/-- Exact discrete constant folding; floating operations are never inspected. -/
@[spec_rule] theorem fold_tile_count {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (n B : Nat) :
    OpEquivalent M s before
      (.div .nat .nil (.sub .nat .nil
        (.add .nat .nil (.constNat n) (.constNat B)) (.constNat 1)) (.constNat B))
      (.constNat ((n + B - 1) / B)) := by
  intro t _
  rfl

end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.FPProofs

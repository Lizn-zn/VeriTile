import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Constant folding of the unused tile count in both complete branch bodies.
Default floating precision is fp32 (ProfiledRewrite); the pointer element casts,
opaque comparisons, unspecified block padding, masks and grid-stride iterations
are unchanged. Empty numerical domains suffice because only exact Nat arithmetic
changes. Equivalence preserves the entire state, including every memory cell and
execution failure, for arbitrary symbolic strides, dimensions and launch grids. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (x y : RegionName) (sx sy n tasks tiles B : Nat)

def oneTile : Program where
  kernel := relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []

def oneTileFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_one_tile_surface x y sx sy n tasks tiles B
  domain := []

def gridStride : Program where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []

def gridStrideFolded : Program where
  kernel := Candidate.relu_forward_kernel_rank_1_grid_stride_surface x y sx sy n tasks tiles B
  domain := []

specification one_tile_equiv (_hB : 0 < B) :
    oneTile x y sx sy n tasks tiles B ≡[assumptions]
      oneTileFolded x y sx sy n tasks tiles B := by
  equiv_decompose
  all_goals fp_prove

specification grid_stride_equiv (_hB : 0 < B) :
    gridStride x y sx sy n tasks tiles B ≡[assumptions]
      gridStrideFolded x y sx sy n tasks tiles B := by
  equiv_decompose
  all_goals fp_prove
end

#print_fp_assumptions one_tile_equiv
#print_fp_assumptions grid_stride_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Equivalence

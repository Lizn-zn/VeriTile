import bench.tritonbench_g.destindex_copy_kv1.Proofs.SourceIO
/-! Real correctness of the complete masked multi-head scatter in Kernels.lean.
Destination metadata remains signed. The IO contract requires a nonnegative
row, finite disjoint allocation, actual-access bounds and zero unspecified
values. The scalar metadata load is always in bounds, even for an inactive
payload tile. Both head and column masks are explicit below. Successful flat
execution, independent initial copy values and the full tagged-cell frame are
proved. Input/output names may alias because all reads precede the store.
The collision-free layout condition concerns the full output tile.

Input/output storage independently ranges over fp16/bf16/fp32, erased only
for these real formulas. Host launches, races and fixed-width overflow are
outside this per-program statement. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv1.Correctness
open VeriTile.Triton
open scoped RealIO
noncomputable section

def originalIO (p : Layout) (input output : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source input output Bool.false).eraseDType
  projection := (SourceIO.sourceIO p input output Bool.false).projection
  readMetadata := id
  read := fun pid j => pid*p.stride_k_bs + p.stride_k_h*j.1.val + p.stride_k_d*j.2.1.val
  write := fun row j => row.toNat*p.stride_o_bs + p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val
  active := fun j => j.1.val<p.head_num ∧ j.2.1.val<p.head_dim

def indexCommutedIO (p : Layout) (input output : TypedSource.Storage) : RealIO.IO p :=
  { originalIO p input output with
    kernel := (p.source input output Bool.true).eraseDType
    projection := (SourceIO.sourceIO p input output Bool.true).projection }

def expected {H D : Nat} (xs : RealIO.Inputs H D) : TileIndex [H,D] → ℝ := xs.values

variable (p : Layout) (input output : TypedSource.Storage)
    (hLayout : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val))
include hLayout

specification destindex_copy_kv_correctness :
    Spec.Real (originalIO p input output ⊨ expected) := by
  exact SourceIO.correct p input output Bool.false hLayout

specification index_commuted_correctness :
    Spec.Real (indexCommutedIO p input output ⊨ expected) := by
  exact SourceIO.correct p input output Bool.true hLayout

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopyKv1.Correctness

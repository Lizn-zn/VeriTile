import bench.tritonbench_g.destindex_copy.Kernels
namespace VeriTile.Bench.TritonBenchG.DestindexCopy
open VeriTile.Triton
/-- The complete source signature, including unused head counts and strides. -/
structure Layout where
  KV_nope : RegionName
  KV_rope : RegionName
  Dest_loc : Region .int
  O_nope : RegionName
  O_rope : RegionName
  stride_kv_nope_bs : Nat
  stride_kv_nope_h : Nat
  stride_kv_nope_d : Nat
  stride_kv_rope_bs : Nat
  stride_kv_rope_h : Nat
  stride_kv_rope_d : Nat
  stride_o_nope_bs : Nat
  stride_o_nope_h : Nat
  stride_o_nope_d : Nat
  stride_o_rope_bs : Nat
  stride_o_rope_h : Nat
  stride_o_rope_d : Nat
  kv_nope_head_num : Nat
  kv_rope_head_num : Nat
  BLOCK_DMODEL_NOPE : Nat
  BLOCK_DMODEL_ROPE : Nat

def Layout.source (p : Layout) (nopeType ropeType : TypedSource.Storage) (swapped : Bool) : ComputeKernel :=
  TypedSource.kernel nopeType ropeType swapped p.KV_nope p.KV_rope p.Dest_loc p.O_nope p.O_rope p.stride_kv_nope_bs p.stride_kv_nope_h p.stride_kv_nope_d p.stride_kv_rope_bs p.stride_kv_rope_h p.stride_kv_rope_d p.stride_o_nope_bs p.stride_o_nope_h p.stride_o_nope_d p.stride_o_rope_bs p.stride_o_rope_h p.stride_o_rope_d p.kv_nope_head_num p.kv_rope_head_num p.BLOCK_DMODEL_NOPE p.BLOCK_DMODEL_ROPE
end VeriTile.Bench.TritonBenchG.DestindexCopy

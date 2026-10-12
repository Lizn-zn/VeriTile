import bench.tritonbench_g.destindex_copy_kv2.Kernels
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2
open VeriTile.Triton
structure Layout where
  K : RegionName
  Dest_loc : Region .int
  Out : RegionName
  stride_k_bs : Nat
  stride_k_h : Nat
  stride_k_d : Nat
  stride_o_bs : Nat
  stride_o_h : Nat
  stride_o_d : Nat
  head_num : Nat
  BLOCK_DMODEL : Nat
  BLOCK_HEAD : Nat

def Layout.source (p : Layout) (input output : TypedSource.Storage) (swapped : Bool) : ComputeKernel :=
  TypedSource.kernel input output swapped p.K p.Dest_loc p.Out p.stride_k_bs p.stride_k_h p.stride_k_d p.stride_o_bs p.stride_o_h p.stride_o_d p.head_num p.BLOCK_DMODEL p.BLOCK_HEAD
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2

import bench.tritonbench_g.cache_transform_triton.Kernels

/-! Runtime buffer names, cache strides and symbolic tile dimensions shared
by both cache gathers. The host reuses these strides for both output buffers. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton
open VeriTile.Triton

structure Layout where
  cos : RegionName
  sin : RegionName
  metadata : Region .int
  cosOut : RegionName
  sinOut : RegionName
  rowStride : Nat
  hiddenStride : Nat
  H : Nat
  B : Nat

def Layout.prefill (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (total N : Nat) : ComputeKernel :=
  TypedSource.prefill cosType sinType commuted p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride total p.H N p.B

def Layout.decode (p : Layout) (cosType sinType : TypedSource.Storage)
    (commuted : Bool) (N : Nat) : ComputeKernel :=
  TypedSource.decode cosType sinType commuted p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride p.H N p.B

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CacheTransformTriton

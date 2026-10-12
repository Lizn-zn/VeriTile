import bench.tritonbench_g.bmm_optimized.Proofs.RealSource

/-! All source mask choices remain explicit. Both CTA-ordering branches stay
in the prefix of the actual K loop; masks and pointer updates are not replaced
by precomputed values. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Parameters where
  A : RegionName
  B : RegionName
  O : RegionName
  M : Nat
  N : Nat
  K : Nat
  groupM : Nat
  divM : Bool
  divN : Bool
  divK : Bool

def source (dtype : TypedSource.Storage) (swapped : Bool) (BM BN BK : Nat) (p : Parameters) :=
  TypedSource.specialize dtype p.divM p.divN p.divK swapped p.A p.B p.O p.M p.N p.K BM BN BK p.groupM

def kernel (swapped : Bool) (BM BN BK : Nat) (p : Parameters) :=
  (source .fp32 swapped BM BN BK p).eraseDType

def parts (swapped : Bool) (BM BN BK : Nat) (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (source .fp32 swapped BM BN BK p).surfaceBody "_i").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

def realPart (code : List ComputeStmt) : List Stmt :=
  Stmt.eraseDTypeList (ComputeKernel.mk [] [] code).toAlgKernel.body

theorem storage_erasure (dtype : TypedSource.Storage) (swapped : Bool) (BM BN BK : Nat) (p : Parameters) :
    (source dtype swapped BM BN BK p).eraseDType = kernel swapped BM BN BK p :=
  RealSource.storage_erasure dtype p.divM p.divN p.divK swapped p.A p.B p.O p.M p.N p.K BM BN BK p.groupM

theorem body_split (swapped : Bool) (BM BN BK : Nat) (p : Parameters) :
    (kernel swapped BM BN BK p).toAlgKernel.body =
      realPart (parts swapped BM BN BK p).before ++
        [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1)
          (realPart (parts swapped BM BN BK p).body)] ++
        realPart (parts swapped BM BN BK p).after := by
  rcases p with ⟨A,B,O,M,N,K,G,dm,dn,dk⟩
  cases dm <;> cases dn <;> cases dk <;> cases swapped
  all_goals
    simp [kernel, source, parts, realPart, TypedSource.specialize,
      Source.dynamicRange?, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealProgram

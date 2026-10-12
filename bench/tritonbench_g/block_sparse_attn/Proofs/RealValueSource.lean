import bench.tritonbench_g.block_sparse_attn.Proofs.RealAccumulationSource

/-! The two actual V-load branches. Signed start_n is retained through the
stride product and the upper-bound mask. The cast at the pointer boundary
is exactly the frontend's source lowering, not a nonnegativity assumption.
Neither masked branch supplies an other value. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueSource
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def pointer (BN BD stride : Nat) (second : Bool) : Op .ptr [BN, BD] :=
  let first := Op.ptrAdd .scalarR (.ref .ptr [BN, BD] "v_ptrs")
    (.castIntToNat (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat stride))))
  if second then .ptrAdd .scalarR first (.constNat BD) else first

def mask (BN BD context : Nat) : Op .bool [BN, BD] :=
  .remap [BN, BD] (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
    (.lt .int .scalarR
      (.add .int .scalarR (.castNatToInt (.expandDim 1 (.ref .nat [BN] "offs_n")))
        (.ref .int [] "start_n"))
      (.castNatToInt (.constNat context)))

def code (BN BD stride context : Nat) (even second : Bool) : List Stmt :=
  [.ifThenElse (.constBool even)
    [.assign .real [BN, BD] "v" (.load .real (.ptr (pointer BN BD stride second)) .none)]
    [.assign .real [BN, BD] "v" (.load .real (.ptr (pointer BN BD stride second))
      (.mask (mask BN BD context)))]]

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem first_matches (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealAccumulationSource.firstLoad swapped BM BN BD p) =
      code BN BD p.vRow p.context p.evenN Bool.false := by
  cases swapped
  all_goals
    simp [code, pointer, mask, RealAccumulationSource.firstLoad,
      RealProgram.realPart, RealProgram.accumulation, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    rfl

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem second_matches (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealAccumulationSource.secondLoad swapped BM BN BD p) =
      code BN BD p.vRow p.context p.evenN Bool.true := by
  cases swapped
  all_goals
    simp [code, pointer, mask, RealAccumulationSource.secondLoad,
      RealAccumulationSource.secondBranch, RealAccumulationSource.writesSecond,
      RealProgram.realPart, RealProgram.accumulation, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, Source.conditional?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueSource

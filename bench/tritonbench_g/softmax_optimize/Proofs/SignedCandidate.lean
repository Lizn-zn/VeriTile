import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def indexPrefixCandidate (X : RegionName) (N B : Nat) : List Stmt :=
  [.assign .int [B] "n_offsets"
      (.add .int .scalarL (.ref .int [] "start_n") (.castNatToInt (.arange B))),
   .assign .int [B] "offset"
      (.add .int .scalarL
        (.mul .int .nil (.constInt N) (.castNatToInt (.ref .nat [] "pid_m")))
        (.ref .int [B] "n_offsets")),
   .assign .ptr [B] "input_ptrs"
      (.ptrAdd .scalarL (.ptrBase X) (.castIntToNat (.ref .int [B] "offset")))]

def inputFullCandidate (X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefixCandidate X N B ++ (loopAbody X N B).drop 3

def inputTailCandidate (X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefixCandidate X N B ++ [maskStatement N B] ++ (loopBbody X N B).drop 4

def outputFullCandidate (Y X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefixCandidate X N B ++ ((loopApass2body Y X N B).drop 3).take 3 ++
    [outputPointer Y B] ++ (loopApass2body Y X N B).drop 7

def outputTailCandidate (Y X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefixCandidate X N B ++ [maskStatement N B] ++
    ((loopBpass2body Y X N B).drop 4).take 3 ++
    [outputPointer Y B] ++ (loopBpass2body Y X N B).drop 8

/-- Exact candidate decomposition retains all four signed loops and the helper at zero. -/
theorem candidate_source_body (Y X : RegionName) (M N B : Nat) :
    (mathematical Bool.true Y X M N B).body =
      [.assign .nat [] "pid_m" (.programId 0),
       .assign .real [B] "m" (.full [B] .negInf),
       .assign .real [B] "z" (.full [B] (.const 0)),
       .assign .int [] "prev_multiple" (previousExpression N B),
       .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
         (inputFullCandidate X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B)
         (inputTailCandidate X N B)] ++ reductionStmts B ++
      [.assign .int [] "prev_multiple" (previousExpression N B),
       .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
         (outputFullCandidate Y X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B)
         (outputTailCandidate Y X N B)] := by
  unfold mathematical
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [kernel, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    ComputeDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType, IntegralDType.eraseDType,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport

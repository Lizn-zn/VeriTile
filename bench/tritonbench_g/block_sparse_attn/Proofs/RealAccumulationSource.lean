import bench.tritonbench_g.block_sparse_attn.Proofs.RealProgram

/-! Decomposition of the remaining source body after probability normalization.
The load fragments are taken directly from the source, keeping signed pointer
arithmetic and EVEN_N masks. Both value-dot assignments and carry commits are
checked explicitly; no claim about load success is made by this decomposition. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAccumulationSource
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def valueDot (BM BN BD : Nat) (name : RegName) : Stmt :=
  .assign .real [BM, BD] name
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BD] name)
      (.dot (batch := []) (.ref .real [BM, BN] "p") (.ref .real [BN, BD] "v")))

def commit (BM : Nat) : List Stmt :=
  [.assign .real [BM] "l_i" (.ref .real [BM] "l_i_new"),
   .assign .real [BM] "m_i" (.ref .real [BM] "m_i_new")]

def firstLoad (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  Source.prefixBefore (RealProgram.accumulation swapped BM BN BD p) "acc"

def writesSecond (code : List ComputeStmt) : Bool := code.any fun st =>
  match st with
  | .assign _ _ name _ => name == "acc2"
  | _ => Bool.false

def secondBranch (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : Source.Conditional :=
  (Source.conditional? (RealProgram.accumulation swapped BM BN BD p) writesSecond).getD
    ⟨[], .constBool Bool.false, [], [], []⟩

def secondLoad (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  Source.prefixBefore (secondBranch swapped BM BN BD p).yes "acc2"

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.accumulation swapped BM BN BD p) =
      RealProgram.realPart (firstLoad swapped BM BN BD p) ++
      [valueDot BM BN BD "acc",
        .ifThen (.constBool (decide (2 ≤ p.blocks)))
          (RealProgram.realPart (secondLoad swapped BM BN BD p) ++ [valueDot BM BN BD "acc2"])] ++
      commit BM := by
  cases swapped
  all_goals
    simp [firstLoad, secondLoad, secondBranch, writesSecond, valueDot, commit,
      RealProgram.realPart, RealProgram.accumulation, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, Source.conditional?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAccumulationSource

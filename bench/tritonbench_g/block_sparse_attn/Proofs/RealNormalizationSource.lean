import bench.tritonbench_g.block_sparse_attn.Proofs.RealProgram

/-! The actual normalization segment, from m_ij through the probability cast.
Its explicit AST is checked against the named slice of both complete sources.
The real cast is an identity; the compute source retains the storage cast. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationSource
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def code (BM BN BD blocks : Nat) : List Stmt :=
  [.assign .real [BM] "m_ij"
     (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk")),
   .assign .real [BM, BN] "p"
     (.exp (.sub .real (.consSame (.consR .nil)) (.ref .real [BM, BN] "qk")
       (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "m_ij")))),
   .assign .real [BM] "l_ij"
     (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p")),
   .assign .real [BM] "m_i_new"
     (.where (.gt .real (.consSame .nil) (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij"))
       (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij")),
   .assign .real [BM] "alpha"
     (.exp (.sub .real (.consSame .nil) (.ref .real [BM] "m_i") (.ref .real [BM] "m_i_new"))),
   .assign .real [BM] "beta"
     (.exp (.sub .real (.consSame .nil) (.ref .real [BM] "m_ij") (.ref .real [BM] "m_i_new"))),
   .assign .real [BM] "l_i_new"
     (.add .real (.consSame .nil)
       (.mul .real (.consSame .nil) (.ref .real [BM] "alpha") (.ref .real [BM] "l_i"))
       (.mul .real (.consSame .nil) (.ref .real [BM] "beta") (.ref .real [BM] "l_ij"))),
   .assign .real [BM] "p_scale"
     (.div .real (.consSame .nil) (.ref .real [BM] "beta") (.ref .real [BM] "l_i_new")),
   .assign .real [BM, BN] "p"
     (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BN] "p")
       (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "p_scale"))),
   .assign .real [BM] "acc_scale"
     (.mul .real (.consSame .nil)
       (.div .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "l_i_new"))
       (.ref .real [BM] "alpha")),
   .assign .real [BM, BD] "acc"
     (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
       (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "acc_scale"))),
   .ifThen (.constBool (decide (2 ≤ blocks)))
     [.assign .real [BM, BD] "acc2"
       (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc2")
         (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "acc_scale")))],
   .assign .real [BM, BN] "p" (.ref .real [BM, BN] "p")]

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.normalization swapped BM BN BD p) = code BM BN BD p.blocks := by
  cases swapped
  all_goals
    simp [RealProgram.realPart, RealProgram.normalization, RealProgram.normalizationRest,
      RealProgram.normalizationPrefix, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, code,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationSource

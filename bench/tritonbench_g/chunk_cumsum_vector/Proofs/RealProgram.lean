import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealSource
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealMath

/-! The complete erased source, exposed as statements for the loop proof.
`original_body` checks the connection to Kernels; none of these fragments is
substituted for the full source in a public specification. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram
open VeriTile.Triton
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
structure Layout where
  headStride : Nat
  timeStride : Nat
  channelStride : Nat
  T : Nat
  S : Nat
  BT : Nat
  BS : Nat

structure Launch where
  feature : Nat
  head : Nat

def numChunks (p : Layout) : Nat := (p.T + p.BT - 1) / p.BT

def stopOp (p : Layout) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.T) (.constNat p.BT))
    (.constNat 1)) (.constNat p.BT)

def pointerOp (p : Layout) (r : RegionName) : Op .blockPtr [p.BT, p.BS] :=
  .makeBlockPtrDynOffsets r (.mul .nat .nil (.ref .nat [] "i_bh") (.constNat p.headStride))
    [p.T, p.S] [p.BT, p.BS] [p.timeStride, p.channelStride]
    [.mul .nat .nil (.ref .nat [] "i_t") (.constNat p.BT),
     .mul .nat .nil (.ref .nat [] "i_s") (.constNat p.BS)]

def carryOp (p : Layout) (swapped : Bool) : Op .real [p.BS] :=
  let a := Op.ref .real [p.BS] "b_z"
  let b := Op.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [p.BT, p.BS] "b_s")
  if swapped then .add .real (.consSame .nil) b a
  else .add .real (.consSame .nil) a b

def loopBody (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BT, p.BS] "p_s" (pointerOp p s),
   .assign .blockPtr [p.BT, p.BS] "p_z" (pointerOp p o),
   .assign .real [p.BT, p.BS] "b_s"
     (.load .real (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_s") [0, 1]) .none .unspecified),
   .assign .real [p.BT, p.BS] "b_c"
     (.add .real (.consL (.consSame .nil))
       (.expandDim ⟨0, by simp⟩ (.ref .real [p.BS] "b_z"))
       (.dot (batch := []) (.ref .real [p.BT, p.BT] "m_s") (.ref .real [p.BT, p.BS] "b_s")
         {allowTF32 := some Bool.false})),
   .store .real [p.BT, p.BS] (.blockPtr (.ref .blockPtr [p.BT, p.BS] "p_z") [0, 1])
     (.ref .real [p.BT, p.BS] "b_c") .none,
   .ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
     [.assign .real [p.BS] "b_z" (carryOp p swapped)]]

def initCode (p : Layout) : List Stmt :=
  [.assign .nat [] "i_s" (.programId 0), .assign .nat [] "i_bh" (.programId 1),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .real [p.BT, p.BT] "m_s"
     (.where (.ge .nat (.consR (.consL .nil))
       (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
       (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i")))
       (.broadcast (.const 1.0) [p.BT, p.BT]) (.broadcast (.const 0.0) [p.BT, p.BT])),
   .assign .real [p.BS] "b_z" (.full [p.BS] (.const 0))]

def body (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  initCode p ++ [.forRangeDyn "i_t" (.constNat 0) (stopOp p) (.constNat 1)
    (loopBody p s o swapped)]

theorem source_body (p : Layout) (s o : RegionName) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.body = body p s o swapped := by
  rw [RealSource.storage_erasure]
  cases swapped
  all_goals simp only [RealSource.source, Bool.false_eq_true, ↓reduceIte,
    TypedSource.original, TypedSource.sumLeft, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealProgram

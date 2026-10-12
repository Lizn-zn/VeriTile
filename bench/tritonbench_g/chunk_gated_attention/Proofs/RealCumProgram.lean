import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumSource
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealInit

/-! Exact complete erased cumulative program. The shared triangular-dot helper
is reused only for expression evaluation; there is no cumsum carry or loop in
this source. `source_body` checks every statement against the typed kernel. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev Layout := ChunkCumsumVector.RealProgram.Layout
abbrev Launch := ChunkCumsumVector.RealProgram.Launch

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 2⟩
def chunk (u : BlockState) : Nat := u.pids 1

def baseOp (p : Layout) (swapped : Bool) : Op .nat [] :=
  if swapped then .mul .nat .nil (.constNat p.headStride) (.ref .nat [] "i_bh")
  else .mul .nat .nil (.ref .nat [] "i_bh") (.constNat p.headStride)

def pointerOp (p : Layout) (r : RegionName) (swapped : Bool) : Op .blockPtr [p.BT,p.BS] :=
  .makeBlockPtrDynOffsets r (baseOp p swapped) [p.T,p.S] [p.BT,p.BS]
    [p.timeStride,p.channelStride]
    [.mul .nat .nil (.ref .nat [] "i_t") (.constNat p.BT),
     .mul .nat .nil (.ref .nat [] "i_s") (.constNat p.BS)]

def initCode (p : Layout) : List Stmt :=
  [.assign .nat [] "i_s" (.programId 0),
   .assign .nat [] "i_t" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [p.BT] "o_i" (.arange p.BT),
   .assign .real [p.BT,p.BT] "m_s"
     (.where (.ge .nat (.consR (.consL .nil))
       (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BT] "o_i"))
       (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BT] "o_i")))
       (.broadcast (.const 1.0) [p.BT,p.BT]) (.broadcast (.const 0.0) [p.BT,p.BT]))]

def computeCode (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  [.assign .blockPtr [p.BT,p.BS] "p_s" (pointerOp p s swapped),
   .assign .blockPtr [p.BT,p.BS] "p_o" (pointerOp p o swapped),
   .assign .real [p.BT,p.BS] "b_s"
     (.load .real (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_s") [0,1]) .none .unspecified),
   .assign .real [p.BT,p.BS] "b_o"
     (.dot (batch := []) (.ref .real [p.BT,p.BT] "m_s") (.ref .real [p.BT,p.BS] "b_s")
       {allowTF32 := some Bool.false}),
   .store .real [p.BT,p.BS] (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_o") [0,1])
     (.ref .real [p.BT,p.BS] "b_o") .none]

def body (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  initCode p ++ computeCode p s o swapped

theorem source_body (p : Layout) (s o : RegionName) (swapped : Bool) (dtype out_dtype : FloatDType) :
    (RealCumSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.body = body p s o swapped := by
  rw [RealCumSource.storage_erasure]
  cases swapped
  all_goals simp only [RealCumSource.source, Bool.false_eq_true, ↓reduceIte,
    TypedSource.cumOriginal, TypedSource.cumIndexFirst, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumProgram

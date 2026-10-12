import bench.tritonbench_g.context_attn_bloom.Proofs.LoopKeysProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresProgram
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 1200000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def scorePrefix (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List ComputeStmt :=
  let body := (parts p r swapped scale).body
  Source.prefixBefore (body.drop (Source.prefixBefore body "qk").length) "m_ij"
def scores (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List Stmt :=
  lower (scorePrefix p r swapped scale)

def causalMask (p : Layout) : Op .bool [p.BM,p.BN] :=
  .ge .int Broadcast.nil.consL.consR
    (.add .int .scalarR
      (Op.expandDim ⟨1,by simp⟩ (Op.ref .nat [p.BM] "offs_m")).castNatToInt
      (z "prompt_cache_len"))
    (.add .int .scalarL (z "start_n")
      (Op.expandDim ⟨0,by simp⟩ (Op.ref .nat [p.BN] "offs_n")).castNatToInt)

def scaling (p : Layout) (swapped : Bool) (scale : ℝ) : Op .real [p.BM,p.BN] :=
  if swapped then .mul .real .scalarL (.const scale) (.ref .real [p.BM,p.BN] "qk")
  else .mul .real .scalarR (.ref .real [p.BM,p.BN] "qk") (.const scale)

def code (p : Layout) (swapped : Bool) (scale : ℝ) : List Stmt :=
  [.assign .real [p.BM,p.BN] "qk" (.full [p.BM,p.BN] (.const 0)),
   .assign .real [p.BM,p.BN] "qk"
     (.add .real Broadcast.nil.consSame.consSame (.ref .real [p.BM,p.BN] "qk")
       (.dot (batch := []) (.ref .real [p.BM,p.D] "q") (.ref .real [p.D,p.BN] "k"))),
   .assign .real [p.BM,p.BN] "qk" (scaling p swapped scale),
   .assign .real [p.BM,p.BN] "qk" (.where (causalMask p) (.ref .real [p.BM,p.BN] "qk")
     (.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (100000000.0 : ℝ))) [p.BM,p.BN]))]

theorem source (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    scores p r swapped scale = code p swapped scale := by
  cases swapped
  all_goals simp only [scores,scorePrefix,lower,parts,mathematical,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    List.drop_succ_cons,List.drop_zero,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    code,causalMask,scaling,n,z]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float]
  all_goals rfl
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ScoresProgram

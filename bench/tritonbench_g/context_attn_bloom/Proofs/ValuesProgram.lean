import bench.tritonbench_g.context_attn_bloom.Proofs.LoopKeysProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesProgram
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def valuePrefix (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List ComputeStmt :=
  let body := (parts p r swapped scale).body
  Source.prefixBefore (body.drop (Source.prefixBefore body "off_v").length) "p"
def values (p : Layout) (r : Regions) : List Stmt := lower (valuePrefix p r Bool.false 0)

def tokenView (p : Layout) := Op.expandDim ⟨1,by simp⟩ (Op.ref .int [p.BN] "kv_loc")
def channelView (p : Layout) := Op.expandDim ⟨0,by simp⟩ (Op.ref .nat [p.D] "offs_d")
def offsets (p : Layout) : Op .int [p.BN,p.D] :=
  .add .int Broadcast.nil.consL.consR
    (.add .int .scalarR (.mul .int .scalarR (tokenView p) (Op.constNat p.vRow).castNatToInt)
      (Op.mul .nat .nil (n "cur_kv_head") (.constNat p.vHead)).castNatToInt)
    (Op.mul .nat .scalarR (channelView p) (.constNat p.vDim)).castNatToInt

def mask (p : Layout) : Op .bool [p.BN,p.D] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .int .scalarR
      (.add .int .scalarL (z "start_n")
        (Op.expandDim ⟨1,by simp⟩ (Op.ref .nat [p.BN] "offs_n")).castNatToInt)
      (z "block_end_loc"))
    (.lt .nat .scalarR (channelView p) (.constNat p.headDim))

def code (p : Layout) (r : Regions) : List Stmt :=
  [.assign .int [p.BN,p.D] "off_v" (offsets p),
   .assign .real [p.BN,p.D] "v"
     (.load .real (.region r.value (Op.ref .int [p.BN,p.D] "off_v").castIntToNat)
       (.maskOther (mask p) (.broadcast (.const (0.0 : ℝ)) [p.BN,p.D])))]

theorem shared (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    lower (valuePrefix p r swapped scale) = values p r := by
  cases swapped <;> rfl

theorem source (p : Layout) (r : Regions) : values p r = code p r := by
  simp only [values,valuePrefix,lower,parts,mathematical,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    List.drop_succ_cons,List.drop_zero,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    code,tokenView,channelView,offsets,mask,n,z]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id]
  rfl
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.ValuesProgram

import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterSource
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterMath

/-! Exact real decomposition of both complete inter-subblock sources. Every
pointer, gate-reference load, K iteration, early-return branch and final store
is retained. `source_body` checks this internal AST against all typed storage
combinations; the independent output formula lives in RealInterMath. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram
open VeriTile.Triton RealInterMath
set_option maxHeartbeats 6000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
structure Regions where
  query : RegionName
  key : RegionName
  gate : RegionName
  output : RegionName

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def times (name : RegName) (n : Nat) : Op .nat [] := .mul .nat .nil (.ref .nat [] name) (.constNat n)
def timeOp (p : Layout) (block : RegName) : Op .nat [] :=
  .add .nat .nil (times "i_t" p.BT) (times block p.BC)
def rowCondition (p : Layout) : Op .bool [] := .boolNot (.ge .nat .nil (timeOp p "i_i") (.constNat p.T))
def triangleCondition : Op .bool [] := .boolNot (.le .nat .nil (.ref .nat [] "i_i") (.ref .nat [] "i_j"))
def queryPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BC,p.BK] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.headStride) [p.T,p.K] [p.BC,p.BK]
    [p.timeStride,1] [timeOp p "i_i",times "i_k" p.BK]
def keyPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BK,p.BC] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.headStride) [p.K,p.T] [p.BK,p.BC]
    [1,p.timeStride] [times "i_k" p.BK,timeOp p "i_j"]
def referencePtr (p : Layout) (r : RegionName) : Op .ptr [p.BK] :=
  .ptrAdd .scalarL (.ptrBase r) (.add .nat .scalarL
    (.add .nat .nil (times "i_bh" p.headStride) (.mul .nat .nil (timeOp p "i_i") (.constNat p.K)))
    (.ref .nat [p.BK] "o_k"))
def outputPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BC,p.BC] :=
  .makeBlockPtrDynOffsets r (.mul .nat .nil (times "i_bh" p.T) (.constNat p.BT))
    [p.T,p.BT] [p.BC,p.BC] [p.BT,1] [timeOp p "i_i",times "i_j" p.BC]
def load2 (a b : Nat) (name : RegName) : Op .real [a,b] :=
  .load .real (.blockPtr (.ref .blockPtr [a,b] name) [0,1]) .none .unspecified
def referenceLoad (p : Layout) : Op .real [p.BK] :=
  .load .real (.ptr (.ref .ptr [p.BK] "p_gn"))
    (.maskOther (.ref .bool [p.BK] "m_k") (.broadcast (.const 0) [p.BK]))
def queryGate (p : Layout) (scale : ℝ) : Op .real [p.BC,p.BK] :=
  .mul .real .scalarR (.mul .real (.consSame (.consSame .nil)) (.ref .real [p.BC,p.BK] "b_q")
    (.exp (.sub .real (.consR (.consSame .nil)) (.ref .real [p.BC,p.BK] "b_g")
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BK] "b_gn"))))) (.const scale)
def keyGate (p : Layout) : Op .real [p.BK,p.BC] :=
  .mul .real (.consSame (.consSame .nil)) (.ref .real [p.BK,p.BC] "b_k")
    (.exp (.sub .real (.consSame (.consL .nil))
      (.expandDim ⟨1,by simp⟩ (.ref .real [p.BK] "b_gn")) (.ref .real [p.BK,p.BC] "b_gk")))
def carry (p : Layout) : Op .real [p.BC,p.BC] := .ref .real [p.BC,p.BC] "b_A"
def dot (p : Layout) : Op .real [p.BC,p.BC] :=
  .dot (batch := []) (.ref .real [p.BC,p.BK] "b_qg") (.ref .real [p.BK,p.BC] "b_kg") {}
def initialization (p : Layout) : List Stmt :=
  [.assign .nat [] "i_t" (.programId 0),.assign .nat [] "i_c" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),
   .assign .nat [] "i_i" (.floorDiv .nat .nil (.ref .nat [] "i_c") (.constNat p.NC)),
   .assign .nat [] "i_j" (.mod .nat .nil (.ref .nat [] "i_c") (.constNat p.NC))]
def setup (p : Layout) : List Stmt := [.assign .real [p.BC,p.BC] "b_A" (.full [p.BC,p.BC] (.const 0))]
def iteration (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  [.assign .nat [p.BK] "o_k" (.add .nat .scalarL (times "i_k" p.BK) (.arange p.BK)),
   .assign .bool [p.BK] "m_k" (.lt .nat .scalarR (.ref .nat [p.BK] "o_k") (.constNat p.K)),
   .assign .blockPtr [p.BC,p.BK] "p_q" (queryPtr p r.query),
   .assign .blockPtr [p.BC,p.BK] "p_g" (queryPtr p r.gate),
   .assign .blockPtr [p.BK,p.BC] "p_k" (keyPtr p r.key),
   .assign .blockPtr [p.BK,p.BC] "p_gk" (keyPtr p r.gate),
   .assign .ptr [p.BK] "p_gn" (referencePtr p r.gate),
   .assign .real [p.BK] "b_gn" (referenceLoad p),
   .assign .real [p.BC,p.BK] "b_q" (load2 p.BC p.BK "p_q"),
   .assign .real [p.BC,p.BK] "b_g" (load2 p.BC p.BK "p_g"),
   .assign .real [p.BC,p.BK] "b_qg" (queryGate p scale),
   .assign .real [p.BK,p.BC] "b_k" (load2 p.BK p.BC "p_k"),
   .assign .real [p.BK,p.BC] "b_gk" (load2 p.BK p.BC "p_gk"),
   .assign .real [p.BK,p.BC] "b_kg" (keyGate p),
   .assign .real [p.BC,p.BC] "b_A" (if swapped then .add .real (.consSame (.consSame .nil)) (dot p) (carry p)
     else .add .real (.consSame (.consSame .nil)) (carry p) (dot p))]
def tripCount (p : Layout) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1)) (.constNat p.BK)
def loop (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : Stmt :=
  .forRangeDyn "i_k" (.constNat 0) (tripCount p) (.constNat 1) (iteration p r scale swapped)
def finalization (p : Layout) (r : Regions) : List Stmt :=
  [.assign .blockPtr [p.BC,p.BC] "p_A" (outputPtr p r.output),
   .store .real [p.BC,p.BC] (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A") [0,1]) (carry p) .none]
def body (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  initialization p ++ [.ifThen (rowCondition p) [.ifThen triangleCondition
    (setup p ++ [loop p r scale swapped] ++ finalization p r)]]
def source (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  TypedInter.source qs ks gs swapped out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC

theorem source_body (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel.body = body p r scale swapped := by
  unfold source
  rw [RealInterSource.storage_erasure]
  cases swapped
  all_goals simp only [TypedInter.source, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterProgram

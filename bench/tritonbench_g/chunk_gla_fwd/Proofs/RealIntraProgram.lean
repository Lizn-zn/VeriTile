import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraSource
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraMath

/-! Exact real AST of the complete intra-subblock source. The early return,
unspecified block loads, explicitly zero-masked flat loads, full column loop,
causal where, store mask and K-strided pointer mutations are retained. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram
open VeriTile.Triton RealIntraMath
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
def timeOp (p : Layout) (name : RegName) : Op .nat [] :=
  .add .nat .nil (times "i_t" p.BT) (times name p.BC)
def condition (p : Layout) : Op .bool [] := .boolNot (.ge .nat .nil (timeOp p "i_i") (.constNat p.T))
def rowsOp (p : Layout) : Op .nat [p.BC] := .add .nat .scalarL (timeOp p "i_i") (.arange p.BC)
def offsetsOp (p : Layout) : Op .nat [p.BC] :=
  .add .nat .scalarR (.add .nat .scalarL
    (.mul .nat .nil (times "i_bh" p.T) (.constNat p.BT))
    (.mul .nat .scalarR (rowsOp p) (.constNat p.BT))) (times "i_j" p.BC)
def queryPtr (p : Layout) (r : RegionName) : Op .blockPtr [p.BC,p.BK] :=
  .makeBlockPtrDynOffsets r (times "i_bh" p.headStride) [p.T,p.K] [p.BC,p.BK]
    [p.timeStride,1] [timeOp p "i_i",.constNat 0]
def keyPtr (p : Layout) (r : RegionName) : Op .ptr [p.BK] :=
  .ptrAdd .scalarL (.ptrBase r) (.add .nat .scalarL
    (.add .nat .nil (times "i_bh" p.headStride) (.mul .nat .nil (timeOp p "i_j") (.constNat p.K)))
    (.ref .nat [p.BK] "o_k"))
def load2 (p : Layout) (name : RegName) : Op .real [p.BC,p.BK] :=
  .load .real (.blockPtr (.ref .blockPtr [p.BC,p.BK] name) [0,1]) .none .unspecified
def keyLoad (p : Layout) (name : RegName) : Op .real [p.BK] :=
  .load .real (.ptr (.ref .ptr [p.BK] name))
    (.maskOther (.ref .bool [p.BK] "m_k") (.broadcast (.const 0) [p.BK]))
def product (p : Layout) : Op .real [p.BC,p.BK] :=
  .mul .real (.consSame (.consSame .nil))
    (.mul .real (.consR (.consSame .nil)) (.ref .real [p.BC,p.BK] "b_q")
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BK] "b_k")))
    (.exp (.sub .real (.consR (.consSame .nil)) (.ref .real [p.BC,p.BK] "b_g")
      (.expandDim ⟨0,by simp⟩ (.ref .real [p.BK] "b_gk"))))
def reducedOp (p : Layout) : Op .real [p.BC] := .reduceSum ⟨1,by simp⟩ Bool.false (product p)
def scaledOp (p : Layout) (scale : ℝ) (swapped : Bool) : Op .real [p.BC] :=
  .where (.ge .nat .scalarR (.ref .nat [p.BC] "o_i") (.ref .nat [] "j"))
    (if swapped then .mul .real .scalarL (.const scale) (.ref .real [p.BC] "b_A")
      else .mul .real .scalarR (.ref .real [p.BC] "b_A") (.const scale))
    (.broadcast (.const 0.0) [p.BC])
def outputOffsets (p : Layout) : Op .nat [p.BC] :=
  .add .nat .scalarR (.ref .nat [p.BC] "o_A") (.ref .nat [] "j")
def advance (p : Layout) (name : RegName) : Op .ptr [p.BK] :=
  .ptrAdd .scalarR (.ref .ptr [p.BK] name) (.constNat p.K)
def remaining (p : Layout) : Op .nat [] :=
  .sub .nat .nil (.sub .nat .nil (.constNat p.T) (times "i_t" p.BT)) (times "i_i" p.BC)
def tripCount (p : Layout) : Op .nat [] :=
  .where (.lt .nat .nil (.constNat p.BC) (remaining p)) (.constNat p.BC) (remaining p)
def initialization : List Stmt :=
  [.assign .nat [] "i_t" (.programId 0),.assign .nat [] "i_i" (.programId 1),
   .assign .nat [] "i_bh" (.programId 2),.assign .nat [] "i_j" (.ref .nat [] "i_i")]
def setup (p : Layout) (r : Regions) : List Stmt :=
  [.assign .nat [p.BC] "o_i" (.arange p.BC),.assign .nat [p.BK] "o_k" (.arange p.BK),
   .assign .nat [p.BC] "o_A" (offsetsOp p),
   .assign .bool [p.BK] "m_k" (.lt .nat .scalarR (.ref .nat [p.BK] "o_k") (.constNat p.K)),
   .assign .bool [p.BC] "m_A" (.lt .nat .scalarR (rowsOp p) (.constNat p.T)),
   .assign .blockPtr [p.BC,p.BK] "p_q" (queryPtr p r.query),
   .assign .blockPtr [p.BC,p.BK] "p_g" (queryPtr p r.gate),
   .assign .ptr [p.BK] "p_k" (keyPtr p r.key),.assign .ptr [p.BK] "p_gk" (keyPtr p r.gate),
   .assign .real [p.BC,p.BK] "b_q" (load2 p "p_q"),.assign .real [p.BC,p.BK] "b_g" (load2 p "p_g")]
def iteration (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  [.assign .real [p.BK] "b_k" (keyLoad p "p_k"),.assign .real [p.BK] "b_gk" (keyLoad p "p_gk"),
   .assign .real [p.BC] "b_A" (reducedOp p),.assign .real [p.BC] "b_A" (scaledOp p scale swapped),
   .store .real [p.BC] (.region r.output (outputOffsets p)) (.ref .real [p.BC] "b_A") (.mask (.ref .bool [p.BC] "m_A")),
   .assign .ptr [p.BK] "p_k" (advance p "p_k"),.assign .ptr [p.BK] "p_gk" (advance p "p_gk")]
def loop (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : Stmt :=
  .forRangeDyn "j" (.constNat 0) (tripCount p) (.constNat 1) (iteration p r scale swapped)
def body (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : List Stmt :=
  initialization ++ [.ifThen (condition p) (setup p r ++ [loop p r scale swapped])]
def source (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage) (scale : ℝ) (swapped : Bool) :=
  TypedIntra.source qs ks gs swapped r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK

theorem source_body (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs scale swapped).eraseDType.toAlgKernel.body = body p r scale swapped := by
  unfold source
  rw [RealIntraSource.storage_erasure]
  cases swapped
  all_goals simp only [TypedIntra.source, Bool.false_eq_true, ↓reduceIte,
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
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraProgram

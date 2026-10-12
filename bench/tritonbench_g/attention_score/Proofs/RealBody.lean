import bench.tritonbench_g.attention_score.Proofs.RealArithmetic
import bench.tritonbench_g.attention_score.Proofs.RealLoads
import bench.tritonbench_g.attention_score.Proofs.RealFrame

/-! The entire real-erased iteration: actual Q/M loads, arithmetic and both
pointer advances. Source equalities retain both load branches and all flags.
The separately checked write sets preserve the registers carried by the loop. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealBody
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def loadQuery (B D : Nat) (evenM : Bool) : Stmt :=
  .ifThenElse (.constBool evenM)
    [.assign .real [B,D] "q" (.load .real (.blockPtr (.ref .blockPtr [B,D] "Q_block_ptr") []) .none .unspecified)]
    [.assign .real [B,D] "q" (.load .real (.blockPtr (.ref .blockPtr [B,D] "Q_block_ptr") [0,1]) .none)]

def loadKey (B D : Nat) (evenN : Bool) : Stmt :=
  .ifThenElse (.constBool evenN)
    [.assign .real [D,B] "k" (.load .real (.blockPtr (.ref .blockPtr [D,B] "K_block_ptr") []) .none .unspecified)]
    [.assign .real [D,B] "k" (.load .real (.blockPtr (.ref .blockPtr [D,B] "K_block_ptr") [0,1]) .none)]

def loadPrefix (B D : Nat) (evenM : Bool) : List Stmt :=
  [.assign .nat [] "start_m" (.ref .nat [] "start_m"), loadQuery B D evenM,
    .assign .real [B] "m" (.load .real (.ptr (.ref .ptr [B] "m_ptrs")) .none)]

def suffix (B D : Nat) : List Stmt :=
  [.assign .blockPtr [B,D] "Q_block_ptr" (.advanceBlockPtr (.ref .blockPtr [B,D] "Q_block_ptr") [B,0]),
    .assign .ptr [B] "m_ptrs" (.ptrAdd .scalarR (.ref .ptr [B] "m_ptrs") (.constNat B))]

def body (B D N : Nat) (offset size : Int)
    (window complement evenM evenN candidate : Bool) : List Stmt :=
  loadPrefix B D evenM ++ RealArithmetic.body B D N offset size window complement evenN candidate ++ suffix B D

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)
theorem original_loop :
    (RealArithmetic.loopBody? (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body) =
      some (body BN BD N offset size window complement evenM evenN Bool.false) := by
  subst BM
  simp only [TypedSource.original, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
  simp only [RealArithmetic.loopBody?, body, loadPrefix, suffix, loadQuery,
    RealArithmetic.body, Bool.false_eq_true, ↓reduceIte]
  rfl

theorem candidate_loop :
    (RealArithmetic.loopBody? (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body) =
      some (body BN BD N offset size window complement evenM evenN Bool.true) := by
  subst BM
  simp only [TypedSource.scalarLeft, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
  simp only [RealArithmetic.loopBody?, body, loadPrefix, suffix, loadQuery,
    RealArithmetic.body, Bool.false_eq_true, ↓reduceIte]
  rfl

end

/-- All writes in either arithmetic version; loop metadata and loaded K survive. -/
def arithmeticWrites : List RegName := ["qk","dist","mask","p","o"]

theorem arithmetic_code (B D N : Nat) (offset size : Int)
    (window complement evenN candidate : Bool) :
    RealFrame.Code arithmeticWrites (RealArithmetic.body B D N offset size window complement evenN candidate) := by
  unfold RealArithmetic.body
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.ifThen
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.nil
    | simp [arithmeticWrites]

def prefixWrites : List RegName := ["start_m","q","m"]
theorem prefix_code (B D : Nat) (evenM : Bool) :
    RealFrame.Code prefixWrites (loadPrefix B D evenM) := by
  unfold loadPrefix loadQuery
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.nil
    | simp [prefixWrites]

def suffixWrites : List RegName := ["Q_block_ptr","m_ptrs"]
theorem suffix_code (B D : Nat) : RealFrame.Code suffixWrites (suffix B D) := by
  unfold suffix
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.nil
    | simp [suffixWrites]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.RealBody

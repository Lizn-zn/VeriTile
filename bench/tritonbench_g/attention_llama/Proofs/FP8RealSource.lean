import bench.tritonbench_g.attention_llama.Kernels
import bench.tritonbench_g.attention_llama.Proofs.RealKernel
import bench.tritonbench_g.attention_llama.Proofs.FP8RealLoads
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.TraceRewrite

/-! Connect the complete FP8 sources to their real execution bodies. The
reference real recurrence supplies the unchanged statements; each K/V load
is replaced by an actual int8 load and E5M2 decoding. The source's three
assignments at each conversion site may then be fused, preserving both the
complete execution result and address safety. Decoding is never erased.
This connection does not yet prove the independent attention formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSource
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

structure ByteLoad where
  shape : TileShape
  name : RegName
  bytes : Op .int shape

/-- Locate K/V loads by their destination, preserving the pointer and mask.
The complete-source correspondence below checks that their padding is zero. -/
def byteLoad? : Stmt → Option ByteLoad
  | .assign .real sh name (.load .real (.ptr ptr) (.maskOther mask _)) =>
      if name == "k" || name == "v" then
        some ⟨sh, name, .load .int (.ptr ptr)
          (.maskOther mask (.broadcast (.constInt 0) sh))⟩
      else none
  | _ => none

def lowerLoad (fused : Bool) (st : Stmt) : List Stmt :=
  match byteLoad? st with
  | none => [st]
  | some load =>
      if fused then
        [.assign .real load.shape load.name (.decodeFloat8E5 .real load.bytes)]
      else FP8RealLoads.conversion load.name load.bytes

private theorem lowerLoad_fuse (st : Stmt) (s : BlockState) :
    stepStmts (lowerLoad Bool.false st) s = stepStmts (lowerLoad Bool.true st) s := by
  cases h : byteLoad? st with
  | none => simp [lowerLoad, h]
  | some load =>
    simp only [lowerLoad, h, Bool.false_eq_true, ↓reduceIte]
    exact FP8RealLoads.fuse_conversion ..

private theorem lowerList_fuse (code : List Stmt) (s : BlockState) :
    stepStmts (code.flatMap (lowerLoad Bool.false)) s =
      stepStmts (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [List.flatMap_cons, ProgramRewrite.append, lowerLoad_fuse, ih]

private theorem ref_safe (bounds : RegionBounds) (s : BlockState) (d : TileDType)
    (sh : TileShape) (name : RegName) :
    (Op.ref d sh name).SafeAt bounds s = True := by rw [Op.SafeAt.eq_def]

private theorem decode_safe (bounds : RegionBounds) (s : BlockState)
    (bytes : Op .int shape) :
    @Op.SafeAt .real shape bounds s (.decodeFloat8E5 .real bytes) =
      bytes.SafeAt bounds s := by rw [Op.SafeAt.eq_def]

private theorem conversion_safe (bounds : RegionBounds) (name : RegName)
    (bytes : Op .int shape) (s : BlockState) :
    Stmt.TraceSafeList bounds (FP8RealLoads.conversion name bytes) s ↔
      bytes.SafeAt bounds s := by
  simp only [FP8RealLoads.conversion, Stmt.TraceSafeList, Stmt.TraceSafe,
    decode_safe, ref_safe, true_and, and_true]
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    split <;> try trivial
    split <;> try trivial
    split <;> trivial

private theorem lowerLoad_safe (bounds : RegionBounds) (st : Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (lowerLoad Bool.false st) s ↔
      Stmt.TraceSafeList bounds (lowerLoad Bool.true st) s := by
  cases h : byteLoad? st with
  | none => simp [lowerLoad, h]
  | some load =>
    simp only [lowerLoad, h, Bool.false_eq_true, ↓reduceIte, conversion_safe,
      Stmt.TraceSafeList, Stmt.TraceSafe, decode_safe, and_true]
    cases stepStmt (.assign .real load.shape load.name
      (.decodeFloat8E5 .real load.bytes)) s <;> simp

private theorem lowerList_safe (bounds : RegionBounds) (code : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.false)) s ↔
      Stmt.TraceSafeList bounds (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [List.flatMap_cons, TraceRewrite.append_iff, lowerLoad_safe, lowerLoad_fuse, ih]

/-- All original recurrence statements are kept, including the source's
block-local key mask and its causal direction `key + SP ≤ query`. -/
def loopBody (fused : Bool) (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) : List Stmt :=
  (alLoopPrefixG scale N BM BN BD).flatMap (lowerLoad fused) ++
    .ifThen (.constBool causal) [alCausalWhereStmt SP BM BN] ::
      (alLoopTailG skn svk N BM BN BD candidate).flatMap (lowerLoad fused)

private theorem loop_fuse (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) (s : BlockState) :
    stepStmts (loopBody Bool.false scale skn svk N SP BM BN BD causal candidate) s =
      stepStmts (loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s := by
  simp only [loopBody, ProgramRewrite.append, lowerList_fuse]
  congr 1
  funext t
  simp only [stepStmts, lowerList_fuse]

private theorem loop_safe (bounds : RegionBounds) (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds (loopBody Bool.false scale skn svk N SP BM BN BD causal candidate) s ↔
      Stmt.TraceSafeList bounds (loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s := by
  simp only [loopBody, TraceRewrite.append_iff, lowerList_safe, lowerList_fuse]
  simp only [Stmt.TraceSafeList, lowerList_safe]

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal candidate : Bool)

/-- The full typed program, before projection to real arithmetic. -/
def source : ComputeKernel :=
  if candidate then
    TypedSource.carryFirstFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal
  else
    TypedSource.originalFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal

def body (fused : Bool) : List Stmt :=
  alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
    .ifThen (.constBool causal) [alBnEndCausalStmt SP BN] ::
    .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
      (loopBody fused scale skn svk N SP BM BN BD causal candidate) ::
      alPostLoopG Out soz soh som son H BM BD

/-- The source's real projection retains both int8 loads and decoders. -/
theorem source_body :
    (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel.body =
      body Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate Bool.false := by
  cases candidate <;>
    simp only [source, TypedSource.originalFP8, TypedSource.carryFirstFP8,
      Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
      MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    rfl

theorem projection_supported :
    (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgorithm? =
      .ok (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel := by
  have h : (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).toAlgorithm? =
      .ok (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).toAlgKernel := by
    cases candidate <;> rfl
  simp only [ComputeKernel.eraseDType, h]
  simp

/-- Proof-body normalization for both complete sources, not a new candidate
FP program. It operates only after numerical precision has been erased. -/
def fusedKernel : Kernel where
  inputs := [Q, K, V]
  outputs := [Out]
  body := body Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate Bool.true

theorem fused_execution (s : BlockState) :
    exec (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel s =
      exec (fusedKernel Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate) s := by
  simp only [exec, source_body, fusedKernel, body, ProgramRewrite.append,
    stepStmts, ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (loop_fuse scale skn svk N SP BM BN BD causal candidate)]

/-- Fusion also preserves the safety of intermediate accesses, not just the
final execution result. The byte load is still the memory-reading operation. -/
theorem fused_safety (bounds : RegionBounds) (s : BlockState) :
    (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel.TraceSafe bounds s ↔
      (fusedKernel Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).TraceSafe bounds s := by
  simp only [Kernel.TraceSafe, source_body, fusedKernel, body, TraceRewrite.append_iff,
    Stmt.TraceSafeList, TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _
      (loop_fuse scale skn svk N SP BM BN BD causal candidate)
      (loop_safe bounds scale skn svk N SP BM BN BD causal candidate),
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _
      (loop_fuse scale skn svk N SP BM BN BD causal candidate)]

/-- The erased byte loads, decoders, loops and stores are all supported by
regional-to-flat memory transport. This is checked on the complete source. -/
theorem flatten_supported :
    (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel.FlattenOk := by
  cases candidate <;>
    simp [Kernel.FlattenOk, source_body, body, alPreLoopG, loopBody, alLoopPrefixG,
    alLoopTailG, alCausalWhereStmt, alBnEndCausalStmt, alPostLoopG,
    lowerLoad, byteLoad?, FP8RealLoads.conversion,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  all_goals simp only [Op.FlattenOk.eq_def, true_and, and_true]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSource

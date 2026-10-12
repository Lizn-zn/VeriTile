import bench.tritonbench_g.attention_score.Proofs.RealBody

/-! Full real-erased source decomposition. The prelude keeps grouped-head
addressing, K load checks and scale initialization. The postlude retains the
source's unit inner output stride and NK mask. Both versions share these
fragments and differ only in the loop's scalar multiplication order. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealProgram
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def prelude (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) (evenN : Bool) :
    List Stmt :=
  [Stmt.assign TileDType.nat [] "start_n" (Op.programId 0),
    Stmt.assign TileDType.nat [] "off_hz" (Op.programId 1),
    Stmt.assign TileDType.nat [] "off_z"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat H)),
    Stmt.assign TileDType.nat [] "off_hkv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat H) (Op.constNat H_KV))),
    Stmt.assign TileDType.nat [] "q_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign TileDType.nat [] "k_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_kz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hkv") (Op.constNat stride_kh))),
    Stmt.assign TileDType.ptr [BN] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_hz") (Op.constNat ROUND_CTX))
          (Op.arange BN))),
    Stmt.assign TileDType.real [BN] "o" (Op.full [BN] (Op.const 0)),
    Stmt.assign TileDType.blockPtr [BN, BD] "Q_block_ptr"
      (Op.makeBlockPtrDyn Q (Op.ref TileDType.nat [] "q_offset") [N_CTX, BD] [BN, BD]
        [stride_qm, stride_qk] [0, 0]),
    Stmt.assign TileDType.blockPtr [BD, BN] "K_block_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.ref TileDType.nat [] "k_offset") [BD, NKV_CTX] [BD, BN]
        [stride_kk, stride_kn]
        [Op.constNat 0,
          Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN)]),
    RealBody.loadKey BN BD evenN,
    Stmt.assign TileDType.nat [] "lo" (Op.constNat 0),
    Stmt.assign TileDType.nat [] "hi" (Op.constNat ROUND_CTX),
    Stmt.assign TileDType.real [] "qk_scale" (Op.const sm_scale),
    Stmt.assign TileDType.real [] "qk_scale"
      (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale")
        (Op.const 1.4426950408889634))]

def postlude (Out : RegionName) (BN NKV_CTX stride_oz stride_oh : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "o_offset"
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_z") (Op.constNat stride_oz))
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "off_h") (Op.constNat stride_oh))),
    Stmt.assign TileDType.nat [BN] "o_range"
      (Op.add NumericDType.nat Broadcast.scalarR (Op.arange BN)
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "start_n") (Op.constNat BN))),
    Stmt.assign TileDType.ptr [BN] "o_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "o_offset")
          (Op.ref TileDType.nat [BN] "o_range"))),
    Stmt.store TileDType.real [BN] (MemAccess.ptr (Op.ref TileDType.ptr [BN] "o_ptrs"))
      (Op.ref TileDType.real [BN] "o")
      (MaskOpt.mask
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BN] "o_range")
          (Op.constNat NKV_CTX)))]

def program (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) : List Stmt :=
  prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN ++
    [.forRangeDyn "start_m" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat B)
      (RealBody.body B D N offset size window complement evenM evenN candidate)] ++
    postlude Out B NK soz soh

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_source :
    (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body =
      program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
        offset size scale window complement evenM evenN Bool.false := by
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
  simp only [program, prelude, postlude, RealBody.body, RealBody.loadPrefix,
    RealBody.suffix, RealBody.loadQuery, RealBody.loadKey, RealArithmetic.body,
    Bool.false_eq_true, ↓reduceIte]
  rfl

theorem candidate_source :
    (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body =
      program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
        offset size scale window complement evenM evenN Bool.true := by
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
  simp only [program, prelude, postlude, RealBody.body, RealBody.loadPrefix,
    RealBody.suffix, RealBody.loadQuery, RealBody.loadKey, RealArithmetic.body,
    ↓reduceIte]
  rfl

end

/-- Every prelude statement writes registers only, including the checked and
unchecked K-load alternatives. -/
def writes : List RegName :=
  ["start_n","off_hz","off_z","off_h","off_hkv","q_offset","k_offset",
    "m_ptrs","o","Q_block_ptr","K_block_ptr","k","lo","hi","qk_scale"]

theorem prelude_code (Q K M : RegionName) (scale : ℝ)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (evenN : Bool) :
    RealFrame.Code writes (prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) := by
  unfold prelude RealBody.loadKey
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.nil
    | simp [writes]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.RealProgram

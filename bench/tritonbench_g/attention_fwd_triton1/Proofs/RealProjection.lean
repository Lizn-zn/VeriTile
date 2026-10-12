import bench.tritonbench_g.attention_fwd_triton1.Kernels
import bench.tritonbench_g.attention_fwd_triton1.Proofs.Legacy
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Source connections for the mathematical proof. The inferred fp32 loads
and explicit casts erase to the real transcription for every branch flag.
RealLoop proves the full output and H-snapshot invariant; RealCandidate
connects the scalar-left implementation to that same mathematical result. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealProjection
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2400000

private theorem erased_projection_supported (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgorithm? = .ok k.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h]
  simp

private theorem erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

section
variable (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool)

theorem original_projection_supported :
    (TypedSource.original q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).eraseDType.toAlgorithm? =
    .ok (TypedSource.original q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by rfl)

theorem scalarLeft_projection_supported :
    (TypedSource.scalarLeft q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).eraseDType.toAlgorithm? =
    .ok (TypedSource.scalarLeft q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by rfl)

theorem original_matches_transcription :
    (TypedSource.original q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).eraseDType.toAlgKernel =
    (attention_fwd_kernel_surface q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND).toAlgKernel := by
  rw [erased_kernel _ (by rfl)]
  simp only [TypedSource.original, attention_fwd_kernel_surface,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind]
  simp only [Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  -- Reduce the dynamic-offset lists first. Unfolding the general erasure
  -- equation under a symbolic map lambda would expand all operation cases.
  simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  rfl

/-- The candidate changes only the query-scale assignment in the erased
source body; the two flags and both stores remain present. -/
noncomputable def scalarLeftBody (q k v h o : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : List Stmt :=
  let body := aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND
  body.take 7 ++ [.assign .real [BT, BD] "b_q"
    (.mul .real .scalarL (.const scale) (.ref .real [BT, BD] "b_q"))] ++ body.drop 8

theorem scalarLeft_matches_body :
    (TypedSource.scalarLeft q k v h o s_qh BD 1 s_hh s_ht (NT * BT) scale
      BT BD NT STORE IFCOND).eraseDType.toAlgKernel.body =
    aft1PrologueG BD ++ [.forRangeDyn "i" (.constNat 0) (aft1StopOpG BT NT) (.constNat 1)
      (scalarLeftBody q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)] := by
  rw [erased_kernel _ (by rfl)]
  simp only [TypedSource.scalarLeft, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind]
  simp only [Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    List.map_cons, List.map_nil]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealProjection

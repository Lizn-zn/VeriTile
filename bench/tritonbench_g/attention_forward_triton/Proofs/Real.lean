import bench.tritonbench_g.attention_forward_triton.Proofs.RealSafety
import VeriTile.Triton.Memory.KernelSpec.StreamAttentionReal

/-! Assembly of independent attention correctness, memory safety and the
five-stream flat-memory contract for both erased sources. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealProofs
open VeriTile.Triton
open VeriTile.Examples.AttentionForwardClosedForm
open VeriTile.Examples.AttentionForwardClosedForm renaming
  qTile → exQTile, kTile → exKTile, vTile → exVTile, keyScale → exKeyScale,
  baseOffset → exBaseOffset, mIndex → exMIndex, outOffset → exOutOffset
open scoped StreamMasked3DKernelIO₅
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

section
variable (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)

def originalIO : StreamMasked3DKernelIO₅ :=
  { attnFwdIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := RealExecution.original Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
    projection := RealNormalization.original_projection_supported .. }

def reciprocalIO : StreamMasked3DKernelIO₅ :=
  { originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
      BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks with
    kernel := RealExecution.reciprocal Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE
    projection := RealNormalization.reciprocal_projection_supported .. }

theorem original_flattenOk :
    (RealExecution.original Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  rw [RealExecution.body_split, RealSafety.prologue_eq]
  simp only [RealLoop.attnLoopBody, attnLoopBody, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, List.nil_append]
  simp [attnIOPre22, attnIOPre20, attnIOPreHead, attnIOPreTail9, attnIOPreLoads,
    attnAccAssign, attnStoreStmt, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

end

section
variable (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)

theorem reciprocal_flattenOk :
    (RealExecution.reciprocal Q K V QScale KScale Out stride_qz stride_qh Z H
      BLOCK_M BLOCK_N numKVBlocks HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.FlattenOk := by
  simp only [RealExecution.reciprocal, TypedSource.reciprocal,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind]
  simp [Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]
end

theorem original_correct
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    originalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        if h : j.val % BLOCK_DMODEL < HEAD_ACTIVE then
          attnFwdIOOutSpec p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs
            ((Lane2D.decode j).1, ⟨j.val % BLOCK_DMODEL, h⟩, PUnit.unit)
        else 0 := by
  apply StreamMasked3DKernelIO₅.Implements.of_identityRounding _ rfl
  refine StreamMasked3DKernelIO₅.ImplementsR.intro _ ?_ ?_ ?_
  · exact original_flattenOk Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks
  · intro bounds st xs ys zs ws vs _hx _hy _hz _hw _hv hbr1 hbr2 hbr3 hbr4 hbr5 hbw
    simp only [originalIO, attnFwdIO] at hbr1 hbr2 hbr3 hbr4 hbr5 hbw ⊢
    exact RealSafety.original_traceSafeR bounds Q K V QScale KScale Out stride_qz stride_qh Z H
      HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks st hBN hnum
      (fun t j hm => hbr1 t j hm) (fun t j hm => hbr2 t j hm) (fun t j hm => hbr3 t j hm)
      (fun t j => hbr4 t j trivial) (fun t j => hbr5 t j trivial) (fun j hm => hbw j hm)
  · intro s₀ xs ys zs ws vs hu hx hy hz hw hv
    simp only [originalIO, attnFwdIO] at hx hy hz hw hv ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    -- the stream-pin tile bridges
    have hqTeq := attnIO_qTm_eq s₀ Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hnum hActiveLe xs (fun t j hm => hx t j hm)
    have hkTeq := attnIO_kTm_eq s₀ K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe ys (fun t j hm => hy t j hm)
    have hvTeq := attnIO_vTm_eq s₀ V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe zs (fun t j hm => hz t j hm)
    have hksEq := attnIO_keyScale_eq s₀ QScale KScale BLOCK_M BLOCK_N numKVBlocks hnum hBN ws vs
      (fun t j => hw t j trivial) (fun t j => hv t j trivial)
    obtain ⟨sF, hexec, hval, hframe⟩ := RealExecution.original_exec_correct
      Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE s₀ hBN hnum hActiveLe hHD hundef'
    refine ⟨sF, by simpa only [execR_triv] using hexec, ?_, ?_⟩
    · -- `Out` readback: the streamed closed form on every write-active lane
      intro j hj
      have hjBD : j.val % BLOCK_DMODEL < HEAD_ACTIVE := hj.2
      have hact : exMIndex s₀ BLOCK_M (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1
            < BLOCK_N * numKVBlocks
          ∧ (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val < HEAD_ACTIVE := by
        simp only [exMIndex, Lane2D.decode_row, Lane2D.decode_col]
        exact hj
      have hOj := hval (Lane2D.decode j) hact
      simp only [RealExecution.expected, dif_pos hact.2] at hOj
      have hspec : attentionRealBase2PerKeyScale
            (exQTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (exKTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exVTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exKeyScale s₀ QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
              ⟨(Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val, hact.2⟩, PUnit.unit)
          = attnFwdIOOutSpec (s₀.pids 0) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
              xs ys zs ws vs
              ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
                ⟨j.val % BLOCK_DMODEL, hjBD⟩, PUnit.unit) := by
        unfold attnFwdIOOutSpec
        rw [← hqTeq, ← hkTeq, ← hvTeq, ← hksEq]
        exact attnIO_spec_row_congr _ _ _ _ _ _ _ (fun e' =>
          (qTileMasked_active s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE
            (BLOCK_N * numKVBlocks)
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1, e', PUnit.unit) hact.1).symm)
      rw [show s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
              + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
            = exOutOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL
                (Lane2D.decode j) from by
        simp only [exOutOffset, exBaseOffset, exMIndex, Lane2D.decode_row, Lane2D.decode_col,
          Nat.mul_one]]
      rw [BlockState.readMemAs_real, hOj, dif_pos hjBD, hspec]
      simp [FloatDType.ofReal, RoundingModel.triv]
    · -- the frame: cells outside the write-active `Out` window are untouched
      intro r' o' hcond
      refine hframe r' o' ?_
      intro hr idx hPidx
      rcases hcond with hne | hguard
      · exact absurd hr hne
      · intro heq
        refine hguard (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) ?_ (heq.trans ?_)
        · simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hPidx
        · simp only [exOutOffset, exMIndex, exBaseOffset, Nat.mul_one, Lane2D.encode_div, Lane2D.encode_mod]

theorem reciprocal_correct
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    reciprocalIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨
      fun p₀ _ _ xs ys zs ws vs j =>
        if h : j.val % BLOCK_DMODEL < HEAD_ACTIVE then
          attnFwdIOOutSpec p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs
            ((Lane2D.decode j).1, ⟨j.val % BLOCK_DMODEL, h⟩, PUnit.unit)
        else 0 := by
  apply StreamMasked3DKernelIO₅.Implements.of_identityRounding _ rfl
  refine StreamMasked3DKernelIO₅.ImplementsR.intro _ ?_ ?_ ?_
  · exact reciprocal_flattenOk Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM
      BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks
  · intro bounds st xs ys zs ws vs _hx _hy _hz _hw _hv hbr1 hbr2 hbr3 hbr4 hbr5 hbw
    simp only [reciprocalIO, originalIO, attnFwdIO] at hbr1 hbr2 hbr3 hbr4 hbr5 hbw ⊢
    exact RealSafety.reciprocal_traceSafeR bounds Q K V QScale KScale Out stride_qz stride_qh Z H
      HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks st hBN hnum
      (fun t j hm => hbr1 t j hm) (fun t j hm => hbr2 t j hm) (fun t j hm => hbr3 t j hm)
      (fun t j => hbr4 t j trivial) (fun t j => hbr5 t j trivial) (fun j hm => hbw j hm)
  · intro s₀ xs ys zs ws vs hu hx hy hz hw hv
    simp only [reciprocalIO, originalIO, attnFwdIO] at hx hy hz hw hv ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hu]
    -- the stream-pin tile bridges
    have hqTeq := attnIO_qTm_eq s₀ Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hnum hActiveLe xs (fun t j hm => hx t j hm)
    have hkTeq := attnIO_kTm_eq s₀ K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe ys (fun t j hm => hy t j hm)
    have hvTeq := attnIO_vTm_eq s₀ V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE numKVBlocks hBN hActiveLe zs (fun t j hm => hz t j hm)
    have hksEq := attnIO_keyScale_eq s₀ QScale KScale BLOCK_M BLOCK_N numKVBlocks hnum hBN ws vs
      (fun t j => hw t j trivial) (fun t j => hv t j trivial)
    obtain ⟨sF, hexec, hval, hframe⟩ := RealExecution.reciprocal_exec_correct
      Q K V QScale KScale Out stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE s₀ hBN hnum hActiveLe hHD hundef'
    refine ⟨sF, by simpa only [execR_triv] using hexec, ?_, ?_⟩
    · -- `Out` readback: the streamed closed form on every write-active lane
      intro j hj
      have hjBD : j.val % BLOCK_DMODEL < HEAD_ACTIVE := hj.2
      have hact : exMIndex s₀ BLOCK_M (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1
            < BLOCK_N * numKVBlocks
          ∧ (Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val < HEAD_ACTIVE := by
        simp only [exMIndex, Lane2D.decode_row, Lane2D.decode_col]
        exact hj
      have hOj := hval (Lane2D.decode j) hact
      simp only [RealExecution.expected, dif_pos hact.2] at hOj
      have hspec : attentionRealBase2PerKeyScale
            (exQTile s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (exKTile s₀ K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exVTile s₀ V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (exKeyScale s₀ QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
              ⟨(Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).2.1.val, hact.2⟩, PUnit.unit)
          = attnFwdIOOutSpec (s₀.pids 0) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
              xs ys zs ws vs
              ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1,
                ⟨j.val % BLOCK_DMODEL, hjBD⟩, PUnit.unit) := by
        unfold attnFwdIOOutSpec
        rw [← hqTeq, ← hkTeq, ← hvTeq, ← hksEq]
        exact attnIO_spec_row_congr _ _ _ _ _ _ _ (fun e' =>
          (qTileMasked_active s₀ Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE
            (BLOCK_N * numKVBlocks)
            ((Lane2D.decode (M := BLOCK_M) (N := BLOCK_DMODEL) j).1, e', PUnit.unit) hact.1).symm)
      rw [show s₀.pids 1 / H * stride_qz + s₀.pids 1 % H * stride_qh
              + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
            = exOutOffset s₀ H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL
                (Lane2D.decode j) from by
        simp only [exOutOffset, exBaseOffset, exMIndex, Lane2D.decode_row, Lane2D.decode_col,
          Nat.mul_one]]
      rw [BlockState.readMemAs_real, hOj, dif_pos hjBD, hspec]
      simp [FloatDType.ofReal, RoundingModel.triv]
    · -- the frame: cells outside the write-active `Out` window are untouched
      intro r' o' hcond
      refine hframe r' o' ?_
      intro hr idx hPidx
      rcases hcond with hne | hguard
      · exact absurd hr hne
      · intro heq
        refine hguard (Lane2D.encode (idx.1, idx.2.1, PUnit.unit)) ?_ (heq.trans ?_)
        · simp only [Lane2D.encode_div, Lane2D.encode_mod]
          exact hPidx
        · simp only [exOutOffset, exMIndex, exBaseOffset, Nat.mul_one, Lane2D.encode_div, Lane2D.encode_mod]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealProofs

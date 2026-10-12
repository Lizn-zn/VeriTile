import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutput
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSource
import VeriTile.Triton.Math.MatrixRewrite

/-! Transfer the complete real execution to every typed storage choice and
the candidate. Both source decompositions are checked by Lean; the only
semantic change is the order of a real matrix addition. This module does not
use FP atoms and is not imported by the floating-point proof. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputCandidate
open VeriTile.Triton RealOutput
set_option maxRecDepth 20000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
section

variable (query value gate : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat)

def before : List Stmt :=
  (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body.take 12

def after : List Stmt :=
  (cgfPostLoop v o A s_v_h s_v_t T V BT BV).drop 7

def intraDot (BT BV : Nat) : Op .real [BT,BV] :=
  .dot (batch := []) (.ref .real [BT,BT] "b_A") (.ref .real [BT,BV] "b_v")
    {allowTF32 := some Bool.false}

theorem original_split :
    (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    before q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ++
      .assign .real [BT,BV] "b_o"
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BT,BV] "b_o") (intraDot BT BV)) ::
      after v o A s_v_h s_v_t T V BT BV := by
  rw [RealOutputSource.storage_erasure]
  change (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body = _
  rw [before,cgf_body_eq]
  rfl

theorem candidate_split :
    (TypedOutput.source query value gate Bool.true dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel.body =
    before q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ++
      .assign .real [BT,BV] "b_o"
        (.add .real (.consSame (.consSame .nil)) (intraDot BT BV) (.ref .real [BT,BV] "b_o")) ::
      after v o A s_v_h s_v_t T V BT BV := by
  rw [RealOutputSource.storage_erasure,before,cgf_body_eq]
  simp only [TypedOutput.source, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.eraseDType,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.map_cons,
    List.map_nil,eraseDType_float,VeriTile.Triton.eraseDType]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType,eraseDType_float,VeriTile.Triton.eraseDType]
  all_goals rfl

theorem candidate_executes_like (s t : BlockState)
    (he : exec (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel s = some t) :
    exec (TypedOutput.source query value gate Bool.true dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel s = some t := by
  rw [exec,original_split] at he
  rw [exec,candidate_split]
  exact MatrixRewrite.add_commute _ _ _ _ _ s t he

theorem source_executes (swapped : Bool) (s : BlockState)
    (hu : s.undef = fun _ _ => 0)
    (hInj : Function.Injective (outOffset s s_v_h s_v_t BT BV)) :
    ∃ sf, exec (TypedOutput.source query value gate swapped dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel s = some sf ∧
      (∀ idx : TileIndex [BT,BV],
        (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) →
        sf.readMem o (outOffset s s_v_h s_v_t BT BV idx) =
          cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV idx.1.val idx.2.1.val) ∧
      BlockState.WriteWithin (footprint o s_v_h s_v_t T V BT BV s) s sf := by
  obtain ⟨sf,he,ho,hf⟩ := RealOutput.source_executes q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV s hu hInj
  have ht : exec (TypedOutput.source query value gate Bool.false dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV).eraseDType.toAlgKernel s = some sf := by
    rw [exec,RealOutputSource.storage_erasure]
    exact he
  refine ⟨sf,?_,ho,hf⟩
  cases swapped
  · exact ht
  · exact candidate_executes_like query value gate dtype out_dtype q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV s sf ht
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputCandidate

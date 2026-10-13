import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileStep
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneTile_body_io (s : BlockState) (Y X : RegionName) (M N K B C row kbase : Nat)
    (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B)
    (hrow : s.regs .nat [] "pid_m" = some (Tile.scalar row))
    (hk : s.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩) :
    ∃ final, stepStmts (oneTileBody Y X M N K B C) s = some final ∧
      BlockState.WriteWithin (outputFootprint Y row N K kbase C) s final ∧
      (∀ n k, n < N → k < C → kbase + k < K →
        final.readMem Y (row * N * K + n * K + (kbase + k)) = expected s X row N K n (kbase + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  obtain ⟨mid, hexec, hmem, hpids, hnum, hundef⟩ := oneTile_step Y X M N K B C row kbase hB s hrow hk
  let val : TileIndex [B, C] → ℝ := storeValue s X row N K kbase 0
    (batchMax s X row N K kbase B C hB) (batchSum s X row N K kbase B C hB)
  let final := scatter mid Y (tileOffset row N K kbase 0) val (tileActive N K kbase 0)
  have hframe : BlockState.WriteWithin (outputFootprint Y row N K kbase C) mid final := by
    apply scatter_frame
    intro i hi
    have hi' : i.1.val < N ∧ kbase + i.2.1.val < K := by simpa [tileActive] using hi
    exact ⟨rfl, i.1.val, i.2.1.val, hi'.1, i.2.1.isLt, hi'.2, by simp [tileOffset]⟩
  have hlaunch := scatter_launch_padding mid Y (tileOffset row N K kbase 0) val (tileActive N K kbase 0)
  refine ⟨final, hexec, ?_, ?_, (scatter_pids _ _ _ _ _).trans hpids,
    hlaunch.1.trans hnum, hlaunch.2.trans hundef⟩
  · intro r o ho
    rw [← hmem]
    exact hframe r o ho
  · intro n k hn hkC hkK
    let ix : TileIndex [B, C] := (⟨n, Nat.lt_of_lt_of_le hn hcover⟩, ⟨k, hkC⟩, PUnit.unit)
    have hi : tileActive N K kbase 0 ix = Bool.true := by simpa [tileActive, ix] using And.intro hn hkK
    have hvalue := scatter_value mid Y row N K kbase 0 val ix hi
    obtain ⟨m, hm, hz⟩ := batch_denominator s X row N K kbase B C hN hB hcover ⟨k, hkC⟩ hkK
    have hv : val ix = expected s X row N K n (kbase + k) := by
      unfold val storeValue score
      simp only [ix, Nat.zero_add, if_pos (And.intro hn hkK), hm, hz]
      exact stored_value _ _ _
    simpa only [final, tileOffset, ix, Nat.zero_add] using hvalue.trans hv

def oneTilePrefix (C : Nat) : List Stmt := RealCandidate.indexPrefix .forwardNonInner Bool.false C

theorem oneTile_prefix_run (s : BlockState) (C : Nat) :
    ∃ st, stepStmts (oneTilePrefix C) s = some st ∧ st.mem = s.mem ∧ st.pids = s.pids ∧
      st.numPids = s.numPids ∧ st.undef = s.undef ∧
      st.regs .nat [] "pid_m" = some (Tile.scalar (s.pids 0)) ∧
      st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ s.pids 1 * C + i.1.val⟩ := by
  simp [oneTilePrefix, RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.add, NumericDType.mul]
  exact ⟨rfl, rfl⟩

theorem oneTile_decomposition (Y X : RegionName) (M N K B C : Nat) :
    ∃ no : List Stmt,
      (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.true).eraseDType.toAlgKernel.body =
        oneTilePrefix C ++ [.ifThenElse (.constBool Bool.true) (oneTileBody Y X M N K B C) no] := by
  let no := match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.true).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
    | some (.ifThenElse _ _ no) => no
    | _ => []
  refine ⟨no, ?_⟩
  simp [oneTilePrefix, RealCandidate.indexPrefix, oneTileBody, no, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

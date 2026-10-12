import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealExecution

/-! Successful real execution of both complete typed source kernels, from the
initial state through END and M/L/Out stores. Results are independent formulas
over the initial memory. This is region-memory correctness; allocation bounds
and flat-memory transport still belong in the public IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecutionSource
open VeriTile.Triton RealExecution
noncomputable section

section
variable (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int)


set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem original_executes (s : BlockState) (hN : 0 < BLOCK_N)
    (hflag : IS_EVEN_N ≠ 0 → NKV_CTX % BLOCK_N = 0) :
    let p : RealPrologue.Parameters :=
      { heads := H, kvHeads := H_KV,
        strides := ⟨stride_qz, stride_qh, stride_kz, stride_kh, stride_vz, stride_vh, stride_oz, stride_oh⟩,
        layout :=
          { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
            queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
            qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
            vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on },
        scale := sm_scale, init := INIT, evenM := IS_EVEN_M, evenN := IS_EVEN_N,
        window := SLIDING_WINDOW, complement := COMPLEMENT_SLIDING_WINDOW,
        offset := sliding_window_offset, size := sliding_window_size }
    OutputLayout BLOCK_M BLOCK_DMODEL p s →
    ∃ sf, stepStmts (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body s = some sf ∧
      Results BLOCK_M BLOCK_N BLOCK_DMODEL END p s sf := by
  dsimp only
  intro hlayout
  apply completes BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.false s _ hlayout
  · exact RealEpilogueSource.original_epilogue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  · exact RealPrologueSource.original_computation Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size s hN hflag

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_executes (s : BlockState) (hN : 0 < BLOCK_N)
    (hflag : IS_EVEN_N ≠ 0 → NKV_CTX % BLOCK_N = 0) :
    let p : RealPrologue.Parameters :=
      { heads := H, kvHeads := H_KV,
        strides := ⟨stride_qz, stride_qh, stride_kz, stride_kh, stride_vz, stride_vh, stride_oz, stride_oh⟩,
        layout :=
          { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
            queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
            qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
            vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on },
        scale := sm_scale, init := INIT, evenM := IS_EVEN_M, evenN := IS_EVEN_N,
        window := SLIDING_WINDOW, complement := COMPLEMENT_SLIDING_WINDOW,
        offset := sliding_window_offset, size := sliding_window_size }
    OutputLayout BLOCK_M BLOCK_DMODEL p s →
    ∃ sf, stepStmts (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body s = some sf ∧
      Results BLOCK_M BLOCK_N BLOCK_DMODEL END p s sf := by
  dsimp only
  intro hlayout
  apply completes BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.true s _ hlayout
  · exact RealEpilogueSource.candidate_epilogue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  · exact RealPrologueSource.candidate_computation Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size s hN hflag

end

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecutionSource

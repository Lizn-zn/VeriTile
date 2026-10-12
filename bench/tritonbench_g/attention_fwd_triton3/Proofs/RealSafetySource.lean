import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSafety

/-! Both complete typed source kernels satisfy real trace safety under the
active-address contract. The source decomposition retains every statement,
including optional carry loads, the signed loop and all END-dependent stores.
Allocation windows must still supply these bounds in the final flat IO proof. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetySource
open VeriTile.Triton RealPrologueSource RealEpilogueSource

theorem body_split (k : ComputeKernel) (BM BN BD finish : Nat) (p : RealPrologue.Parameters) (reciprocal : Bool)
    (hk : k.toAlgorithm? = .ok k.toAlgKernel)
    (hshape : throughLoop k.toAlgKernel.body =
      beforeLoop k.toAlgKernel.body ++ (RealLoopSource.loop? "start_n" k.toAlgKernel.body).toList)
    (hbefore : beforeLoop k.eraseDType.toAlgKernel.body = RealPrologue.code BM BN BD p)
    (hloop : RealLoopSource.loop? "start_n" k.eraseDType.toAlgKernel.body =
      some (RealLoopSource.statement BM BN BD p.layout.kvContext p.window p.complement p.evenN p.offset p.size))
    (hafter : afterLoop k.eraseDType.toAlgKernel.body = RealEpilogue.code BM BD finish reciprocal) :
    k.eraseDType.toAlgKernel.body = RealSafety.code BM BN BD finish p reciprocal := by
  rw [← through_after_loop k.eraseDType.toAlgKernel.body, erased_computation k hk hshape,
    hbefore, hloop, hafter]
  rfl

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
theorem original_safe (s : BlockState) (bounds : RegionBounds) (hN : 0 < BLOCK_N)
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
    RealSafety.Bounds BLOCK_M BLOCK_N BLOCK_DMODEL END p s bounds →
    (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.TraceSafe bounds s := by
  dsimp only
  intro hb
  unfold Kernel.TraceSafe
  rw [body_split _ BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.false (by rfl) (by rfl)
    (RealPrologueSource.original_prologue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)
    (RealLoopSource.original_loop_source Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)
    (RealEpilogueSource.original_epilogue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)]
  exact RealSafety.complete_safe BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.false s bounds hN hflag hb

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_safe (s : BlockState) (bounds : RegionBounds) (hN : 0 < BLOCK_N)
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
    RealSafety.Bounds BLOCK_M BLOCK_N BLOCK_DMODEL END p s bounds →
    (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.TraceSafe bounds s := by
  dsimp only
  intro hb
  unfold Kernel.TraceSafe
  rw [body_split _ BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.true (by rfl) (by rfl)
    (RealPrologueSource.candidate_prologue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)
    (RealLoopSource.candidate_loop_source Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)
    (RealEpilogueSource.candidate_epilogue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size)]
  exact RealSafety.complete_safe BLOCK_M BLOCK_N BLOCK_DMODEL END _ Bool.true s bounds hN hflag hb

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealSafetySource

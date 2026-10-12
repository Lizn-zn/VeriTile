import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealPrologue

/-! The entire source prefix before the signed loop equals the executed
prologue. The decomposition checks order and contiguity, not just the presence
of individual assignments or branches. Both source implementations are checked. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologueSource
open VeriTile.Triton

def beforeLoop : List Stmt → List Stmt
  | .forRangeInt _ _ _ _ _ :: _ => []
  | st :: rest => st :: beforeLoop rest
  | [] => []

def throughLoop : List Stmt → List Stmt
  | (.forRangeInt name lo hi stride body) :: _ => [.forRangeInt name lo hi stride body]
  | st :: rest => st :: throughLoop rest
  | [] => []

def pieces (body : List Stmt) : List Stmt :=
  RealHeaderSource.leadingAssignments body ++ (RealInitSource.branch? "m_i" body).toList ++
    RealInitSource.assignments "qk_scale" body ++ (RealInitSource.branch? "q" body).toList ++
    (RealWindowSource.window? body).toList

theorem before_loop_erase (body : List Stmt) :
    beforeLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (beforeLoop body) := by
  induction body with
  | nil => simp [beforeLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [beforeLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem through_loop_erase (body : List Stmt) :
    throughLoop (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (throughLoop body) := by
  induction body with
  | nil => simp [throughLoop, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [throughLoop, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem option_list_erase (st : Option Stmt) :
    (st.map Stmt.eraseDType).toList = Stmt.eraseDTypeList st.toList := by
  cases st <;> simp [Stmt.eraseDTypeList]

theorem pieces_erase (body : List Stmt) :
    pieces (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (pieces body) := by
  simp only [pieces, RealHeaderSource.prefix_erase, RealInitSource.branch_erase,
    RealInitSource.assignments_erase, RealWindowSource.window_erase, option_list_erase,
    RealHeaderSource.erase_append]

theorem erased_pieces (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel)
    (hp : beforeLoop k.toAlgKernel.body = pieces k.toAlgKernel.body) :
    beforeLoop k.eraseDType.toAlgKernel.body = pieces k.eraseDType.toAlgKernel.body := by
  rw [RealInitSource.erased_body k h]
  exact ((before_loop_erase _).trans (congrArg Stmt.eraseDTypeList hp)).trans (pieces_erase _).symm

theorem erased_computation (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel)
    (hp : throughLoop k.toAlgKernel.body =
      beforeLoop k.toAlgKernel.body ++ (RealLoopSource.loop? "start_n" k.toAlgKernel.body).toList) :
    throughLoop k.eraseDType.toAlgKernel.body = beforeLoop k.eraseDType.toAlgKernel.body ++
      (RealLoopSource.loop? "start_n" k.eraseDType.toAlgKernel.body).toList := by
  rw [RealInitSource.erased_body k h, through_loop_erase, hp]
  simp only [before_loop_erase, RealLoopSource.loop_erase, option_list_erase, RealHeaderSource.erase_append]

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
theorem original_prologue :
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
    beforeLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealPrologue.code BLOCK_M BLOCK_N BLOCK_DMODEL p := by
  dsimp only
  have hs : beforeLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      pieces (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body := by rfl
  have he := erased_pieces _ (by rfl) hs
  have hh := RealHeaderSource.original_prefix Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  have hi := RealInitSource.original_fragments Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  have hw := RealWindowSource.original_window Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  rw [pieces, hh, hi.1, hi.2.1, hi.2.2, hw] at he
  exact he

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_prologue :
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
    beforeLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealPrologue.code BLOCK_M BLOCK_N BLOCK_DMODEL p := by
  dsimp only
  have hs : beforeLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      pieces (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body := by rfl
  have he := erased_pieces _ (by rfl) hs
  have hh := RealHeaderSource.candidate_prefix Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  have hi := RealInitSource.candidate_fragments Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  have hw := RealWindowSource.candidate_window Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size
  rw [pieces, hh, hi.1, hi.2.1, hi.2.2, hw] at he
  exact he

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem original_computation (s : BlockState) (hN : 0 < BLOCK_N)
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
    let lp := RealPrologue.loopParameters BLOCK_M BLOCK_N BLOCK_DMODEL p s
    ∃ s', stepStmts (throughLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body) s = some s' ∧
      RealLoop.Registers lp (RealLoopContext.fullResult s lp (RealPrologue.seed BLOCK_M BLOCK_DMODEL p s))
        (RealLoop.count lp) s' ∧ RealScores.Frame (RealPrologue.writes ++ RealLoop.writes) s s' ∧
      RealPrologue.OutputRegisters BLOCK_M BLOCK_DMODEL p s s' := by
  dsimp only
  have hshape : throughLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      beforeLoop (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body ++
      (RealLoopSource.loop? "start_n" (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body).toList := by rfl
  have hcomp := erased_computation _ (by rfl) hshape
  rw [original_prologue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size, RealLoopSource.original_loop_source Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size] at hcomp
  rw [hcomp]
  exact RealPrologue.through_loop_executes s BLOCK_M BLOCK_N BLOCK_DMODEL _ hN hflag

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_computation (s : BlockState) (hN : 0 < BLOCK_N)
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
    let lp := RealPrologue.loopParameters BLOCK_M BLOCK_N BLOCK_DMODEL p s
    ∃ s', stepStmts (throughLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body) s = some s' ∧
      RealLoop.Registers lp (RealLoopContext.fullResult s lp (RealPrologue.seed BLOCK_M BLOCK_DMODEL p s))
        (RealLoop.count lp) s' ∧ RealScores.Frame (RealPrologue.writes ++ RealLoop.writes) s s' ∧
      RealPrologue.OutputRegisters BLOCK_M BLOCK_DMODEL p s s' := by
  dsimp only
  have hshape : throughLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      beforeLoop (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body ++
      (RealLoopSource.loop? "start_n" (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body).toList := by rfl
  have hcomp := erased_computation _ (by rfl) hshape
  rw [candidate_prologue Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size, RealLoopSource.candidate_loop_source Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size] at hcomp
  rw [hcomp]
  exact RealPrologue.through_loop_executes s BLOCK_M BLOCK_N BLOCK_DMODEL _ hN hflag

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealPrologueSource

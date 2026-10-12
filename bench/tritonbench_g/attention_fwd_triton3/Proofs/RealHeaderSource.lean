import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealPointers

/-! Check the complete leading assignment sequence, from program_id through
l_ptrs, in both source kernels. The selector stops at the first control-flow
statement, retaining every assignment in order without positional indices. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeaderSource
open VeriTile.Triton

def leadingAssignments : List Stmt → List Stmt
  | .assign dt shape name value :: rest => .assign dt shape name value :: leadingAssignments rest
  | _ => []

theorem prefix_erase (body : List Stmt) :
    leadingAssignments (Stmt.eraseDTypeList body) = Stmt.eraseDTypeList (leadingAssignments body) := by
  induction body with
  | nil => simp [leadingAssignments, Stmt.eraseDTypeList]
  | cons st rest ih => cases st <;> simp [leadingAssignments, Stmt.eraseDTypeList, Stmt.eraseDType, ih]

theorem erased_prefix (k : ComputeKernel) (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    leadingAssignments k.eraseDType.toAlgKernel.body = Stmt.eraseDTypeList (leadingAssignments k.toAlgKernel.body) := by
  rw [RealInitSource.erased_body k h]
  exact prefix_erase _

theorem erase_append (a b : List Stmt) :
    Stmt.eraseDTypeList (a ++ b) = Stmt.eraseDTypeList a ++ Stmt.eraseDTypeList b := by
  induction a with
  | nil => simp [Stmt.eraseDTypeList]
  | cons st rest ih => simp [Stmt.eraseDTypeList, ih]

theorem coordinates_erase (heads kvHeads : Nat) (p : RealHeader.Strides) :
    Stmt.eraseDTypeList (RealHeader.code heads kvHeads p) = RealHeader.code heads kvHeads p := by
  simp only [RealHeader.code, RealHeader.coordinateCode, RealHeader.offsetCode, RealHeader.offsetExpr,
    List.cons_append, List.nil_append, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, IntegralDType.eraseDType, VeriTile.Triton.eraseDType]

theorem dynamic_rows_erase (region : RegionName) (name : RegName)
    (BM BD context strideRow strideDim : Nat) :
    (Op.makeBlockPtrDynOffsets region (.ref .nat [] name) [context, BD] [BM, BD]
      [strideRow, strideDim] [RealPointers.rowOffsetExpr BM, .constNat 0]).eraseDType =
    .makeBlockPtrDynOffsets region (.ref .nat [] name) [context, BD] [BM, BD]
      [strideRow, strideDim] [RealPointers.rowOffsetExpr BM, .constNat 0] := by
  rw [Op.eraseDType.eq_def]
  change Op.makeBlockPtrDynOffsets region (Op.ref .nat [] name).eraseDType [context, BD] [BM, BD]
    [strideRow, strideDim] [(RealPointers.rowOffsetExpr BM).eraseDType, (Op.constNat 0).eraseDType] = _
  simp only [RealPointers.rowOffsetExpr, Op.eraseDType.eq_def, NumericDType.eraseDType]
  rfl

theorem pointers_erase (BM BN BD : Nat) (p : RealPointers.Layout) :
    Stmt.eraseDTypeList (RealPointers.code BM BN BD p) = RealPointers.code BM BN BD p := by
  simp only [RealPointers.code, Stmt.eraseDTypeList, Stmt.eraseDType, RealPointers.queryExpr, RealPointers.outputExpr]
  rw [dynamic_rows_erase, dynamic_rows_erase]
  simp only [RealPointers.valueExpr, RealPointers.keyExpr, RealPointers.rowPointerExpr, RealPointers.rowOffsetExpr,
    Op.eraseDType.eq_def, NumericDType.eraseDType, VeriTile.Triton.eraseDType]
  rfl

theorem code_erase (BM BN BD heads kvHeads : Nat) (p : RealHeader.Strides) (m : RealPointers.Layout) :
    Stmt.eraseDTypeList (RealHeader.code heads kvHeads p ++ RealPointers.code BM BN BD m) =
      RealHeader.code heads kvHeads p ++ RealPointers.code BM BN BD m := by
  rw [erase_append, coordinates_erase, pointers_erase]

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
theorem original_prefix :
    let strides : RealHeader.Strides := ⟨stride_qz, stride_qh, stride_kz, stride_kh,
      stride_vz, stride_vh, stride_oz, stride_oh⟩
    let layout : RealPointers.Layout :=
      { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
        queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
        qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
        vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on }
    leadingAssignments (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealHeader.code H H_KV strides ++ RealPointers.code BLOCK_M BLOCK_N BLOCK_DMODEL layout := by
  dsimp only
  have h : leadingAssignments (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      RealHeader.code H H_KV ⟨stride_qz, stride_qh, stride_kz, stride_kh, stride_vz, stride_vh, stride_oz, stride_oh⟩ ++
      RealPointers.code BLOCK_M BLOCK_N BLOCK_DMODEL
        { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
          queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
          qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
          vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on } := by rfl
  exact (erased_prefix _ (by rfl)).trans ((congrArg Stmt.eraseDTypeList h).trans (code_erase _ _ _ _ _ _ _))

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_prefix :
    let strides : RealHeader.Strides := ⟨stride_qz, stride_qh, stride_kz, stride_kh,
      stride_vz, stride_vh, stride_oz, stride_oh⟩
    let layout : RealPointers.Layout :=
      { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
        queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
        qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
        vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on }
    leadingAssignments (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      RealHeader.code H H_KV strides ++ RealPointers.code BLOCK_M BLOCK_N BLOCK_DMODEL layout := by
  dsimp only
  have h : leadingAssignments (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).toAlgKernel.body =
      RealHeader.code H H_KV ⟨stride_qz, stride_qh, stride_kz, stride_kh, stride_vz, stride_vh, stride_oz, stride_oh⟩ ++
      RealPointers.code BLOCK_M BLOCK_N BLOCK_DMODEL
        { Q := Q, K := K, V := V, M := M, Out := Out, L := L,
          queryContext := N_CTX, roundedContext := ROUND_CTX, kvContext := NKV_CTX,
          qRow := stride_qm, qDim := stride_qk, kDim := stride_kk, kRow := stride_kn,
          vRow := stride_vk, vDim := stride_vn, oRow := stride_om, oDim := stride_on } := by rfl
  exact (erased_prefix _ (by rfl)).trans ((congrArg Stmt.eraseDTypeList h).trans (code_erase _ _ _ _ _ _ _))

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealHeaderSource

import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealExpressions
import VeriTile.Triton.Float.Erasure

/-! Named parts of the precision-erased source loop. The source equalities
below check the complete body of both kernels, including tail broadcasts,
NaN-repair syntax, probability cast erasure, loads and pointer advances.
These are internal execution helpers, not an alternative kernel specification. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealBody
open VeriTile.Triton

def tailMask (BM BN context : Nat) : Op .bool [BM, BN] :=
  .remap [BM, BN] (fun i => (⟨0, by omega⟩, i.2.1, PUnit.unit))
    (.expandDim ⟨0, by simp⟩
      (.lt .int .scalarR
        (.add .int .scalarR (.castNatToInt (.arange BN)) (.ref .int [] "start_n"))
        (.castNatToInt (.constNat context))))

def scores (BM BN BD context window complement even : Nat) (offset size : Int) : List Stmt :=
  [.assign .real [BM, BN] "qk" (.full [BM, BN] (.const 0)),
   .assign .real [BM, BN] "qk" (.add .real (.consSame (.consSame .nil))
     (.ref .real [BM, BN] "qk")
     (.dot (batch := []) (.ref .real [BM, BD] "q") (.ref .real [BD, BN] "k"))),
   .assign .real [BM, BN] "qk" (.mul .real .scalarR
     (.ref .real [BM, BN] "qk") (.ref .real [] "qk_scale")),
   .ifThen (.ne .nat .nil (.constNat window) (.constNat 0))
     [.assign .int [BM, BN] "dist" (RealExpressions.distanceExpr BM BN offset),
      RealExpressions.maskStmt BM BN complement size,
      .assign .real [BM, BN] "qk" (.where (.ref .bool [BM, BN] "mask")
        (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))],
   .ifThen (.eq .nat .nil (.constNat even) (.constNat 0))
     [.assign .real [BM, BN] "qk" (.where (tailMask BM BN context)
       (.ref .real [BM, BN] "qk") (.broadcast .negInf [BM, BN]))]]

def probabilities (BM BN context window even : Nat) : List Stmt :=
  [.assign .real [BM] "m_ij"
     (.where (.gt .real (.consSame .nil) (.ref .real [BM] "m_i")
       (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk")))
       (.ref .real [BM] "m_i")
       (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "qk"))),
   .assign .real [BM, BN] "qk" (.sub .real (.consSame (.consR .nil))
     (.ref .real [BM, BN] "qk") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "m_ij"))),
   .assign .real [BM, BN] "p" (.exp2 (.ref .real [BM, BN] "qk")),
   .ifThen (.ne .nat .nil (.constNat window) (.constNat 0))
     [.assign .real [BM, BN] "p" (.where (.ref .bool [BM, BN] "mask")
       (.ref .real [BM, BN] "p") (.broadcast (.const 0.0) [BM, BN]))],
   .ifThen (.eq .nat .nil (.constNat even) (.constNat 0))
     [.assign .real [BM, BN] "p" (.where (tailMask BM BN context)
       (.ref .real [BM, BN] "p") (.broadcast (.const 0.0) [BM, BN]))]]

def rescale (BM BN BD : Nat) : List Stmt :=
  [.assign .real [BM] "l_ij" (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p")),
   .assign .real [BM] "tmp" (.sub .real (.consSame .nil)
     (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij")),
   .assign .bool [BM] "alpha_mask" (.ne .real (.consSame .nil)
     (.ref .real [BM] "tmp") (.ref .real [BM] "tmp")),
   .assign .real [BM] "alpha" (.exp2 (.ref .real [BM] "tmp")),
   .assign .real [BM] "alpha" (.where (.ref .bool [BM] "alpha_mask")
     (.broadcast (.const 1.0) [BM]) (.ref .real [BM] "alpha")),
   .assign .real [BM] "l_i" (.add .real (.consSame .nil)
     (.mul .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "alpha"))
     (.ref .real [BM] "l_ij")),
   .assign .real [BM, BD] "acc" (.mul .real (.consSame (.consR .nil))
     (.ref .real [BM, BD] "acc") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "alpha")))]

def loadKey (BD BN even : Nat) : Stmt :=
  .ifThenElse (.ne .nat .nil (.constNat even) (.constNat 0))
    [.assign .real [BD, BN] "k" (.load .real
      (.blockPtr (.ref .blockPtr [BD, BN] "K_block_ptr") []) .none .unspecified)]
    [.assign .real [BD, BN] "k" (.load .real
      (.blockPtr (.ref .blockPtr [BD, BN] "K_block_ptr") [0, 1]) .none)]

def accumulate (BM BN BD even : Nat) : List Stmt :=
  [.ifThenElse (.ne .nat .nil (.constNat even) (.constNat 0))
     [.assign .real [BN, BD] "v" (.load .real
       (.blockPtr (.ref .blockPtr [BN, BD] "V_block_ptr") []) .none .unspecified)]
     [.assign .real [BN, BD] "v" (.load .real
       (.blockPtr (.ref .blockPtr [BN, BD] "V_block_ptr") [0, 1]) .none)],
   .assign .real [BM, BD] "acc" (.add .real (.consSame (.consSame .nil))
     (.ref .real [BM, BD] "acc")
     (.dot (batch := []) (.ref .real [BM, BN] "p") (.ref .real [BN, BD] "v"))),
   .assign .real [BM] "m_i" (.ref .real [BM] "m_ij"),
   .assign .blockPtr [BN, BD] "V_block_ptr"
     (.advanceBlockPtr (.ref .blockPtr [BN, BD] "V_block_ptr") [BN, 0]),
   .assign .blockPtr [BD, BN] "K_block_ptr"
     (.advanceBlockPtr (.ref .blockPtr [BD, BN] "K_block_ptr") [0, BN])]

def body (BM BN BD context window complement even : Nat) (offset size : Int) : List Stmt :=
  [.assign .int [] "start_n" (.ref .int [] "start_n"), loadKey BD BN even] ++
    scores BM BN BD context window complement even offset size ++
    probabilities BM BN context window even ++ rescale BM BN BD ++ accumulate BM BN BD even

/-- Locate a top-level signed source loop by its index register. -/
def sourceLoop? (index : RegName) : List Stmt → Option (List Stmt)
  | [] => none
  | .forRangeInt name _ _ _ loop :: rest => if name = index then some loop else sourceLoop? index rest
  | _ :: rest => sourceLoop? index rest

/-- Selecting the loop before erasing avoids traversing unrelated address
arithmetic while proving the same complete source connection. -/
theorem sourceLoop_erase (index : RegName) (code : List Stmt) :
    sourceLoop? index (Stmt.eraseDTypeList code) =
      (sourceLoop? index code).map Stmt.eraseDTypeList := by
  induction code with
  | nil => simp only [Stmt.eraseDTypeList, sourceLoop?, Option.map_none]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, sourceLoop?]
    all_goals try exact ih
    split <;> simp_all

theorem erased_loop (k : ComputeKernel) (index : RegName)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    sourceLoop? index k.eraseDType.toAlgKernel.body =
      (sourceLoop? index k.toAlgKernel.body).map Stmt.eraseDTypeList := by
  rw [ComputeKernel.eraseDType, h]
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody, Kernel.eraseDType]
  exact sourceLoop_erase index k.toAlgKernel.body

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
theorem original_loop_source :
    sourceLoop? "start_n" (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (body BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        IS_EVEN_N sliding_window_offset sliding_window_size) := by
  rw [erased_loop _ _ (by rfl)]
  dsimp only [TypedSource.original, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, sourceLoop?, Option.map]
  simp only [ite_true, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    body, scores, probabilities, rescale, accumulate, loadKey, tailMask,
    RealExpressions.maskStmt, RealExpressions.distanceExpr]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 16000 in
theorem candidate_loop_source :
    sourceLoop? "start_n" (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).eraseDType.toAlgKernel.body =
      some (body BLOCK_M BLOCK_N BLOCK_DMODEL NKV_CTX SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
        IS_EVEN_N sliding_window_offset sliding_window_size) := by
  rw [erased_loop _ _ (by rfl)]
  dsimp only [TypedSource.reciprocal, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, sourceLoop?, Option.map]
  simp only [ite_true, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    body, scores, probabilities, rescale, accumulate, loadKey, tailMask,
    RealExpressions.maskStmt, RealExpressions.distanceExpr]
  rfl
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealBody

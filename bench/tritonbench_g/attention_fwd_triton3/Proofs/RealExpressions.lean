import bench.tritonbench_g.attention_fwd_triton3.Kernels
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealKeys
import VeriTile.Triton.Semantics.Integer
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Real-side expression facts for the full typed source. The syntax checks
locate actual assignments in both kernels; evaluation then relates their
signed coordinates to the independent window predicate. These are helpers for
the remaining whole-kernel real execution proof, not replacements for it. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExpressions
open VeriTile.Triton

def distanceExpr (BM BN : Nat) (offset : Int) : Op .int [BM, BN] :=
  .add .int .scalarR
    (.sub .int .scalarR
      (.add .int .scalarR
        (.sub .int (.consR (.consL .nil))
          (.castNatToInt (.expandDim ⟨1, by simp⟩ (.arange BM)))
          (.castNatToInt (.expandDim ⟨0, by simp⟩ (.arange BN))))
        (.mul .int .nil (.castNatToInt (.ref .nat [] "start_m")) (.constInt BM)))
      (.ref .int [] "start_n")) (.constInt offset)

def distances (BM BN startM : Nat) (startN offset : Int) : Tile .int [BM, BN] :=
  ⟨fun i => ((startM * BM : Nat) : Int) + i.1.val - startN - i.2.1.val + offset⟩

theorem eval_distance (s : BlockState) (BM BN startM : Nat) (startN offset : Int)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN)) :
    evalOp (distanceExpr BM BN offset) s = some (distances BM BN startM startN offset) := by
  simp only [distanceExpr, evalOp.eq_def, hm, hn, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  rcases i with ⟨row, key, ⟨⟩⟩
  simp [distances, Tile.bop_data, Tile.scalar, Tile.vec, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.sub, NumericDType.mul,
    -Tile.scalar_eta]
  ring

/-- Once the signed loop index is known nonnegative, a local lane has exactly
the global distance used by the independent window formula. -/
theorem distance_global_key (BM BN startM startN : Nat) (offset : Int)
    (row : Fin BM) (key : Fin BN) :
    (distances BM BN startM startN offset).data (row, key, PUnit.unit) =
      WindowBounds.distance startM BM row.val (startN + key.val) offset := by
  simp [distances, WindowBounds.distance]
  ring

def regularMaskExpr (BM BN : Nat) (size : Int) : Op .bool [BM, BN] :=
  .boolAnd (.consSame (.consSame .nil))
    (.ge .int .scalarR (.ref .int [BM, BN] "dist") (.castNatToInt (.constNat 0)))
    (.lt .int .scalarR (.ref .int [BM, BN] "dist") (.constInt size))

/-- The source's complement arm keeps distances at least `size`. It is not
the Boolean negation of the ordinary two-sided window. -/
def complementMaskExpr (BM BN : Nat) (size : Int) : Op .bool [BM, BN] :=
  .ge .int .scalarR (.ref .int [BM, BN] "dist") (.constInt size)

def regularMask {BM BN : Nat} (d : Tile .int [BM, BN]) (size : Int) :
    Tile .bool [BM, BN] :=
  ⟨fun i => decide (0 ≤ d.data i ∧ d.data i < size)⟩

def complementMask {BM BN : Nat} (d : Tile .int [BM, BN]) (size : Int) :
    Tile .bool [BM, BN] :=
  ⟨fun i => decide (size ≤ d.data i)⟩

theorem eval_regular_mask (s : BlockState) (BM BN : Nat)
    (d : Tile .int [BM, BN]) (size : Int)
    (hd : s.regs .int [BM, BN] "dist" = some d) :
    evalOp (regularMaskExpr BM BN size) s = some (regularMask d size) := by
  simp only [regularMaskExpr, evalOp.eq_def, hd, bind, Option.bind]
  congr 1
  apply Tile.ext
  intro i
  rcases i with ⟨row, key, ⟨⟩⟩
  simp [regularMask, Tile.bop_data, Tile.cop_data, Tile.scalar,
    ComparableDType.ge, ComparableDType.lt, -Tile.scalar_eta]
  rfl

theorem eval_complement_mask (s : BlockState) (BM BN : Nat)
    (d : Tile .int [BM, BN]) (size : Int)
    (hd : s.regs .int [BM, BN] "dist" = some d) :
    evalOp (complementMaskExpr BM BN size) s = some (complementMask d size) := by
  simp only [complementMaskExpr, evalOp.eq_def, hd, bind, Option.bind]
  congr 1

theorem regular_mask_global_key (BM BN startM startN : Nat) (offset size : Int)
    (row : Fin BM) (key : Fin BN) :
    (regularMask (distances BM BN startM startN offset) size).data (row, key, PUnit.unit) = Bool.true ↔
      WindowBounds.retained startM BM row.val (startN + key.val) offset size := by
  change decide (0 ≤ (distances BM BN startM startN offset).data (row, key, PUnit.unit) ∧
    (distances BM BN startM startN offset).data (row, key, PUnit.unit) < size) = Bool.true ↔ _
  rw [distance_global_key BM BN startM startN offset row key]
  simp [WindowBounds.retained]

theorem complement_mask_global_key (BM BN startM startN : Nat) (offset size : Int)
    (row : Fin BM) (key : Fin BN) :
    (complementMask (distances BM BN startM startN offset) size).data (row, key, PUnit.unit) = Bool.true ↔
      size ≤ WindowBounds.distance startM BM row.val (startN + key.val) offset := by
  change decide (size ≤ (distances BM BN startM startN offset).data
    (row, key, PUnit.unit)) = Bool.true ↔ _
  rw [distance_global_key BM BN startM startN offset row key]
  simp

def maskChoice (BM BN complement : Nat) (size : Int) : ComputeStmt :=
  .ifThenElse (.ne .nat .nil (.constNat complement) (.constNat 0))
    [.assign .bool [BM, BN] "mask" (.alg (complementMaskExpr BM BN size))]
    [.assign .bool [BM, BN] "mask" (.alg (regularMaskExpr BM BN size))]

def maskStmt (BM BN complement : Nat) (size : Int) : Stmt :=
  .ifThenElse (.ne .nat .nil (.constNat complement) (.constNat 0))
    [.assign .bool [BM, BN] "mask" (complementMaskExpr BM BN size)]
    [.assign .bool [BM, BN] "mask" (regularMaskExpr BM BN size)]

theorem mask_choice_projection (BM BN complement : Nat) (size : Int) :
    (maskChoice BM BN complement size).toAlgorithm? =
      .ok (maskStmt BM BN complement size) := rfl

def selectedMask {BM BN : Nat} (d : Tile .int [BM, BN]) (complement : Nat) (size : Int) :
    Tile .bool [BM, BN] :=
  if complement ≠ 0 then complementMask d size else regularMask d size

/-- Connect the executed mask choice to the full-context key predicate. The
source evaluates this fragment only when its outer window flag is enabled. -/
theorem selected_mask_matches_keep (BM BN startM startN window complement : Nat)
    (offset size : Int) (row : Fin BM) (key : Fin BN) (hw : window ≠ 0) :
    (selectedMask (distances BM BN startM startN offset) complement size).data
        (row, key, PUnit.unit) =
      RealKeys.keep startM BM row.val window complement offset size (startN + key.val) := by
  apply Bool.eq_iff_iff.mpr
  by_cases hc : complement = 0
  · simp only [selectedMask, hc, ne_eq, not_true_eq_false, if_false]
    rw [regular_mask_global_key]
    simp [RealKeys.keep, hw, WindowBounds.retained]
  · simp only [selectedMask, if_pos hc]
    rw [complement_mask_global_key]
    simp [RealKeys.keep, hw, hc]

theorem mask_branch_executes (s : BlockState) (BM BN complement : Nat)
    (d : Tile .int [BM, BN]) (size : Int)
    (hd : s.regs .int [BM, BN] "dist" = some d) :
    stepStmt (maskStmt BM BN complement size) s =
      some (s.setReg "mask" .bool [BM, BN] (selectedMask d complement size)) := by
  have hregular := eval_regular_mask s BM BN d size hd
  have hcomplement := eval_complement_mask s BM BN d size hd
  simp only [maskStmt, stepStmt, evalOp.eq_def, bind, Option.bind,
    Tile.cop_data, Tile.scalar, ComparableDType.ne]
  by_cases h : complement = 0
  · simp [h, selectedMask, stepStmts, stepStmt, hregular]
  · simp [h, selectedMask, stepStmts, stepStmt, hcomplement]

/-- Successful execution of the actual distance and mask fragment for either
branch. Only registers change; the statement equality includes the memory
frame. This is a loop-body fragment, not the whole attention kernel. -/
theorem mask_fragment_executes (s : BlockState) (BM BN startM complement : Nat)
    (startN offset size : Int)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hn : s.regs .int [] "start_n" = some (Tile.scalar startN)) :
    stepStmts [.assign .int [BM, BN] "dist" (distanceExpr BM BN offset),
      maskStmt BM BN complement size] s =
      some ((s.setReg "dist" .int [BM, BN] (distances BM BN startM startN offset)).setReg
        "mask" .bool [BM, BN] (selectedMask (distances BM BN startM startN offset) complement size)) := by
  have hd := eval_distance s BM BN startM startN offset hm hn
  simp only [stepStmts, stepStmt, hd, bind, Option.bind]
  rw [mask_branch_executes _ BM BN complement (distances BM BN startM startN offset)
    size (by simp)]

/-- Find a named assignment inside source control flow, without inspecting
floating values or assuming that a branch executes. The returned statement
retains its dtype, shape and complete expression. -/
def assignment? : Nat → RegName → List ComputeStmt → Option ComputeStmt
  | 0, _, _ => none
  | _ + 1, _, [] => none
  | n + 1, name, st :: rest =>
      let found := match st with
        | .assign _ _ out _ => if out = name then some st else none
        | .ifThen _ body => assignment? n name body
        | .ifThenElse _ yes no => assignment? n name yes <|> assignment? n name no
        | .forRangeInt _ _ _ _ body => assignment? n name body
        | _ => none
      found <|> assignment? n name rest

/-- Locate the source branch whose two arms assign the requested register.
Returning the whole conditional also checks its selector and both expressions. -/
def assignmentChoice? : Nat → RegName → List ComputeStmt → Option ComputeStmt
  | 0, _, _ => none
  | _ + 1, _, [] => none
  | n + 1, name, st :: rest =>
      let found := match st with
        | .ifThenElse _ [.assign _ _ a _] [.assign _ _ b _] =>
            if a = name ∧ b = name then some st else none
        | .ifThen _ body => assignmentChoice? n name body
        | .ifThenElse _ yes no => assignmentChoice? n name yes <|> assignmentChoice? n name no
        | .forRangeInt _ _ _ _ body => assignmentChoice? n name body
        | _ => none
      found <|> assignmentChoice? n name rest

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

theorem original_distance_source :
    assignment? 128 "dist" (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).surfaceBody =
      some (.assign .int [BLOCK_M, BLOCK_N] "dist"
        (.alg (distanceExpr BLOCK_M BLOCK_N sliding_window_offset))) := rfl

theorem candidate_distance_source :
    assignment? 128 "dist" (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).surfaceBody =
      some (.assign .int [BLOCK_M, BLOCK_N] "dist"
        (.alg (distanceExpr BLOCK_M BLOCK_N sliding_window_offset))) := rfl

theorem original_mask_source :
    assignmentChoice? 128 "mask" (TypedSource.original Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).surfaceBody =
      some (maskChoice BLOCK_M BLOCK_N COMPLEMENT_SLIDING_WINDOW sliding_window_size) := rfl

theorem candidate_mask_source :
    assignmentChoice? 128 "mask" (TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW
      sliding_window_offset sliding_window_size).surfaceBody =
      some (maskChoice BLOCK_M BLOCK_N COMPLEMENT_SLIDING_WINDOW sliding_window_size) := rfl

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExpressions

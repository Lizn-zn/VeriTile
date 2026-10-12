import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealAccess

/-! Register-only portions of the actual source require no allocation bounds.
The local certificate is constructed from each concrete AST, then translated
into the framework's trace-safety predicate. Loads and stores have no certificate
constructor and are handled by separate active-address proofs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
open VeriTile.Triton RealAccess

inductive RegisterOnly : Stmt → Prop
  | assign {dt shape name} {e : Op dt shape}
      (safe : ∀ bounds s, e.SafeAt bounds s) : RegisterOnly (.assign dt shape name e)
  | branch {c : Op .bool []} {yes no : List Stmt}
      (safe : ∀ bounds s, c.SafeAt bounds s)
      (hy : ∀ st ∈ yes, RegisterOnly st) (hn : ∀ st ∈ no, RegisterOnly st) :
      RegisterOnly (.ifThenElse c yes no)
  | optional {c : Op .bool []} {body : List Stmt}
      (safe : ∀ bounds s, c.SafeAt bounds s) (hb : ∀ st ∈ body, RegisterOnly st) :
      RegisterOnly (.ifThen c body)

@[simp] theorem assign_iff {dt shape name} {e : Op dt shape} :
    RegisterOnly (.assign dt shape name e) ↔ ∀ bounds s, e.SafeAt bounds s := by
  constructor
  · intro h; cases h with | assign he => exact he
  · exact RegisterOnly.assign

@[simp] theorem branch_iff {c : Op .bool []} {yes no : List Stmt} :
    RegisterOnly (.ifThenElse c yes no) ↔ (∀ bounds s, c.SafeAt bounds s) ∧
      (∀ st ∈ yes, RegisterOnly st) ∧ (∀ st ∈ no, RegisterOnly st) := by
  constructor
  · intro h; cases h with | branch hc hy hn => exact ⟨hc, hy, hn⟩
  · rintro ⟨hc, hy, hn⟩; exact RegisterOnly.branch hc hy hn

@[simp] theorem optional_iff {c : Op .bool []} {body : List Stmt} :
    RegisterOnly (.ifThen c body) ↔ (∀ bounds s, c.SafeAt bounds s) ∧ (∀ st ∈ body, RegisterOnly st) := by
  constructor
  · intro h; cases h with | optional hc hb => exact ⟨hc, hb⟩
  · rintro ⟨hc, hb⟩; exact RegisterOnly.optional hc hb

theorem RegisterOnly.safe {st : Stmt} (h : RegisterOnly st) : ∀ bounds s, Stmt.TraceSafe bounds st s := by
  induction h with
  | assign he => intro bounds s; simpa only [Stmt.TraceSafe] using he bounds s
  | branch hc _ _ ihy ihn =>
    intro bounds s
    exact if_else_safe (hc bounds s)
      (all_safe _ s (fun st hs t => ihy st hs bounds t))
      (all_safe _ s (fun st hs t => ihn st hs bounds t))
  | optional hc _ ih =>
    intro bounds s
    exact if_safe (hc bounds s) (all_safe _ s (fun st hs t => ih st hs bounds t))

theorem list_safe (body : List Stmt) (h : ∀ st ∈ body, RegisterOnly st) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds body s := all_safe body s (fun st hs => (h st hs).safe bounds)

@[simp] theorem dynamic_offsets_safe (bounds : RegionBounds) (s : BlockState) (region : RegionName)
    (base : Op .nat []) (parent shape strides : List Nat) (a b : Op .nat []) :
    (Op.makeBlockPtrDynOffsets region base parent shape strides [a, b]).SafeAt bounds s ↔
      base.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  simp only [List.forall_mem_cons]
  simp

@[simp] theorem dynamic_advance_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (ptr : Op .blockPtr shape) (a b : Op .int []) :
    (Op.advanceBlockPtrDyn ptr [a, b]).SafeAt bounds s ↔
      ptr.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s := by
  rw [Op.SafeAt.eq_def]
  simp only [List.forall_mem_cons]
  simp

theorem header_safe (heads kvHeads : Nat) (strides : RealHeader.Strides) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealHeader.code heads kvHeads strides) s := by
  apply list_safe
  simp [RealHeader.code, RealHeader.coordinateCode, RealHeader.offsetCode, RealHeader.offsetExpr, Op.SafeAt.eq_def]

theorem pointers_safe (BM BN BD : Nat) (p : RealPointers.Layout) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealPointers.code BM BN BD p) s := by
  apply list_safe
  simp only [RealPointers.code, List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const,
    and_true, assign_iff, RealPointers.queryExpr, RealPointers.valueExpr, RealPointers.keyExpr,
    RealPointers.outputExpr, dynamic_offsets_safe]
  simp [RealPointers.rowPointerExpr, RealPointers.rowOffsetExpr, Op.SafeAt.eq_def]

theorem scale_safe (scale : ℝ) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealInit.scaleCode scale) s := by
  apply list_safe
  simp [RealInit.scaleCode, Op.SafeAt.eq_def]

theorem window_safe (BM BN BD context window complement : Nat) (offset size : Int)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (RealWindow.code BM BN BD context window complement offset size) s := by
  apply RegisterOnly.safe
  simp only [RealWindow.code, RealWindow.clampLower, RealWindow.clampUpper, RealWindow.advance,
    branch_iff, optional_iff, List.cons_append, List.nil_append, List.forall_mem_cons,
    List.not_mem_nil, false_implies, forall_const, and_true, assign_iff, dynamic_advance_safe]
  simp [RealWindow.lowerExpr, RealWindow.upperExpr, RealWindow.complementUpperExpr, Op.SafeAt.eq_def]

theorem scores_safe (BM BN BD context window complement even : Nat) (offset size : Int)
    (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.scores BM BN BD context window complement even offset size) s := by
  apply list_safe
  simp [RealBody.scores, RealBody.tailMask, RealExpressions.distanceExpr, RealExpressions.maskStmt,
    RealExpressions.regularMaskExpr, RealExpressions.complementMaskExpr, Op.SafeAt.eq_def]

theorem probabilities_safe (BM BN context window even : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.probabilities BM BN context window even) s := by
  apply list_safe
  simp [RealBody.probabilities, RealBody.tailMask, Op.SafeAt.eq_def]

theorem rescale_safe (BM BN BD : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.rescale BM BN BD) s := by
  apply list_safe
  simp [RealBody.rescale, Op.SafeAt.eq_def]

theorem accumulate_tail_safe (BM BN BD even : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.accumulate BM BN BD even).tail s := by
  apply list_safe
  simp [RealBody.accumulate, Op.SafeAt.eq_def]

theorem normalize_safe (BM BD : Nat) (reciprocal : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealEpilogue.normalizeCode BM BD reciprocal) s := by
  apply list_safe
  cases reciprocal <;> simp [RealEpilogue.normalizeCode, RealEpilogue.quotientExpr,
    RealEpilogue.denominatorExpr, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety

import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealExecutionSource
import VeriTile.Triton.Memory.TraceProofs

/-! Compositional access-safety helpers for the actual source. Bounds refer to
active addresses, including unchecked output stores. Checked load lanes outside
the parent shape are inactive; no numerical or output equation is assumed. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Triton

theorem append_safe {bounds : RegionBounds} :
    ∀ (first : List Stmt) {rest : List Stmt} (s : BlockState),
      Stmt.TraceSafeList bounds first s →
      (∀ t, stepStmts first s = some t → Stmt.TraceSafeList bounds rest t) →
      Stmt.TraceSafeList bounds (first ++ rest) s
  | [], _, s, _, h => h s stepStmts.nil
  | st :: first, rest, s, h, hn => by
      rw [Stmt.TraceSafeList] at h
      refine Stmt.TraceSafeList.cons_intro h.1 (fun t ht => ?_)
      have hrest := h.2
      rw [ht] at hrest
      exact append_safe first t hrest
        (fun u hu => hn u ((stepStmts.cons_some ht).trans hu))

theorem append_of_run {bounds : RegionBounds} {first rest : List Stmt} {s t : BlockState}
    (hs : Stmt.TraceSafeList bounds first s) (hr : stepStmts first s = some t)
    (ht : Stmt.TraceSafeList bounds rest t) : Stmt.TraceSafeList bounds (first ++ rest) s := by
  apply append_safe first s hs
  intro u hu
  have he : t = u := Option.some.inj (hr.symm.trans hu)
  simpa only [← he] using ht

theorem cons_of_run {bounds : RegionBounds} {st : Stmt} {rest : List Stmt} {s t : BlockState}
    (hs : Stmt.TraceSafe bounds st s) (hr : stepStmt st s = some t)
    (ht : Stmt.TraceSafeList bounds rest t) : Stmt.TraceSafeList bounds (st :: rest) s := by
  apply Stmt.TraceSafeList.cons_intro hs
  intro u hu
  have he : t = u := Option.some.inj (hr.symm.trans hu)
  simpa only [← he] using ht

theorem all_safe {bounds : RegionBounds} :
    ∀ (body : List Stmt) (s : BlockState),
      (∀ st ∈ body, ∀ t, Stmt.TraceSafe bounds st t) → Stmt.TraceSafeList bounds body s
  | [], _, _ => Stmt.TraceSafeList.nil_intro
  | st :: rest, s, h => by
      apply Stmt.TraceSafeList.cons_intro (h st List.mem_cons_self s)
      intro t _
      exact all_safe rest t (fun st hs => h st (List.mem_cons_of_mem _ hs))

theorem if_else_safe {bounds : RegionBounds} {s : BlockState} {c : Op .bool []} {yes no : List Stmt}
    (hc : c.SafeAt bounds s) (hy : Stmt.TraceSafeList bounds yes s)
    (hn : Stmt.TraceSafeList bounds no s) : Stmt.TraceSafe bounds (.ifThenElse c yes no) s := by
  rw [Stmt.TraceSafe]
  refine ⟨hc, ?_⟩
  cases evalOp c s with
  | none => trivial
  | some v => dsimp; split <;> assumption

theorem if_safe {bounds : RegionBounds} {s : BlockState} {c : Op .bool []} {body : List Stmt}
    (hc : c.SafeAt bounds s) (hb : Stmt.TraceSafeList bounds body s) :
    Stmt.TraceSafe bounds (.ifThen c body) s := by
  rw [Stmt.TraceSafe]
  refine ⟨hc, ?_⟩
  cases evalOp c s with
  | none => trivial
  | some v => dsimp; split <;> trivial

theorem if_nonzero_safe (bounds : RegionBounds) (s : BlockState) (flag : Nat) (yes no : List Stmt) :
    Stmt.TraceSafe bounds (.ifThenElse (.ne .nat .nil (.constNat flag) (.constNat 0)) yes no) s ↔
      if flag ≠ 0 then Stmt.TraceSafeList bounds yes s else Stmt.TraceSafeList bounds no s := by
  rw [Stmt.TraceSafe]
  simp only [Op.SafeAt.eq_def, true_and, evalOp.eq_def, bind, Option.bind_some,
    Tile.scalar, Tile.cop, ComparableDType.ne]
  simp

def BlockBounds {shape : TileShape} (bounds : RegionBounds) (ptrs : Tile .blockPtr shape)
    (checked : List Nat) : Prop := ∀ i,
  (ptrs.data i).inBounds (TileShape.indexToList shape i) checked = Bool.true →
    (ptrs.data i).address (TileShape.indexToList shape i) < bounds (ptrs.data i).region

theorem block_access_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .blockPtr shape) (checked : List Nat)
    (hp : s.regs .blockPtr shape name = some ptrs) (hb : BlockBounds bounds ptrs checked) :
    (MemAccess.blockPtr (d := .real) (.ref .blockPtr shape name) checked).ActiveAddressSafe bounds s
      (fun _ => True) := by
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro tile ht i _
  rw [evalOp_ref, hp] at ht
  cases ht
  dsimp only
  exact hb i

theorem block_load_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .blockPtr shape) (checked : List Nat)
    (hp : s.regs .blockPtr shape name = some ptrs) (hb : BlockBounds bounds ptrs checked)
    (padding : PaddingOption := .zero) :
    (Op.load .real (.blockPtr (.ref .blockPtr shape name) checked) .none padding).SafeAt bounds s := by
  simp only [Op.SafeAt.eq_def, true_and]
  exact block_access_safe bounds s name ptrs checked hp hb

theorem pointer_access_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape)
    (hp : s.regs .ptr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr shape name)).ActiveAddressSafe bounds s (fun _ => True) := by
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro tile ht i _
  rw [evalOp_ref, hp] at ht
  cases ht
  exact hb i

theorem pointer_load_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape)
    (hp : s.regs .ptr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (Op.load .real (.ptr (.ref .ptr shape name)) .none).SafeAt bounds s := by
  simp only [Op.SafeAt.eq_def, true_and]
  exact pointer_access_safe bounds s name ptrs hp hb

theorem pointer_store_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name value : RegName) (ptrs : Tile .ptr shape)
    (hp : s.regs .ptr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    Stmt.TraceSafe bounds (.store .real shape (.ptr (.ref .ptr shape name)) (.ref .real shape value) .none) s := by
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt, true_and]
  exact pointer_access_safe bounds s name ptrs hp hb

theorem block_store_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name value : RegName) (ptrs : Tile .blockPtr shape)
    (hp : s.regs .blockPtr shape name = some ptrs) (hb : BlockBounds bounds ptrs []) :
    Stmt.TraceSafe bounds (.store .real shape (.blockPtr (.ref .blockPtr shape name) [])
      (.ref .real shape value) .none) s := by
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt, true_and]
  exact block_access_safe bounds s name ptrs [] hp hb

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess

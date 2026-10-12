import bench.tritonbench_g.attention_fwd_triton1.Proofs.Legacy
import VeriTile.Meta.StatementAudit

/-! Unmasked two-dimensional stores used by both O and the optional H
snapshots. The frame is stated for memory cells, so it preserves their tags
as well as their real-valued observations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealStore
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def offset {rows cols : Nat} (base row stride : Nat) (i : TileIndex [rows, cols]) : Nat :=
  base + (row + i.1.val) * stride + i.2.1.val

noncomputable def writeTile {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) : BlockState :=
  (TileShape.allIndices [rows, cols]).foldl
    (fun acc i => acc.writeMem region (offset base row stride i) (value i.1 i.2.1)) st

theorem executes (st : BlockState) (region : RegionName) (ptr val : RegName)
    (parentRows parentCols rows cols base row stride : Nat)
    (value : Fin rows → Fin cols → ℝ)
    (hp : st.regs .blockPtr [rows, cols] ptr = some
      ⟨fun _ => BlockPtr.mk region base [parentRows, parentCols] [rows, cols]
        [stride, 1] [row, 0]⟩)
    (hv : st.regs .real [rows, cols] val = some ⟨fun i => some (value i.1 i.2.1)⟩) :
    stepStmt (.store .real [rows, cols] (.blockPtr (.ref .blockPtr [rows, cols] ptr) [])
      (.ref .real [rows, cols] val) .none) st =
      some (writeTile st region base row stride value) := by
  unfold stepStmt
  simp only [evalOp_ref, hv, hp, Option.bind, Option.map]
  refine congrArg some (congrArg
    (fun f => List.foldl f st (TileShape.allIndices [rows, cols])) ?_)
  funext acc i
  obtain ⟨t, d, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds,
    List.all_nil, Bool.and_true, if_true, Nat.zero_add, Nat.mul_one,
    BlockState.writeMemTyped_real, FloatDType.real_storeValue]
  rfl

theorem offset_injective {rows cols : Nat} (base row stride : Nat) (hstride : cols ≤ stride) :
    Function.Injective (@offset rows cols base row stride) := by
  rintro ⟨a, b, u⟩ ⟨a', b', u'⟩ h
  have hcol := b.isLt
  have hcol' := b'.isLt
  simp only [offset, Nat.add_left_cancel_iff] at h
  have ha : a.val = a'.val := by
    by_contra hn
    rcases lt_or_gt_of_ne hn with hlt | hgt
    · have hm := Nat.mul_le_mul_right stride (show row + a.val + 1 ≤ row + a'.val by omega)
      simp only [Nat.add_mul, Nat.one_mul] at h hm
      omega
    · have hm := Nat.mul_le_mul_right stride (show row + a'.val + 1 ≤ row + a.val by omega)
      simp only [Nat.add_mul, Nat.one_mul] at h hm
      omega
  have hb : b.val = b'.val := by simpa only [ha, Nat.add_left_cancel_iff] using h
  have : a = a' := Fin.ext ha
  have : b = b' := Fin.ext hb
  subst a'
  subst b'
  rfl

/-- A written output carries its real tag, even when its value is zero.
The real-valued observer alone would not distinguish a zero from a mismatched
cell, so the public typed IO contract uses this stronger fact. -/
theorem readback_cell {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (hstride : cols ≤ stride)
    (r : Fin rows) (c : Fin cols) :
    (writeTile st region base row stride value).mem region
      (base + (row + r.val) * stride + c.val) = MemCell.real (value r c) := by
  let i : TileIndex [rows, cols] := (r, c, PUnit.unit)
  have hinj := offset_injective (rows := rows) (cols := cols) base row stride hstride
  obtain ⟨before, after, hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [rows, cols] i)
  have hn := TileShape.allIndices_nodup [rows, cols]
  rw [hsplit, List.nodup_append, List.nodup_cons] at hn
  have hafter : ∀ k ∈ after, offset base row stride k ≠ offset base row stride i := by
    intro k hk heq
    have he := hinj heq
    subst k
    exact hn.2.1.1 hk
  change ((TileShape.allIndices [rows, cols]).foldl
    (fun acc k => acc.writeMem region (offset base row stride k) (value k.1 k.2.1)) st).mem
      region (offset base row stride i) = _
  rw [hsplit, List.foldl_append, List.foldl_cons,
    BlockState.foldl_writeMem_mem_preserve_unhit _ _ after _ hafter,
    BlockState.writeMem_mem, if_pos ⟨rfl, rfl⟩]

theorem readback {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (hstride : cols ≤ stride)
    (r : Fin rows) (c : Fin cols) :
    (writeTile st region base row stride value).readMem region
      (base + (row + r.val) * stride + c.val) = value r c := by
  exact BlockState.scatter_readback_nd st (offset base row stride)
    (fun i => value i.1 i.2.1) (offset_injective base row stride hstride) (r, c, PUnit.unit)

theorem frame {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (r : RegionName) (off : Nat)
    (hout : r ≠ region ∨ ∀ (i : Fin rows) (j : Fin cols),
      off ≠ base + (row + i.val) * stride + j.val) :
    (writeTile st region base row stride value).mem r off = st.mem r off := by
  rcases hout with hr | hoff
  · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr off st
  · by_cases hr : r = region
    · subst r
      apply BlockState.foldl_writeMem_mem_preserve_unhit
      intro i _
      exact (hoff i.1 i.2.1).symm
    · exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr off st

@[simp] theorem regs {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) :
    (writeTile st region base row stride value).regs = st.regs := by
  funext dtype shape name
  simp [writeTile]

@[simp] theorem pids {rows cols : Nat} (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) :
    (writeTile st region base row stride value).pids = st.pids := by
  unfold writeTile
  generalize TileShape.allIndices [rows, cols] = indices
  induction indices generalizing st with
  | nil => rfl
  | cons i is ih =>
    rw [List.foldl_cons, ih]
    rfl

/-- The STORE constexpr either performs the real store or leaves state intact. -/
noncomputable def save {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) : BlockState :=
  if STORE then writeTile st region base row stride value else st

theorem save_executes (STORE : Bool) (st : BlockState) (region : RegionName) (ptr val : RegName)
    (parentRows parentCols rows cols base row stride : Nat)
    (value : Fin rows → Fin cols → ℝ)
    (hp : st.regs .blockPtr [rows, cols] ptr = some
      ⟨fun _ => BlockPtr.mk region base [parentRows, parentCols] [rows, cols]
        [stride, 1] [row, 0]⟩)
    (hv : st.regs .real [rows, cols] val = some ⟨fun i => some (value i.1 i.2.1)⟩) :
    stepStmt (.ifThen (.constBool STORE)
      [.store .real [rows, cols] (.blockPtr (.ref .blockPtr [rows, cols] ptr) [])
        (.ref .real [rows, cols] val) .none]) st =
      some (save STORE st region base row stride value) := by
  have hs := executes st region ptr val parentRows parentCols rows cols base row stride value hp hv
  cases STORE with
  | «false» => exact aft1_ifThen_false_noop _ _
  | «true» =>
    have hif (body : List Stmt) : stepStmt (.ifThen (.constBool Bool.true) body) st =
        stepStmts body st := by simp [stepStmt, evalOp]
    rw [hif, stepStmts.cons_some hs, stepStmts.nil]
    rfl

@[simp] theorem save_regs {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) :
    (save STORE st region base row stride value).regs = st.regs := by
  cases STORE <;> simp [save]

@[simp] theorem save_pids {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) :
    (save STORE st region base row stride value).pids = st.pids := by
  cases STORE <;> simp [save]

theorem save_frame {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (r : RegionName) (off : Nat)
    (hout : STORE = Bool.false ∨ r ≠ region ∨ ∀ (i : Fin rows) (j : Fin cols),
      off ≠ base + (row + i.val) * stride + j.val) :
    (save STORE st region base row stride value).mem r off = st.mem r off := by
  cases STORE with
  | «false» => rfl
  | «true» => exact frame st region base row stride value r off (hout.resolve_left (by decide))

theorem save_read_other {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (r : RegionName) (off : Nat)
    (hr : r ≠ region) :
    (save STORE st region base row stride value).readMem r off = st.readMem r off := by
  simp only [BlockState.readMem, save_frame STORE st region base row stride value r off
    (Or.inr (Or.inl hr))]

theorem save_readback {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (hstride : cols ≤ stride)
    (hs : STORE = Bool.true) (r : Fin rows) (c : Fin cols) :
    (save STORE st region base row stride value).readMem region
      (base + (row + r.val) * stride + c.val) = value r c := by
  subst STORE
  exact readback st region base row stride value hstride r c

theorem save_readback_cell {rows cols : Nat} (STORE : Bool) (st : BlockState) (region : RegionName)
    (base row stride : Nat) (value : Fin rows → Fin cols → ℝ) (hstride : cols ≤ stride)
    (hs : STORE = Bool.true) (r : Fin rows) (c : Fin cols) :
    (save STORE st region base row stride value).mem region
      (base + (row + r.val) * stride + c.val) = MemCell.real (value r c) := by
  subst STORE
  exact readback_cell st region base row stride value hstride r c

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealStore

import bench.tritonbench_g.block_sparse_attn.Proofs.RealStores

/-! Output-address geometry, tagged cell readback and complete memory frame
for the actual masked final stores. Row stride must cover the written channel
blocks; query rows outside context never write, including EVEN_M variants. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores
open VeriTile.Triton RealNormalizationValues
noncomputable section

private theorem row_column_unique (base stride r r' c c' : Nat)
    (hc : c < stride) (hc' : c' < stride)
    (he : base + r * stride + c = base + r' * stride + c') : r = r' ∧ c = c' := by
  have hr : r = r' := by
    by_contra hn
    rcases lt_or_gt_of_ne hn with hlt | hgt
    · have hm := Nat.mul_le_mul_right stride (show r + 1 ≤ r' by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hm
      omega
    · have hm := Nat.mul_le_mul_right stride (show r' + 1 ≤ r by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hm
      omega
  exact ⟨hr, by simpa only [hr, Nat.add_left_cancel_iff] using he⟩

theorem offset_injective (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (hstride : BD ≤ p.oRow) :
    Function.Injective (fun i : TileIndex [BM, BD] => offset s BM BD p second i.1 i.2.1) := by
  rintro ⟨r, d, ⟨⟩⟩ ⟨r', d', ⟨⟩⟩ he
  simp only [offset, Nat.add_right_cancel_iff] at he
  have h := row_column_unique _ _ _ _ _ _ (lt_of_lt_of_le d.isLt hstride)
    (lt_of_lt_of_le d'.isLt hstride) he
  have hr : r = r' := Fin.ext (by simpa only [RealSetup.rows, Nat.add_left_cancel_iff] using h.1)
  have hd : d = d' := Fin.ext h.2
  subst r'
  subst d'
  rfl

theorem channels_disjoint (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (hstride : 2 * BD ≤ p.oRow) (r r' : Fin BM) (d d' : Fin BD) :
    offset s BM BD p Bool.true r d ≠ offset s BM BD p Bool.false r' d' := by
  intro he
  have hc := d.isLt
  have hc' := d'.isLt
  simp only [offset, Bool.false_eq_true, ↓reduceIte, Nat.add_zero, Nat.add_assoc] at he
  have h := row_column_unique (RealSetup.batch s p * p.oBatch + RealSetup.head s p * p.oHead)
    p.oRow (RealSetup.rows s BM r) (RealSetup.rows s BM r') (d.val + BD) d'.val
    (by omega) (by omega) (by simpa only [Nat.add_assoc] using he)
  omega

theorem write_readback_cell (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ) (r : Fin BM) (d : Fin BD)
    (hr : RealSetup.rows s BM r < p.context) (hstride : BD ≤ p.oRow) :
    (write s t BM BD p second values).mem p.Out (offset s BM BD p second r d) =
      MemCell.real (values r d) := by
  let i : TileIndex [BM, BD] := (r, d, PUnit.unit)
  have hinj := offset_injective s BM BD p second hstride
  obtain ⟨before, after, hsplit⟩ := List.append_of_mem (TileShape.mem_allIndices [BM, BD] i)
  have hn := TileShape.allIndices_nodup [BM, BD]
  rw [hsplit, List.nodup_append, List.nodup_cons] at hn
  have hafter : ∀ k ∈ after, RealSetup.rows s BM k.1 < p.context →
      ¬(p.Out = p.Out ∧ offset s BM BD p second k.1 k.2.1 = offset s BM BD p second r d) := by
    intro k hk _ he
    have heq : k = i := hinj he.2
    subst k
    exact hn.2.1.1 hk
  unfold write
  rw [hsplit, List.foldl_append, List.foldl_cons]
  rw [if_pos hr, foldl_store_preserve_cell _ _ _ _ _ after _ hafter,
    BlockState.writeMem_mem, if_pos ⟨rfl, rfl⟩]

theorem result_frame (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) (region : RegionName) (address : Nat)
    (hfirst : ∀ r d, RealSetup.rows s BM r < p.context →
      ¬(p.Out = region ∧ offset s BM BD p Bool.false r d = address))
    (hsecond : 2 ≤ p.blocks → ∀ r d, RealSetup.rows s BM r < p.context →
      ¬(p.Out = region ∧ offset s BM BD p Bool.true r d = address)) :
    (result s t BM BN BD p x).mem region address = t.mem region address := by
  by_cases hb : 2 ≤ p.blocks
  · simp only [result, hb, ↓reduceIte]
    rw [write_frame _ _ _ _ _ _ _ _ _ (hsecond hb), write_frame _ _ _ _ _ _ _ _ _ hfirst]
    rfl
  · simp only [result, hb, ↓reduceIte]
    rw [write_frame _ _ _ _ _ _ _ _ _ hfirst]
    rfl

theorem first_output (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) (r : Fin BM) (d : Fin BD)
    (hr : RealSetup.rows s BM r < p.context) (hstride : BD ≤ p.oRow)
    (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    (result s t BM BN BD p x).mem p.Out (offset s BM BD p Bool.false r d) =
      MemCell.real (x.accumulator r d) := by
  by_cases hb : 2 ≤ p.blocks
  · simp only [result, hb, ↓reduceIte]
    rw [write_frame _ _ _ _ _ _ _ _ _ (fun r' d' _ he =>
      channels_disjoint s BM BD p (htwo hb) r' r d' d he.2)]
    exact write_readback_cell s _ BM BD p Bool.false x.accumulator r d hr hstride
  · simp only [result, hb, ↓reduceIte]
    exact write_readback_cell s _ BM BD p Bool.false x.accumulator r d hr hstride

theorem second_output (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) (r : Fin BM) (d : Fin BD)
    (hr : RealSetup.rows s BM r < p.context) (hb : 2 ≤ p.blocks) (hstride : BD ≤ p.oRow) :
    (result s t BM BN BD p x).mem p.Out (offset s BM BD p Bool.true r d) =
      MemCell.real (x.secondAccumulator r d) := by
  simp only [result, hb, ↓reduceIte]
  exact write_readback_cell s _ BM BD p Bool.true x.secondAccumulator r d hr hstride

/-- Recover a source-independent cell outside both sets of active stores.
The fixed loop invariant supplies the complete pre-store memory frame. -/
theorem kernel_frame (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) (hf : RealLoop.Fixed s BM BN BD p t) (region : RegionName) (address : Nat)
    (hfirst : ∀ r d, RealSetup.rows s BM r < p.context →
      ¬(p.Out = region ∧ offset s BM BD p Bool.false r d = address))
    (hsecond : 2 ≤ p.blocks → ∀ r d, RealSetup.rows s BM r < p.context →
      ¬(p.Out = region ∧ offset s BM BD p Bool.true r d = address)) :
    (result s t BM BN BD p x).mem region address = s.mem region address := by
  rw [result_frame s t BM BN BD p x region address hfirst hsecond]
  exact congrArg (fun u => u.mem region address) hf.frame

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores

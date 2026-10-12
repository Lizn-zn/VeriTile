import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealAddressBounds

/-! Ordinary output strides imply nonoverlapping stores within one program.
The inner stride may exceed one and rows may contain padding. Cross-program
race freedom is a separate grid-level property. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealOutputLayout
open VeriTile.Triton RealPrologue

theorem strided_injective (BM BD base start rowStride dimStride : Nat)
    (hd : 0 < dimStride) (hr : BD * dimStride ≤ rowStride) :
    Function.Injective (fun i : TileIndex [BM, BD] =>
      base + (start + i.1.val) * rowStride + i.2.1.val * dimStride) := by
  rintro ⟨i, d, ⟨⟩⟩ ⟨j, e, ⟨⟩⟩ h
  have hdBound : d.val * dimStride < rowStride :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right d.isLt hd) hr
  have heBound : e.val * dimStride < rowStride :=
    lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right e.isLt hd) hr
  simp only [Nat.add_mul] at h
  have hij : i.val = j.val := by
    rcases lt_trichotomy i.val j.val with hij | hij | hij
    · have hsep := Nat.mul_le_mul_right rowStride (Nat.succ_le_of_lt hij)
      simp only [Nat.succ_mul] at hsep
      omega
    · exact hij
    · have hsep := Nat.mul_le_mul_right rowStride (Nat.succ_le_of_lt hij)
      simp only [Nat.succ_mul] at hsep
      omega
  rw [hij] at h
  have hde : d.val = e.val := (Nat.mul_right_cancel_iff hd).mp (by omega)
  have hi : i = j := Fin.ext hij
  have he : d = e := Fin.ext hde
  subst j
  subst e
  rfl

theorem output_injective (BM BD : Nat) (p : Parameters) (s : BlockState)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow) :
    Function.Injective (RealExecution.outputAddress BM BD p s) := by
  intro i j h
  simp only [RealAddressBounds.output_address] at h
  exact strided_injective BM BD _ _ _ _ hd hr h

/-- M, L and Out are distinct output allocations; Q/K/V may still alias
inputs or outputs because their reads finish before the first store. -/
theorem output_layout (BM BD : Nat) (p : Parameters) (s : BlockState)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M)
    (hlo : p.layout.L ≠ p.layout.Out) : RealExecution.OutputLayout BM BD p s :=
  ⟨output_injective BM BD p s hd hr, hm, hlm, hlo⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealOutputLayout

import bench.tritonbench_g.block_sparse_attn.Proofs.RealWindows
import bench.tritonbench_g.block_sparse_attn.Proofs.RealFlatten

/-! The shared flat-memory bridge transports the complete real source:
successful execution, the independent initial-array attention formula and
the frame of every other flat cell. Access safety is derived from arithmetic
input/output windows, not assumed as a trace or whole-body hypothesis. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFlat
open VeriTile.Triton RealAddressFormula
noncomputable section

/-- Observable flat output cells and the complete memory frame. Padding and
inactive query rows are outside the write window, including with EVEN_M. -/
structure Results (A : FlatAlloc) (s t : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs) : Prop where
  first : ∀ (r : Fin BM) (d : Fin BD), RealSetup.rows s BM r < p.context →
    t.mem A.flat (A.addr p.Out (RealWindows.output s BM p r d.val)) =
      MemCell.real (CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) d.val p.scale)
  second : 2 ≤ p.blocks → ∀ (r : Fin BM) (d : Fin BD), RealSetup.rows s BM r < p.context →
    t.mem A.flat (A.addr p.Out (RealWindows.output s BM p r (BD + d.val))) =
      MemCell.real (CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) (BD + d.val) p.scale)
  frame : ∀ region address,
    (∀ (r : Fin BM) (d : Fin BD), RealSetup.rows s BM r < p.context →
      ¬(A.flat = region ∧ A.addr p.Out (RealWindows.output s BM p r d.val) = address)) →
    (2 ≤ p.blocks → ∀ (r : Fin BM) (d : Fin BD), RealSetup.rows s BM r < p.context →
      ¬(A.flat = region ∧ A.addr p.Out (RealWindows.output s BM p r (BD + d.val)) = address)) →
    t.mem region address = (A.flattenState s).mem region address

theorem source_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (swapped : Bool)
    (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters) (xs : CSRMath.Inputs)
    (hm : RealInputs.Memory s BD p xs) (hv : RealInputs.Valid s p xs)
    (hw : RealWindows.Bounds A.extent s BM BN BD p xs) (hu : s.undef = fun _ _ => 0)
    (hN : 0 < BN) (hstride : BD ≤ p.oRow) (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    ∃ t, exec (A.flattenKernel (RealProgram.kernel swapped BM BN BD p).toAlgKernel)
      (A.flattenState s) = some t ∧ Results A s t BM BN BD p xs := by
  obtain ⟨t, he, hf, hs, hframe⟩ := RealFormula.source_correct swapped s BM BN BD p xs hm hv hN hstride htwo
  have hsafe := RealWindows.source_safe A.extent swapped s BM BN BD p xs hm hv hw hN
  have hbridge := A.exec_flatten hd hcov _ s hsafe (RealFlatten.source_ok swapped BM BN BD p) hu
  have hmem (r : Fin BM) (d : Nat) (hdim : d < RealInputs.width BD p)
      (hr : RealSetup.rows s BM r < p.context) : p.Out ∈ A.regions := by
    by_contra h
    have hb := hw.output r d hdim hr
    rw [hcov p.Out h] at hb
    omega
  refine ⟨A.flattenState t, ?_, ?_, ?_, ?_⟩
  · rw [hbridge]
    change (stepStmts _ s).map A.flattenState = _
    rw [he, Option.map_some]
  · intro r d hr
    have hdim := RealInputs.channel_lt BD p Bool.false d (by simp)
    rw [A.flattenState_mem_addr hd t (hmem r d.val hdim hr) (hw.output r d.val hdim hr)]
    have hy := hf r d hr
    rw [RealWindows.output_address] at hy
    change t.mem p.Out (RealWindows.output s BM p r d.val) = _ at hy
    rw [hy]
    rfl
  · intro hb r d hr
    have hdim := RealInputs.channel_lt BD p Bool.true d (fun _ => hb)
    rw [A.flattenState_mem_addr hd t (hmem r (BD + d.val) hdim hr) (hw.output r (BD + d.val) hdim hr)]
    have hy := hs hb r d hr
    rw [RealWindows.output_address] at hy
    change t.mem p.Out (RealWindows.output s BM p r (BD + d.val)) = _ at hy
    rw [hy]
    rfl
  · intro region address hfirst hsecond
    by_cases hr : region = A.flat
    · subst region
      simp only [FlatAlloc.flattenState, ↓reduceIte]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode address with
      | none => rfl
      | some pair =>
        obtain ⟨region, offset⟩ := pair
        obtain ⟨_, ha, _⟩ := A.decode_sound hdec
        apply congrArg A.trCell
        apply hframe region offset
        · intro r d hr hhit
          obtain ⟨hrg, hoff⟩ := hhit
          apply hfirst r d hr
          refine ⟨rfl, ?_⟩
          rw [RealWindows.output_address] at hoff
          change RealWindows.output s BM p r d.val = offset at hoff
          rw [hoff, hrg]
          exact ha.symm
        · intro hb r d hr hhit
          obtain ⟨hrg, hoff⟩ := hhit
          apply hsecond hb r d hr
          refine ⟨rfl, ?_⟩
          rw [RealWindows.output_address] at hoff
          change RealWindows.output s BM p r (BD + d.val) = offset at hoff
          rw [hoff, hrg]
          exact ha.symm
    · simp only [FlatAlloc.flattenState, if_neg hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFlat

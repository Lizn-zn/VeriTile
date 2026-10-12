import bench.tritonbench_g.block_sparse_attn.Proofs.RealInputs
import bench.tritonbench_g.block_sparse_attn.Proofs.RealObservedFormula
import bench.tritonbench_g.block_sparse_attn.Proofs.RealStoreOutputs

/-! The complete source computes the independent CSR attention formula on
every stored row, with original and score-commuted variants. The assumptions
describe initial arrays, valid CSR indices and output address separation.
Allocation safety and transport to flat memory are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFormula
open VeriTile.Triton RealNormalizationValues RealAddressFormula
noncomputable section

theorem accumulated_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : RealInputs.Memory s BD p xs) (hv : RealInputs.Valid s p xs)
    (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks)
    (r : Fin BM) (d : Fin BD) (hr : RealSetup.rows s BM r < p.context) :
    StreamingMath.accumulated (fun n => RealLoop.score s BM BN BD p n r)
      (fun n j => RealLoop.value s BM BN BD p second n j d) (RealLoop.count s p) =
      CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) (channel BD second d) p.scale := by
  rw [RealInputs.count_matches s BD p xs hm]
  apply RealObservedFormula.attention
  · intro n hn j
    exact RealInputs.score_matches s BM BN BD p xs hm hv n hn r j hr
  · intro n hn j hvisible
    apply RealInputs.value_matches s BM BN BD p xs hm hv second hb n hn j d
    omega

theorem accumulator_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : RealInputs.Memory s BD p xs) (hv : RealInputs.Valid s p xs)
    (r : Fin BM) (d : Fin BD) (hr : RealSetup.rows s BM r < p.context) :
    (RealLoop.carry s BM BN BD p (RealLoop.count s p)).accumulator r d =
      CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) d.val p.scale :=
  accumulated_matches s BM BN BD p xs hm hv Bool.false (by simp) r d hr

theorem second_accumulator_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : RealInputs.Memory s BD p xs) (hv : RealInputs.Valid s p xs)
    (hb : 2 ≤ p.blocks) (r : Fin BM) (d : Fin BD) (hr : RealSetup.rows s BM r < p.context) :
    (RealLoop.carry s BM BN BD p (RealLoop.count s p)).secondAccumulator r d =
      CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) (BD + d.val) p.scale :=
  accumulated_matches s BM BN BD p xs hm hv Bool.true (fun _ => hb) r d hr

/-- Full actual source execution, independent initial-array formula and
complete logical-region memory frame. Empty/reversed CSR rows, repeated keys,
partial query tiles and either source direction are all included. -/
theorem source_correct (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : CSRMath.Inputs)
    (hm : RealInputs.Memory s BD p xs) (hv : RealInputs.Valid s p xs)
    (hN : 0 < BN) (hstride : BD ≤ p.oRow) (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    ∃ t, stepStmts (RealProgram.kernel swapped BM BN BD p).toAlgKernel.body s = some t ∧
      (∀ r d, RealSetup.rows s BM r < p.context →
        t.mem p.Out (RealStores.offset s BM BD p Bool.false r d) =
          MemCell.real (CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) d.val p.scale)) ∧
      (2 ≤ p.blocks → ∀ r d, RealSetup.rows s BM r < p.context →
        t.mem p.Out (RealStores.offset s BM BD p Bool.true r d) =
          MemCell.real (CSRMath.attention (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) (BD + d.val) p.scale)) ∧
      (∀ region address,
        (∀ r d, RealSetup.rows s BM r < p.context →
          ¬(p.Out = region ∧ RealStores.offset s BM BD p Bool.false r d = address)) →
        (2 ≤ p.blocks → ∀ r d, RealSetup.rows s BM r < p.context →
          ¬(p.Out = region ∧ RealStores.offset s BM BD p Bool.true r d = address)) →
        t.mem region address = s.mem region address) := by
  obtain ⟨t, hf, he⟩ := RealStores.kernel_executes swapped s BM BN BD p hN
  refine ⟨_, he, ?_, ?_, ?_⟩
  · intro r d hr
    rw [RealStores.first_output s t BM BN BD p _ r d hr hstride htwo,
      accumulator_matches s BM BN BD p xs hm hv r d hr]
  · intro hb r d hr
    rw [RealStores.second_output s t BM BN BD p _ r d hr hb hstride,
      second_accumulator_matches s BM BN BD p xs hm hv hb r d hr]
  · exact RealStores.kernel_frame s t BM BN BD p _ hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealFormula

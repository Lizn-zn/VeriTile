import bench.tritonbench_g.attn_fwd_causal.Proofs.RealContract
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealCounterexample

/-! The exact-mask counterexample satisfies physical allocation, input pins
and active output separation. Both complete erased sources execute through
the flat IO contract, and the first stored result is strictly positive. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFlatCounterexample
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def parameters : RealContract.Parameters where
  Q := "Q"
  K := "K"
  V := "V"
  QS := "QS"
  KS := "KS"
  Out := "Out"
  queryBatch := 1
  queryHead := 1
  queryRow := 1
  queryDim := 1
  keyBatch := 1
  keyHead := 1
  keyRow := 1
  keyDim := 1
  valueBatch := 1
  valueHead := 1
  valueRow := 1
  valueDim := 1
  outputBatch := 1
  outputHead := 1
  outputRow := 1
  outputDim := 1
  batches := 1
  heads := 1
  context := 2
  headDim := 1
  activeHead := 1
  stage := 3
  validStage := by decide

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Q","K","V","QS","KS","Out"]
  base := fun r => if r = "Q" then 0 else if r = "K" then 2 else if r = "V" then 4
    else if r = "QS" then 6 else if r = "KS" then 8 else 10
  extent := fun r => if r ∈ (["Q","K","V","QS","KS","Out"] : List RegionName) then 2 else 0

theorem allocation_disjoint : allocation.Disjoint := by simp +decide [FlatAlloc.Disjoint]

theorem address_bounds :
    RealAccessSafety.Windows RealCounterexample.configuration "Out" 2 2 1 3 allocation.extent := by
  constructor
  · intro row d hr hd
    simpa +decide [RealCounterexample.configuration, RealPrologue.configuration, RealCoordinates.values,
      RealCounterexample.initial, RealEpilogue.address, allocation, Fin.eq_zero d] using hr
  · intro v hv j d hj hd
    have hv0 : v = ⟨0,Bool.true⟩ := by
      simpa [RealFormula.allVisits, RealFormula.firstVisits, RealFormula.secondVisits,
        RealFormula.visits, RealFormula.tripCount, RealCounterexample.configuration,
        RealPrologue.configuration, RealCoordinates.values, RealCounterexample.initial] using hv
    subst v
    simpa +decide [RealCounterexample.configuration, RealPrologue.configuration, RealCoordinates.values,
      RealCounterexample.initial, allocation, Fin.eq_zero d] using hj
  · intro v hv j d hj hd
    have hv0 : v = ⟨0,Bool.true⟩ := by
      simpa [RealFormula.allVisits, RealFormula.firstVisits, RealFormula.secondVisits,
        RealFormula.visits, RealFormula.tripCount, RealCounterexample.configuration,
        RealPrologue.configuration, RealCoordinates.values, RealCounterexample.initial] using hv
    subst v
    simpa +decide [RealCounterexample.configuration, RealPrologue.configuration, RealCoordinates.values,
      RealCounterexample.initial, allocation, Fin.eq_zero d] using hj
  · decide
  · intro v hv
    have hv0 : v = ⟨0,Bool.true⟩ := by
      simpa [RealFormula.allVisits, RealFormula.firstVisits, RealFormula.secondVisits,
        RealFormula.visits, RealFormula.tripCount, RealCounterexample.configuration,
        RealPrologue.configuration, RealCoordinates.values, RealCounterexample.initial] using hv
    subst v
    decide
  · rintro ⟨r,d,⟨⟩⟩ hi
    simpa +decide [RealEpilogue.active, RealEpilogue.address, RealCounterexample.configuration,
      RealPrologue.configuration, RealCoordinates.values, RealCounterexample.initial,
      allocation, Fin.eq_zero d] using hi

theorem executes (candidate : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealCounterexample.kernel candidate))
        (allocation.flattenState RealCounterexample.initial) = some t ∧
      t.readMem "flat" 10 = pow2 (-1000000) / (1 + pow2 (-1000000)) := by
  have proof := RealContract.implements 2 2 1 parameters candidate (by decide)
  obtain ⟨t, he, hy, _⟩ := proof allocation allocation_disjoint
    (by intro r hr; simp_all [allocation]) RealCounterexample.initial
    (RealLoads.inputs RealCounterexample.initial RealCounterexample.configuration 2 2 1)
    rfl rfl address_bounds
    (RealInputPins.pins_from_memory 2 2 1 RealCounterexample.initial RealCounterexample.configuration 3)
    (fun a b _ _ hab => RealCounterexample.output_address_injective a b hab)
  refine ⟨t, he, ?_⟩
  have h := hy ⟨0,0,⟨⟩⟩ (by change RealEpilogue.active RealCounterexample.configuration 2 1 ⟨0,0,⟨⟩⟩ = Bool.true; decide)
  exact h.trans RealCounterexample.formula_value

theorem positive_output (candidate : Bool) :
    ∃ t, exec (allocation.flattenKernel (RealCounterexample.kernel candidate))
        (allocation.flattenState RealCounterexample.initial) = some t ∧ 0 < t.readMem "flat" 10 := by
  obtain ⟨t, he, hv⟩ := executes candidate
  refine ⟨t, he, ?_⟩
  rw [hv]
  exact div_pos (pow2_pos _) (by have h := pow2_pos (-1000000); linarith)

refutation source_is_exact_causal_mask (candidate : Bool) :
    ∀ t, exec (allocation.flattenKernel (RealCounterexample.kernel candidate))
      (allocation.flattenState RealCounterexample.initial) = some t → t.readMem "flat" 10 = 0 := by
  obtain ⟨t, he, hv⟩ := positive_output candidate
  disprove [t, he, ne_of_gt hv]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFlatCounterexample

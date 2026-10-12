import bench.tritonbench_g.attn_fwd_causal.Proofs.RealExecution
import bench.tritonbench_g.attn_fwd_causal.Proofs.FinitePenalty

/-! A complete source-execution counterexample to exact causal masking in
real arithmetic. Two keys have zero scores and values [0,1]. The future key
receives the finite -1e6 penalty, so the first output remains positive.
Both the original and reciprocal source execute from empty registers.
This theorem uses region memory; physical flat-memory safety is separate. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCounterexample
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

def initial : BlockState where
  mem := fun r i => MemCell.real
    (if r = "QS" ∨ r = "KS" then 1 else if r = "V" ∧ i = 1 then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0

def kernel (candidate : Bool) : AlgKernel :=
  ((if candidate then TypedSource.reciprocal else TypedSource.original)
    "Q" "K" "V" "QS" "KS" "Out"
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 2 1 2 2 1 1 3 (by decide)).eraseDType.toAlgKernel

def configuration : RealLoads.Config :=
  RealPrologue.configuration "Q" "K" "V" "QS" "KS" 1 1 1 2 1 1
    (RealCoordinates.values initial 1 2 2 2 1 1 1)

theorem initial_read (r : RegionName) (i : Nat) :
    initial.readMem r i =
      (if r = "QS" ∨ r = "KS" then 1 else if r = "V" ∧ i = 1 then 1 else 0) := rfl

theorem formula_value :
    RealFormula.expected (RealLoads.inputs initial configuration 2 2 1) 0 2 1 3 0 0 =
      pow2 (-1000000) / (1 + pow2 (-1000000)) := by
  norm_num [RealFormula.expected, RealFormula.blocks, RealFormula.allVisits,
    RealFormula.firstVisits, RealFormula.secondVisits, RealFormula.visits,
    RealFormula.tripCount, RealFormula.blockPairs, List.finRange_succ,
    RealFormula.logit, RealFormula.score, RealFormula.penalty, RealFormula.loadedValue,
    RealLoads.inputs, configuration, RealPrologue.configuration, RealCoordinates.values,
    RealCoordinates.ceil, initial_read, Fin.sum_univ_succ]
  simp +decide

theorem output_address_injective (a b : TileIndex [2,1]) :
    RealEpilogue.address configuration 2 1 a = RealEpilogue.address configuration 2 1 b → a = b := by
  rcases a with ⟨a,d,⟨⟩⟩
  rcases b with ⟨b,e,⟨⟩⟩
  have hd : d = 0 := Fin.eq_zero d
  have he : e = 0 := Fin.eq_zero e
  subst d e
  simp [RealEpilogue.address, configuration, RealPrologue.configuration,
    RealCoordinates.values, initial, Fin.ext_iff]

theorem source_output (candidate : Bool) :
    ∃ t, exec (kernel candidate) initial = some t ∧
      t.readMem "Out" 0 = pow2 (-1000000) / (1 + pow2 (-1000000)) := by
  obtain ⟨t, ht, hread, _⟩ := RealExecution.executes initial
    "Q" "K" "V" "QS" "KS" "Out"
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 2 1 2 2 1 1 3 (by decide) candidate (by decide) (by intros; rfl)
  refine ⟨t, ht, ?_⟩
  have h := hread ⟨0,0,⟨⟩⟩ (by decide)
    (fun a b _ _ hab => output_address_injective a b hab)
  change t.readMem "Out" 0 =
    RealFormula.expected (RealLoads.inputs initial configuration 2 2 1) 0 2 1 3 0 0 at h
  exact h.trans formula_value

theorem source_output_positive (candidate : Bool) :
    ∃ t, exec (kernel candidate) initial = some t ∧ 0 < t.readMem "Out" 0 := by
  obtain ⟨t, ht, hv⟩ := source_output candidate
  refine ⟨t, ht, ?_⟩
  rw [hv]
  exact div_pos (pow2_pos _) (by have h := pow2_pos (-1000000); linarith)

/-- Exact causal masking keeps only the zero-valued first key at row zero. -/
refutation source_is_exact_causal_mask (candidate : Bool) :
    ∀ t, exec (kernel candidate) initial = some t → t.readMem "Out" 0 = 0 := by
  obtain ⟨t, ht, hv⟩ := source_output_positive candidate
  disprove [t, ht, ne_of_gt hv]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCounterexample

import bench.tritonbench_g.attention_llama.Correct

/-! The IO contract admits a nonzero causal start position and a later program
ID. The successful execution is nonempty and retains a masked output cell. -/
namespace RealAttentionLlamaIOTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

private def initial : BlockState :=
  { mem := fun r o =>
      if r = "V" then .real (if o = 0 then 7 else 11)
      else if r = "Out" then .real 99 else .real 0
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 0 then 1 else 0 }

private def allocation : FlatAlloc where
  flat := "flat"
  regions := ["Q", "K", "V", "Out"]
  base := fun r => if r = "Q" then 0 else if r = "K" then 8 else if r = "V" then 16 else 24
  extent := fun r => if r = "Q" then 3 else if r ∈ (["K", "V"] : List RegionName) then 6
    else if r = "Out" then 4 else 0

private def contract (candidate : Bool) : StreamMasked3DKernelIO₃ :=
  Correctness.io "Q" "K" "V" "Out" 1
    6 6 1 1 6 6 1 1 6 6 1 1 6 6 1 1 1 3 6 2 2 2 1 3 candidate Bool.true

private def queries : Fin 3 → Fin (2 * 1) → ℝ := fun _ _ => 0
private def keys : Fin 3 → Fin (2 * 1) → ℝ := fun _ _ => 0
private def values : Fin 3 → Fin (2 * 1) → ℝ :=
  fun t j => if t.val * 2 + j.val = 0 then 7 else 11

/-- The contract permits the intended program, but does not grant the same
three-block span to program zero. This is a launch condition, not an output mask. -/
theorem causal_launch_condition (candidate : Bool) :
    (contract candidate).pre 1 0 0 ∧ ¬ (contract candidate).pre 0 0 0 := by
  simp [contract, Correctness.io, RealIO.launch]

/-- Row two only sees key zero because start_position is two. Both complete
source variants produce seven, while the masked row and the V input survive. -/
theorem shifted_causal_io_executes (candidate : Bool) :
    ∃ sF, exec (allocation.flattenKernel (contract candidate).kernel.toAlgKernel)
        (allocation.flattenState initial) = some sF ∧
      sF.readMem "flat" 26 = 7 ∧ sF.mem "flat" 27 = .real 99 ∧
      sF.mem "flat" 16 = .real 7 := by
  have hd : allocation.Disjoint := by
    simp +decide [FlatAlloc.Disjoint]
  have hc : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
    intro r hr
    simp_all [allocation]
  have hi : Function.Injective (fun i : TileIndex [2, 1] => i.1.val * 1 + i.2.1.val * 1) := by
    simpa only [Nat.zero_mul, Nat.zero_add, Nat.mul_one] using
      (rowMajor2D_inj (I := 2) (J := 1) 0 1 (by decide))
  have proof := Correctness.attention_correctness "Q" "K" "V" "Out" 1
    6 6 1 1 6 6 1 1 6 6 1 1 6 6 1 1 1 3 6 2 2 2 1 3 candidate Bool.true
    (by decide) (by decide) (by decide) hi
  obtain ⟨sF, he, hread, hframe⟩ := proof allocation hd rfl hc 1 0 0
    queries keys values initial (causal_launch_condition candidate).1 rfl rfl rfl rfl
    (by
      intro t j hj
      simpa +decide [Correctness.io, allocation, Nat.mod_one] using hj)
    (by
      intro t j _
      simp +decide [Correctness.io, allocation]
      omega)
    (by
      intro t j _
      simp +decide [Correctness.io, allocation]
      omega)
    (by
      intro j _
      simp +decide [Correctness.io, allocation]
      omega)
    (by intros; rfl)
    (by intros; rfl)
    (by
      intro t j _
      simp [Correctness.io, initial, BlockState.readMem, MemCell.real, values, Nat.mod_one])
  refine ⟨sF, he, ?_, ?_, ?_⟩
  · have hr := hread (⟨0, by decide⟩ : Fin (2 * 1)) (by change 2 + 0 < 3; decide)
    change sF.readMem "flat" 26 = _ at hr
    norm_num [Correctness.expected, Correctness.score, Correctness.query, Correctness.stream, RealIO.query, RealIO.stream, RealIO.stepOf, RealIO.laneOf,
      queries, keys, values, attentionRealCausalBlock, scaledScore, Fin.sum_univ_succ,
      Lane2D.encode, Lane2D.decode] at hr
    exact hr
  · have hf := hframe "flat" 27 (Or.inr (by
      intro j hj heq
      simp +decide [Correctness.io, FlatAlloc.addr, allocation] at hj heq
      omega))
    exact hf.trans (by rfl)
  · have hf := hframe "flat" 16 (Or.inr (by
      intro j _ heq
      simp +decide [Correctness.io, FlatAlloc.addr, allocation] at heq
      omega))
    exact hf.trans (by rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end RealAttentionLlamaIOTests

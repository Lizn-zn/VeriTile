import bench.tritonbench_g.max_reduction.Proofs.Real
import bench.tritonbench_g.max_reduction.Proofs.Stage2Real
import bench.tritonbench_g.max_reduction.Proofs.Candidate
import bench.tritonbench_g.max_reduction.Proofs.DimensionIO

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Per-kernel output summary for the 2D value/index `max_kernel`: the DSL
surface lowers to the algorithm layer, and the masked value/index stores are
compute-correct — every active row lane `i` holds the row-wise maximum
(`maxKernelValueSpec`) in `out_value` and its argmax (`maxKernelIndexSpec`) in
`out_index`. Carries the existing side conditions of
`max_kernel_compute_correct`: `hOutInj` (injective output offsets) and
`hOutRegions` (`out_value ≠ out_index`). -/
specification max_kernel_output_summary
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i))
    (hOutRegions : out_value ≠ out_index) :
    (∃ alg, (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.OutputPairWhere
      (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N)
      s out_value out_index
      (maxKernelOutOffset s K BLOCK_M)
      (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
      (maxKernelValueSpec s inp M N K BLOCK_M BLOCK_N)
      (maxKernelIndexSpec s inp M N K BLOCK_M BLOCK_N) := by
  refine ⟨?_, ?_⟩
  · simp [max_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact max_kernel_compute_correct inp out_value out_index M N K BLOCK_M BLOCK_N
      s hOutInj hOutRegions

/-- **The headline on the IO surface** for `max_reduction.py`'s `max_kernel_1`:
for every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose input window holds `xs`
at the active lanes, the translated pointer kernel terminates, `mid[pid]` holds
the genuine block max — `Tile.reduceMax` of `xs` over the active lanes, with the
`other=-float("inf")` padding modeled as `⊥` — and every other memory cell is
unchanged.

This is the first *reduction* on an `io ⊨ f` face, and it is what the spec `f`'s
program-id argument is for: the reduced index set is `{i | pid·BLOCK_SIZE + i < M}`,
so the value is irreducibly pid-dependent (the tail block reduces fewer lanes
than a full block) and a pid-independent spec would be falsifiable.

Dimension-general in `M` and `BLOCK_SIZE`. Honest side-condition:
`0 < BLOCK_SIZE` — with an empty tile `tl.max` has no axis to reduce and the
kernel faults, so termination genuinely fails there. -/
specification max_kernel_1_io_correctness
    (inp mid : RegionName) (M BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE) :
    maxKernel1IO inp mid M BLOCK_SIZE
      ⊨ fun pid xs _ => maxTileSpecOf M BLOCK_SIZE pid xs := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact max_kernel_1_flattenOk inp mid M BLOCK_SIZE
  · intro bounds s h1 h2
    exact max_kernel_1_traceSafe inp mid M BLOCK_SIZE hB bounds s
      (fun i hi => h1 (i, PUnit.unit) hi)
      (h2 (⟨0, hB⟩, PUnit.unit) rfl)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      max_kernel_1_region_run inp mid M BLOCK_SIZE hB s₀ xs
        (fun idx hact => hin idx hact)
    refine ⟨s1, hexec, hval, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    -- a non-degenerate tile always has a write-active lane, which turns the
    -- skin's lane-wise disjunct into the single-cell one
    · exact Or.inr (h (⟨0, hB⟩, PUnit.unit) rfl)


/-- Full second-stage IO correctness against a finite mathematical maximum. -/
specification stage2_math_correct (mid out : RegionName) (M B : Nat) (hB : 0 < B) :
    stage2IO mid out M B ⊨ fun _ xs _ => blockMaximum M B 0 xs := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact stage2_flattenOk mid out M B
  · intro bounds s hi ho
    exact stage2_traceSafe mid out M B hB bounds s
      (fun i h => hi (i, PUnit.unit) h) (ho (⟨0,hB⟩, PUnit.unit) rfl)
  · intro s xs hx
    obtain ⟨t, he⟩ := stage2_terminates mid out M B hB s
    refine ⟨t, he, ?_, ?_⟩
    · intro i hi
      change t.readMem out 0 = blockMaximum M B 0 xs
      rw [max_kernel_2_correct mid out M B s t he]
      exact stage2_formula mid M B hB s xs (fun i h => hx (i, PUnit.unit) h)
    · intro r o hf
      apply stage2_frame mid out M B hB s t he r o
      rcases hf with hr | ho
      · exact Or.inl hr
      · exact Or.inr (ho (⟨0,hB⟩, PUnit.unit) rfl)

def typedStage1IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { maxKernel1IO inp out M B with
    kernel := (stage1 storage v inp out M B).eraseDType
    projection := stage1_supported storage v inp out M B }

def typedStage2IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { stage2IO inp out M B with
    kernel := (stage2 storage v inp out M B).eraseDType
    projection := stage2_supported storage v inp out M B }

/-- Each source version independently implements the finite maximum of the
masked original input window, terminates, and preserves every other flat cell. -/
specification stage1_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage1IO storage v inp out M B ⊨
      fun pid xs _ => blockMaximum M B pid xs) := by
  intro A hd hr hc pid hin hout xs s hp hu hx
  obtain ⟨t, he, hv, hf⟩ := max_kernel_1_io_correctness inp out M B hB
    A hd hr hc pid hin hout xs s hp hu hx
  refine ⟨t, ?_, ?_, hf⟩
  · exact (stage1_flat_exec A storage v inp out M B (A.flattenState s)).trans he
  · intro i hi
    change t.readMem A.flat (A.addr out pid) = blockMaximum M B pid xs
    rw [blockMaximum_eq_tile M B pid hB]
    exact hv i hi

/-- Each second-stage source independently computes the finite maximum of the
masked intermediate window, with successful execution and a full flat frame. -/
specification stage2_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage2IO storage v inp out M B ⊨
      fun _ xs _ => blockMaximum M B 0 xs) := by
  intro A hd hr hc pid hin hout xs s hp hu hx
  obtain ⟨t, he, hv, hf⟩ := stage2_math_correct inp out M B hB
    A hd hr hc pid hin hout xs s hp hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (stage2_flat_exec A storage v inp out M B (A.flattenState s)).trans he

/-- Both full indexed-reduction sources independently compute row maxima and
leftmost indices from the complete masked input window. The signed index
output, value output, successful execution and full flat frame are all covered.
Distinct output regions and injective row addresses prevent conflicting stores. -/
specification dimension_real_correct (storage : Storage) (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (hRegions : ov ≠ oi)
    (hOutInj : Function.Injective (fun i : Fin BM => i.val * K)) :
    VeriTile.Spec.Real (DimensionIO storage v inp ov oi M N K BM BN) := by
  exact dimensionIO_correct storage v inp ov oi M N K BM BN hN hRegions hOutInj

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.MaxReduction

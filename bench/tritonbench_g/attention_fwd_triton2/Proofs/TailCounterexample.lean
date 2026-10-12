import bench.tritonbench_g.attention_fwd_triton2.Proofs.RealProjection
import bench.tritonbench_g.attention_forward_triton.Proofs.TailCounterexample

/-! Both complete real projections include padded scores in the denominator.
The extra V cast and bf16 output do not change that real-arithmetic behavior.
This refutes unqualified length-N attention on a partial KV block; it says
nothing about the admission of an FP atom or a GPU execution. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.TailCounterexample
open VeriTile.Triton

/-- One real key in a two-key block, with the source's actual head layout. -/
def kernel : ComputeKernel :=
  (TypedSource.original "Q" "K" "V" "QS" "KS" "Out"
    128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
    1 1 1 128 1 2 128 96 1).eraseDType

/-- Zero queries/keys and unit active values; masked loads use zero. -/
noncomputable def initial : BlockState where
  mem := fun r _ => MemCell.real (if r = "V" then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  undef := fun _ _ => 0

theorem stored_output :
    (exec kernel initial).map (·.readMem "Out" 0) = some (1 / 2 : ℝ) := by
  unfold kernel
  rw [RealProjection.original_real_source]
  exact AttentionForwardTriton.TailCounterexample.stored_output

/-- Standard attention over exactly the one real key, using the original
input memory and source scale/address maps, with no padded score term. -/
noncomputable def standardOutput : ℝ :=
  attentionRealBase2PerKeyScale
    (AttentionForwardTriton.qTile initial "Q" 1 128 128 128 1 96)
    (AttentionForwardTriton.kTile initial "K" 1 128 128 128 1 96)
    (AttentionForwardTriton.vTile initial "V" 1 128 128 128 1 96)
    (AttentionForwardTriton.keyScale initial "QS" "KS" 1 1 2 1) (0, 0, PUnit.unit)

theorem standard_output : standardOutput = 1 := by
  simp [standardOutput, attentionRealBase2PerKeyScale, AttentionForwardTriton.qTile, AttentionForwardTriton.kTile, AttentionForwardTriton.vTile,
    AttentionForwardTriton.keyScale, AttentionForwardTriton.baseOffset, AttentionForwardTriton.offZ, AttentionForwardTriton.offH, AttentionForwardTriton.mIndex, initial, BlockState.readMem,
    MemCell.real, MemCell.of, MemCell.readAs]

/-- Success is part of the observed counterexample, not an assumption. -/
theorem full_kernel_executes : ∃ t, exec kernel initial = some t := by
  have h := stored_output
  cases he : exec kernel initial with
  | none => simp [he] at h
  | some t => exact ⟨t, rfl⟩

/-- The full original kernel does not compute length-N attention for all
tail sizes. This refutes the displayed real specification, not an FP atom. -/
refutation standard_attention_on_tail :
    (exec kernel initial).map (·.readMem "Out" 0) = some standardOutput := by
  rw [stored_output, standard_output]
  norm_num

/-- Reciprocal normalization leaves the same padded score in the denominator. -/
def reciprocalKernel : ComputeKernel :=
  (TypedSource.reciprocal "Q" "K" "V" "QS" "KS" "Out"
    128 128 128 1 128 128 128 1 128 128 128 1 128 128 128 1
    1 1 1 128 1 2 128 96 1).eraseDType

theorem reciprocal_stored_output :
    (exec reciprocalKernel initial).map (·.readMem "Out" 0) = some (1 / 2 : ℝ) := by
  unfold reciprocalKernel
  rw [RealProjection.reciprocal_real_source]
  exact AttentionForwardTriton.TailCounterexample.reciprocal_stored_output

theorem reciprocal_kernel_executes : ∃ t, exec reciprocalKernel initial = some t := by
  have h := reciprocal_stored_output
  cases he : exec reciprocalKernel initial with
  | none => simp [he] at h
  | some t => exact ⟨t, rfl⟩

refutation reciprocal_standard_attention_on_tail :
    (exec reciprocalKernel initial).map (·.readMem "Out" 0) = some standardOutput := by
  rw [reciprocal_stored_output, standard_output]
  norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton2.TailCounterexample

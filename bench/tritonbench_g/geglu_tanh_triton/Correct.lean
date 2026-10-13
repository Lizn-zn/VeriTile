import bench.tritonbench_g.geglu_tanh_triton.Proofs.Precision
import bench.tritonbench_g.geglu_tanh_triton.Proofs.SourceChecks
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Real
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Math
import bench.tritonbench_g.geglu_tanh_triton.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂
/-- **The forward headline**: `_geglu_tanh_forward_kernel` implements the
tanh-GeGLU forward oracle on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `as`/`bs`,
the translated pointer kernel terminates, every active output-row lane `j`
holds `TiledActivation.geluTanhFwd (as j) (bs j)`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification geglu_tanh_forward_kernel_correctness
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    gegluTanhFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => TiledActivation.geluTanhFwd (as i) (bs i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact geglu_tanh_forward_kernel_flattenOk A B C stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact geglu_tanh_forward_kernel_traceSafe A B C stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := geglu_tanh_forward_kernel_region_run
      A B C stride n_cols BLOCK_SIZE s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- **The backward headline**: `_geglu_tanh_backward_kernel` implements the
tanh-GeGLU backward oracles on its masked in-place IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input-row lanes hold
`dcs`/`as`/`bs`, the translated pointer kernel terminates, every active lane
of the gate buffer ends up holding `geluTanhBwdA` and of the value buffer
`geluTanhBwdB` — applied to the *originally loaded* windows (the standard
before/after reading of an in-place Hoare triple) — and every other memory
cell is unchanged. Assumes the two output regions are distinct (`A ≠ B`) so
the second store cannot clobber the first channel. Proof:
`MaskedKernelIO₃ₓ₂.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification geglu_tanh_backward_kernel_correctness
    (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (hAB : A ≠ B) :
    gegluTanhBwdIO DC A B stride n_cols BLOCK_SIZE ⊨ fun dcs as bs =>
      (fun i => TiledActivation.geluTanhBwdA (dcs i) (as i) (bs i),
       fun i => TiledActivation.geluTanhBwdB (dcs i) (as i)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [gegluTanhBwdIO]) (by simp [gegluTanhBwdIO]) ?_ ?_ ?_
  · exact geglu_tanh_backward_kernel_flattenOk DC A B stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 h3 _ _
    exact geglu_tanh_backward_kernel_traceSafe DC A B stride n_cols BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ dcs as bs hdc ha hb
    exact geglu_tanh_backward_kernel_region_run DC A B stride n_cols BLOCK_SIZE
      hAB s₀ dcs as bs hdc ha hb



/-- Complete forward flat IO against an independent scalar formula using the
source decimal coefficient, with the existing active-lane bounds and full frame. -/
specification forward_formula_correct (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) :
    gegluTanhFwdIO A B C stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => Math.forward (as i) (bs i) := by
  simpa only [Math.forward_eq] using
    geglu_tanh_forward_kernel_correctness A B C stride n_cols BLOCK_SIZE

/-- Both in-place output formulas use the old input windows. The two output
regions are distinct, so the second store preserves the first gradient. -/
specification backward_formula_correct (DC A B : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) (hAB : A ≠ B) :
    gegluTanhBwdIO DC A B stride n_cols BLOCK_SIZE ⊨ fun dcs as bs =>
      (fun i => (Math.backward (dcs i) (as i) (bs i)).1,
       fun i => (Math.backward (dcs i) (as i) (bs i)).2) := by
  simpa only [Math.backward_eq] using
    geglu_tanh_backward_kernel_correctness DC A B stride n_cols BLOCK_SIZE hAB

def typedFwdIO (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ :=
  { gegluTanhFwdIO a b c stride n_cols BLOCK_SIZE with
    kernel := (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).eraseDType
    projection := fwd_supported storage partner v a b c stride n_cols BLOCK_SIZE }

def typedBwdIO (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₃ₓ₂ :=
  { gegluTanhBwdIO dc a b stride n_cols BLOCK_SIZE with
    kernel := (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).eraseDType
    projection := bwd_supported storage partner gradient v dc a b stride n_cols BLOCK_SIZE }

/-- Each complete typed forward version realizes the independent pointwise
formula on the whole logical row, with successful execution and the full frame. -/
specification forward_typed_real_correct (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat)
    (_hB : 0 < BLOCK_SIZE) (_hN : n_cols ≤ BLOCK_SIZE) :
    VeriTile.Spec.Real (typedFwdIO storage partner v a b c stride n_cols BLOCK_SIZE
      ⊨ fun as bs i => Math.forward (as i) (bs i)) := by
  intro A hd hr hc pid hab hbb hcb hs as bs s hp hu ha hb
  obtain ⟨t, he, hv, hf⟩ := forward_formula_correct a b c stride n_cols BLOCK_SIZE
    A hd hr hc pid hab hbb hcb hs as bs s hp hu ha hb
  refine ⟨t, ?_, hv, hf⟩
  exact (fwd_flat_exec A storage partner v a b c stride n_cols BLOCK_SIZE
    (A.flattenState s)).trans he

/-- Each complete typed backward version writes both independent gradient
formulas from the old inputs and preserves all other memory cells. -/
specification backward_typed_real_correct (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) (hab : a ≠ b)
    (_hB : 0 < BLOCK_SIZE) (_hN : n_cols ≤ BLOCK_SIZE) :
    VeriTile.Spec.Real (typedBwdIO storage partner gradient v dc a b stride n_cols BLOCK_SIZE
      ⊨ fun dcs as bs =>
        (fun i => (Math.backward (dcs i) (as i) (bs i)).1,
         fun i => (Math.backward (dcs i) (as i) (bs i)).2)) := by
  intro A hd hr hc pid hdcb habound hbbound haout hbout dcs as bs s hp hu hdc ha hb
  obtain ⟨t, he, hva, hvb, hf⟩ := backward_formula_correct dc a b stride n_cols BLOCK_SIZE hab
    A hd hr hc pid hdcb habound hbbound haout hbout dcs as bs s hp hu hdc ha hb
  refine ⟨t, ?_, hva, hvb, hf⟩
  exact (bwd_flat_exec A storage partner gradient v dc a b stride n_cols BLOCK_SIZE
    (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton

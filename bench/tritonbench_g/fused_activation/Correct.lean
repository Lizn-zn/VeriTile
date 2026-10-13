import bench.tritonbench_g.fused_activation.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.FusedActivation
open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

/-- `fused_add_mul_activation_kernel`'s grouped masked **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`GroupedMasked2DKernelIO`, the vector-channel genre; the general per-lane
windows are what the broadcast bias read needs):

* `bufs = [x_ptr, bias_ptr, in_ptr]` — every buffer once; the output channel
  names `x_ptr` again, i.e. the update is **in place**;
* `nIn = 3` — channel 0 = `x_ptr` at `pid₀·BLOCK_SIZE + j`, channel 1 =
  `bias_ptr` at `(pid₀·BLOCK_SIZE + j) % num_weights` (the broadcast bias),
  channel 2 = `in_ptr` at `pid₀·BLOCK_SIZE + j`;
* `nOut = 1` — `x_ptr` at `pid₀·BLOCK_SIZE + j`;
* every read/write gate is the kernel's single mask `pid₀·BLOCK_SIZE + j <
  xnumel`;
* `B = BLOCK_SIZE`; the kernel is a 1-D launch, so the family's second program
  id is ignored by every field.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the declared lanes. -/
def fusedActivationIO
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) : GroupedMasked2DKernelIO where
  kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights
    xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  nIn := 3
  nOut := 1
  bufs := [x_ptr, bias_ptr, in_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => x_ptr
    | ⟨1, _⟩ => bias_ptr
    | ⟨_ + 2, _⟩ => in_ptr
  out := fun _ => x_ptr
  B := BLOCK_SIZE
  read := fun i pid₀ _ j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_SIZE + j.val
    | ⟨1, _⟩ => (pid₀ * BLOCK_SIZE + j.val) % num_weights
    | ⟨_ + 2, _⟩ => pid₀ * BLOCK_SIZE + j.val
  readMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
  write := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  writeMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel

/-- **The headline**: `fused_add_mul_activation_kernel` implements the fused
biased combine + activation on its grouped IO signature — for every disjoint
flat placement of the three buffers, every program id whose masked lanes are in
bounds, and every launch state whose three read windows (including the
broadcast bias window at `index % num_weights`) hold `xs`, the translated
pointer kernel terminates, every active lane `j` (`pid₀·BLOCK_SIZE + j <
xnumel`) of the *same* buffer `x_ptr` ends up holding
`fusedActivationSpec` — the selected activation of
`multiplier·in + x + bias` on the originally-loaded values — and every other
memory cell, including the out-of-bounds lanes of the window, is unchanged.
Both activation branches (`true` = `tl.sigmoid`, `false` = `tl.maximum(0, ·)`)
are covered: `ACTIVATION_SIGMOID` is a free parameter. Proof:
`GroupedMasked2DKernelIO.Implements.intro` assembles the region-model grouped
triple with the flat-memory bridge side conditions. -/
specification fused_add_mul_activation_kernel_correctness
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) :
    fusedActivationIO x_ptr bias_ptr in_ptr num_weights xnumel BLOCK_SIZE
        multiplier ACTIVATION_SIGMOID
      ⊨ fun _ _ xs _ j =>
          fusedActivationSpec ACTIVATION_SIGMOID
            (xs (⟨0, by decide⟩ : Fin 3) j)
            (xs (⟨1, by decide⟩ : Fin 3) j)
            (xs (⟨2, by decide⟩ : Fin 3) j) multiplier := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [fusedActivationIO]
  · exact fused_add_mul_activation_kernel_flattenOk x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  · intro bounds s hread _hwrite
    exact fused_add_mul_activation_kernel_traceSafe x_ptr bias_ptr in_ptr
      num_weights xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID bounds s
      (fun j hj => hread (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hread (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hread (⟨2, by decide⟩ : Fin 3) j hj)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      fused_add_mul_activation_kernel_region_run x_ptr bias_ptr in_ptr num_weights
        xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID s₀
        (fun j => xs (⟨0, by decide⟩ : Fin 3) j)
        (fun j => xs (⟨1, by decide⟩ : Fin 3) j)
        (fun j => xs (⟨2, by decide⟩ : Fin 3) j)
        (fun j hj => hx (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hx (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hx (⟨2, by decide⟩ : Fin 3) j hj)
    exact ⟨s1, hexec, fun _o j hj => hval j hj,
      fun r o hcond => hframe r o (fun j hj => hcond (⟨0, by decide⟩ : Fin 1) j hj)⟩

/-- The candidate retains all three read windows and the in-place write window. -/
def commutedIO (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) :
    GroupedMasked2DKernelIO :=
  { fusedActivationIO x b i W N B m a with
    kernel := fused_add_mul_activation_commuted_kernel x b i W N B m a
    projection := by cases a <;> rfl }

/-- Mathematical scalar operation, independent of either source program. -/
noncomputable def activationFormula (a : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := multiplier * input + x + bias
  if a then Real.sigmoid z else max 0 z

private theorem some_lt_some (x y : ℝ) :
    @LT.lt (WithBot ℝ) WithBot.instLT (some x) (some y) ↔ x < y := WithBot.coe_lt_coe

private theorem spec_eq_formula (a : Bool) (x bias input multiplier : ℝ) :
    fusedActivationSpec a x bias input multiplier =
      activationFormula a x bias input multiplier := by
  cases a
  · by_cases h : multiplier * input + x + bias < 0
    · simp [fusedActivationSpec, fusedActivationInput, activationFormula,
        ComparableDType.gt, some_lt_some, h, max_eq_left (le_of_lt h)]
    · simp [fusedActivationSpec, fusedActivationInput, activationFormula,
        ComparableDType.gt, some_lt_some, h, max_eq_right (le_of_not_gt h)]
  · rfl

/-- Independent real formula for the original, with full grouped flat-memory IO. -/
specification original_real_correct (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) :
    VeriTile.Spec.Real (fusedActivationIO x b i W N B m a
      ⊨ fun _ _ xs _ j => activationFormula a
        (xs (⟨0, by decide⟩ : Fin 3) j) (xs (⟨1, by decide⟩ : Fin 3) j)
        (xs (⟨2, by decide⟩ : Fin 3) j) m) := by
  simpa only [spec_eq_formula] using
    fused_add_mul_activation_kernel_correctness x b i W N B m a

/-- The candidate implements the same mathematical sigmoid/ReLU formula.
The proof uses real addition commutativity, independently of FP rules. -/
specification commuted_real_correct (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) :
    VeriTile.Spec.Real (commutedIO x b i W N B m a
      ⊨ fun _ _ xs _ j => activationFormula a
        (xs (⟨0, by decide⟩ : Fin 3) j) (xs (⟨1, by decide⟩ : Fin 3) j)
        (xs (⟨2, by decide⟩ : Fin 3) j) m) := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · intro o
    simp [commutedIO, fusedActivationIO]
  · exact commuted_flattenOk x b i W N B m a
  · intro bounds s hr _hw
    exact commuted_traceSafe x b i W N B m a bounds s
      (fun j hj => hr (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hr (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hr (⟨2, by decide⟩ : Fin 3) j hj)
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := fused_add_mul_activation_kernel_region_run
      x b i W N B m a s
      (fun j => xs (⟨0, by decide⟩ : Fin 3) j)
      (fun j => xs (⟨1, by decide⟩ : Fin 3) j)
      (fun j => xs (⟨2, by decide⟩ : Fin 3) j)
      (fun j hj => hx (⟨0, by decide⟩ : Fin 3) j hj)
      (fun j hj => hx (⟨1, by decide⟩ : Fin 3) j hj)
      (fun j hj => hx (⟨2, by decide⟩ : Fin 3) j hj)
    exact ⟨s1, (commuted_exec x b i W N B m a s).trans he, fun _ j hj => (hv j hj).trans (spec_eq_formula _ _ _ _ _),
      fun r o hm => hf r o (fun j hj => hm (⟨0, by decide⟩ : Fin 1) j hj)⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.FusedActivation

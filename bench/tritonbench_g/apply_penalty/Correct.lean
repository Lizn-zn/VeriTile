import bench.tritonbench_g.apply_penalty.Kernels
import bench.tritonbench_g.apply_penalty.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of both sources in Kernels.lean. The independent formula is
  (if logit > 0 then logit / repetition else logit * repetition)
    - count * frequency - presence,
written as TiledOptimizer.lionPenalty. The inputs are the initially loaded
logits and counts, together with the three scalar penalties for this program.
This is an in-place gather/scatter contract for symbolic block size and strides;
it requires active-write injectivity for the per-lane values and includes
termination, active-address bounds and the memory frame. Host launch composition and IEEE arithmetic are outside its scope.
As in the source, stride_logit_s is unused: each row is contiguous. -/
namespace VeriTile.Bench.TritonBenchG.ApplyPenalty
open VeriTile.Triton
open scoped VeriTile.Triton.MetaScatterMasked2DKernelIO₁

/-- Scalar penalties at pid; cumsum bounds at pid and pid+1; ids and counts at
start+j. Logits are read and overwritten at pid * stride_logit_b + ids[j], with
the same tail mask start+j < stop. The IO contract checks injectivity only on
active write lanes, so padding does not need distinct token ids. -/
def applyPenaltyIO
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    MetaScatterMasked2DKernelIO₁ where
  kernel := apply_penalty Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P
  fbuf1 := presence_penalty
  fbuf2 := freqency_penalty
  fbuf3 := repetition_penalty
  mbuf := p_cumsum_seq_len
  idbuf := p_token_ids
  cntbuf := p_token_counts
  inp := Logits
  out := Logits
  B := BLOCK_P
  fwin1 := fun pid₀ _ => pid₀
  fwin2 := fun pid₀ _ => pid₀
  fwin3 := fun pid₀ _ => pid₀
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀ + 1
  readi := fun _ _ m₁ _ j => m₁ + j.val
  readc := fun _ _ m₁ _ j => m₁ + j.val
  read := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
  mask := fun _ _ m₁ m₂ j => m₁ + j.val < m₂
  write := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j

/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
specification apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence) := by
  refine MetaScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact apply_penalty_flattenOk Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  · intro bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB _
    exact traceSafe_pinned Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB
  · intro s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx
    exact apply_penalty_region_run Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx

/-- Identical metadata slots, masks, in-place gather/scatter and active-index requirement. -/
def reciprocalIO (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : MetaScatterMasked2DKernelIO₁ :=
  { applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P with
    kernel := apply_penalty_reciprocal Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
    projection := by rfl }

/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
specification apply_penalty_reciprocal_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (reciprocalIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence) := by
  refine MetaScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact apply_penalty_reciprocal_flattenOk Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  · intro bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB _
    exact reciprocal_traceSafe_pinned Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB
  · intro s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx
    obtain ⟨t, ht, hv, hframe⟩ := apply_penalty_region_run Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx
    exact ⟨t, (reciprocal_exec Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P s).trans ht, hv, hframe⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.ApplyPenalty

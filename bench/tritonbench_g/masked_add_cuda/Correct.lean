import bench.tritonbench_g.masked_add_cuda.Proofs.Real
import bench.tritonbench_g.masked_add_cuda.Proofs.IntegerReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton
open scoped VeriTile.Triton.BoolMasked2DKernelIO₂

/-- `masked_add_kernel`'s masked **in-place IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`mbuf` — which buffer is which argument (data `p_ptr`, running
  buffer `grad_ptr`, boolean veto `p_mask_ptr`);
* `out = in2 = grad_ptr` — the update is **in-place** (duplicate-region
  wiring, the `triton_mul2` precedent): the triple reads the *old* window
  contents into `ys` and asserts the *new* ones;
* `B = BLOCK_SIZE`, all four windows at lane `j` = `pid₀ * BLOCK_SIZE + j`
  (the host-side 1-D `cdiv(n_elements, BLOCK_SIZE)` launch convention; the
  family's second program id is ignored);
* `mask` — the **static** bounds lanes `pid₀ * BLOCK_SIZE + j < n_elements`
  (the trace-safety/input superset);
* `writeMask` — the **data-dependent** store gate
  `pid₀ * BLOCK_SIZE + j < n_elements ∧ bs j = false`: exactly the kernel's
  `mask = mask & ~p_mask` narrowing.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and `p_mask` narrowing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the static-mask lanes. -/
def maskedAddIO (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    BoolMasked2DKernelIO₂ where
  kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
    BLOCK_SIZE
  in1 := p_ptr
  in2 := grad_ptr
  mbuf := p_mask_ptr
  out := grad_ptr
  B := BLOCK_SIZE
  read1 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  read2 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readm := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  write := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  mask := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < n_elements
  writeMask := fun pid₀ _ bs j =>
    pid₀ * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.false

/-- **The headline**: `masked_add_kernel` implements the masked in-place fused
add on its bool-input IO signature — for every disjoint flat placement of the
buffers, every program id whose static-mask lanes are in bounds, and every
launch state whose `p`/`grad`/`p_mask` windows hold `xs`/`ys`/`bs` at the
static-mask lanes, the translated pointer kernel terminates, every
write-active lane `j` (in-bounds and `bs j = false`, the kernel's
`mask & ~p_mask`) of the *same* buffer `grad_ptr` ends up holding
`ys j + xs j * alpha` (the fused update of the originally-loaded window), and
every other memory cell — including the in-bounds lanes vetoed by `p_mask` —
is unchanged. Proof: `BoolMasked2DKernelIO₂.Implements.intro` assembles the
region-model masked in-place triple with the flat-memory bridge side
conditions. -/
specification masked_add_kernel_correctness
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    maskedAddIO grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
      ⊨ fun _ _ _ xs ys j => ys j + xs j * alpha := by
  refine BoolMasked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_ ?_
  · exact masked_add_kernel_flattenOk grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE
  · exact fun _ _ _ j h => h.1
  · intro bounds s h1 h2 h3 _
    exact masked_add_kernel_traceSafe grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ bs xs ys hx hy hb
    exact masked_add_kernel_region_run grad_ptr p_ptr p_mask_ptr n_elements
      alpha BLOCK_SIZE s₀ bs xs ys hx hy hb


/-- Preserved boolean-storage IO, with the commuted candidate source. -/
def boolCommutedIO (g p m : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    BoolMasked2DKernelIO₂ :=
  { maskedAddIO g p m n alpha B with
    kernel := masked_add_commuted_kernel g p m n alpha B
    projection := by rfl }

specification masked_add_bool_commuted_real_correct (g p m : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real (boolCommutedIO g p m n alpha B ⊨
      fun _ _ _ xs ys j => ys j + xs j * alpha) := by
  refine BoolMasked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_ ?_
  · exact masked_add_commuted_kernel_flattenOk g p m n alpha B
  · exact fun _ _ _ _ h => h.1
  · intro bounds s hp hg hm _
    exact masked_add_commuted_kernel_traceSafe g p m n alpha B bounds s hp hg hm
  · intro s bs xs ys hx hy hm
    obtain ⟨s1, he, hv, hf⟩ := masked_add_kernel_region_run g p m n alpha B
      s bs xs ys hx hy hm
    exact ⟨s1, (bool_commuted_exec g p m n alpha B s).trans he, hv, hf⟩

section IntegerMask
open scoped IntMasked2DKernelIO₂

/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
def integerIO (storage : MaskStorage) (candidate : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (integer_kernel storage candidate grad p mask n alpha B).eraseDType
  projection := integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0

/-- Both complete tested source profiles implement the in-place mathematical
formula. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
specification masked_add_integer_real_correct (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((integerIO storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha)) := by
  refine IntMasked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_ ?_
  · exact integer_flattenOk storage candidate grad p mask n alpha B
  · exact fun _ _ _ _ h => h.1
  · intro bounds s hp hg hm _
    exact integer_traceSafe storage candidate grad p mask n alpha B bounds s hp hg hm
  · intro s ms xs ys hx hy hm
    obtain ⟨s1, he, hv, hf⟩ := integer_math_kernel_region_run grad p mask n alpha B
      s ms xs ys hx hy hm
    exact ⟨s1, (integer_exec storage candidate grad p mask n alpha B s).trans he, hv, hf⟩
end IntegerMask

section FP64IntegerMask
open scoped IntMasked2DKernelIO₂

/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
def fp64IntegerIO (storage : MaskStorage) (candidate : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (integer_fp64_kernel storage candidate grad p mask n alpha B).eraseDType
  projection := fp64_integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0

/-- Both float64-region transcriptions satisfy the mathematical formula.
This is a real-only result; the pending FP translation gap is documented in README. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
specification masked_add_fp64_integer_real_correct (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((fp64IntegerIO storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha)) := by
  refine IntMasked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_ ?_
  · exact fp64_integer_flattenOk storage candidate grad p mask n alpha B
  · exact fun _ _ _ _ h => h.1
  · intro bounds s hp hg hm _
    exact fp64_integer_traceSafe storage candidate grad p mask n alpha B bounds s hp hg hm
  · intro s ms xs ys hx hy hm
    obtain ⟨s1, he, hv, hf⟩ := integer_math_kernel_region_run grad p mask n alpha B
      s ms xs ys hx hy hm
    exact ⟨s1, (fp64_integer_exec storage candidate grad p mask n alpha B s).trans he, hv, hf⟩
end FP64IntegerMask

section FullIntegerMask
open scoped IntMasked2DKernelIO₂

/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
def fullIntegerIO (gradType pType : FloatStorage) (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (typed_integer_kernel gradType pType storage candidate grad p mask n alpha B).eraseDType
  projection := full_integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0

/-- Both complete typed source profiles implement the in-place mathematical
formula. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
specification masked_add_typed_integer_real_correct (gradType pType : FloatStorage)
    (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((fullIntegerIO gradType pType storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha)) := by
  refine IntMasked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_ ?_
  · exact full_integer_flattenOk gradType pType storage candidate grad p mask n alpha B
  · exact fun _ _ _ _ h => h.1
  · intro bounds s hp hg hm _
    exact full_integer_traceSafe gradType pType storage candidate grad p mask n alpha B bounds s hp hg hm
  · intro s ms xs ys hx hy hm
    obtain ⟨s1, he, hv, hf⟩ := integer_math_kernel_region_run grad p mask n alpha B
      s ms xs ys hx hy hm
    exact ⟨s1, (full_integer_exec gradType pType storage candidate grad p mask n alpha B s).trans he, hv, hf⟩
end FullIntegerMask

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaskedAddCuda

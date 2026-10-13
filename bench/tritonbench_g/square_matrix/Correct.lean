import bench.tritonbench_g.square_matrix.Proofs.Real
import bench.tritonbench_g.square_matrix.Proofs.Candidate
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `square_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def squareIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := square_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols

/-- **The headline**: `square_kernel` implements the lane-wise square
`xs i * xs i` on its masked IO signature — for every disjoint flat placement
of the two buffers, every program id whose active lanes are in bounds, and
every launch state whose active input-row lanes hold `xs`, the translated
pointer kernel terminates, every active output-row lane holds `xs i * xs i`,
and every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification square_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    squareIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => xs i * xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact square_kernel_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact square_kernel_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := square_kernel_region_run output_ptr
      input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Source-facing IO retains explicit storage in the Compute source and uses its
checked real projection for the independent mathematical contract. -/
def typedIO (storage : Storage) (commuted : Bool) (out inp : RegionName)
    (si so n B : Nat) : MaskedKernelIO₁ :=
  { squareIO out inp si so n B with
    kernel := (typedKernel storage commuted out inp si so n B).eraseDType
    projection := typed_supported storage commuted out inp si so n B }

/-- Both complete typed source versions square the active input lanes. This
includes successful flat execution and the full frame outside active writes;
different input/output row strides and partial final tiles remain symbolic. -/
specification square_typed_real_correct (storage : Storage) (commuted : Bool)
    (out inp : RegionName) (si so n B : Nat) (_hB : 0 < B) :
    VeriTile.Spec.Real (typedIO storage commuted out inp si so n B ⊨
      fun xs i => xs i * xs i) := by
  intro A hd hr hc pid hin hout hscratch xs s hpid hu hx
  obtain ⟨t, he, hv, hf⟩ := square_kernel_correctness out inp si so n B
    A hd hr hc pid hin hout hscratch xs s hpid hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A storage commuted out inp si so n B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.SquareMatrix

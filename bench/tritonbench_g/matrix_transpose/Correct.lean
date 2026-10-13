import bench.tritonbench_g.matrix_transpose.Proofs.Real
import bench.tritonbench_g.matrix_transpose.Proofs.Candidate
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁

/-- `kernel`'s **IO signature** — the whole kernel-specific audit surface of
the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`M`, `Out`);
* `B = D_HEAD * SIZE_M` — the output tile flattened row-major into lanes,
  lane `j` = logical output cell `(j / SIZE_M, j % SIZE_M)` (`laneIdx`);
* `read` — lane `j`'s **transposed** source address
  `m * matrix_stridex + d * matrix_stridey`;
* `write` — lane `j`'s output address `d * out_stridex + m * out_stridey`;
* `mask` — every lane (the kernel's load and store are both unmasked), so
  `writeMask` keeps the struct default.

The grid is `(1,)`, so no field depends on either program id. -/
def transposeIO (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    Masked2DKernelIO₁ where
  kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
    SIZE_M D_HEAD
  inp := M
  out := Out
  B := D_HEAD * SIZE_M
  read := fun _ _ j =>
    matrixAddr matrix_stridex matrix_stridey (Lane2D.decode j)
  write := fun _ _ j =>
    outAddr out_stridex out_stridey (Lane2D.decode j)
  mask := fun _ _ _ => True

/-- **The headline**: `kernel` implements the matrix transpose on its IO
signature — for every disjoint flat placement of `M`/`Out`, every program id
whose lanes are in bounds, and every launch state whose transposed source
window holds the tile `xs`, the translated pointer kernel terminates, output
lane `j` holds `xs j` (i.e. the output cell `(d, m)` holds the source cell
`(m, d)` — the transposition lives in the signature's windows, which this
theorem proves the kernel actually uses), and every other memory cell is
unchanged.

`hOutInj` is an honest, **truth-required** side condition: two distinct output
lanes must not alias the same memory cell, otherwise only the last store
survives and the per-lane claim is false. The host's contiguous `out` buffer
satisfies it. -/
specification kernel_correctness
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        outAddr out_stridex out_stridey idx)) :
    transposeIO M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD
      ⊨ fun _ _ xs j => xs j := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact kernel_flattenOk M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD
  · intro bounds s hin hout _
    exact kernel_traceSafe M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD bounds s
      (fun j => hin j trivial) (fun j => hout j trivial)
  · intro s₀ xs hx
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      kernel_region_run M Out matrix_stridex matrix_stridey out_stridex
        out_stridey SIZE_M D_HEAD s₀ xs hOutInj (fun j => hx j trivial)
    exact ⟨s1, hs1, fun j _ => hval j, fun r o hcond _ =>
      hframe r o (by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · exact Or.inr fun j => hno j trivial)⟩


/-- The full typed source retains the independently declared transposed input
window and strided output window under its checked mathematical projection. -/
def typedIO (v : Bool) (M Out : RegionName) (sx sy ox oy m d : Nat) : Masked2DKernelIO₁ :=
  { transposeIO M Out sx sy ox oy m d with
    kernel := (typedKernel v M Out sx sy ox oy m d).eraseDType
    projection := typed_supported v M Out sx sy ox oy m d }

/-- Both float16 source versions transpose the input window, terminate with
successful flat execution, and preserve every cell outside the output window.
Output-address injectivity is necessary for a per-lane result. -/
specification transpose_typed_real_correct (v : Bool) (M Out : RegionName)
    (sx sy ox oy m d : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [d, m] => outAddr ox oy idx)) :
    VeriTile.Spec.Real (typedIO v M Out sx sy ox oy m d ⊨ fun _ _ xs j => xs j) := by
  intro A hd hr hc pid0 pid1 hin hout hscratch xs s hpid0 hpid1 hu hx
  obtain ⟨t, he, hv, hf⟩ := kernel_correctness M Out sx sy ox oy m d hOutInj
    A hd hr hc pid0 pid1 hin hout hscratch xs s hpid0 hpid1 hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A v M Out sx sy ox oy m d (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose

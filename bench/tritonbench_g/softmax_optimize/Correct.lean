import bench.tritonbench_g.softmax_optimize.Proofs.Contracts
import bench.tritonbench_g.softmax_optimize.Proofs.Witnesses

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Per-kernel output summary for the one-tile slice of
`softmax_kernel_online_v2`: the slice lowers to the algorithm layer, and the
masked store to `output_ptr` is compute-correct — every active lane (`i < N`)
holds the stable-softmax value `softmaxOptimizeSpec`, out-of-bounds lanes are
preserved. (Covers the `N ≤ TILE_N` collapsed-loop regime; the full online
surface's exec-level value correctness is established separately, see
`softmax_kernel_online_v2_surface_exec_correct`.) -/
specification softmax_kernel_online_v2_one_tile_output_summary
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    (∃ alg, (softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_online_v2_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxOptimizeSpec s input_ptr N TILE_N i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact softmax_kernel_online_v2_one_tile_compute_correct output_ptr input_ptr N TILE_N s

/-- **Public output summary (headline theorem).** The full online-softmax
surface `softmax_kernel_online_v2_surface` realizes the genuine full-row softmax
`softmaxOptimizeFullSpec` at every output column `j < N` of row `pid`: the total
write map sends column `j` to `output_ptr` at `linearOffset s N j`, and the
value read back there after any successful run is exactly the numerically
stabilized full-row softmax of the loaded input row. Stated via
`ComputeCorrect.Realizes_without_Rounding` (per `bench/MAIN_THEOREM_CONVENTIONS.md` §4), wrapping
the exec-level engine lemma `softmax_kernel_online_v2_surface_exec_correct`. -/
specification softmax_kernel_online_v2_output_summary
    (output_ptr input_ptr : RegionName) (M N TILE_N : Nat)
    (hN : 0 < N) (hT : 0 < TILE_N) (hne : output_ptr ≠ input_ptr)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_online_v2_surface output_ptr input_ptr M N TILE_N)
      (initialState := s)
      (write := fun j : Fin N => some (output_ptr, linearOffset s N j))
      (expected := fun j : Fin N => softmaxOptimizeFullSpec s input_ptr N j) := by
  unfold ComputeCorrect.Realizes_without_Rounding
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_online_v2_surface, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro j
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact softmax_kernel_online_v2_surface_exec_correct output_ptr input_ptr M N TILE_N
    hN hT hne s s' hExec j

section IOFace

/-- **The headline on the IO surface** for `softmax_optimize.py`'s one-tile path:
for every disjoint flat placement of `input_ptr` / `output_ptr`, every program id
whose active lanes are in bounds, and every launch state whose input row holds
`xs` on the `i < N` lanes, the kernel terminates, every active lane of the output
row holds `softmaxTileSpec N TILE_N xs i`, and every other memory cell is
unchanged.

`softmaxTileSpec` is the port's own reduction chain — running max, shift,
`exp`, running sum, divide — read over the **loaded lane values** instead of over
memory, so this is a genuine input/output statement rather than a read-back of
the kernel's own buffer. Masked-out lanes stay `⊥`, exactly as the kernel's
`other = -float("inf")` load leaves them, which is what makes the face true for a
partial tile (`N < TILE_N`, the case the one-tile path is *for*) and not only for
a full one.

Dimension-general in `N` and `TILE_N`. **Zero side-conditions on addresses**: the
window is `pid·N + i`, so injectivity is discharged inline. The one honest
hypothesis is `0 < TILE_N` — an empty tile makes `tl.max` fault, exactly as in
Python. -/
specification softmax_kernel_online_v2_one_tile_io_correctness
    (output_ptr input_ptr : RegionName) (N TILE_N : Nat) (hT : 0 < TILE_N) :
    oneTileIO output_ptr input_ptr N TILE_N
      ⊨ fun _pid xs idx => softmaxTileSpec N TILE_N xs idx.1 := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact oneTile_flattenOk output_ptr input_ptr N TILE_N
  · intro bounds s h1 h2
    exact oneTile_traceSafe output_ptr input_ptr N TILE_N hT bounds s
      (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      oneTile_region_run output_ptr input_ptr N TILE_N hT s₀ xs
        (fun j hj => hx j hj)
    exact ⟨s1, hexec, fun idx hidx => hval idx.1 hidx, fun r n hc =>
      hframe r n (by
        rcases hc with hr | hn
        · exact Or.inl hr
        · exact Or.inr fun i hi => hn (i, PUnit.unit) hi)⟩

end IOFace
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Correctness
open VeriTile Triton FullSource

/-- Complete signed four-loop original: successful full-row softmax IO, exact
row frame and access safety, with the flat bridge's zero-undef premise explicit.
The numerical domain excludes never-active accumulator lanes; the exact source
and its FP equivalence separately retain N=0 signed control flow. -/
specification full_source_original_correct
    (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.false Y X M N B) Y X N := by
  exact original_contract input output A hd hcov s Y X M N B hdom hYX hr hX hY hYr

/-- Complete signed four-loop candidate: successful full-row softmax IO, exact
row frame and access safety, with the flat bridge's zero-undef premise explicit.
The numerical domain excludes never-active accumulator lanes; the exact source
and its FP equivalence separately retain N=0 signed control flow. -/
specification full_source_candidate_correct
    (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.true Y X M N B) Y X N := by
  exact candidate_contract input output A hd hcov s Y X M N B hdom hYX hr hX hY hYr

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Correctness

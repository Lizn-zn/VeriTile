import bench.tritonbench_g.softmax_flaggems.Proofs.Contracts
import bench.tritonbench_g.softmax_flaggems.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Correctness
open VeriTile Triton FullSource
open scoped VeriTile.Spec

/-! Complete source contracts for both versions of all four kernels. Real inputs
are finite mathematical reals. Forward results are full-column softmax; backward
results retain the source's arbitrary masked-load padding contribution. See the
separate formula lemmas for the conventional VJP under vanishing padding.
SourceProfileSupported excludes original fp64 fallback specializations whose
loop-carried types change. ForwardSourceDomain ensures every fallback lane has
an initially active finite input. Mathematical projection lemmas remain more general. -/

specification forward_non_inner_correctness (input output : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N K B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported input one)
    (_hforward : ForwardSourceDomain one N B)
    (h : RealForwardNonInner.ExecutionDomain one X Y N B)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ A.extent X)
    (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_kernel_non_inner input output Y X M N K B C one)
      (RealForwardNonInner.IOResult s · Y X N K C) (RealForwardNonInner.FlatIOResult A s · Y X N K C) ∧
    SourceCorrect A s (softmax_kernel_non_inner_candidate input output Y X M N K B C one)
      (RealForwardNonInner.IOResult s · Y X N K C) (RealForwardNonInner.FlatIOResult A s · Y X N K C) := by
  constructor
  · refine ⟨RealForwardNonInner.original_all_safe input output A.extent s Y X M N K B C one h hr hX hY,
      RealForwardNonInner.original_all_io input output s Y X M N K B C one h, ?_⟩
    intro hu
    exact RealForwardNonInner.original_all_flat_io input output A hd hcov s Y X M N K B C one h hr hX hY hYr hu
  · refine ⟨RealForwardNonInner.candidate_all_safe input output A.extent s Y X M N K B C one h hr hX hY,
      RealForwardNonInner.candidate_all_io input output s Y X M N K B C one h, ?_⟩
    intro hu
    exact RealForwardNonInner.candidate_all_flat_io input output A hd hcov s Y X M N K B C one h hr hX hY hYr hu

specification forward_inner_correctness (input output : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported input one)
    (_hforward : ForwardSourceDomain one N B)
    (h : RealForwardInner.ExecutionDomain one X Y N B)
    (hr : s.pids 0 < M) (hX : M * N ≤ A.extent X)
    (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_kernel_inner input output Y X M N B one)
      (RealForwardInner.IOResult s · Y X N) (RealForwardInner.FlatIOResult A s · Y X N) ∧
    SourceCorrect A s (softmax_kernel_inner_candidate input output Y X M N B one)
      (RealForwardInner.IOResult s · Y X N) (RealForwardInner.FlatIOResult A s · Y X N) := by
  constructor
  · refine ⟨RealForwardInner.original_all_safe input output A.extent s Y X M N B one h hr hX hY,
      RealForwardInner.original_all_io input output s Y X M N B one h, ?_⟩
    intro hu
    exact RealForwardInner.original_all_flat_io input output A hd hcov s Y X M N B one h hr hX hY hYr hu
  · refine ⟨RealForwardInner.candidate_all_safe input output A.extent s Y X M N B one h hr hX hY,
      RealForwardInner.candidate_all_io input output s Y X M N B one h, ?_⟩
    intro hu
    exact RealForwardInner.candidate_all_flat_io input output A hd hcov s Y X M N B one h hr hX hY hYr hu

specification backward_non_inner_correctness (storage : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N K B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported storage one)
    (h : RealBackwardNonInner.ExecutionDomain one P G Y N B)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ A.extent P)
    (hG : M * N * K ≤ A.extent G)
    (hY : M * N * K ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_backward_kernel_non_inner storage P G Y M N K B C one)
      (RealBackwardNonInner.IOResult s · P G Y N K B C one) (RealBackwardNonInner.FlatIOResult A s · P G Y N K B C one) ∧
    SourceCorrect A s (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C one)
      (RealBackwardNonInner.IOResult s · P G Y N K B C one) (RealBackwardNonInner.FlatIOResult A s · P G Y N K B C one) := by
  constructor
  · refine ⟨RealBackwardNonInner.original_all_safe storage A.extent s P G Y M N K B C one h hr hP hG hY,
      RealBackwardNonInner.original_all_io storage s P G Y M N K B C one h, ?_⟩
    intro hu
    exact RealBackwardNonInner.original_all_flat_io storage A hd hcov s P G Y M N K B C one h hr hP hG hY hYr hu
  · refine ⟨RealBackwardNonInner.candidate_all_safe storage A.extent s P G Y M N K B C one h hr hP hG hY,
      RealBackwardNonInner.candidate_all_io storage s P G Y M N K B C one h, ?_⟩
    intro hu
    exact RealBackwardNonInner.candidate_all_flat_io storage A hd hcov s P G Y M N K B C one h hr hP hG hY hYr hu

specification backward_inner_correctness (storage : Storage)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (P G Y : RegionName) (M N B C : Nat) (one : Bool)
    (_hprofile : SourceProfileSupported storage one)
    (h : RealBackwardInner.ExecutionDomain one P G Y N B)
    (hP : M * N ≤ A.extent P)
    (hG : M * N ≤ A.extent G)
    (hY : M * N ≤ A.extent Y)
    (hYr : Y ∈ A.regions) :
    SourceCorrect A s (softmax_backward_kernel_inner storage P G Y M N C B one)
      (RealBackwardInner.IOResult s · P G Y M N B C one) (RealBackwardInner.FlatIOResult A s · P G Y M N B C one) ∧
    SourceCorrect A s (softmax_backward_kernel_inner_candidate storage P G Y M N C B one)
      (RealBackwardInner.IOResult s · P G Y M N B C one) (RealBackwardInner.FlatIOResult A s · P G Y M N B C one) := by
  constructor
  · refine ⟨RealBackwardInner.original_all_safe storage A.extent s P G Y M N B C one h hP hG hY,
      RealBackwardInner.original_all_io storage s P G Y M N B C one h, ?_⟩
    intro hu
    exact RealBackwardInner.original_all_flat_io storage A hd hcov s P G Y M N B C one h hP hG hY hYr hu
  · refine ⟨RealBackwardInner.candidate_all_safe storage A.extent s P G Y M N B C one h hP hG hY,
      RealBackwardInner.candidate_all_io storage s P G Y M N B C one h, ?_⟩
    intro hu
    exact RealBackwardInner.candidate_all_flat_io storage A hd hcov s P G Y M N B C one h hP hG hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.Correctness

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Compute-facing correctness for the inner one-tile FlagGems softmax. -/
specification softmax_kernel_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, linearOffset s N i)))
      (expected := fun i => softmaxFlaggemsSpec s input_ptr N TILE_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_inner_one_tile]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_inner_one_tile_correct output_ptr input_ptr N TILE_N
    s s' hExec i
  simpa [hActive] using h


/-- Compute-facing correctness for the non-inner one-tile FlagGems softmax. -/
specification softmax_kernel_non_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_kernel_non_inner_one_tile_surface
        output_ptr input_ptr N K TILE_N TILE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K)
        (fun idx => (output_ptr, nonInnerOffset s N K TILE_K idx)))
      (expected := fun idx =>
        softmaxFlaggemsNonInnerSpec s input_ptr N K TILE_N TILE_K idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_non_inner_one_tile_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := softmax_kernel_non_inner_one_tile_correct output_ptr input_ptr
    N K TILE_N TILE_K s s' hRange hExec idx
  simpa [hActive] using h


/-- Compute-facing correctness for the inner one-tile FlagGems softmax
backward. -/
specification softmax_backward_kernel_inner_one_tile_compute_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat)
    (s : BlockState)
    (hOffInj : Function.Injective
      (fun idx : TileIndex [TILE_M, TILE_N] => innerBwdOffset s N TILE_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_backward_kernel_inner_one_tile_surface
        out_ptr out_grad_ptr in_grad_ptr M N TILE_M TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_M, TILE_N] =>
          innerBwdActive s M N TILE_M idx)
        (fun idx => (in_grad_ptr, innerBwdOffset s N TILE_M idx)))
      (expected := fun idx =>
        innerBwdSpec s out_ptr out_grad_ptr M N TILE_M TILE_N idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_backward_kernel_inner_one_tile_surface,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := softmax_backward_kernel_inner_one_tile_correct out_ptr out_grad_ptr
    in_grad_ptr M N TILE_M TILE_N s s' hOffInj hExec idx
  simpa [hActive] using h

/-- Compute-facing correctness for the non-inner one-tile FlagGems softmax
backward. -/
specification softmax_backward_kernel_non_inner_one_tile_compute_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := softmax_backward_kernel_non_inner_one_tile_surface
        out_ptr out_grad_ptr in_grad_ptr N K TILE_N TILE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K)
        (fun idx => (in_grad_ptr, nonInnerOffset s N K TILE_K idx)))
      (expected := fun idx =>
        nonInnerBwdSpec s out_ptr out_grad_ptr N K TILE_N TILE_K idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_backward_kernel_non_inner_one_tile_surface,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := softmax_backward_kernel_non_inner_one_tile_correct out_ptr
    out_grad_ptr in_grad_ptr N K TILE_N TILE_K s s' hRange hExec idx
  simpa [hActive] using h

/-- **The headline on the IO surface**: the inner one-tile FlagGems softmax
implements the exact stable softmax over the active row prefix. `0 < TILE_N` is
required — the `max` reduce is only defined on non-empty tiles. -/
specification softmax_kernel_inner_one_tile_correctness
    (output_ptr input_ptr : RegionName) (N TILE_N : Nat) (hT : 0 < TILE_N) :
    softmaxFlaggemsInnerIO output_ptr input_ptr N TILE_N
      ⊨ fun xs i => softmaxFlaggemsSpecOf N TILE_N xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_inner_one_tile_flattenOk output_ptr input_ptr N TILE_N
  · intro bounds s h1 h2 _
    exact softmax_kernel_inner_one_tile_traceSafe output_ptr input_ptr N TILE_N
      hT bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      softmax_kernel_inner_one_tile_region_run output_ptr input_ptr N TILE_N hT
        s₀ xs hx
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems

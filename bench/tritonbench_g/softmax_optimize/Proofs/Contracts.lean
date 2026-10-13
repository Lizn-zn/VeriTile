import bench.tritonbench_g.softmax_optimize.Proofs.FlatIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Correctness
open VeriTile Triton FullSource SignedTransport

/-- Independent full-row formula and the complete raw-memory frame. -/
def RegionResult (Y X : RegionName) (N : Nat) (s t : BlockState) : Prop :=
  (∀ j : Fin N, t.readMem Y (linearOffset s N j) = softmaxOptimizeFullSpec s X N j) ∧
    RowFrame Y N s t

def FlatResult (A : FlatAlloc) (Y X : RegionName) (N : Nat) (s t : BlockState) : Prop :=
  (∀ j : Fin N, t.readMem A.flat (A.addr Y (linearOffset s N j)) = softmaxOptimizeFullSpec s X N j) ∧
    BlockState.WriteWithin
      (SoftmaxFlaggems.RealFlatFrame.footprint A (FullIO.footprint Y s.pid N)) (A.flattenState s) t

/-- Successful exact-source IO and access safety. The region clause retains
arbitrary undefined memory; zero undef appears only in the existing flat bridge. -/
def SourceCorrect (A : FlatAlloc) (s : BlockState) (k : ComputeKernel)
    (Y X : RegionName) (N : Nat) : Prop :=
  k.eraseDType.toAlgKernel.TraceSafe A.extent s ∧
    (∃ t, exec k.eraseDType.toAlgKernel s = some t ∧ RegionResult Y X N s t) ∧
    (s.undef=(fun _ _ ↦ 0) →
      ∃ t, exec (A.flattenKernel k.eraseDType.toAlgKernel) (A.flattenState s) = some t ∧
        FlatResult A Y X N s t)

theorem original_contract (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.false Y X M N B) Y X N := by
  have hN := numericalDomain_positive hdom
  refine ⟨Safety.original_source_safe input output A.extent s Y X M N B hN hdom.1 hr hX hY,
    ?_, ?_⟩
  · exact original_io input output Y X M N B hN hdom.1 hYX s
  · intro hu
    exact FullIO.original_flat_io input output A hd hcov s Y X M N B hN hdom.1 hYX hr hX hY hYr hu

theorem candidate_contract (input : Storage) (output : Accumulation)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ r, r∉A.regions → A.extent r=0)
    (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hdom : NumericalDomain N B) (hYX : Y≠X) (hr : s.pids 0<M)
    (hX : M*N≤A.extent X) (hY : M*N≤A.extent Y) (hYr : Y∈A.regions) :
    SourceCorrect A s (kernel input output Bool.true Y X M N B) Y X N := by
  have hN := numericalDomain_positive hdom
  refine ⟨Safety.candidate_source_safe input output A.extent s Y X M N B hN hdom.1 hr hX hY,
    ?_, ?_⟩
  · exact candidate_io input output Y X M N B hN hdom.1 hYX s
  · intro hu
    exact FullIO.candidate_flat_io input output A hd hcov s Y X M N B hN hdom.1 hYX hr hX hY hYr hu

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Correctness

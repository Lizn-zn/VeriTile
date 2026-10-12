import bench.tritonbench_g.f8_conversion_utils.Proofs.RealIO
/-! Precision-erased correctness of the complete fp8 conversion kernels:
each active output equals its initially loaded input value. Actual destination
casts remain in the typed kernels and in FP equivalence; this real contract
makes no quantization or round-trip identity claim. Finite bounds, full
execution and tagged-cell frame are explicit. Region names may alias.
Host reinterpret/launches, grid races and fixed-width overflow are excluded. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val

def originalToHalf (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toHalf format Bool.false Y X N B).eraseDType

def indexCommutedToHalf (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toHalf format Bool.true Y X N B).eraseDType

def originalToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toFP8 input format Bool.false Y X N B).eraseDType

def indexCommutedToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toFP8 input format Bool.true Y X N B).eraseDType

structure Bounds (Y X : RegionName) (N B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<N → offset u B j<bounds X
  output : ∀ j : Fin B, offset u B j<N → offset u B j<bounds Y

def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)

/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
def correct_target (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)

specification f8_to_f16_correctness (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (originalToHalf format Y X N B) Y X N B) := by
  intro A hd hc u hu hb
  exact RealIO.to_half_correct format Y X N B A hd hc u hu ⟨hb.input,hb.output⟩

specification f8_to_f16_index_commuted_correctness (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (indexCommutedToHalf format Y X N B) Y X N B) := by
  intro A hd hc u hu hb
  exact RealIO.to_half_candidate_correct format Y X N B A hd hc u hu ⟨hb.input,hb.output⟩

specification to_fp8_correctness (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (originalToFP8 input format Y X N B) Y X N B) := by
  intro A hd hc u hu hb
  exact RealIO.to_fp8_correct input format Y X N B A hd hc u hu ⟨hb.input,hb.output⟩

specification to_fp8_index_commuted_correctness (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (indexCommutedToFP8 input format Y X N B) Y X N B) := by
  intro A hd hc u hu hb
  exact RealIO.to_fp8_candidate_correct input format Y X N B A hd hc u hu ⟨hb.input,hb.output⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.Correctness

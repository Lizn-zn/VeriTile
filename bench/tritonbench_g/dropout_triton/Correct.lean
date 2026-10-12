import bench.tritonbench_g.dropout_triton.Proofs.RealIO
/-! Complete-source inverted dropout over initial real data and a given keep
buffer. Integer keep values use nonzero truthiness; bool values are read as
bools. Finite bounds, successful flat execution and the full tagged memory
frame are explicit below. Input/output region names may alias because all
reads precede the store. Both storage versions erase precision only for the
real formula. Division here is Lean's total real division (including p=1).
Mask generation, host launches, concurrent races and physical rounding are
outside this per-program specification. -/
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def original (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.false x k y n B p).eraseDType

def indexCommuted (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.true x k y n B p).eraseDType

def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val

def keepBit (keep : KeepStorage) (u : BlockState) (r : RegionName) (o : Nat) : Bool :=
  match keep with
  | .int32 => decide (u.readMemValue .int r o≠0)
  | .bool => u.readMemValue .bool r o

def expected (keep : KeepStorage) (u : BlockState) (x k : RegionName) (p : ℝ) (o : Nat) : ℝ :=
  if keepBit keep u k o then u.readMem x o/(1-p) else 0

structure Bounds (x k y : RegionName) (n B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<n → offset u B j<bounds x
  keep : ∀ j : Fin B, offset u B j<n → offset u B j<bounds k
  output : ∀ j : Fin B, offset u B j<n → offset u B j<bounds y

def Outside (A : FlatAlloc) (u : BlockState) (y : RegionName) (n B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<n → o≠A.addr y (offset u B j)

def correct_target (kernel : ComputeKernel) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds x k y n B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<n →
        v.readMem A.flat (A.addr y (offset u B j))=expected keep u x k p (offset u B j)) ∧
      (∀ r o, Outside A u y n B r o → v.mem r o=(A.flattenState u).mem r o)

specification dropout_correctness (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (correct_target (original storage keep x k y n B p) keep x k y n B p) := by
  intro A hd hc u hu hb
  exact RealIO.original_correct storage keep x k y n B p A hd hc u hu ⟨hb.input,hb.keep,hb.output⟩

specification index_commuted_correctness (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (correct_target (indexCommuted storage keep x k y n B p) keep x k y n B p) := by
  intro A hd hc u hu hb
  exact RealIO.candidate_correct storage keep x k y n B p A hd hc u hu ⟨hb.input,hb.keep,hb.output⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DropoutTriton.Correctness

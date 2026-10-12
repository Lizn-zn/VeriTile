import bench.tritonbench_g.embedding_triton_kernel.Proofs.RealIO
/-! Complete chunked lookup with signed int32 token IDs. Matching fp16/bf16/fp32
payload storage is erased only for real correctness. The formula uses initial
inputs, actual vocabulary/row/column masks and the entire strided loop. A last
chunk may extend past BLOCK_N. Output stride need only cover active columns.
The two input regions must be separate from output so subsequent iterations
still read initial data. This per-program contract excludes host/grid races,
fixed-width index overflow and physical rounding. Zero chunk stride follows
the formal range semantics; it is not a Triton compilation claim. -/
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def original (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType

def indexCommuted (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.true weight ids out first last sw so n h D N NN).eraseDType

def row (pid N NN t lane : Nat) : Nat := pid*N+t*NN+lane

def token (u : BlockState) (ids : RegionName) (r : Nat) : Int := u.readMemValue .int ids r

/-- Independent mathematical lookup, before any output stores. Negative IDs
are valid exactly when they lie in the declared signed vocabulary interval. -/
def expected (u : BlockState) (weight ids : RegionName) (first last : Int)
    (weightStride r d : Nat) : ℝ :=
  let id := token u ids r
  if first ≤ id ∧ id < last then u.readMem weight ((id-first).toNat*weightStride+d) else 0

/-- All source store locations, including a possible final overshooting chunk. -/
def Written (pid N NN D n hidden outStride : Nat) (o : Nat) : Prop :=
  ∃ t, t*NN<N ∧ ∃ r, r<NN ∧ ∃ d, d<D ∧
    row pid N NN t r<n ∧ d<hidden ∧ o=row pid N NN t r*outStride+d

/-- Only source-active token, weight and output addresses require storage.
Bounds are on initial inputs, independently of any execution result. -/
structure Bounds (u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) (bounds : RegionBounds) : Prop where
  idAccess : ∀ t, t*NN<N → ∀ jr : Fin NN,
    row u.pid N NN t jr.val<n → row u.pid N NN t jr.val<bounds ids
  weightAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    first≤token u ids (row u.pid N NN t jr.val) →
    token u ids (row u.pid N NN t jr.val)<last →
    (token u ids (row u.pid N NN t jr.val)-first).toNat*sw+jd.val<bounds weight
  outputAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    row u.pid N NN t jr.val*so+jd.val<bounds out

def Outside (A : FlatAlloc) (u : BlockState) (out : RegionName)
    (so n h D N NN : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ z, Written u.pid N NN D n h so z → o≠A.addr out z

/-- Actual finite allocation and successful flat execution are part of the
contract, together with the independent lookup and full tagged-cell frame. -/
def correct_target (kernel : ComputeKernel) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : Prop :=
  weight≠out → ids≠out → min D h≤so →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds u weight ids out first last sw so n h D N NN A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
        row u.pid N NN t jr.val<n → jd.val<h →
        v.readMem A.flat (A.addr out (row u.pid N NN t jr.val*so+jd.val))=
          expected u weight ids first last sw (row u.pid N NN t jr.val) jd.val) ∧
      (∀ r o, Outside A u out so n h D N NN r o → v.mem r o=(A.flattenState u).mem r o)

specification embedding_correctness (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (correct_target (original storage weight ids out first last sw so n h D N NN)
      weight ids out first last sw so n h D N NN) := by
  intro hw hid hs A hd hc u hu hb
  exact RealIO.original_correct storage weight ids out first last sw so n h D N NN hw hid hs A hd hc u hu
    ⟨hb.idAccess,hb.weightAccess,hb.outputAccess⟩

specification index_commuted_correctness (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (correct_target (indexCommuted storage weight ids out first last sw so n h D N NN)
      weight ids out first last sw so n h D N NN) := by
  intro hw hid hs A hd hc u hu hb
  exact RealIO.candidate_correct storage weight ids out first last sw so n h D N NN hw hid hs A hd hc u hu
    ⟨hb.idAccess,hb.weightAccess,hb.outputAccess⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Correctness

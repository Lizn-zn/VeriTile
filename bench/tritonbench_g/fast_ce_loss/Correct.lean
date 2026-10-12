import bench.tritonbench_g.fast_ce_loss.Proofs.ForwardIO
import bench.tritonbench_g.fast_ce_loss.Proofs.BackwardIO
import bench.tritonbench_g.fast_ce_loss.Proofs.ChunkedIO
import bench.tritonbench_g.fast_ce_loss.Proofs.Counterexamples
/-! Complete finite-flat input/output contracts for the actual JIT bodies.
Capped padding participates in forward LSE. Chunked forward writes only in
chunk zero; it does not establish the host aggregation's correctness.
Domains retain finite bounds and typed valid labels. Forward enabled scalars
are positive; backward accepts arbitrary scaling and any nonzero enabled softcap.
Host launches, races, fixed-width overflow and physical rounding are excluded. -/
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

/-! Pure formulas use initial values, independently of execution. `lse`
includes capped padded lanes; `partialLoss` is only the chunk-zero partial
term. `gradient` uses the caller-supplied LSE, without claiming a host
forward/backward composition or a calculus theorem. -/
def transform (cap scale : Bool) (SC SL x : ℝ) : ℝ :=
  let x := if scale then SL*x else x
  if cap then SC*Real.tanh (x/SC) else x

/-- A padded softcap lane contributes exp(-SC); an uncapped masked lane contributes zero. -/
def lse {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V start : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ j, if start+j.val<V then Real.exp (transform cap scale SC SL (xs j))
    else if cap then Real.exp (-SC) else 0)

def loss {B : Nat} (cap scale : Bool) (SC SL : ℝ) (V : Nat)
    (lab : Int) (xs : Fin B → ℝ) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else lse cap scale SC SL V 0 xs-transform cap scale SC SL gather

def gradient (cap scale : Bool) (SC SL : ℝ) (lab : Int) (col : Nat)
    (raw lse upstream : ℝ) : ℝ :=
  let scaled := if scale then raw * SL else raw
  let t := Real.tanh (scaled / SC)
  let x := if cap then SC * t else scaled
  let p := Real.exp (x - lse)
  let y := if (col : Int) = lab then p - 1 else p
  let y := if scale then y * SL else y
  let y := if cap then y * (1 - t*t) else y
  (if lab = -100 then 0 else upstream) * y

def col (u : BlockState) (B : Nat) (j : Fin B) := u.pids 1*B+j.val
def address (u : BlockState) (R B : Nat) (j : Fin B) := u.pids 0*R+col u B j

def partialLoss (cap scale : Bool) (SC SL : ℝ) (lab : Int) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else -transform cap scale SC SL gather

structure ForwardBounds (X Y Z : RegionName) (L : Region .int) (V R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 0 < bounds Y
  lse : u.pids 0 < bounds Z
  gather : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, j.val<V → u.pids 0*R+j.val < bounds X

def forwardOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0))

def forward_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ForwardBounds X Y Z L V R B u A.extent →
  0<V → V≤B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      v.readMem A.flat (A.addr Z (u.pids 0)) = lse cap scale SC SL V 0
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem A.flat (A.addr Y (u.pids 0)) = loss cap scale SC SL V
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, forwardOutside A u Y Z r o → v.mem r o=(A.flattenState u).mem r o)

specification forward_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (forward_correct_target
      (forward storage cap scale Bool.false X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hb' : ForwardIO.Bounds X Y Z L V R B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.loss,hb.lse,hb.gather,hb.logits⟩
  exact ForwardIO.typed_correct storage cap scale X Y Z L V R B SC SL
    hSC hSL A hd hc u hu hb' hv hB hne

specification forward_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    Spec.Real (forward_correct_target
      (forward storage cap scale Bool.true X Y Z L V R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hb' : ForwardIO.Bounds X Y Z L V R B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.loss,hb.lse,hb.gather,hb.logits⟩
  exact ForwardIO.candidate_correct storage cap scale X Y Z L V R B SC SL
    hSC hSL A hd hc u hu hb' hv hB hne


structure ChunkedBounds (X Y Z : RegionName) (L : Region .int) (V C R B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  loss : u.pids 1=0 → u.pids 0 < bounds Y
  lse : u.pids 1=0 → u.pids 0*C < bounds Z
  gather : u.pids 1=0 → u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
    u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X
  logits : ∀ j : Fin B, u.pids 1*B+j.val<V → u.pids 0*R+(u.pids 1*B+j.val) < bounds X

def chunkedOutside (A : FlatAlloc) (u : BlockState) (Y Z : RegionName) (C : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ (o≠A.addr Y (u.pids 0) ∧ o≠A.addr Z (u.pids 0*C))

def chunkedOutputs (A : FlatAlloc) (u v : BlockState) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) : Prop :=
  if u.pids 1=0 then
    v.readMem A.flat (A.addr Z (u.pids 0*C)) = lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
    v.readMem A.flat (A.addr Y (u.pids 0)) = partialLoss cap scale SC SL
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
    (∀ r o, chunkedOutside A u Y Z C r o → v.mem r o=(A.flattenState u).mem r o)
  else v.mem=(A.flattenState u).mem

/-- Conditional source behavior: only chunk zero writes its partial loss and LSE. -/
def chunked_correct_target (kernel : Kernel) (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → 0<SC) → (scale=Bool.true → 0<SL) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → ChunkedBounds X Y Z L V C R B u A.extent →
  0<V → 0<B → Y≠Z →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      chunkedOutputs A u v cap scale X Y Z L V C R B SC SL

specification chunked_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (chunked_correct_target
      (chunked storage cap scale Bool.false X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hb' : ChunkedIO.Bounds X Y Z L V C R B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.loss,hb.lse,hb.gather,hb.logits⟩
  exact ChunkedIO.typed_correct storage cap scale X Y Z L V C R B SC SL
    hSC hSL A hd hc u hu hb' hv hB hne

specification chunked_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    Spec.Real (chunked_correct_target
      (chunked storage cap scale Bool.true X Y Z L V C R B SC SL).eraseDType.toAlgKernel
      cap scale X Y Z L V C R B SC SL) := by
  intro hSC hSL A hd hc u hu hb hv hB hne
  have hb' : ChunkedIO.Bounds X Y Z L V C R B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.loss,hb.lse,hb.gather,hb.logits⟩
  exact ChunkedIO.candidate_correct storage cap scale X Y Z L V C R B SC SL
    hSC hSL A hd hc u hu hb' hv hB hne


structure BackwardBounds (X D Z : RegionName) (L : Region .int) (V R DR B : Nat)
    (u : BlockState) (bounds : RegionBounds) : Prop where
  label : u.pids 0 < bounds (Region.cast L)
  labelTyped : ∃ lab : Int, u.mem (Region.cast L) (u.pids 0)=MemCell.of .int lab
  labelValid : let lab := u.readMemValue .int (Region.cast L) (u.pids 0)
    lab = -100 ∨ (0≤lab ∧ lab < (V : Int))
  lse : u.pids 0 < bounds Z
  upstream : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 → u.pids 0*DR < bounds D
  logits : ∀ j : Fin B, col u B j<V → address u R B j < bounds X

def backwardOutside (A : FlatAlloc) (u : BlockState) (X : RegionName) (V R B : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, col u B j<V → o≠A.addr X (address u R B j)

def backward_correct_target (kernel : Kernel) (cap scale : Bool) (X D Z : RegionName) (L : Region .int)
    (V R DR B : Nat) (SC SL : ℝ) : Prop :=
  (cap=Bool.true → SC≠0) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → BackwardBounds X D Z L V R DR B u A.extent →
    ∃ v, exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, col u B j<V → v.readMem A.flat (A.addr X (address u R B j)) =
        gradient cap scale SC SL (u.readMemValue .int (Region.cast L) (u.pids 0))
          (col u B j) (u.readMem X (address u R B j))
          (u.readMem Z (u.pids 0)) (u.readMem D (u.pids 0*DR))) ∧
      (∀ r o, backwardOutside A u X V R B r o → v.mem r o=(A.flattenState u).mem r o)

specification backward_correctness (storage : Storage) (cap scale : Bool)
    (X D Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (backward_correct_target
      (backward storage cap scale Bool.false X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL) := by
  intro hSC A hd hc u hu hb
  have hb' : BackwardIO.Bounds X D Z L V R DR B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.lse,hb.upstream,hb.logits⟩
  exact BackwardIO.typed_correct storage cap scale X D Z L V R DR B SC SL
    hSC A hd hc u hu hb'

specification backward_index_commuted_correctness (storage : Storage) (cap scale : Bool)
    (X D Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    Spec.Real (backward_correct_target
      (backward storage cap scale Bool.true X D Z L V R DR B SC SL).eraseDType.toAlgKernel
      cap scale X D Z L V R DR B SC SL) := by
  intro hSC A hd hc u hu hb
  have hb' : BackwardIO.Bounds X D Z L V R DR B u A.extent :=
    ⟨hb.label,hb.labelTyped,hb.labelValid,hb.lse,hb.upstream,hb.logits⟩
  exact BackwardIO.candidate_correct storage cap scale X D Z L V R DR B SC SL
    hSC A hd hc u hu hb'

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.Correctness

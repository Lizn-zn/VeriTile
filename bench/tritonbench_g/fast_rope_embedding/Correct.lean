import bench.tritonbench_g.fast_rope_embedding.Proofs.IO
/-! Complete dynamic four-head group, initial-memory rotary formulas, all
active outputs and every other tagged cell. Both halves are loaded before
writing. The sine/cosine tables are loaded before any head executes, so the
contract permits aliases with Q. Positive sequence length and a block covering
the half-head are explicit; odd head sizes leave the final channel unchanged.
This per-program real contract does not certify host scheduling, races,
fixed-width index overflow or physical floating-point accuracy. -/
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def first (row QR H head lane : Nat) := row*QR+head*H+lane
def second (row QR H head lane : Nat) := first row QR H head lane+H/2
def coefficientAddress (row stride seq lane : Nat) := (row%seq)*stride+lane

def signedSine (backwards : Bool) (x : ℝ) := if backwards then -x else x

def firstValue (backwards : Bool) (a b c s : ℝ) := a*c-b*signedSine backwards s
def secondValue (backwards : Bool) (a b c s : ℝ) := b*c+a*signedSine backwards s

def start (group : Nat) := group*4
def stop (group heads : Nat) := min (start group+4) heads

structure Bounds (u : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (bounds : RegionBounds) : Prop where
  cosine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid CR seq j.val<bounds C
  sine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid SR seq j.val<bounds S
  first : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → first u.pid QR H k j.val<bounds Q
  second : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → second u.pid QR H k j.val<bounds Q

def Outside (A : FlatAlloc) (u : BlockState) (Q : RegionName) (QR H heads B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 →
      o≠A.addr Q (first u.pid QR H k j.val) ∧ o≠A.addr Q (second u.pid QR H k j.val)

def correct_target (kernel : Kernel) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds u Q C S QR CR SR seq H heads B A.extent →
  0<seq → H/2≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
        ∀ j : Fin B,j.val<H/2 →
        v.readMem A.flat (A.addr Q (first u.pid QR H k j.val))=
          firstValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val)) ∧
        v.readMem A.flat (A.addr Q (second u.pid QR H k j.val))=
          secondValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val))) ∧
      (∀ r o,Outside A u Q QR H heads B r o → v.mem r o=(A.flattenState u).mem r o)

specification original_correctness (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (correct_target (rope input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards) := by
  intro A hd ha u hu hb hseq hB
  have hb' : Safety.Bounds u Q C S QR CR SR seq H heads B A.extent :=
    ⟨hb.cosine,hb.sine,hb.first,hb.second⟩
  exact IOProof.typed_correct input cosine sine Q C S QR CR SR seq H heads B backwards A hd ha u hu hb' hseq hB

specification index_commuted_correctness (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (correct_target (rope input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards) := by
  intro A hd ha u hu hb hseq hB
  have hb' : Safety.Bounds u Q C S QR CR SR seq H heads B A.extent :=
    ⟨hb.cosine,hb.sine,hb.first,hb.second⟩
  exact IOProof.candidate_correct input cosine sine Q C S QR CR SR seq H heads B backwards A hd ha u hu hb' hseq hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Correctness

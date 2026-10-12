import FifthSphDraft.ForwardIO
import FifthSphDraft.BackwardIO
/-! Complete original/candidate real IO contracts for both source JITs.
The polynomial tables use the source decimal coefficients as exact rationals.
Backward adds the source update to the old coordinate gradient. Its rounded
coefficient table is not the exact derivative of the forward table; the
formal coefficient discrepancy is in Proofs/RealMath. Host launch composition,
fixed-width overflow, races and physical floating-point accuracy are excluded.
-/
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section
def output (c : Fin 11) (x y z : ℝ) : ℝ := match c.val with
  | 0 => (232681380862329 / 100000000000000 : ℝ) * x^5 + (-232681380862329 / 10000000000000 : ℝ) * x^3 * z^2 + (29085172607791 / 2500000000000 : ℝ) * x * z^4
  | 1 => (-294321253055229 / 10000000000000 : ℝ) * x^3 * y * z + (294321253055229 / 10000000000000 : ℝ) * x * y * z^3
  | 2 => (34686092313779 / 20000000000000 : ℝ) * x^5 + (-34686092313779 / 2500000000000 : ℝ) * x^3 * y^2 + (-34686092313779 / 10000000000000 : ℝ) * x^3 * z^2 + (104058276941337 / 2500000000000 : ℝ) * x * y^2 * z^2 + (-104058276941337 / 20000000000000 : ℝ) * x * z^4
  | 3 => (-10620403417479 / 625000000000 : ℝ) * x^3 * y * z + (339852909359329 / 10000000000000 : ℝ) * x * y^3 * z + (-10620403417479 / 625000000000 : ℝ) * x * y * z^3
  | 4 => (80282703616657 / 50000000000000 : ℝ) * x^5 + (-192678488679977 / 10000000000000 : ℝ) * x^3 * y^2 + (80282703616657 / 25000000000000 : ℝ) * x^3 * z^2 + (128452325786651 / 10000000000000 : ℝ) * x * y^4 + (-192678488679977 / 10000000000000 : ℝ) * x * y^2 * z^2 + (80282703616657 / 50000000000000 : ℝ) * x * z^4
  | 5 => (621867148191637 / 100000000000000 : ℝ) * x^4 * y + (-16583123951777 / 1000000000000 : ℝ) * x^2 * y^3 + (124373429638327 / 10000000000000 : ℝ) * x^2 * y * z^2 + (16583123951777 / 5000000000000 : ℝ) * y^5 + (-16583123951777 / 1000000000000 : ℝ) * y^3 * z^2 + (621867148191637 / 100000000000000 : ℝ) * y * z^4
  | 6 => (80282703616657 / 50000000000000 : ℝ) * x^4 * z + (-192678488679977 / 10000000000000 : ℝ) * x^2 * y^2 * z + (80282703616657 / 25000000000000 : ℝ) * x^2 * z^3 + (128452325786651 / 10000000000000 : ℝ) * y^4 * z + (-192678488679977 / 10000000000000 : ℝ) * y^2 * z^3 + (80282703616657 / 50000000000000 : ℝ) * z^5
  | 7 => (849632273398321 / 100000000000000 : ℝ) * x^4 * y + (-10620403417479 / 625000000000 : ℝ) * x^2 * y^3 + (10620403417479 / 625000000000 : ℝ) * y^3 * z^2 + (-849632273398321 / 100000000000000 : ℝ) * y * z^4
  | 8 => (104058276941337 / 20000000000000 : ℝ) * x^4 * z + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 * z + (34686092313779 / 10000000000000 : ℝ) * x^2 * z^3 + (34686092313779 / 2500000000000 : ℝ) * y^2 * z^3 + (-34686092313779 / 20000000000000 : ℝ) * z^5
  | 9 => (91975391579759 / 12500000000000 : ℝ) * x^4 * y + (-441481879582843 / 10000000000000 : ℝ) * x^2 * y * z^2 + (91975391579759 / 12500000000000 : ℝ) * y * z^4
  | 10 => (29085172607791 / 2500000000000 : ℝ) * x^4 * z + (-232681380862329 / 10000000000000 : ℝ) * x^2 * z^3 + (232681380862329 / 100000000000000 : ℝ) * z^5
  | _ => 0

/-- Decimal coefficients of the source backward update. They are not
asserted to be the exact derivatives of the source's rounded forward table. -/
def backwardWeight (component : Fin 3) (c : Fin 11) (x y z : ℝ) : ℝ := match component.val,c.val with
  | 0,0 => (29085172607791 / 2500000000000 : ℝ) * x^4 + (-349022071293493 / 5000000000000 : ℝ) * x^2 * z^2 + (29085172607791 / 2500000000000 : ℝ) * z^4
  | 0,1 => (-441481879582843 / 5000000000000 : ℝ) * x^2 * y * z + (294321253055229 / 10000000000000 : ℝ) * y * z^3
  | 0,2 => (216788076961119 / 25000000000000 : ℝ) * x^4 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 + (-104058276941337 / 10000000000000 : ℝ) * x^2 * z^2 + (104058276941337 / 2500000000000 : ℝ) * y^2 * z^2 + (-104058276941337 / 20000000000000 : ℝ) * z^4
  | 0,3 => (-509779364038993 / 10000000000000 : ℝ) * x^2 * y * z + (339852909359329 / 10000000000000 : ℝ) * y^3 * z + (-10620403417479 / 625000000000 : ℝ) * y * z^3
  | 0,4 => (802827036166571 / 100000000000000 : ℝ) * x^4 + (-578035466039931 / 10000000000000 : ℝ) * x^2 * y^2 + (240848110849971 / 25000000000000 : ℝ) * x^2 * z^2 + (128452325786651 / 10000000000000 : ℝ) * y^4 + (-192678488679977 / 10000000000000 : ℝ) * y^2 * z^2 + (80282703616657 / 50000000000000 : ℝ) * z^4
  | 0,5 => (49749371855331 / 2000000000000 : ℝ) * x^3 * y + (-16583123951777 / 500000000000 : ℝ) * x * y^3 + (49749371855331 / 2000000000000 : ℝ) * x * y * z^2
  | 0,6 => (80282703616657 / 12500000000000 : ℝ) * x^3 * z + (-192678488679977 / 5000000000000 : ℝ) * x * y^2 * z + (80282703616657 / 12500000000000 : ℝ) * x * z^3
  | 0,7 => (339852909359329 / 10000000000000 : ℝ) * x^3 * y + (-339852909359329 / 10000000000000 : ℝ) * x * y^3
  | 0,8 => (104058276941337 / 5000000000000 : ℝ) * x^3 * z + (-104058276941337 / 1250000000000 : ℝ) * x * y^2 * z + (34686092313779 / 5000000000000 : ℝ) * x * z^3
  | 0,9 => (294321253055229 / 10000000000000 : ℝ) * x^3 * y + (-441481879582843 / 5000000000000 : ℝ) * x * y * z^2
  | 0,10 => (465362761724657 / 10000000000000 : ℝ) * x^3 * z + (-465362761724657 / 10000000000000 : ℝ) * x * z^3
  | 1,1 => (-294321253055229 / 10000000000000 : ℝ) * x^3 * z + (294321253055229 / 10000000000000 : ℝ) * x * z^3
  | 1,2 => (-34686092313779 / 1250000000000 : ℝ) * x^3 * y + (104058276941337 / 1250000000000 : ℝ) * x * y * z^2
  | 1,3 => (-10620403417479 / 625000000000 : ℝ) * x^3 * z + (101955872807799 / 1000000000000 : ℝ) * x * y^2 * z + (-10620403417479 / 625000000000 : ℝ) * x * z^3
  | 1,4 => (-192678488679977 / 5000000000000 : ℝ) * x^3 * y + (102761860629321 / 2000000000000 : ℝ) * x * y^3 + (-192678488679977 / 5000000000000 : ℝ) * x * y * z^2
  | 1,5 => (621867148191637 / 100000000000000 : ℝ) * x^4 + (-49749371855331 / 1000000000000 : ℝ) * x^2 * y^2 + (124373429638327 / 10000000000000 : ℝ) * x^2 * z^2 + (16583123951777 / 1000000000000 : ℝ) * y^4 + (-49749371855331 / 1000000000000 : ℝ) * y^2 * z^2 + (621867148191637 / 100000000000000 : ℝ) * z^4
  | 1,6 => (-192678488679977 / 5000000000000 : ℝ) * x^2 * y * z + (102761860629321 / 2000000000000 : ℝ) * y^3 * z + (-192678488679977 / 5000000000000 : ℝ) * y * z^3
  | 1,7 => (849632273398321 / 100000000000000 : ℝ) * x^4 + (-31861210252437 / 625000000000 : ℝ) * x^2 * y^2 + (31861210252437 / 625000000000 : ℝ) * y^2 * z^2 + (-849632273398321 / 100000000000000 : ℝ) * z^4
  | 1,8 => (-104058276941337 / 1250000000000 : ℝ) * x^2 * y * z + (34686092313779 / 1250000000000 : ℝ) * y * z^3
  | 1,9 => (91975391579759 / 12500000000000 : ℝ) * x^4 + (-441481879582843 / 10000000000000 : ℝ) * x^2 * z^2 + (91975391579759 / 12500000000000 : ℝ) * z^4
  | 2,0 => (-465362761724657 / 10000000000000 : ℝ) * x^3 * z + (465362761724657 / 10000000000000 : ℝ) * x * z^3
  | 2,1 => (-294321253055229 / 10000000000000 : ℝ) * x^3 * y + (441481879582843 / 5000000000000 : ℝ) * x * y * z^2
  | 2,2 => (-34686092313779 / 5000000000000 : ℝ) * x^3 * z + (104058276941337 / 1250000000000 : ℝ) * x * y^2 * z + (-104058276941337 / 5000000000000 : ℝ) * x * z^3
  | 2,3 => (-10620403417479 / 625000000000 : ℝ) * x^3 * y + (339852909359329 / 10000000000000 : ℝ) * x * y^3 + (-509779364038993 / 10000000000000 : ℝ) * x * y * z^2
  | 2,4 => (80282703616657 / 12500000000000 : ℝ) * x^3 * z + (-192678488679977 / 5000000000000 : ℝ) * x * y^2 * z + (642261628933257 / 100000000000000 : ℝ) * x * z^3
  | 2,5 => (49749371855331 / 2000000000000 : ℝ) * x^2 * y * z + (-16583123951777 / 500000000000 : ℝ) * y^3 * z + (49749371855331 / 2000000000000 : ℝ) * y * z^3
  | 2,6 => (80282703616657 / 50000000000000 : ℝ) * x^4 + (-192678488679977 / 10000000000000 : ℝ) * x^2 * y^2 + (240848110849971 / 25000000000000 : ℝ) * x^2 * z^2 + (128452325786651 / 10000000000000 : ℝ) * y^4 + (-578035466039931 / 10000000000000 : ℝ) * y^2 * z^2 + (802827036166571 / 100000000000000 : ℝ) * z^4
  | 2,7 => (339852909359329 / 10000000000000 : ℝ) * y^3 * z + (-339852909359329 / 10000000000000 : ℝ) * y * z^3
  | 2,8 => (104058276941337 / 20000000000000 : ℝ) * x^4 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 + (104058276941337 / 10000000000000 : ℝ) * x^2 * z^2 + (104058276941337 / 2500000000000 : ℝ) * y^2 * z^2 + (-34686092313779 / 4000000000000 : ℝ) * z^4
  | 2,9 => (-441481879582843 / 5000000000000 : ℝ) * x^2 * y * z + (294321253055229 / 10000000000000 : ℝ) * y * z^3
  | 2,10 => (29085172607791 / 2500000000000 : ℝ) * x^4 + (-349022071293493 / 5000000000000 : ℝ) * x^2 * z^2 + (29085172607791 / 2500000000000 : ℝ) * z^4
  | _,_ => 0

def gradient (component : Fin 3) (x y z old : ℝ) (g : Fin 11 → ℝ) : ℝ :=
  old+∑ c,g c*backwardWeight component c x y z


def coord (pid B j c : Nat) := j*3+B*3*pid+c
def outputAddress (pid B stride off j c : Nat) := j*stride+B*stride*pid+off+c

structure ForwardBounds (X Y : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  coordinates : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds X
  output : ∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M → outputAddress u.pid B stride off j.val c.val<bounds Y

def ForwardCover (B N M off stride : Nat) : Prop := ∀ pid,∀ j : Fin B,
  outputAddress pid B stride off j.val 0<M → coord pid B j.val 2<N

def forwardOutside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (B M off stride : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M →
    o≠A.addr Y (outputAddress u.pid B stride off j.val c.val)

def forward_correct_target (kernel : Kernel) (X Y : RegionName) (B N M off stride : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → ForwardBounds X Y B N M off stride u A.extent →
  10<stride → ForwardCover B N M off stride →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M →
        v.readMem A.flat (A.addr Y (outputAddress u.pid B stride off j.val c.val))=
          output c (u.readMem X (coord u.pid B j.val 0))
            (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))) ∧
      (∀ r o,forwardOutside A u Y B M off stride r o → v.mem r o=(A.flattenState u).mem r o)

specification forward_correctness (input outputStorage : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (forward_correct_target (forward input outputStorage Bool.false X Y B N M off stride).eraseDType.toAlgKernel X Y B N M off stride) := by
  intro A hd ha u hu hb hstride hcover
  have hb' : ForwardIO.Bounds X Y B N M off stride u A.extent := ⟨hb.coordinates,hb.output⟩
  exact ForwardIO.typed_correct input outputStorage X Y B N M off stride A hd ha u hu hb' hstride hcover

specification forward_index_commuted_correctness (input outputStorage : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (forward_correct_target (forward input outputStorage Bool.true X Y B N M off stride).eraseDType.toAlgKernel X Y B N M off stride) := by
  intro A hd ha u hu hb hstride hcover
  have hb' : ForwardIO.Bounds X Y B N M off stride u A.extent := ⟨hb.coordinates,hb.output⟩
  exact ForwardIO.candidate_correct input outputStorage X Y B N M off stride A hd ha u hu hb' hstride hcover

structure BackwardBounds (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  coordinates : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds X
  gradient : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds dX
  upstream : ∀ c : Fin 11,∀ j : Fin B,outputAddress u.pid B stride off j.val c.val<M → outputAddress u.pid B stride off j.val c.val<bounds g


/-- Every active coordinate belongs to a complete xyz row with all eleven
upstream channels available. This is the domain of the unmasked polynomial
formula; masked loads remain present in the implementation. -/
def BackwardCover (pid B N M off stride : Nat) : Prop := ∀ j : Fin B,
  coord pid B j.val 0<N → coord pid B j.val 2<N ∧ outputAddress pid B stride off j.val 10<M

def backwardOutside (A : FlatAlloc) (u : BlockState) (dX : RegionName) (B N : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N →
    o≠A.addr dX (coord u.pid B j.val c.val)

def backward_correct_target (kernel : Kernel) (X dX g : RegionName) (B N M off stride : Nat) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → BackwardBounds X dX g B N M off stride u A.extent →
  BackwardCover u.pid B N M off stride →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N →
        v.readMem A.flat (A.addr dX (coord u.pid B j.val c.val))=
          gradient c (u.readMem X (coord u.pid B j.val 0))
            (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
            (u.readMem dX (coord u.pid B j.val c.val))
            (fun d => u.readMem g (outputAddress u.pid B stride off j.val d.val))) ∧
      (∀ r o,backwardOutside A u dX B N r o → v.mem r o=(A.flattenState u).mem r o)

specification backward_correctness (input gradientStorage upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (backward_correct_target (backward input gradientStorage upstream Bool.false X dX g B N M off stride).eraseDType.toAlgKernel X dX g B N M off stride) := by
  intro A hd ha u hu hb hcover
  have hb' : BackwardBase.Bounds X dX g B N M off stride u A.extent := ⟨hb.coordinates,hb.gradient,hb.upstream⟩
  exact BackwardIO.typed_correct input gradientStorage upstream X dX g B N M off stride A hd ha u hu hb' hcover

specification backward_index_commuted_correctness (input gradientStorage upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (backward_correct_target (backward input gradientStorage upstream Bool.true X dX g B N M off stride).eraseDType.toAlgKernel X dX g B N M off stride) := by
  intro A hd ha u hu hb hcover
  have hb' : BackwardBase.Bounds X dX g B N M off stride u A.extent := ⟨hb.coordinates,hb.gradient,hb.upstream⟩
  exact BackwardIO.candidate_correct input gradientStorage upstream X dX g B N M off stride A hd ha u hu hb' hcover

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Correctness

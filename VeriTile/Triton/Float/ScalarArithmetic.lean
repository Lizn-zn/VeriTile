/- Scalar fp32 arithmetic interpreted through the current admitted fragments.
Every numerical equality below is obtained from a selected scalar rule, with
its operand guards retained. No ring instance or IEEE equality is assumed. -/
import VeriTile.Triton.Float.Reciprocal

namespace VeriTile.Triton.FP.ScalarArithmetic
open Structural Guarded

inductive Atom where
  | addCommute | addAssociate | mulCommute | mulAssociate | mulDistribute | cancel
  | addZero | mulOne | divOne | divMulRcp | mulRcpCancel
  deriving DecidableEq

def atoms : List Atom := [.addCommute, .addAssociate, .mulCommute, .mulAssociate,
  .mulDistribute, .cancel, .addZero, .mulOne, .divOne, .divMulRcp, .mulRcpCancel]

def guards : Atom → List OperandGuard
  | .addCommute | .mulCommute | .cancel => [⟨"a", .finite⟩, ⟨"b", .finite⟩]
  | .addAssociate | .mulAssociate | .mulDistribute =>
      [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"c", .finite⟩]
  | .addZero | .mulOne | .divOne => [⟨"a", .finite⟩]
  | .divMulRcp => [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"b", .nonzero⟩]
  | .mulRcpCancel => [⟨"a", .finite⟩, ⟨"a", .nonzero⟩]

def report : Atom → ReportedScalarRule
  | .addCommute => ⟨ReportedAdmission.fp32_add_commute, guards .addCommute⟩
  | .addAssociate => ⟨ReportedAdmission.fp32_add_assoc, guards .addAssociate⟩
  | .mulCommute => ⟨ReportedAdmission.fp32_mul_commute, guards .mulCommute⟩
  | .mulAssociate => ⟨ReportedAdmission.fp32_mul_assoc, guards .mulAssociate⟩
  | .mulDistribute => ⟨ReportedAdmission.fp32_mul_distrib, guards .mulDistribute⟩
  | .cancel => ⟨ReportedAdmission.fp32_cancel, guards .cancel⟩
  | .addZero => SupplementalAdmission.fp32_add_zero
  | .mulOne => SupplementalAdmission.fp32_mul_one
  | .divOne => SupplementalAdmission.fp32_div_one
  | .divMulRcp => SupplementalAdmission.fp32_div_mul_rcp
  | .mulRcpCancel => SupplementalAdmission.fp32_mul_rcp_cancel

def Atom.ruleID : Atom → String
  | .addCommute => "ADD-COMMUTE"
  | .addAssociate => "ADD-ASSOC"
  | .mulCommute => "MUL-COMMUTE"
  | .mulAssociate => "MUL-ASSOC"
  | .mulDistribute => "MUL-DISTRIB"
  | .cancel => "CANCEL"
  | .addZero => "ADD-ZERO"
  | .mulOne => "MUL-ONE"
  | .divOne => "DIV-ONE"
  | .divMulRcp => "DIV-MUL-RCP"
  | .mulRcpCancel => "MUL-RCP-CANCEL"

/-- Report refreshes must preserve the selected relation, precision and domain.
In particular an accepted bf16 output cast cannot supply a bare fp32 law. -/
theorem report_matches (a : Atom) :
    (report a).report.ruleID = a.ruleID ∧
    (report a).report.input = "fp32" ∧ (report a).report.compute = "fp32" ∧
    (report a).report.accumulator = "fp32" ∧ (report a).report.output = "fp32" ∧
    (report a).guards = guards a := by
  cases a <;> decide

def ref (name : String) : Op .real [] := .ref .real [] name
def plus (a b : Op .real []) : Op .real [] := .add .real .nil a b
def minus (a b : Op .real []) : Op .real [] := .sub .real .nil a b
def times (a b : Op .real []) : Op .real [] := .mul .real .nil a b
def divide (a b : Op .real []) : Op .real [] := .div .real .nil a b
def fragment (e : Op .real []) : List ComputeStmt :=
  [.assign .real [] "out" (.compute (.alg .fp32 e))]

def lhsCode : Atom → List ComputeStmt
  | .addCommute => fragment (plus (ref "a") (ref "b"))
  | .addAssociate => fragment (plus (plus (ref "a") (ref "b")) (ref "c"))
  | .mulCommute => fragment (times (ref "a") (ref "b"))
  | .mulAssociate => fragment (times (times (ref "a") (ref "b")) (ref "c"))
  | .mulDistribute => fragment (times (ref "a") (plus (ref "b") (ref "c")))
  | .cancel => fragment (plus (minus (ref "a") (ref "b")) (ref "b"))
  | .addZero => fragment (plus (ref "a") (.const 0))
  | .mulOne => fragment (times (ref "a") (.const 1))
  | .divOne => fragment (divide (ref "a") (.const 1))
  | .divMulRcp => (Reciprocal.lhs .fp32).code
  | .mulRcpCancel => fragment (times (ref "a") (divide (.const 1) (ref "a")))

def rhsCode : Atom → List ComputeStmt
  | .addCommute => fragment (plus (ref "b") (ref "a"))
  | .addAssociate => fragment (plus (ref "a") (plus (ref "b") (ref "c")))
  | .mulCommute => fragment (times (ref "b") (ref "a"))
  | .mulAssociate => fragment (times (ref "a") (times (ref "b") (ref "c")))
  | .mulDistribute => fragment (plus (times (ref "a") (ref "b")) (times (ref "a") (ref "c")))
  | .cancel | .addZero | .mulOne | .divOne => fragment (ref "a")
  | .divMulRcp => (Reciprocal.rhs .fp32).code
  | .mulRcpCancel => fragment (.const 1)

def lhs (a : Atom) : GuardedFragment := ⟨guards a, lhsCode a⟩
def rhs (a : Atom) : GuardedFragment := ⟨guards a, rhsCode a⟩
def entry (a : Atom) := (report a).bind (lhsCode a) (rhsCode a)

structure Rules where
  validated : ∀ a, Spec.EvidenceValidated (entry a).rule (entry a).evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment := atoms.map entry

theorem admitted (R : Rules) (a : Atom) :
    Spec.Derivation R.assumptions [lhs a] [rhs a] := by
  have ha : a ∈ atoms := by cases a <;> simp [atoms]
  have h := Spec.Derivation.atom (entry a) (List.mem_map.mpr ⟨a, ha, rfl⟩)
    ((report a).admit _ _ (R.validated a))
  cases a <;> exact h

def add {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .add
def sub {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .sub
def mul {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .mul
def div {α : Type} (M : Algebra α) := M.binary (some .fp32) .real .div
def zero {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 0
def one {α : Type} (M : Algebra α) := M.literal (some .fp32) .real 1

def leftValue {α : Type} (M : Algebra α) (a b c : α) : Atom → α
  | .addCommute => add M a b
  | .addAssociate => add M (add M a b) c
  | .mulCommute => mul M a b
  | .mulAssociate => mul M (mul M a b) c
  | .mulDistribute => mul M a (add M b c)
  | .cancel => add M (sub M a b) b
  | .addZero => add M a (zero M)
  | .mulOne => mul M a (one M)
  | .divOne => div M a (one M)
  | .divMulRcp => div M a b
  | .mulRcpCancel => mul M a (div M (one M) a)

def rightValue {α : Type} (M : Algebra α) (a b c : α) : Atom → α
  | .addCommute => add M b a
  | .addAssociate => add M a (add M b c)
  | .mulCommute => mul M b a
  | .mulAssociate => mul M a (mul M b c)
  | .mulDistribute => add M (mul M a b) (mul M a c)
  | .cancel | .addZero | .mulOne | .divOne => a
  | .divMulRcp => mul M a (div M (one M) b)
  | .mulRcpCancel => one M

def Inputs {α : Type} (D : Domain α) (a b c : α) : Atom → Prop
  | .addCommute | .mulCommute | .cancel => D .finite a ∧ D .finite b
  | .addAssociate | .mulAssociate | .mulDistribute => D .finite a ∧ D .finite b ∧ D .finite c
  | .addZero | .mulOne | .divOne => D .finite a
  | .divMulRcp => D .finite a ∧ D .finite b ∧ D .nonzero b
  | .mulRcpCancel => D .finite a ∧ D .nonzero a

set_option maxHeartbeats 1600000 in
/-- Execute the actual scalar fragments before interpreting their admitted law.
The arbitrary surrounding state contributes no numerical premise. -/
theorem apply_atom {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (atom : Atom) (a b c : α) (hg : Inputs D a b c atom) :
    leftValue M a b c atom = rightValue M a b c atom := by
  let t := ((s.setReg "a" .real [] (fun _ => a)).setReg "b" .real []
    (fun _ => b)).setReg "c" .real [] (fun _ => c)
  have hd : ScalarDomain D (guards atom) t := by
    intro g hg'
    cases atom <;> simp [guards] at hg'
    all_goals
      simp only [Inputs] at hg
      rcases hg' with rfl | rfl | rfl
      all_goals simp only [t, State.setReg]; simp_all
  have h := hM (lhs atom) (rhs atom) (admitted R atom) t hd hd
  cases atom <;>
    simp only [lhs, rhs, lhsCode, rhsCode, fragment, Reciprocal.lhs, Reciprocal.rhs,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, ref, plus, minus, times, divide,
      Reciprocal.div, Reciprocal.inv, Reciprocal.mul, Reciprocal.ref,
      ComputeDType.eraseDType, numeric, State.setReg_same, t] at h
  all_goals
    simp [State.setReg, leftValue, rightValue, add, sub, mul, div, zero, one] at h ⊢
    exact congrFun h PUnit.unit

section Laws
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

theorem add_comm (a b : α) (ha : D .finite a) (hb : D .finite b) :
    add M a b = add M b a :=
  apply_atom R M D hM s .addCommute a b a ⟨ha, hb⟩

theorem add_assoc (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    add M (add M a b) c = add M a (add M b c) :=
  apply_atom R M D hM s .addAssociate a b c ⟨ha, hb, hc⟩

/-- Exchange the middle terms of two paired sums using only association and
commutation. Finite cross sums are explicit because leaves alone do not imply
that reassociated intermediate operations stay in the admitted domain. -/
theorem add_add_swap (a b c d : α)
    (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) (hd : D .finite d)
    (hcd : D .finite (add M c d)) (hbd : D .finite (add M b d)) :
    add M (add M a b) (add M c d) = add M (add M a c) (add M b d) := by
  rw [add_assoc R M D hM s a b _ ha hb hcd,
    ← add_assoc R M D hM s b c d hb hc hd,
    add_comm R M D hM s b c hb hc,
    add_assoc R M D hM s c b d hc hb hd,
    ← add_assoc R M D hM s a c _ ha hc hbd]

theorem mul_comm (a b : α) (ha : D .finite a) (hb : D .finite b) :
    mul M a b = mul M b a :=
  apply_atom R M D hM s .mulCommute a b a ⟨ha, hb⟩

theorem mul_assoc (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    mul M (mul M a b) c = mul M a (mul M b c) :=
  apply_atom R M D hM s .mulAssociate a b c ⟨ha, hb, hc⟩

theorem mul_distrib (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    mul M a (add M b c) = add M (mul M a b) (mul M a c) :=
  apply_atom R M D hM s .mulDistribute a b c ⟨ha, hb, hc⟩

/-- Distribution with the sum on the left is derived from the admitted
orientation and commutation. The intermediate sum must also be finite. -/
theorem add_mul (a b c : α) (ha : D .finite a) (hb : D .finite b)
    (hc : D .finite c) (hsum : D .finite (add M a b)) :
    mul M (add M a b) c = add M (mul M a c) (mul M b c) := by
  rw [mul_comm R M D hM s _ c hsum hc,
    mul_distrib R M D hM s c a b hc ha hb,
    mul_comm R M D hM s c a hc ha, mul_comm R M D hM s c b hc hb]

theorem sub_add_cancel (a b : α) (ha : D .finite a) (hb : D .finite b) :
    add M (sub M a b) b = a :=
  apply_atom R M D hM s .cancel a b a ⟨ha, hb⟩

theorem add_zero (a : α) (ha : D .finite a) : add M a (zero M) = a :=
  apply_atom R M D hM s .addZero a a a ha

theorem mul_one (a : α) (ha : D .finite a) : mul M a (one M) = a :=
  apply_atom R M D hM s .mulOne a a a ha

theorem div_one (a : α) (ha : D .finite a) : div M a (one M) = a :=
  apply_atom R M D hM s .divOne a a a ha

theorem div_mul_rcp (a b : α) (ha : D .finite a) (hb : D .finite b) (hn : D .nonzero b) :
    div M a b = mul M a (div M (one M) b) :=
  apply_atom R M D hM s .divMulRcp a b a ⟨ha, hb, hn⟩

theorem mul_rcp_cancel (a : α) (ha : D .finite a) (hn : D .nonzero a) :
    mul M a (div M (one M) a) = one M :=
  apply_atom R M D hM s .mulRcpCancel a a a ⟨ha, hn⟩

/-- Subtracting zero is derived, not separately admitted. -/
theorem sub_zero (a : α) (ha : D .finite a) (hz : D .finite (zero M))
    (hs : D .finite (sub M a (zero M))) : sub M a (zero M) = a :=
  (add_zero R M D hM s _ hs).symm.trans (sub_add_cancel R M D hM s a _ ha hz)

theorem zero_add (a : α) (ha : D .finite a) (hz : D .finite (zero M)) :
    add M (zero M) a = a :=
  (add_comm R M D hM s _ _ hz ha).trans (add_zero R M D hM s a ha)

/-- Recover the summand by adding the derived inverse of the common term.
Cancellation is proved here; it is not supplied by a group structure. -/
theorem add_right_cancel (a b c : α) (ha : D .finite a) (hb : D .finite b)
    (hc : D .finite c) (hz : D .finite (zero M))
    (hn : D .finite (sub M (zero M) c))
    (h : add M a c = add M b c) : a = b := by
  have inverse : add M c (sub M (zero M) c) = zero M :=
    (add_comm R M D hM s c _ hc hn).trans (sub_add_cancel R M D hM s _ c hz hc)
  have recover (v : α) (hv : D .finite v) :
      add M (add M v c) (sub M (zero M) c) = v := by
    rw [add_assoc R M D hM s v c _ hv hc hn, inverse, add_zero R M D hM s v hv]
  exact (recover a ha).symm.trans
    ((congrArg (fun v => add M v (sub M (zero M) c)) h).trans (recover b hb))

/-- Self-subtraction follows from CANCEL and derived addition cancellation. -/
theorem sub_self (a : α) (ha : D .finite a) (hz : D .finite (zero M))
    (hs : D .finite (sub M a a)) (hn : D .finite (sub M (zero M) a)) :
    sub M a a = zero M := by
  apply add_right_cancel R M D hM s _ _ a hs hz ha hz hn
  exact (sub_add_cancel R M D hM s a a ha ha).trans
    (zero_add R M D hM s a ha hz).symm

/-- Joining two differences follows from CANCEL, association and the derived
addition cancellation rule, retaining every guard needed at these sites. -/
theorem sub_add_sub_cancel (a b c : α)
    (ha : D .finite a) (hb : D .finite b) (hc : D .finite c)
    (hab : D .finite (sub M a b)) (hbc : D .finite (sub M b c))
    (hsum : D .finite (add M (sub M a b) (sub M b c)))
    (hac : D .finite (sub M a c)) (hz : D .finite (zero M))
    (hn : D .finite (sub M (zero M) c)) :
    add M (sub M a b) (sub M b c) = sub M a c := by
  apply add_right_cancel R M D hM s _ _ c hsum hac hc hz hn
  rw [add_assoc R M D hM s _ _ c hab hbc hc,
    sub_add_cancel R M D hM s b c hb hc,
    sub_add_cancel R M D hM s a b ha hb, sub_add_cancel R M D hM s a c ha hc]

/-- Explicit padding zeros may be scaled away only after this derivation.
The finite intermediate requirements are domain facts, not numerical equations. -/
theorem mul_zero (a : α) (ha : D .finite a) (hz : D .finite (zero M))
    (hp : D .finite (mul M a (zero M)))
    (hn : D .finite (sub M (zero M) (mul M a (zero M)))) :
    mul M a (zero M) = zero M := by
  let p := mul M a (zero M)
  let n := sub M (zero M) p
  have hpp : add M p p = p := by
    change add M (mul M a (zero M)) (mul M a (zero M)) = mul M a (zero M)
    rw [← mul_distrib R M D hM s a _ _ ha hz hz, add_zero R M D hM s _ hz]
  have hnp : add M n p = zero M := sub_add_cancel R M D hM s _ _ hz hp
  have hzp : add M (zero M) p = zero M := by
    calc
      add M (zero M) p = add M (add M n p) p := congrArg (fun v => add M v p) hnp.symm
      _ = add M n (add M p p) := add_assoc R M D hM s n p p hn hp hp
      _ = add M n p := congrArg (add M n) hpp
      _ = zero M := hnp
  exact (zero_add R M D hM s p hp hz).symm.trans hzp

/-- Opposite summands have equal squares, derived using distribution and
cancellation. The opposite-sum equation is consumed as a local lemma premise;
no negation operation or ring structure is introduced. -/
theorem square_eq_of_add_eq_zero (a b : α)
    (ha : D .finite a) (hb : D .finite b) (hz : D .finite (zero M))
    (haa : D .finite (mul M a a)) (hbb : D .finite (mul M b b))
    (hab : D .finite (mul M a b)) (hnab : D .finite (sub M (zero M) (mul M a b)))
    (ha0 : D .finite (mul M a (zero M)))
    (hna0 : D .finite (sub M (zero M) (mul M a (zero M))))
    (hb0 : D .finite (mul M b (zero M)))
    (hnb0 : D .finite (sub M (zero M) (mul M b (zero M))))
    (h : add M a b = zero M) : mul M a a = mul M b b := by
  have left : add M (mul M a a) (mul M a b) = zero M := by
    rw [← mul_distrib R M D hM s a a b ha ha hb, h,
      mul_zero R M D hM s a ha hz ha0 hna0]
  have right : add M (mul M b b) (mul M a b) = zero M := by
    rw [mul_comm R M D hM s a b ha hb,
      ← mul_distrib R M D hM s b b a hb hb ha, add_comm R M D hM s b a hb ha, h,
      mul_zero R M D hM s b hb hz hb0 hnb0]
  exact add_right_cancel R M D hM s _ _ _ haa hbb hab hz hnab (left.trans right.symm)

/-- A derived cancellation principle for a common nonzero right factor. -/
theorem mul_right_cancel (a b c : α) (ha : D .finite a) (hb : D .finite b)
    (hc : D .finite c) (hn : D .nonzero c) (hi : D .finite (div M (one M) c))
    (h : mul M a c = mul M b c) : a = b := by
  have recover (v : α) (hv : D .finite v) : mul M (mul M v c) (div M (one M) c) = v := by
    rw [mul_assoc R M D hM s v c _ hv hc hi, mul_rcp_cancel R M D hM s c hc hn,
      mul_one R M D hM s v hv]
  exact (recover a ha).symm.trans ((congrArg (fun v => mul M v (div M (one M) c)) h).trans
    (recover b hb))

theorem div_mul_cancel (a b : α) (ha : D .finite a) (hb : D .finite b)
    (hn : D .nonzero b) (hi : D .finite (div M (one M) b)) :
    mul M (div M a b) b = a := by
  rw [div_mul_rcp R M D hM s a b ha hb hn,
    mul_assoc R M D hM s a _ b ha hi hb,
    mul_comm R M D hM s (div M (one M) b) b hi hb,
    mul_rcp_cancel R M D hM s b hb hn, mul_one R M D hM s a ha]

/-- Normalization by a shared scale, derived entirely from scalar atoms.
Every extra premise is a finite/nonzero guard at a concrete arithmetic site. -/
theorem normalize_scale (a b c : α)
    (ha : D .finite a) (hb : D .finite b) (hc : D .finite c)
    (hbn : D .nonzero b) (hcn : D .nonzero c)
    (hac : D .finite (mul M a c)) (hbc : D .finite (mul M b c))
    (hbcn : D .nonzero (mul M b c))
    (hib : D .finite (div M (one M) b)) (hic : D .finite (div M (one M) c))
    (hibc : D .finite (div M (one M) (mul M b c)))
    (hsmall : D .finite (div M a b))
    (hlarge : D .finite (div M (mul M a c) (mul M b c)))
    (hpartial : D .finite (mul M (div M (mul M a c) (mul M b c)) b)) :
    div M (mul M a c) (mul M b c) = div M a b := by
  let q := div M (mul M a c) (mul M b c)
  have hq : mul M (mul M q b) c = mul M a c := by
    rw [mul_assoc R M D hM s q b c hlarge hb hc]
    exact div_mul_cancel R M D hM s _ _ hac hbc hbcn hibc
  have hqb : mul M q b = a :=
    mul_right_cancel R M D hM s _ a c hpartial ha hc hcn hic hq
  apply mul_right_cancel R M D hM s q (div M a b) b hlarge hsmall hb hbn hib
  exact hqb.trans (div_mul_cancel R M D hM s a b ha hb hbn hib).symm

end Laws

end VeriTile.Triton.FP.ScalarArithmetic

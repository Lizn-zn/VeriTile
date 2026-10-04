import VeriTile.Triton.Float.ScalarArithmetic

/-! Arithmetic laws derived from the currently selected scalar candidates.
Availability is checked at each use below. The candidate catalog itself can
still compile if a report removes a relation required by these derivations. -/
namespace VeriTile.Triton.FP.ScalarArithmetic
open Structural Guarded

section Laws
variable {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α)
  (hM : Models R.assumptions M D) (s : State α)
include R hM s

theorem add_comm (a b : α) (ha : D .finite a) (hb : D .finite b) :
    add M a b = add M b a :=
  apply_atom R M D hM s .addCommute (by decide) a b a ⟨ha, hb⟩

theorem add_assoc (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    add M (add M a b) c = add M a (add M b c) :=
  apply_atom R M D hM s .addAssociate (by decide) a b c ⟨ha, hb, hc⟩

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
  apply_atom R M D hM s .mulCommute (by decide) a b a ⟨ha, hb⟩

theorem mul_assoc (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    mul M (mul M a b) c = mul M a (mul M b c) :=
  apply_atom R M D hM s .mulAssociate (by decide) a b c ⟨ha, hb, hc⟩

theorem mul_distrib (a b c : α) (ha : D .finite a) (hb : D .finite b) (hc : D .finite c) :
    mul M a (add M b c) = add M (mul M a b) (mul M a c) :=
  apply_atom R M D hM s .mulDistribute (by decide) a b c ⟨ha, hb, hc⟩

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
  apply_atom R M D hM s .cancel (by decide) a b a ⟨ha, hb⟩

theorem add_zero (a : α) (ha : D .finite a) : add M a (zero M) = a :=
  apply_atom R M D hM s .addZero (by decide) a a a ha

theorem mul_one (a : α) (ha : D .finite a) : mul M a (one M) = a :=
  apply_atom R M D hM s .mulOne (by decide) a a a ha

theorem div_one (a : α) (ha : D .finite a) : div M a (one M) = a :=
  apply_atom R M D hM s .divOne (by decide) a a a ha

theorem div_mul_rcp (a b : α) (ha : D .finite a) (hb : D .finite b) (hn : D .nonzero b) :
    div M a b = mul M a (div M (one M) b) :=
  apply_atom R M D hM s .divMulRcp (by decide) a b a ⟨ha, hb, hn⟩

theorem mul_rcp_cancel (a : α) (ha : D .finite a) (hn : D .nonzero a) :
    mul M a (div M (one M) a) = one M :=
  apply_atom R M D hM s .mulRcpCancel (by decide) a a a ⟨ha, hn⟩

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

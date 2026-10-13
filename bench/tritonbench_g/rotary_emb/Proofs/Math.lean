import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

abbrev Lane (BS BH D : Nat) := TileIndex [BS,BH,D/2]

def sequence (s : BlockState) (BS : Nat) {BH D : Nat} (i : Lane BS BH D) : Nat :=
  s.pids 1 * BS + i.1.val

def head (s : BlockState) (BH : Nat) {BS D : Nat} (i : Lane BS BH D) : Nat :=
  s.pids 0 * BH + i.2.1.val

def dimension {BS BH D : Nat} (odd : Bool) (i : Lane BS BH D) : Nat :=
  i.2.2.1.val*2 + if odd then 1 else 0

def address (s : BlockState) (sb sh sd BS BH D : Nat) (odd : Bool)
    (i : Lane BS BH D) : Nat :=
  sequence s BS i * sb + head s BH i * sh + dimension odd i * sd

/-- The source uses cosine strides for BOTH cosine and sine addresses. -/
def trigAddress (s : BlockState) (scb scd BS BH D : Nat) (odd : Bool)
    (i : Lane BS BH D) : Nat := sequence s BS i * scb + dimension odd i * scd

def active (s : BlockState) (L H BS BH D : Nat) (i : Lane BS BH D) : Prop :=
  sequence s BS i < L ∧ head s BH i < H

/-- Independent pair rotation from initial memory, with the actual even/odd
trigonometric reads. It does not mention the program or final memory. -/
noncomputable def expected (s : BlockState) (X Cos Sin : RegionName)
    (sb sh sd scb scd BS BH D : Nat) (odd : Bool) (i : Lane BS BH D) : ℝ :=
  let x0 := s.readMem X (address s sb sh sd BS BH D Bool.false i)
  let x1 := s.readMem X (address s sb sh sd BS BH D Bool.true i)
  let c := s.readMem Cos (trigAddress s scb scd BS BH D odd i)
  let sn := s.readMem Sin (trigAddress s scb scd BS BH D odd i)
  if odd then x0*sn+x1*c else x0*c-x1*sn

theorem expected_congr (s t : BlockState) (X Cos Sin : RegionName)
    (sb sh sd scb scd BS BH D : Nat) (odd : Bool) (i : Lane BS BH D)
    (hp : s.pids=t.pids) (hX : ∀ o, s.readMem X o=t.readMem X o)
    (hCos : ∀ o, s.readMem Cos o=t.readMem Cos o)
    (hSin : ∀ o, s.readMem Sin o=t.readMem Sin o) :
    expected s X Cos Sin sb sh sd scb scd BS BH D odd i =
      expected t X Cos Sin sb sh sd scb scd BS BH D odd i := by
  simp only [expected, address, trigAddress, sequence, head, hp, hX, hCos, hSin]

theorem masked_pair_formula (seq head : Prop) [Decidable seq] [Decidable head]
    (x0 x1 c sn : ℝ) (odd : Bool) :
    (if odd then
      (if seq ∧ head then x0 else 0)*(if seq then sn else 0) +
        (if seq ∧ head then x1 else 0)*(if seq then c else 0)
    else
      (if seq ∧ head then x0 else 0)*(if seq then c else 0) -
        (if seq ∧ head then x1 else 0)*(if seq then sn else 0)) =
    if seq ∧ head then (if odd then x0*sn+x1*c else x0*c-x1*sn) else 0 := by
  by_cases hs : seq <;> by_cases hh : head <;> cases odd <;> simp [hs,hh]

def PairInjective (s : BlockState) (sb sh sd BS BH D : Nat) : Prop :=
  Function.Injective (fun p : Lane BS BH D × Bool ↦ address s sb sh sd BS BH D p.2 p.1)

def RegionsSeparate (Q K Cos Sin : RegionName) : Prop :=
  Q≠K ∧ Q≠Cos ∧ Q≠Sin ∧ K≠Cos ∧ K≠Sin

def footprint (s : BlockState) (Q K : RegionName)
    (sqb sqh sqd skb skh skd L HQ HK BS BH D : Nat) : WriteFootprint := fun a ↦
  (a.1=Q ∧ ∃ i : Lane BS BH D, ∃ odd, active s L HQ BS BH D i ∧
    a.2=address s sqb sqh sqd BS BH D odd i) ∨
  (a.1=K ∧ ∃ i : Lane BS BH D, ∃ odd, active s L HK BS BH D i ∧
    a.2=address s skb skh skd BS BH D odd i)

theorem dimension_lt {BS BH D : Nat} (odd : Bool) (i : Lane BS BH D) :
    dimension odd i < D := by
  have hi := i.2.2.1.isLt
  have hmod := Nat.mod_lt D (by omega : 0<2)
  have hdiv := Nat.mod_add_div D 2
  cases odd <;> simp only [dimension, Bool.false_eq_true, ↓reduceIte] <;> omega

/-- No odd final dimension is addressed when the requested rotary width is odd. -/
theorem dimension_lt_evenWidth {BS BH D : Nat} (odd : Bool) (i : Lane BS BH D) :
    dimension odd i < 2*(D/2) := by
  have hi := i.2.2.1.isLt
  cases odd <;> simp only [dimension, Bool.false_eq_true, ↓reduceIte] <;> omega

theorem regions_nonempty : RegionsSeparate "Q" "K" "Cos" "Sin" := by
  simp [RegionsSeparate]

/-- A nonempty block with two sequences, two heads, two rotary pairs, and
noncontiguous sequence storage satisfies the cross-store geometry domain. -/
theorem pair_geometry_nonempty (s : BlockState) :
    PairInjective s 16 4 1 2 2 4 := by
  rintro ⟨⟨a, b, c, u⟩, x⟩ ⟨⟨d, e, f, v⟩, y⟩ h
  have ha := a.isLt
  have hb := b.isLt
  have hc := c.isLt
  have hd := d.isLt
  have he := e.isLt
  have hf := f.isLt
  cases x <;> cases y <;>
    simp only [address, sequence, head, dimension, Bool.false_eq_true, ↓reduceIte,
      Nat.mul_one] at h
  all_goals
    have had : a=d := Fin.ext (by omega)
    have hbe : b=e := Fin.ext (by omega)
    have hcf : c=f := Fin.ext (by omega)
    subst d
    subst e
    subst f
    first | exact False.elim (by omega) | congr 3

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath

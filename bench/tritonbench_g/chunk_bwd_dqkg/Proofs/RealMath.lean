import VeriTile.Triton.Core
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

/-! Independent input-array formulas for the complete value-axis contraction.
All arrays denote inputs, never intermediate registers or executed outputs.
The H/DH column mask is the source's `NT * K`, including columns that cross a
key-block tail into a later time chunk. A narrower logical-K mask would describe
a different program. These real sum identities are not FP reassociations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealMath
open scoped BigOperators
noncomputable section

structure Layout where
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  NT : Nat
  s_k_h : Nat
  s_k_t : Nat
  s_v_h : Nat
  s_v_t : Nat
  s_h_h : Nat
  s_h_t : Nat
  scale : ℝ

structure Launch where
  key : Nat
  time : Nat
  head : Nat
  heads : Nat

structure Inputs where
  q : Nat → ℝ
  k : Nat → ℝ
  v : Nat → ℝ
  h : Nat → ℝ
  g : Nat → ℝ
  dout : Nat → ℝ
  dh : Nat → ℝ

def blocks (p : Layout) : Nat := (p.V + p.BV - 1) / p.BV

/-- The original value/dout mask; the value coordinate is global. -/
def value (p : Layout) (l : Launch) (a : Nat → ℝ) (r v : Nat) : ℝ :=
  if l.time * p.BT + r < p.T ∧ v < p.V then
    a (l.head * p.s_v_h + (l.time * p.BT + r) * p.s_v_t + v)
  else 0

/-- The original H/DH mask and strides, without a logical-K tail repair. -/
def history (p : Layout) (l : Launch) (a : Nat → ℝ) (v e : Nat) : ℝ :=
  if v < p.V ∧ l.time * p.K + l.key * p.BK + e < p.NT * p.K then
    a (l.head * p.s_h_h + v + (l.time * p.K + l.key * p.BK + e) * p.s_h_t)
  else 0

def queryKey (p : Layout) (l : Launch) (a : Nat → ℝ) (r e : Nat) : ℝ :=
  if l.time * p.BT + r < p.T ∧ l.key * p.BK + e < p.K then
    a (l.head * p.s_k_h + (l.time * p.BT + r) * p.s_k_t + l.key * p.BK + e)
  else 0

def gate (p : Layout) (l : Launch) (a : Inputs) (r : Nat) : ℝ :=
  if l.time * p.BT + r < p.T then a.g (l.head * p.T + l.time * p.BT + r) else 0

def lastGate (p : Layout) (l : Launch) (a : Inputs) : ℝ :=
  a.g (l.head * p.T + (min (l.time * p.BT + p.BT) p.T - 1))

def dqTerm (p : Layout) (l : Launch) (a : Inputs) (r e v : Nat) : ℝ :=
  value p l a.dout r v * history p l a.h v e

def dkTerm (p : Layout) (l : Launch) (a : Inputs) (r e v : Nat) : ℝ :=
  value p l a.v r v * history p l a.dh v e

def dsTerm (p : Layout) (l : Launch) (a : Inputs) (r s v : Nat) : ℝ :=
  value p l a.dout r v * value p l a.v s v

def dgLastTerm (p : Layout) (l : Launch) (a : Inputs) (v : Nat) : ℝ :=
  ∑ e : Fin p.BK, history p l a.h v e.val * history p l a.dh v e.val

structure Accumulators (BT BK : Nat) where
  dq : Fin BT → Fin BK → ℝ
  dk : Fin BT → Fin BK → ℝ
  ds : Fin BT → Fin BT → ℝ
  dgLast : ℝ

def prefixSum (width count : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ v ∈ Finset.range (count * width), f v

def blockSum (width count : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ v : Fin width, f (count * width + v.val)

def carries (p : Layout) (l : Launch) (a : Inputs) (count : Nat) :
    Accumulators p.BT p.BK where
  dq r e := prefixSum p.BV count (dqTerm p l a r.val e.val)
  dk r e := prefixSum p.BV count (dkTerm p l a r.val e.val)
  ds r s := prefixSum p.BV count (dsTerm p l a r.val s.val)
  dgLast := prefixSum p.BV count (dgLastTerm p l a)

def block (p : Layout) (l : Launch) (a : Inputs) (count : Nat) :
    Accumulators p.BT p.BK where
  dq r e := blockSum p.BV count (dqTerm p l a r.val e.val)
  dk r e := blockSum p.BV count (dkTerm p l a r.val e.val)
  ds r s := blockSum p.BV count (dsTerm p l a r.val s.val)
  dgLast := blockSum p.BV count (dgLastTerm p l a)

def total (p : Layout) (l : Launch) (a : Inputs) : Accumulators p.BT p.BK where
  dq r e := ∑ v ∈ Finset.range p.V, dqTerm p l a r.val e.val v
  dk r e := ∑ v ∈ Finset.range p.V, dkTerm p l a r.val e.val v
  ds r s := ∑ v ∈ Finset.range p.V, dsTerm p l a r.val s.val v
  dgLast := ∑ v ∈ Finset.range p.V, dgLastTerm p l a v

def add {BT BK : Nat} (x y : Accumulators BT BK) : Accumulators BT BK where
  dq r e := x.dq r e + y.dq r e
  dk r e := x.dk r e + y.dk r e
  ds r s := x.ds r s + y.ds r s
  dgLast := x.dgLast + y.dgLast

theorem prefix_zero (width : Nat) (f : Nat → ℝ) : prefixSum width 0 f = 0 := by
  simp [prefixSum]

theorem prefix_succ (width count : Nat) (f : Nat → ℝ) :
    prefixSum width (count + 1) f = prefixSum width count f + blockSum width count f := by
  simp only [prefixSum, Nat.add_mul, Nat.one_mul]
  rw [Finset.sum_range_add]
  simp only [blockSum, Finset.sum_range]

theorem partial_succ (p : Layout) (l : Launch) (a : Inputs) (count : Nat) :
    carries p l a (count + 1) = add (carries p l a count) (block p l a count) := by
  simp only [carries, add, block, prefix_succ]

theorem blocks_cover (p : Layout) (hBV : 0 < p.BV) : p.V ≤ blocks p * p.BV := by
  have h := Nat.lt_mul_div_succ (p.V + p.BV - 1) hBV
  rw [Nat.mul_add, Nat.mul_one, Nat.mul_comm p.BV] at h
  unfold blocks
  omega

theorem prefix_complete (p : Layout) (hBV : 0 < p.BV) (f : Nat → ℝ)
    (hzero : ∀ v, p.V ≤ v → f v = 0) :
    prefixSum p.BV (blocks p) f = ∑ v ∈ Finset.range p.V, f v := by
  unfold prefixSum
  symm
  apply Finset.sum_subset (Finset.range_mono (blocks_cover p hBV))
  intro v _ hv
  exact hzero v (Nat.le_of_not_lt (by simpa using hv))

/-- The complete loop includes every value coordinate exactly once, and the
partial final block adds only the source's zero-padded terms. -/
theorem partial_complete (p : Layout) (l : Launch) (a : Inputs) (hBV : 0 < p.BV) :
    carries p l a (blocks p) = total p l a := by
  have hq (r e : Nat) := prefix_complete p hBV (dqTerm p l a r e)
    (by intro v hv; simp [dqTerm, value, history, Nat.not_lt_of_ge hv])
  have hk (r e : Nat) := prefix_complete p hBV (dkTerm p l a r e)
    (by intro v hv; simp [dkTerm, value, history, Nat.not_lt_of_ge hv])
  have hs (r s : Nat) := prefix_complete p hBV (dsTerm p l a r s)
    (by intro v hv; simp [dsTerm, value, Nat.not_lt_of_ge hv])
  have hg := prefix_complete p hBV (dgLastTerm p l a)
    (by intro v hv; simp [dgLastTerm, history, Nat.not_lt_of_ge hv])
  simp only [carries, total, hq, hk, hs, hg]

/-- Pre-update gated DK. The last-row correction uses this quantity before
the later `transpose(ds) @ q` update, as in the source. -/
def gatedDk (p : Layout) (l : Launch) (a : Inputs) (r : Fin p.BT) (e : Fin p.BK) : ℝ :=
  (total p l a).dk r e * Real.exp (0 - gate p l a r.val + lastGate p l a)

def gatedDq (p : Layout) (l : Launch) (a : Inputs) (r : Fin p.BT) (e : Fin p.BK) : ℝ :=
  (total p l a).dq r e * Real.exp (gate p l a r.val) * p.scale

def score (p : Layout) (l : Launch) (a : Inputs) (r s : Fin p.BT) : ℝ :=
  if s.val ≤ r.val then
    (total p l a).ds r s * p.scale * Real.exp (gate p l a r.val - gate p l a s.val)
  else 0

def dq (p : Layout) (l : Launch) (a : Inputs) (r : Fin p.BT) (e : Fin p.BK) : ℝ :=
  gatedDq p l a r e + ∑ s : Fin p.BT, score p l a r s * queryKey p l a.k s.val e.val

def dk (p : Layout) (l : Launch) (a : Inputs) (r : Fin p.BT) (e : Fin p.BK) : ℝ :=
  gatedDk p l a r e + ∑ s : Fin p.BT, score p l a s r * queryKey p l a.q s.val e.val

def lastCorrection (p : Layout) (l : Launch) (a : Inputs) : ℝ :=
  (total p l a).dgLast * Real.exp (lastGate p l a) +
    ∑ r : Fin p.BT, ∑ e : Fin p.BK, gatedDk p l a r e * queryKey p l a.k r.val e.val

/-- No reverse cumulative sum is present: that line is commented out in the
source. Every active row past the stated cutoff receives the last correction. -/
def dg (p : Layout) (l : Launch) (a : Inputs) (r : Fin p.BT) : ℝ :=
  (∑ e : Fin p.BK, (queryKey p l a.q r.val e.val * dq p l a r e -
    queryKey p l a.k r.val e.val * dk p l a r e)) +
  if r.val < min p.BT (p.T - l.time * p.BT) - 1 then 0 else lastCorrection p l a

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealMath

import bench.tritonbench_g.chunk_bwd_dqkg.Kernels
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of the two complete sources in Kernels. Both compute the
three independent dq/dk/dg formulas below over ALL V coordinates, with the
source's masks, strides, partial tiles and final-row correction retained.

The IO contract includes successful flat execution and preservation of every
cell outside active outputs. It requires disjoint allocation intervals, the
active access bounds below, and zero unspecified values for omitted padding.
The source itself does not guarantee zero padding. Proofs/UndefinedTail.lean
refutes removing that premise: with V=1, a 16×16 tile and all-zero inputs,
one-valued unspecified lanes produce dq[0,0]=15 instead of 0.
This is a per-program real specification with mathematical integer addresses;
launch composition and concrete IEEE behavior are separate claims. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.Correctness
open VeriTile.Triton RealMath
open scoped BigOperators VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealIO
noncomputable section

/-- Dimensions T/K/V, tile sizes BT/BK/BV, NT, the six source strides and scale. -/
abbrev Parameters := RealMath.Layout
abbrev Inputs := RealMath.Inputs

section
variable (p : Parameters) (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)

/-- Read windows: q/k use (time*BT+r, key*BK+e), checked against T/K;
v/dout use (time*BT+r, j), checked against T/V. H/DH use
(j, time*K+key*BK+e), checked against V/(NT*K), with strides (1,s_h_t).
The latter is deliberately not a logical-K mask. Gate tiles check T, while
head*T + min(time*BT+BT,T)-1 is an unmasked scalar read.

`bounds` covers those windows and the three output windows shown here.
All seven input arrays are taken from INITIAL memory, before any stores. -/
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut
    p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
    p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealIO.projection p qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
  read := fun s =>
    { q := s.readMem q, k := s.readMem k, v := s.readMem v, h := s.readMem h,
      g := s.readMem g, dout := s.readMem dout, dh := s.readMem dh }
  bounds := fun s => RealSafety.Bounds s p q k v h g dout dh dqOut dkOut dgOut
  dq := dqOut
  dk := dkOut
  dg := dgOut
  qkWindow := fun l i => l.head*p.s_k_h + (l.time*p.BT+i.1.val)*p.s_k_t + l.key*p.BK+i.2.1.val
  dgWindow := fun l i => (l.key*l.heads+l.head)*p.T + l.time*p.BT+i.1.val
  qkMask := fun l i => l.time*p.BT+i.1.val < p.T ∧ l.key*p.BK+i.2.1.val < p.K
  dgMask := fun l i => l.time*p.BT+i.1.val < p.T

/-- The candidate only reverses the final dq addition; reads and writes agree. -/
def dotLeftIO : RealIO.IO p :=
  { originalIO p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut with
    kernel := (TypedSource.dotLeft qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut
      p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealIO.projection p qk valueStorage hist dhist Bool.true q k v h g dout dh dqOut dkOut dgOut }
end

/-- Independent input-array formula. `value`, `history`, `queryKey` and `gate`
are masked INITIAL array reads, zero outside their source windows. The whole-V
contractions are mathematical real sums. The source does not perform a reverse
cumulative sum of dg; its last-row correction is retained exactly. -/
def expected (p : Parameters) (l : Launch) (a : Inputs) : RealIO.Outputs p :=
  let dq0 := fun r e => ∑ j ∈ Finset.range p.V,
    value p l a.dout r j * history p l a.h j e
  let dk0 := fun r e => ∑ j ∈ Finset.range p.V,
    value p l a.v r j * history p l a.dh j e
  let ds := fun (r t : Fin p.BT) => if t.val ≤ r.val then
    (∑ j ∈ Finset.range p.V, value p l a.dout r.val j * value p l a.v t.val j) *
      p.scale * Real.exp (gate p l a r.val - gate p l a t.val) else 0
  let dq := fun (r : Fin p.BT) (e : Fin p.BK) =>
    dq0 r.val e.val * Real.exp (gate p l a r.val) * p.scale +
      ∑ t : Fin p.BT, ds r t * queryKey p l a.k t.val e.val
  let dkGated := fun (r : Fin p.BT) (e : Fin p.BK) =>
    dk0 r.val e.val * Real.exp (0 - gate p l a r.val + lastGate p l a)
  let dk := fun (r : Fin p.BT) (e : Fin p.BK) =>
    dkGated r e + ∑ t : Fin p.BT, ds t r * queryKey p l a.q t.val e.val
  let correction :=
    (∑ j ∈ Finset.range p.V, ∑ e : Fin p.BK, history p l a.h j e.val * history p l a.dh j e.val) *
      Real.exp (lastGate p l a) +
    ∑ r : Fin p.BT, ∑ e : Fin p.BK, dkGated r e * queryKey p l a.k r.val e.val
  { dq := dq, dk := dk, dg := fun r =>
      (∑ e : Fin p.BK, (queryKey p l a.q r.val e.val * dq r e - queryKey p l a.k r.val e.val * dk r e)) +
        if r.val < min p.BT (p.T-l.time*p.BT)-1 then 0 else correction }

/-- K≤row stride separates active output addresses, including a partial BK
block. There is no V≤BV, K≥BK, or divisibility requirement. -/
specification chunk_bwd_dqkg_correctness (p : Parameters)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (hBV : 0 < p.BV) (hstride : p.K ≤ p.s_k_t)
    (hDqDk : dqOut ≠ dkOut) (hDqDg : dqOut ≠ dgOut) (hDkDg : dkOut ≠ dgOut) :
    Spec.Real (originalIO p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut ⊨ expected p) := by
  exact RealIO.implements p qk valueStorage hist dhist Bool.false q k v h g dout dh dqOut dkOut dgOut
    hBV hstride hDqDk hDqDg hDkDg

specification chunk_dot_left_correctness (p : Parameters)
    (qk valueStorage hist dhist : TypedSource.Storage)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)
    (hBV : 0 < p.BV) (hstride : p.K ≤ p.s_k_t)
    (hDqDk : dqOut ≠ dkOut) (hDqDg : dqOut ≠ dgOut) (hDkDg : dkOut ≠ dgOut) :
    Spec.Real (dotLeftIO p qk valueStorage hist dhist q k v h g dout dh dqOut dkOut dgOut ⊨ expected p) := by
  exact RealIO.implements p qk valueStorage hist dhist Bool.true q k v h g dout dh dqOut dkOut dgOut
    hBV hstride hDqDk hDqDg hDkDg

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.Correctness

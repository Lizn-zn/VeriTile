import bench.tritonbench_g.context_attn_nopad.Kernels
import bench.tritonbench_g.context_attn_nopad.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Both complete sources compute the independent initial-array natural-exp
weighted sum below in real semantics, with successful finite flat execution
and preservation of every tagged cell outside active output stores.

Kernels.lean retains all fp16/bf16/fp32 storage branches, signed metadata,
causal negative-infinity masks, natural tl.exp, the normalized online
accumulator and probability/output casts. Precision is erased only for these
real specifications. BN must be positive; source starts are nonnegative,
active addresses are in bounds, and active outputs are distinct. Dimensions,
strides and program IDs remain symbolic. This is per-program correctness;
fixed-width index overflow and concurrent launch behavior are outside this
contract. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Correctness
open VeriTile VeriTile.Triton Formula Program
open scoped VeriTile.Bench.TritonBenchG.ContextAttnNopad.RealIO
noncomputable section
set_option maxRecDepth 5000
set_option maxHeartbeats 800000

def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence }
  valid := fun l a => ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
    RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Two metadata cells, masked Q, reached K/V loads and active stores.
  -- The start is nonnegative so converting the K/V pointer advance preserves
  -- its signed value; every active address is also checked before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < Formula.sequence p l a

def scaleFirstIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p :=
  { originalIO p r storage scale with
    kernel := (kernel p r storage Bool.true scale).eraseDType
    projection := RealIO.projection p r storage Bool.true scale }

/-- Query/key/value lookups use only the initial arrays and source time masks. -/
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale

/-- The causal negative-infinity mask has zero real weight; unmasked lanes use
natural exponential, matching tl.exp after precision erasure. -/
def weight (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ queryRow p l i then Real.exp (score p l a scale i j) else 0

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  (∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val * value p l a j.val i.2.1.val) /
    ∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val

specification original_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (originalIO p r storage scale ⊨ expected p scale) := by
  exact RealIO.implements_closed p r storage Bool.false scale hBN

specification scaleFirst_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (scaleFirstIO p r storage scale ⊨ expected p scale) := by
  exact RealIO.implements_closed p r storage Bool.true scale hBN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Correctness

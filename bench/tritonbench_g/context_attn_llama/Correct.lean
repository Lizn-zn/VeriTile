import bench.tritonbench_g.context_attn_llama.Kernels
import bench.tritonbench_g.context_attn_llama.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Both complete sources compute the initial-array weighted sum below in
real semantics, with successful finite flat execution and a full-cell frame.

Kernels.lean preserves fp16/bf16/fp32 storage, signed int32 metadata, grouped
KV heads, time masks, base-2 exponential and the fused accumulator dot.
Precision is erased only for this real specification. BN must be positive;
active addresses must be nonnegative and in bounds, and active outputs must
be distinct. Strides, dimensions and program IDs remain symbolic. This is
per-program correctness, without a claim about fixed-width index overflow or
concurrent launch behavior. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.Correctness
open VeriTile VeriTile.Triton Formula Program
open scoped VeriTile.Bench.TritonBenchG.ContextAttnLlama.RealIO
noncomputable section
set_option maxRecDepth 5000
set_option maxHeartbeats 800000

def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence,
    prompt := u.readMemValue .int r.prompt, request := u.readMemValue .int r.request,
    pages := u.readMemValue .int r.pages }
  valid := fun l a => ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
    RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Four metadata cells, masked Q, reached page/K/V lanes and active output stores.
  -- Each active address is checked before its signed-to-natural conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start (batch p l)+queryRow p l i.1.val)*p.outRow+
    head p l*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < Formula.sequence p l a

def scaleFirstIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p :=
  { originalIO p r storage scale with
    kernel := (kernel p r storage Bool.true scale).eraseDType
    projection := RealIO.projection p r storage Bool.true scale }

/-- Query/key/value lookups use the initial arrays and the source time masks.
The finite causal sentinel gives future lanes positive real weights. Padded
values are zero, while their finite scores still affect the denominator. -/
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt (batch p l) then
    (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale
  else -100000000

/-- 2^score, matching the source's tl.math.exp2 operation in real semantics. -/
def weight (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  Real.exp (Real.log 2 * score p l a scale i j)

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
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.Correctness

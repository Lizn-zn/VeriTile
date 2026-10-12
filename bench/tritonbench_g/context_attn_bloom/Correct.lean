import bench.tritonbench_g.context_attn_bloom.Kernels
import bench.tritonbench_g.context_attn_bloom.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Both complete sources compute the guarded normalized-block formula below
in real semantics. The contract includes successful finite flat execution,
active-address bounds and preservation of every other tagged cell.

Q/K/V/Out storage can be fp16, bf16 or fp32; precision is erased only for real
correctness. Sizes, strides, launch IDs, grouped KV heads, signed metadata,
gathered pages and partial time/channel masks remain symbolic. Required:
positive BN, nonnegative prompt length, distinct active output addresses and
nonnegative in-bounds active addresses. This is per-program correctness; it
does not prove fixed-width index arithmetic or concurrent launch behavior. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.Correctness
open VeriTile VeriTile.Triton Math Program RealSetup
open scoped VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealIO
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
    tokens := u.readMemValue .int r.tokens }
  valid := fun l a => 0 ≤ a.prompt l.batch ∧
    ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
      RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Four metadata cells, masked Q, reached page-table/K/V lanes and output.
  -- Each address is checked as signed and nonnegative before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < sequence a l ∧ i.2.1.val < p.headDim

def scaleFirstIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p :=
  { originalIO p r storage scale with
    kernel := (kernel p r storage Bool.true scale).eraseDType
    projection := RealIO.projection p r storage Bool.true scale }

/-- Masked query/key/value lookups read the initial arrays and signed page
indices. The causal mask uses the actual finite sentinel, so future lanes
still have positive mathematical exponential weights. -/
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt l.batch then
    scale * ∑ e : Fin p.D, query p l a i e.val * key p l a j e.val
  else -100000000

def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN), if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*value p l a j.val d else 0

/-- Independent prefix-sum recurrence. The separate acc_scale guard leaves
the earlier accumulator unscaled for wholly future blocks. Replacing this
formula with ideal causal softmax would change the source's real behavior. -/
def normalizedBlocks (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 => normalizedBlocks p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) / denominator p l a scale i (c+1)

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  normalizedBlocks p l a scale i.1.val i.2.1.val (count p l a)

specification original_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (originalIO p r storage scale ⊨ expected p scale) := by
  exact RealIO.implements_formula p r storage Bool.false scale hBN
    (fun l a i => normalizedBlocks p l a scale i.1.val i.2.1.val) (by intros; rfl) (by intros; rfl)

specification scaleFirst_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (scaleFirstIO p r storage scale ⊨ expected p scale) := by
  exact RealIO.implements_formula p r storage Bool.true scale hBN
    (fun l a i => normalizedBlocks p l a scale i.1.val i.2.1.val) (by intros; rfl) (by intros; rfl)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.Correctness

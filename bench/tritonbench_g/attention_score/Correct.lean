import bench.tritonbench_g.attention_score.Kernels
import bench.tritonbench_g.attention_score.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of the complete score kernels after precision erasure.
Each output key receives a sum of base-2 weights over the visited query rows.
Q, K and the row normalization M are independent inputs; no omitted producer
of M is assumed correct. The contract supplies successful flat execution,
active-address safety, all masked outputs and preservation of other memory.

Both window flags, both load flags, grouped-head strides, signed windows,
empty loops and partial query blocks remain symbolic. As in the source,
IS_EVEN_N also controls the query-tail selection. The source wrapper uses
one common query/key block size B. Host launch composition and fixed-width
integer overflow are outside this per-program mathematical statement. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.AttentionScore.RealIO
noncomputable section

/-- Q/K/M/Out regions; independent batch, head, row and inner strides; head
counts and context sizes; scale; signed offset/size; window and even flags.
The source's unused output inner stride and batch count remain parameters. -/
abbrev Parameters := RealSource.Parameters
abbrev Inputs := RealInputs.Inputs

def blocks (B : Nat) (p : Parameters) : Nat := (p.roundedContext + B - 1) / B
def rows (B : Nat) (p : Parameters) : Nat := blocks B p * B

/-- Checked Q/K loads require only their active addresses; unchecked loads
require whole windows. M is always read for every visited row, including
padding. K is read even if ROUND_CTX=0. Only active output keys are written.
The shared flat IO contract checks physical allocation and disjoint regions. -/
def originalIO (B D : Nat) (p : Parameters) : RealIO.IO (rows B p) B D where
  kernel := (TypedSource.original p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
    p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl).eraseDType
  projection := RealSource.projection B D p Bool.false
  Q := p.Q
  K := p.K
  M := p.M
  Out := p.Out
  queryWindow := fun _ headBatch _ i =>
    headBatch / p.heads * p.queryBatch + headBatch % p.heads * p.queryHead +
      i.1.val * p.queryRow + i.2.1.val * p.queryDim
  keyWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.keyBatch + (headBatch % p.heads) / (p.heads / p.kvHeads) * p.keyHead +
      i.1.val * p.keyDim + (startN * B + i.2.1.val) * p.keyRow
  normalizerWindow := fun _ headBatch _ i => headBatch * p.roundedContext + i.1.val
  outputWindow := fun startN headBatch _ i =>
    headBatch / p.heads * p.outputBatch + headBatch % p.heads * p.outputHead + startN * B + i.val
  queryMask := fun _ _ _ i => p.evenM = Bool.true ∨ i.1.val < p.queryContext
  keyMask := fun startN _ _ i => p.evenN = Bool.true ∨ startN * B + i.2.1.val < p.keyContext
  outputMask := fun startN _ _ i => startN * B + i.val < p.keyContext

def scalarLeftIO (B D : Nat) (p : Parameters) : RealIO.IO (rows B p) B D :=
  { originalIO B D p with
    kernel := (TypedSource.scalarLeft p.Q p.K p.M p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
      p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl).eraseDType
    projection := RealSource.projection B D p Bool.true }

/-- The exact decimal is the source's conversion of scale to base 2.
Boundary-checked inactive Q/K lanes contribute zero to the dot product. -/
def score (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  (∑ d : Fin D,
    (if p.evenM = Bool.true ∨ row.val < p.queryContext then xs.query row d else 0) *
    (if p.evenN = Bool.true ∨ startN*B+key.val < p.keyContext then xs.key key d else 0)) *
      (p.scale * 1.4426950408889634)

/-- The regular window keeps 0 ≤ query-key+offset < size. The complement
arm keeps distance ≥ size. With windows disabled all distances are allowed. -/
def windowAllows (p : Parameters) (row key : Nat) : Prop :=
  p.window = Bool.false ∨ if p.complement then p.size ≤ (row : Int) - key + p.offset
    else 0 ≤ (row : Int) - key + p.offset ∧ (row : Int) - key + p.offset < p.size

instance (p : Parameters) (row key : Nat) : Decidable (windowAllows p row key) := by
  unfold windowAllows
  infer_instance

/-- The source selects query rows using evenN, independently of the Q-load
flag evenM. Keep this distinction, including padded rows when evenN=true. -/
def weight (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D)
    (row : Fin (rows B p)) (key : Fin B) : ℝ :=
  if windowAllows p row.val (startN*B+key.val) ∧
      (p.evenN = Bool.true ∨ row.val < p.queryContext) then
    pow2 (score B D startN p xs row key - xs.normalizer row)
  else 0

def expected (B D startN : Nat) (p : Parameters) (xs : Inputs (rows B p) B D) (key : Fin B) : ℝ :=
  ∑ block : Fin (blocks B p), ∑ row : Fin B,
    weight B D startN p xs (RealInputs.rowIndex block row) key

specification score_correctness (B D : Nat) (p : Parameters) (hB : 0 < B) :
    Spec.Real (originalIO B D p ⊨ fun startN _ xs => expected B D startN p xs) := by
  exact RealContract.implements B D p Bool.false hB

specification score_scalar_left_correctness (B D : Nat) (p : Parameters) (hB : 0 < B) :
    Spec.Real (scalarLeftIO B D p ⊨ fun startN _ xs => expected B D startN p xs) := by
  exact RealContract.implements B D p Bool.true hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.Correctness

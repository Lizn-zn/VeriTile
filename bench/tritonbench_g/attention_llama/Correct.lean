import bench.tritonbench_g.attention_llama.Kernels
import bench.tritonbench_g.attention_llama.Proofs.RealIO
import bench.tritonbench_g.attention_llama.Proofs.FP8RealIO
import VeriTile.Meta.StatementAudit

/-! Independent natural-exp attention for both original/carry-first sources,
both causal modes and both K/V storage formats. `candidate` selects carry-first;
`causal` selects the source's shifted visibility. H is the query length, NH the
head count, N the key length and SP the nonnegative start_position.

The contracts erase numerical precision, retain actual FP8 byte decoding,
and include successful flat execution, active outputs and all other cells.
The displayed precondition admits whole streamed blocks and nonempty visible
causal rows. It does not claim conventional attention for partial key tails or
empty visible rows; the source's local offs_n score mask is retained unchanged.
No FP admission rule is used in this file. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.Correctness
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked3DKernelIO₃
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

/-- Q is static across steps; K/V advance by BN rows. All four buffers keep
their independent batch/head/row/column strides. Query and output masks use H;
the whole-block launch precondition makes all streamed K/V lanes active. -/
def io (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ where
  kernel := (if candidate then
    TypedSource.carryFirst Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  else
    TypedSource.original Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).eraseDType
  projection := by
    cases candidate
    · exact RealSource.original_projection_supported ..
    · exact RealSource.candidate_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  T := T
  B1 := BM * BD
  B2 := BN * BD
  B3 := BN * BD
  C := BM * BD
  outDType := .real
  pre := fun p _ _ => if causal then (p + 1) * BN + SP = BN * T ∧ BN * T ≤ N ∧ SP ≤ p * BM
    else N = BN * T
  read1 := fun p h _ _ j => h / NH * sqz + h % NH * sqh +
    (p * BM + j.val / BD) * sqm + j.val % BD * sqk
  read2 := fun _ h _ t j => h / NH * skz + h % NH * skh +
    (t.val * BN + j.val / BD) * skn + j.val % BD * skk
  read3 := fun _ h _ t j => h / NH * svz + h % NH * svh +
    (t.val * BN + j.val / BD) * svk + j.val % BD * svn
  write := fun p h _ j => h / NH * soz + h % NH * soh +
    (p * BM + j.val / BD) * som + j.val % BD * son
  mask1 := fun p _ _ _ j => p * BM + j.val / BD < H
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask := fun p _ _ j => p * BM + j.val / BD < H

/-- The FP8 sources use the same addresses and masks, loading signed integer
K/V cells before bit reinterpretation and fp16 conversion inside the kernel. -/
def byteIO (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ :=
  { io Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal with
    kernel := (if candidate then
      TypedSource.carryFirstFP8 Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
    else
      TypedSource.originalFP8 Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).eraseDType
    projection := FP8RealSource.projection_supported .. }

/-- Array indexing only: query reads the first stream slice, zeroing masked
rows; stream reconstructs global key k from step k/BN and lane k%BN. -/
noncomputable abbrev query := RealIO.query
noncomputable abbrev stream := RealIO.stream

noncomputable def score (T BM BN BD H : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (scale : ℝ) (p : Nat) (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys : Fin T → Fin (BN * BD) → ℝ) (r : Fin BM) (k : Fin (BN * T)) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin BD =>
    query T BM BD p H hT xs (r, d, PUnit.unit) * stream T BN BD hBN ys (k, d, PUnit.unit))

/-- Independent Σ exp(score)*V / Σ exp(score). Under the causal launch
precondition, k ≤ p*BM-SP+r is exactly k+SP ≤ p*BM+r. This source shrinks
visibility by SP; it does not add a prefix cache to the visible keys. -/
noncomputable def expected (T BM BN BD H SP : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (causal : Bool) (scale : ℝ) (p : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys zs : Fin T → Fin (BN * BD) → ℝ) (j : Fin (BM * BD)) : ℝ :=
  let i := Lane2D.decode j
  let s := score T BM BN BD H hBN hT scale p xs ys i.1
  if causal then
    let weight := fun k : Fin (BN * T) =>
      if k.val ≤ p * BM - SP + i.1.val then Real.exp (s k) else 0
    (Finset.univ.sum fun k => weight k * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => weight k)
  else
    (Finset.univ.sum fun k => Real.exp (s k) * stream T BN BD hBN zs (k, i.2.1, PUnit.unit)) /
      (Finset.univ.sum fun k => Real.exp (s k))

/-- Finite signed E5M2 bytes: invalid int8 values, NaNs and infinities have no
real interpretation here. Inputs pin exact .int cells, including signed zero;
only the mathematical formula decodes them. -/
abbrev Byte := FP8RealIO.Byte

def byteCell (b : Byte) : MemCell := .of .int b.val

noncomputable def decode (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.getD 0 : ℚ) : ℝ)

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool)

/-- Each active output equals the formula for the initial input arrays; all
other cells remain unchanged. Injectivity prevents overlapping output lanes. -/
specification attention_correctness (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    Spec.Real (io Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal ⊨
      fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs ys zs) := by
  exact RealIO.correct Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal hBM hBN hT hinj

/-- The same formula over the original integer K/V buffers, with exact E5M2
decoding. Q/Out are distinct from K/V so real cells cannot replace byte cells. -/
specification attention_fp8_correctness (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    Spec.Real ((byteIO Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal).ImplementsCells
      MemCell.real byteCell byteCell
      (fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs
        (fun t i => decode (ys t i)) (fun t i => decode (zs t i)))) := by
  exact FP8RealIO.correct Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal
    hBM hBN hT hQK hQV hOK hOV hinj

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.Correctness

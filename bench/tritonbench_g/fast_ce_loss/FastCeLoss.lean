import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FastCeLoss

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented backward final-store slice of `fast_ce_loss.py`'s
`_cross_entropy_backward`.

The full kernel builds `y` from logits, logsumexp, labels, optional logit
scaling, and optional softcapping. This slice starts from a precomputed gradient
tile `Grad` and proves the final masked in-place writeback
`logits[col_offsets] = dloss * Grad[col_offsets]`. -/
def cross_entropy_backward_store_slice
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(axis=0)
  block_idx = tl.program_id(axis=1)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  y = tl.load(Grad + row_idx * $(grad_row_stride) + col_offsets,
    mask=mask, other=0.0)
  tl.store(logits_ptr + row_idx * $(logits_row_stride) + col_offsets,
    dloss * y, mask=mask)
}

def colOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * BLOCK_SIZE + i.val

def active (s : BlockState) (VOCAB_SIZE BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Prop :=
  colOffset s BLOCK_SIZE i < VOCAB_SIZE

instance activeDecidable (s : BlockState) (VOCAB_SIZE BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s VOCAB_SIZE BLOCK_SIZE i) := by
  unfold active
  infer_instance

def logitsOffset
    (s : BlockState) (logits_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * logits_row_stride + colOffset s BLOCK_SIZE i

def gradOffset
    (s : BlockState) (grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * grad_row_stride + colOffset s BLOCK_SIZE i

noncomputable def expectedBackward
    (s : BlockState) (dloss_ptr Grad : RegionName)
    (dloss_row_stride grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) *
    s.readMem Grad (gradOffset s grad_row_stride BLOCK_SIZE i)

/-- Algorithm-layer correctness for the masked backward writeback. -/
theorem cross_entropy_backward_store_slice_correct
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (cross_entropy_backward_store_slice logits_ptr dloss_ptr Grad
        VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem logits_ptr (logitsOffset s logits_row_stride BLOCK_SIZE i) =
        if active s VOCAB_SIZE BLOCK_SIZE i then
          expectedBackward s dloss_ptr Grad dloss_row_stride grad_row_stride
            BLOCK_SIZE i
        else
          s.readMem logits_ptr (logitsOffset s logits_row_stride BLOCK_SIZE i) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * logits_row_stride + (s.pids 1 * BLOCK_SIZE + idx.1.val)) := by
    intro a b h
    have hInner :
        s.pids 1 * BLOCK_SIZE + a.1.val =
          s.pids 1 * BLOCK_SIZE + b.1.val := by
      exact Nat.add_left_cancel h
    have hab : a.1 = b.1 := Fin.ext (Nat.add_left_cancel hInner)
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, cross_entropy_backward_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [logitsOffset, colOffset, active, expectedBackward, gradOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hActive : s.pids 1 * BLOCK_SIZE + i.val < VOCAB_SIZE
  · simp [hActive]
  · simp [hActive]

/-- Compute-facing correctness for the masked backward writeback. -/
theorem cross_entropy_backward_store_slice_compute_correct
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_backward_store_slice logits_ptr dloss_ptr Grad
        VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s VOCAB_SIZE BLOCK_SIZE i)
        (fun i => (logits_ptr, logitsOffset s logits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedBackward s dloss_ptr Grad dloss_row_stride grad_row_stride
          BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_backward_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := cross_entropy_backward_store_slice_correct logits_ptr dloss_ptr Grad
    VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE
    s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.FastCeLoss

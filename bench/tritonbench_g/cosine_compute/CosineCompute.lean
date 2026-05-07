import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.CosineCompute

open VeriTile.Triton VeriTile.Examples

/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
-/
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, b_value, mask=mask)
}

/-! ## Correctness -/

/-- Algorithm-layer correctness for `cos_func`.

For each lane `i ∈ Fin BLOCK_SIZE`:
* In-bounds (`pid * BLOCK_SIZE + i < n_elements`): the output region holds
  `Real.cos (xs i)`.
* Out-of-bounds: the value at `pid * BLOCK_SIZE + i` is preserved from the
  initial state (mask=false → no store).

The `(a_value).to(tl.float32)` cast at the compute layer reduces to
`Op.castFloat`, which is the identity at the algorithm layer (post-erasure
all dtypes unify to `ℝ`). -/
theorem cos_func_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let addr := s.pid * BLOCK_SIZE + i.val
      observeAt (exec (cos_func a b n_elements BLOCK_SIZE) s)
                b BLOCK_SIZE s.pid i
        = some (if addr < n_elements then Real.cos (xs i)
                else s.readMem b addr) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, cos_func, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, WithBot.realCos,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hi, h_x]
  · simp [hi]

/-- **Compute-facing `ComputeCorrect` for `cos_func`.**

For any region names, any block-size > 0, any pre-loaded input tile `xs`, the
kernel writes `Real.cos (xs i)` at every in-bounds lane and preserves
out-of-bounds lanes verbatim. -/
theorem cos_func_compute_correct
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    ComputeKernel.ComputeCorrect
      (cos_func a b n_elements BLOCK_SIZE)
      (fun s0 s' =>
        s0 = s →
        ∀ i : Fin BLOCK_SIZE,
          let addr := s.pid * BLOCK_SIZE + i.val
          observeAt (some s') b BLOCK_SIZE s.pid i
            = some (if addr < n_elements then Real.cos (xs i)
                    else s.readMem b addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := cos_func_correct a b n_elements BLOCK_SIZE
    hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa using hi

end VeriTile.Bench.TritonBenchG.CosineCompute

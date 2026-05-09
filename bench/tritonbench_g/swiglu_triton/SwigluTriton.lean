import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SwigluTriton

open VeriTile.Triton

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python pointer mutation `a += ...` / `b += ...` / `c += ...` -> explicit
  base pointer registers.
- The helper `silu(x)` is inlined as `x * tl.sigmoid(x)`. -/
def swiglu_forward_kernel
    (A B C : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  A_base = A + program_id * $(stride)
  B_base = B + program_id * $(stride)
  C_base = C + program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(A_base + col_offsets, mask=mask, other=0.0).to(tl.float32)
  b_row = tl.load(B_base + col_offsets, mask=mask, other=0.0)
  c_row = a_row * tl.sigmoid(a_row) * b_row
  tl.store(C_base + col_offsets, c_row, mask=mask)
}

noncomputable def swigluSpec (a b : ℝ) : ℝ :=
  a * Real.sigmoid a * b

def swigluOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride + i.val

/-- Algorithm-layer correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i)
    (hExec : exec (swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := swigluOffset s stride i
      s'.readMem C outAddr =
        if i.val < n_cols then
          swigluSpec (as i) (bs i)
        else s.readMem C outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, swiglu_forward_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        NumericDType.mul, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot] at hExec
  subst s'
  simp only [swigluOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · have ha := h_a i
    have hb := h_b i
    simp [swigluOffset] at ha hb
    simp [hi, swigluSpec, FloatDType.toWithBot, ha, hb]
  · simp [hi]

/-- Compute-facing correctness for `_swiglu_forward_kernel`. -/
theorem swiglu_forward_kernel_compute_correct
    (A B C : RegionName)
    (stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (as bs : Fin BLOCK_SIZE → ℝ)
    (h_a : ∀ i : Fin BLOCK_SIZE, s.readMem A (swigluOffset s stride i) = as i)
    (h_b : ∀ i : Fin BLOCK_SIZE, s.readMem B (swigluOffset s stride i) = bs i) :
    ComputeCorrect.Realizes
      (kernel := swiglu_forward_kernel A B C stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (C, swigluOffset s stride i)))
      (expected := fun i => swigluSpec (as i) (bs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [swiglu_forward_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := swiglu_forward_kernel_correct A B C stride n_cols BLOCK_SIZE
    s s' as bs h_a h_b hExec i
  simp [hActive] at hi
  exact hi

end VeriTile.Bench.TritonBenchG.SwigluTriton

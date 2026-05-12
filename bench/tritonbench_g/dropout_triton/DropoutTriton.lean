import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DropoutTriton

open VeriTile.Triton

/-- Faithful transcription of `dropout_triton.py`'s `_dropout`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `x_keep_ptr` is a typed Lean boolean region so its `tl.load` call does not
  need an extra `dtype=` kwarg. -/
def dropout_kernel
    (x_ptr : RegionName) (x_keep_ptr : Region .bool) (output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load($((x_keep_ptr : Region .bool)) + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}

def dropoutOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

noncomputable def dropoutSpec
    (s : BlockState) (x_ptr x_keep_ptr : RegionName)
    (p : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  if s.readMemValue .bool x_keep_ptr (dropoutOffset s BLOCK_SIZE i) then
    s.readMem x_ptr (dropoutOffset s BLOCK_SIZE i) / (1 - p)
  else
    0.0

/-- Algorithm-layer correctness for `_dropout`.

Active lanes write the scaled-or-zero dropout result; inactive tail lanes are
preserved. -/
theorem dropout_kernel_correct
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (dropout_kernel x_ptr x_keep_ptr output_ptr
          n_elements p BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := dropoutOffset s BLOCK_SIZE i
      s'.readMem output_ptr outAddr =
        if outAddr < n_elements then
          dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i
        else s.readMem output_ptr outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pid * BLOCK_SIZE + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, dropout_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt] at hExec
  subst s'
  simp only [dropoutOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hBounds : s.pid * BLOCK_SIZE + i.val < n_elements
  · simp [hBounds, dropoutSpec, dropoutOffset]
    cases hKeep : s.readMemValue .bool x_keep_ptr (s.pid * BLOCK_SIZE + i.val) <;>
      simp
  · simp [hBounds]

/-- Compute-facing correctness for `_dropout`. -/
theorem dropout_kernel_compute_correct
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dropout_kernel x_ptr x_keep_ptr output_ptr
        n_elements p BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => dropoutOffset s BLOCK_SIZE i < n_elements)
        (fun i => (output_ptr, dropoutOffset s BLOCK_SIZE i)))
      (expected := fun i => dropoutSpec s x_ptr x_keep_ptr p BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [dropout_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := dropout_kernel_correct x_ptr x_keep_ptr output_ptr
    n_elements p BLOCK_SIZE s s' hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.DropoutTriton

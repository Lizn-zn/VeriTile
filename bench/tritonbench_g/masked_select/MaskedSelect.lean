import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MaskedSelect

open VeriTile.Triton

/-- Faithful transcription of `masked_select.py`'s `masked_select_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- The in-body `tl.region` directive declares element dtypes for the
  `select_mask_ptr` (Boolean mask buffer) and `prefix_sum_ptr` (int64
  prefix-sum buffer). The `select_mask` load picks up `tl.int1` via the
  region directive; the expression-position `prefix_sum_ptr` load similarly
  picks up `tl.uint64` before subtracting one. -/
def masked_select_kernel
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region select_mask_ptr = tl.int1, prefix_sum_ptr = tl.uint64
  pid = tl.program_id(axis=0)
  offsets = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
  select_mask = tl.load(select_mask_ptr + offsets, mask=mask)
  out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=$(0)) - $(1)
  tl.store(out_ptr + out_offset, inp, mask=select_mask and mask)
}

def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

def maskedSelectOutOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i) - 1

def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  (if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1

def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true

instance activeDecidable
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s select_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold active
  infer_instance

/-- Algorithm-layer cellwise correctness for active masked-select lanes. -/
theorem masked_select_kernel_correct
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)) :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      (exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
          n_elements BLOCK_SIZE) s).map
          (fun s' => s'.readMem out_ptr
            (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i))
        = some (s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  intro i hActive
  simp [exec, masked_select_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, tile_elementwise,
        NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ h
    have hab : a = b := hOutInj (by
      simpa [maskedSelectStoreOffset, maskedSelectOffset, BlockState.pid_eq,
        BlockState.readMemValue] using h)
    subst b
    rfl
  simp [maskedSelectOffset, maskedSelectStoreOffset, BlockState.readMemValue]
  rcases hActive with ⟨hBounds, hMask⟩
  have hBoundsRaw : s.pid * BLOCK_SIZE + i.val < n_elements := by
    simpa [maskedSelectOffset] using hBounds
  have hMaskRaw :
      s.readMemValue .bool select_mask_ptr (s.pid * BLOCK_SIZE + i.val) = Bool.true := by
    simpa [maskedSelectOffset] using hMask
  simpa [hBoundsRaw, hMaskRaw] using
    (BlockState.scatter_readback_prop_masked_nd
      (region := out_ptr)
      (shape := [BLOCK_SIZE])
      (offsetFn := fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            (match s.readMemTyped TileDType.nat prefix_sum_ptr
                (s.pid * BLOCK_SIZE + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat)
          else
            0) - 1)
      (valueFn := fun idx : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            some (s.readMem inp_ptr (s.pid * BLOCK_SIZE + idx.1.val))
          else
            some 0.0))
      (P := fun idx : TileIndex [BLOCK_SIZE] =>
        (if s.pid * BLOCK_SIZE + idx.1.val < n_elements then
            s.readMemValue .bool select_mask_ptr (s.pid * BLOCK_SIZE + idx.1.val) = Bool.true
          else
            BlockState.defaultCarrier TileDType.bool = Bool.true) ∧
          s.pid * BLOCK_SIZE + idx.1.val < n_elements)
      _ hRawInj (i, PUnit.unit))

/-- Executed-state form of `masked_select_kernel_correct`. -/
theorem masked_select_kernel_correct_of_exec
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i))
    (s' : BlockState)
    (hExec : exec (masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      active s select_mask_ptr n_elements BLOCK_SIZE i →
      s'.readMem out_ptr
          (maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)
        = s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i) := by
  intro i hActive
  have h := masked_select_kernel_correct inp_ptr select_mask_ptr prefix_sum_ptr
    out_ptr n_elements BLOCK_SIZE s hOutInj i hActive
  rw [hExec] at h
  simpa using h

/-- Compute-facing correctness for active masked-select lanes. -/
theorem masked_select_kernel_compute_correct
    (inp_ptr select_mask_ptr prefix_sum_ptr out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)) :
    ComputeCorrect.Realizes
      (kernel := masked_select_kernel inp_ptr select_mask_ptr prefix_sum_ptr out_ptr
        n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s select_mask_ptr n_elements BLOCK_SIZE)
        (fun i => (out_ptr,
          maskedSelectStoreOffset s prefix_sum_ptr n_elements BLOCK_SIZE i)))
      (expected := fun i => s.readMem inp_ptr (maskedSelectOffset s BLOCK_SIZE i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [masked_select_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  exact masked_select_kernel_correct_of_exec inp_ptr select_mask_ptr prefix_sum_ptr
    out_ptr n_elements BLOCK_SIZE s hOutInj s' hExec i hActive

end VeriTile.Bench.TritonBenchG.MaskedSelect

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul

open VeriTile.Triton

/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}

def bOffset (s : BlockState) (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn

def fpbOffset (s : BlockState) (stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn

def nOffset (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def dequantizeActive (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
    s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N

instance dequantizeActiveDecidable
    (s : BlockState) (K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) :
    Decidable (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  unfold dequantizeActive
  infer_instance

/-- Exact dequantized value written at active tile lane `idx`. -/
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N idx.2.1)

private noncomputable def dequantizeStoreBase
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    BlockState :=
  ((((((((((s.setReg "k_block_idx" .nat [] (Tile.scalar (s.pids 0))).setReg
    "n_block_idx" .nat [] (Tile.scalar (s.pids 1))).setReg
    "offs_k" .nat [BLOCK_SIZE_K] (Tile.vec fun i => i.val)).setReg
    "offs_n" .nat [BLOCK_SIZE_N] (Tile.vec fun i => i.val)).setReg
    "b_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "fpb_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "bs_offs" .nat [1, BLOCK_SIZE_N]
      { data := fun i => nOffset s BLOCK_SIZE_N i.2.1 }).setReg
    "n_mask" .bool [1, BLOCK_SIZE_N]
      { data := fun i => decide (nOffset s BLOCK_SIZE_N i.2.1 < N) }).setReg
    "mask" .bool [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => decide (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K i) }).setReg
    "int_b" .real [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i =>
          if dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K i then
            some (s.readMem b_ptr (bOffset s stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K i))
          else some (0.0 : ℝ) }).setReg
    "scale_b" .real [1, BLOCK_SIZE_N]
      { data := fun i =>
          if nOffset s BLOCK_SIZE_N i.2.1 < N then
            some (s.readMem b_scale_ptr (nOffset s BLOCK_SIZE_N i.2.1))
          else some (0.0 : ℝ) }

/-- Algorithm-layer correctness for `dequantize_kernel`. -/
theorem dequantize_kernel_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K))
    (hExec : exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
          stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N],
      s'.readMem fpb_ptr
          (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)
        = if dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K idx then
          dequantizeSpec s b_ptr b_scale_ptr stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx
        else
          s.readMem fpb_ptr
            (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  intro idx
  simp [exec, dequantize_kernel, stepStmts, stepStmt, evalOp, Option.bind,
        Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
        (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
  let active : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  let valueFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (if active idx then
            some
              (s.readMem b_ptr
                ((s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
                  (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn))
          else some (0.0 : ℝ))
          (if s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N then
            some
              (s.readMem b_scale_ptr
                (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))
          else some (0.0 : ℝ)))
  have hOffsetInj : Function.Injective
      offsetFn := by
    simpa [offsetFn, fpbOffset] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := fpb_ptr)
    (s := dequantizeStoreBase s b_ptr b_scale_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := active)
    hOffsetInj idx
  by_cases hActive : active idx
  · rcases hActive with ⟨hk, hn⟩
    simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hk, hn,
      dequantizeStoreBase, TileShape.dropInsertedIndex, Bool.and_eq_true] using hscatter
  · simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hActive,
      dequantizeStoreBase, TileShape.dropInsertedIndex] using hscatter

/-- Compute-facing correctness for `dequantize_kernel`. -/
theorem dequantize_kernel_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)) :
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s K N BLOCK_SIZE_N BLOCK_SIZE_K)
          (fun idx => (fpb_ptr,
            fpbOffset s stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := dequantize_kernel_correct b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s s' hOutInj hExec idx
  simpa [hActive] using h

/-- Python test path `(K,N)=(128,256)`, contiguous `fp_b`, config
`BLOCK_SIZE_N=128`, `BLOCK_SIZE_K=128`. -/
theorem dequantize_matmul_python_128x128_offset_injective
    (s : BlockState) :
    Function.Injective (fpbOffset s 256 1 128 128) := by
  intro a b h
  rcases a with ⟨ka, na, ha⟩
  rcases b with ⟨kb, nb, hb⟩
  rcases na with ⟨na, hna⟩
  rcases nb with ⟨nb, hnb⟩
  simp [fpbOffset] at h
  have hk : ka = kb := by omega
  have hn : na = nb := by omega
  subst kb
  subst nb
  rfl

/-- Python test path `(K,N)=(128,256)`, contiguous `fp_b`, config
`BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=256`. -/
theorem dequantize_matmul_python_64x256_offset_injective
    (s : BlockState) :
    Function.Injective (fpbOffset s 256 1 64 256) := by
  intro a b h
  rcases a with ⟨ka, na, ha⟩
  rcases b with ⟨kb, nb, hb⟩
  rcases na with ⟨na, hna⟩
  rcases nb with ⟨nb, hnb⟩
  simp [fpbOffset] at h
  have hk : ka = kb := by omega
  have hn : na = nb := by omega
  subst kb
  subst nb
  rfl

/-- Python test path `(K,N)=(128,256)`, contiguous `fp_b`, config
`BLOCK_SIZE_N=32`, `BLOCK_SIZE_K=256`. -/
theorem dequantize_matmul_python_32x256_offset_injective
    (s : BlockState) :
    Function.Injective (fpbOffset s 256 1 32 256) := by
  intro a b h
  rcases a with ⟨ka, na, ha⟩
  rcases b with ⟨kb, nb, hb⟩
  rcases na with ⟨na, hna⟩
  rcases nb with ⟨nb, hnb⟩
  simp [fpbOffset] at h
  have hk : ka = kb := by omega
  have hn : na = nb := by omega
  subst kb
  subst nb
  rfl

/-- Python test path `(K,N)=(128,256)`, contiguous `fp_b`, config
`BLOCK_SIZE_N=256`, `BLOCK_SIZE_K=64`. -/
theorem dequantize_matmul_python_256x64_offset_injective
    (s : BlockState) :
    Function.Injective (fpbOffset s 256 1 256 64) := by
  intro a b h
  rcases a with ⟨ka, na, ha⟩
  rcases b with ⟨kb, nb, hb⟩
  rcases na with ⟨na, hna⟩
  rcases nb with ⟨nb, hnb⟩
  simp [fpbOffset] at h
  have hk : ka = kb := by omega
  have hn : na = nb := by omega
  subst kb
  subst nb
  rfl

/-- Python `test_matmul_dequantize_int8`, config
`BLOCK_SIZE_N=128`, `BLOCK_SIZE_K=128`. -/
theorem dequantize_matmul_python_128x128_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr 128 256 256 1 256 1
        128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 128 128)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 128 128 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 128 128 idx) := by
  exact dequantize_kernel_compute_correct b_ptr b_scale_ptr fpb_ptr
    128 256 256 1 256 1 128 128 s
    (dequantize_matmul_python_128x128_offset_injective s)

/-- Python `test_matmul_dequantize_int8`, config
`BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=256`. -/
theorem dequantize_matmul_python_64x256_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr 128 256 256 1 256 1
        64 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 64 256)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 64 256 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 64 256 idx) := by
  exact dequantize_kernel_compute_correct b_ptr b_scale_ptr fpb_ptr
    128 256 256 1 256 1 64 256 s
    (dequantize_matmul_python_64x256_offset_injective s)

/-- Python `test_matmul_dequantize_int8`, config
`BLOCK_SIZE_N=32`, `BLOCK_SIZE_K=256`. -/
theorem dequantize_matmul_python_32x256_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr 128 256 256 1 256 1
        32 256)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 32 256)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 32 256 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 32 256 idx) := by
  exact dequantize_kernel_compute_correct b_ptr b_scale_ptr fpb_ptr
    128 256 256 1 256 1 32 256 s
    (dequantize_matmul_python_32x256_offset_injective s)

/-- Python `test_matmul_dequantize_int8`, config
`BLOCK_SIZE_N=256`, `BLOCK_SIZE_K=64`. -/
theorem dequantize_matmul_python_256x64_compute_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr 128 256 256 1 256 1
        256 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (dequantizeActive s 128 256 256 64)
          (fun idx => (fpb_ptr, fpbOffset s 256 1 256 64 idx)))
      (expected := fun idx =>
        dequantizeSpec s b_ptr b_scale_ptr 256 1 256 64 idx) := by
  exact dequantize_kernel_compute_correct b_ptr b_scale_ptr fpb_ptr
    128 256 256 1 256 1 256 64 s
    (dequantize_matmul_python_256x64_offset_injective s)

end VeriTile.Bench.TritonBenchG.DequantizeMatmul

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.ScatterStore

/-!
# `triton_attention` — strict per-kernel correctness

`triton_attention.py` is a full FlashAttention training pipeline of three
`@triton.jit` kernels: `_fwd_kernel` (online-softmax forward, stores the output
`Out` plus the running `L`/`M` log-sum-exp rows), `_bwd_preprocess` (computes
`NewDO = DO` and the per-row `Delta = sum(DO·O)`), and `_bwd_kernel` (the main
backward producing the `DQ`/`DK`/`DV` gradients, with the score-side `P`/`DS`
arithmetic as an inner step). Scaling is `1/√D`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`grid = (cdiv(T, BLOCK), B·H, 1)`, the
`torch.autograd.Function` orchestration that chains forward → preprocess →
backward, and how the runtime composes per-program writes into each buffer) is
the *trusted boundary*, not a proof obligation here. Because the program ids
are universally quantified (via `s`), the per-program statements cover every
program of each grid.

## Proof architecture

```
triton_attention_bwd_python_test_shape_complete_summary             ← TOP THEOREM (bwd grads + score)
  ├─ triton_attention_bwd_grads_python_test_shape_output_summary
  │    ├─ triton_attention_bwd_kernel_toAlgorithm_supported
  │    └─ triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
  │         └─ triton_attention_bwd_kernel_{dq,dk,dv}_python_test_shape_compute_correct
  │              └─ triton_attention_bwd_{dq,dkdv}_store_slice_compute_correct ...
  └─ triton_attention_bwd_score_python_test_shape_formula_summary
       └─ triton_attention_bwd_score_{p,ds}_formula_python_test_shape_compute_correct
            └─ triton_attention_bwd_score_{p,ds}_formula_slice_compute_correct   ← closed-form P/DS

triton_attention_forward_python_test_shape_output_summary           ← TOP (forward Out/L/M)
  └─ triton_attention_forward_surface_{out,l,m}_python_test_shape_compute_correct
       └─ triton_attention_forward_{output,l,m}_store_slice_compute_correct → ..._correct

triton_attention_bwd_preprocess_python_test_shape_output_summary    ← TOP (preprocess NewDO/Delta)
  └─ triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
       └─ triton_attention_bwd_preprocess_{newdo,delta}_surface_compute_correct
            └─ ..._{store,formula}_slice_compute_correct → ..._correct
```
(Offset injectivity discharged by the `triton_attention_python_*_offset_injective` lemmas.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp2`/`log2`, the
`tl.dot` `float16` accumulations, and `1/√D` scaling are not modeled at the bit
level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled. The
modeling depth differs by kernel:

* **Forward** (`Out`, `L`, `M`) and **backward grads** (`DQ`, `DK`, `DV`) are
  **final-store scoped**: the proofs establish the masked/full stores write the
  accumulator slices at the correct, injective offsets and preserve inactive
  lanes; the written values are the opaque `producedTritonAttentionForward*` /
  `producedBwdKernelD{Q,K,V}Value` carriers for the online-softmax forward and
  backward recurrences, which are **not** re-derived as closed-form formulas.
* **Backward preprocess** and the **backward score `P`/`DS` step** are verified
  against explicit closed-form specs (`bwdScorePFormulaSpec`,
  `bwdScoreDSFormulaSpec`, and the `newdo`/`delta` formula slices), not opaque
  carriers — these inner arithmetic steps are checked, the surrounding loop
  composition is trusted.

Side conditions: the test-shape summaries fix `(B,H,T,D) = (2,4,128,64)`,
`BLOCK_M = 128`, `BLOCK_DMODEL = 64`, strides `(32768, 8192, 64, 1)`,
`num_block = 1`, `sm_scale = 1/√64`; the complete backward summary additionally
requires `PTile ≠ DSTile`.
-/

namespace VeriTile.Bench.TritonBenchG.TritonAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- DSL port of `triton_attention.py`'s `_fwd_kernel`. -/
def triton_attention_fwd_kernel
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)

  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_hz * stride_qh_2d, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))
  out_tile_ptr = tl.make_block_ptr(base=Out,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(off_hz * stride_qh_2d + start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  q = tl.load(q_tile_ptr)

  for start_n in range($(0), (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_N), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_N), $(0)])
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_ptrs = L + off_hz * $(N_CTX) + offs_m
  m_ptrs = M + off_hz * $(N_CTX) + offs_m
  tl.store(l_ptrs, l_prev)
  tl.store(m_ptrs, m_prev)

  acc = (acc).to(tl.float16)
  tl.store(out_tile_ptr, acc, boundary_check=(0, 1))
}

/-- The full Python-shaped forward attention surface lowers to the algorithm
layer, including the streaming softmax loop and final L/M/O stores. -/
theorem triton_attention_fwd_kernel_toAlgorithm_supported
    (Q K V L M Out : RegionName)
    (sm_scale : ℝ)
    (_stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _stride_oz _stride_oh stride_om stride_on
      _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (triton_attention_fwd_kernel Q K V L M Out sm_scale _stride_qz
      stride_qh stride_qm stride_qk _stride_kz _stride_kh stride_kn
      stride_kk _stride_vz _stride_vh stride_vk stride_vn _stride_oz
      _stride_oh stride_om stride_on _Z _H N_CTX D0 BLOCK_M BLOCK_DMODEL
      BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented forward output-store slice of
`triton_attention.py`'s `_fwd_kernel`.

The Python kernel writes `acc` through a block pointer with
`boundary_check=(0, 1)`. This slice spells the same write as explicit pointer
arithmetic and an explicit two-axis boundary mask. The inner `tl.float32`
streaming-softmax accumulator is outside this slice. -/
def triton_attention_forward_output_store_slice
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] + $(hzRowOffset) < $(D0)) &
    (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + offs_m[:, None] * $(BLOCK_DMODEL) + offs_d[None, :],
    mask=mask, other=0.0)
  tl.store(Out + (offs_m[:, None] + $(hzRowOffset)) * $(stride_om) +
      offs_d[None, :] * $(stride_on), (acc).to(tl.float16), mask=mask)
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (hzRowOffset D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  rowIndex s BLOCK_M idx.1 + hzRowOffset < D0

instance activeDecidable (s : BlockState) (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s hzRowOffset D0 BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  rowIndex s BLOCK_M idx.1 * BLOCK_DMODEL + dIndex idx

def outOffset (s : BlockState) (hzRowOffset stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (rowIndex s BLOCK_M idx.1 + hzRowOffset) * stride_om + dIndex idx * stride_on

noncomputable def storeValue (s : BlockState) (Acc : RegionName)
    (hzRowOffset D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s hzRowOffset D0 BLOCK_M idx then
      some (s.readMem Acc (accOffset s BLOCK_M BLOCK_DMODEL idx))
    else some (0.0 : ℝ))

theorem triton_attention_forward_output_store_slice_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s hzRowOffset stride_om stride_on BLOCK_M idx
      (exec (triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
            stride_om stride_on BLOCK_M BLOCK_DMODEL) s).map
          (·.mem Out outAddr)
        = some (if active s hzRowOffset D0 BLOCK_M idx then
            MemCell.of .fp16
              (FloatDType.real.cast FloatDType.fp16
                (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
          else s.mem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_forward_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, dIndex, active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset) * stride_om +
        idx.2.1.val * stride_on
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (WithBot.unbotD 0
          (if s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0 then
            some (s.readMem Acc
              ((s.pids 0 * BLOCK_M + idx.1.val) * BLOCK_DMODEL + idx.2.1.val))
          else some (0.0 : ℝ))))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, rowIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMemTyped .fp16 Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).mem Out
        (offsetFn idx) =
    if P idx then
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))
    else s.mem Out (offsetFn idx)
  rw [scatter_memcell_fp16_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val + hzRowOffset < D0
  · rfl
  · rfl

theorem triton_attention_forward_output_store_slice_compute_correct
    (Acc Out : RegionName) (hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out hzRowOffset D0
        stride_om stride_on BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s hzRowOffset D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s hzRowOffset stride_om stride_on BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset D0 BLOCK_M BLOCK_DMODEL idx)))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_forward_output_store_slice_correct Acc Out
    hzRowOffset D0 stride_om stride_on BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented L (log-sum-exp) row store slice of `triton_attention.py`'s
forward kernel. Writes a precomputed `LPrev` vector into `L` at the per-row
`off_hz * N_CTX + offs_m` strided offset. Companion to the output store
slice. -/
def triton_attention_forward_l_store_slice
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  l_prev = tl.load(LPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(L + $(off_hz) * $(N_CTX) + offs_m, l_prev)
}

def lRowOffset (s : BlockState) (off_hz N_CTX BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  off_hz * N_CTX + (s.pids 0 * BLOCK_M + i.val)

noncomputable def lStoreSpec (s : BlockState) (LPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem LPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_l_store_slice_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
          s).map (·.readMem L outAddr)
        = some (lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_l_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [lStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_l_store_slice_compute_correct
    (LPrev L : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => lStoreSpec s LPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_l_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_l_store_slice_correct LPrev L
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented M (max) row store slice of `triton_attention.py`'s forward
kernel. Mirrors the L-row store slice. -/
def triton_attention_forward_m_store_slice
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  m_prev = tl.load(MPrev + $(off_hz) * $(N_CTX) + offs_m)
  tl.store(M + $(off_hz) * $(N_CTX) + offs_m, m_prev)
}

noncomputable def mStoreSpec (s : BlockState) (MPrev : RegionName)
    (off_hz N_CTX BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem MPrev (lRowOffset s off_hz N_CTX BLOCK_M i)

theorem triton_attention_forward_m_store_slice_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ∀ i : Fin BLOCK_M,
      let outAddr := lRowOffset s off_hz N_CTX BLOCK_M i
      (exec (triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
          s).map (·.readMem M outAddr)
        = some (mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] =>
        off_hz * N_CTX + (s.pids 0 * BLOCK_M + idx.1.val)) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : lRowOffset s off_hz N_CTX BLOCK_M a =
        lRowOffset s off_hz N_CTX BLOCK_M b := by
      simpa [lRowOffset, Nat.add_assoc] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  simp [exec, triton_attention_forward_m_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul]
  simp only [lRowOffset, Nat.add_assoc]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [mStoreSpec, lRowOffset, Nat.add_assoc]

theorem triton_attention_forward_m_store_slice_compute_correct
    (MPrev M : RegionName) (off_hz N_CTX BLOCK_M : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lRowOffset s off_hz N_CTX BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz N_CTX BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (M, lRowOffset s off_hz N_CTX BLOCK_M i))
      (expected := fun i => mStoreSpec s MPrev off_hz N_CTX BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_forward_m_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_forward_m_store_slice_correct MPrev M
    off_hz N_CTX BLOCK_M s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ### Auxiliary backward-preprocess slices

The `_bwd_preprocess` kernel in `triton_attention.py` is much simpler than the
forward / backward main kernels: it loads `O`, `DO`, `L`, computes
`do = do / L[:, None]`, then writes `do` to `NewDO` and `sum(o * do, axis=1)`
to `Delta`. The streaming softmax / tl.dot / make_block_ptr / advance pieces
that block the main attention loop are absent. The two store-back regions
support clean proof-oriented slices analogous to the forward `L`/`M` row store
slices already in this file.
-/

/-- DSL port of `triton_attention.py`'s `_bwd_preprocess`. -/
def triton_attention_bwd_preprocess
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  do_val = do_val / denom[:, None]
  delta = tl.sum(o * do_val, axis=1)
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
  tl.store(Delta + off_m, delta)
}

theorem triton_attention_bwd_preprocess_toAlgorithm_supported
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD).toAlgorithm? = Except.ok alg := by
  simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- DSL port of `triton_attention.py`'s main `_bwd_kernel`.

The Python test shape has `num_block = 1`, so the post-inner-loop pointer reset
`lo + (1 - num_block) * BLOCK_M` is exactly zero. Under that checked launch,
this surface preserves the block-pointer construction, nested loops, causal
mask, DQ accumulation store, DK/DV accumulator stores, and pointer advances. -/
def triton_attention_bwd_kernel
    (Q K V _Out DO DQ DK DV _L M Delta : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL _BLOCK_N : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  stride_qz_2d = $(stride_qz) // $(stride_qm) // $(stride_qk)
  stride_qh_2d = $(stride_qh) // $(stride_qm) // $(stride_qk)
  q_tile_ptr = tl.make_block_ptr(base=Q,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  k_tile_ptr = tl.make_block_ptr(base=K,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_kn), $(stride_kk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  v_tile_ptr = tl.make_block_ptr(base=V,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  do_tile_ptr = tl.make_block_ptr(base=DO,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dq_tile_ptr = tl.make_block_ptr(base=DQ,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dk_tile_ptr = tl.make_block_ptr(base=DK,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  dv_tile_ptr = tl.make_block_ptr(base=DV,
    shape=($(D0), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  DQ = DQ + off_z * $(stride_qz) + off_h * $(stride_qh)
  for start_n in range($(0), $(num_block), $(1)) {
    lo = start_n * $(BLOCK_M)
    offs_qm = lo + tl.arange(0, $(BLOCK_M))
    offs_n = start_n * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    offs_m = tl.arange(0, $(BLOCK_M))
    offs_k = tl.arange(0, $(BLOCK_DMODEL))
    dq_ptrs = DQ + (offs_qm[:, None] * $(stride_qm) +
      offs_k[None, :] * $(stride_qk))
    D_ptrs = Delta + off_hz * $(N_CTX)
    m_ptrs = M + off_hz * $(N_CTX)
    dv = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    dk = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    k = tl.load(k_tile_ptr, boundary_check=(0, 1))
    v = tl.load(v_tile_ptr, boundary_check=(0, 1))
    for start_m in range(lo, $(num_block) * $(BLOCK_M), $(BLOCK_M)) {
      offs_m_curr = start_m + offs_m
      q = tl.load(q_tile_ptr, boundary_check=(0, 1))
      qk = tl.dot(q, tl.trans(k))
      qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk,
        float("-inf"))
      m = tl.load(m_ptrs + offs_m_curr)
      p = tl.exp(qk * $((sm_scale : ℝ)) - m[:, None])
      do_val = tl.load(do_tile_ptr, boundary_check=(0, 1))
      dv += tl.dot(tl.trans((p).to(tl.float16)), do_val)
      Di = tl.load(D_ptrs + offs_m_curr)
      dp = tl.zeros([$(BLOCK_M), $(BLOCK_M)], dtype=tl.float32) - Di[:, None]
      dp += tl.dot(do_val, tl.trans(v))
      ds = p * dp * $((sm_scale : ℝ))
      dk += tl.dot(tl.trans((ds).to(tl.float16)), q)
      dq = tl.load(dq_tile_ptr)
      dq += tl.dot((ds).to(tl.float16), k)
      tl.store(dq_tile_ptr, dq)
      dq_ptrs += $(BLOCK_M) * $(stride_qm)
      q_tile_ptr = tl.advance(q_tile_ptr, [$(BLOCK_M), $(0)])
      do_tile_ptr = tl.advance(do_tile_ptr, [$(BLOCK_M), $(0)])
      dq_tile_ptr = tl.advance(dq_tile_ptr, [$(BLOCK_M), $(0)])
    }
    q_tile_ptr = tl.advance(q_tile_ptr, [$(0), $(0)])
    do_tile_ptr = tl.advance(do_tile_ptr, [$(0), $(0)])
    dq_tile_ptr = tl.advance(dq_tile_ptr, [$(0), $(0)])
    k_tile_ptr = tl.advance(k_tile_ptr, [$(BLOCK_M), $(0)])
    v_tile_ptr = tl.advance(v_tile_ptr, [$(BLOCK_M), $(0)])
    tl.store(dv_tile_ptr, (dv).to(tl.float16), boundary_check=(0, 1))
    tl.store(dk_tile_ptr, (dk).to(tl.float16), boundary_check=(0, 1))
    dv_tile_ptr = tl.advance(dv_tile_ptr, [$(BLOCK_M), $(0)])
    dk_tile_ptr = tl.advance(dk_tile_ptr, [$(BLOCK_M), $(0)])
  }
}

theorem triton_attention_bwd_kernel_toAlgorithm_supported
    (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn stride_kk
      _stride_vz _stride_vh stride_vk stride_vn
      _Z H N_CTX D0 num_block BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      sm_scale stride_qz stride_qh stride_qm stride_qk _stride_kz _stride_kh
      stride_kn stride_kk _stride_vz _stride_vh stride_vk stride_vn _Z H
      N_CTX D0 num_block BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? =
        Except.ok alg := by
  simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented `NewDO` 2D store slice of `triton_attention.py`'s
`_bwd_preprocess`. The kernel stores a (precomputed) `NewDOAcc` tile to
`NewDO` at strided offset `off_m[:, None] * D_HEAD + off_n[None, :]`. -/
def triton_attention_bwd_preprocess_newdo_store_slice
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = tl.load(NewDOAcc + off_m[:, None] * $(D_HEAD) + off_n[None, :])
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], do_val)
}

def newdoMIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def newdoNIndex (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  idx.2.1.val

def newdoOffset (s : BlockState) (BLOCK_M D_HEAD : Nat)
    (idx : TileIndex [BLOCK_M, D_HEAD]) : Nat :=
  newdoMIndex s BLOCK_M idx.1 * D_HEAD + newdoNIndex idx

noncomputable def newdoStoreSpec (s : BlockState) (NewDOAcc : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem NewDOAcc (newdoOffset s BLOCK_M D_HEAD idx)

theorem triton_attention_bwd_preprocess_newdo_store_slice_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_store_slice
            NewDOAcc NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        newdoOffset, newdoMIndex, newdoNIndex, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem NewDOAcc
      ((s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoStoreSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    (NewDOAcc NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoStoreSpec s NewDOAcc BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_store_slice_correct
    NewDOAcc NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Formula-level `NewDO` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers the Python arithmetic

`do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)`
`denom = tl.load(L + off_m).to(tl.float32)`
`do = do / denom[:, None]`
`tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)`.
-/
def triton_attention_bwd_preprocess_newdo_formula_slice
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  tl.store(NewDO + off_m[:, None] * $(D_HEAD) + off_n[None, :], new_do)
}

noncomputable def newdoFormulaSpec (s : BlockState) (DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  s.readMem DO (newdoOffset s BLOCK_M D_HEAD idx) /
    s.readMem L (newdoMIndex s BLOCK_M idx.1)

theorem triton_attention_bwd_preprocess_newdo_formula_slice_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ∀ idx : TileIndex [BLOCK_M, D_HEAD],
      let outAddr := newdoOffset s BLOCK_M D_HEAD idx
      (exec (triton_attention_bwd_preprocess_newdo_formula_slice
            DO L NewDO BLOCK_M D_HEAD) s).map (·.readMem NewDO outAddr)
        = some (newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  intro idx
  simp [exec, triton_attention_bwd_preprocess_newdo_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, newdoOffset, newdoMIndex, newdoNIndex,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, D_HEAD] → Nat :=
    fun idx => (s.pids 0 * BLOCK_M + idx.1.val) * D_HEAD + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, D_HEAD] → ℝ :=
    fun idx => s.readMem DO (offsetFn idx) /
      s.readMem L (s.pids 0 * BLOCK_M + idx.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, newdoOffset, newdoMIndex, newdoNIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        acc.writeMem NewDO (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, D_HEAD])).readMem NewDO
        (offsetFn idx) =
    newdoFormulaSpec s DO L BLOCK_M D_HEAD idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [newdoFormulaSpec, newdoOffset, newdoMIndex, newdoNIndex,
    offsetFn, valueFn]

theorem triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    (DO L NewDO : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        newdoOffset s BLOCK_M D_HEAD idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx => newdoFormulaSpec s DO L BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_newdo_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_preprocess_newdo_formula_slice_correct
    DO L NewDO BLOCK_M D_HEAD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Proof-oriented `Delta` 1D row store slice of `triton_attention.py`'s
`_bwd_preprocess`. Mirrors the L-row store slice of the forward kernel:
load a (precomputed) `DeltaAcc` row vector and write it to `Delta` at
`off_m`. -/
def triton_attention_bwd_preprocess_delta_store_slice
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  delta_val = tl.load(DeltaAcc + off_m)
  tl.store(Delta + off_m, delta_val)
}

def deltaOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

noncomputable def producedBwdPreprocessNewDOValue
    (s : BlockState) (Out DO L NewDO Delta : RegionName)
    (BLOCK_M D_HEAD : Nat) (idx : TileIndex [BLOCK_M, D_HEAD]) : ℝ :=
  match exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD) s with
  | some s' => s'.readMem NewDO (newdoOffset s BLOCK_M D_HEAD idx)
  | none => 0.0

noncomputable def producedBwdPreprocessDeltaValue
    (s : BlockState) (Out DO L NewDO Delta : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  match exec (triton_attention_bwd_preprocess Out DO L NewDO Delta
      BLOCK_M D_HEAD) s with
  | some s' => s'.readMem Delta (deltaOffset s BLOCK_M i)
  | none => 0.0

noncomputable def deltaStoreSpec (s : BlockState) (DeltaAcc : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem DeltaAcc (deltaOffset s BLOCK_M i)

/-- Formula-level `Delta` slice of `triton_attention.py`'s `_bwd_preprocess`.
It covers `do = do / L[:, None]` followed by
`delta = tl.sum(o * do, axis=1)` and the row store to `Delta`. -/
def triton_attention_bwd_preprocess_delta_formula_slice
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) :
    ComputeKernel := triton {
  off_m = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(D_HEAD))
  o = (tl.load(Out + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  do_val = (tl.load(DO + off_m[:, None] * $(D_HEAD) + off_n[None, :])).to(tl.float32)
  denom = (tl.load(L + off_m)).to(tl.float32)
  new_do = do_val / denom[:, None]
  delta = tl.sum(o * new_do, axis=1)
  tl.store(Delta + off_m, delta)
}

noncomputable def deltaFormulaSpec (s : BlockState) (Out DO L : RegionName)
    (BLOCK_M D_HEAD : Nat) (i : Fin BLOCK_M) : ℝ :=
  ∑ j : Fin D_HEAD,
    let idx : TileIndex [BLOCK_M, D_HEAD] :=
      TileShape.insertAxisIndex [BLOCK_M, D_HEAD] 1
        (TileShape.insertAxisIndex [BLOCK_M] 0 PUnit.unit i) j
    s.readMem Out (newdoOffset s BLOCK_M D_HEAD idx) *
      newdoFormulaSpec s DO L BLOCK_M D_HEAD idx

theorem triton_attention_bwd_preprocess_delta_formula_slice_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s s' : BlockState)
    (hExec : exec (triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem Delta (deltaOffset s BLOCK_M i) =
        deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i := by
  intro i
  simp [exec, triton_attention_bwd_preprocess_delta_formula_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, NumericDType.add,
        NumericDType.mul, NumericDType.div, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaFormulaSpec, newdoFormulaSpec, newdoOffset, newdoMIndex,
    newdoNIndex, deltaOffset, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.mul, NumericDType.div]
  congr

theorem triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    (Out DO L Delta : RegionName) (BLOCK_M D_HEAD : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaFormulaSpec s Out DO L BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_formula_slice,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact triton_attention_bwd_preprocess_delta_formula_slice_correct Out DO L
    Delta BLOCK_M D_HEAD s s' hExec i

theorem triton_attention_bwd_preprocess_delta_store_slice_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ∀ i : Fin BLOCK_M,
      let outAddr := deltaOffset s BLOCK_M i
      (exec (triton_attention_bwd_preprocess_delta_store_slice
            DeltaAcc Delta BLOCK_M) s).map (·.readMem Delta outAddr)
        = some (deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M] => s.pids 0 * BLOCK_M + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, triton_attention_bwd_preprocess_delta_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul]
  simp only [deltaOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [deltaStoreSpec, deltaOffset]

theorem triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    (DeltaAcc Delta : RegionName) (BLOCK_M : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i => deltaStoreSpec s DeltaAcc BLOCK_M i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess_delta_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := triton_attention_bwd_preprocess_delta_store_slice_correct
    DeltaAcc Delta BLOCK_M s i
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_preprocess_newdo_surface_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        some (NewDO, newdoOffset s BLOCK_M D_HEAD idx))
      (expected := fun idx : TileIndex [BLOCK_M, D_HEAD] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta
          BLOCK_M D_HEAD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdPreprocessNewDOValue, hExec]

theorem triton_attention_bwd_preprocess_delta_surface_compute_correct
    (Out DO L NewDO Delta : RegionName) (BLOCK_M D_HEAD : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        BLOCK_M D_HEAD)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (Delta, deltaOffset s BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta
          BLOCK_M D_HEAD i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_preprocess, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedBwdPreprocessDeltaValue, hExec]

/-! ### Main backward gradient store slices

The main `_bwd_kernel` accumulates `dq`, `dk`, and `dv` through nested dot
loops. The slices below start after those accumulators have been materialized
and cover the Python-observed gradient writebacks. `DQ` is stored without a
boundary check in the source kernel; `DK` and `DV` use block-pointer
`boundary_check=(0, 1)`.
-/

def triton_attention_bwd_dq_store_slice
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def triton_attention_bwd_dkdv_store_slice
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(D0)) & (offs_k[None, :] < $(BLOCK_DMODEL))
  grad = tl.load(GradPre + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (grad).to(Out.dtype.element_ty), mask=mask)
}

def bwdOffZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 / H

def bwdOffH (s : BlockState) (H : Nat) : Nat :=
  s.pids 0 % H

def bwdRowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * BLOCK_M + i.val

def bwdColIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def bwdGradOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  bwdOffZ s H * stride_qz + bwdOffH s H * stride_qh +
    bwdRowIndex s BLOCK_M idx.1 * stride_qm + bwdColIndex idx * stride_qk

def bwdGradActive (s : BlockState) (D0 BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  bwdRowIndex s BLOCK_M idx.1 < D0

instance bwdGradActiveDecidable
    (s : BlockState) (D0 BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (bwdGradActive s D0 BLOCK_M idx) := by
  unfold bwdGradActive
  infer_instance

noncomputable def bwdGradStoreSpec
    (s : BlockState) (GradPre : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  s.readMem GradPre
    (bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)

/-- One inner-loop DQ update slice from `_bwd_kernel`:
`dq += tl.dot(ds.to(tl.float16), k)` followed by the DQ tile store. The
precomputed `DS` and `KTile` regions stand for the source kernel's `ds` tile
and loaded `k` tile at one loop step. -/
def triton_attention_bwd_dq_dot_step_slice
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  dq = tl.load(DQPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  ds = (tl.load(DS + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :])).to(tl.float16)
  k = tl.load(KTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  dq += tl.dot(ds, k)
  tl.store(DQ + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), dq)
}

def bwdDsOffset (s : BlockState) (BLOCK_M : Nat)
    (row : Fin BLOCK_M) (k : Fin BLOCK_M) : Nat :=
  bwdRowIndex s BLOCK_M row * BLOCK_M + k.val

def bwdKTileOffset (BLOCK_DMODEL : Nat) (k : Fin BLOCK_M)
    (col : Fin BLOCK_DMODEL) : Nat :=
  k.val * BLOCK_DMODEL + col.val

noncomputable def bwdDqDotStepSpec
    (s : BlockState) (DQPrev DS KTile : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s DQPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ k : Fin BLOCK_M,
      FloatDType.fp16.storeValue
        (FloatDType.real.cast FloatDType.fp16
          (some (s.readMem DS (bwdDsOffset s BLOCK_M idx.1 k)))) *
        s.readMem KTile (bwdKTileOffset BLOCK_DMODEL k idx.2.1)

/-! ### Main backward score/DS arithmetic

The dot-step proofs below consume `P` and `DS` tiles. This slice covers the
source `_bwd_kernel` inner-loop arithmetic that produces those two tiles for
one query/key block:

* `qk = tl.dot(q, tl.trans(k))`
* causal masking of `qk`
* `p = tl.exp(qk * sm_scale - m[:, None])`
* `dp = tl.dot(do, tl.trans(v)) - D[:, None]`
* `ds = p * dp * sm_scale`

For the checked Python launch `num_block = 1`, this is the only inner-loop
score update feeding the public DQ/DK/DV writeback summaries.
-/

def triton_attention_bwd_score_formula_slice
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  offs_m = tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  q = tl.load(QTile + offs_m[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  k = tl.load(KTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  v = tl.load(VTile + offs_n[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  do_val = tl.load(DOTile + offs_m[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  qk = tl.dot(q, tl.trans(k))
  qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
  m = tl.load(MVec + offs_m)
  p = tl.exp(qk * $((sm_scale : ℝ)) - m[:, None])
  Di = tl.load(DeltaVec + offs_m)
  dp = tl.zeros([$(BLOCK_M), $(BLOCK_M)], dtype=tl.float32) - Di[:, None]
  dp += tl.dot(do_val, tl.trans(v))
  ds = p * dp * $((sm_scale : ℝ))
  tl.store(PTile + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :], p)
  tl.store(DSTile + offs_m[:, None] * $(BLOCK_M) + offs_n[None, :], ds)
}

def bwdScoreOffset (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_M]) : Nat :=
  idx.1.val * BLOCK_M + idx.2.1.val

def bwdLocalTileOffset (BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.1.val * BLOCK_DMODEL + idx.2.1.val

noncomputable def bwdLocalTile
    (s : BlockState) (R : RegionName) (BLOCK_M BLOCK_DMODEL : Nat) :
    Tile .real [BLOCK_M, BLOCK_DMODEL] :=
  { data := fun idx =>
      some (s.readMem R (bwdLocalTileOffset (BLOCK_M := BLOCK_M)
        BLOCK_DMODEL idx)) }

noncomputable def bwdScoreQKTile
    (s : BlockState) (QTile KTile : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  Tile.dot [] (bwdLocalTile s QTile BLOCK_M BLOCK_DMODEL)
    (Tile.transpose [] (bwdLocalTile s KTile BLOCK_M BLOCK_DMODEL))

noncomputable def bwdScoreDotTile
    (s : BlockState) (DOTile VTile : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  Tile.dot [] (bwdLocalTile s DOTile BLOCK_M BLOCK_DMODEL)
    (Tile.transpose [] (bwdLocalTile s VTile BLOCK_M BLOCK_DMODEL))

noncomputable def bwdScorePTile
    (s : BlockState) (QTile KTile MVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  { data := fun idx =>
      WithBot.realExp
        (Option.map (fun scaled => scaled - s.readMem MVec idx.1.val)
          (Option.map (fun qk => qk * sm_scale)
            (if idx.1.val >= idx.2.1.val then
              (bwdScoreQKTile s QTile KTile BLOCK_M BLOCK_DMODEL).data idx
            else none))) }

noncomputable def bwdScoreDPTile
    (s : BlockState) (DOTile VTile DeltaVec : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) :
  Tile .real [BLOCK_M, BLOCK_M] :=
  { data := fun idx =>
      Option.map (fun dot => -s.readMem DeltaVec idx.1.val + dot)
        ((bwdScoreDotTile s DOTile VTile BLOCK_M BLOCK_DMODEL).data idx) }

noncomputable def bwdScorePFormulaSpec
    (s : BlockState) (QTile KTile MVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_M]) : ℝ :=
  WithBot.unbotD 0
    ((bwdScorePTile s QTile KTile MVec sm_scale BLOCK_M BLOCK_DMODEL).data idx)

noncomputable def bwdScoreDSFormulaSpec
    (s : BlockState) (QTile KTile VTile DOTile MVec DeltaVec : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_M]) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun acc => acc * sm_scale)
      (Option.map₂ (fun p dp => p * dp)
        ((bwdScorePTile s QTile KTile MVec sm_scale BLOCK_M BLOCK_DMODEL).data idx)
        ((bwdScoreDPTile s DOTile VTile DeltaVec BLOCK_M BLOCK_DMODEL).data idx)))

theorem triton_attention_bwd_score_formula_slice_toAlgorithm_supported
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) :
    ∃ alg, (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
      MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL).toAlgorithm? =
        Except.ok alg := by
  simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

private theorem foldl_writeMem_other_region_preserves {α : Type}
    {readRegion writeRegion : RegionName} (offsetFn : α → Nat)
    (valueFn : α → ℝ) (o : Nat) (l : List α)
    (hRegions : readRegion ≠ writeRegion) (s : BlockState) :
    ((l.foldl
      (fun acc k => acc.writeMem writeRegion (offsetFn k) (valueFn k))
      s).readMem readRegion o) = s.readMem readRegion o := by
  induction l generalizing s with
  | nil =>
      rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      rw [BlockState.writeMem_readMem]
      rw [if_neg (by
        intro h
        exact hRegions h.1)]

theorem triton_attention_bwd_score_formula_slice_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    (∀ idx : TileIndex [BLOCK_M, BLOCK_M],
      let outAddr := bwdScoreOffset BLOCK_M idx
      (exec (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
            MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem PTile outAddr)
        = some (bwdScorePFormulaSpec s QTile KTile MVec sm_scale BLOCK_M
            BLOCK_DMODEL idx)) ∧
    (∀ idx : TileIndex [BLOCK_M, BLOCK_M],
      let outAddr := bwdScoreOffset BLOCK_M idx
      (exec (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
            MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem DSTile outAddr)
        = some (bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec
            DeltaVec sm_scale BLOCK_M BLOCK_DMODEL idx)) := by
  constructor
  · intro idx
    simp [exec, triton_attention_bwd_score_formula_slice,
          ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
          Tile.transpose, NumericDType.add, NumericDType.sub, NumericDType.mul,
          ComparableDType.ge, bwdScoreOffset, bwdLocalTileOffset,
          bwdLocalTile, bwdScoreQKTile, bwdScoreDotTile, bwdScorePTile,
          bwdScoreDPTile, bwdScorePFormulaSpec, TileShape.dropInsertedIndex]
    let offsetFn : TileIndex [BLOCK_M, BLOCK_M] → Nat :=
      fun i => i.1.val * BLOCK_M + i.2.1.val
    have hInj : Function.Injective offsetFn := by
      simpa [offsetFn, bwdScoreOffset] using hOffsetInj
    rw [foldl_writeMem_other_region_preserves (readRegion := PTile)
      (writeRegion := DSTile) offsetFn _ (offsetFn idx)
      (TileShape.allIndices [BLOCK_M, BLOCK_M]) hRegions]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
    rfl
  · intro idx
    simp [exec, triton_attention_bwd_score_formula_slice,
          ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
          Tile.transpose, NumericDType.add, NumericDType.sub, NumericDType.mul,
          ComparableDType.ge, bwdScoreOffset, bwdLocalTileOffset,
          bwdLocalTile, bwdScoreQKTile, bwdScoreDotTile, bwdScorePTile,
          bwdScoreDPTile, bwdScoreDSFormulaSpec, TileShape.dropInsertedIndex]
    let offsetFn : TileIndex [BLOCK_M, BLOCK_M] → Nat :=
      fun i => i.1.val * BLOCK_M + i.2.1.val
    have hInj : Function.Injective offsetFn := by
      simpa [offsetFn, bwdScoreOffset] using hOffsetInj
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
    rfl

theorem triton_attention_bwd_score_p_formula_slice_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (PTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScorePFormulaSpec s QTile KTile MVec sm_scale BLOCK_M
          BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := (triton_attention_bwd_score_formula_slice_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL
    s hOffsetInj hRegions).1 idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_score_ds_formula_slice_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (sm_scale : ℝ) (BLOCK_M BLOCK_DMODEL : Nat) (s : BlockState)
    (hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_M] => bwdScoreOffset BLOCK_M idx))
    (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        some (DSTile, bwdScoreOffset BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_M] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          sm_scale BLOCK_M BLOCK_DMODEL idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_score_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := (triton_attention_bwd_score_formula_slice_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile sm_scale BLOCK_M BLOCK_DMODEL
    s hOffsetInj hRegions).2 idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dq_dot_step_slice_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem DQ outAddr)
        = some (bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh
            stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, bwdGradStoreSpec,
        bwdDsOffset, bwdKTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem DQPrev (offsetFn idx) +
        ∑ k : Fin BLOCK_M,
          FloatDType.fp16.storeValue
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem DS
                ((s.pids 1 * BLOCK_M + idx.1.val) * BLOCK_M + k.val)))) *
            s.readMem KTile (k.val * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
      stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdDqDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH,
    bwdRowIndex, bwdColIndex, bwdDsOffset, bwdKTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_dq_dot_step_slice_compute_correct
    (DQPrev DS KTile DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_dot_step_slice DQPrev DS KTile DQ H
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdDqDotStepSpec s DQPrev DS KTile H stride_qz stride_qh stride_qm
          stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_dot_step_slice_correct DQPrev DS KTile DQ
    H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Shared inner-loop transpose-dot update slice for `_bwd_kernel`'s DK/DV
accumulators. It covers both Python paths
`dv += tl.dot(tl.trans(p.to(tl.float16)), do)` and
`dk += tl.dot(tl.trans(ds.to(tl.float16)), q)` by parameterizing the left and
right tiles and the query-row block participating in this loop step. -/
def triton_attention_bwd_trans_dot_step_slice
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  off_hz = tl.program_id(0)
  block = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_out = block * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_query = $(queryBlock) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(AccPrev + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk))
  left = (tl.load(LeftTile +
      offs_query[:, None] * $(BLOCK_M) + offs_out[None, :])).to(tl.float16)
  right = tl.load(RightTile +
      offs_query[:, None] * $(BLOCK_DMODEL) + offs_k[None, :])
  acc += tl.dot(tl.trans(left), right)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_out[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk), acc)
}

def bwdQueryIndex (queryBlock BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  queryBlock * BLOCK_M + i.val

def bwdLeftTileOffset (s : BlockState) (queryBlock BLOCK_M : Nat)
    (query : Fin BLOCK_M) (outRow : Fin BLOCK_M) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_M +
    bwdRowIndex s BLOCK_M outRow

def bwdRightTileOffset (queryBlock BLOCK_M BLOCK_DMODEL : Nat)
    (query : Fin BLOCK_M) (col : Fin BLOCK_DMODEL) : Nat :=
  bwdQueryIndex queryBlock BLOCK_M query * BLOCK_DMODEL + col.val

noncomputable def bwdTransDotStepSpec
    (s : BlockState) (AccPrev LeftTile RightTile : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  bwdGradStoreSpec s AccPrev H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx +
    ∑ query : Fin BLOCK_M,
      s.readMem LeftTile
          (bwdLeftTileOffset s queryBlock BLOCK_M query idx.1) *
        s.readMem RightTile
          (bwdRightTileOffset queryBlock BLOCK_M BLOCK_DMODEL query idx.2.1)

theorem triton_attention_bwd_trans_dot_step_slice_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
            RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H
            stride_qz stride_qh stride_qm stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_trans_dot_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, Tile.transpose, NumericDType.add,
        NumericDType.mul, IntegralDType.floorDiv, IntegralDType.mod,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradStoreSpec, bwdQueryIndex, bwdLeftTileOffset,
        bwdRightTileOffset, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      s.readMem AccPrev (offsetFn idx) +
        ∑ query : Fin BLOCK_M,
          s.readMem LeftTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_M +
                (s.pids 1 * BLOCK_M + idx.1.val)) *
            s.readMem RightTile
              ((queryBlock * BLOCK_M + query.val) * BLOCK_DMODEL + idx.2.1.val)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem Out (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
      stride_qh stride_qm stride_qk BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdTransDotStepSpec, bwdGradStoreSpec, bwdGradOffset, bwdOffZ,
    bwdOffH, bwdRowIndex, bwdColIndex, bwdQueryIndex, bwdLeftTileOffset,
    bwdRightTileOffset, offsetFn, valueFn]

theorem triton_attention_bwd_trans_dot_step_slice_compute_correct
    (AccPrev LeftTile RightTile Out : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice AccPrev LeftTile
        RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s AccPrev LeftTile RightTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_trans_dot_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_trans_dot_step_slice_correct AccPrev LeftTile
    RightTile Out queryBlock H stride_qz stride_qh stride_qm stride_qk
    BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dv_dot_step_slice_compute_correct
    (DVPrev PTile DOTile DV : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DVPrev PTile DOTile
        DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DVPrev PTile DOTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DVPrev PTile
    DOTile DV queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dk_dot_step_slice_compute_correct
    (DKPrev DSTile QTile DK : RegionName)
    (queryBlock H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_trans_dot_step_slice DKPrev DSTile QTile
        DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdTransDotStepSpec s DKPrev DSTile QTile queryBlock H stride_qz
          stride_qh stride_qm stride_qk BLOCK_M idx) := by
  exact triton_attention_bwd_trans_dot_step_slice_compute_correct DKPrev DSTile
    QTile DK queryBlock H stride_qz stride_qh stride_qm stride_qk BLOCK_M
    BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dq_store_slice_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem DQ outAddr)
        = some (bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm
            stride_qk BLOCK_M idx) := by
  intro idx
  simp [exec, triton_attention_bwd_dq_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, bwdOffZ, bwdOffH,
        bwdRowIndex, bwdColIndex, bwdGradOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem DQPre (offsetFn idx)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem DQ (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem DQ
        (offsetFn idx) =
    bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
      BLOCK_M idx
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
    bwdColIndex, offsetFn, valueFn]

theorem triton_attention_bwd_dq_store_slice_compute_correct
    (DQPre DQ : RegionName)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ H stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        some (DQ, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DQPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dq_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := triton_attention_bwd_dq_store_slice_correct DQPre DQ H stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem triton_attention_bwd_dkdv_store_slice_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr :=
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx
      (exec (triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
            stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if bwdGradActive s D0 BLOCK_M idx then
            bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm
              stride_qk BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, triton_attention_bwd_dkdv_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex, bwdGradOffset,
        bwdGradActive, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 0 / H * stride_qz + s.pids 0 % H * stride_qh +
        (s.pids 1 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx => s.readMem GradPre (offsetFn idx)
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => s.pids 1 * BLOCK_M + idx.1.val < D0
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK_M + idx.1.val < D0
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]
  · simp [P, bwdGradStoreSpec, bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex,
      bwdColIndex, offsetFn, valueFn, hActive]

theorem triton_attention_bwd_dkdv_store_slice_compute_correct
    (GradPre Out : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice GradPre Out H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s GradPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_dkdv_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := triton_attention_bwd_dkdv_store_slice_correct GradPre Out H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem triton_attention_bwd_dk_store_slice_compute_correct
    (DKPre DK : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DK, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DKPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DKPre DK H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

theorem triton_attention_bwd_dv_store_slice_compute_correct
    (DVPre DV : RegionName)
    (H D0 stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV H D0 stride_qz
        stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          bwdGradActive s D0 BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (DV, bwdGradOffset s H stride_qz stride_qh stride_qm stride_qk
            BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bwdGradStoreSpec s DVPre H stride_qz stride_qh stride_qm stride_qk
          BLOCK_M idx) := by
  exact triton_attention_bwd_dkdv_store_slice_compute_correct DVPre DV H D0
    stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj

/-! ## Python test-shape wrappers

The checked Python test uses `q/k/v/o` with shape `(2, 4, 128, 64)`, so
the contiguous tensor strides are `(32768, 8192, 64, 1)`. The forward and
backward launchers use `BLOCK_M = BLOCK_N = 128`, `BLOCK_DMODEL = 64`,
`H = 4`, and `D0 = batch * heads * seq_len = 1024`. -/

theorem triton_attention_python_output_offset_injective
    (s : BlockState) (hzRowOffset : Nat) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        outOffset s hzRowOffset 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, rowIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_row_offset_injective
    (s : BlockState) (off_hz : Nat) :
    Function.Injective
      (fun i : Fin 128 => lRowOffset s off_hz 128 128 i) := by
  intro a b h
  simp [lRowOffset] at h
  exact Fin.ext (by omega)

theorem triton_attention_python_newdo_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] => newdoOffset s 128 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [newdoOffset, newdoMIndex, newdoNIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_bwd_grad_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [128, 64] =>
        bwdGradOffset s 4 32768 8192 64 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [bwdGradOffset, bwdOffZ, bwdOffH, bwdRowIndex, bwdColIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem triton_attention_python_bwd_score_offset_injective :
    Function.Injective
      (fun idx : TileIndex [128, 128] => bwdScoreOffset 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨na, hna⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨nb, hnb⟩, _⟩ h
  simp [bwdScoreOffset] at h
  have hm : ma = mb := by omega
  have hn : na = nb := by omega
  subst mb
  subst nb
  rfl

theorem triton_attention_forward_output_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (hzRowOffset : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx)))) := by
  exact triton_attention_forward_output_store_slice_compute_correct Acc Out
    hzRowOffset 1024 64 1 128 64 s
    (triton_attention_python_output_offset_injective s hzRowOffset)

theorem triton_attention_forward_l_store_python_test_shape_compute_correct
    (LPrev L : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i) := by
  exact triton_attention_forward_l_store_slice_compute_correct LPrev L
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_forward_m_store_python_test_shape_compute_correct
    (MPrev M : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i) := by
  exact triton_attention_forward_m_store_slice_compute_correct MPrev M
    off_hz 128 128 s (triton_attention_python_row_offset_injective s off_hz)

theorem triton_attention_bwd_preprocess_newdo_store_python_test_shape_compute_correct
    (NewDOAcc NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_store_slice
        NewDOAcc NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoStoreSpec s NewDOAcc 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_store_slice_compute_correct
    NewDOAcc NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_newdo_formula_python_test_shape_compute_correct
    (DO L NewDO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_newdo_formula_slice
        DO L NewDO 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        newdoFormulaSpec s DO L 128 64 idx) := by
  exact triton_attention_bwd_preprocess_newdo_formula_slice_compute_correct
    DO L NewDO 128 64 s (triton_attention_python_newdo_offset_injective s)

theorem triton_attention_bwd_preprocess_delta_formula_python_test_shape_compute_correct
    (Out DO L Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_formula_slice
        Out DO L Delta 128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaFormulaSpec s Out DO L 128 64 i) := by
  exact triton_attention_bwd_preprocess_delta_formula_slice_compute_correct
    Out DO L Delta 128 64 s

theorem triton_attention_bwd_preprocess_delta_store_python_test_shape_compute_correct
    (DeltaAcc Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess_delta_store_slice
        DeltaAcc Delta 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 => deltaStoreSpec s DeltaAcc 128 i) := by
  exact triton_attention_bwd_preprocess_delta_store_slice_compute_correct
    DeltaAcc Delta 128 s

theorem triton_attention_bwd_dq_store_python_test_shape_compute_correct
    (DQPre DQ : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dq_store_slice DQPre DQ 4
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DQPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dq_store_slice_compute_correct DQPre DQ 4
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dk_store_python_test_shape_compute_correct
    (DKPre DK : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DKPre DK 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DKPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dk_store_slice_compute_correct DKPre DK 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

theorem triton_attention_bwd_dv_store_python_test_shape_compute_correct
    (DVPre DV : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_dkdv_store_slice DVPre DV 4 1024
        32768 8192 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        bwdGradStoreSpec s DVPre 4 32768 8192 64 1 128 idx) := by
  exact triton_attention_bwd_dv_store_slice_compute_correct DVPre DV 4 1024
    32768 8192 64 1 128 64 s
    (triton_attention_python_bwd_grad_offset_injective s)

noncomputable def producedBwdKernelDQValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DQ (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

noncomputable def producedBwdKernelDKValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DK (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

noncomputable def producedBwdKernelDVValue
    (s : BlockState) (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128) s with
  | some s' => s'.readMem DV (bwdGradOffset s 4 32768 8192 64 1 128 idx)
  | none => 0.0

theorem triton_attention_bwd_kernel_dq_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [producedBwdKernelDQValue, hExec]

theorem triton_attention_bwd_kernel_dk_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBwdKernelDKValue, hExec]

theorem triton_attention_bwd_kernel_dv_python_test_shape_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_bwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBwdKernelDVValue, hExec]

theorem triton_attention_bwd_score_p_formula_python_test_shape_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx) := by
  exact triton_attention_bwd_score_p_formula_slice_compute_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
    128 64 s triton_attention_python_bwd_score_offset_injective hRegions

theorem triton_attention_bwd_score_ds_formula_python_test_shape_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx) := by
  exact triton_attention_bwd_score_ds_formula_slice_compute_correct QTile KTile
    VTile DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
    128 64 s triton_attention_python_bwd_score_offset_injective hRegions

noncomputable def producedTritonAttentionForwardOutValue
    (s : BlockState) (Q K V L M Out : RegionName)
    (hzRowOffset : Nat) (idx : TileIndex [128, 64]) : ℝ :=
  match exec (triton_attention_fwd_kernel Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128) s with
  | some s' => s'.readMem Out (outOffset s hzRowOffset 64 1 128 idx)
  | none => 0.0

noncomputable def producedTritonAttentionForwardLValue
    (s : BlockState) (Q K V L M Out : RegionName)
    (off_hz : Nat) (i : Fin 128) : ℝ :=
  match exec (triton_attention_fwd_kernel Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128) s with
  | some s' => s'.readMem L (lRowOffset s off_hz 128 128 i)
  | none => 0.0

noncomputable def producedTritonAttentionForwardMValue
    (s : BlockState) (Q K V L M Out : RegionName)
    (off_hz : Nat) (i : Fin 128) : ℝ :=
  match exec (triton_attention_fwd_kernel Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128) s with
  | some s' => s'.readMem M (lRowOffset s off_hz 128 128 i)
  | none => 0.0

theorem triton_attention_forward_surface_out_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (hzRowOffset : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedTritonAttentionForwardOutValue s Q K V L M Out hzRowOffset idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedTritonAttentionForwardOutValue, hExec]

theorem triton_attention_forward_surface_l_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 =>
        producedTritonAttentionForwardLValue s Q K V L M Out off_hz i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedTritonAttentionForwardLValue, hExec]

theorem triton_attention_forward_surface_m_python_test_shape_compute_correct
    (Q K V L M Out : RegionName) (off_hz : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 =>
        producedTritonAttentionForwardMValue s Q K V L M Out off_hz i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [triton_attention_fwd_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [producedTritonAttentionForwardMValue, hExec]

/-- Python forward shape summary: final output plus the row-wise `L` and `M`
side stores are compute-correct for the tested block shape. -/
theorem triton_attention_forward_python_test_shape_all_outputs_compute_correct
    (Acc LPrev MPrev Out L M : RegionName) (hzRowOffset off_hz : Nat)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_output_store_slice Acc Out
        hzRowOffset 1024 64 1 128 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (storeValue s Acc hzRowOffset 1024 128 64 idx))))) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_l_store_slice LPrev L off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => lStoreSpec s LPrev off_hz 128 128 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_forward_m_store_slice MPrev M off_hz 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 => mStoreSpec s MPrev off_hz 128 128 i)) := by
  constructor
  · exact triton_attention_forward_output_store_python_test_shape_compute_correct
      Acc Out hzRowOffset s
  constructor
  · exact triton_attention_forward_l_store_python_test_shape_compute_correct
      LPrev L off_hz s
  · exact triton_attention_forward_m_store_python_test_shape_compute_correct
      MPrev M off_hz s

/-- Python backward-preprocess shape summary: the full `_bwd_preprocess`
surface realizes both observable outputs for the tested block shape. -/
theorem triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
    (Out DO L NewDO Delta : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta 128 64 i)) := by
  constructor
  · exact triton_attention_bwd_preprocess_newdo_surface_compute_correct
      Out DO L NewDO Delta 128 64 s
  · exact triton_attention_bwd_preprocess_delta_surface_compute_correct
      Out DO L NewDO Delta 128 64 s

/-- Python backward gradient shape summary: the full `_bwd_kernel` realizes the
final `DQ`, `DK`, and `DV` outputs for the tested block shape. -/
theorem triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
    (Q K V Out DO DQ DK DV L M Delta : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx)) := by
  constructor
  · exact triton_attention_bwd_kernel_dq_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s
  constructor
  · exact triton_attention_bwd_kernel_dk_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s
  · exact triton_attention_bwd_kernel_dv_python_test_shape_compute_correct
      Q K V Out DO DQ DK DV L M Delta s

/-- Python backward score arithmetic shape summary: the `P` probability tile
and `DS` score-gradient tile are compute-correct for the tested block shape. -/
theorem triton_attention_bwd_score_python_test_shape_all_outputs_compute_correct
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) := by
  constructor
  · exact triton_attention_bwd_score_p_formula_python_test_shape_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions
  · exact triton_attention_bwd_score_ds_formula_python_test_shape_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions

/-- Python-shape arithmetic surface for the main `_bwd_kernel` score step.

This pins the checked launch's one-block backward inner step at
`BLOCK_M = BLOCK_N = 128`, `BLOCK_DMODEL = 64`, and `sm_scale = 1 / sqrt(64)`.
It exposes compute-correct `P` and `DS` tiles that feed the checked DQ/DK/DV
dot-step proofs, instead of treating those score-side inputs as opaque
precomputed regions. -/
theorem triton_attention_bwd_score_python_test_shape_formula_summary
    (QTile KTile VTile DOTile MVec DeltaVec PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    (∃ alg, (triton_attention_bwd_score_formula_slice QTile KTile VTile DOTile
      MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹)
      128 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s QTile KTile MVec ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice QTile KTile VTile
        DOTile MVec DeltaVec PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s QTile KTile VTile DOTile MVec DeltaVec
          ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) := by
  constructor
  · exact triton_attention_bwd_score_formula_slice_toAlgorithm_supported
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile
      ((Real.sqrt (64 : ℝ))⁻¹) 128 64
  · exact triton_attention_bwd_score_python_test_shape_all_outputs_compute_correct
      QTile KTile VTile DOTile MVec DeltaVec PTile DSTile s hRegions























/-- Public Python forward summary for `triton_attention.py`.

The surface conjunct pins the faithful `_fwd_kernel` launch for the checked
shape `(B, H, T, D) = (2, 4, 128, 64)`, contiguous Q/K/V/O strides
`(32768, 8192, 64, 1)`, `BLOCK_M = BLOCK_N = 128`, and
`BLOCK_DMODEL = 64`. The output conjuncts read back the Python-observable
`Out`, `L`, and `M` stores from that full surface. -/
theorem triton_attention_forward_python_test_shape_output_summary
    (Q K V L M Out : RegionName)
    (hzRowOffset off_hz : Nat) (s : BlockState) :
    (∃ alg, (triton_attention_fwd_kernel Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => active s hzRowOffset 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (Out, outOffset s hzRowOffset 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedTritonAttentionForwardOutValue s Q K V L M Out hzRowOffset idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 =>
        producedTritonAttentionForwardLValue s Q K V L M Out off_hz i)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_fwd_kernel Q K V L M Out
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 128 64 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (M, lRowOffset s off_hz 128 128 i))
      (expected := fun i : Fin 128 =>
        producedTritonAttentionForwardMValue s Q K V L M Out off_hz i)) := by
  constructor
  · exact triton_attention_fwd_kernel_toAlgorithm_supported Q K V L M Out
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 128 64 128
  constructor
  · exact triton_attention_forward_surface_out_python_test_shape_compute_correct
      Q K V L M Out hzRowOffset s
  constructor
  · exact triton_attention_forward_surface_l_python_test_shape_compute_correct
      Q K V L M Out off_hz s
  · exact triton_attention_forward_surface_m_python_test_shape_compute_correct
      Q K V L M Out off_hz s

/-- Public Python backward-preprocess summary for `triton_attention.py`.

This records the faithful `_bwd_preprocess` full surface at `BLOCK_M = 128`
and `D_HEAD = 64`, and connects both Python-observable `NewDO` and `Delta`
outputs directly to the produced full-surface values. -/
theorem triton_attention_bwd_preprocess_python_test_shape_output_summary
    (Out DO L NewDO Delta : RegionName) (s : BlockState) :
    (∃ alg, (triton_attention_bwd_preprocess Out DO L NewDO Delta
      128 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (NewDO, newdoOffset s 128 64 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdPreprocessNewDOValue s Out DO L NewDO Delta 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_preprocess Out DO L NewDO Delta
        128 64)
      (initialState := s)
      (write := fun i : Fin 128 => some (Delta, deltaOffset s 128 i))
      (expected := fun i : Fin 128 =>
        producedBwdPreprocessDeltaValue s Out DO L NewDO Delta 128 64 i)) := by
  constructor
  · exact triton_attention_bwd_preprocess_toAlgorithm_supported
      Out DO L NewDO Delta 128 64
  · exact triton_attention_bwd_preprocess_python_test_shape_all_outputs_compute_correct
      Out DO L NewDO Delta s

/-- Public Python backward-gradient summary for `triton_attention.py`.

The surface conjunct pins the checked Python launch of the main `_bwd_kernel`
for `(B, H, T, D) = (2, 4, 128, 64)` with `num_block = 1`; the output
conjuncts connect the Python-observable `DQ`, `DK`, and `DV` writes directly to
the produced full-kernel values. -/
theorem triton_attention_bwd_grads_python_test_shape_output_summary
    (Q K V Out DO DQ DK DV L M Delta : RegionName)
    (s : BlockState) :
    (∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx)) := by
  constructor
  · exact triton_attention_bwd_kernel_toAlgorithm_supported Q K V Out DO DQ DK
      DV L M Delta ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128
  · exact triton_attention_bwd_grads_python_test_shape_all_outputs_compute_correct
      Q K V Out DO DQ DK DV L M Delta s

/-- Combined checked-shape backward summary for `triton_attention.py`.

This exposes the main `_bwd_kernel` surface, final `DQ`/`DK`/`DV` writebacks,
and the score-side `P`/`DS` arithmetic producer in one public target. -/
theorem triton_attention_bwd_python_test_shape_complete_summary
    (Q K V Out DO DQ DK DV L M Delta PTile DSTile : RegionName)
    (s : BlockState) (hRegions : PTile ≠ DSTile) :
    ((∃ alg, (triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
      ((Real.sqrt (64 : ℝ))⁻¹)
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 1024 1 128 64 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (DQ, bwdGradOffset s 4 32768 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDQValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DK, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDKValue s Q K V Out DO DQ DK DV L M Delta idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_kernel Q K V Out DO DQ DK DV L M Delta
        ((Real.sqrt (64 : ℝ))⁻¹)
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 1024 1 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 64] => bwdGradActive s 1024 128 idx)
        (fun idx : TileIndex [128, 64] =>
          (DV, bwdGradOffset s 4 32768 8192 64 1 128 idx)))
      (expected := fun idx : TileIndex [128, 64] =>
        producedBwdKernelDVValue s Q K V Out DO DQ DK DV L M Delta idx))) ∧
    ((∃ alg, (triton_attention_bwd_score_formula_slice Q K V DO M Delta
      PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice Q K V DO M Delta
        PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (PTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScorePFormulaSpec s Q K M ((Real.sqrt (64 : ℝ))⁻¹) 128 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := triton_attention_bwd_score_formula_slice Q K V DO M Delta
        PTile DSTile ((Real.sqrt (64 : ℝ))⁻¹) 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (DSTile, bwdScoreOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bwdScoreDSFormulaSpec s Q K V DO M Delta ((Real.sqrt (64 : ℝ))⁻¹)
          128 64 idx))) := by
  constructor
  · exact triton_attention_bwd_grads_python_test_shape_output_summary Q K V Out
      DO DQ DK DV L M Delta s
  · exact triton_attention_bwd_score_python_test_shape_formula_summary Q K V
      DO M Delta PTile DSTile s hRegions

end VeriTile.Bench.TritonBenchG.TritonAttention

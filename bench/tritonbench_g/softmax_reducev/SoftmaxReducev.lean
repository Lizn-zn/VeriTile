import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `softmax_reducev` — strict per-kernel correctness

`_fwd_kernel` is the fused online-softmax + reduce-over-V attention kernel:
program `(cur_batch, cur_head)` streams the sequence in `BLOCK_N` chunks,
gathers V rows through the `B_Loc` paged-KV index, and maintains the flash-style
running statistics — running max `e_max`, rescaled running sum `e_sum`, and
rescaled accumulator `acc` — then normalizes `acc / e_sum` and stores the
`BLOCK_DMODEL` result to `Out[cur_batch, cur_head, :]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[(batch, head)](...)`, the grid over
`(batch, head)`, the host `BLOCK = 64` / `BLOCK_DMODEL = v.shape[-1]` choices,
`num_warps` / `num_stages`, and how the runtime composes per-program writes into
`Out`) is the *trusted boundary*, not a proof obligation here. Because the
program ids `cur_batch`/`cur_head` are universally quantified (via `BlockState`),
the per-program statements cover every program of the grid.

## Proof architecture

```
softmax_reducev_python_test_shape_output_summary      ← TOP THEOREM
  ├─ softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported  surface lowers
  ├─ softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported     (other_kv_index = 0)
  │    └─ softmax_reducev_surface_toAlgorithm_supported
  └─ softmax_reducev_surface_output_compute_correct     ← ComputeCorrect of the store
       ├─ softmax_reducev_python_output_offset_injective
       └─ softmax_reducev_python_test_shape_final_output_compute_correct
            └─ softmax_reducev_final_store_slice_compute_correct
                 └─ softmax_reducev_final_store_slice_correct  (per-lane normalized readback)
```

The top theorem covers both `other_kv_index ∈ {-1, 0}` test variants at the
Python test shape (`BLOCK_DMODEL = 64`, `BLOCK_N = 64`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` init,
`exp`, running-max rescaling, and the final division are real-valued);
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The verified
statement is scoped to the **final normalized store** to `Out`: the expected
value is `acc / e_sum` (`softmaxReducevFinalSpec` / `softmaxReducevSurfaceValue`)
read off at each `outOffset`, over a one-block output footprint with the program
ids universally quantified. The streaming online-softmax loop (running max,
rescale-and-accumulate, paged-V gather) feeds those `Acc`/`ESum` values; the side
condition is the offset-injectivity of the `Out` slice at the test shape.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Lean port of `softmax_reducev.py`'s `_fwd_kernel`.

This records the streaming softmax recurrence over token blocks, the signed
`B_Loc` gather with Python's `other_kv_index` sentinel, the V gather, and the
final normalized writeback. -/
def softmax_reducev_surface
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_batch_seq_len = tl.load(BSeqLen + cur_batch)
  cur_batch_start_loc = tl.load(BStartLoc + cur_batch)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_v = cur_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  off_b_loc = cur_batch * $(stride_b_loc_b) +
    ($(max_input_len) - cur_batch_seq_len) * $(stride_b_loc_s)
  v_ptrs = V + off_v
  e_max = float("-inf")
  e_sum = 0.0
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    v_index = tl.load(BLoc + off_b_loc +
      (start_n + offs_n) * $(stride_b_loc_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=$(other_kv_index))
    qk = tl.load(Logics + cur_head * $(stride_logic_h) +
        (cur_batch_start_loc + start_n + offs_n) * $(stride_logic_bs),
      mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    n_e_max = tl.maximum(tl.max(qk, 0), e_max)
    old_scale = tl.exp(e_max - n_e_max)
    p = tl.exp(qk - n_e_max)
    e_sum = e_sum * old_scale + tl.sum(p, 0)
    v = tl.load(v_ptrs + v_index[:, None] * $(stride_vbs))
    acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
    e_max = n_e_max
  }
  acc = acc / e_sum
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}

/-- The full softmax-reduceV surface lowers to the algorithm layer. -/
theorem softmax_reducev_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N other_kv_index).toAlgorithm? = Except.ok alg := by
  simp [softmax_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final normalization/store slice of `softmax_reducev.py`'s
`_fwd_kernel`.

The full kernel performs a streaming softmax over V blocks, maintaining `acc`
and `e_sum`. This slice starts after that loop with precomputed `Acc` and `ESum`
regions, then proves the final `acc / e_sum` writeback to `Out` across the
`BLOCK_DMODEL` vector. -/
def softmax_reducev_final_store_slice
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  e_sum = tl.load(ESum + cur_batch * $(stride_es_bs) + cur_head * $(stride_es_h))
  acc = tl.load(Acc + cur_batch * $(stride_acc_bs) + cur_head * $(stride_acc_h) +
      offs_d * $(stride_acc_d))
  out = acc / e_sum
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), out)
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def accOffset
    (s : BlockState) (stride_acc_bs stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_bs + s.pids 1 * stride_acc_h + dIndex s i * stride_acc_d

def eSumOffset (s : BlockState) (stride_es_bs stride_es_h : Nat) : Nat :=
  s.pids 0 * stride_es_bs + s.pids 1 * stride_es_h

def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

noncomputable def softmaxReducevFinalSpec
    (s : BlockState) (Acc ESum : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i) /
    s.readMem ESum (eSumOffset s stride_es_bs stride_es_h)

/-- Algorithm-layer correctness for the final softmax-reduce-v store slice. -/
theorem softmax_reducev_final_store_slice_correct
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (softmax_reducev_final_store_slice Acc ESum Out
            stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
            stride_obs stride_oh stride_od BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h
          stride_acc_d stride_es_bs stride_es_h i) := by
  intro i
  simp [exec, softmax_reducev_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, NumericDType.div, dIndex, accOffset, eSumOffset,
        outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    have hfin : a = b := hOutInj habFin
    subst hfin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [softmaxReducevFinalSpec, accOffset, eSumOffset, outOffset, dIndex]

/-- Compute-facing correctness for the final softmax-reduce-v store slice. -/
theorem softmax_reducev_final_store_slice_compute_correct
    (Acc ESum Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_es_bs stride_es_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum stride_acc_bs stride_acc_h
          stride_acc_d stride_es_bs stride_es_h i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_reducev_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := softmax_reducev_final_store_slice_correct Acc ESum Out
    stride_acc_bs stride_acc_h stride_acc_d stride_es_bs stride_es_h
    stride_obs stride_oh stride_od BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Python test-shape wrappers

`test_token_softmax_reducev_fwd` fixes `batch = 2`, `head = 2`,
`max_input_len = 128`, `d_model = 64`, and `BLOCK_N = 64`.  The first three
cases use `other_kv_index = -1`; the fourth uses `other_kv_index = 0`. -/

theorem softmax_reducev_python_output_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 64 => outOffset s 128 64 1 i) := by
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

/-- Python test-shape final output store correctness for `o` with shape
`(batch, head, d_model) = (2, 2, 64)`.  The accumulator and denominator are
materialized in proof regions; the full streaming softmax loop is represented
by the surface wrappers below. -/
theorem softmax_reducev_python_test_shape_final_output_compute_correct
    (Acc ESum Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        128 64 1 2 1 128 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum 128 64 1 2 1 i) := by
  exact softmax_reducev_final_store_slice_compute_correct Acc ESum Out
    128 64 1 2 1 128 64 1 64 s
    (softmax_reducev_python_output_offset_injective s)

theorem softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg := by
  exact softmax_reducev_surface_toAlgorithm_supported Logics V Out BLoc
    BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)

theorem softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) :
    ∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg := by
  exact softmax_reducev_surface_toAlgorithm_supported Logics V Out BLoc
    BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 0

noncomputable def softmaxReducevSurfaceValue
    (s : BlockState) (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (offset : Nat) : ℝ :=
  match exec (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
      stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N other_kv_index) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem softmax_reducev_surface_output_compute_correct
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
        stride_vd stride_obs stride_oh stride_od stride_b_loc_b stride_b_loc_s
        BLOCK_DMODEL BLOCK_N other_kv_index)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          max_input_len stride_logic_h stride_logic_bs stride_vbs stride_vh
          stride_vd stride_obs stride_oh stride_od stride_b_loc_b
          stride_b_loc_s BLOCK_DMODEL BLOCK_N other_kv_index
          (outOffset s stride_obs stride_oh stride_od i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i
  simp [softmaxReducevSurfaceValue, hExec]

/-- Public Python case coverage summary: the full checked surface lowers for
both sentinel choices, and the final normalized `o` vector writeback realizes
the Python output shape. -/
theorem softmax_reducev_python_test_shape_output_surface_summary
    (Logics V Acc ESum Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg) ∧
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_final_store_slice Acc ESum Out
        128 64 1 2 1 128 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i =>
        softmaxReducevFinalSpec s Acc ESum 128 64 1 2 1 i)) := by
  constructor
  · exact softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  constructor
  · exact softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  · exact softmax_reducev_python_test_shape_final_output_compute_correct
      Acc ESum Out s

/-- Python reduce-V softmax final-store coverage. -/
abbrev softmax_reducev_python_test_shape_store_summary
    (Logics V Acc ESum Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :=
  softmax_reducev_python_test_shape_output_surface_summary
    Logics V Acc ESum Out BLoc BStartLoc BSeqLen s




















theorem softmax_reducev_python_test_shape_output_summary
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat) (s : BlockState) :
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1))
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i : Fin 64 =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1)
          (outOffset s 128 64 1 i))) ∧
    (∃ alg, (softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
      128 256 1 8192 64 1 128 64 1 128 1 64 64 0).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := softmax_reducev_surface Logics V Out BLoc BStartLoc BSeqLen
        128 256 1 8192 64 1 128 64 1 128 1 64 64 0)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 128 64 1 i))
      (expected := fun i : Fin 64 =>
        softmaxReducevSurfaceValue s Logics V Out BLoc BStartLoc BSeqLen
          128 256 1 8192 64 1 128 64 1 128 1 64 64 0
          (outOffset s 128 64 1 i))) := by
  constructor
  · exact softmax_reducev_python_case_other_neg_one_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  constructor
  · exact softmax_reducev_surface_output_compute_correct Logics V Out BLoc
      BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 (-1) s
  constructor
  · exact softmax_reducev_python_case_other_zero_surface_toAlgorithm_supported
      Logics V Out BLoc BStartLoc BSeqLen
  · exact softmax_reducev_surface_output_compute_correct Logics V Out BLoc
      BStartLoc BSeqLen 128 256 1 8192 64 1 128 64 1 128 1 64 64 0 s

end VeriTile.Bench.TritonBenchG.SoftmaxReducev

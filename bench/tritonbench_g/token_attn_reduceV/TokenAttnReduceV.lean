import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_attn_reduceV.py`'s
`_fwd_kernel_token_att2`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
def token_attn_reducev_surface
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = 0
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}

/-- The full token-attention reduce-V surface lowers to the algorithm layer. -/
theorem token_attn_reducev_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen stride_req_to_tokens_b stride_req_to_tokens_s
      stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh
      stride_od kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [token_attn_reducev_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final output-store slice of `token_attn_reduceV.py`'s
`_fwd_kernel_token_att2`.

The full kernel streams over token blocks, gathers V through `Req_to_tokens`,
and accumulates `sum(prob * v)`. This slice starts from a precomputed `Acc`
vector and proves the final `BLOCK_DMODEL` writeback into `Out`. -/
def token_attn_reducev_final_store_slice
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_bs) + cur_head * $(stride_acc_h) +
      offs_d * $(stride_acc_d))
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), (acc).to(Out.dtype.element_ty))
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def accOffset
    (s : BlockState) (stride_acc_bs stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_bs + s.pids 1 * stride_acc_h + dIndex s i * stride_acc_d

def outOffset
    (s : BlockState) (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

/-- Algorithm-layer correctness for the token-attention reduce-V final store. -/
theorem token_attn_reducev_final_store_slice_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (token_attn_reducev_final_store_slice Acc Out stride_acc_bs
            stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  intro i
  simp [exec, token_attn_reducev_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, dIndex, accOffset, outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]

/-- Compute-facing correctness for the token-attention reduce-V final store. -/
theorem token_attn_reducev_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (stride_acc_bs stride_acc_h stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out stride_acc_bs
        stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (Out, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i =>
        s.readMem Acc (accOffset s stride_acc_bs stride_acc_h stride_acc_d i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_reducev_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := token_attn_reducev_final_store_slice_correct Acc Out stride_acc_bs
    stride_acc_h stride_acc_d stride_obs stride_oh stride_od BLOCK_DMODEL
    s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Python test-shape wrapper

The checked Python test uses `batch_size = 2`, `num_heads = 4`,
`seq_len = 128`, and `d_model = 64`. The output tensor has shape
`(2, 4, 64)` and contiguous strides `(256, 64, 1)`. -/

theorem token_attn_reducev_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 64 => outOffset s 256 64 1 i) := by
  intro a b h
  simp [outOffset, dIndex] at h
  exact Fin.ext (by omega)

theorem token_attn_reducev_final_store_python_test_shape_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i)) := by
  exact token_attn_reducev_final_store_slice_compute_correct Acc Out
    256 64 1 256 64 1 64 s
    (token_attn_reducev_python_test_shape_offset_injective s)

/-- Python case 1 full reduce-V surface lowering for `batch = 2`,
`seq_len = 128`, `num_heads = 4`, and `d_model = 64`. -/
theorem token_attn_reducev_python_case1_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
    256 64 1 1 64 128

/-- Python case 2 full reduce-V surface lowering for the `seq_len = 64`
variant. -/
theorem token_attn_reducev_python_case2_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1
    256 64 1 1 64 128

/-- Python case 3 full reduce-V surface lowering for the `batch = 3`,
`seq_len = 128` variant. -/
theorem token_attn_reducev_python_case3_surface_toAlgorithm_supported
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat) :
    ∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_reducev_surface_toAlgorithm_supported Prob V Out
    Req_to_tokens B_req_idx B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1
    256 64 1 1 64 128

/-- Public Python case 1 coverage summary: the full gather/reduceV surface
lowers and the final output vector store realizes the checked output shape. -/
theorem token_attn_reducev_python_case1_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case1_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 2 coverage summary. -/
theorem token_attn_reducev_python_case2_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 64 1 64 1 4096 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case2_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- Public Python case 3 coverage summary. -/
theorem token_attn_reducev_python_case3_output_surface_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :
    (∃ alg, (token_attn_reducev_surface Prob V Out Req_to_tokens B_req_idx
      B_Start_Loc B_Seqlen 128 1 128 1 8192 64 1 256 64 1 1 64 128).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_reducev_final_store_slice Acc Out
        256 64 1 256 64 1 64)
      (initialState := s)
      (write := fun i : Fin 64 => some (Out, outOffset s 256 64 1 i))
      (expected := fun i : Fin 64 =>
        s.readMem Acc (accOffset s 256 64 1 i))) := by
  constructor
  · exact token_attn_reducev_python_case3_surface_toAlgorithm_supported
      Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen
  · exact token_attn_reducev_final_store_python_test_shape_compute_correct
      Acc Out s

/-- `output_summary` alias for Python reduce-V token-attention case 1. -/
abbrev token_attn_reducev_python_case1_output_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case1_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s

/-- `output_summary` alias for Python reduce-V token-attention case 2. -/
abbrev token_attn_reducev_python_case2_output_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case2_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s

/-- `output_summary` alias for Python reduce-V token-attention case 3. -/
abbrev token_attn_reducev_python_case3_output_summary
    (Prob V Acc Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (s : BlockState) :=
  token_attn_reducev_python_case3_output_surface_summary
    Prob V Acc Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen s

end VeriTile.Bench.TritonBenchG.TokenAttnReduceV

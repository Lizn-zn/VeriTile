import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
  simp [exec, softmax_reducev_final_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.SoftmaxReducev

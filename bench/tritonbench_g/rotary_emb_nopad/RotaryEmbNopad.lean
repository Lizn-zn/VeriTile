import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Surface transcription of the Q part of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This writes both rotary halves for Q over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. The conditional K
branch is represented by `rotary_embedding_k_surface`; the cache-writing v2
branch is represented by `fused_rotary_embedding_v2_surface`. -/
def rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  dim_range1 = dim_range0 + $(HEAD_HALF)
  off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
    dim_range0[None, :] * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(Sin + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  off_q0 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range0[None, None, :] * $(head_dim_stride)
  off_q1 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range1[None, None, :] * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(Q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
  out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]
  tl.store(Q + off_q0, out_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
  tl.store(Q + off_q1, out_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens)))
}

/-- Surface transcription of the conditional K part of
`rotary_emb_nopad.py`'s `rotary_embedding_kernel` over the full
`[BLOCK_TOKENS, 1, HEAD_HALF]` token/head/dimension tile. -/
def rotary_embedding_k_surface
    (K Cos Sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens KV_GROUP_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  handle_kv = (cur_head_idx % $(KV_GROUP_NUM)) == 0
  if handle_kv {
    k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
    dim_range0 = tl.arange(0, $(HEAD_HALF))
    dim_range1 = dim_range0 + $(HEAD_HALF)
    off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
      dim_range0[None, :] * $(cos_stride)
    loaded_cos = tl.load(Cos + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    loaded_sin = tl.load(Sin + off_cos_sin,
      mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
    off_k0 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range0[None, None, :] * $(head_dim_stride)
    off_k1 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range1[None, None, :] * $(head_dim_stride)
    loaded_k0 = tl.load(K + off_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    loaded_k1 = tl.load(K + off_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens), other=0.0)
    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
    tl.store(K + off_k0, out_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
    tl.store(K + off_k1, out_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
  }
}

/-- Surface transcription of `rotary_emb_nopad.py`'s
`fused_rotary_embedding_kernel_v2`.

Python returns early when `block_head_index >= Q_HEAD_NUM`; this surface
represents that with a guarded body. The benchmark initializes
`context_lengths >= 1`, so the `past_kv_seq_len = context_lengths[...] - 1`
Nat subtraction follows the exercised path. -/
def fused_rotary_embedding_v2_surface
    (Q K Cos Sin KVCache : RegionName) (BlockTables ContextLengths : Region .nat)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride cacheb_stride cacheh_stride cachebs_stride
      cached_stride bts_stride btb_stride block_size q_total_tokens
      Q_HEAD_NUM HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  if block_head_index < $(Q_HEAD_NUM) {
    block_token_index = tl.program_id(1)
    dim = tl.arange(0, $(HEAD_HALF))
    dim1 = dim + $(HEAD_HALF)

    off_q0 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim * $(head_dim_stride)
    off_q1 = block_token_index * $(q_token_stride) +
      block_head_index * $(q_head_stride) + dim1 * $(head_dim_stride)
    off_k0 = block_token_index * $(k_token_stride) +
      block_head_index * $(k_head_stride) + dim * $(head_dim_stride)
    off_k1 = block_token_index * $(k_token_stride) +
      block_head_index * $(k_head_stride) + dim1 * $(head_dim_stride)

    loaded_q0 = tl.load(Q + off_q0)
    loaded_q1 = tl.load(Q + off_q1)
    loaded_k0 = tl.load(K + off_k0)
    loaded_k1 = tl.load(K + off_k1)

    off_cos_sin = block_token_index * $(cos_token_stride) + dim * $(cos_stride)
    loaded_cos = tl.load(Cos + off_cos_sin,
      mask=block_token_index < $(q_total_tokens), other=0.0)
    loaded_sin = tl.load(Sin + off_cos_sin,
      mask=block_token_index < $(q_total_tokens), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    past_kv_seq_len = tl.load($((ContextLengths : Region .nat)) + block_token_index) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_table_ptr = $((BlockTables : Region .nat)) + block_token_index * $(bts_stride)
    block_ids = tl.load(block_table_ptr + last_block_idx * $(btb_stride),
      mask=block_token_index < $(q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % $(block_size)) * $(cachebs_stride)

    kv_range0 = block_ids * $(cacheb_stride) +
      block_head_index * $(cacheh_stride) + offsets_in_last_block +
      dim * $(cached_stride)
    kv_range1 = block_ids * $(cacheb_stride) +
      block_head_index * $(cacheh_stride) + offsets_in_last_block +
      dim1 * $(cached_stride)

    tl.store(KVCache + kv_range0, out_k0)
    tl.store(KVCache + kv_range1, out_k1)
    tl.store(Q + off_q0, out_q0)
    tl.store(Q + off_q1, out_q1)
  }
}

/-- Proof-oriented Q first-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This captures one token block and one Q head for the first Q store:
`out_q0 = q0 * cos - q1 * sin`. -/
def rotary_embedding_q0_block
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)
  token = cur_token_block_idx * $(BLOCK_TOKENS)
  dim = tl.arange(0, $(HEAD_HALF))
  off_cos_sin = token * $(cos_token_stride) + dim * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=token < $(q_total_tokens), other=0.0)
  off_q0 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    dim * $(head_dim_stride)
  off_q1 = token * $(q_token_stride) + cur_head_idx * $(q_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_q1 = tl.load(q + off_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def tokenIndex (s : BlockState) (BLOCK_TOKENS : Nat) : Nat :=
  s.pids 1 * BLOCK_TOKENS

def headIndex (s : BlockState) : Nat :=
  s.pids 0

def dimIndex (i : Fin HEAD_HALF) : Nat :=
  i.val

def active (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat) :
    Prop :=
  headIndex s < Q_HEAD_NUM ∧ tokenIndex s BLOCK_TOKENS < q_total_tokens

instance activeDecidable
    (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat) :
    Decidable (active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS : Nat) (dim : Nat) : Nat :=
  tokenIndex s BLOCK_TOKENS * q_token_stride +
    headIndex s * q_head_stride + dim * head_dim_stride

def cosOffset
    (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * cos_token_stride + dimIndex i * cos_stride

noncomputable def rotaryNopadQ0Spec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) -
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

/-- Algorithm-layer correctness for the Q first-half nopad rotary store. -/
theorem rotary_embedding_q0_block_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i)))
    (hExec : exec (rotary_embedding_q0_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem q
          (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i)) =
        if active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadQ0Spec s q cos sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem q
            (qOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * q_token_stride +
          s.pids 0 * q_head_stride + idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_q0_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [qOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < Q_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [active, rotaryNopadQ0Spec, qOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [active, tokenIndex, headIndex, hHead, hTok]
    · simp [active, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q first-half nopad rotary store. -/
theorem rotary_embedding_q0_block_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_embedding_q0_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS)
        (fun i => (q,
          qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i))))
      (expected := fun i =>
        rotaryNopadQ0Spec s q cos sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_q0_block_correct q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.RotaryEmbNopad

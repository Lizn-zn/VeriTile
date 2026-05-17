import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes and the conditional K rotary path
in one kernel; the smaller Q/K kernels below remain available for local
correctness arguments. -/
def rotary_embedding_kernel_surface
    (q k cos sin : RegionName)
    (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride q_total_tokens Q_HEAD_NUM KV_GROUP_NUM
      HEAD_DIM BLOCK_TOKENS : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_block_idx = tl.program_id(1)

  tokens_range = cur_token_block_idx * $(BLOCK_TOKENS) + tl.arange(0, $(BLOCK_TOKENS))
  dim_range0 = tl.arange(0, $(HEAD_DIM) // $(2))
  dim_range1 = tl.arange($(HEAD_DIM) // $(2), $(HEAD_DIM))

  off_cos_sin = tokens_range[:, None] * $(cos_token_stride) +
    dim_range0[None, :] * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)
  loaded_sin = tl.load(sin + off_cos_sin,
    mask=tokens_range[:, None] < $(q_total_tokens), other=0.0)

  off_q0 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range0[None, None, :] * $(head_dim_stride)
  off_q1 = tokens_range[:, None, None] * $(q_token_stride) +
    cur_head_idx * $(q_head_stride) +
    dim_range1[None, None, :] * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))),
    other=0.0)
  loaded_q1 = tl.load(q + off_q1,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))),
    other=0.0)
  out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
  out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

  tl.store(q + off_q0, out_q0,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))))
  tl.store(q + off_q1, out_q1,
    mask=((cur_head_idx < $(Q_HEAD_NUM)) &
      (tokens_range[:, None, None] < $(q_total_tokens))))

  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    off_k0 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range0[None, None, :] * $(head_dim_stride)
    off_k1 = tokens_range[:, None, None] * $(k_token_stride) +
      k_head_idx * $(k_head_stride) +
      dim_range1[None, None, :] * $(head_dim_stride)
    loaded_k0 = tl.load(k + off_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens),
      other=0.0)
    loaded_k1 = tl.load(k + off_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens),
      other=0.0)
    out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
    out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
    tl.store(k + off_k0, out_k0,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
    tl.store(k + off_k1, out_k1,
      mask=tokens_range[:, None, None] < $(q_total_tokens))
  }
}

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

    past_kv_seq_len = tl.load(ContextLengths + block_token_index) - $(1)
    last_block_idx = past_kv_seq_len // $(block_size)
    block_table_ptr = BlockTables + block_token_index * $(bts_stride)
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

/-- Proof-oriented Q second-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`.

This captures one token block and one Q head for the second Q store:
`out_q1 = q0 * sin + q1 * cos` written to `q + off_q1`. -/
def rotary_embedding_q1_block
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
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(q + off_q1, out_q1,
    mask=(cur_head_idx < $(Q_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def qSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * q_token_stride +
    headIndex s * q_head_stride + (dimIndex i + HEAD_HALF) * head_dim_stride

noncomputable def rotaryNopadQ1Spec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) +
  s.readMem q
      (qOffset s q_token_stride q_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

/-- Algorithm-layer correctness for the Q second-half nopad rotary store. -/
theorem rotary_embedding_q1_block_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i))
    (hExec : exec (rotary_embedding_q1_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem q
          (qSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i) =
        if active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadQ1Spec s q cos sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem q
            (qSecondOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * q_token_stride +
          s.pids 0 * q_head_stride + (idx.1.val + HEAD_HALF) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qSecondOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_q1_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [qSecondOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < Q_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [active, rotaryNopadQ1Spec, qOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [active, tokenIndex, headIndex, hHead, hTok]
    · simp [active, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q second-half nopad rotary store. -/
theorem rotary_embedding_q1_block_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rotary_embedding_q1_block q cos sin q_token_stride
        q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          active s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i)))
      (expected := fun i =>
        rotaryNopadQ1Spec s q cos sin q_token_stride q_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_q1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_q1_block_correct q cos sin q_token_stride
    q_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K first-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`. K analog of `rotary_embedding_q0_block` writing
`out_k0 = k0 * cos - k1 * sin` to `k + off_k0`. The slice does NOT include the
`KV_GROUP_NUM` modular gate (this slice models a one-head-block view; the
group gate is captured separately when proving the full surface). -/
def rotary_embedding_k0_block
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
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
  off_k0 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    dim * $(head_dim_stride)
  off_k1 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_k1 = tl.load(k + off_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  tl.store(k + off_k0, out_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def kOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      BLOCK_TOKENS : Nat) (dim : Nat) : Nat :=
  tokenIndex s BLOCK_TOKENS * k_token_stride +
    headIndex s * k_head_stride + dim * head_dim_stride

def activeK (s : BlockState) (q_total_tokens K_HEAD_NUM BLOCK_TOKENS : Nat) :
    Prop :=
  headIndex s < K_HEAD_NUM ∧ tokenIndex s BLOCK_TOKENS < q_total_tokens

instance activeKDecidable
    (s : BlockState) (q_total_tokens K_HEAD_NUM BLOCK_TOKENS : Nat) :
    Decidable (activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS) := by
  unfold activeK
  infer_instance

noncomputable def rotaryNopadK0Spec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) -
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

theorem rotary_embedding_k0_block_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i)))
    (hExec : exec (rotary_embedding_k0_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem k
          (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i)) =
        if activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadK0Spec s k cos sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem k
            (kOffset s k_token_stride k_head_stride head_dim_stride
              BLOCK_TOKENS (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * k_token_stride +
          s.pids 0 * k_head_stride + idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_k0_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [kOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < K_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [activeK, rotaryNopadK0Spec, kOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [activeK, tokenIndex, headIndex, hHead, hTok]
    · simp [activeK, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_embedding_k0_block_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
          (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rotary_embedding_k0_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS)
        (fun i => (k,
          kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
            (dimIndex i))))
      (expected := fun i =>
        rotaryNopadK0Spec s k cos sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k0_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_k0_block_correct k cos sin k_token_stride
    k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    K_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented K second-half slice of `rotary_emb_nopad.py`'s
`rotary_embedding_kernel`. Writes `out_k1 = k0 * sin + k1 * cos` to
`k + off_k1`. -/
def rotary_embedding_k1_block
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat) :
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
  off_k0 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    dim * $(head_dim_stride)
  off_k1 = token * $(k_token_stride) + cur_head_idx * $(k_head_stride) +
    (dim + $(HEAD_HALF)) * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  loaded_k1 = tl.load(k + off_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)),
    other=0.0)
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  tl.store(k + off_k1, out_k1,
    mask=(cur_head_idx < $(K_HEAD_NUM)) and (token < $(q_total_tokens)))
}

def kSecondOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  tokenIndex s BLOCK_TOKENS * k_token_stride +
    headIndex s * k_head_stride + (dimIndex i + HEAD_HALF) * head_dim_stride

noncomputable def rotaryNopadK1Spec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HEAD_HALF BLOCK_TOKENS : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i)) *
    s.readMem sin (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i) +
  s.readMem k
      (kOffset s k_token_stride k_head_stride head_dim_stride BLOCK_TOKENS
        (dimIndex i + HEAD_HALF)) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride BLOCK_TOKENS i)

theorem rotary_embedding_k1_block_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i))
    (hExec : exec (rotary_embedding_k1_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem k
          (kSecondOffset s k_token_stride k_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i) =
        if activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS then
          rotaryNopadK1Spec s k cos sin k_token_stride k_head_stride
            head_dim_stride cos_token_stride cos_stride HEAD_HALF
            BLOCK_TOKENS i
        else
          s.readMem k
            (kSecondOffset s k_token_stride k_head_stride head_dim_stride
              BLOCK_TOKENS HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        s.pids 1 * BLOCK_TOKENS * k_token_stride +
          s.pids 0 * k_head_stride + (idx.1.val + HEAD_HALF) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kSecondOffset, tokenIndex, headIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHH : 0 < HEAD_HALF
  · simp [exec, rotary_embedding_k1_block, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hHH] at hExec
    rw [← hExec]
    simp only [kSecondOffset, tokenIndex, headIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : s.pids 0 < K_HEAD_NUM
    · by_cases hTok : s.pids 1 * BLOCK_TOKENS < q_total_tokens
      · simp [activeK, rotaryNopadK1Spec, kOffset, cosOffset, tokenIndex,
              headIndex, dimIndex, hHead, hTok, Option.bind, Option.map]
      · simp [activeK, tokenIndex, headIndex, hHead, hTok]
    · simp [activeK, headIndex, hHead]
  · exact False.elim (hHH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rotary_embedding_k1_block_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride
          BLOCK_TOKENS HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rotary_embedding_k1_block k cos sin k_token_stride
        k_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens K_HEAD_NUM HEAD_HALF BLOCK_TOKENS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_HALF =>
          activeK s q_total_tokens K_HEAD_NUM BLOCK_TOKENS)
        (fun i => (k,
          kSecondOffset s k_token_stride k_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF i)))
      (expected := fun i =>
        rotaryNopadK1Spec s k cos sin k_token_stride k_head_stride
          head_dim_stride cos_token_stride cos_stride HEAD_HALF BLOCK_TOKENS i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_embedding_k1_block]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_embedding_k1_block_correct k cos sin k_token_stride
    k_head_stride head_dim_stride cos_token_stride cos_stride q_total_tokens
    K_HEAD_NUM HEAD_HALF BLOCK_TOKENS s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented kv_cache first-half store slice of
`rotary_emb_nopad.py`'s `fused_rotary_embedding_kernel_v2`. Takes a
precomputed `OutK0Pre` tile and proves the writeback into `kv_cache` at the
cache-layout first-half offsets. Parameterized over
`(block_ids, offsets_in_last_block)`. -/
def fused_rotary_v2_kv_cache_first_half_store_slice
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  dim_range0 = tl.arange(0, $(HEAD_HALF))
  out_k0 = tl.load(OutK0Pre + dim_range0)
  kv_range0 = $(block_ids) * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) +
    $(offsets_in_last_block) +
    dim_range0 * $(cached_stride)
  tl.store(kv_cache + kv_range0, out_k0)
}

def kvCacheFirstOffset
    (s : BlockState)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride : Nat)
    (i : Fin HEAD_HALF) : Nat :=
  block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
    offsets_in_last_block + i.val * cached_stride

noncomputable def kvCacheFirstStoreSpec
    (s : BlockState) (OutK0Pre : RegionName) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem OutK0Pre i.val

theorem fused_rotary_v2_kv_cache_first_half_store_slice_correct
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i))
    (hExec : exec (fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem kv_cache
          (kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
            cacheh_stride cached_stride i) =
        kvCacheFirstStoreSpec s OutK0Pre i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
          offsets_in_last_block + idx.1.val * cached_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kvCacheFirstOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, fused_rotary_v2_kv_cache_first_half_store_slice, stepStmts,
        stepStmt, evalOp, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kvCacheFirstOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kvCacheFirstStoreSpec]

theorem fused_rotary_v2_kv_cache_first_half_store_slice_compute_correct
    (OutK0Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i)) :
    ComputeCorrect.Realizes
      (kernel := fused_rotary_v2_kv_cache_first_half_store_slice OutK0Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheFirstOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride i))
      (expected := fun i => kvCacheFirstStoreSpec s OutK0Pre i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_kv_cache_first_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact fused_rotary_v2_kv_cache_first_half_store_slice_correct OutK0Pre kv_cache
    block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
    HEAD_HALF s s' hOutInj hExec i

/-- Proof-oriented kv_cache second-half store slice of
`rotary_emb_nopad.py`'s `fused_rotary_embedding_kernel_v2`. Companion to
the first-half kv_cache slice. -/
def fused_rotary_v2_kv_cache_second_half_store_slice
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) :
    ComputeKernel := triton {
  block_head_index = tl.program_id(0)
  dim_range1 = tl.arange(0, $(HEAD_HALF)) + $(HEAD_HALF)
  out_k1 = tl.load(OutK1Pre + dim_range1)
  kv_range1 = $(block_ids) * $(cacheb_stride) +
    block_head_index * $(cacheh_stride) +
    $(offsets_in_last_block) +
    dim_range1 * $(cached_stride)
  tl.store(kv_cache + kv_range1, out_k1)
}

def kvCacheSecondOffset
    (s : BlockState)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat) (i : Fin HEAD_HALF) : Nat :=
  block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
    offsets_in_last_block + (i.val + HEAD_HALF) * cached_stride

noncomputable def kvCacheSecondStoreSpec
    (s : BlockState) (OutK1Pre : RegionName) (HEAD_HALF : Nat) (i : Fin HEAD_HALF) : ℝ :=
  s.readMem OutK1Pre (i.val + HEAD_HALF)

theorem fused_rotary_v2_kv_cache_second_half_store_slice_correct
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i))
    (hExec : exec (fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF) s = some s') :
    ∀ i : Fin HEAD_HALF,
      s'.readMem kv_cache
          (kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
            cacheh_stride cached_stride HEAD_HALF i) =
        kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_HALF] =>
        block_ids * cacheb_stride + s.pids 0 * cacheh_stride +
          offsets_in_last_block + (idx.1.val + HEAD_HALF) * cached_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kvCacheSecondOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, fused_rotary_v2_kv_cache_second_half_store_slice, stepStmts,
        stepStmt, evalOp, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kvCacheSecondOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kvCacheSecondStoreSpec]

theorem fused_rotary_v2_kv_cache_second_half_store_slice_compute_correct
    (OutK1Pre kv_cache : RegionName)
    (block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
      HEAD_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_HALF =>
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := fused_rotary_v2_kv_cache_second_half_store_slice OutK1Pre
        kv_cache block_ids offsets_in_last_block cacheb_stride cacheh_stride
        cached_stride HEAD_HALF)
      (initialState := s)
      (write := fun i : Fin HEAD_HALF => some (kv_cache,
        kvCacheSecondOffset s block_ids offsets_in_last_block cacheb_stride
          cacheh_stride cached_stride HEAD_HALF i))
      (expected := fun i => kvCacheSecondStoreSpec s OutK1Pre HEAD_HALF i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_rotary_v2_kv_cache_second_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact fused_rotary_v2_kv_cache_second_half_store_slice_correct OutK1Pre kv_cache
    block_ids offsets_in_last_block cacheb_stride cacheh_stride cached_stride
    HEAD_HALF s s' hOutInj hExec i

/-! ## Full-kernel Q-surface store proofs

The full Q surface `rotary_embedding_q_surface` writes the Q first-half and
Q second-half tiles consecutively over the 3D `[BLOCK_TOKENS, 1, HEAD_HALF]`
tile. We characterize each store at the algorithm layer using the substrate
helpers `BlockState.scatter_readback_prop_masked_nd` (for the active scatter)
and `BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem` (for
stripping the sibling Q store, which targets the same region at disjoint
offsets). -/

/-- Per-tile-index Q first-half offset for the full Q surface kernel.
The 3D tile shape is `[BLOCK_TOKENS, 1, HEAD_HALF]`. -/
def qFullFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + idx.2.2.1.val * head_dim_stride

/-- Per-tile-index Q second-half offset for the full Q surface kernel. -/
def qFullSecondOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * q_token_stride +
    s.pids 0 * q_head_stride + (idx.2.2.1.val + HEAD_HALF) * head_dim_stride

/-- Per-tile-index cos/sin offset (the cos/sin tile is `[BLOCK_TOKENS,
HEAD_HALF]` and broadcasts into the 3D Q tile along the singleton head
axis). -/
def cosFullOffset
    (s : BlockState) (cos_token_stride cos_stride BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Nat :=
  (s.pids 1 * BLOCK_TOKENS + idx.1.val) * cos_token_stride +
    idx.2.2.1.val * cos_stride

/-- Active predicate for the Q stores in the full Q surface kernel: head
index in range AND the per-tile token index in range. -/
def activeFull (s : BlockState) (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : Prop :=
  s.pids 0 < Q_HEAD_NUM ∧
    s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens

instance activeFullDecidable (s : BlockState)
    (q_total_tokens Q_HEAD_NUM BLOCK_TOKENS : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) :
    Decidable (activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx) := by
  unfold activeFull
  infer_instance

/-- Q first-half rotary spec for the full Q surface kernel:
`q0 * cos - q1 * sin`. -/
noncomputable def rotaryNopadQ0FullSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem q
      (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) -
  s.readMem q
      (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- Q second-half rotary spec for the full Q surface kernel:
`q0 * sin + q1 * cos`. -/
noncomputable def rotaryNopadQ1FullSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      BLOCK_TOKENS HEAD_HALF : Nat)
    (idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF]) : ℝ :=
  s.readMem q
      (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS idx) *
    s.readMem sin (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx) +
  s.readMem q
      (qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
        BLOCK_TOKENS HEAD_HALF idx) *
    s.readMem cos (cosFullOffset s cos_token_stride cos_stride BLOCK_TOKENS idx)

/-- Algorithm-layer correctness for the Q first-half store in the full
3D-tile Q surface kernel `rotary_embedding_q_surface`. The Q second-half
store (same region, disjoint offsets) is stripped via
`foldl_writeMem_same_region_disjoint_offsets_readMem`. -/
theorem rotary_embedding_q_surface_q0_correct
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
          BLOCK_TOKENS idx))
    (hOutDisjoint :
      ∀ idx idx' : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
        qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx ≠
          qFullSecondOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS HEAD_HALF idx')
    (hExec : exec (rotary_embedding_q_surface Q Cos Sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        q_total_tokens Q_HEAD_NUM HEAD_HALF BLOCK_TOKENS) s = some s') :
    ∀ idx : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF],
      s'.readMem Q
          (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
            BLOCK_TOKENS idx) =
        if activeFull s q_total_tokens Q_HEAD_NUM BLOCK_TOKENS idx then
          rotaryNopadQ0FullSpec s Q Cos Sin q_token_stride q_head_stride
            head_dim_stride cos_token_stride cos_stride BLOCK_TOKENS HEAD_HALF idx
        else
          s.readMem Q
            (qFullFirstOffset s q_token_stride q_head_stride head_dim_stride
              BLOCK_TOKENS idx) := by
  intro idx
  by_cases hTok : 0 < BLOCK_TOKENS
  · by_cases hHalf : 0 < HEAD_HALF
    · have hRawInj : Function.Injective
          (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
            (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
              s.pids 0 * q_head_stride + j.2.2.1.val * head_dim_stride) := by
        simpa [qFullFirstOffset] using hOutInj
      simp [exec, rotary_embedding_q_surface, stepStmts, stepStmt, evalOp,
            Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            ComparableDType.lt, hTok, hHalf,
            TileShape.dropInsertedIndex_two_pair,
            TileShape.dropInsertedIndex_zero_pair] at hExec
      rw [← hExec]
      simp only [qFullFirstOffset]
      -- Strip the outer Q second-half foldl: same region Q, disjoint offsets.
      rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
            Q (fun j : TileIndex [BLOCK_TOKENS, 1, HEAD_HALF] =>
                (s.pids 1 * BLOCK_TOKENS + j.1.val) * q_token_stride +
                  s.pids 0 * q_head_stride +
                  (j.2.2.1.val + HEAD_HALF) * head_dim_stride)
            _ _ _ _ _
            (fun k _ _ => by
              have := hOutDisjoint idx k
              simpa [qFullFirstOffset, qFullSecondOffset] using this)]
      -- The inner foldl is the Q first-half scatter.
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
      by_cases hHead : s.pids 0 < Q_HEAD_NUM
      · by_cases hTokIdx : s.pids 1 * BLOCK_TOKENS + idx.1.val < q_total_tokens
        · simp [activeFull, rotaryNopadQ0FullSpec, qFullFirstOffset,
                qFullSecondOffset, cosFullOffset, hHead, hTokIdx,
                Option.bind, Option.map]
        · simp [activeFull, hHead, hTokIdx]
      · simp [activeFull, hHead]
    · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.2.1.isLt))
  · exact False.elim (hTok (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

end VeriTile.Bench.TritonBenchG.RotaryEmbNopad

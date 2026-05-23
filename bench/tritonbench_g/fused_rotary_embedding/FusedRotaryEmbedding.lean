import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

This keeps the unconditional Q rotary writes plus the conditional K rotary and
K/V cache-fill path guarded by `cur_head_idx % KV_GROUP_NUM == 0`. -/
def decoding_fused_rotary_embedding_kernel_surface
    (q k v cos sin k_cache v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (x q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM
      : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)

  dim_range = tl.arange(0, $(HEAD_DIM))
  dim_range0 = tl.arange(0, $(HEAD_DIM) // $(2))
  dim_range1 = tl.arange($(HEAD_DIM) // $(2), $(HEAD_DIM))

  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)

  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)

  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(q + off_q0, out_q0)
  tl.store(q + off_q1, out_q1)

  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
    off_k0 = off_kv + dim_range0 * $(head_dim_stride)
    off_k1 = off_kv + dim_range1 * $(head_dim_stride)
    loaded_k0 = tl.load(k + off_k0)
    loaded_k1 = tl.load(k + off_k1)

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    past_kv_seq_len = tl.load(context_lengths + cur_token_idx) - $(1)

    last_block_idx = past_kv_seq_len // $(block_size)
    block_ids = tl.load(BLOCK_TABLES + cur_token_idx * $(bts_stride) +
      last_block_idx * $(btb_stride))
    offsets_in_last_block = past_kv_seq_len % $(block_size)
    offsets_cache_base = block_ids * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride)
    k_range0 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    k_range1 = offsets_cache_base +
      offsets_in_last_block * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
    tl.store(k_cache + k_range1, out_k1)

    off_v = off_kv + dim_range * $(head_dim_stride)
    loaded_v = tl.load(v + off_v)
    v_range = block_ids * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      offsets_in_last_block * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

/-- The full Python-shaped fused rotary decoding surface lowers to the
algorithm layer, including both Q halves, conditional K rotation, and K/V cache
stores. -/
theorem decoding_fused_rotary_embedding_kernel_surface_toAlgorithm_supported
    (q k v cos sin k_cache v_cache : RegionName)
    (BLOCK_TABLES context_lengths : Region .nat)
    (x q_token_stride q_head_stride k_token_stride k_head_stride
      head_dim_stride cos_token_stride cos_stride kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride vcb_stride vch_stride vcs_stride
      vcd_stride bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM
      : Nat) :
    ∃ alg, (decoding_fused_rotary_embedding_kernel_surface q k v cos sin
      k_cache v_cache BLOCK_TABLES context_lengths x q_token_stride
      q_head_stride k_token_stride k_head_stride head_dim_stride
      cos_token_stride cos_stride kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride vcb_stride vch_stride vcs_stride vcd_stride
      bts_stride btb_stride block_size KV_GROUP_NUM HEAD_DIM).toAlgorithm?
        = Except.ok alg := by
  simp [decoding_fused_rotary_embedding_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the unconditional Q rotary part of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.

The full Python kernel also conditionally rotates K and fills K/V caches when
`cur_head_idx % KV_GROUP_NUM == 0`. That branch depends on context-length
metadata and cache block tables, so this surface covers the unconditional Q
updates that every program instance performs: both the first and second rotary
halves are written. -/
def decoding_fused_rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0)
  loaded_q1 = tl.load(Q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin)
  loaded_sin = tl.load(Sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0)
  tl.store(Q + off_q1, out_q1)
}

/-- The standalone Q rotary decoding surface lowers to the algorithm layer. -/
theorem decoding_fused_rotary_embedding_q_surface_toAlgorithm_supported
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ∃ alg, (decoding_fused_rotary_embedding_q_surface Q Cos Sin q_token_stride
      q_head_stride head_dim_stride cos_token_stride cos_stride _HEAD_DIM
      HALF_DIM).toAlgorithm? = Except.ok alg := by
  simp [decoding_fused_rotary_embedding_q_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented Q first-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

The full kernel updates Q, conditionally rotates K, and fills K/V caches. This
slice captures the unconditional Q first-half rotary writeback:
`q0 * cos - q1 * sin`. -/
def decoding_fused_rotary_embedding_q_first_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0)
}

def dimIndex (i : Fin HALF_DIM) : Nat :=
  i.val

def qBase (s : BlockState) (q_token_stride q_head_stride : Nat) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride

def qFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + dimIndex i * head_dim_stride

def qSecondOffset
    (s : BlockState)
    (q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride

def cosOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

def sinOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

noncomputable def qFirstSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

/-- Algorithm-layer correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) =
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
          idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qFirstOffset, qBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_q_first_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [qFirstOffset, qBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := q)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_q" TileDType.nat []
            (Tile.scalar (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride))
          |>.setReg "off_q0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_q1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) -
                s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            idx.1.val * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) -
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [qFirstSpec, qFirstOffset, qSecondOffset, cosOffset, sinOffset, qBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q, qFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_q_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact decoding_fused_rotary_embedding_q_first_half_correct q cos sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-! ## Q second-half rotary writeback (`out_q1 = q1 * cos + q0 * sin`) -/

/-- Proof-oriented Q second-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`. Captures the unconditional Q
second-half rotary writeback: `q1' = q1 * cos + q0 * sin`. -/
def decoding_fused_rotary_embedding_q_second_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q1 = loaded_q1 * loaded_cos + loaded_q0 * loaded_sin
  tl.store(q + off_q1, out_q1)
}

noncomputable def qSecondSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) +
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

/-- Algorithm-layer correctness for the Q second-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_second_half_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem q
          (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) =
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
          (idx.1.val + HALF_DIM) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qSecondOffset, qBase, dimIndex] using h
    cases a; cases b; simp only at hab; cases hab; rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_q_second_half, stepStmts,
          stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [qSecondOffset, qBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := q)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_q" TileDType.nat []
            (Tile.scalar (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride))
          |>.setReg "off_q0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_q1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) +
                s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            (idx.1.val + HALF_DIM) * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) +
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [qSecondSpec, qFirstOffset, qSecondOffset, cosOffset, sinOffset, qBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q second-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_second_half_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q,
          qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i)))
      (expected := fun i =>
        qSecondSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_q_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact decoding_fused_rotary_embedding_q_second_half_correct q cos sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented K first-half slice of
`decoding_fused_rotary_embedding_kernel`. K analog of the Q first-half slice
writing `out_k0 = k0 * cos - k1 * sin` to `k + off_k0`. The KV_GROUP gating
that ungates the K branch in the full kernel is omitted here; the slice models
the K rotary stores assuming the gate is active. -/
def decoding_fused_rotary_embedding_k_first_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
  tl.store(k + off_k0, out_k0)
}

def kBase (s : BlockState) (k_token_stride k_head_stride : Nat) : Nat :=
  s.pids 1 * k_token_stride + s.pids 0 * k_head_stride

def kFirstOffset
    (s : BlockState) (k_token_stride k_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + dimIndex i * head_dim_stride

def kSecondOffset
    (s : BlockState)
    (k_token_stride k_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kBase s k_token_stride k_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride

noncomputable def kFirstSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

theorem decoding_fused_rotary_embedding_k_first_half_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k
          (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) =
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
          idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kFirstOffset, kBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_k_first_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [kFirstOffset, kBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := k)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_k_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_kv" TileDType.nat []
            (Tile.scalar (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride))
          |>.setReg "off_k0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_k1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) -
                s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
            idx.1.val * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) -
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [kFirstSpec, kFirstOffset, kSecondOffset, cosOffset, sinOffset, kBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem decoding_fused_rotary_embedding_k_first_half_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kFirstOffset s k_token_stride k_head_stride head_dim_stride i))
      (expected := fun i =>
        kFirstSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_first_half_correct k cos sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented K second-half slice of
`decoding_fused_rotary_embedding_kernel`. Writes `out_k1 = k0 * sin + k1 * cos`
to `k + off_k1`. -/
def decoding_fused_rotary_embedding_k_second_half
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_kv = cur_token_idx * $(k_token_stride) + cur_k_head_idx * $(k_head_stride)
  off_k0 = off_kv + dim_range0 * $(head_dim_stride)
  off_k1 = off_kv + dim_range1 * $(head_dim_stride)
  loaded_k0 = tl.load(k + off_k0)
  loaded_k1 = tl.load(k + off_k1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
  tl.store(k + off_k1, out_k1)
}

noncomputable def kSecondSpec
    (s : BlockState) (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem k (kFirstOffset s k_token_stride k_head_stride head_dim_stride i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i) +
  s.readMem k
      (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i)

theorem decoding_fused_rotary_embedding_k_second_half_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k
          (kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i) =
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
          (idx.1.val + HALF_DIM) * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kSecondOffset, kBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_k_second_half, stepStmts, stepStmt,
          evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [kSecondOffset, kBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := k)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_k_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_kv" TileDType.nat []
            (Tile.scalar (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride))
          |>.setReg "off_k0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_k1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_k0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem k
                (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_k1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride) +
                s.readMem k
                    (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
            (idx.1.val + HALF_DIM) * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) +
          s.readMem k
              (s.pids 1 * k_token_stride + s.pids 0 * k_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [kSecondSpec, kFirstOffset, kSecondOffset, cosOffset, sinOffset, kBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem decoding_fused_rotary_embedding_k_second_half_compute_correct
    (k cos sin : RegionName)
    (k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        k_token_stride k_head_stride head_dim_stride cos_token_stride
        cos_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k,
        kSecondOffset s k_token_stride k_head_stride head_dim_stride HALF_DIM i))
      (expected := fun i =>
        kSecondSpec s k cos sin k_token_stride k_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_second_half]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_second_half_correct k cos sin
    k_token_stride k_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented v_cache store slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`. Takes a precomputed `LoadedV` tile
and proves the writeback into `v_cache` at the canonical cache-layout offset
parameterized on `(block_id, offsets_in_last_block)`. -/
def decoding_fused_rotary_embedding_v_cache_store_slice
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range = tl.arange(0, $(HEAD_DIM))
  loaded_v = tl.load(LoadedV + dim_range)
  v_range = $(block_id) * $(vcb_stride) +
    cur_k_head_idx * $(vch_stride) +
    $(offsets_in_last_block) * $(vcs_stride) +
    dim_range * $(vcd_stride)
  tl.store(v_cache + v_range, loaded_v)
}

def vCacheOffset
    (s : BlockState)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  block_id * vcb_stride + s.pids 0 * vch_stride +
    offsets_in_last_block * vcs_stride + i.val * vcd_stride

noncomputable def vCacheStoreSpec
    (s : BlockState) (LoadedV : RegionName) (i : Fin HEAD_DIM) : ℝ :=
  s.readMem LoadedV i.val

theorem decoding_fused_rotary_embedding_v_cache_store_slice_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_v_cache_store_slice LoadedV
        v_cache block_id offsets_in_last_block vcb_stride vch_stride vcs_stride
        vcd_stride HEAD_DIM) s = some s') :
    ∀ i : Fin HEAD_DIM,
      s'.readMem v_cache
          (vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
            vcs_stride vcd_stride i) =
        vCacheStoreSpec s LoadedV i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_DIM] =>
        block_id * vcb_stride + s.pids 0 * vch_stride +
          offsets_in_last_block * vcs_stride + idx.1.val * vcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [vCacheOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_v_cache_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [vCacheOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [vCacheStoreSpec]

theorem decoding_fused_rotary_embedding_v_cache_store_slice_compute_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
      HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_store_slice LoadedV
        v_cache block_id offsets_in_last_block vcb_stride vch_stride vcs_stride
        vcd_stride HEAD_DIM)
      (initialState := s)
      (write := fun i : Fin HEAD_DIM => some (v_cache,
        vCacheOffset s block_id offsets_in_last_block vcb_stride vch_stride
          vcs_stride vcd_stride i))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_v_cache_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_v_cache_store_slice_correct LoadedV v_cache
    block_id offsets_in_last_block vcb_stride vch_stride vcs_stride vcd_stride
    HEAD_DIM s s' hOutInj hExec i

/-- Guarded V-cache store slice for the `handle_kv` branch of
`decoding_fused_rotary_embedding_kernel`.

This keeps the Python branch guard `cur_head_idx % KV_GROUP_NUM == 0` and uses
the derived `cur_k_head_idx = cur_head_idx // KV_GROUP_NUM` in the cache
address. `block_id` and `offsets_in_last_block` remain parameters here; the
companion full surface above records their origin from `context_lengths` and
`BLOCK_TABLES`. -/
def decoding_fused_rotary_embedding_v_cache_guarded_store_slice
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range = tl.arange(0, $(HEAD_DIM))
    loaded_v = tl.load(LoadedV + dim_range)
    v_range = $(block_id) * $(vcb_stride) +
      cur_k_head_idx * $(vch_stride) +
      $(offsets_in_last_block) * $(vcs_stride) +
      dim_range * $(vcd_stride)
    tl.store(v_cache + v_range, loaded_v)
  }
}

def handleKv (s : BlockState) (KV_GROUP_NUM : Nat) : Prop :=
  s.pids 0 % KV_GROUP_NUM = 0

instance handleKvDecidable (s : BlockState) (KV_GROUP_NUM : Nat) :
    Decidable (handleKv s KV_GROUP_NUM) := by
  unfold handleKv
  infer_instance

def vCacheGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  block_id * vcb_stride + (s.pids 0 / KV_GROUP_NUM) * vch_stride +
    offsets_in_last_block * vcs_stride + i.val * vcd_stride

theorem decoding_fused_rotary_embedding_v_cache_guarded_store_slice_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
        vch_stride vcs_stride vcd_stride HEAD_DIM) s = some s') :
    ∀ i : Fin HEAD_DIM,
      s'.readMem v_cache
          (vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
            vcb_stride vch_stride vcs_stride vcd_stride i) =
        if handleKv s KV_GROUP_NUM then
          vCacheStoreSpec s LoadedV i
        else
          s.readMem v_cache
            (vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
              vcb_stride vch_stride vcs_stride vcd_stride i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HEAD_DIM] =>
        block_id * vcb_stride + (s.pids 0 / KV_GROUP_NUM) * vch_stride +
          offsets_in_last_block * vcs_stride + idx.1.val * vcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [vCacheGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_v_cache_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, vCacheGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [vCacheStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, vCacheGuardedOffset]

theorem decoding_fused_rotary_embedding_v_cache_guarded_store_slice_compute_correct
    (LoadedV v_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM vcb_stride vch_stride
      vcs_stride vcd_stride HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
        vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          vCacheGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
            vcb_stride vch_stride vcs_stride vcd_stride i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_v_cache_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := decoding_fused_rotary_embedding_v_cache_guarded_store_slice_correct
    LoadedV v_cache block_id offsets_in_last_block KV_GROUP_NUM vcb_stride
    vch_stride vcs_stride vcd_stride HEAD_DIM s s' hOutInj hExec i
  simpa [hActive] using h

/-- Proof-oriented k_cache first-half store slice of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.
Takes a precomputed `OutK0Pre` tile (the post-rotary K first-half values)
and proves the writeback into `k_cache` at the cache-layout first-half
offsets. Parameterized over `(block_id, offsets_in_last_block, x)`, covering
both the legacy 4D cache layout (`kcsplit_x_stride = 0`, `x = head_dim`) and
the new split layout `[num_blocks, heads, head_dim // x, block_size, x]`. -/
def decoding_fused_rotary_embedding_k_cache_first_half_store_slice
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  out_k0 = tl.load(OutK0Pre + dim_range0)
  k_range0 = $(block_id) * $(kcb_stride) +
    cur_k_head_idx * $(kch_stride) +
    $(offsets_in_last_block) * $(kcs_stride) +
    (dim_range0 // $(x)) * $(kcsplit_x_stride) +
    (dim_range0 % $(x)) * $(kcd_stride)
  tl.store(k_cache + k_range0, out_k0)
}

def kCacheFirstOffset
    (s : BlockState)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + s.pids 0 * kch_stride +
    offsets_in_last_block * kcs_stride +
    (i.val / x) * kcsplit_x_stride + (i.val % x) * kcd_stride

noncomputable def kCacheFirstStoreSpec
    (s : BlockState) (OutK0Pre : RegionName) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK0Pre i.val

theorem decoding_fused_rotary_embedding_k_cache_first_half_store_slice_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_k_cache_first_half_store_slice
        OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
            kch_stride kcsplit_x_stride kcs_stride kcd_stride i) =
        kCacheFirstStoreSpec s OutK0Pre i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + s.pids 0 * kch_stride +
          offsets_in_last_block * kcs_stride +
          (idx.1.val / x) * kcsplit_x_stride + (idx.1.val % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheFirstOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_first_half_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kCacheFirstOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kCacheFirstStoreSpec]

theorem decoding_fused_rotary_embedding_k_cache_first_half_store_slice_compute_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_cache_first_half_store_slice
        OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k_cache,
        kCacheFirstOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_first_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_cache_first_half_store_slice_correct
    OutK0Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
    kcsplit_x_stride kcs_stride kcd_stride HALF_DIM s s' hOutInj hExec i

/-- Proof-oriented k_cache second-half store slice. K-side analog of the
first-half slice. -/
def decoding_fused_rotary_embedding_k_cache_second_half_store_slice
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_k_head_idx = tl.program_id(0)
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  out_k1 = tl.load(OutK1Pre + dim_range1)
  k_range1 = $(block_id) * $(kcb_stride) +
    cur_k_head_idx * $(kch_stride) +
    $(offsets_in_last_block) * $(kcs_stride) +
    (dim_range1 // $(x)) * $(kcsplit_x_stride) +
    (dim_range1 % $(x)) * $(kcd_stride)
  tl.store(k_cache + k_range1, out_k1)
}

def kCacheSecondOffset
    (s : BlockState)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat) (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + s.pids 0 * kch_stride +
    offsets_in_last_block * kcs_stride +
    ((i.val + HALF_DIM) / x) * kcsplit_x_stride +
    ((i.val + HALF_DIM) % x) * kcd_stride

noncomputable def kCacheSecondStoreSpec
    (s : BlockState) (OutK1Pre : RegionName) (HALF_DIM : Nat) (i : Fin HALF_DIM) : ℝ :=
  s.readMem OutK1Pre (i.val + HALF_DIM)

theorem decoding_fused_rotary_embedding_k_cache_second_half_store_slice_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i))
    (hExec : exec (decoding_fused_rotary_embedding_k_cache_second_half_store_slice
        OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
            kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i) =
        kCacheSecondStoreSpec s OutK1Pre HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + s.pids 0 * kch_stride +
          offsets_in_last_block * kcs_stride +
          ((idx.1.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((idx.1.val + HALF_DIM) % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheSecondOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_second_half_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hExec
  rw [← hExec]
  simp only [kCacheSecondOffset]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
  simp [kCacheSecondStoreSpec]

theorem decoding_fused_rotary_embedding_k_cache_second_half_store_slice_compute_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block x kcb_stride kch_stride kcsplit_x_stride
      kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_cache_second_half_store_slice
        OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
        kcsplit_x_stride kcs_stride kcd_stride HALF_DIM)
      (initialState := s)
      (write := fun i : Fin HALF_DIM => some (k_cache,
        kCacheSecondOffset s block_id offsets_in_last_block x kcb_stride
          kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM i))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_second_half_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact decoding_fused_rotary_embedding_k_cache_second_half_store_slice_correct
    OutK1Pre k_cache block_id offsets_in_last_block x kcb_stride kch_stride
    kcsplit_x_stride kcs_stride kcd_stride HALF_DIM s s' hOutInj hExec i

/-- Guarded K-cache first-half store slice for the `handle_kv` branch. -/
def decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range0 = tl.arange(0, $(HALF_DIM))
    out_k0 = tl.load(OutK0Pre + dim_range0)
    k_range0 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range0 // $(x)) * $(kcsplit_x_stride) +
      (dim_range0 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range0, out_k0)
  }
}

def kCacheFirstGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    (i.val / x) * kcsplit_x_stride + (i.val % x) * kcd_stride

theorem decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride i))
    (hExec : exec
        (decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheFirstGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride i) =
        if handleKv s KV_GROUP_NUM then
          kCacheFirstStoreSpec s OutK0Pre i
        else
          s.readMem k_cache
            (kCacheFirstGuardedOffset s block_id offsets_in_last_block
              KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
              kcd_stride i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
          offsets_in_last_block * kcs_stride +
          (idx.1.val / x) * kcsplit_x_stride + (idx.1.val % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheFirstGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheFirstGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [kCacheFirstStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheFirstGuardedOffset]

theorem decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_compute_correct
    (OutK0Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheFirstGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride i)) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          kCacheFirstGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h :=
    decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_correct
      OutK0Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
      kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM
      s s' hOutInj hExec i
  simpa [hActive] using h

/-- Guarded K-cache second-half store slice for the `handle_kv` branch. -/
def decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  handle_kv = cur_head_idx % $(KV_GROUP_NUM) == $(0)
  if handle_kv {
    cur_k_head_idx = cur_head_idx // $(KV_GROUP_NUM)
    dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
    out_k1 = tl.load(OutK1Pre + dim_range1)
    k_range1 = $(block_id) * $(kcb_stride) +
      cur_k_head_idx * $(kch_stride) +
      $(offsets_in_last_block) * $(kcs_stride) +
      (dim_range1 // $(x)) * $(kcsplit_x_stride) +
      (dim_range1 % $(x)) * $(kcd_stride)
    tl.store(k_cache + k_range1, out_k1)
  }
}

def kCacheSecondGuardedOffset
    (s : BlockState)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
    offsets_in_last_block * kcs_stride +
    ((i.val + HALF_DIM) / x) * kcsplit_x_stride +
    ((i.val + HALF_DIM) % x) * kcd_stride

theorem decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM i))
    (hExec : exec
        (decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem k_cache
          (kCacheSecondGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride HALF_DIM i) =
        if handleKv s KV_GROUP_NUM then
          kCacheSecondStoreSpec s OutK1Pre HALF_DIM i
        else
          s.readMem k_cache
            (kCacheSecondGuardedOffset s block_id offsets_in_last_block
              KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
              kcd_stride HALF_DIM i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        block_id * kcb_stride + (s.pids 0 / KV_GROUP_NUM) * kch_stride +
          offsets_in_last_block * kcs_stride +
          ((idx.1.val + HALF_DIM) / x) * kcsplit_x_stride +
          ((idx.1.val + HALF_DIM) % x) * kcd_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheSecondGuardedOffset] using h
    cases a; cases b
    simp only at hab; cases hab; rfl
  simp [exec, decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.eq] at hExec
  by_cases hHandle : s.pids 0 % KV_GROUP_NUM = 0
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheSecondGuardedOffset]
    rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]
    simp [kCacheSecondStoreSpec]
  · simp [hHandle] at hExec
    rw [← hExec]
    simp [hHandle, handleKv, kCacheSecondGuardedOffset]

theorem decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_compute_correct
    (OutK1Pre k_cache : RegionName)
    (block_id offsets_in_last_block KV_GROUP_NUM x kcb_stride kch_stride
      kcsplit_x_stride kcs_stride kcd_stride HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        kCacheSecondGuardedOffset s block_id offsets_in_last_block KV_GROUP_NUM
          x kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM i)) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
          kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride
          HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          kCacheSecondGuardedOffset s block_id offsets_in_last_block
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h :=
    decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_correct
      OutK1Pre k_cache block_id offsets_in_last_block KV_GROUP_NUM x
      kcb_stride kch_stride kcsplit_x_stride kcs_stride kcd_stride HALF_DIM
      s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Python metadata-specialized guarded cache writebacks -/

/-- Python decode metadata:
`past_kv_seq_len = context_lengths[cur_token_idx] - 1`. -/
def decodingPastKvSeqLen (s : BlockState) (context_lengths : RegionName) : Nat :=
  s.readMemValue .nat context_lengths (s.pids 1) - 1

/-- Python decode metadata: `last_block_idx = past_kv_seq_len // block_size`. -/
def decodingLastBlockIdx
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths / block_size

/-- Python decode metadata:
`block_ids = BLOCK_TABLES[cur_token_idx, last_block_idx]`. -/
def decodingBlockId
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (bts_stride btb_stride block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 1 * bts_stride +
      decodingLastBlockIdx s context_lengths block_size * btb_stride)

/-- Python decode metadata:
`offsets_in_last_block = past_kv_seq_len % block_size`. -/
def decodingOffsetsInLastBlock
    (s : BlockState) (context_lengths : RegionName) (block_size : Nat) : Nat :=
  decodingPastKvSeqLen s context_lengths % block_size

def decodingKCacheFirstGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheFirstGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride i

def decodingKCacheSecondGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  kCacheSecondGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
    kcd_stride HALF_DIM i

def decodingVCacheGuardedOffset
    (s : BlockState) (BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
      btb_stride block_size : Nat)
    (i : Fin HEAD_DIM) : Nat :=
  vCacheGuardedOffset s
    (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
      block_size)
    (decodingOffsetsInLastBlock s context_lengths block_size)
    KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride i

/-- Public guarded K-cache first-half theorem specialized to the Python
`context_lengths` and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct
    (OutK0Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size i)) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  simpa [decodingKCacheFirstGuardedOffset]
    using
      decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice_compute_correct
        OutK0Pre k_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride HALF_DIM s hOutInj

/-- Public guarded K-cache second-half theorem specialized to the Python
`context_lengths` and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct
    (OutK1Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
      kcd_stride bts_stride btb_stride block_size HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride bts_stride btb_stride block_size HALF_DIM i)) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
            block_size)
          (decodingOffsetsInLastBlock s context_lengths block_size)
          KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
          kcd_stride HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HALF_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
            kcd_stride bts_stride btb_stride block_size HALF_DIM i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre HALF_DIM i) := by
  simpa [decodingKCacheSecondGuardedOffset]
    using
      decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice_compute_correct
        OutK1Pre k_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM x kcb_stride kch_stride kcsplit_x_stride kcs_stride
        kcd_stride HALF_DIM s hOutInj

/-- Public guarded V-cache theorem specialized to the Python `context_lengths`
and `BLOCK_TABLES` metadata path. -/
theorem decoding_fused_rotary_embedding_context_v_cache_guarded_store_slice_compute_correct
    (LoadedV v_cache BLOCK_TABLES context_lengths : RegionName)
    (KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride bts_stride
      btb_stride block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths KV_GROUP_NUM
          vcb_stride vch_stride vcs_stride vcd_stride bts_stride btb_stride
          block_size i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
          block_size)
        (decodingOffsetsInLastBlock s context_lengths block_size)
        KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin HEAD_DIM => handleKv s KV_GROUP_NUM)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride
            bts_stride btb_stride block_size i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  simpa [decodingVCacheGuardedOffset]
    using decoding_fused_rotary_embedding_v_cache_guarded_store_slice_compute_correct
      LoadedV v_cache
      (decodingBlockId s BLOCK_TABLES context_lengths bts_stride btb_stride
        block_size)
      (decodingOffsetsInLastBlock s context_lengths block_size)
      KV_GROUP_NUM vcb_stride vch_stride vcs_stride vcd_stride HEAD_DIM s
      hOutInj

/-! ## Python test-shape wrappers

The checked `fused_rotary_embedding.py` test uses `total_tokens = 16`,
`q_head_num = 8`, `kv_head_num = 4`, `head_dim = 64`, `block_size = 4`,
and the default 4D K/V cache layout in its first case. The wrappers below pin
those strides and expose the Q/K rotary writes plus guarded K/V cache writes
for that concrete Python shape. -/

theorem decoding_fused_rotary_embedding_q_first_half_python_test_shape_compute_correct
    (q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qFirstOffset s 512 64 1 i)))
      (expected := fun i => qFirstSpec s q cos sin 512 64 1 64 1 32 i) := by
  apply decoding_fused_rotary_embedding_q_first_half_compute_correct
  intro a b h
  apply Fin.ext
  simp [qFirstOffset, qBase, dimIndex] at h
  omega

theorem decoding_fused_rotary_embedding_q_second_half_python_test_shape_compute_correct
    (q cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qSecondOffset s 512 64 1 32 i)))
      (expected := fun i => qSecondSpec s q cos sin 512 64 1 64 1 32 i) := by
  apply decoding_fused_rotary_embedding_q_second_half_compute_correct
  intro a b h
  apply Fin.ext
  simp [qSecondOffset, qBase, dimIndex] at h
  omega

theorem decoding_fused_rotary_embedding_k_first_half_python_test_shape_compute_correct
    (k cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kFirstOffset s 256 64 1 i))
      (expected := fun i => kFirstSpec s k cos sin 256 64 1 64 1 32 i) := by
  apply decoding_fused_rotary_embedding_k_first_half_compute_correct
  intro a b h
  apply Fin.ext
  simp [kFirstOffset, kBase, dimIndex] at h
  omega

theorem decoding_fused_rotary_embedding_k_second_half_python_test_shape_compute_correct
    (k cos sin : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kSecondOffset s 256 64 1 32 i))
      (expected := fun i => kSecondSpec s k cos sin 256 64 1 64 1 32 i) := by
  apply decoding_fused_rotary_embedding_k_second_half_compute_correct
  intro a b h
  apply Fin.ext
  simp [kSecondOffset, kBase, dimIndex] at h
  omega

theorem decoding_fused_rotary_embedding_context_k_cache_first_half_old_layout_python_test_shape_compute_correct
    (OutK0Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  apply
    decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  have ha64 : a.val < 64 := by omega
  have hb64 : b.val < 64 := by omega
  simp [decodingKCacheFirstGuardedOffset, kCacheFirstGuardedOffset,
    Nat.div_eq_of_lt ha64, Nat.div_eq_of_lt hb64,
    Nat.mod_eq_of_lt ha64, Nat.mod_eq_of_lt hb64] at h
  omega

theorem decoding_fused_rotary_embedding_context_k_cache_second_half_old_layout_python_test_shape_compute_correct
    (OutK1Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i) := by
  apply
    decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  have ha64 : a.val + 32 < 64 := by omega
  have hb64 : b.val + 32 < 64 := by omega
  simp [decodingKCacheSecondGuardedOffset, kCacheSecondGuardedOffset,
    Nat.div_eq_of_lt ha64, Nat.div_eq_of_lt hb64,
    Nat.mod_eq_of_lt ha64, Nat.mod_eq_of_lt hb64] at h
  omega

/-- Python test case 2: new 5D K-cache layout
`[num_blocks, kv_head_num, head_dim // 16, block_size, 16]`. -/
theorem decoding_fused_rotary_embedding_context_k_cache_first_half_new_layout_python_test_shape_compute_correct
    (OutK0Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i) := by
  apply
    decoding_fused_rotary_embedding_context_k_cache_first_half_guarded_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [decodingKCacheFirstGuardedOffset, kCacheFirstGuardedOffset] at h
  omega

/-- Python test case 2: new 5D K-cache layout, second rotary half. -/
theorem decoding_fused_rotary_embedding_context_k_cache_second_half_new_layout_python_test_shape_compute_correct
    (OutK1Pre k_cache BLOCK_TABLES context_lengths : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i) := by
  apply
    decoding_fused_rotary_embedding_context_k_cache_second_half_guarded_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [decodingKCacheSecondGuardedOffset, kCacheSecondGuardedOffset] at h
  omega

theorem decoding_fused_rotary_embedding_context_v_cache_python_test_shape_compute_correct
    (LoadedV v_cache BLOCK_TABLES context_lengths : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
        (decodingOffsetsInLastBlock s context_lengths 4)
        2 1024 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => handleKv s 2)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            2 1024 256 64 1 4 1 4 i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i) := by
  apply decoding_fused_rotary_embedding_context_v_cache_guarded_store_slice_compute_correct
  intro a b h
  apply Fin.ext
  simp [decodingVCacheGuardedOffset, vCacheGuardedOffset] at h
  omega

/-- Public Python-shape summary for the default 4D cache layout used by
`fused_rotary_embedding.py`: Q and K rotary writebacks plus guarded K/V cache
stores are all exposed as compute-correct outputs. -/
theorem decoding_fused_rotary_embedding_old_layout_python_test_shape_all_outputs_compute_correct
    (q k cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qFirstOffset s 512 64 1 i)))
      (expected := fun i => qFirstSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qSecondOffset s 512 64 1 32 i)))
      (expected := fun i => qSecondSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kFirstOffset s 256 64 1 i))
      (expected := fun i => kFirstSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kSecondOffset s 256 64 1 32 i))
      (expected := fun i => kSecondSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
        (decodingOffsetsInLastBlock s context_lengths 4)
        2 1024 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => handleKv s 2)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            2 1024 256 64 1 4 1 4 i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i)) := by
  constructor
  · exact decoding_fused_rotary_embedding_q_first_half_python_test_shape_compute_correct
      q cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_q_second_half_python_test_shape_compute_correct
      q cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_k_first_half_python_test_shape_compute_correct
      k cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_k_second_half_python_test_shape_compute_correct
      k cos sin s
  constructor
  · exact
      decoding_fused_rotary_embedding_context_k_cache_first_half_old_layout_python_test_shape_compute_correct
        OutK0Pre k_cache BLOCK_TABLES context_lengths s
  constructor
  · exact
      decoding_fused_rotary_embedding_context_k_cache_second_half_old_layout_python_test_shape_compute_correct
        OutK1Pre k_cache BLOCK_TABLES context_lengths s
  · exact decoding_fused_rotary_embedding_context_v_cache_python_test_shape_compute_correct
      LoadedV v_cache BLOCK_TABLES context_lengths s

/-- Public Python-shape summary for the new 5D K-cache layout used by
`fused_rotary_embedding.py`'s second test case: Q and K rotary writebacks plus
guarded K/V cache stores are all exposed as compute-correct outputs. -/
theorem decoding_fused_rotary_embedding_new_layout_python_test_shape_all_outputs_compute_correct
    (q k cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qFirstOffset s 512 64 1 i)))
      (expected := fun i => qFirstSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qSecondOffset s 512 64 1 32 i)))
      (expected := fun i => qSecondSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kFirstOffset s 256 64 1 i))
      (expected := fun i => kFirstSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kSecondOffset s 256 64 1 32 i))
      (expected := fun i => kSecondSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
        (decodingOffsetsInLastBlock s context_lengths 4)
        2 1024 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => handleKv s 2)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            2 1024 256 64 1 4 1 4 i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i)) := by
  constructor
  · exact decoding_fused_rotary_embedding_q_first_half_python_test_shape_compute_correct
      q cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_q_second_half_python_test_shape_compute_correct
      q cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_k_first_half_python_test_shape_compute_correct
      k cos sin s
  constructor
  · exact decoding_fused_rotary_embedding_k_second_half_python_test_shape_compute_correct
      k cos sin s
  constructor
  · exact
      decoding_fused_rotary_embedding_context_k_cache_first_half_new_layout_python_test_shape_compute_correct
        OutK0Pre k_cache BLOCK_TABLES context_lengths s
  constructor
  · exact
      decoding_fused_rotary_embedding_context_k_cache_second_half_new_layout_python_test_shape_compute_correct
        OutK1Pre k_cache BLOCK_TABLES context_lengths s
  · exact decoding_fused_rotary_embedding_context_v_cache_python_test_shape_compute_correct
      LoadedV v_cache BLOCK_TABLES context_lengths s

/-- Public Python-shape summary for test case 1 of
`decoding_fused_rotary_embedding`: default 4D K-cache layout
`[num_blocks, kv_head_num, block_size, head_dim]`.

The surface conjunct pins the faithful full kernel launch for
`total_tokens = 16`, `q_head_num = 8`, `kv_head_num = 4`, `head_dim = 64`,
`block_size = 4`, and `KV_GROUP_NUM = 2`; the output conjunct reuses the
checked Q/K rotary writebacks plus guarded K/V cache stores. -/
theorem decoding_fused_rotary_embedding_old_layout_python_test_shape_output_summary
    (q k v cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (s : BlockState) :
    (∃ alg, (decoding_fused_rotary_embedding_kernel_surface q k v cos sin
      k_cache v_cache BLOCK_TABLES context_lengths
      64 512 64 256 64 1 64 1 1024 256 0 64 1 1024 256 64 1 4 1 4 2 64
      ).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qFirstOffset s 512 64 1 i)))
      (expected := fun i => qFirstSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qSecondOffset s 512 64 1 32 i)))
      (expected := fun i => qSecondSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kFirstOffset s 256 64 1 i))
      (expected := fun i => kFirstSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kSecondOffset s 256 64 1 32 i))
      (expected := fun i => kSecondSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 64 1024 256 0 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 64 1024 256 0 64 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
        (decodingOffsetsInLastBlock s context_lengths 4)
        2 1024 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => handleKv s 2)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            2 1024 256 64 1 4 1 4 i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i)) := by
  constructor
  · exact decoding_fused_rotary_embedding_kernel_surface_toAlgorithm_supported
      q k v cos sin k_cache v_cache BLOCK_TABLES context_lengths
      64 512 64 256 64 1 64 1 1024 256 0 64 1 1024 256 64 1 4 1 4 2 64
  · exact
      decoding_fused_rotary_embedding_old_layout_python_test_shape_all_outputs_compute_correct
        q k cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
        context_lengths s

/-- Public Python-shape summary for test case 2 of
`decoding_fused_rotary_embedding`: new 5D K-cache layout
`[num_blocks, kv_head_num, head_dim // 16, block_size, 16]`.

Compared with the old layout, this pins `x = 16` and the split K-cache strides
`kcsplit_x_stride = 64`, `kcs_stride = 16`, `kcd_stride = 1`, while preserving
the same Q/K/V/cos/sin and V-cache strides from the Python test. -/
theorem decoding_fused_rotary_embedding_new_layout_python_test_shape_output_summary
    (q k v cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
      context_lengths : RegionName)
    (s : BlockState) :
    (∃ alg, (decoding_fused_rotary_embedding_kernel_surface q k v cos sin
      k_cache v_cache BLOCK_TABLES context_lengths
      16 512 64 256 64 1 64 1 1024 256 64 16 1 1024 256 64 1 4 1 4 2 64
      ).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qFirstOffset s 512 64 1 i)))
      (expected := fun i => qFirstSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_second_half q cos sin
        512 64 1 64 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin 32 => True)
        (fun i => (q, qSecondOffset s 512 64 1 32 i)))
      (expected := fun i => qSecondSpec s q cos sin 512 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_first_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kFirstOffset s 256 64 1 i))
      (expected := fun i => kFirstSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_k_second_half k cos sin
        256 64 1 64 1 32)
      (initialState := s)
      (write := fun i : Fin 32 => some (k, kSecondOffset s 256 64 1 32 i))
      (expected := fun i => kSecondSpec s k cos sin 256 64 1 64 1 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_first_half_guarded_store_slice
          OutK0Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheFirstGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 i)))
      (expected := fun i => kCacheFirstStoreSpec s OutK0Pre i)) ∧
    (ComputeCorrect.Realizes
      (kernel :=
        decoding_fused_rotary_embedding_k_cache_second_half_guarded_store_slice
          OutK1Pre k_cache
          (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
          (decodingOffsetsInLastBlock s context_lengths 4)
          2 16 1024 256 64 16 1 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 32 => handleKv s 2)
        (fun i => (k_cache,
          decodingKCacheSecondGuardedOffset s BLOCK_TABLES context_lengths
            2 16 1024 256 64 16 1 4 1 4 32 i)))
      (expected := fun i => kCacheSecondStoreSpec s OutK1Pre 32 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_v_cache_guarded_store_slice
        LoadedV v_cache
        (decodingBlockId s BLOCK_TABLES context_lengths 4 1 4)
        (decodingOffsetsInLastBlock s context_lengths 4)
        2 1024 256 64 1 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin 64 => handleKv s 2)
        (fun i => (v_cache,
          decodingVCacheGuardedOffset s BLOCK_TABLES context_lengths
            2 1024 256 64 1 4 1 4 i)))
      (expected := fun i => vCacheStoreSpec s LoadedV i)) := by
  constructor
  · exact decoding_fused_rotary_embedding_kernel_surface_toAlgorithm_supported
      q k v cos sin k_cache v_cache BLOCK_TABLES context_lengths
      16 512 64 256 64 1 64 1 1024 256 64 16 1 1024 256 64 1 4 1 4 2 64
  · exact
      decoding_fused_rotary_embedding_new_layout_python_test_shape_all_outputs_compute_correct
        q k cos sin OutK0Pre OutK1Pre LoadedV k_cache v_cache BLOCK_TABLES
        context_lengths s

end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding

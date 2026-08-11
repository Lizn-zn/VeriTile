import VeriTile.Triton

/-!
# `chunk_retention` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_retention_fwd_kernel_h` (the file's first kernel), the output
forward `fwd_kernel_o`, the state-recurrence backward `bwd_kernel_dh`, and the
fused gradient backward `bwd_kernel_dqkv`. This file covers the two
**state-recurrence** kernels — the retention (decayed) siblings of the ported
`chunk_linear_attn` pair:

```
fwd_h:  h[·,·,t] = H_t,  H_0 = (h0 or 0),
        H_{t+1} = d_b(t)·H_t + k_tᵀ · (v_t ⊙ d_i(t))
bwd_dh: one accumulation loop over chunks (descending), then a single store
        of d_b · Σ_t (do_t ⊙ d_i)ᵀ… — see the faithfulness notes below.
```

with the per-head decay `b_b = log2(1 - 2^(-5 - i_h))` and per-chunk factors
`d_b(t) = 2^(len(t)·b_b)`, `d_i(t)[c] = 2^((len(t) - c - 1)·b_b)`, where
`len(t)` is `T % BT` on a ragged last chunk and `BT` otherwise (the kernel
rebinds `d_b`/`d_i` inside the loop on that chunk).

Translation-surface blocker: the decay prologue's implicit int→float
promotions (`i_h * 1.0`, `BT * b_b`, `(BT - o_i - 1) * b_b`, …) have no
implicit-coercion analogue in the shape/dtype-typed DSL and are spelled with
the explicit nat→real cast `tl.toReal(...)` (the `rbe_triton_transform`
precedent), so the `tl.*` call set differs from the Python surface by exactly
these casts.

## Faithfulness notes (both honest, both invisible to the stores)

1. **Nat-truncated ragged decay tail.** On the ragged last chunk the kernel
   computes `(T % BT) - o_i - 1` on an *integer* tile, which goes negative for
   lanes `o_i ≥ T % BT`; the DSL's `.nat` channel truncates those lanes at
   `0`, so the ported `d_i` register holds `2^0·…` where the CUDA kernel holds
   a positive power. The divergence is unobservable through every store: those
   lanes multiply `v` rows at `t·BT + o_i ≥ T`, which the block-pointer
   boundary check already zeroes.
2. **`bwd_kernel_dh` is only compilable square.** Its accumulation
   `b_dh += tl.dot(b_o ⊙ d_i, b_v)` multiplies two `[BT, BV]` tiles into a
   `[BK, BV]` accumulator, which Triton itself only compiles when
   `BK = BT = BV`; the surface binds that single shared value as `BT` where
   the Python spells `BK`/`BV`. Its final store also reads the loop variable
   `i_t` *after* the loop (Python leaves `i_t = 0`; the ascending respelling
   leaves the same value), which requires `0 < NT` — with `NT = 0` the Python
   kernel would `NameError` on the same line; because the DSL scopes loop-body
   names, the surface pre-initializes `i_t = 0` before the loop — dead for
   `NT ≥ 1`, where the last iteration writes the same value. The loads of
   `dh` inside its loop are dead (`b_h` is never used) and are transcribed
   anyway; its dead signature params (`q`, `s_qk_h`/`s_qk_t`/`s_qk_d`,
   `scale` — never read) are omitted from the surface binders (the
   `fused_recurrent_retention` precedent).

## The descending loop, spelled ascending

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`; the surface
spells the identical iteration sequence as `for j in range(0, NT)` with
`i_t = NT - 1 - j` as the body's first statement — the established respelling
(`chunk_linear_attn`, `triton_linear_activation`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the
3-D grid `(NK, NV, B*H)` and the host-computed `NT`) is the *trusted
boundary*. Every dimension and stride stays a symbolic parameter (forward
kernel), as do the two `constexpr` gates. The `.to(...)` dtype round-trips
erase to the identity at the algorithm layer. Spelling notes, per
`bench/MAIN_THEOREM_CONVENTIONS.md`: integer literals inside index arithmetic
are written `$(n)`; `boundary_check=(0, 1)` is written
`boundary_check=([0, 1] : List Nat)`; the tuple assignments
(`i_k, i_v, i_bh = …`, `d_b, d_i = …`) are split into one statement each.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkRetention

open VeriTile.Triton

section Correct_without_Rounding

/-- Faithful transcription of `chunk_retention_fwd_kernel_h`. -/
def crh_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_i = tl.arange(0, $(BT))
  d_b = tl.math.exp2(tl.toReal($(BT)) * b_b)
  d_i = tl.math.exp2(tl.toReal($(BT) - o_i - $(1)) * b_b)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=h0 + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
      d_b = tl.math.exp2(tl.toReal($(T) % $(BT)) * b_b)
      d_i = tl.math.exp2(tl.toReal($(T) % $(BT) - o_i - $(1)) * b_b)
    }
    b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The forward state surface lowers to the algorithm layer. -/
theorem crh_fwd_h_surface_toAlgorithm_supported
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (crh_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
      s_vo_d s_h_h s_h_t H T K V BT BK BV NT
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [crh_fwd_h_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_retention_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable, and the single
shared block size (forced by its `tl.dot` shapes — see the preamble) bound as
`BT` where the Python spells `BK`/`BV`. -/
def crh_bwd_dh_surface
    (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_i = tl.arange(0, $(BT))
  d_b = tl.math.exp2(tl.toReal($(BT)) * b_b)
  d_i = tl.math.exp2(tl.toReal(o_i + $(1)) * b_b)
  b_dh = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  i_t = $(0)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_o = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    p_h = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BT), i_v * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
    b_o = tl.load(p_o, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot((b_o * d_i[:, None]).to(b_o.dtype), b_v, allow_tf32=false)
  }
  b_dh *= d_b
  p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_k * $(K) * $(V),
    shape=($(K), $(V)), strides=($(s_h_t), $(1)),
    offsets=(i_v * $(BT), i_t * $(BT)), block_shape=($(BT), $(BT)), order=(1, 0))
  tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The backward state surface lowers to the algorithm layer. -/
theorem crh_bwd_dh_surface_toAlgorithm_supported
    (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) :
    ∃ alg, (crh_bwd_dh_surface v do_ dh s_vo_h s_vo_t
      s_vo_d s_h_h s_h_t H T K V BT NT).toAlgorithm? = Except.ok alg := by
  simp [crh_bwd_dh_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkRetention

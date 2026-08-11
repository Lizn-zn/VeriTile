import VeriTile.Triton

/-!
# `chunk_linear_attn` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_linear_attn_fwd_kernel_h`, the output forward
`chunk_linear_attn_fwd_kernel_o`, the state-recurrence backward
`chunk_linear_attn_bwd_kernel_dh`, and the fused gradient backward
`chunk_linear_attn_bwd_kernel_dqkv`. This file covers the two **state-recurrence**
kernels — the file's first kernel, `chunk_linear_attn.py`'s
`chunk_linear_attn_fwd_kernel_h`, and its descending mirror `bwd_kernel_dh` —
which share one proof skeleton: a `[BK, BV]` running state is **stored to memory
at every chunk** and then advanced by one `tl.dot`.

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 or 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (and ht = H_NT when STORE_FINAL_STATE)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t) · do_t
```

The store-then-accumulate order matters and is carried by the spec: chunk `t`
receives the state *before* chunk `t`'s own contribution.

## The descending loop, spelled ascending

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`. The DSL's `forRange`
counts up (`stepForRangeAux` advances while `cur < stop`), so the faithful surface
spells the change of variable explicitly:

    for j in range(0, NT, 1) { i_t = NT - $(1) - j; … }

— the identical iteration sequence, with the descending index computed from the
ascending counter as the body's first statement. This is the established
respelling (`triton_linear_activation`'s "descending K-loop as antiquoted
ascending trip count", `completion_audit.md`), and it is observable in the
surface rather than hidden in a spec.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the 3-D
grid `(NK, NV, B*H)` and the host-computed `NT`) is the *trusted boundary*. Every
dimension and stride stays a symbolic parameter, as do the two `constexpr` gates
`USE_INITIAL_STATE` / `STORE_FINAL_STATE` — one theorem covers all four
configurations. The `.to(...)` dtype round-trips erase to the identity at the
algorithm layer.

Spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, all surface syntax
rather than semantics: integer literals inside index arithmetic are written
`$(n)`; `boundary_check=(0, 1)` is written `boundary_check=([0, 1] : List Nat)`.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn

open VeriTile.Triton

section Correct_without_Rounding

/-- Faithful transcription of `chunk_linear_attn_fwd_kernel_h`. -/
def cla_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
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
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The forward state surface lowers to the algorithm layer. -/
theorem cla_fwd_h_surface_toAlgorithm_supported
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t T K V BT BK BV NT
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [cla_fwd_h_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_linear_attn_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the preamble). -/
def cla_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot(b_q, (b_do).to(b_q.dtype), allow_tf32=false)
  }
}

/-- The backward state surface lowers to the algorithm layer. -/
theorem cla_bwd_dh_surface_toAlgorithm_supported
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ∃ alg, (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? = Except.ok alg := by
  simp [cla_bwd_dh_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn

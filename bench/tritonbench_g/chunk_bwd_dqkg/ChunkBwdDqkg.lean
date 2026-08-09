import VeriTile.Triton

/-!
# `chunk_bwd_dqkg` — strict per-kernel correctness

`chunk_bwd_dqkg.py`'s `chunk_simple_gla_bwd_kernel_dqkg` is the simple-GLA
chunked **backward** for the three gradients `dq`, `dk` and `dg`. It is the
backward partner of the already-ported `chunk_gla_simple` forward and shares its
gate vocabulary: a per-row log-decay `g`, a chunk-local causal mask
`o_i[:, None] >= o_i[None, :]`, and the score decay `exp(g[r] - g[r'])`.

One program owns the tile `(i_k, i_t, i_bh)` = (key block, time chunk,
batch*head). It streams the value axis in `ceil(V / BV)` steps, accumulating four
quantities, then applies the gate decays, the causal mask and two more
contractions, and stores `dq`, `dk` and a per-row `dg`.

## Scope

The value-axis loop is stated for the **single value-block regime** `V <= BV`,
i.e. `ceil(V / BV) = 1`, which is the launcher's own regime: it sets
`BV = min(next_power_of_2(V), 64)`, so `V <= BV` holds exactly when `V <= 64`
(the checked shape has `V = 64`). That is the same kind of documented,
launcher-consistent narrowing as `chunk_delta_fwd`'s `BC = BT`, and it is stated
as an explicit hypothesis rather than baked into a literal - every dimension and
stride stays symbolic.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and the
host launch (the 3-D grid `(NK, NT, B*H)`, the host-computed `BT/BK/BV/NT`, and
the `dg` pre-fill with `-1e9`) are the *trusted boundary*. The `.to(...)` dtype
round-trips erase to the identity at the algorithm layer.

Two mechanical transcription notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`:

* `b_dg_last` is a `tl.zeros([1,], ...)` one-lane tile in the source, used only
  ever as a scalar (`+=` a full reduction, `*=` a scalar, and broadcast into a
  `[BT]` add). It is modelled as a `[]`-shaped scalar tile - same values, one
  fewer index to carry.
* Integer literals inside index arithmetic are written `$(1)` rather than `1`.
  A bare literal is inferred `.real` by the DSL's expression typing, so
  `min(...) - 1` does not elaborate while `min(...) - $(1)` is the intended
  `.nat` reading. Spelling, not semantics.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription)

The source's `do` parameter is spelled `do_` (`do` is Lean syntax). The
`(V, NT*K)` logical shape of the `h` / `dh` block pointers is antiquoted as a
single `$(NT * K)`: the DSL requires each `tl.make_block_ptr` shape entry to be a
literal or one antiquote, and `NT`/`K` are both `tl.constexpr`, so the product is
a compile-time constant either way. -/
def chunk_bwd_dqkg_surface
    (q k v h g do_ dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  n_bh = tl.num_programs(2)
  o_i = tl.arange(0, $(BT))
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  last_idx = min(i_t * $(BT) + $(BT), $(T)) - $(1)
  b_g_last = tl.load(g + i_bh * $(T) + last_idx)
  b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
  b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
  b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  b_dg_last = tl.zeros([], dtype=tl.float32)
  b_dg = tl.zeros([$(BT)], dtype=tl.float32)
  for i_v in range($(0), tl.cdiv($(V), $(BV)), $(1)) {
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h),
      shape=($(V), $(NT * K)), strides=($(1), $(s_h_t)),
      offsets=(i_v * $(BV), i_t * $(K) + i_k * $(BK)),
      block_shape=($(BV), $(BK)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_v_h),
      shape=($(T), $(V)), strides=($(s_v_t), $(1)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h),
      shape=($(V), $(NT * K)), strides=($(1), $(s_h_t)),
      offsets=(i_v * $(BV), i_t * $(K) + i_k * $(BK)),
      block_shape=($(BV), $(BK)), order=(0, 1))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat))
    b_dg_last += tl.sum(b_h * b_dh)
    b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=false)
    b_dq += tl.dot(b_do, (b_h).to(b_do.dtype), allow_tf32=false)
    b_dk += tl.dot(b_v, (b_dh).to(b_v.dtype), allow_tf32=false)
  }
  p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(1)),
    offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(1)),
    offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
  b_dg_last = b_dg_last * tl.exp(b_g_last)
  b_dq = b_dq * tl.exp(b_g)[:, None] * $(scale)
  b_dk = b_dk * tl.exp(0.0 - b_g + b_g_last)[:, None]
  b_dg_last += tl.sum(b_dk * b_k)
  b_ds = tl.where(o_i[:, None] >= o_i[None, :],
    b_ds * $(scale) * tl.exp(b_g[:, None] - b_g[None, :]), 0.0)
  b_dq += tl.dot((b_ds).to(b_k.dtype), b_k, allow_tf32=false)
  b_dk += tl.dot(tl.trans((b_ds).to(b_k.dtype)), b_q, allow_tf32=false)
  b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
  b_dg = tl.where(o_i < min($(BT), $(T) - i_t * $(BT)) - $(1), b_dg, b_dg + b_dg_last)
  p_dq = tl.make_block_ptr(base=dq + i_bh * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(1)),
    offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
  p_dk = tl.make_block_ptr(base=dk + i_bh * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(1)),
    offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
  p_dg = tl.make_block_ptr(base=dg + (i_k * n_bh + i_bh) * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  tl.store(p_dq, (b_dq).to(p_dq.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dg, (b_dg).to(p_dg.dtype.element_ty), boundary_check=([0] : List Nat))
}

theorem chunk_bwd_dqkg_surface_toAlgorithm_supported
    (q k v h g do_ dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ∃ alg, (chunk_bwd_dqkg_surface q k v h g do_ dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm?
        = Except.ok alg := by
  simp [chunk_bwd_dqkg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]



/-! ## Masked element accessors

Each accessor is the kernel's own block-pointer address arithmetic, guarded by
the `boundary_check` its load carries: an off-region lane reads `0`, which is what
`tl.load(..., boundary_check=...)` yields. The program's ids are read off the
launch state, so `i_k = s.pids 0`, `i_t = s.pids 1`, `i_bh = s.pids 2`. -/

/-- `b_g[r]` — the per-row log decay of this time chunk. -/
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (r : Nat) : ℝ :=
  if s.pids 1 * BT + r < T then s.readMem g (s.pids 2 * T + s.pids 1 * BT + r) else 0

/-- `b_g_last` — the decay of this chunk's last in-range row,
`last_idx = min(i_t*BT + BT, T) - 1`. Unmasked in the source: a raw pointer load. -/
noncomputable def gLastElem (s : BlockState) (g : RegionName) (T BT : Nat) : ℝ :=
  s.readMem g (s.pids 2 * T + (min (s.pids 1 * BT + BT) T - 1))

/-- `b_v[r, c]` / `b_do[r, c]` — block ptr `(T,V)`, strides `(s_v_t, 1)`,
offsets `(i_t*BT, 0)` at the single value block. -/
noncomputable def voElem (s : BlockState) (rg : RegionName)
    (s_v_h s_v_t T V BT : Nat) (r c : Nat) : ℝ :=
  if s.pids 1 * BT + r < T ∧ c < V then
    s.readMem rg (s.pids 2 * s_v_h + (s.pids 1 * BT + r) * s_v_t + c) else 0

/-- `b_h[a, e]` / `b_dh[a, e]` — block ptr `(V, NT*K)`, strides `(1, s_h_t)`,
offsets `(0, i_t*K + i_k*BK)`. -/
noncomputable def hElem (s : BlockState) (rg : RegionName)
    (s_h_h s_h_t K V BK NT : Nat) (a e : Nat) : ℝ :=
  if a < V ∧ s.pids 1 * K + s.pids 0 * BK + e < NT * K then
    s.readMem rg (s.pids 2 * s_h_h + a + (s.pids 1 * K + s.pids 0 * BK + e) * s_h_t)
  else 0

/-- `b_q[r, e]` / `b_k[r, e]` — block ptr `(T,K)`, strides `(s_k_t, 1)`,
offsets `(i_t*BT, i_k*BK)`. -/
noncomputable def qkElem (s : BlockState) (rg : RegionName)
    (s_k_h s_k_t T K BT BK : Nat) (r e : Nat) : ℝ :=
  if s.pids 1 * BT + r < T ∧ s.pids 0 * BK + e < K then
    s.readMem rg (s.pids 2 * s_k_h + (s.pids 1 * BT + r) * s_k_t + s.pids 0 * BK + e)
  else 0

/-! ## The four value-axis accumulators, at the single block `i_v = 0`

`V <= BV` makes `ceil(V/BV) = 1`, so each accumulator is one term rather than a
fold. The contraction ranges are the *block* extents `BV` / `BT`, with
out-of-region lanes contributing `0` through the accessors above — exactly the
`tl.dot` the kernel issues. -/

/-- `b_dg_last` after the loop: `tl.sum(b_h * b_dh)`, a full reduction. -/
noncomputable def dgLastAcc (s : BlockState) (h dh : RegionName)
    (s_h_h s_h_t K V BK BV NT : Nat) : ℝ :=
  Finset.univ.sum fun a : Fin BV => Finset.univ.sum fun e : Fin BK =>
    hElem s h s_h_h s_h_t K V BK NT a.val e.val
      * hElem s dh s_h_h s_h_t K V BK NT a.val e.val

/-- `b_ds[r, r']` after the loop: `tl.dot(b_do, tl.trans(b_v))`. -/
noncomputable def dsAcc (s : BlockState) (v do_ : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) (r r' : Nat) : ℝ :=
  Finset.univ.sum fun c : Fin BV =>
    voElem s do_ s_v_h s_v_t T V BT r c.val * voElem s v s_v_h s_v_t T V BT r' c.val

/-- `b_dq[r, e]` after the loop: `tl.dot(b_do, b_h)`. -/
noncomputable def dqAcc (s : BlockState) (h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  Finset.univ.sum fun a : Fin BV =>
    voElem s do_ s_v_h s_v_t T V BT r a.val
      * hElem s h s_h_h s_h_t K V BK NT a.val e

/-- `b_dk[r, e]` after the loop: `tl.dot(b_v, b_dh)`. -/
noncomputable def dkAcc (s : BlockState) (v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  Finset.univ.sum fun a : Fin BV =>
    voElem s v s_v_h s_v_t T V BT r a.val
      * hElem s dh s_h_h s_h_t K V BK NT a.val e

/-! ## The post-loop chain

Order matters and is preserved: `b_dg_last` picks up its `b_dk * b_k` reduction
**before** `b_dk` receives the `dot(trans(b_ds), b_q)` term, so the two uses of
`b_dk` are at different stages of the computation. -/

/-- `b_dq` after the gate decay and scale, before the `b_ds` contraction. -/
noncomputable def dqGated (s : BlockState) (g h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  dqAcc s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r e
    * Real.exp (gElem s g T BT r) * scale

/-- `b_dk` after its gate decay, before the `b_ds` contraction — the value the
`b_dg_last` reduction sees. -/
noncomputable def dkGated (s : BlockState) (g v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  dkAcc s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r e
    * Real.exp (0 - gElem s g T BT r + gLastElem s g T BT)

/-- `b_ds` after the causal mask and score decay. -/
noncomputable def dsMasked (s : BlockState) (g v do_ : RegionName)
    (s_v_h s_v_t : Nat) (scale : ℝ) (T V BT BV : Nat) (r r' : Nat) : ℝ :=
  if r' ≤ r then
    dsAcc s v do_ s_v_h s_v_t T V BT BV r r' * scale
      * Real.exp (gElem s g T BT r - gElem s g T BT r')
  else 0

/-- `b_dg_last` at the point the per-row `dg` consumes it: the gated `h·dh`
reduction plus the `b_dk * b_k` reduction taken at the pre-contraction `b_dk`. -/
noncomputable def dgLastFinal (s : BlockState) (g k v h dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : ℝ :=
  dgLastAcc s h dh s_h_h s_h_t K V BK BV NT * Real.exp (gLastElem s g T BT)
    + Finset.univ.sum fun r : Fin BT => Finset.univ.sum fun e : Fin BK =>
        dkGated s g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r.val e.val
          * qkElem s k s_k_h s_k_t T K BT BK r.val e.val

/-! ## The three stored gradients -/

/-- `dq[r, e]` — the gated accumulator plus the masked-score contraction. -/
noncomputable def dqSpec (s : BlockState) (g k v h do_ : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  dqGated s g h do_ s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT r e
    + Finset.univ.sum fun r' : Fin BT =>
        dsMasked s g v do_ s_v_h s_v_t scale T V BT BV r r'.val
          * qkElem s k s_k_h s_k_t T K BT BK r'.val e

/-- `dk[r, e]` — the gated accumulator plus the transposed contraction. -/
noncomputable def dkSpec (s : BlockState) (g q v do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (r e : Nat) : ℝ :=
  dkGated s g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r e
    + Finset.univ.sum fun r' : Fin BT =>
        dsMasked s g v do_ s_v_h s_v_t scale T V BT BV r'.val r
          * qkElem s q s_k_h s_k_t T K BT BK r'.val e

/-- `dg[r]` — the row reduction of `q*dq - k*dk` over the key block, with the
chunk's last in-range row additionally receiving `b_dg_last`. -/
noncomputable def dgSpec (s : BlockState) (g q k v h do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (r : Nat) : ℝ :=
  (Finset.univ.sum fun e : Fin BK =>
      qkElem s q s_k_h s_k_t T K BT BK r e.val
          * dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
              T K V BT BK BV NT r e.val
        - qkElem s k s_k_h s_k_t T K BT BK r e.val
          * dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
              T K V BT BK BV NT r e.val)
    + (if r < min BT (T - s.pids 1 * BT) - 1 then 0
       else dgLastFinal s g k v h dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
              T K V BT BK BV NT)

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg

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


/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`.
Two lowerings worth naming because they are not guessable from the source text:
`min(a, b)` becomes `Op.where (Op.lt …) a b` (there is no `Op.min`), and an
axis-less `tl.sum` on a rank-2 tile drops the **last** remaining axis each time,
so it is `reduceSum ⟨0,_⟩ (reduceSum ⟨1,_⟩ ·)`. -/

/-- `tl.cdiv(V, BV)` — the value-axis loop's trip count. -/
def cbdStopOp (V BV : Nat) : Op .nat [] :=
  Op.div .nat Broadcast.nil
    (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV)) (Op.constNat 1))
    (Op.constNat BV)

/-- The compiled value-axis loop body: four block-pointer constructions, four
masked loads, and the four accumulations. -/
def cbdLoopBody (v h do_ dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BV, BK] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h)) [V, NT * K]
        [BV, BK] [1, s_h_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV),
          Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))]),
    Stmt.assign .blockPtr [BT, BV] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BV, BK] "p_dh"
      (Op.makeBlockPtrDynOffsets dh
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h)) [V, NT * K]
        [BV, BK] [1, s_h_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV),
          Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))]),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1]) .none),
    Stmt.assign .real [BT, BV] "b_do"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BV] "p_do") [0, 1]) .none),
    Stmt.assign .real [BV, BK] "b_h"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BV, BK] "p_h") [0, 1]) .none),
    Stmt.assign .real [BV, BK] "b_dh"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BV, BK] "p_dh") [0, 1]) .none),
    Stmt.assign .real [] "b_dg_last"
      (Op.add .real Broadcast.nil (Op.ref .real [] "b_dg_last")
        (Op.reduceSum ⟨0, by simp [TileShape.eraseAxis]⟩ Bool.false
          (Op.reduceSum ⟨1, by simp⟩ Bool.false
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BV, BK] "b_h") (Op.ref .real [BV, BK] "b_dh"))))),
    Stmt.assign .real [BT, BT] "b_ds"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_ds")
        (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_do")
          (Op.transpose (Op.ref .real [BT, BV] "b_v")))),
    Stmt.assign .real [BT, BK] "b_dq"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dq")
        (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_do")
          (Op.ref .real [BV, BK] "b_h"))),
    Stmt.assign .real [BT, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dk")
        (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_v")
          (Op.ref .real [BV, BK] "b_dh"))) ]


/-- The compiled prologue: the four ids, `o_i`, the `g` block pointer and its
load, `last_idx` / `b_g_last`, and the five zero-initialised accumulators. -/
def cbdPreLoop (g : RegionName) (T BT BK : Nat) : List Stmt :=
[ Stmt.assign .nat [] "i_k" (Op.programId 0),
    Stmt.assign .nat [] "i_t" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .nat [] "n_bh" (Op.numPrograms 2),
    Stmt.assign .nat [BT] "o_i" (Op.arange BT),
    Stmt.assign .blockPtr [BT] "p_g"
      (Op.makeBlockPtrDynOffsets g
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
        [T] [BT] [1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .real [BT] "b_g"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT] "p_g") [0]) .none),
    Stmt.assign .nat [] "last_idx"
      (Op.sub .nat Broadcast.nil
        (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
          (Op.constNat BT))
          (Op.constNat T))
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
          (Op.constNat BT))
        (Op.constNat T))
        (Op.constNat 1)),
    Stmt.assign .real [] "b_g_last"
      (Op.load .real
          (MemAccess.region g
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
          (Op.ref .nat [] "last_idx"))) MaskOpt.none),
    Stmt.assign .real [BT, BK] "b_dq" (Op.full [BT, BK] (Op.const 0)),
    Stmt.assign .real [BT, BK] "b_dk" (Op.full [BT, BK] (Op.const 0)),
    Stmt.assign .real [BT, BT] "b_ds" (Op.full [BT, BT] (Op.const 0)),
    Stmt.assign .real [] "b_dg_last" (Op.full [] (Op.const 0)),
    Stmt.assign .real [BT] "b_dg" (Op.full [BT] (Op.const 0)) ]


/-- The compiled post-loop tail: the `q`/`k` block pointers and loads, the gate
decays, the causal-masked score, the two remaining contractions, the per-row `dg`
reduction with its last-row correction, and the three stores. -/
def cbdPostLoop (q k dq dk dg : RegionName)
    (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BK : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BT, BK] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BT, BK] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .real [BT, BK] "b_k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BK] "p_k") [0, 1]) .none),
    Stmt.assign .real [BT, BK] "b_q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BK] "p_q") [0, 1]) .none),
    Stmt.assign .real [] "b_dg_last"
      (Op.mul .real Broadcast.nil (Op.ref .real [] "b_dg_last")
        (Op.exp (Op.ref .real [] "b_g_last"))),
    Stmt.assign .real [BT, BK] "b_dq"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [BT, BK] "b_dq")
          (Op.expandDim ⟨1, by simp⟩ (Op.exp (Op.ref .real [BT] "b_g"))))
        (Op.const scale)),
    Stmt.assign .real [BT, BK] "b_dk"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dk")
        (Op.expandDim ⟨1, by simp⟩
          (Op.exp (Op.add .real Broadcast.scalarR
            (Op.sub .real Broadcast.scalarL (Op.const 0.0) (Op.ref .real [BT] "b_g"))
            (Op.ref .real [] "b_g_last"))))),
    Stmt.assign .real [] "b_dg_last"
      (Op.add .real Broadcast.nil (Op.ref .real [] "b_dg_last")
        (Op.reduceSum ⟨0, by simp [TileShape.eraseAxis]⟩ Bool.false
          (Op.reduceSum ⟨1, by simp⟩ Bool.false
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BT, BK] "b_dk") (Op.ref .real [BT, BK] "b_k"))))),
    Stmt.assign .real [BT, BT] "b_ds"
      (Op.where
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BT] "o_i"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BT] "o_i")))
        (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BT] "b_ds") (Op.const scale))
          (Op.exp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "b_g"))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BT] "b_g")))))
        (Op.broadcast (Op.const 0.0) [BT, BT])),
    Stmt.assign .real [BT, BK] "b_dq"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dq")
        (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_ds") (Op.ref .real [BT, BK] "b_k"))),
    Stmt.assign .real [BT, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dk")
        (Op.dot (batch := []) (Op.transpose (Op.ref .real [BT, BT] "b_ds"))
          (Op.ref .real [BT, BK] "b_q"))),
    Stmt.assign .real [BT] "b_dg"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BT] "b_dg")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.sub .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BT, BK] "b_dq"))
            (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BT, BK] "b_k") (Op.ref .real [BT, BK] "b_dk"))))),
    Stmt.assign .real [BT] "b_dg"
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [BT] "o_i")
          (Op.sub .nat Broadcast.nil
            (Op.where
              (Op.lt ComparableDType.nat Broadcast.nil (Op.constNat BT)
                (Op.sub .nat Broadcast.nil (Op.constNat T)
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))))
              (Op.constNat BT)
              (Op.sub .nat Broadcast.nil (Op.constNat T)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))))
            (Op.constNat 1)))
        (Op.ref .real [BT] "b_dg")
        (Op.add .real Broadcast.scalarR (Op.ref .real [BT] "b_dg")
          (Op.ref .real [] "b_dg_last"))),
    Stmt.assign .blockPtr [BT, BK] "p_dq"
      (Op.makeBlockPtrDynOffsets dq
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BT, BK] "p_dk"
      (Op.makeBlockPtrDynOffsets dk
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BT] "p_dg"
      (Op.makeBlockPtrDynOffsets dg
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.ref .nat [] "n_bh"))
            (Op.ref .nat [] "i_bh"))
          (Op.constNat T))
        [T] [BT] [1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.store .real [BT, BK]
      (.blockPtr (Op.ref .blockPtr [BT, BK] "p_dq") [0, 1])
      (Op.ref .real [BT, BK] "b_dq") .none,
    Stmt.store .real [BT, BK]
      (.blockPtr (Op.ref .blockPtr [BT, BK] "p_dk") [0, 1])
      (Op.ref .real [BT, BK] "b_dk") .none,
    Stmt.store .real [BT]
      (.blockPtr (Op.ref .blockPtr [BT] "p_dg") [0])
      (Op.ref .real [BT] "b_dg") .none ]

set_option maxRecDepth 8000 in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`cbdPreLoop ++ [forRangeDyn "i_v" 0 (cbdStopOp V BV) 1 cbdLoopBody] ++ cbdPostLoop`
— 34 statements, every one checked against the macro output rather than assumed. -/
theorem cbd_body_eq (q k v h g do_ dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat) :
    (chunk_bwd_dqkg_surface q k v h g do_ dh dq dk dg s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel.body
      = cbdPreLoop g T BT BK
        ++ [Stmt.forRangeDyn "i_v" (Op.constNat 0) (cbdStopOp V BV) (Op.constNat 1)
              (cbdLoopBody v h do_ dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)]
        ++ cbdPostLoop q k dq dk dg s_k_h s_k_t scale T K BT BK := by
  rfl



/-! ## Per-statement eval recipes

Private copies, since bench ports never import each other. Each is the shape the
compiled statement list emits, so the step-through can `rw` straight through. -/

/-- `name * c` on a `nat` scalar register. -/
private theorem cbd_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `nameA * nameB` on two `nat` scalar registers (the `dg` row base). -/
private theorem cbd_mulRefRef_eval (t : BlockState) (a b : RegName) (va vb : Nat)
    (ha : t.regs .nat [] a = some (Tile.scalar va))
    (hb : t.regs .nat [] b = some (Tile.scalar vb)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] a) (Op.ref .nat [] b)) t
      = some (Tile.scalar (va * vb)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- 1-D `tl.make_block_ptr` with a computed base and a computed offset. -/
private theorem cbd_mkptr_1d (R : RegionName) (len stride BT : Nat)
    (baseOp offOp : Op .nat []) (t : BlockState) (base off : Nat)
    (hbase : evalOp baseOp t = some (Tile.scalar base))
    (hoff : evalOp offOp t = some (Tile.scalar off)) :
    evalOp (Op.makeBlockPtrDynOffsets R baseOp [len] [BT] [stride] [offOp]) t
      = some (⟨fun _ : TileIndex [BT] =>
          { region := R, baseOffset := base, parentShape := [len],
            blockShape := [BT], strides := [stride], offsets := [off] }⟩
          : Tile .blockPtr [BT]) := by
  rw [makeBlockPtr2_eval]
  simp only [hbase, hoff, List.mapM_cons, List.mapM_nil, Option.bind_some,
    Option.pure_def, Option.bind_eq_bind, Tile.scalar_data]

/-- 2-D `tl.make_block_ptr` with a computed base and two computed offsets. -/
private theorem cbd_mkptr_2d (R : RegionName) (d0 d1 st0 st1 B0 B1 : Nat)
    (baseOp o0 o1 : Op .nat []) (t : BlockState) (base f0 f1 : Nat)
    (hbase : evalOp baseOp t = some (Tile.scalar base))
    (h0 : evalOp o0 t = some (Tile.scalar f0))
    (h1 : evalOp o1 t = some (Tile.scalar f1)) :
    evalOp (Op.makeBlockPtrDynOffsets R baseOp [d0, d1] [B0, B1] [st0, st1] [o0, o1]) t
      = some (⟨fun _ : TileIndex [B0, B1] =>
          { region := R, baseOffset := base, parentShape := [d0, d1],
            blockShape := [B0, B1], strides := [st0, st1], offsets := [f0, f1] }⟩
          : Tile .blockPtr [B0, B1]) := by
  rw [makeBlockPtr2_eval]
  simp only [hbase, h0, h1, List.mapM_cons, List.mapM_nil, Option.bind_some,
    Option.pure_def, Option.bind_eq_bind, Tile.scalar_data]

/-- Boundary-checked 1-D block-pointer load: in-region lanes read
`base + (off + i)*stride`, the rest read `0`. -/
private theorem cbd_load_1d (R : RegionName) (len stride BT base off : Nat)
    (bpName : RegName) (t : BlockState)
    (hbp : t.regs .blockPtr [BT] bpName = some (⟨fun _ : TileIndex [BT] =>
        { region := R, baseOffset := base, parentShape := [len],
          blockShape := [BT], strides := [stride], offsets := [off] }⟩ :
        Tile .blockPtr [BT])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT] bpName) [0]) MaskOpt.none) t
      = some (⟨fun idx : TileIndex [BT] =>
          if off + idx.1.val < len then
            some (t.readMem R (base + (off + idx.1.val) * stride))
          else some 0⟩ : Tile .real [BT]) := by
  simp only [evalOp, evalOp_ref, hbp]
  refine congrArg some ?_
  congr 1
  funext idx
  obtain ⟨i1, u⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d,
    BlockState.readMemValue_real, decide_eq_true_eq]
  by_cases hh : off + i1.val < len
  · simp [hh]
  · simp [hh, BlockState.defaultCarrier]


/-- `min(a, b)` on `nat` scalars. The DSL lowers `min` to
`Op.where (Op.lt …) a b` (there is no `Op.min`), so this is the bridge from that
shape to `Nat.min`. -/
private theorem cbd_min_eval (t : BlockState) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a t = some (Tile.scalar va))
    (hb : evalOp b t = some (Tile.scalar vb)) :
    evalOp (Op.where (Op.lt ComparableDType.nat Broadcast.nil a b) a b) t
      = some (Tile.scalar (min va vb)) := by
  rw [evalOp_where, evalOp_lt]
  simp only [ha, hb, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  intro z
  simp only [Tile.select, Tile.cop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  by_cases hlt : va < vb
  · simp [hlt, Nat.le_of_lt hlt]
  · simp [hlt, Nat.not_lt.mp hlt]


/-- `a + b` on `nat` scalars. -/
private theorem cbd_add_eval (t : BlockState) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a t = some (Tile.scalar va))
    (hb : evalOp b t = some (Tile.scalar vb)) :
    evalOp (Op.add .nat Broadcast.nil a b) t = some (Tile.scalar (va + vb)) := by
  rw [evalOp_add]
  simp only [ha, hb, Option.bind_some, Option.bind_eq_bind]
  rfl

/-- `a - b` on `nat` scalars (truncated, as `Nat` subtraction). -/
private theorem cbd_sub_eval (t : BlockState) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a t = some (Tile.scalar va))
    (hb : evalOp b t = some (Tile.scalar vb)) :
    evalOp (Op.sub .nat Broadcast.nil a b) t = some (Tile.scalar (va - vb)) := by
  rw [evalOp_sub]
  simp only [ha, hb, Option.bind_some, Option.bind_eq_bind]
  rfl


/-! ## Prologue execution

The loaded tiles are named exactly as the eval recipes emit them (so the step
chain closes by `rw`); the bridges to the `gElem` / `gLastElem` accessors are
separate, below. -/

/-- `b_g` as `cbd_load_1d` emits it. -/
noncomputable def cbdBgTile (s : BlockState) (g : RegionName) (T BT : Nat) :
    Tile .real [BT] :=
  ⟨fun idx => if s.pids 1 * BT + idx.1.val < T then
      some (s.readMem g (s.pids 2 * T + (s.pids 1 * BT + idx.1.val) * 1)) else some 0⟩

/-- `b_g_last` as the region-load recipe emits it, after the `.real`
`readMemValue` collapses to `readMem`. -/
noncomputable def cbdBgLastTile (s : BlockState) (g : RegionName) (T BT : Nat) :
    Tile .real [] :=
  ⟨fun _ => some (s.readMem g (s.pids 2 * T + (min (s.pids 1 * BT + BT) T - 1)))⟩

/-- The all-zero tile of a given shape, as `Op.full … (Op.const 0)` emits it. -/
noncomputable def cbdZero (sh : TileShape) : Tile .real sh := ⟨fun _ => some 0⟩

/-- `cbdBgTile` agrees with the `gElem` accessor lane by lane. -/
theorem cbdBgTile_data (s : BlockState) (g : RegionName) (T BT : Nat)
    (i : Fin BT) :
    (cbdBgTile s g T BT).data (i, PUnit.unit) = some (gElem s g T BT i.val) := by
  simp only [cbdBgTile, gElem, Nat.mul_one, Nat.add_assoc]
  split <;> rfl

/-- `cbdBgLastTile` agrees with the `gLastElem` accessor. -/
theorem cbdBgLastTile_data (s : BlockState) (g : RegionName) (T BT : Nat) :
    (cbdBgLastTile s g T BT).data PUnit.unit = some (gLastElem s g T BT) := by
  rfl

/-- `tl.zeros(shape)` — the DSL emits `Op.full shape (Op.const 0)`. -/
private theorem cbd_full_zero_eval (sh : TileShape) (t : BlockState) :
    evalOp (Op.full sh (Op.const 0)) t = some (cbdZero sh) := by
  simp only [evalOp_full, evalOp_const]
  rfl

/-- The `b_g` load, phrased against the *launch* state's memory: the prologue's
earlier assignments do not touch memory, so the tile only depends on `s`. -/
private theorem cbd_load_bg_eq (s t : BlockState) (g : RegionName) (T BT : Nat)
    (hmem : ∀ off, t.readMem g off = s.readMem g off)
    (hbp : t.regs .blockPtr [BT] "p_g" = some (⟨fun _ : TileIndex [BT] =>
        { region := g, baseOffset := s.pids 2 * T, parentShape := [T],
          blockShape := [BT], strides := [1], offsets := [s.pids 1 * BT] }⟩ :
        Tile .blockPtr [BT])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT] "p_g") [0]) MaskOpt.none) t
      = some (cbdBgTile s g T BT) := by
  rw [cbd_load_1d g T 1 BT (s.pids 2 * T) (s.pids 1 * BT) "p_g" t hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [cbdBgTile]
  split <;> simp [hmem]

/-- The `b_g_last` load: a raw region access at `i_bh * T + last_idx`. -/
private theorem cbd_load_glast_eq (s t : BlockState) (g : RegionName) (T BT : Nat)
    (hmem : ∀ off, t.readMem g off = s.readMem g off)
    (hibh : t.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hlast : t.regs .nat [] "last_idx"
        = some (Tile.scalar (min (s.pids 1 * BT + BT) T - 1))) :
    evalOp (Op.load .real
        (MemAccess.region g
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
            (Op.ref .nat [] "last_idx"))) MaskOpt.none) t
      = some (cbdBgLastTile s g T BT) := by
  rw [evalOp_load_region_none,
    cbd_add_eval t _ _ (s.pids 2 * T) (min (s.pids 1 * BT + BT) T - 1)
      (cbd_mulRef_eval t "i_bh" (s.pids 2) T hibh)
      (by rw [evalOp_ref]; exact hlast)]
  refine congrArg some ?_
  apply Tile.ext
  intro _
  simp only [cbdBgLastTile, Region.cast_id, Tile.scalar,
    BlockState.readMemValue_real, hmem]

/-! ### The prologue's exit state

Fourteen statements: the four ids, `o_i`, the `g` block pointer and its load,
`last_idx` / `b_g_last`, and the five zero accumulators. Memory is untouched, so
every register value is phrased against the launch state `s`. -/

set_option maxHeartbeats 1000000 in
/-- **Prologue run.** From a launch state `s`, `cbdPreLoop` reaches a state
carrying the ids, the `arange`, the two `g` tiles and the five zeroed
accumulators, with memory and the grid coordinates unchanged. -/
theorem cbdPreLoop_run (g : RegionName) (T BT BK : Nat) (s : BlockState) :
    ∃ s0, stepStmts (cbdPreLoop g T BT BK) s = some s0
      ∧ s0.pids = s.pids
      ∧ s0.numPids = s.numPids
      ∧ (∀ rg off, s0.readMem rg off = s.readMem rg off)
      ∧ s0.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s0.regs .nat [] "n_bh" = some (Tile.scalar (s.numPids 2))
      ∧ s0.regs .nat [BT] "o_i" = some (Tile.vec (fun i : Fin BT => i.val))
      ∧ s0.regs .real [BT] "b_g" = some (cbdBgTile s g T BT)
      ∧ s0.regs .real [] "b_g_last" = some (cbdBgLastTile s g T BT)
      ∧ s0.regs .real [BT, BK] "b_dq" = some (cbdZero [BT, BK])
      ∧ s0.regs .real [BT, BK] "b_dk" = some (cbdZero [BT, BK])
      ∧ s0.regs .real [BT, BT] "b_ds" = some (cbdZero [BT, BT])
      ∧ s0.regs .real [] "b_dg_last" = some (cbdZero [])
      ∧ s0.regs .real [BT] "b_dg" = some (cbdZero [BT]) := by
  unfold cbdPreLoop
  -- the four grid coordinates
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 2 _))]
  -- o_i = arange(BT)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BT _))]
  -- p_g
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_1d g T 1 BT _ _ _ (s.pids 2 * T) (s.pids 1 * BT)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) T (by simp))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp))))]
  -- b_g
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_bg_eq s _ g T BT (by intro off; simp) (by simp)))]
  -- last_idx = min(i_t*BT + BT, T) - 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_sub_eval _ _ _ (min (s.pids 1 * BT + BT) T) 1
      (cbd_min_eval _ _ _ (s.pids 1 * BT + BT) T
        (cbd_add_eval _ _ _ (s.pids 1 * BT) BT
          (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp))
          (evalOp_constNat BT _))
        (evalOp_constNat T _))
      (evalOp_constNat 1 _)))]
  -- b_g_last
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_glast_eq s _ g T BT (by intro off; simp) (by simp) (by simp)))]
  -- the five zeroed accumulators
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cbd_full_zero_eval [BT, BK] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cbd_full_zero_eval [BT, BK] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cbd_full_zero_eval [BT, BT] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cbd_full_zero_eval [] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cbd_full_zero_eval [BT] _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, by simp, by simp, by intro rg off; simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_⟩ <;> simp

/-! ## The single value-block regime

`0 < V` and `V ≤ BV` make `ceil(V/BV) = 1`, so the value-axis loop runs exactly
once with `i_v = 0`. This is the launcher's regime — it sets
`BV = min(next_power_of_2(V), 64)`, so `V ≤ BV` holds whenever `V ≤ 64`. -/

/-- The compiled `tl.cdiv(V, BV)` trip count is `1` when `0 < V ≤ BV`. -/
theorem cbdStopOp_eval (V BV : Nat) (hV : 0 < V) (hVB : V ≤ BV) (s : BlockState) :
    evalOp (cbdStopOp V BV) s = some (Tile.scalar 1) := by
  have hdiv : (V + BV - 1) / BV = 1 := by
    refine Nat.div_eq_of_lt_le ?_ ?_ <;> omega
  simp only [cbdStopOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  apply Tile.ext
  intro z
  simp only [Tile.bop_data, Tile.scalar, Broadcast.leftIndex,
    NumericDType.div, NumericDType.sub, NumericDType.add, hdiv]

/-- **Loop collapse.** In the single value-block regime the `forRangeDyn` reduces
to one pass of the body with `i_v` pinned to `0`. -/
theorem cbdLoop_collapse (v h do_ dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hV : 0 < V) (hVB : V ≤ BV) (s : BlockState) :
    stepStmt (Stmt.forRangeDyn "i_v" (Op.constNat 0) (cbdStopOp V BV) (Op.constNat 1)
        (cbdLoopBody v h do_ dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)) s
      = stepStmts (cbdLoopBody v h do_ dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)
          (s.setReg "i_v" .nat [] (Tile.scalar 0)) := by
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [evalOp_constNat, cbdStopOp_eval V BV hV hVB s,
    Option.bind_some, Tile.scalar_data]
  rw [stepForRangeAux.step_lt one_ne_zero (by norm_num)]
  cases hb : stepStmts (cbdLoopBody v h do_ dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)
      (s.setReg "i_v" .nat [] (Tile.scalar 0)) with
  | none => simp
  | some s' =>
      simp only [Option.bind_some]
      exact stepForRangeAux.step_ge one_ne_zero (by norm_num)

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg

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
        (Op.reduceSum (shape := [BV]) ⟨0, by simp⟩ Bool.false
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
        (Op.reduceSum (shape := [BT]) ⟨0, by simp⟩ Bool.false
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

/-- `a * c` on a `nat` scalar with a literal. -/
private theorem cbd_mulConst_eval (t : BlockState) (a : Op .nat []) (va c : Nat)
    (ha : evalOp a t = some (Tile.scalar va)) :
    evalOp (Op.mul .nat Broadcast.nil a (Op.constNat c)) t
      = some (Tile.scalar (va * c)) := by
  rw [evalOp_mul]
  simp only [ha, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
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

/-! ## The value-axis loop body's loads

Four block pointers and four boundary-checked loads. The `i_v = 0` pinning shows
up only as the column offset `0 * BV`, which the bridges normalise away. -/

/-- Boundary-checked 2-D block-pointer load: in-region lanes read
`base + (o0 + i)*st0 + (o1 + j)*st1`, the rest read `0`. -/
private theorem cbd_load_2d (R : RegionName) (d0 d1 st0 st1 B0 B1 o0 o1 base : Nat)
    (bpName : RegName) (t : BlockState)
    (hbp : t.regs .blockPtr [B0, B1] bpName = some (⟨fun _ : TileIndex [B0, B1] =>
        { region := R, baseOffset := base, parentShape := [d0, d1],
          blockShape := [B0, B1], strides := [st0, st1], offsets := [o0, o1] }⟩ :
        Tile .blockPtr [B0, B1])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [B0, B1] bpName) [0, 1]) MaskOpt.none) t
      = some (⟨fun idx : TileIndex [B0, B1] =>
          if o0 + idx.1.val < d0 ∧ o1 + idx.2.1.val < d1 then
            some (t.readMem R
              (base + (o0 + idx.1.val) * st0 + (o1 + idx.2.1.val) * st1))
          else some 0⟩ : Tile .real [B0, B1]) := by
  simp only [evalOp, evalOp_ref, hbp]
  refine congrArg some ?_
  congr 1
  funext idx
  obtain ⟨i0, i1, u⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets, BlockState.readMemValue_real, decide_eq_true_eq]
  by_cases hh : o0 + i0.val < d0 ∧ o1 + i1.val < d1
  · simp [hh]
  · simp [hh, BlockState.defaultCarrier]

/-- `b_v` / `b_do` — the `(T, V)` value tile of this time chunk. -/
noncomputable def cbdVoTile (s : BlockState) (rg : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) : Tile .real [BT, BV] :=
  Tile.mat fun r c => some (voElem s rg s_v_h s_v_t T V BT r.val c.val)

/-- `b_h` / `b_dh` — the `(V, NT*K)` state tile of this `(i_t, i_k)` block. -/
noncomputable def cbdHTile (s : BlockState) (rg : RegionName)
    (s_h_h s_h_t K V BK BV NT : Nat) : Tile .real [BV, BK] :=
  Tile.mat fun a e => some (hElem s rg s_h_h s_h_t K V BK NT a.val e.val)

/-- `b_q` / `b_k` — the `(T, K)` key tile of this `(i_t, i_k)` block. -/
noncomputable def cbdQkTile (s : BlockState) (rg : RegionName)
    (s_k_h s_k_t T K BT BK : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e => some (qkElem s rg s_k_h s_k_t T K BT BK r.val e.val)

/-- The `b_v` / `b_do` load, phrased against the launch state's memory. -/
private theorem cbd_load_vo_eq (s t : BlockState) (rg : RegionName) (bpName : RegName)
    (s_v_h s_v_t T V BT BV : Nat)
    (hmem : ∀ off, t.readMem rg off = s.readMem rg off)
    (hbp : t.regs .blockPtr [BT, BV] bpName = some (⟨fun _ : TileIndex [BT, BV] =>
        { region := rg, baseOffset := s.pids 2 * s_v_h, parentShape := [T, V],
          blockShape := [BT, BV], strides := [s_v_t, 1],
          offsets := [s.pids 1 * BT, 0 * BV] }⟩ : Tile .blockPtr [BT, BV])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] bpName) [0, 1]) MaskOpt.none) t
      = some (cbdVoTile s rg s_v_h s_v_t T V BT BV) := by
  rw [cbd_load_2d rg T V s_v_t 1 BT BV (s.pids 1 * BT) (0 * BV)
    (s.pids 2 * s_v_h) bpName t hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨i0, i1, u⟩ := idx
  simp only [cbdVoTile, Tile.mat_data, voElem, Nat.zero_mul, Nat.zero_add, Nat.mul_one]
  split <;> simp [hmem]

/-- The `b_h` / `b_dh` load, phrased against the launch state's memory. -/
private theorem cbd_load_h_eq (s t : BlockState) (rg : RegionName) (bpName : RegName)
    (s_h_h s_h_t K V BK BV NT : Nat)
    (hmem : ∀ off, t.readMem rg off = s.readMem rg off)
    (hbp : t.regs .blockPtr [BV, BK] bpName = some (⟨fun _ : TileIndex [BV, BK] =>
        { region := rg, baseOffset := s.pids 2 * s_h_h, parentShape := [V, NT * K],
          blockShape := [BV, BK], strides := [1, s_h_t],
          offsets := [0 * BV, s.pids 1 * K + s.pids 0 * BK] }⟩ :
        Tile .blockPtr [BV, BK])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BV, BK] bpName) [0, 1]) MaskOpt.none) t
      = some (cbdHTile s rg s_h_h s_h_t K V BK BV NT) := by
  rw [cbd_load_2d rg V (NT * K) 1 s_h_t BV BK (0 * BV) (s.pids 1 * K + s.pids 0 * BK)
    (s.pids 2 * s_h_h) bpName t hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨i0, i1, u⟩ := idx
  simp only [cbdHTile, Tile.mat_data, hElem, Nat.zero_mul, Nat.zero_add, Nat.mul_one,
    Nat.add_assoc]
  split <;> simp [hmem]

/-- The `b_q` / `b_k` load, phrased against the launch state's memory. -/
private theorem cbd_load_qk_eq (s t : BlockState) (rg : RegionName) (bpName : RegName)
    (s_k_h s_k_t T K BT BK : Nat)
    (hmem : ∀ off, t.readMem rg off = s.readMem rg off)
    (hbp : t.regs .blockPtr [BT, BK] bpName = some (⟨fun _ : TileIndex [BT, BK] =>
        { region := rg, baseOffset := s.pids 2 * s_k_h, parentShape := [T, K],
          blockShape := [BT, BK], strides := [s_k_t, 1],
          offsets := [s.pids 1 * BT, s.pids 0 * BK] }⟩ : Tile .blockPtr [BT, BK])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BK] bpName) [0, 1]) MaskOpt.none) t
      = some (cbdQkTile s rg s_k_h s_k_t T K BT BK) := by
  rw [cbd_load_2d rg T K s_k_t 1 BT BK (s.pids 1 * BT) (s.pids 0 * BK)
    (s.pids 2 * s_k_h) bpName t hbp]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨i0, i1, u⟩ := idx
  simp only [cbdQkTile, Tile.mat_data, qkElem, Nat.mul_one, Nat.add_assoc]
  split <;> simp [hmem]


/-! ### The four accumulations

Each accumulator enters the (single) pass at zero, so its exit value is one term.
The four tiles below are the element specs of the same name, lifted to tiles. -/

/-- `b_dg_last` after the loop: the full `h · dh` reduction. -/
noncomputable def cbdDgLastTile (s : BlockState) (h dh : RegionName)
    (s_h_h s_h_t K V BK BV NT : Nat) : Tile .real [] :=
  Tile.scalar (some (dgLastAcc s h dh s_h_h s_h_t K V BK BV NT))

/-- `b_ds` after the loop. -/
noncomputable def cbdDsTile (s : BlockState) (v do_ : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) : Tile .real [BT, BT] :=
  Tile.mat fun r r' => some (dsAcc s v do_ s_v_h s_v_t T V BT BV r.val r'.val)

/-- `b_dq` after the loop. -/
noncomputable def cbdDqTile (s : BlockState) (h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e =>
    some (dqAcc s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r.val e.val)

/-- `b_dk` after the loop. -/
noncomputable def cbdDkTile (s : BlockState) (v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e =>
    some (dkAcc s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT r.val e.val)

/-- A `WithBot ℝ` sum of `some`s is the `some` of the `ℝ` sum. -/
private theorem cbd_coe_sum1 {n : Nat} (f : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e => (some (f e) : WithBot ℝ))
      = some (∑ e : Fin n, f e) := by
  show (Finset.univ.sum fun e => ((f e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]
  rfl

/-- A `WithBot ℝ` double sum of pointwise products collapses to the `ℝ` double sum.
`tl.sum` on a rank-2 tile lands in this shape after both axes are dropped. -/
private theorem cbd_coe_sum2 {m n : Nat} (f g : Fin m → Fin n → ℝ) :
    (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
        @Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          Option.map₂ (fun x y : ℝ => x * y) (some (f a e)) (some (g a e)))
      = some (∑ a : Fin m, ∑ e : Fin n, f a e * g a e) := by
  rw [show (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
        @Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
          Option.map₂ (fun x y : ℝ => x * y) (some (f a e)) (some (g a e)))
      = (@Finset.sum (Fin m) (WithBot ℝ) _ Finset.univ fun a =>
          (some (∑ e : Fin n, f a e * g a e) : WithBot ℝ))
      from Finset.sum_congr rfl fun a _ => cbd_coe_sum1 _]
  exact cbd_coe_sum1 _

/-- `b_dg_last += tl.sum(b_h * b_dh)` from zero. -/
private theorem cbd_dgLastAcc_eval (s t : BlockState) (h dh : RegionName)
    (s_h_h s_h_t K V BK BV NT : Nat)
    (hacc : t.regs .real [] "b_dg_last" = some (cbdZero []))
    (hh : t.regs .real [BV, BK] "b_h" = some (cbdHTile s h s_h_h s_h_t K V BK BV NT))
    (hdh : t.regs .real [BV, BK] "b_dh" = some (cbdHTile s dh s_h_h s_h_t K V BK BV NT)) :
    evalOp (Op.add .real Broadcast.nil (Op.ref .real [] "b_dg_last")
      (Op.reduceSum (shape := [BV]) ⟨0, by simp⟩ Bool.false
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BV, BK] "b_h") (Op.ref .real [BV, BK] "b_dh"))))) t
      = some (cbdDgLastTile s h dh s_h_h s_h_t K V BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hacc, evalOp_reduceSum, evalOp_reduceSum, evalOp_mul,
    evalOp_ref, hh, evalOp_ref, hdh]
  refine congrArg some ?_
  apply Tile.ext
  intro _
  simp only [cbdDgLastTile, cbdZero, cbdHTile, Tile.scalar, Tile.mat_data,
    Tile.bop_data, Tile.reduceSum_false, Tile.reduceSumDrop_data, TileShape.axisDim,
    TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Broadcast.leftIndex, Broadcast.rightIndex, dgLastAcc]
  erw [cbd_coe_sum2]
  simp

/-- `b_ds += tl.dot(b_do, tl.trans(b_v))` from zero. -/
private theorem cbd_dsAcc_eval (s t : BlockState) (v do_ : RegionName)
    (s_v_h s_v_t T V BT BV : Nat)
    (hds : t.regs .real [BT, BT] "b_ds" = some (cbdZero [BT, BT]))
    (hdo : t.regs .real [BT, BV] "b_do" = some (cbdVoTile s do_ s_v_h s_v_t T V BT BV))
    (hv : t.regs .real [BT, BV] "b_v" = some (cbdVoTile s v s_v_h s_v_t T V BT BV)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [BT, BT] "b_ds")
      (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_do")
        (Op.transpose (Op.ref .real [BT, BV] "b_v")))) t
      = some (cbdDsTile s v do_ s_v_h s_v_t T V BT BV) := by
  erw [evalOp_add, evalOp_ref, hds, evalOp_dot, evalOp_ref, hdo, evalOp_transpose,
    evalOp_ref, hv]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, r', u⟩ := idx
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data BT BV BT _ _ r r'
      (fun c => voElem s do_ s_v_h s_v_t T V BT r.val c.val)
      (fun c => voElem s v s_v_h s_v_t T V BT r'.val c.val)
      (fun _ => rfl) (fun _ => rfl)]
  simp [cbdDsTile, cbdZero, Tile.mat_data, dsAcc, NumericDType.add,
    WithBot.realAdd, Broadcast.leftIndex]

/-- `b_dq += tl.dot(b_do, b_h)` from zero. -/
private theorem cbd_dqAcc_eval (s t : BlockState) (h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hdq : t.regs .real [BT, BK] "b_dq" = some (cbdZero [BT, BK]))
    (hdo : t.regs .real [BT, BV] "b_do" = some (cbdVoTile s do_ s_v_h s_v_t T V BT BV))
    (hh : t.regs .real [BV, BK] "b_h" = some (cbdHTile s h s_h_h s_h_t K V BK BV NT)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [BT, BK] "b_dq")
      (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_do")
        (Op.ref .real [BV, BK] "b_h"))) t
      = some (cbdDqTile s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hdq, evalOp_dot, evalOp_ref, hdo, evalOp_ref, hh]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data BT BV BK _ _ r e
      (fun c => voElem s do_ s_v_h s_v_t T V BT r.val c.val)
      (fun c => hElem s h s_h_h s_h_t K V BK NT c.val e.val)
      (fun _ => rfl) (fun _ => rfl)]
  simp [cbdDqTile, cbdZero, Tile.mat_data, dqAcc, NumericDType.add,
    WithBot.realAdd, Broadcast.leftIndex]

/-- `b_dk += tl.dot(b_v, b_dh)` from zero. -/
private theorem cbd_dkAcc_eval (s t : BlockState) (v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hdk : t.regs .real [BT, BK] "b_dk" = some (cbdZero [BT, BK]))
    (hv : t.regs .real [BT, BV] "b_v" = some (cbdVoTile s v s_v_h s_v_t T V BT BV))
    (hdh : t.regs .real [BV, BK] "b_dh" = some (cbdHTile s dh s_h_h s_h_t K V BK BV NT)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [BT, BK] "b_dk")
      (Op.dot (batch := []) (Op.ref .real [BT, BV] "b_v")
        (Op.ref .real [BV, BK] "b_dh"))) t
      = some (cbdDkTile s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hdk, evalOp_dot, evalOp_ref, hv, evalOp_ref, hdh]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data BT BV BK _ _ r e
      (fun c => voElem s v s_v_h s_v_t T V BT r.val c.val)
      (fun c => hElem s dh s_h_h s_h_t K V BK NT c.val e.val)
      (fun _ => rfl) (fun _ => rfl)]
  simp [cbdDkTile, cbdZero, Tile.mat_data, dkAcc, NumericDType.add,
    WithBot.realAdd, Broadcast.leftIndex]


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

set_option maxHeartbeats 1000000 in
/-- **Loop-body run.** One pass of the value-axis body at `i_v = 0`: the four
block pointers, the four loads, and the four accumulations. Memory is untouched,
and every register the post-loop still needs (`o_i`, `b_g`, `b_g_last`, `b_dg`,
the ids) is carried through unchanged. -/
theorem cbdLoopBody_run (v h do_ dh : RegionName) (s t : BlockState)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hmemV : ∀ off, t.readMem v off = s.readMem v off)
    (hmemH : ∀ off, t.readMem h off = s.readMem h off)
    (hmemDo : ∀ off, t.readMem do_ off = s.readMem do_ off)
    (hmemDh : ∀ off, t.readMem dh off = s.readMem dh off)
    (hik : t.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)))
    (hibh : t.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hiv : t.regs .nat [] "i_v" = some (Tile.scalar 0))
    (hdq0 : t.regs .real [BT, BK] "b_dq" = some (cbdZero [BT, BK]))
    (hdk0 : t.regs .real [BT, BK] "b_dk" = some (cbdZero [BT, BK]))
    (hds0 : t.regs .real [BT, BT] "b_ds" = some (cbdZero [BT, BT]))
    (hdgl0 : t.regs .real [] "b_dg_last" = some (cbdZero [])) :
    ∃ s1, stepStmts (cbdLoopBody v h do_ dh s_v_h s_v_t s_h_h s_h_t
          T K V BT BK BV NT) t = some s1
      ∧ s1.pids = t.pids
      ∧ s1.numPids = t.numPids
      ∧ (∀ rg off, s1.readMem rg off = t.readMem rg off)
      ∧ s1.regs .nat [] "i_k" = t.regs .nat [] "i_k"
      ∧ s1.regs .nat [] "i_t" = t.regs .nat [] "i_t"
      ∧ s1.regs .nat [] "i_bh" = t.regs .nat [] "i_bh"
      ∧ s1.regs .nat [] "n_bh" = t.regs .nat [] "n_bh"
      ∧ s1.regs .nat [BT] "o_i" = t.regs .nat [BT] "o_i"
      ∧ s1.regs .real [BT] "b_g" = t.regs .real [BT] "b_g"
      ∧ s1.regs .real [] "b_g_last" = t.regs .real [] "b_g_last"
      ∧ s1.regs .real [BT] "b_dg" = t.regs .real [BT] "b_dg"
      ∧ s1.regs .real [] "b_dg_last"
          = some (cbdDgLastTile s h dh s_h_h s_h_t K V BK BV NT)
      ∧ s1.regs .real [BT, BT] "b_ds" = some (cbdDsTile s v do_ s_v_h s_v_t T V BT BV)
      ∧ s1.regs .real [BT, BK] "b_dq"
          = some (cbdDqTile s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)
      ∧ s1.regs .real [BT, BK] "b_dk"
          = some (cbdDkTile s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT) := by
  unfold cbdLoopBody
  -- p_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d v T V s_v_t 1 BT BV _ _ _ t (s.pids 2 * s_v_h) (s.pids 1 * BT) (0 * BV)
      (cbd_mulRef_eval t "i_bh" (s.pids 2) s_v_h hibh)
      (cbd_mulRef_eval t "i_t" (s.pids 1) BT hit)
      (cbd_mulRef_eval t "i_v" 0 BV hiv)))]
  -- p_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d h V (NT * K) 1 s_h_t BV BK _ _ _ _ (s.pids 2 * s_h_h) (0 * BV)
      (s.pids 1 * K + s.pids 0 * BK)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_h_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_v" 0 BV (by simp [hiv]))
      (cbd_add_eval _ _ _ (s.pids 1 * K) (s.pids 0 * BK)
        (cbd_mulRef_eval _ "i_t" (s.pids 1) K (by simp [hit]))
        (cbd_mulRef_eval _ "i_k" (s.pids 0) BK (by simp [hik])))))]
  -- p_do
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d do_ T V s_v_t 1 BT BV _ _ _ _ (s.pids 2 * s_v_h) (s.pids 1 * BT) (0 * BV)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_v_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp [hit]))
      (cbd_mulRef_eval _ "i_v" 0 BV (by simp [hiv]))))]
  -- p_dh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d dh V (NT * K) 1 s_h_t BV BK _ _ _ _ (s.pids 2 * s_h_h) (0 * BV)
      (s.pids 1 * K + s.pids 0 * BK)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_h_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_v" 0 BV (by simp [hiv]))
      (cbd_add_eval _ _ _ (s.pids 1 * K) (s.pids 0 * BK)
        (cbd_mulRef_eval _ "i_t" (s.pids 1) K (by simp [hit]))
        (cbd_mulRef_eval _ "i_k" (s.pids 0) BK (by simp [hik])))))]
  -- b_v, b_do, b_h, b_dh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_vo_eq s _ v "p_v" s_v_h s_v_t T V BT BV
      (by intro off; simp [hmemV]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_vo_eq s _ do_ "p_do" s_v_h s_v_t T V BT BV
      (by intro off; simp [hmemDo]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_h_eq s _ h "p_h" s_h_h s_h_t K V BK BV NT
      (by intro off; simp [hmemH]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_h_eq s _ dh "p_dh" s_h_h s_h_t K V BK BV NT
      (by intro off; simp [hmemDh]) (by simp)))]
  -- the four accumulations
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dgLastAcc_eval s _ h dh s_h_h s_h_t K V BK BV NT
      (by simp [hdgl0]) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dsAcc_eval s _ v do_ s_v_h s_v_t T V BT BV
      (by simp [hds0]) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dqAcc_eval s _ h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT
      (by simp [hdq0]) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dkAcc_eval s _ v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT
      (by simp [hdk0]) (by simp) (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, by simp, by simp, by intro rg off; simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_⟩ <;> simp

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

/-! ## The post-loop chain's register values

Nine compute statements, in the order the kernel issues them. The order is the
whole content of `dgLastFinal`: `b_dg_last` picks up its `b_dk · b_k` reduction at
the *gated* `b_dk`, before `b_dk` receives the `dot(trans(b_ds), b_q)` term. -/

/-- The `axis=1` row reduction of `b_q * b_dq - b_k * b_dk`, collapsed out of
`WithBot ℝ`. -/
private theorem cbd_sum_map2_sub {n : Nat} (a b c d : Fin n → ℝ) :
    (@Finset.sum (Fin n) (WithBot ℝ) _ Finset.univ fun e =>
        Option.map₂ (fun x y : ℝ => x - y)
          (Option.map₂ (fun x y : ℝ => x * y) (some (a e)) (some (b e)))
          (Option.map₂ (fun x y : ℝ => x * y) (some (c e)) (some (d e))))
      = some (∑ e : Fin n, (a e * b e - c e * d e)) :=
  cbd_coe_sum1 (fun e => a e * b e - c e * d e)

/-- `>=` on the `nat` channel — there is no `evalOp_ge` in the library. -/
private theorem cbd_ge_nat_eval {a b shape : TileShape} (bc : Broadcast a b shape)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState) (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.ge ComparableDType.nat bc x y) t
      = some (Tile.cop ComparableDType.nat.ge bc vx vy) := by
  simp only [evalOp, hx, hy, Option.bind_some, Option.bind_eq_bind]
  rfl

/-- `tl.broadcast_to` of a scalar — there is no `evalOp_broadcast` in the library. -/
private theorem cbd_broadcast_eval {dtype : TileDType} (e : Op dtype [])
    (sh : TileShape) (t : BlockState) (v : Tile dtype [])
    (hv : evalOp e t = some v) :
    evalOp (Op.broadcast e sh) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  simp only [evalOp, hv, Option.bind_some, Option.bind_eq_bind]

/-- `b_dg_last` after `*= tl.exp(b_g_last)`. -/
noncomputable def cbdDgLastGatedTile (s : BlockState) (g h dh : RegionName)
    (s_h_h s_h_t T K V BT BK BV NT : Nat) : Tile .real [] :=
  Tile.scalar (some (dgLastAcc s h dh s_h_h s_h_t K V BK BV NT
    * Real.exp (gLastElem s g T BT)))

/-- `b_dq` after the gate decay and the `scale`. -/
noncomputable def cbdDqGatedTile (s : BlockState) (g h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat) :
    Tile .real [BT, BK] :=
  Tile.mat fun r e => some (dqGated s g h do_ s_v_h s_v_t s_h_h s_h_t scale
    T K V BT BK BV NT r.val e.val)

/-- `b_dk` after its gate decay. -/
noncomputable def cbdDkGatedTile (s : BlockState) (g v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e => some (dkGated s g v dh s_v_h s_v_t s_h_h s_h_t
    T K V BT BK BV NT r.val e.val)

/-- `b_dg_last` after its `b_dk · b_k` reduction — the value `b_dg` consumes. -/
noncomputable def cbdDgLastFinalTile (s : BlockState) (g k v h dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat) : Tile .real [] :=
  Tile.scalar (some (dgLastFinal s g k v h dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
    T K V BT BK BV NT))

/-- `b_ds` after the causal mask and the score decay. -/
noncomputable def cbdDsMaskedTile (s : BlockState) (g v do_ : RegionName)
    (s_v_h s_v_t : Nat) (scale : ℝ) (T V BT BV : Nat) : Tile .real [BT, BT] :=
  Tile.mat fun r r' => some (dsMasked s g v do_ s_v_h s_v_t scale T V BT BV r.val r'.val)

/-- `b_dq` at the store. -/
noncomputable def cbdDqFinalTile (s : BlockState) (g k v h do_ : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e => some (dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
    scale T K V BT BK BV NT r.val e.val)

/-- `b_dk` at the store. -/
noncomputable def cbdDkFinalTile (s : BlockState) (g q v do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : Tile .real [BT, BK] :=
  Tile.mat fun r e => some (dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
    scale T K V BT BK BV NT r.val e.val)

/-- `b_dg` after its row reduction, before the last-row correction. -/
noncomputable def cbdDgRowTile (s : BlockState) (g q k v h do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : Tile .real [BT] :=
  Tile.vec fun r => some (∑ e : Fin BK,
    (qkElem s q s_k_h s_k_t T K BT BK r.val e.val
        * dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
            T K V BT BK BV NT r.val e.val
      - qkElem s k s_k_h s_k_t T K BT BK r.val e.val
        * dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
            T K V BT BK BV NT r.val e.val))

/-- `b_dg` at the store. -/
noncomputable def cbdDgFinalTile (s : BlockState) (g q k v h do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : Tile .real [BT] :=
  Tile.vec fun r => some (dgSpec s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
    scale T K V BT BK BV NT r.val)

/-- `b_dg_last *= tl.exp(b_g_last)`. -/
private theorem cbd_dgLastGated_eval (s t : BlockState) (g h dh : RegionName)
    (s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hacc : t.regs .real [] "b_dg_last"
        = some (cbdDgLastTile s h dh s_h_h s_h_t K V BK BV NT))
    (hgl : t.regs .real [] "b_g_last" = some (cbdBgLastTile s g T BT)) :
    evalOp (Op.mul .real Broadcast.nil (Op.ref .real [] "b_dg_last")
      (Op.exp (Op.ref .real [] "b_g_last"))) t
      = some (cbdDgLastGatedTile s g h dh s_h_h s_h_t T K V BT BK BV NT) := by
  erw [evalOp_mul, evalOp_ref, hacc, evalOp_exp, evalOp_ref, hgl]
  refine congrArg some ?_
  apply Tile.ext
  intro _
  simp only [Tile.bop_data, Tile.uop_data, cbdDgLastTile, cbdDgLastGatedTile,
    cbdBgLastTile, gLastElem, Tile.scalar, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, WithBot.realExp_some]
  rfl

/-- `b_dq = b_dq * tl.exp(b_g)[:, None] * scale`. -/
private theorem cbd_dqGated_eval (s t : BlockState) (g h do_ : RegionName)
    (s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hdq : t.regs .real [BT, BK] "b_dq"
        = some (cbdDqTile s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT))
    (hg : t.regs .real [BT] "b_g" = some (cbdBgTile s g T BT)) :
    evalOp (Op.mul .real Broadcast.scalarR
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BK] "b_dq")
        (Op.expandDim ⟨1, by simp⟩ (Op.exp (Op.ref .real [BT] "b_g"))))
      (Op.const scale)) t
      = some (cbdDqGatedTile s g h do_ s_v_h s_v_t s_h_h s_h_t scale
          T K V BT BK BV NT) := by
  erw [evalOp_mul, evalOp_mul, evalOp_ref, hdq, evalOp_expandDim, evalOp_exp,
    evalOp_ref, hg, evalOp_const]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [cbdDqGatedTile, cbdDqTile, dqGated, Tile.expandDim,
    TileShape.dropInsertedIndex]
  rw [cbdBgTile_data]
  rfl

/-- `b_dk = b_dk * tl.exp(0.0 - b_g + b_g_last)[:, None]`. -/
private theorem cbd_dkGated_eval (s t : BlockState) (g v dh : RegionName)
    (s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hdk : t.regs .real [BT, BK] "b_dk"
        = some (cbdDkTile s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT))
    (hg : t.regs .real [BT] "b_g" = some (cbdBgTile s g T BT))
    (hgl : t.regs .real [] "b_g_last" = some (cbdBgLastTile s g T BT)) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BT, BK] "b_dk")
      (Op.expandDim ⟨1, by simp⟩
        (Op.exp (Op.add .real Broadcast.scalarR
          (Op.sub .real Broadcast.scalarL (Op.const 0.0) (Op.ref .real [BT] "b_g"))
          (Op.ref .real [] "b_g_last"))))) t
      = some (cbdDkGatedTile s g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT) := by
  erw [evalOp_mul, evalOp_ref, hdk, evalOp_expandDim, evalOp_exp, evalOp_add,
    evalOp_sub, evalOp_const, evalOp_ref, hg, evalOp_ref, hgl]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [cbdDkGatedTile, cbdDkTile, dkGated, Tile.expandDim,
    TileShape.dropInsertedIndex]
  rw [cbdBgTile_data, cbdBgLastTile_data]
  simp [NumericDType.mul, NumericDType.add, NumericDType.sub]
  norm_num

/-- `b_dg_last += tl.sum(b_dk * b_k)`, at the *gated* `b_dk`. -/
private theorem cbd_dgLastFinal_eval (s t : BlockState) (g k v h dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT : Nat)
    (hacc : t.regs .real [] "b_dg_last"
        = some (cbdDgLastGatedTile s g h dh s_h_h s_h_t T K V BT BK BV NT))
    (hdk : t.regs .real [BT, BK] "b_dk"
        = some (cbdDkGatedTile s g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT))
    (hk : t.regs .real [BT, BK] "b_k" = some (cbdQkTile s k s_k_h s_k_t T K BT BK)) :
    evalOp (Op.add .real Broadcast.nil (Op.ref .real [] "b_dg_last")
      (Op.reduceSum (shape := [BT]) ⟨0, by simp⟩ Bool.false
        (Op.reduceSum ⟨1, by simp⟩ Bool.false
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BT, BK] "b_dk") (Op.ref .real [BT, BK] "b_k"))))) t
      = some (cbdDgLastFinalTile s g k v h dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hacc, evalOp_reduceSum, evalOp_reduceSum, evalOp_mul,
    evalOp_ref, hdk, evalOp_ref, hk]
  refine congrArg some ?_
  apply Tile.ext
  intro _
  simp only [cbdDgLastFinalTile, cbdDgLastGatedTile, cbdDkGatedTile, cbdQkTile,
    Tile.scalar, Tile.mat_data, Tile.bop_data, Tile.reduceSum_false,
    Tile.reduceSumDrop_data, TileShape.axisDim, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Broadcast.leftIndex, Broadcast.rightIndex, dgLastFinal]
  erw [cbd_coe_sum2]
  simp

/-- `b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(...), 0.0)`. -/
private theorem cbd_dsMasked_eval (s t : BlockState) (g v do_ : RegionName)
    (s_v_h s_v_t : Nat) (scale : ℝ) (T V BT BV : Nat)
    (hds : t.regs .real [BT, BT] "b_ds" = some (cbdDsTile s v do_ s_v_h s_v_t T V BT BV))
    (hoi : t.regs .nat [BT] "o_i" = some (Tile.vec (fun i : Fin BT => i.val)))
    (hg : t.regs .real [BT] "b_g" = some (cbdBgTile s g T BT)) :
    evalOp (Op.where
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BT] "o_i"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BT] "o_i")))
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BT] "b_ds") (Op.const scale))
        (Op.exp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "b_g"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BT] "b_g")))))
      (Op.broadcast (Op.const 0.0) [BT, BT])) t
      = some (cbdDsMaskedTile s g v do_ s_v_h s_v_t scale T V BT BV) := by
  have hge : evalOp
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BT] "o_i"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BT] "o_i"))) t
      = some (Tile.cop ComparableDType.nat.ge
          (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BT => i.val)))
          (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun i : Fin BT => i.val)))) :=
    cbd_ge_nat_eval _ _ _ t _ _
      (evalOp_expandDim_ref_of_regs .nat [BT] _ "o_i" t _ hoi)
      (evalOp_expandDim_ref_of_regs .nat [BT] _ "o_i" t _ hoi)
  have hzero : evalOp (Op.broadcast (Op.const 0.0) [BT, BT]) t
      = some (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [BT, BT]) :=
    cbd_broadcast_eval _ _ t _ (evalOp_const 0.0 t)
  erw [evalOp_where, hge, evalOp_mul, evalOp_mul, evalOp_ref, hds, evalOp_const,
    evalOp_exp, evalOp_sub, evalOp_expandDim, evalOp_ref, hg, evalOp_expandDim,
    evalOp_ref, hg, hzero]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, r', u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.uop_data,
    Tile.expandDim_data, Tile.vec, Tile.mat_data, Tile.scalar,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.ge, NumericDType.mul, NumericDType.sub, WithBot.realMul,
    WithBot.realSub, cbdDsTile, cbdDsMaskedTile, dsMasked, decide_eq_true_eq]
  by_cases hle : r'.val ≤ r.val
  · simp [hle]
    rw [cbdBgTile_data, cbdBgTile_data]
    rfl
  · simp [hle]
    norm_num

/-- `b_dq += tl.dot(b_ds, b_k)`. -/
private theorem cbd_dqFinal_eval (s t : BlockState) (g k v h do_ : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hdq : t.regs .real [BT, BK] "b_dq"
        = some (cbdDqGatedTile s g h do_ s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT))
    (hds : t.regs .real [BT, BT] "b_ds"
        = some (cbdDsMaskedTile s g v do_ s_v_h s_v_t scale T V BT BV))
    (hk : t.regs .real [BT, BK] "b_k" = some (cbdQkTile s k s_k_h s_k_t T K BT BK)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [BT, BK] "b_dq")
      (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_ds")
        (Op.ref .real [BT, BK] "b_k"))) t
      = some (cbdDqFinalTile s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hdq, evalOp_dot, evalOp_ref, hds, evalOp_ref, hk]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data BT BT BK _ _ r e
    (fun r' => dsMasked s g v do_ s_v_h s_v_t scale T V BT BV r.val r'.val)
    (fun r' => qkElem s k s_k_h s_k_t T K BT BK r'.val e.val)
    (fun _ => rfl) (fun _ => rfl)]
  simp [cbdDqFinalTile, cbdDqGatedTile, Tile.mat_data, dqSpec, NumericDType.add,
    WithBot.realAdd, Broadcast.leftIndex]

/-- `b_dk += tl.dot(tl.trans(b_ds), b_q)`. -/
private theorem cbd_dkFinal_eval (s t : BlockState) (g q v do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hdk : t.regs .real [BT, BK] "b_dk"
        = some (cbdDkGatedTile s g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT))
    (hds : t.regs .real [BT, BT] "b_ds"
        = some (cbdDsMaskedTile s g v do_ s_v_h s_v_t scale T V BT BV))
    (hq : t.regs .real [BT, BK] "b_q" = some (cbdQkTile s q s_k_h s_k_t T K BT BK)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .real [BT, BK] "b_dk")
      (Op.dot (batch := []) (Op.transpose (Op.ref .real [BT, BT] "b_ds"))
        (Op.ref .real [BT, BK] "b_q"))) t
      = some (cbdDkFinalTile s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hdk, evalOp_dot, evalOp_transpose, evalOp_ref, hds,
    evalOp_ref, hq]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.rightIndex_consSame, Broadcast.rightIndex_nil]
  rw [tile_dot_data BT BT BK _ _ r e
    (fun r' => dsMasked s g v do_ s_v_h s_v_t scale T V BT BV r'.val r.val)
    (fun r' => qkElem s q s_k_h s_k_t T K BT BK r'.val e.val)
    (fun _ => rfl) (fun _ => rfl)]
  simp [cbdDkFinalTile, cbdDkGatedTile, Tile.mat_data, dkSpec, NumericDType.add,
    WithBot.realAdd, Broadcast.leftIndex]

/-- `b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)`, from zero. -/
private theorem cbd_dgRow_eval (s t : BlockState) (g q k v h do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hdg : t.regs .real [BT] "b_dg" = some (cbdZero [BT]))
    (hq : t.regs .real [BT, BK] "b_q" = some (cbdQkTile s q s_k_h s_k_t T K BT BK))
    (hk : t.regs .real [BT, BK] "b_k" = some (cbdQkTile s k s_k_h s_k_t T K BT BK))
    (hdq : t.regs .real [BT, BK] "b_dq"
        = some (cbdDqFinalTile s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
            scale T K V BT BK BV NT))
    (hdk : t.regs .real [BT, BK] "b_dk"
        = some (cbdDkFinalTile s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
            scale T K V BT BK BV NT)) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BT] "b_dg")
      (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.sub .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BT, BK] "b_dq"))
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BT, BK] "b_k") (Op.ref .real [BT, BK] "b_dk"))))) t
      = some (cbdDgRowTile s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV NT) := by
  erw [evalOp_add, evalOp_ref, hdg, evalOp_reduceSum, evalOp_sub, evalOp_mul,
    evalOp_ref, hq, evalOp_ref, hdq, evalOp_mul, evalOp_ref, hk, evalOp_ref, hdk]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  simp only [Tile.bop_data, Tile.reduceSum_false, Tile.reduceSumDrop_data,
    TileShape.axisDim, TileShape.insertAxisIndex, Tile.mat_data, cbdQkTile,
    cbdDqFinalTile, cbdDkFinalTile, NumericDType.sub, NumericDType.mul,
    WithBot.realSub, WithBot.realMul, Broadcast.leftIndex, Broadcast.rightIndex]
  erw [cbd_sum_map2_sub]
  simp [cbdDgRowTile, cbdZero, Tile.vec, NumericDType.add, WithBot.realAdd]

/-- `b_dg = tl.where(o_i < min(BT, T - i_t*BT) - 1, b_dg, b_dg + b_dg_last)`. -/
private theorem cbd_dgFinal_eval (s t : BlockState) (g q k v h do_ dh : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)))
    (hoi : t.regs .nat [BT] "o_i" = some (Tile.vec (fun i : Fin BT => i.val)))
    (hdg : t.regs .real [BT] "b_dg"
        = some (cbdDgRowTile s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
            scale T K V BT BK BV NT))
    (hdgl : t.regs .real [] "b_dg_last"
        = some (cbdDgLastFinalTile s g k v h dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
            T K V BT BK BV NT)) :
    evalOp (Op.where
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
        (Op.ref .real [] "b_dg_last"))) t
      = some (cbdDgFinalTile s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV NT) := by
  have hmin : evalOp
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil (Op.constNat BT)
          (Op.sub .nat Broadcast.nil (Op.constNat T)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))))
        (Op.constNat BT)
        (Op.sub .nat Broadcast.nil (Op.constNat T)
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)))) t
      = some (Tile.scalar (min BT (T - s.pids 1 * BT))) :=
    cbd_min_eval t _ _ BT (T - s.pids 1 * BT) (evalOp_constNat BT t)
      (cbd_sub_eval t _ _ T (s.pids 1 * BT) (evalOp_constNat T t)
        (cbd_mulRef_eval t "i_t" (s.pids 1) BT hit))
  erw [evalOp_where, evalOp_lt, evalOp_ref, hoi,
    cbd_sub_eval t _ _ (min BT (T - s.pids 1 * BT)) 1 hmin (evalOp_constNat 1 t),
    evalOp_ref, hdg, evalOp_add, evalOp_ref, hdg, evalOp_ref, hdgl]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.vec, Tile.scalar,
    Broadcast.leftIndex, ComparableDType.lt, NumericDType.add,
    WithBot.realAdd, cbdDgRowTile, cbdDgLastFinalTile, cbdDgFinalTile, dgSpec,
    decide_eq_true_eq]
  by_cases hlt : r.val < min BT (T - s.pids 1 * BT) - 1
  · simp [hlt]
  · simp [hlt]


/-! ## The three stores

`dq` and `dk` share one block-pointer layout, so one injectivity hypothesis
covers both; the `dg` layout is `base + (i_t*BT + r)`, injective outright. -/

/-- The `dq` / `dk` block-ptr store offset at lane `(r, e)`. -/
def cbdQkAddr (s : BlockState) (s_k_h s_k_t BT BK : Nat) (idx : TileIndex [BT, BK]) : Nat :=
  s.pids 2 * s_k_h + (s.pids 1 * BT + idx.1.val) * s_k_t + (s.pids 0 * BK + idx.2.1.val) * 1

/-- The `dg` block-ptr store offset at lane `r`. -/
def cbdDgAddr (s : BlockState) (T BT : Nat) (idx : TileIndex [BT]) : Nat :=
  (s.pids 0 * s.numPids 2 + s.pids 2) * T + (s.pids 1 * BT + idx.1.val) * 1

/-- Post-store state of a boundary-checked 2-D block-pointer store. -/
noncomputable def cbdStore2State (R : RegionName) (d0 d1 st0 st1 B0 B1 o0 o1 base : Nat)
    (f : TileIndex [B0, B1] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [B0, B1]).foldl
    (fun acc i => if (o0 + i.1.val < d0 ∧ o1 + i.2.1.val < d1) then
        acc.writeMem R (base + (o0 + i.1.val) * st0 + (o1 + i.2.1.val) * st1) (f i)
      else acc) t

/-- Post-store state of a boundary-checked 1-D block-pointer store. -/
noncomputable def cbdStore1State (R : RegionName) (len stride B0 off base : Nat)
    (f : TileIndex [B0] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [B0]).foldl
    (fun acc i => if off + i.1.val < len then
        acc.writeMem R (base + (off + i.1.val) * stride) (f i) else acc) t

private theorem cbd_store_2d_eq (R : RegionName) (d0 d1 st0 st1 B0 B1 o0 o1 base : Nat)
    (bpName vName : RegName) (t : BlockState) (vt : Tile .real [B0, B1])
    (f : TileIndex [B0, B1] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hbp : t.regs .blockPtr [B0, B1] bpName = some (⟨fun _ : TileIndex [B0, B1] =>
        { region := R, baseOffset := base, parentShape := [d0, d1],
          blockShape := [B0, B1], strides := [st0, st1], offsets := [o0, o1] }⟩ :
        Tile .blockPtr [B0, B1]))
    (hv : t.regs .real [B0, B1] vName = some vt) :
    stepStmt (Stmt.store .real [B0, B1]
        (.blockPtr (Op.ref .blockPtr [B0, B1] bpName) [0, 1])
        (Op.ref .real [B0, B1] vName) .none) t
      = some (cbdStore2State R d0 d1 st0 st1 B0 B1 o0 o1 base f t) := by
  unfold stepStmt cbdStore2State
  simp only [evalOp_ref, hv, hbp]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [B0, B1])) ?_)
  funext acc i
  obtain ⟨i0, i1, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and]
  by_cases hb : o0 + i0.val < d0 ∧ o1 + i1.val < d1
  · simp only [hb, BlockState.writeMemTyped_real, hfv]
    rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false]

private theorem cbd_store_1d_eq (R : RegionName) (len stride B0 off base : Nat)
    (bpName vName : RegName) (t : BlockState) (vt : Tile .real [B0])
    (f : TileIndex [B0] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hbp : t.regs .blockPtr [B0] bpName = some (⟨fun _ : TileIndex [B0] =>
        { region := R, baseOffset := base, parentShape := [len],
          blockShape := [B0], strides := [stride], offsets := [off] }⟩ :
        Tile .blockPtr [B0]))
    (hv : t.regs .real [B0] vName = some vt) :
    stepStmt (Stmt.store .real [B0]
        (.blockPtr (Op.ref .blockPtr [B0] bpName) [0])
        (Op.ref .real [B0] vName) .none) t
      = some (cbdStore1State R len stride B0 off base f t) := by
  unfold stepStmt cbdStore1State
  simp only [evalOp_ref, hv, hbp]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [B0])) ?_)
  funext acc i
  obtain ⟨i0, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_1d, BlockPtr.inBounds_1d,
    Bool.true_and]
  by_cases hb : off + i0.val < len
  · simp only [hb, BlockState.writeMemTyped_real, hfv]
    rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false]

private theorem cbd_store_2d_props (R : RegionName)
    (d0 d1 st0 st1 B0 B1 o0 o1 base : Nat) (t : BlockState) (f : TileIndex [B0, B1] → ℝ)
    (hInj : Function.Injective (fun i : TileIndex [B0, B1] =>
        base + (o0 + i.1.val) * st0 + (o1 + i.2.1.val) * st1)) :
    (cbdStore2State R d0 d1 st0 st1 B0 B1 o0 o1 base f t).pids = t.pids
      ∧ (cbdStore2State R d0 d1 st0 st1 B0 B1 o0 o1 base f t).regs = t.regs
      ∧ (∀ i : TileIndex [B0, B1], (o0 + i.1.val < d0 ∧ o1 + i.2.1.val < d1) →
          (cbdStore2State R d0 d1 st0 st1 B0 B1 o0 o1 base f t).readMem R
              (base + (o0 + i.1.val) * st0 + (o1 + i.2.1.val) * st1) = f i)
      ∧ (∀ rg off, rg ≠ R →
          (cbdStore2State R d0 d1 st0 st1 B0 B1 o0 o1 base f t).readMem rg off
            = t.readMem rg off) := by
  classical
  unfold cbdStore2State
  refine ⟨BlockState.foldl_writeMem_prop_masked_pids _ _ _ _ _ _, ?_, ?_, ?_⟩
  · funext dtype shape name; rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro i hi
    have h := BlockState.scatter_readback_prop_masked_nd (region := R) t
      (fun j : TileIndex [B0, B1] =>
        base + (o0 + j.1.val) * st0 + (o1 + j.2.1.val) * st1) f
      (fun j : TileIndex [B0, B1] => o0 + j.1.val < d0 ∧ o1 + j.2.1.val < d1) hInj i
    rw [h, if_pos hi]
  · intro rg off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      R _ f _ _ t rg off hrg

private theorem cbd_store_1d_props (R : RegionName) (len stride B0 off base : Nat)
    (t : BlockState) (f : TileIndex [B0] → ℝ) (hstride : 0 < stride) :
    (cbdStore1State R len stride B0 off base f t).pids = t.pids
      ∧ (cbdStore1State R len stride B0 off base f t).regs = t.regs
      ∧ (∀ i : TileIndex [B0], off + i.1.val < len →
          (cbdStore1State R len stride B0 off base f t).readMem R
              (base + (off + i.1.val) * stride) = f i)
      ∧ (∀ rg o, rg ≠ R →
          (cbdStore1State R len stride B0 off base f t).readMem rg o = t.readMem rg o) := by
  classical
  have hInj : Function.Injective
      (fun i : TileIndex [B0] => base + (off + i.1.val) * stride) := by
    intro a b hab
    obtain ⟨a0, ua⟩ := a
    obtain ⟨b0, ub⟩ := b
    simp only at hab
    have : a0.val = b0.val := by
      have := Nat.eq_of_mul_eq_mul_right hstride
        (show (off + a0.val) * stride = (off + b0.val) * stride by omega)
      omega
    simp only [Prod.mk.injEq]
    exact ⟨Fin.ext this, trivial⟩
  unfold cbdStore1State
  refine ⟨BlockState.foldl_writeMem_prop_masked_pids _ _ _ _ _ _, ?_, ?_, ?_⟩
  · funext dtype shape name; rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro i hi
    have h := BlockState.scatter_readback_prop_masked_nd (region := R) t
      (fun j : TileIndex [B0] => base + (off + j.1.val) * stride) f
      (fun j : TileIndex [B0] => off + j.1.val < len) hInj i
    rw [h, if_pos hi]
  · intro rg o hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      R _ f _ _ t rg o hrg


set_option maxHeartbeats 2000000 in
/-- **Post-loop run.** The nineteen tail statements: the `q`/`k` loads, the nine
compute steps, the three output block pointers, and the three stores. Concludes
about the three regions' readback at every in-region lane. -/
theorem cbdPostLoop_run (q k dq dk dg g v h do_ dh : RegionName) (s t : BlockState)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV NT : Nat)
    (hmemQ : ∀ off, t.readMem q off = s.readMem q off)
    (hmemK : ∀ off, t.readMem k off = s.readMem k off)
    (hDqDk : dq ≠ dk) (hDqDg : dq ≠ dg) (hDkDg : dk ≠ dg)
    (hInj : Function.Injective
      (fun i : TileIndex [BT, BK] => cbdQkAddr s s_k_h s_k_t BT BK i))
    (hik : t.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)))
    (hibh : t.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hnbh : t.regs .nat [] "n_bh" = some (Tile.scalar (s.numPids 2)))
    (hoi : t.regs .nat [BT] "o_i" = some (Tile.vec (fun i : Fin BT => i.val)))
    (hg : t.regs .real [BT] "b_g" = some (cbdBgTile s g T BT))
    (hgl : t.regs .real [] "b_g_last" = some (cbdBgLastTile s g T BT))
    (hdg0 : t.regs .real [BT] "b_dg" = some (cbdZero [BT]))
    (hdgl : t.regs .real [] "b_dg_last"
        = some (cbdDgLastTile s h dh s_h_h s_h_t K V BK BV NT))
    (hds : t.regs .real [BT, BT] "b_ds" = some (cbdDsTile s v do_ s_v_h s_v_t T V BT BV))
    (hdq : t.regs .real [BT, BK] "b_dq"
        = some (cbdDqTile s h do_ s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT))
    (hdk : t.regs .real [BT, BK] "b_dk"
        = some (cbdDkTile s v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)) :
    ∃ sF, stepStmts (cbdPostLoop q k dq dk dg s_k_h s_k_t scale T K BT BK) t = some sF
      ∧ (∀ i : TileIndex [BT, BK],
          (s.pids 1 * BT + i.1.val < T ∧ s.pids 0 * BK + i.2.1.val < K) →
          sF.readMem dq (cbdQkAddr s s_k_h s_k_t BT BK i)
            = dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV NT i.1.val i.2.1.val)
      ∧ (∀ i : TileIndex [BT, BK],
          (s.pids 1 * BT + i.1.val < T ∧ s.pids 0 * BK + i.2.1.val < K) →
          sF.readMem dk (cbdQkAddr s s_k_h s_k_t BT BK i)
            = dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV NT i.1.val i.2.1.val)
      ∧ (∀ i : TileIndex [BT], s.pids 1 * BT + i.1.val < T →
          sF.readMem dg (cbdDgAddr s T BT i)
            = dgSpec s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV NT i.1.val) := by
  unfold cbdPostLoop
  -- p_q, p_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d q T K s_k_t 1 BT BK _ _ _ t (s.pids 2 * s_k_h) (s.pids 1 * BT)
      (s.pids 0 * BK)
      (cbd_mulRef_eval t "i_bh" (s.pids 2) s_k_h hibh)
      (cbd_mulRef_eval t "i_t" (s.pids 1) BT hit)
      (cbd_mulRef_eval t "i_k" (s.pids 0) BK hik)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d k T K s_k_t 1 BT BK _ _ _ _ (s.pids 2 * s_k_h) (s.pids 1 * BT)
      (s.pids 0 * BK)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_k_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp [hit]))
      (cbd_mulRef_eval _ "i_k" (s.pids 0) BK (by simp [hik]))))]
  -- b_k, b_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_qk_eq s _ k "p_k" s_k_h s_k_t T K BT BK
      (by intro off; simp [hmemK]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_load_qk_eq s _ q "p_q" s_k_h s_k_t T K BT BK
      (by intro off; simp [hmemQ]) (by simp)))]
  -- the nine compute steps
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dgLastGated_eval s _ g h dh s_h_h s_h_t T K V BT BK BV NT
      (by simp [hdgl]) (by simp [hgl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dqGated_eval s _ g h do_ s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT
      (by simp [hdq]) (by simp [hg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dkGated_eval s _ g v dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT
      (by simp [hdk]) (by simp [hg]) (by simp [hgl])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dgLastFinal_eval s _ g k v h dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      T K V BT BK BV NT (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dsMasked_eval s _ g v do_ s_v_h s_v_t scale T V BT BV
      (by simp [hds]) (by simp [hoi]) (by simp [hg])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dqFinal_eval s _ g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dkFinal_eval s _ g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dgRow_eval s _ g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT (by simp [hdg0]) (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_dgFinal_eval s _ g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT (by simp [hit]) (by simp [hoi]) (by simp) (by simp)))]
  -- p_dq, p_dk, p_dg
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d dq T K s_k_t 1 BT BK _ _ _ _ (s.pids 2 * s_k_h) (s.pids 1 * BT)
      (s.pids 0 * BK)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_k_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp [hit]))
      (cbd_mulRef_eval _ "i_k" (s.pids 0) BK (by simp [hik]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_2d dk T K s_k_t 1 BT BK _ _ _ _ (s.pids 2 * s_k_h) (s.pids 1 * BT)
      (s.pids 0 * BK)
      (cbd_mulRef_eval _ "i_bh" (s.pids 2) s_k_h (by simp [hibh]))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp [hit]))
      (cbd_mulRef_eval _ "i_k" (s.pids 0) BK (by simp [hik]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cbd_mkptr_1d dg T 1 BT _ _ _ ((s.pids 0 * s.numPids 2 + s.pids 2) * T)
      (s.pids 1 * BT)
      (cbd_mulConst_eval _ _ (s.pids 0 * s.numPids 2 + s.pids 2) T
        (cbd_add_eval _ _ _ (s.pids 0 * s.numPids 2) (s.pids 2)
          (cbd_mulRefRef_eval _ "i_k" "n_bh" (s.pids 0) (s.numPids 2)
            (by simp [hik]) (by simp [hnbh]))
          (by rw [evalOp_ref]; simp [hibh])))
      (cbd_mulRef_eval _ "i_t" (s.pids 1) BT (by simp [hit]))))]
  -- the three stores
  rw [stepStmts.cons_some (cbd_store_2d_eq dq T K s_k_t 1 BT BK (s.pids 1 * BT)
    (s.pids 0 * BK) (s.pids 2 * s_k_h) "p_dq" "b_dq" _
    (cbdDqFinalTile s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT)
    (fun i => dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val i.2.1.val)
    (fun _ => rfl) (by simp) (by simp))]
  obtain ⟨hp1, hr1, hread1, hoth1⟩ := cbd_store_2d_props dq T K s_k_t 1 BT BK
    (s.pids 1 * BT) (s.pids 0 * BK) (s.pids 2 * s_k_h) _
    (fun i => dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val i.2.1.val) hInj
  rw [stepStmts.cons_some (cbd_store_2d_eq dk T K s_k_t 1 BT BK (s.pids 1 * BT)
    (s.pids 0 * BK) (s.pids 2 * s_k_h) "p_dk" "b_dk" _
    (cbdDkFinalTile s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT)
    (fun i => dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val i.2.1.val)
    (fun _ => rfl) (by rw [hr1]; simp) (by rw [hr1]; simp))]
  obtain ⟨hp2, hr2, hread2, hoth2⟩ := cbd_store_2d_props dk T K s_k_t 1 BT BK
    (s.pids 1 * BT) (s.pids 0 * BK) (s.pids 2 * s_k_h)
    (cbdStore2State dq T K s_k_t 1 BT BK (s.pids 1 * BT) (s.pids 0 * BK)
      (s.pids 2 * s_k_h)
      (fun i => dqSpec s g k v h do_ s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
          T K V BT BK BV NT i.1.val i.2.1.val) _)
    (fun i => dkSpec s g q v do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val i.2.1.val) hInj
  rw [stepStmts.cons_some (cbd_store_1d_eq dg T 1 BT (s.pids 1 * BT)
    ((s.pids 0 * s.numPids 2 + s.pids 2) * T) "p_dg" "b_dg" _
    (cbdDgFinalTile s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV NT)
    (fun i => dgSpec s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val)
    (fun _ => rfl) (by rw [hr2, hr1]; simp) (by rw [hr2, hr1]; simp))]
  obtain ⟨hp3, hr3, hread3, hoth3⟩ := cbd_store_1d_props dg T 1 BT (s.pids 1 * BT)
    ((s.pids 0 * s.numPids 2 + s.pids 2) * T) _
    (fun i => dgSpec s g q k v h do_ dh s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV NT i.1.val) Nat.one_pos
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · intro i hi
    rw [hoth3 dq _ hDqDg, hoth2 dq _ hDqDk]
    exact hread1 i hi
  · intro i hi
    rw [hoth3 dk _ hDkDg]
    exact hread2 i hi
  · intro i hi
    exact hread3 i hi

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg

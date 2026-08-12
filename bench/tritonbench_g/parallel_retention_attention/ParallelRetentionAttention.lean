import VeriTile.Triton

/-!
# `parallel_retention_attention` — strict per-kernel correctness

`parallel_retention_attention.py` implements chunk-parallel *retention*
attention: causal attention whose score is the plain dot product `q·k`
weighted by the per-head exponential decay `γ^(pos_q − pos_k)` with
`γ = 2^b_b`, `b_b = log2(1 − 2^(−5 − i_h))`. The forward JIT streams K/V
blocks in two phases — the strictly-below-diagonal blocks unmasked with a
decayed accumulator recurrence (`b_o *= 2^(b_b·BTS)` per block), then the
diagonal block causally masked — and the backward dk/dv helper streams Q/dO
blocks in the *opposite* order: the strictly-above-diagonal blocks first via
a **descending** `-BTS` loop with the mirrored decay recurrence
(`b_dk *= d_b`, `b_dv *= d_b`), then the diagonal block masked.

## Scope

This file ports two of the four `@triton.jit` kernels:

* `parallel_retention_attention.py`'s `parallel_retention_fwd_kernel` — the
  launched forward JIT (program `(i_kv, i_c, i_bh)` on grid
  `(NK·NV, cdiv(T, BTL), B·H)`), and
* `_parallel_retention_bwd_dkv` — the backward dk/dv helper JIT, verified as
  a standalone kernel over universally-quantified scalar arguments
  `i_bh, i_c, i_k, i_v, i_h` (the values the shell
  `parallel_retention_bwd_kernel` would pass it).

The backward shell `parallel_retention_bwd_kernel` (a prologue plus two
helper calls separated by `tl.debug_barrier()`) and the dq helper
`_parallel_retention_bwd_dq` are not transcribed. The host launch and the
host-side `o.sum(0)` / `dk.sum(0)`-style NK/NV reductions are the *trusted
boundary*, not proof obligations here.

## Translation-surface blocker

Translation-surface blocker: the DSL has no cross-`@triton.jit`
function-call surface, so `_parallel_retention_bwd_dkv` is ported as a
standalone kernel whose Python scalar arguments `i_bh, i_c, i_k, i_v, i_h`
become universally-quantified Lean binders (the shell
`parallel_retention_bwd_kernel` and `_parallel_retention_bwd_dq` are out of
scope), its trailing bare `return` is dropped, its **descending** loop
`for i in range(cdiv(T, BTS)·BTS − BTS, (i_c+1)·BTL − BTS, −BTS)` is
respelled as the ascending change of variable
`for j in range(0, cdiv(cdiv(T, BTS)·BTS − (i_c+1)·BTL, BTS))` with
body-first `i = cdiv(T, BTS)·BTS − BTS − j·BTS` (the `± BTS` in Python's
`start − stop` cancels exactly in ℤ, and ℕ-truncated subtraction reproduces
Python's empty-range behaviour, so the trip counts agree for **all**
parameter values — the `parallel_attention` respelling verbatim), and its
diagonal decay's unary-minus index spelling `-o_k[:, None] + o_q[None, :]`
is respelled as the subtraction `o_q[None, :] - o_k[:, None]` (the DSL has
no unary tile negation; the two agree on every lane the `tl.where` keep mask
`m_s` retains, and the masked-off lanes multiply the hard `0` branch).
Additionally the decay prologue's implicit int→float promotions
(`i_h * 1.0`, `b_b * BTS`, `(BTS − o_k) * b_b`, …) are spelled with the
explicit nat→real cast `tl.toReal(...)` (the `chunk_retention` precedent).
The Lean surfaces are therefore not line-for-line textual matches of the
Python bodies, and the textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker (registered
in `proof_blockers.md`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `.to(b_q.dtype)` /
`.to(tl.float32)` / `.to(p_o.dtype.element_ty)` casts are identities at
`tl.float32`); `num_warps`/`num_stages` and `tl.debug_barrier()` (an
intra-program no-op fence, transcribed verbatim) are not modeled at the
scheduling level. `boundary_check=(0, 1)` block-pointer loads zero-pad
out-of-window lanes; the specs bake that window directly into the guarded
value functions, so the headlines hold for **arbitrary ragged tails** with
no divisibility hypothesis on `T`. The only trip-count hypothesis is the
host's own `assert BTL % BTS == 0`.

Faithfulness note (honest, invisible to the stores): the diagonal decay
computes `(o_q − o_k) · b_b` on an *integer* tile, which goes negative on
the lanes the causal mask `m_s` rejects; the DSL's `.nat` channel truncates
those lanes at `0`, so the ported `d_s` register holds `2^0·…` where the
CUDA kernel holds a sub-unit power — on exactly the lanes `tl.where` then
replaces with `0` (both here and in the backward diagonal loop). The
divergence is unobservable through every store.

The port targets `parallel_retention_attention.py`'s
`parallel_retention_fwd_kernel`.
-/

namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `pra_fwd_o_exec_genuine`, `pra_bwd_dkv_exec_genuine` -/

section Correct_without_Rounding

/-- Faithful transcription of `parallel_retention_fwd_kernel`. -/
def pra_fwd_surface
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
  i_h = i_bh % $(H)
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal(i_h) * 1.0))
  o_k = tl.arange(0, $(BTS))
  d_h = tl.math.exp2(tl.toReal($(BTS) - o_k) * b_b)
  p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=(i_c * $(BTL), i_k * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), 0), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(0, i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_o = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_h[None, :]
    b_o = b_o * tl.math.exp2(b_b * tl.toReal($(BTS)))
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTL))) * b_b)
  b_o *= d_q[:, None]
  o_k = tl.arange(0, $(BTS))
  p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
    shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
    offsets=(i_k * $(BK), i_c * $(BTL)), block_shape=($(BK), $(BTS)), order=(0, 1))
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTS), $(BV)), order=(1, 0))
  for _i in range(i_c * $(BTL), (i_c + $(1)) * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    m_s = o_q[:, None] >= o_k[None, :]
    d_s = tl.where(m_s, tl.math.exp2(tl.toReal(o_q[:, None] - o_k[None, :]) * b_b), 0.0)
    b_s = tl.dot(b_q, b_k, allow_tf32=false) * d_s
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The forward surface lowers to the algorithm layer. -/
theorem pra_fwd_surface_toAlgorithm_supported
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ∃ alg, (pra_fwd_surface q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      scale B H T K V BTL BTS BK BV).toAlgorithm? = Except.ok alg := by
  simp [pra_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `_parallel_retention_bwd_dkv`, with the
helper's scalar arguments `i_bh, i_c, i_k, i_v, i_h` as
universally-quantified binders, the descending loop spelled as its ascending
change of variable, the diagonal unary-minus decay respelled as a
subtraction (see the preamble), and the trailing bare `return` dropped. -/
def pra_bwd_dkv_surface
    (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  b_b = tl.math.log2(1.0 - tl.math.exp2(0.0 - 5.0 - tl.toReal($(i_h)) * 1.0))
  d_b = tl.math.exp2(b_b * tl.toReal($(BTS)))
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_qk_h),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  d_h = tl.math.exp2(tl.toReal($(BTL) - tl.arange(0, $(BTL))) * b_b)
  b_kd = (b_k * d_h[:, None]).to(b_k.dtype)
  d_q = tl.math.exp2(tl.toReal(tl.arange(0, $(BTS))) * b_b)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_do = (b_do * d_q[None, :]).to(b_do.dtype)
    b_dv *= d_b
    b_s = tl.dot((b_kd).to(b_q.dtype), b_q, allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk *= d_b
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  b_dk *= d_h[:, None] * $((scale : ℝ))
  b_dv *= $((scale : ℝ))
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_vo_h),
      shape=($(V), $(T)), strides=($(s_vo_d), $(s_vo_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    m_s = o_k[:, None] <= o_q[None, :]
    d_s = tl.where(m_s,
      tl.math.exp2(tl.toReal(o_q[None, :] - o_k[:, None]) * (b_b).to(tl.float32)),
      0.0) * $((scale : ℝ))
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * d_s
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * d_s
    b_dk += tl.dot((b_ds).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    b_dv += tl.dot((b_s).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_qk_h : Nat)),
    shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_vo_h : Nat)),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The backward dk/dv surface lowers to the algorithm layer. -/
theorem pra_bwd_dkv_surface_toAlgorithm_supported
    (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ∃ alg, (pra_bwd_dkv_surface q k v do_ dk dv i_bh i_c i_k i_v i_h
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      B H T K V BTL BTS BK BV).toAlgorithm? = Except.ok alg := by
  simp [pra_bwd_dkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. `tl.debug_barrier()` erases to the no-op `Stmt.ifThen (constBool
false) []`; `tl.trans` is `Op.transpose`; `tl.toReal` is `Op.natToReal` and
`tl.math.exp2`/`log2` are `Op.exp2`/`Op.log2` (the `chunk_retention`
lowerings); the `.to(x.dtype)` casts erase at lowering while the diagonal
`b_b.to(tl.float32)` survives as the identity
`Op.castFloat FloatDType.real FloatDType.real`. -/

/-- The forward non-diagonal streaming body: two block-pointer loads, the
decay-weighted score, the accumulator decay-and-add recurrence, and the two
`tl.advance`s. -/
def praFwdLoop1Body (BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .real [BK, BTS] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTS, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q") (Op.ref .real [BK, BTS] "b_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "d_h"))),
    Stmt.assign .real [BTL, BV] "b_o"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_o")
        (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
          (Op.natToReal (Op.constNat BTS))))),
    Stmt.assign .real [BTL, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTS, BV] "b_v"))),
    Stmt.assign .blockPtr [BK, BTS] "p_k"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [(0 : Nat), BTS]),
    Stmt.assign .blockPtr [BTS, BV] "p_v"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [BTS, (0 : Nat)]) ]

/-- The forward diagonal body: the causal mask, the intra-block decay tile
`d_s`, the weighted score, and the `o_k += BTS` carry. -/
def praFwdLoop2Body (BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .real [BK, BTS] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTS, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .bool [BTL, BTS] "m_s"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_q"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_k"))),
    Stmt.assign .real [BTL, BTS] "d_s"
      (Op.where (Op.ref .bool [BTL, BTS] "m_s")
        (Op.exp2 (Op.mul .real Broadcast.scalarR
          (Op.natToReal (Op.sub .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_q"))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_k"))))
          (Op.ref .real [] "b_b")))
        (Op.broadcast (Op.const 0.0) [BTL, BTS])),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q") (Op.ref .real [BK, BTS] "b_k"))
        (Op.ref .real [BTL, BTS] "d_s")),
    Stmt.assign .real [BTL, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTS, BV] "b_v"))),
    Stmt.assign .blockPtr [BK, BTS] "p_k"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [(0 : Nat), BTS]),
    Stmt.assign .blockPtr [BTS, BV] "p_v"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [BTS, (0 : Nat)]),
    Stmt.assign .nat [BTS] "o_k"
      (Op.add .nat Broadcast.scalarR (Op.ref .nat [BTS] "o_k") (Op.constNat BTS)) ]

set_option maxRecDepth 8000 in
/-- **Forward body split (by `rfl`).** Twenty-seven top-level statements. -/
theorem pra_fwd_body_eq (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    (pra_fwd_surface q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale B H T K V BTL BTS BK BV).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_kv" (Op.programId 0),
          Stmt.assign .nat [] "i_c" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .nat [] "NV"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
                (Op.constNat 1))
              (Op.constNat BV)),
          Stmt.assign .nat [] "i_k"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_v"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_h"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_bh")
              (Op.constNat H)),
          Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
                  (Op.const 1.0)))))),
          Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
          Stmt.assign .real [BTS] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTS)
                (Op.ref .nat [BTS] "o_k")))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .blockPtr [BTL, BK] "p_q"
            (Op.makeBlockPtrDynOffsets q
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
          Stmt.assign .blockPtr [BK, BTS] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [K, T] [BK, BTS] [s_qk_d, s_qk_t]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                Op.constNat 0]),
          Stmt.assign .blockPtr [BTS, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
              [T, V] [BTS, BV] [s_vo_t, s_vo_d]
              [Op.constNat 0,
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_q") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
              (Op.const scale)),
          Stmt.assign .real [BTL, BV] "b_o" (Op.full [BTL, BV] (Op.const 0)),
          Stmt.forRangeDyn "_i" (Op.constNat 0)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
            (Op.constNat BTS) (praFwdLoop1Body BTL BTS BK BV),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
          Stmt.assign .real [BTL] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTL))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BV] "b_o"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BV] "b_o")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_q"))),
          Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
          Stmt.assign .blockPtr [BK, BTS] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [K, T] [BK, BTS] [s_qk_d, s_qk_t]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)]),
          Stmt.assign .blockPtr [BTS, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
              [T, V] [BTS, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.forRangeDyn "_i"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
              (Op.constNat BTL))
            (Op.constNat BTS) (praFwdLoop2Body BTL BTS BK BV),
          Stmt.assign .blockPtr [BTL, BV] "p_o"
            (Op.makeBlockPtrDynOffsets o
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                    (Op.ref .nat [] "i_k")))
                (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
            (Op.ref .real [BTL, BV] "b_o") MaskOpt.none ] := by
  rfl

/-- The backward descending-loop body (respelled ascending; see the
preamble): the block index `i`, two pointer makes, two loads, the `d_q`
rescale of `b_do`, and the two decay-and-add accumulations. -/
def praBwdDescBody (q do_ : RegionName)
    (i_bh i_k i_v : Nat) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat T) (Op.constNat BTS))
                (Op.constNat 1))
              (Op.constNat BTS))
            (Op.constNat BTS))
          (Op.constNat BTS))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "j") (Op.constNat BTS))),
    Stmt.assign .blockPtr [BK, BTS] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
        [K, T] [BK, BTS] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK),
          Op.ref .nat [] "i"]),
    Stmt.assign .blockPtr [BV, BTS] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
        [V, T] [BV, BTS] [s_vo_d, s_vo_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV),
          Op.ref .nat [] "i"]),
    Stmt.assign .real [BK, BTS] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BV, BTS] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BV, BTS] "b_do"
      (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BV, BTS] "b_do")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "d_q"))),
    Stmt.assign .real [BTL, BV] "b_dv"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
        (Op.ref .real [] "d_b")),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_kd")
        (Op.ref .real [BK, BTS] "b_q")),
    Stmt.assign .real [BTL, BV] "b_dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s")
          (Op.transpose (Op.ref .real [BV, BTS] "b_do")))),
    Stmt.assign .real [BTL, BK] "b_dk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_dk")
        (Op.ref .real [] "d_b")),
    Stmt.assign .real [BTL, BTS] "b_ds"
      (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
        (Op.ref .real [BV, BTS] "b_do")),
    Stmt.assign .real [BTL, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_ds")
          (Op.transpose (Op.ref .real [BK, BTS] "b_q")))) ]

/-- The backward diagonal body: two pointer makes, two loads, the causal
mask, the scaled decay tile `d_s` (with the respelled subtraction and the
identity `b_b.to(tl.float32)` cast), both weighted contractions, the two
accumulations, and the `o_q += BTS` carry. -/
def praBwdDiagBody (q do_ : RegionName)
    (i_bh i_k i_v : Nat) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BTS] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
        [K, T] [BK, BTS] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK),
          Op.ref .nat [] "i"]),
    Stmt.assign .blockPtr [BV, BTS] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
        [V, T] [BV, BTS] [s_vo_d, s_vo_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV),
          Op.ref .nat [] "i"]),
    Stmt.assign .real [BK, BTS] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BV, BTS] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .bool [BTL, BTS] "m_s"
      (Op.le ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))),
    Stmt.assign .real [BTL, BTS] "d_s"
      (Op.mul .real Broadcast.scalarR
        (Op.where (Op.ref .bool [BTL, BTS] "m_s")
          (Op.exp2 (Op.mul .real Broadcast.scalarR
            (Op.natToReal (Op.sub .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))))
            (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [] "b_b"))))
          (Op.broadcast (Op.const 0.0) [BTL, BTS]))
        (Op.const scale)),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
          (Op.ref .real [BK, BTS] "b_q"))
        (Op.ref .real [BTL, BTS] "d_s")),
    Stmt.assign .real [BTL, BTS] "b_ds"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
          (Op.ref .real [BV, BTS] "b_do"))
        (Op.ref .real [BTL, BTS] "d_s")),
    Stmt.assign .real [BTL, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_ds")
          (Op.transpose (Op.ref .real [BK, BTS] "b_q")))),
    Stmt.assign .real [BTL, BV] "b_dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s")
          (Op.transpose (Op.ref .real [BV, BTS] "b_do")))),
    Stmt.assign .nat [BTS] "o_q"
      (Op.add .nat Broadcast.scalarR (Op.ref .nat [BTS] "o_q") (Op.constNat BTS)) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Twenty top-level statements. -/
theorem pra_bwd_body_eq (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    (pra_bwd_dkv_surface q k v do_ dk dv i_bh i_c i_k i_v i_h
        s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        B H T K V BTL BTS BK BV).toAlgKernel.body
      = [ Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat i_h))
                  (Op.const 1.0)))))),
          Stmt.assign .real [] "d_b"
            (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
              (Op.natToReal (Op.constNat BTS)))),
          Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_k"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_k") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BV] "b_v"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_v") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_dk" (Op.full [BTL, BK] (Op.const 0)),
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)),
          Stmt.assign .real [BTL] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTL)
                (Op.arange BTL)))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BK] "b_kd"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_k")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))),
          Stmt.assign .real [BTS] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTS))
              (Op.ref .real [] "b_b"))),
          Stmt.forRangeDyn "j" (Op.constNat 0)
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil
                      (Op.div .nat Broadcast.nil
                        (Op.sub .nat Broadcast.nil
                          (Op.add .nat Broadcast.nil (Op.constNat T) (Op.constNat BTS))
                          (Op.constNat 1))
                        (Op.constNat BTS))
                      (Op.constNat BTS))
                    (Op.constNat ((i_c + 1) * BTL)))
                  (Op.constNat BTS))
                (Op.constNat 1))
              (Op.constNat BTS))
            (Op.constNat 1)
            (praBwdDescBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d
              s_vo_h s_vo_t s_vo_d T K V BTL BTS BK BV),
          Stmt.assign .real [BTL, BK] "b_dk"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_dk")
              (Op.mul .real Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))
                (Op.const scale))),
          Stmt.assign .real [BTL, BV] "b_dv"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
              (Op.const scale)),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL),
          Stmt.forRangeDyn "i"
            (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
            (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
            (praBwdDiagBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d
              s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV),
          Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.store .real [BTL, BK]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk") [0, 1])
            (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv") [0, 1])
            (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ] := by
  rfl

/-! ## Closed-form specification (forward)

Every load is boundary-checked, so the guarded value functions bake the
window directly in: out-of-window lanes are `0` and the closed forms hold
for arbitrary ragged tails. The decay values mirror the walk exactly:
`exp2 x = Real.exp (x·log 2)` and `log2 y = Real.log y / log 2` are the
semantics' projections (the `chunk_retention` convention). -/

/-- `NV = tl.cdiv(V, BV)` as the prologue computes it. -/
def praNV (V BV : Nat) : Nat := (V + BV - 1) / BV

/-- `i_k = i_kv // NV`. -/
def praIk (s : BlockState) (V BV : Nat) : Nat := s.pids 0 / praNV V BV

/-- `i_v = i_kv % NV`. -/
def praIv (s : BlockState) (V BV : Nat) : Nat := s.pids 0 % praNV V BV

/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))` with
`i_h = i_bh % H`, exactly as the walk computes it. -/
noncomputable def praBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2

/-- The retention decay weight `γ^n = 2^(n·b_b)`, in the walk's exact form. -/
noncomputable def praW (β : ℝ) (n : Nat) : ℝ :=
  Real.exp ((n : ℝ) * β * Real.log 2)

theorem praW_zero (β : ℝ) : praW β 0 = 1 := by simp [praW]

theorem praW_add (β : ℝ) (m n : Nat) :
    praW β (m + n) = praW β m * praW β n := by
  unfold praW
  rw [← Real.exp_add]
  congr 1
  push_cast
  ring

/-- The scaled, guarded `b_q` lane `(a, e)`: row `i_c·BTL + a` of the `(T, K)`
parent, column `i_k·BK + e`, times `scale`; `0` outside the window. -/
noncomputable def praQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a e : Nat) : ℝ :=
  if s.pids 1 * BTL + a < T ∧ praIk s V BV * BK + e < K then
    s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + a) * s_qk_t
      + (praIk s V BV * BK + e) * s_qk_d) * scale
  else 0

/-- The guarded transposed `k` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
noncomputable def praKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV : Nat) (e t : Nat) : ℝ :=
  if praIk s V BV * BK + e < K ∧ t < T then
    s.readMem k (s.pids 2 * s_qk_h + (praIk s V BV * BK + e) * s_qk_d
      + t * s_qk_t)
  else 0

/-- The guarded `v` lane `(t, p)` at absolute key `t`. -/
noncomputable def praVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV : Nat) (t p : Nat) : ℝ :=
  if t < T ∧ praIv s V BV * BV + p < V then
    s.readMem v (s.pids 2 * s_vo_h + t * s_vo_t
      + (praIv s V BV * BV + p) * s_vo_d)
  else 0

/-- The retention score at `(row a, key t)`: `(scale·q) · k` over the `BK`
head window. -/
noncomputable def praScore (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a t : Nat) : ℝ :=
  ∑ e : Fin BK,
    praQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a e.val
      * praKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV e.val t

/-- The non-diagonal accumulator after the streaming loop has consumed keys
`[0, n)`, **anchored at `n`**: `Σ_t 2^((n − t)·b_b) · score · v` — the
kernel's per-block decay `b_o *= 2^(b_b·BTS)` shifts the anchor by `BTS`
each iteration. -/
noncomputable def praOAcc (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (n a p : Nat) : ℝ :=
  ∑ t ∈ Finset.range n,
    praW (praBeta s H) (n - t)
      * praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p

/-- The diagonal (causally masked) accumulator over keys `[i_c·BTL, i)`:
kept iff `t ≤ i_c·BTL + a`, with the full row-anchored decay
`2^((i_c·BTL + a − t)·b_b)`. -/
noncomputable def praODiag (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (i a p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico (s.pids 1 * BTL) i,
    if t ≤ s.pids 1 * BTL + a then
      praW (praBeta s H) (s.pids 1 * BTL + a - t)
        * praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
    else 0

/-- **The stored `o` lane** — causal retention attention as one sum:
`Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) · v[t,p]` over
the keys `[0, (i_c+1)·BTL)` the program consumes. -/
noncomputable def praOOut (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (a p : Nat) : ℝ :=
  ∑ t ∈ Finset.range ((s.pids 1 + 1) * BTL),
    if t ≤ s.pids 1 * BTL + a then
      praW (praBeta s H) (s.pids 1 * BTL + a - t)
        * praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
    else 0

/-- `praOOut` splits into the `d_q`-rescaled streamed accumulator plus the
diagonal block (the shape the walk actually produces). -/
theorem praOOut_split (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (a p : Nat) :
    praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        H T K V BTL BK BV a p
      = praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BTL BK BV (s.pids 1 * BTL) a p
            * praW (praBeta s H) a
          + praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
              H T K V BTL BK BV ((s.pids 1 + 1) * BTL) a p := by
  unfold praOOut praOAcc praODiag
  have hle : s.pids 1 * BTL ≤ (s.pids 1 + 1) * BTL := by
    have : (s.pids 1 + 1) * BTL = s.pids 1 * BTL + BTL := by ring
    omega
  rw [Finset.range_eq_Ico,
    ← Finset.sum_Ico_consecutive _ (Nat.zero_le (s.pids 1 * BTL)) hle,
    Finset.sum_mul, ← Finset.range_eq_Ico]
  congr 1
  refine Finset.sum_congr rfl fun t ht => ?_
  have htlt : t < s.pids 1 * BTL := Finset.mem_range.mp ht
  rw [if_pos (by omega)]
  rw [show s.pids 1 * BTL + a - t = a + (s.pids 1 * BTL - t) by omega,
    praW_add]
  ring

/-- The `o` store base `(i_bh + B·H·i_k) · s_vo_h`. -/
def praOBase (s : BlockState) (B H s_vo_h V BV : Nat) : Nat :=
  (s.pids 2 + B * H * praIk s V BV) * s_vo_h

/-- The `o` store address at lane `(a, p)` (strides `(s_vo_t, 1)` after the
`s_vo_d = 1` substitution). -/
def praOOffset (s : BlockState) (B H s_vo_h s_vo_t s_vo_d V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  praOBase s B H s_vo_h V BV + (s.pids 1 * BTL + idx.1.val) * s_vo_t
    + (praIv s V BV * BV + idx.2.1.val) * s_vo_d

/-- An `o` store lane is *active* when it maps inside the `T × V` window. -/
def praOActive (s : BlockState) (T V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Prop :=
  s.pids 1 * BTL + idx.1.val < T ∧ praIv s V BV * BV + idx.2.1.val < V

/-! ## Eval recipes (local copies from the `parallel_attention` /
`chunk_retention` pairs, plus the decay family). -/

private theorem pra_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (rowOffOp colOffOp : Op .nat [])
    (parentShape blockShape strides : List Nat)
    (base rowOff colOff : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hrow : evalOp rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOp colOffOp s = some (Tile.scalar colOff)) :
    evalOp (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [rowOffOp, colOffOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [rowOff, colOff]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOp, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.nil_append,
    List.cons_append]

private theorem pra_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BR BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs .blockPtr [BR, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BR, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load .real
      (MemAccess.blockPtr (Op.ref .blockPtr [BR, BS] name) [0, 1]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BR, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
  simp only [evalOp, evalOp_ref, hreg, bind, Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, BlockState.readMemValue_real]
  by_cases h : rowOff + i.val < rows ∧ colOff + j.val < cols
  · simp only [h, and_self, decide_true, if_true]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier]
    rfl

private theorem pra_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `NV = tl.cdiv(V, BV)` on spliced constants. -/
private theorem pra_NV_eval (s : BlockState) (V BV : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
          (Op.constNat 1))
        (Op.constNat BV)) s
      = some (Tile.scalar (praNV V BV)) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

/-- `i_k = i_kv // NV` on refs. -/
private theorem pra_floorDiv_refs_eval (s : BlockState) (na nb : RegName)
    (va vb : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va))
    (hb : s.regs .nat [] nb = some (Tile.scalar vb)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.ref .nat [] nb)) s
      = some (Tile.scalar (va / vb)) := by
  rw [evalOp_floorDiv]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `i_v = i_kv % NV` on refs. -/
private theorem pra_mod_refs_eval (s : BlockState) (na nb : RegName)
    (va vb : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va))
    (hb : s.regs .nat [] nb = some (Tile.scalar vb)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.ref .nat [] nb)) s
      = some (Tile.scalar (va % vb)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `i_h = i_bh % H` on a ref and a spliced constant. -/
private theorem pra_mod_ref_const_eval (s : BlockState) (na : RegName)
    (va c : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.constNat c)) s
      = some (Tile.scalar (va % c)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, evalOp_constNat, ha, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem pra_zeros_eval (sh : TileShape) (t : BlockState) :
    evalOp (Op.full sh (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real sh) := by
  simp [evalOp_full, evalOp_const]

private theorem pra_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

private theorem pra_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
    (b : Tile .real [K, N]) (m : Fin M) (n : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ e : Fin K, a.data (m, e, PUnit.unit) = some (fa e))
    (hb : ∀ e : Fin K, b.data (e, n, PUnit.unit) = some (fb e)) :
    (Tile.dot [] a b).data (m, n, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fa e * fb e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (m, e, PUnit.unit))
          (b.data (e, n, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
          (fun e => (some (fa e * fb e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [ha e, hb e]; rfl)]
  exact withBot_sum_some _

private theorem pra_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

private theorem pra_addTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add .real bc x y) t
      = some (Tile.bop NumericDType.real.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem pra_mulTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul .real bc x y) t
      = some (Tile.bop NumericDType.real.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- Advance a 2-D block pointer along the column axis. -/
private theorem pra_advance_col_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff colOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, colOff] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [(0 : Nat), d]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, colOff + d] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

/-- Advance a 2-D block pointer along the row axis. -/
private theorem pra_advance_row_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff colOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, colOff] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, (0 : Nat)]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff + d, colOff] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

/-! ### The decay family -/

/-- An arange register tile. -/
def praArangeTile (n : Nat) : Tile .nat [n] :=
  Tile.vec fun i : Fin n => (i.val : Nat)

/-- The decay exponent statement lands on `praBeta` (forward spelling, on the
`i_h` register). -/
private theorem pra_bb_eval (s t : BlockState) (H : Nat)
    (hih : t.regs .nat [] "i_h" = some (Tile.scalar (s.pids 2 % H))) :
    evalOp (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
        (Op.exp2 (Op.sub .real Broadcast.nil
          (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
          (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
            (Op.const 1.0)))))) t
      = some (Tile.scalar (some (praBeta s H))) := by
  simp only [evalOp, evalOp_ref, hih, bind, Option.bind]
  rfl

/-- The `d_h` tile: lane `c` holds `2^((BTS - c)·b_b)` (int-exact: `c < BTS`
so the ℕ subtraction never truncates). -/
noncomputable def praDhTile (s : BlockState) (H BTS : Nat) : Tile .real [BTS] :=
  ⟨fun idx => some (praW (praBeta s H) (BTS - idx.1.val))⟩

/-- The `d_h` statement lands on `praDhTile`. -/
private theorem pra_dh_eval (s t : BlockState) (H BTS : Nat)
    (hok : t.regs .nat [BTS] "o_k" = some (praArangeTile BTS))
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTS)
          (Op.ref .nat [BTS] "o_k")))
        (Op.ref .real [] "b_b"))) t
      = some (praDhTile s H BTS) := by
  simp only [evalOp, evalOp_ref, hok, hbb, bind, Option.bind]
  rfl

/-- The `d_q` tile: lane `a` holds `2^(a·b_b)`. -/
noncomputable def praDqTile (s : BlockState) (H BTL : Nat) : Tile .real [BTL] :=
  ⟨fun idx => some (praW (praBeta s H) idx.1.val)⟩

/-- The `d_q` statement (inline `tl.arange`) lands on `praDqTile`. -/
private theorem pra_dq_eval (s t : BlockState) (H BTL : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTL))
        (Op.ref .real [] "b_b"))) t
      = some (praDqTile s H BTL) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The loop-1 inline decay `tl.math.exp2(b_b * BTS)` evaluates to
`2^(BTS·b_b)` (walk-order product; `= praW β BTS` by `praW_comm`). -/
private theorem pra_dbInline_eval (s t : BlockState) (H BTS : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
        (Op.natToReal (Op.constNat BTS)))) t
      = some (Tile.scalar (some (Real.exp (praBeta s H * ((BTS : Nat) : ℝ)
          * Real.log 2)))) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The walk-order product is `praW`. -/
theorem praW_comm (β : ℝ) (n : Nat) :
    Real.exp (β * (n : ℝ) * Real.log 2) = praW β n := by
  unfold praW
  ring_nf

/-! ## Block store machinery (same-region masked frame; the strides-`[σ, 1]`
form applies after the `s_vo_d = 1` / `s_qk_d = 1` substitutions). -/

/-- One boundary-checked `[BR, BS]` block store's lane address. -/
def praStoreAddr (base σ rowOff colOff : Nat) (BR BS : Nat)
    (i : TileIndex [BR, BS]) : Nat :=
  base + (rowOff + i.1.val) * σ + (colOff + i.2.1.val) * 1

/-- The post-store state: every in-window lane written. -/
noncomputable def praStoreState (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BR, BS]).foldl
    (fun acc i => if (rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
        then acc.writeMem rg (praStoreAddr base σ rowOff colOff BR BS i)
          (f i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Block store step (eq).** -/
theorem praStore_step_eq (sin : BlockState) (rg : RegionName)
    (bname pname : RegName) (base σ rowOff colOff K V BR BS : Nat)
    (f : Nat → Nat → ℝ) (bT : Tile .real [BR, BS])
    (hbf : ∀ e p, bT.data (e, p, PUnit.unit) = some (f e.val p.val))
    (hb : sin.regs .real [BR, BS] bname = some bT)
    (hp : sin.regs .blockPtr [BR, BS] pname = some
      ⟨fun _ => BlockPtr.mk rg base [K, V] [BR, BS] [σ, 1]
        [rowOff, colOff]⟩) :
    stepStmt (Stmt.store .real [BR, BS]
        (MemAccess.blockPtr (Op.ref .blockPtr [BR, BS] pname) [0, 1])
        (Op.ref .real [BR, BS] bname) MaskOpt.none) sin
      = some (praStoreState sin rg base σ rowOff colOff K V BR BS f) := by
  unfold stepStmt praStoreState
  simp only [evalOp_ref, hb, hp]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [BR, BS])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, praStoreAddr]
  by_cases hbnd : rowOff + e.val < K ∧ colOff + p.val < V
  · simp only [hbnd, BlockState.writeMemTyped_real, hbf, Nat.mul_one]
    rfl
  · simp only [hbnd, decide_false, Bool.false_eq_true, if_false]

private theorem pra_block_index_inj {Q j c A B : Nat} (hA : A < Q) (hB : B < Q)
    (heq : j * Q + A = c * Q + B) : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A) hA
  have hj : (j * Q + A) / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B) / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- One block's store lanes are pairwise distinct once a `BS` segment fits
under the row stride. -/
theorem praStoreAddr_injective (base σ rowOff colOff BR BS : Nat)
    (hBSσ : BS ≤ σ) :
    Function.Injective (praStoreAddr base σ rowOff colOff BR BS) := by
  rintro ⟨e, p, u⟩ ⟨e', p', u'⟩ heq
  simp only [praStoreAddr] at heq
  have hp := p.isLt
  have hp' := p'.isLt
  have h2 : (rowOff + e.val) * σ + p.val = (rowOff + e'.val) * σ + p'.val := by
    omega
  have hlt : p.val < σ := by omega
  have hlt' : p'.val < σ := by omega
  have hjj : rowOff + e.val = rowOff + e'.val :=
    pra_block_index_inj hlt hlt' h2
  have he : e = e' := Fin.ext (by omega)
  have hpv : p = p' := Fin.ext (by
    have hσ : (rowOff + e.val) * σ = (rowOff + e'.val) * σ := by rw [hjj]
    omega)
  subst he
  subst hpv
  rfl

set_option maxHeartbeats 4000000 in
/-- **Block store readback** (mask-restricted same-region frame). -/
theorem praStore_step_props (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ)
    (hInj : Function.Injective (praStoreAddr base σ rowOff colOff BR BS)) :
    (praStoreState sin rg base σ rowOff colOff K V BR BS f).pids = sin.pids
      ∧ (praStoreState sin rg base σ rowOff colOff K V BR BS f).regs = sin.regs
      ∧ (∀ idx : TileIndex [BR, BS],
          (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
          (praStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg
              (praStoreAddr base σ rowOff colOff BR BS idx)
            = f idx.1.val idx.2.1.val)
      ∧ (∀ rg' off, rg' ≠ rg →
          (praStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg' off
            = sin.readMem rg' off)
      ∧ (∀ off, (∀ idx : TileIndex [BR, BS],
            (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
            off ≠ praStoreAddr base σ rowOff colOff BR BS idx) →
          (praStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold praStoreState
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name
    rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
      (praStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      hInj idx
    rw [h, if_pos ⟨h1, h2⟩]
  · intro rg' off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      rg (praStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin rg' off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      rg (praStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin off (fun i _ hPi => hoff i hPi)

/-! ## Value tiles and load bridges (forward) -/

private theorem pra_readMem_congr (s s0 : BlockState) (h : s.mem = s0.mem)
    (rg : RegionName) (off : Nat) : s.readMem rg off = s0.readMem rg off := by
  unfold BlockState.readMem
  rw [h]

/-- The scaled loaded `b_q` tile. -/
noncomputable def praQTile (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat) :
    Tile .real [BTL, BK] :=
  ⟨fun idx => some (praQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
    idx.1.val idx.2.1.val)⟩

/-- The loaded `b_k` block at key-column offset `i`: lane `(e, s') ↦ k[e, i + s']`. -/
noncomputable def praKTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV BTS : Nat) (i : Nat) :
    Tile .real [BK, BTS] :=
  ⟨fun idx => some (praKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV
    idx.1.val (i + idx.2.1.val))⟩

/-- The loaded `b_v` block at key-row offset `i`. -/
noncomputable def praVTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV BTS : Nat) (i : Nat) :
    Tile .real [BTS, BV] :=
  ⟨fun idx => some (praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV
    (i + idx.1.val) idx.2.1.val)⟩

/-- Loading `p_k` at offsets `[i_k·BK, i]` lands on `praKTile … i`. -/
private theorem pra_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BK, BTS] name = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [praIk s V BV * BK, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (praKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i) := by
  rw [pra_load_bp_2d k sin name (s.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
    (praIk s V BV * BK) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, c, u⟩ := idx
  simp only [praKTile, praKGuarded]
  by_cases h : praIk s V BV * BK + e.val < K ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_v` at offsets `[i, i_v·BV]` lands on `praVTile … i`. -/
private theorem pra_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTS, BV] name = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, praIv s V BV * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTS, BV] name) [0, 1])
        MaskOpt.none) sin
      = some (praVTile s v s_vo_h s_vo_t s_vo_d T V BV BTS i) := by
  rw [pra_load_bp_2d v sin name (s.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
    i (praIv s V BV * BV) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨c, p, u⟩ := idx
  simp only [praVTile, praVGuarded]
  by_cases h : i + c.val < T ∧ praIv s V BV * BV + p.val < V
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-! ## Forward per-statement recipes -/

/-- `tl.dot(b_q, b_k)` lands lanewise on `praScore · (i + s')`. -/
private theorem pra_score_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV BTS : Nat)
    (i : Nat)
    (hq : sin.regs .real [BTL, BK] "b_q"
      = some (praQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV))
    (hk : sin.regs .real [BK, BTS] "b_k"
      = some (praKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q")
        (Op.ref .real [BK, BTS] "b_k")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
            idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hq)
    (by rw [evalOp_ref]; exact hk)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rw [pra_dot2d_elem _ _ a c
    (fun e => praQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      a.val e.val)
    (fun e => praKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV
      e.val (i + c.val))
    (fun e => rfl) (fun e => rfl)]
  rfl

/-- `b_s = tl.dot(b_q, b_k) * d_h[None, :]`: the streamed decay-weighted
score. -/
private theorem pra_sWeighted1_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (H T K V BTL BK BV BTS : Nat)
    (i : Nat)
    (hq : sin.regs .real [BTL, BK] "b_q"
      = some (praQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV))
    (hk : sin.regs .real [BK, BTS] "b_k"
      = some (praKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i))
    (hdh : sin.regs .real [BTS] "d_h" = some (praDhTile s H BTS)) :
    evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q")
          (Op.ref .real [BK, BTS] "b_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "d_h"))) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
              idx.1.val (i + idx.2.1.val)
            * praW (praBeta s H) (BTS - idx.2.1.val))⟩ := by
  erw [pra_mulTile_eval (Broadcast.consR (Broadcast.consSame Broadcast.nil)) _ _ sin _ _
    (pra_score_eval s sin q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV BTS i hq hk)
    (evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdh)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rfl

/-- `b_o = b_o * tl.math.exp2(b_b * BTS)`: the per-block accumulator decay. -/
private theorem pra_oDecay_eval (s sin : BlockState) (H BTL BTS BV : Nat)
    (g : Nat → Nat → ℝ)
    (ho : sin.regs .real [BTL, BV] "b_o"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hbb : sin.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s H)))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_o")
        (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
          (Op.natToReal (Op.constNat BTS))))) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val
            * Real.exp (praBeta s H * ((BTS : Nat) : ℝ) * Real.log 2))⟩ := by
  rw [pra_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (by rw [evalOp_ref]; exact ho) (pra_dbInline_eval s sin H BTS hbb)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, p, u⟩ := idx
  rfl

/-- `b_o (+)= tl.dot(b_s, b_v)` on all-`some` tiles: lanewise contraction. -/
private theorem pra_oAdd_eval (sin : BlockState) (BTL BTS BV : Nat)
    (g : Nat → Nat → ℝ) (f : Nat → Nat → ℝ) (w : Nat → Nat → ℝ)
    (ho : sin.regs .real [BTL, BV] "b_o"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩)
    (hv : sin.regs .real [BTS, BV] "b_v"
      = some ⟨fun idx => some (w idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s")
          (Op.ref .real [BTS, BV] "b_v"))) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin BTS, f idx.1.val c.val * w c.val idx.2.1.val)⟩ := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s")
      (Op.ref .real [BTS, BV] "b_v")) sin
      = some (Tile.dot [] (⟨fun idx => some (f idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BTS])
          (⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [BTS, BV])) :=
    pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hs)
      (by rw [evalOp_ref]; exact hv)
  erw [pra_addTile_eval _ _ _ sin _ _ (by rw [evalOp_ref]; exact ho) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [pra_dot2d_elem _ _ a p (fun c => f a.val c.val) (fun c => w c.val p.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

private theorem pra_evalOp_ge_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- The `o_q[:, None] >= o_k[None, :]` mask at diagonal offset `off`. -/
def praMaskTile (BTL BTS off : Nat) : Tile .bool [BTL, BTS] :=
  ⟨fun idx => decide (idx.2.1.val + off ≤ idx.1.val)⟩

private theorem pra_msGe_eval (sin : BlockState) (BTL BTS off : Nat)
    (hoq : sin.regs .nat [BTL] "o_q" = some (Tile.vec fun r => r.val))
    (hok : sin.regs .nat [BTS] "o_k" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_q"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_k"))) sin
      = some (praMaskTile BTL BTS off) := by
  rw [pra_evalOp_ge_def]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoq,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hok]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  show decide (a.val ≥ c.val + off) = decide (c.val + off ≤ a.val)
  rfl

/-- `d_s = tl.where(m_s, exp2((o_q − o_k)·b_b), 0)`: the causal decay tile
at diagonal offset `off` (kept lanes hold `2^((a − (c+off))·b_b)`; the
ℕ-truncated masked-off lanes are replaced by the hard `0`). -/
private theorem pra_dsTile_eval (s sin : BlockState) (H BTL BTS off : Nat)
    (hm : sin.regs .bool [BTL, BTS] "m_s" = some (praMaskTile BTL BTS off))
    (hoq : sin.regs .nat [BTL] "o_q" = some (Tile.vec fun r => r.val))
    (hok : sin.regs .nat [BTS] "o_k" = some (Tile.vec fun r => r.val + off))
    (hbb : sin.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s H)))) :
    evalOp (Op.where (Op.ref .bool [BTL, BTS] "m_s")
        (Op.exp2 (Op.mul .real Broadcast.scalarR
          (Op.natToReal (Op.sub .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_q"))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_k"))))
          (Op.ref .real [] "b_b")))
        (Op.broadcast (Op.const 0.0) [BTL, BTS])) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (if idx.2.1.val + off ≤ idx.1.val
            then praW (praBeta s H) (idx.1.val - (idx.2.1.val + off))
            else 0)⟩ := by
  rw [evalOp_where]
  simp only [evalOp.eq_def, evalOp_ref, evalOp_const, hm, hoq, hok, hbb,
    Option.bind, Option.map, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rw [Tile.select_data]
  simp only [praMaskTile]
  by_cases h : c.val + off ≤ a.val
  · rw [if_pos (by simpa using h), if_pos h]
    rfl
  · rw [if_neg (by simpa using h), if_neg h]
    norm_num

/-- `b_s = tl.dot(b_q, b_k) * d_s` on an all-`some` decay tile. -/
private theorem pra_sWeighted2_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV BTS : Nat)
    (i : Nat) (fd : Nat → Nat → ℝ)
    (hq : sin.regs .real [BTL, BK] "b_q"
      = some (praQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV))
    (hk : sin.regs .real [BK, BTS] "b_k"
      = some (praKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i))
    (hd : sin.regs .real [BTL, BTS] "d_s"
      = some ⟨fun idx => some (fd idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q")
          (Op.ref .real [BK, BTS] "b_k"))
        (Op.ref .real [BTL, BTS] "d_s")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
              idx.1.val (i + idx.2.1.val)
            * fd idx.1.val idx.2.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _
    (pra_score_eval s sin q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV BTS i hq hk)
    (by rw [evalOp_ref]; exact hd)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rfl

/-- `o_k += BTS` shifts the arange window. -/
private theorem pra_okAdd_eval (sin : BlockState) (BTS off : Nat)
    (hok : sin.regs .nat [BTS] "o_k" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BTS] "o_k")
        (Op.constNat BTS)) sin
      = some (Tile.vec fun r : Fin BTS => r.val + (off + BTS)) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hok, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, u⟩ := idx
  show r.val + off + BTS = r.val + (off + BTS)
  omega

/-- `b_q = (b_q * scale)` after the boundary-checked load: the scaled guarded
tile. -/
private theorem pra_qScale_eval (s sin : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (hq : sin.regs .real [BTL, BK] "b_q" = some
      ⟨fun idx : TileIndex [BTL, BK] =>
        if (s.pids 1 * BTL + idx.1.val < T ∧ praIk s V BV * BK + idx.2.1.val < K) then
          some (s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + idx.1.val) * s_qk_t
            + (praIk s V BV * BK + idx.2.1.val) * s_qk_d))
        else some 0⟩) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
        (Op.const scale)) sin
      = some (praQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) := by
  rw [pra_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (by rw [evalOp_ref]; exact hq) (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    praQTile, praQGuarded]
  by_cases h : s.pids 1 * BTL + a.val < T ∧ praIk s V BV * BK + e.val < K
  · rw [if_pos h, if_pos h]
    rfl
  · rw [if_neg h, if_neg h]
    show Option.map₂ _ (some (0 : ℝ)) (some scale) = some 0
    show some ((0 : ℝ) * scale) = some 0
    rw [zero_mul]

/-- `b_o *= d_q[:, None]`: the mid-section row rescale. -/
private theorem pra_oRescale_eval (s sin : BlockState) (H BTL BV : Nat)
    (g : Nat → Nat → ℝ)
    (ho : sin.regs .real [BTL, BV] "b_o"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hdq : sin.regs .real [BTL] "d_q" = some (praDqTile s H BTL)) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_o")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_q"))) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val
            * praW (praBeta s H) idx.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact ho)
    (evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdq)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, p, u⟩ := idx
  rfl

/-! ## Block-sum split lemmas for the specs -/

private theorem pra_sum_range_add_block (n BTS : Nat) (f : Nat → ℝ) :
    ∑ t ∈ Finset.range (n + BTS), f t
      = (∑ t ∈ Finset.range n, f t) + ∑ c : Fin BTS, f (n + c.val) := by
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun c => f (n + c))]

private theorem pra_sum_Ico_add_block (lo i BTS : Nat) (hlo : lo ≤ i)
    (f : Nat → ℝ) :
    ∑ t ∈ Finset.Ico lo (i + BTS), f t
      = (∑ t ∈ Finset.Ico lo i, f t) + ∑ c : Fin BTS, f (i + c.val) := by
  rw [← Finset.sum_Ico_consecutive _ hlo (Nat.le_add_right i BTS)]
  congr 1
  rw [Finset.sum_Ico_eq_sum_range]
  simp only [Nat.add_sub_cancel_left]
  rw [Fin.sum_univ_eq_sum_range (fun c => f (i + c))]

/-! ## Spec step lemmas -/

private theorem pra_dvd_step_le {BTS i X : Nat} (hd : BTS ∣ i) (hX : BTS ∣ X)
    (h : i < X) : i + BTS ≤ X := by
  obtain ⟨a, rfl⟩ := hd
  obtain ⟨b, rfl⟩ := hX
  have hab : a < b := by
    by_contra hab
    rw [Nat.not_lt] at hab
    exact absurd (Nat.mul_le_mul_left BTS hab) (Nat.not_le.mpr h)
  calc BTS * a + BTS = BTS * (a + 1) := by ring
    _ ≤ BTS * b := Nat.mul_le_mul_left BTS hab

/-- One streamed block: decay the anchored accumulator by `2^(BTS·b_b)`
(walk order) and add the block's `d_h`-weighted contribution. -/
private theorem praOAcc_step (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (i BTS a p : Nat) :
    praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        H T K V BTL BK BV i a p
        * Real.exp (praBeta s H * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS,
          praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * praW (praBeta s H) (BTS - c.val)
            * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
      = praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BK BV (i + BTS) a p := by
  unfold praOAcc
  rw [praW_comm, pra_sum_range_add_block, Finset.sum_mul]
  congr 1
  · refine Finset.sum_congr rfl fun t ht => ?_
    have htlt : t < i := Finset.mem_range.mp ht
    rw [show i + BTS - t = (i - t) + BTS from by omega, praW_add]
    ring
  · refine Finset.sum_congr rfl fun c _ => ?_
    rw [show i + BTS - (i + c.val) = BTS - c.val from by omega]
    ring

/-- One diagonal block appended to the masked accumulator. -/
private theorem praODiag_step (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (i BTS a p : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        H T K V BTL BK BV i a p
      + ∑ c : Fin BTS,
          (if c.val + (i - s.pids 1 * BTL) ≤ a then
            praW (praBeta s H) (a - (c.val + (i - s.pids 1 * BTL)))
              * praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
          else 0)
      = praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BK BV (i + BTS) a p := by
  unfold praODiag
  rw [pra_sum_Ico_add_block _ _ _ hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (c.val + (i - s.pids 1 * BTL) ≤ a) ↔ (i + c.val ≤ s.pids 1 * BTL + a) := by
    omega
  by_cases h : i + c.val ≤ s.pids 1 * BTL + a
  · rw [if_pos h, if_pos (hcond.mpr h)]
    rw [show s.pids 1 * BTL + a - (i + c.val)
      = a - (c.val + (i - s.pids 1 * BTL)) from by omega]
  · rw [if_neg h, if_neg (fun hc => h (hcond.mp hc))]

/-- The walk-shaped diagonal step: the `b_o` clause of the loop-2 invariant
absorbs one masked block. -/
private theorem praODiag_step' (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BK BV : Nat) (i BTS a p : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    (praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        H T K V BTL BK BV (s.pids 1 * BTL) a p
        * praW (praBeta s H) a
      + praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BK BV i a p)
      + ∑ c : Fin BTS,
          praScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * (if c.val + (i - s.pids 1 * BTL) ≤ a then
                  praW (praBeta s H) (a - (c.val + (i - s.pids 1 * BTL)))
                else 0)
            * praVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
      = praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BK BV (s.pids 1 * BTL) a p
          * praW (praBeta s H) a
        + praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BTL BK BV (i + BTS) a p := by
  rw [add_assoc]
  congr 1
  rw [← praODiag_step s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
    H T K V BTL BK BV i BTS a p hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  by_cases h : c.val + (i - s.pids 1 * BTL) ≤ a
  · rw [if_pos h, if_pos h]
    ring
  · rw [if_neg h, if_neg h, mul_zero, zero_mul]

/-! ## The forward non-diagonal loop -/

/-- The forward loop-1 invariant at counter `i` (the absolute key offset):
the anchored decayed accumulator plus the decay registers `b_b`/`d_h`. -/
def praInv1 (s0 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧
  BTS ∣ i ∧ i ≤ s0.pids 1 * BTL ∧
  s.regs .nat [] "i_c" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2)) ∧
  s.regs .nat [] "i_k" = some (Tile.scalar (praIk s0 V BV)) ∧
  s.regs .nat [] "i_v" = some (Tile.scalar (praIv s0 V BV)) ∧
  s.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s0 H))) ∧
  s.regs .real [BTS] "d_h" = some (praDhTile s0 H BTS) ∧
  s.regs .real [BTL, BK] "b_q"
    = some (praQTile s0 q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) ∧
  s.regs .real [BTL, BV] "b_o"
    = some ⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale H T K V BTL BK BV i idx.1.val idx.2.1.val)⟩ ∧
  s.regs .blockPtr [BK, BTS] "p_k"
    = some ⟨fun _ => BlockPtr.mk k (s0.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [praIk s0 V BV * BK, i]⟩ ∧
  s.regs .blockPtr [BTS, BV] "p_v"
    = some ⟨fun _ => BlockPtr.mk v (s0.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, praIv s0 V BV * BV]⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One non-diagonal block.** -/
theorem praFwdLoop1_step (s0 sin : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat) (i : Nat)
    (hBTS : BTL % BTS = 0)
    (hlt : i < s0.pids 1 * BTL)
    (hInv : praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      H T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (praFwdLoop1Body BTL BTS BK BV)
        (sin.setReg "_i" .nat [] (Tile.scalar i)) = some s'
      ∧ praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hpids, hmem, hdvd, hle, hic, hibh, hik, hiv, hbb, hdh, hbq, hbo,
    hpk, hpv⟩ := hInv
  unfold praFwdLoop1Body
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_kLoad_eq s0 _ k "p_k" s_qk_h s_qk_t s_qk_d T K V BK BV BTS i
      (by simpa using hmem) (by simpa using hpk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_vLoad_eq s0 _ v "p_v" s_vo_h s_vo_t s_vo_d T V BV BTS i
      (by simpa using hmem) (by simpa using hpv)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_sWeighted1_eval s0 _ q k s_qk_h s_qk_t s_qk_d scale H T K V BTL BK BV
      BTS i (by simpa using hbq) (by simp [praKTile]) (by simpa using hdh)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_oDecay_eval s0 _ H BTL BTS BV
      (fun a p => praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale H T K V BTL BK BV i a p)
      (by simpa using hbo) (by simpa using hbb)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_oAdd_eval _ BTL BTS BV
      (fun a p => praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale H T K V BTL BK BV i a p
        * Real.exp (praBeta s0 H * ((BTS : Nat) : ℝ) * Real.log 2))
      (fun a c => praScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          a (i + c)
        * praW (praBeta s0 H) (BTS - c))
      (fun c p => praVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c) p)
      (by simp) (by simp) (by simp [praVTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_advance_col_eval _ k (s0.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
      (praIk s0 V BV * BK) i BTS "p_k" (by simp [hpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_advance_row_eval _ v (s0.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
      i (praIv s0 V BV * BV) BTS "p_v" (by simp [hpv])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · exact Dvd.dvd.add hdvd dvd_rfl
  · exact pra_dvd_step_le hdvd
      (Dvd.dvd.mul_left (Nat.dvd_of_mod_eq_zero hBTS) (s0.pids 1)) hlt
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbb
  · simpa using hdh
  · simpa using hbq
  · rw [show (⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale H T K V BTL BK BV (i + BTS)
          idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale H T K V BTL BK BV i idx.1.val idx.2.1.val
            * Real.exp (praBeta s0 H * ((BTS : Nat) : ℝ) * Real.log 2)
          + ∑ c : Fin BTS,
              praScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                  idx.1.val (i + c.val)
                * praW (praBeta s0 H) (BTS - c.val)
                * praVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c.val)
                    idx.2.1.val)⟩
      from Tile.ext fun idx =>
        congrArg some (praOAcc_step s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d scale H T K V BTL BK BV i BTS idx.1.val idx.2.1.val).symm]
    simp
  · simp
  · simp

set_option maxHeartbeats 4000000 in
/-- **The full non-diagonal loop**: from the prologue state to keys
`[0, i_c·BTL)` consumed. -/
theorem praFwdLoop1_run (s0 sPre : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      H T K V BTL BTS BK BV 0 sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "_i" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.constNat BTS) (praFwdLoop1Body BTL BTS BK BV)) sPre = some sL
      ∧ praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV (s0.pids 1 * BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "_i")
      (P := fun i s => praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale H T K V BTL BTS BK BV i s)
      (evalOp_constNat 0 sPre)
      (pra_mulConst_eval sPre "i_c" (s0.pids 1) BTL hInv.2.2.2.2.1)
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        praFwdLoop1_step s0 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale H T K V BTL BTS BK BV i hBTS hi hPi)
  have hfin : final = s0.pids 1 * BTL := le_antisymm hP.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-! ## The forward diagonal loop -/

/-- The forward loop-2 invariant at counter `i` (absolute key offset within
the diagonal chunk): the rescaled streamed accumulator plus the masked
diagonal partial sum. -/
def praInv2 (s0 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧
  s0.pids 1 * BTL ≤ i ∧ BTS ∣ (i - s0.pids 1 * BTL) ∧
  i ≤ (s0.pids 1 + 1) * BTL ∧
  s.regs .nat [] "i_c" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2)) ∧
  s.regs .nat [] "i_k" = some (Tile.scalar (praIk s0 V BV)) ∧
  s.regs .nat [] "i_v" = some (Tile.scalar (praIv s0 V BV)) ∧
  s.regs .real [] "b_b" = some (Tile.scalar (some (praBeta s0 H))) ∧
  s.regs .real [BTL, BK] "b_q"
    = some (praQTile s0 q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) ∧
  s.regs .real [BTL, BV] "b_o"
    = some ⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale H T K V BTL BK BV (s0.pids 1 * BTL) idx.1.val idx.2.1.val
        * praW (praBeta s0 H) idx.1.val
      + praODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BK BV i idx.1.val idx.2.1.val)⟩ ∧
  s.regs .nat [BTL] "o_q" = some (Tile.vec fun r => r.val) ∧
  s.regs .nat [BTS] "o_k"
    = some (Tile.vec fun r => r.val + (i - s0.pids 1 * BTL)) ∧
  s.regs .blockPtr [BK, BTS] "p_k"
    = some ⟨fun _ => BlockPtr.mk k (s0.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [praIk s0 V BV * BK, i]⟩ ∧
  s.regs .blockPtr [BTS, BV] "p_v"
    = some ⟨fun _ => BlockPtr.mk v (s0.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, praIv s0 V BV * BV]⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One diagonal block** (causally masked, decay-weighted). -/
theorem praFwdLoop2_step (s0 sin : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat) (i : Nat)
    (hBTS : BTL % BTS = 0)
    (hlt : i < (s0.pids 1 + 1) * BTL)
    (hInv : praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      H T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (praFwdLoop2Body BTL BTS BK BV)
        (sin.setReg "_i" .nat [] (Tile.scalar i)) = some s'
      ∧ praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hpids, hmem, hlo, hdvd, hle, hic, hibh, hik, hiv, hbb, hbq, hbo,
    hoq, hok, hpk, hpv⟩ := hInv
  unfold praFwdLoop2Body
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_kLoad_eq s0 _ k "p_k" s_qk_h s_qk_t s_qk_d T K V BK BV BTS i
      (by simpa using hmem) (by simpa using hpk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_vLoad_eq s0 _ v "p_v" s_vo_h s_vo_t s_vo_d T V BV BTS i
      (by simpa using hmem) (by simpa using hpv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_msGe_eval _ BTL BTS (i - s0.pids 1 * BTL)
      (by simpa using hoq) (by simpa using hok)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_dsTile_eval s0 _ H BTL BTS (i - s0.pids 1 * BTL)
      (by simp) (by simpa using hoq) (by simpa using hok)
      (by simpa using hbb)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_sWeighted2_eval s0 _ q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      BTS i
      (fun a c => if c + (i - s0.pids 1 * BTL) ≤ a
        then praW (praBeta s0 H) (a - (c + (i - s0.pids 1 * BTL)))
        else 0)
      (by simpa using hbq) (by simp [praKTile]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_oAdd_eval _ BTL BTS BV
      (fun a p => praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale H T K V BTL BK BV (s0.pids 1 * BTL) a p
          * praW (praBeta s0 H) a
        + praODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BTL BK BV i a p)
      (fun a c => praScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          a (i + c)
        * (if c + (i - s0.pids 1 * BTL) ≤ a
            then praW (praBeta s0 H) (a - (c + (i - s0.pids 1 * BTL)))
            else 0))
      (fun c p => praVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c) p)
      (by simpa using hbo) (by simp) (by simp [praVTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_advance_col_eval _ k (s0.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
      (praIk s0 V BV * BK) i BTS "p_k" (by simp [hpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_advance_row_eval _ v (s0.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
      i (praIv s0 V BV * BV) BTS "p_v" (by simp [hpv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_okAdd_eval _ BTS (i - s0.pids 1 * BTL) (by simpa using hok)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · omega
  · have : i + BTS - s0.pids 1 * BTL = (i - s0.pids 1 * BTL) + BTS := by omega
    rw [this]
    exact Dvd.dvd.add hdvd dvd_rfl
  · have hexp : (s0.pids 1 + 1) * BTL = s0.pids 1 * BTL + BTL := by ring
    have hBTLd : BTS ∣ BTL := Nat.dvd_of_mod_eq_zero hBTS
    have h2 : i - s0.pids 1 * BTL < BTL := by omega
    have h3 : (i - s0.pids 1 * BTL) + BTS ≤ BTL :=
      pra_dvd_step_le hdvd hBTLd h2
    omega
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbb
  · simpa using hbq
  · rw [show (⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale H T K V BTL BK BV (s0.pids 1 * BTL)
          idx.1.val idx.2.1.val
          * praW (praBeta s0 H) idx.1.val
        + praODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BTL BK BV (i + BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun idx => some ((praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale H T K V BTL BK BV (s0.pids 1 * BTL)
            idx.1.val idx.2.1.val
            * praW (praBeta s0 H) idx.1.val
          + praODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
              H T K V BTL BK BV i idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              praScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                  idx.1.val (i + c.val)
                * (if c.val + (i - s0.pids 1 * BTL) ≤ idx.1.val
                    then praW (praBeta s0 H)
                      (idx.1.val - (c.val + (i - s0.pids 1 * BTL)))
                    else 0)
                * praVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c.val)
                    idx.2.1.val)⟩
      from Tile.ext fun idx =>
        congrArg some (praODiag_step' s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale H T K V BTL BK BV i BTS
          idx.1.val idx.2.1.val hlo).symm]
    simp
  · simpa using hoq
  · rw [show ((Tile.vec fun r : Fin BTS => r.val + (i + BTS - s0.pids 1 * BTL))
          : Tile .nat [BTS])
        = ((Tile.vec fun r : Fin BTS => r.val + ((i - s0.pids 1 * BTL) + BTS))
          : Tile .nat [BTS])
      from Tile.ext fun r => by
        show r.1.val + (i + BTS - s0.pids 1 * BTL)
          = r.1.val + ((i - s0.pids 1 * BTL) + BTS)
        omega]
    simp
  · simp
  · simp

set_option maxHeartbeats 4000000 in
/-- **The full diagonal loop.** -/
theorem praFwdLoop2_run (s0 sPre : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      H T K V BTL BTS BK BV (s0.pids 1 * BTL) sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "_i"
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
          (Op.constNat BTL))
        (Op.constNat BTS) (praFwdLoop2Body BTL BTS BK BV)) sPre = some sL
      ∧ praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV ((s0.pids 1 + 1) * BTL) sL := by
  have hic := hInv.2.2.2.2.2.1
  have hstop : evalOp (Op.mul .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
      (Op.constNat BTL)) sPre
      = some (Tile.scalar ((s0.pids 1 + 1) * BTL)) := by
    rw [evalOp_mul, evalOp_add]
    simp only [evalOp_ref, hic, evalOp_constNat, Option.bind_eq_bind,
      Option.bind_some]
    rfl
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "_i")
      (P := fun i s => praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale H T K V BTL BTS BK BV i s)
      (pra_mulConst_eval sPre "i_c" (s0.pids 1) BTL hic)
      hstop
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        praFwdLoop2_step s0 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale H T K V BTL BTS BK BV i hBTS hi hPi)
  have hfin : final = (s0.pids 1 + 1) * BTL :=
    le_antisymm hP.2.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-- **The mid-section**: barrier no-op, the `o_q`/`o_k` aranges, the `d_q`
decay tile, the `b_o *= d_q[:, None]` rescale, and the diagonal pointer
remakes carry `praInv1` at `i_c·BTL` into `praInv2` at `i_c·BTL`. -/
theorem praFwdMid_run (s0 sL1 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat)
    (hInv : praInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      H T K V BTL BTS BK BV (s0.pids 1 * BTL) sL1) :
    ∃ sM, stepStmts
        [ Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
          Stmt.assign .real [BTL] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTL))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BV] "b_o"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BV] "b_o")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_q"))),
          Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
          Stmt.assign .blockPtr [BK, BTS] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [K, T] [BK, BTS] [s_qk_d, s_qk_t]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)]),
          Stmt.assign .blockPtr [BTS, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
              [T, V] [BTS, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]) ]
        sL1 = some sM
      ∧ praInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV (s0.pids 1 * BTL) sM := by
  obtain ⟨hpids, hmem, hdvd, hle, hic, hibh, hik, hiv, hbb, hdh, hbq, hbo,
    hpk, hpv⟩ := hInv
  rw [stepStmts.cons_some (pra_ifThen_false_noop [] sL1)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTL sL1))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_dq_eval s0 _ H BTL (by simpa using hbb)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_oRescale_eval s0 _ H BTL BV
      (fun a p => praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale H T K V BTL BK BV (s0.pids 1 * BTL) a p)
      (by simpa using hbo) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTS _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (s0.pids 2 * s_qk_h) (praIk s0 V BV * BK) (s0.pids 1 * BTL)
      (pra_mulConst_eval _ "i_bh" (s0.pids 2) s_qk_h (by simpa using hibh))
      (pra_mulConst_eval _ "i_k" (praIk s0 V BV) BK (by simpa using hik))
      (pra_mulConst_eval _ "i_c" (s0.pids 1) BTL (by simpa using hic))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTS, BV] [s_vo_t, s_vo_d]
      (s0.pids 2 * s_vo_h) (s0.pids 1 * BTL) (praIv s0 V BV * BV)
      (pra_mulConst_eval _ "i_bh" (s0.pids 2) s_vo_h (by simpa using hibh))
      (pra_mulConst_eval _ "i_c" (s0.pids 1) BTL (by simpa using hic))
      (pra_mulConst_eval _ "i_v" (praIv s0 V BV) BV (by simpa using hiv))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, le_rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · simp
  · have : (s0.pids 1 + 1) * BTL = s0.pids 1 * BTL + BTL := by ring
    omega
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbb
  · simpa using hbq
  · rw [show (⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale H T K V BTL BK BV (s0.pids 1 * BTL)
          idx.1.val idx.2.1.val
          * praW (praBeta s0 H) idx.1.val
        + praODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            H T K V BTL BK BV (s0.pids 1 * BTL) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun idx => some (praOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale H T K V BTL BK BV (s0.pids 1 * BTL)
            idx.1.val idx.2.1.val
            * praW (praBeta s0 H) idx.1.val)⟩
      from Tile.ext fun idx => by simp [praODiag]]
    simp
  · simp
  · rw [show ((Tile.vec fun r : Fin BTS =>
          r.val + (s0.pids 1 * BTL - s0.pids 1 * BTL)) : Tile .nat [BTS])
        = ((Tile.vec fun r : Fin BTS => r.val) : Tile .nat [BTS])
      from Tile.ext fun r => by
        show r.1.val + (s0.pids 1 * BTL - s0.pids 1 * BTL) = r.1.val
        omega]
    simp
  · simp
  · simp

/-! ## Prologue -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The forward prologue** (statements 1–16): program ids, `NV`/`i_k`/`i_v`,
`i_h`, the decay registers `b_b`/`d_h`, the three block pointers, the scaled
`b_q`, and the zero accumulator establish `praInv1` at `0`. -/
theorem praFwdPrologue_run (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (H T K V BTL BTS BK BV : Nat) :
    ∃ sP, stepStmts
        [ Stmt.assign .nat [] "i_kv" (Op.programId 0),
          Stmt.assign .nat [] "i_c" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .nat [] "NV"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
                (Op.constNat 1))
              (Op.constNat BV)),
          Stmt.assign .nat [] "i_k"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_v"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_h"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_bh")
              (Op.constNat H)),
          Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
                  (Op.const 1.0)))))),
          Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
          Stmt.assign .real [BTS] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTS)
                (Op.ref .nat [BTS] "o_k")))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .blockPtr [BTL, BK] "p_q"
            (Op.makeBlockPtrDynOffsets q
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
          Stmt.assign .blockPtr [BK, BTS] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [K, T] [BK, BTS] [s_qk_d, s_qk_t]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                Op.constNat 0]),
          Stmt.assign .blockPtr [BTS, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
              [T, V] [BTS, BV] [s_vo_t, s_vo_d]
              [Op.constNat 0,
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_q") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
              (Op.const scale)),
          Stmt.assign .real [BTL, BV] "b_o" (Op.full [BTL, BV] (Op.const 0)) ]
        s = some sP
      ∧ praInv1 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          H T K V BTL BTS BK BV 0 sP := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pra_NV_eval _ V BV))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_floorDiv_refs_eval _ "i_kv" "NV" (s.pids 0) (praNV V BV)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_mod_refs_eval _ "i_kv" "NV" (s.pids 0) (praNV V BV)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_mod_ref_const_eval _ "i_bh" (s.pids 2) H (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_bb_eval s _ H (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTS _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_dh_eval s _ H BTS (by simp [praArangeTile]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval q _ _ _ _ [T, K] [BTL, BK] [s_qk_t, s_qk_d]
      (s.pids 2 * s_qk_h) (s.pids 1 * BTL) (s.pids 0 / praNV V BV * BK)
      (pra_mulConst_eval _ "i_bh" (s.pids 2) s_qk_h (by simp))
      (pra_mulConst_eval _ "i_c" (s.pids 1) BTL (by simp))
      (pra_mulConst_eval _ "i_k" (s.pids 0 / praNV V BV) BK (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (s.pids 2 * s_qk_h) (s.pids 0 / praNV V BV * BK) 0
      (pra_mulConst_eval _ "i_bh" (s.pids 2) s_qk_h (by simp))
      (pra_mulConst_eval _ "i_k" (s.pids 0 / praNV V BV) BK (by simp))
      (evalOp_constNat 0 _)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTS, BV] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) 0 (s.pids 0 % praNV V BV * BV)
      (pra_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simp))
      (evalOp_constNat 0 _)
      (pra_mulConst_eval _ "i_v" (s.pids 0 % praNV V BV) BV (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_load_bp_2d q _ "p_q" (s.pids 2 * s_qk_h) T K BTL BK s_qk_t s_qk_d
      (s.pids 1 * BTL) (s.pids 0 / praNV V BV * BK) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_qScale_eval s _ q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      (by
        have h : (⟨fun idx : TileIndex [BTL, BK] =>
            if (s.pids 1 * BTL + idx.1.val < T
                ∧ s.pids 0 / praNV V BV * BK + idx.2.1.val < K) then
              some (s.readMem q (s.pids 2 * s_qk_h
                + (s.pids 1 * BTL + idx.1.val) * s_qk_t
                + (s.pids 0 / praNV V BV * BK + idx.2.1.val) * s_qk_d))
            else some 0⟩ : Tile .real [BTL, BK])
            = ⟨fun idx : TileIndex [BTL, BK] =>
            if (s.pids 1 * BTL + idx.1.val < T
                ∧ praIk s V BV * BK + idx.2.1.val < K) then
              some (s.readMem q (s.pids 2 * s_qk_h
                + (s.pids 1 * BTL + idx.1.val) * s_qk_t
                + (praIk s V BV * BK + idx.2.1.val) * s_qk_d))
            else some 0⟩ := rfl
        rw [← h]
        simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_zeros_eval [BTL, BV] _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, dvd_zero BTS, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · simp
  · simp
  · simp [praIk]
  · simp [praIv]
  · simp
  · simp
  · simp
  · rw [show (⟨fun idx => some (praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale H T K V BTL BK BV 0 idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [praOAcc]]
    simp
  · simp [praIk]
  · simp [praIv]

/-! ## Final store recipe and ★ main theorem -/

private theorem pra_oBase_eval (sin : BlockState) (B H s_vo_h ibh ik : Nat)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar ik)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
            (Op.ref .nat [] "i_k")))
        (Op.constNat s_vo_h)) sin
      = some (Tile.scalar ((ibh + B * H * ik) * s_vo_h)) := by
  rw [evalOp_mul, evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hibh, hik, evalOp_constNat, Option.bind_eq_bind,
    Option.bind_some]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The forward epilogue**: the `p_o` make and the boundary-checked store,
from the loop-2 exit invariant, with the readback. -/
theorem praFwdStores_run (s sL2 : BlockState) (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hσ : BV ≤ s_vo_t)
    (hInvF : praInv2 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      H T K V BTL BTS BK BV ((s.pids 1 + 1) * BTL) sL2) :
    ∃ sF, stepStmts
        [ Stmt.assign .blockPtr [BTL, BV] "p_o"
            (Op.makeBlockPtrDynOffsets o
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                    (Op.ref .nat [] "i_k")))
                (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, 1]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
            (Op.ref .real [BTL, BV] "b_o") MaskOpt.none ] sL2 = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], praOActive s T V BV BTL idx →
          sF.readMem o (praOOffset s B H s_vo_h s_vo_t 1 V BV BTL idx)
            = praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
                H T K V BTL BK BV idx.1.val idx.2.1.val) := by
  obtain ⟨hFpids, hFmem, hFlo, hFdvd, hFle, hFic, hFibh, hFik, hFiv, hFbb,
    hFbq, hFbo, hFoq, hFok, hFpk, hFpv⟩ := hInvF
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BTL, BV] [s_vo_t, 1]
      ((s.pids 2 + B * H * praIk s V BV) * s_vo_h) (s.pids 1 * BTL)
      (praIv s V BV * BV)
      (pra_oBase_eval _ B H s_vo_h (s.pids 2) (praIk s V BV) hFibh hFik)
      (pra_mulConst_eval _ "i_c" (s.pids 1) BTL hFic)
      (pra_mulConst_eval _ "i_v" (praIv s V BV) BV hFiv)))]
  rw [stepStmts.cons_some (praStore_step_eq _ o "b_o" "p_o"
    ((s.pids 2 + B * H * praIk s V BV) * s_vo_h) s_vo_t (s.pids 1 * BTL)
    (praIv s V BV * BV) T V BTL BV
    (fun a p => praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      H T K V BTL BK BV a p)
    ⟨fun idx => some (praOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1
        scale H T K V BTL BK BV (s.pids 1 * BTL) idx.1.val idx.2.1.val
        * praW (praBeta s H) idx.1.val
      + praODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
          H T K V BTL BK BV ((s.pids 1 + 1) * BTL) idx.1.val idx.2.1.val)⟩
    (fun e p => congrArg some (praOOut_split s q k v s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t 1 scale H T K V BTL BK BV e.val p.val).symm)
    (by simpa using hFbo)
    (by simp))]
  rw [stepStmts.nil]
  obtain ⟨hSpids, hSregs, hSread, hSother, hSuntouched⟩ := praStore_step_props
    (sL2.setReg "p_o" .blockPtr [BTL, BV]
        ⟨fun _ => BlockPtr.mk o ((s.pids 2 + B * H * praIk s V BV) * s_vo_h)
          [T, V] [BTL, BV] [s_vo_t, 1]
          [s.pids 1 * BTL, praIv s V BV * BV]⟩)
    o ((s.pids 2 + B * H * praIk s V BV) * s_vo_h) s_vo_t (s.pids 1 * BTL)
    (praIv s V BV * BV) T V BTL BV
    (fun a p => praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      H T K V BTL BK BV a p)
    (praStoreAddr_injective _ _ _ _ BTL BV hσ)
  refine ⟨_, rfl, ?_⟩
  intro idx hact
  exact hSread idx hact

/-! ## ★ Forward main theorem -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **★ Forward main theorem: the `o` store is the genuine causal retention
attention closed form.**

For every program `(i_kv, i_c, i_bh)` (universally quantified through
`s.pids`), executing the full forward surface succeeds and the `o` block
store holds `Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) ·
v[t,p]` at every in-window lane `(a, p)`, with the per-head decay
`b_b = log2(1 − 2^(−5 − i_bh % H))` and `score(a,t) = (scale·q[a]) · k[t]`
over the `BK` head window.

Side conditions: the host's contiguous last-dim stride (`s_vo_d = 1`,
`BV ≤ s_vo_t`, store-lane injectivity) and the host's own
`assert BTL % BTS == 0`. The loads are boundary-checked, so **no**
divisibility hypothesis on `T` is needed: ragged tails are exact. -/
specification pra_fwd_o_exec_genuine
    (s : BlockState) (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hSd : s_vo_d = 1) (hσ : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pra_fwd_surface q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], praOActive s T V BV BTL idx →
          sF.readMem o (praOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = praOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                H T K V BTL BK BV idx.1.val idx.2.1.val) := by
  subst hSd
  obtain ⟨sP, hPro, hInvP⟩ := praFwdPrologue_run s q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale H T K V BTL BTS BK BV
  obtain ⟨sL1, hL1, hInv1⟩ := praFwdLoop1_run s sP q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale H T K V BTL BTS BK BV hBTS hBTSpos hInvP
  obtain ⟨sM, hMid, hInv2⟩ := praFwdMid_run s sL1 q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale H T K V BTL BTS BK BV hInv1
  obtain ⟨sL2, hL2, hInvF⟩ := praFwdLoop2_run s sM q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale H T K V BTL BTS BK BV hBTS hBTSpos hInv2
  rw [exec, pra_fwd_body_eq]
  rw [show ([ Stmt.assign .nat [] "i_kv" (Op.programId 0),
      Stmt.assign .nat [] "i_c" (Op.programId 1),
      Stmt.assign .nat [] "i_bh" (Op.programId 2),
      Stmt.assign .nat [] "NV"
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
            (Op.constNat 1))
          (Op.constNat BV)),
      Stmt.assign .nat [] "i_k"
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
          (Op.ref .nat [] "NV")),
      Stmt.assign .nat [] "i_v"
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
          (Op.ref .nat [] "NV")),
      Stmt.assign .nat [] "i_h"
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_bh")
          (Op.constNat H)),
      Stmt.assign .real [] "b_b"
        (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
          (Op.exp2 (Op.sub .real Broadcast.nil
            (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
            (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
              (Op.const 1.0)))))),
      Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
      Stmt.assign .real [BTS] "d_h"
        (Op.exp2 (Op.mul .real Broadcast.scalarR
          (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTS)
            (Op.ref .nat [BTS] "o_k")))
          (Op.ref .real [] "b_b"))),
      Stmt.assign .blockPtr [BTL, BK] "p_q"
        (Op.makeBlockPtrDynOffsets q
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
          [T, K] [BTL, BK] [s_qk_t, s_qk_d]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
            Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
      Stmt.assign .blockPtr [BK, BTS] "p_k"
        (Op.makeBlockPtrDynOffsets k
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
          [K, T] [BK, BTS] [s_qk_d, s_qk_t]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
            Op.constNat 0]),
      Stmt.assign .blockPtr [BTS, BV] "p_v"
        (Op.makeBlockPtrDynOffsets v
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
          [T, V] [BTS, BV] [s_vo_t, 1]
          [Op.constNat 0,
            Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
      Stmt.assign .real [BTL, BK] "b_q"
        (Op.load .real
          (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_q") [0, 1])
          MaskOpt.none),
      Stmt.assign .real [BTL, BK] "b_q"
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
          (Op.const scale)),
      Stmt.assign .real [BTL, BV] "b_o" (Op.full [BTL, BV] (Op.const 0)),
      Stmt.forRangeDyn "_i" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.constNat BTS) (praFwdLoop1Body BTL BTS BK BV),
      Stmt.ifThen (Op.constBool Bool.false) [],
      Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
      Stmt.assign .real [BTL] "d_q"
        (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTL))
          (Op.ref .real [] "b_b"))),
      Stmt.assign .real [BTL, BV] "b_o"
        (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [BTL, BV] "b_o")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_q"))),
      Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
      Stmt.assign .blockPtr [BK, BTS] "p_k"
        (Op.makeBlockPtrDynOffsets k
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
          [K, T] [BK, BTS] [s_qk_d, s_qk_t]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
            Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)]),
      Stmt.assign .blockPtr [BTS, BV] "p_v"
        (Op.makeBlockPtrDynOffsets v
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
          [T, V] [BTS, BV] [s_vo_t, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
            Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
      Stmt.forRangeDyn "_i"
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
          (Op.constNat BTL))
        (Op.constNat BTS) (praFwdLoop2Body BTL BTS BK BV),
      Stmt.assign .blockPtr [BTL, BV] "p_o"
        (Op.makeBlockPtrDynOffsets o
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                (Op.ref .nat [] "i_k")))
            (Op.constNat s_vo_h))
          [T, V] [BTL, BV] [s_vo_t, 1]
          [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
            Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
      Stmt.store .real [BTL, BV]
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
        (Op.ref .real [BTL, BV] "b_o") MaskOpt.none ] : List Stmt)
      = [ Stmt.assign .nat [] "i_kv" (Op.programId 0),
          Stmt.assign .nat [] "i_c" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .nat [] "NV"
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
                (Op.constNat 1))
              (Op.constNat BV)),
          Stmt.assign .nat [] "i_k"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_v"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_kv")
              (Op.ref .nat [] "NV")),
          Stmt.assign .nat [] "i_h"
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_bh")
              (Op.constNat H)),
          Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
                  (Op.const 1.0)))))),
          Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
          Stmt.assign .real [BTS] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTS)
                (Op.ref .nat [BTS] "o_k")))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .blockPtr [BTL, BK] "p_q"
            (Op.makeBlockPtrDynOffsets q
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
          Stmt.assign .blockPtr [BK, BTS] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              [K, T] [BK, BTS] [s_qk_d, s_qk_t]
              [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                Op.constNat 0]),
          Stmt.assign .blockPtr [BTS, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
              [T, V] [BTS, BV] [s_vo_t, 1]
              [Op.constNat 0,
                Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_q") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_q"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
              (Op.const scale)),
          Stmt.assign .real [BTL, BV] "b_o" (Op.full [BTL, BV] (Op.const 0)) ]
        ++ (Stmt.forRangeDyn "_i" (Op.constNat 0)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
            (Op.constNat BTS) (praFwdLoop1Body BTL BTS BK BV)
          :: ([ Stmt.ifThen (Op.constBool Bool.false) [],
              Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
              Stmt.assign .real [BTL] "d_q"
                (Op.exp2 (Op.mul .real Broadcast.scalarR
                  (Op.natToReal (Op.arange BTL))
                  (Op.ref .real [] "b_b"))),
              Stmt.assign .real [BTL, BV] "b_o"
                (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                  (Op.ref .real [BTL, BV] "b_o")
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_q"))),
              Stmt.assign .nat [BTS] "o_k" (Op.arange BTS),
              Stmt.assign .blockPtr [BK, BTS] "p_k"
                (Op.makeBlockPtrDynOffsets k
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
                  [K, T] [BK, BTS] [s_qk_d, s_qk_t]
                  [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
                    Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)]),
              Stmt.assign .blockPtr [BTS, BV] "p_v"
                (Op.makeBlockPtrDynOffsets v
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
                  [T, V] [BTS, BV] [s_vo_t, 1]
                  [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                    Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]) ]
            ++ (Stmt.forRangeDyn "_i"
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
                (Op.mul .nat Broadcast.nil
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
                  (Op.constNat BTL))
                (Op.constNat BTS) (praFwdLoop2Body BTL BTS BK BV)
              :: [ Stmt.assign .blockPtr [BTL, BV] "p_o"
                    (Op.makeBlockPtrDynOffsets o
                      (Op.mul .nat Broadcast.nil
                        (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                          (Op.mul .nat Broadcast.nil
                            (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                            (Op.ref .nat [] "i_k")))
                        (Op.constNat s_vo_h))
                      [T, V] [BTL, BV] [s_vo_t, 1]
                      [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL),
                        Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
                  Stmt.store .real [BTL, BV]
                    (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
                    (Op.ref .real [BTL, BV] "b_o") MaskOpt.none ])))
      from rfl]
  rw [stepStmts.append_some hPro, stepStmts.cons_some hL1,
    stepStmts.append_some hMid, stepStmts.cons_some hL2]
  obtain ⟨sF, hSt, hRead⟩ := praFwdStores_run s sL2 q k v o
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t scale B H T K V BTL BTS BK BV
    hσ hInvF
  exact ⟨sF, hSt, hRead⟩


/-! ## Closed-form specification (backward dk/dv)

The helper's scalar arguments `i_bh, i_c, i_k, i_v, i_h` are plain
universally-quantified binders (the surface splices them as constants), so
the backward closed forms are parameterized by them directly — no `s.pids`
projections. Every load is boundary-checked, so the guarded value functions
bake the `T × K` / `T × V` windows in and the closed forms hold for
arbitrary ragged tails. -/

/-- The top of the streamed key range: `cdiv(T, BTS)·BTS`. -/
def prbNB (T BTS : Nat) : Nat := (T + BTS - 1) / BTS * BTS

/-- The respelled descending trip count (see the preamble). -/
def prbTrip (T BTS i_c BTL : Nat) : Nat :=
  ((T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL + BTS - 1) / BTS

/-- The per-head decay exponent `b_b = log2(1 − 2^(−5 − i_h))` on the spliced
`i_h` constant, exactly as the walk computes it. -/
noncomputable def prbBeta (i_h : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0) - ((i_h : Nat) : ℝ) * 1.0)
      * Real.log 2)) / Real.log 2

/-- The guarded `b_k` lane `(r, e)` of the fixed chunk block. -/
noncomputable def prbKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (r e : Nat) : ℝ :=
  if i_c * BTL + r < T ∧ i_k * BK + e < K then
    s.readMem k (i_bh * s_qk_h + (i_c * BTL + r) * s_qk_t
      + (i_k * BK + e) * s_qk_d)
  else 0

/-- The guarded `b_v` lane `(r, p)` of the fixed chunk block. -/
noncomputable def prbVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (r p : Nat) : ℝ :=
  if i_c * BTL + r < T ∧ i_v * BV + p < V then
    s.readMem v (i_bh * s_vo_h + (i_c * BTL + r) * s_vo_t
      + (i_v * BV + p) * s_vo_d)
  else 0

/-- The guarded transposed `q` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
noncomputable def prbQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_k : Nat) (T K BK : Nat)
    (e t : Nat) : ℝ :=
  if i_k * BK + e < K ∧ t < T then
    s.readMem q (i_bh * s_qk_h + (i_k * BK + e) * s_qk_d + t * s_qk_t)
  else 0

/-- The guarded transposed `do` lane `(p, t)` at absolute key `t`. -/
noncomputable def prbDoGuarded (s : BlockState) (do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_v : Nat) (T V BV : Nat)
    (p t : Nat) : ℝ :=
  if i_v * BV + p < V ∧ t < T then
    s.readMem do_ (i_bh * s_vo_h + (i_v * BV + p) * s_vo_d + t * s_vo_t)
  else 0

/-- The (unscaled) backward score core at `(row r, key t)`: `k[r] · qᵀ[t]`
over the `BK` head window. -/
noncomputable def prbSVal (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (r t : Nat) : ℝ :=
  ∑ e : Fin BK,
    prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r e.val
      * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e.val t

/-- The (unscaled) backward dscore core at `(row r, key t)`: `v[r] · doᵀ[t]`
over the `BV` value window. -/
noncomputable def prbDsVal (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (r t : Nat) : ℝ :=
  ∑ p : Fin BV,
    prbVGuarded s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r p.val
      * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p.val t

/-- The `b_dk` accumulator over keys `[lo, hi)` with the causal keep
`i_c·BTL + r ≤ t` and the row-anchored retention weight
`2^((t − (i_c·BTL+r))·b_b) · scale`. -/
noncomputable def prbDkPart (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (lo hi : Nat) (r e : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
        * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t
    else 0

/-- The `b_dv` accumulator over keys `[lo, hi)` with the causal keep and the
row-anchored retention weight. -/
noncomputable def prbDvPart (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (lo hi : Nat) (r p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
        * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t
    else 0

/-- **The stored `dk` lane** — the retention key-gradient closed form. Over
the full key sweep (up to the larger of the streamed top `cdiv(T,BTS)·BTS`
and the diagonal end `(i_c+1)·BTL`; loads beyond `T` read as zero, so the
tail summands vanish), every causally kept key `t ≥ i_c·BTL + r` contributes
its decayed, scaled dscore times the transposed-`q` lane:
`Σ_t 2^((t − (i_c·BTL+r))·b_b) · scale · (v[r]·doᵀ[t]) · q[e, t]`. -/
noncomputable def prbDkOut (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (r e : Nat) : ℝ :=
  ∑ t ∈ Finset.range (max (prbNB T BTS) ((i_c + 1) * BTL)),
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
        * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t
    else 0

/-- **The stored `dv` lane** — the retention value-gradient closed form:
`Σ_t 2^((t − (i_c·BTL+r))·b_b) · scale · (k[r]·qᵀ[t]) · do[p, t]` over the
same causally kept key sweep. -/
noncomputable def prbDvOut (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (r p : Nat) : ℝ :=
  ∑ t ∈ Finset.range (max (prbNB T BTS) ((i_c + 1) * BTL)),
    if i_c * BTL + r ≤ t then
      praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
        * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
        * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t
    else 0

/-- The streamed `b_dk` accumulator **anchored at `A`**: keys `[A, prbNB)`
with the anchor-relative decay `2^((t − A)·b_b)` — the per-iteration
`b_dk *= d_b` shifts the anchor down by `BTS`; the row factor
`2^((BTL−r)·b_b) · scale` is applied only after the loop. -/
noncomputable def prbDkDesc (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (A : Nat) (r e : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico A (prbNB T BTS),
    praW (prbBeta i_h) (t - A)
      * (prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
        * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t)

/-- The streamed `b_dv` accumulator anchored at `A` (the row factor
`2^((BTL−r)·b_b)` rides separately, inside the loop via `b_kd`). -/
noncomputable def prbDvDesc (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (A : Nat) (r p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico A (prbNB T BTS),
    praW (prbBeta i_h) (t - A)
      * (prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
        * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t)

/-! ## Backward block-sum, decay-step, and trip-count arithmetic -/

/-- Bottom-extension of an `Ico` sum by one `BTS` block. -/
private theorem prb_sum_Ico_add_block_bot (lo hi BTS : Nat) (h : lo + BTS ≤ hi)
    (f : Nat → ℝ) :
    ∑ t ∈ Finset.Ico lo hi, f t
      = (∑ c : Fin BTS, f (lo + c.val)) + ∑ t ∈ Finset.Ico (lo + BTS) hi, f t := by
  rw [← Finset.sum_Ico_consecutive _ (Nat.le_add_right lo BTS) h]
  congr 1
  rw [Finset.sum_Ico_eq_sum_range]
  simp only [Nat.add_sub_cancel_left]
  rw [Fin.sum_univ_eq_sum_range (fun c => f (lo + c))]

/-- Under the host's `BTL % BTS = 0`, the respelled trip count covers the
streamed range exactly: `prbTrip · BTS = prbNB − (i_c+1)·BTL`. -/
private theorem prb_trip_mul (T BTS i_c BTL : Nat) (hBTS : BTL % BTS = 0)
    (hBTSpos : 0 < BTS) :
    prbTrip T BTS i_c BTL * BTS = prbNB T BTS - (i_c + 1) * BTL := by
  obtain ⟨m, hm⟩ : BTS ∣ prbNB T BTS - (i_c + 1) * BTL :=
    Nat.dvd_sub (dvd_mul_left BTS ((T + BTS - 1) / BTS))
      (Dvd.dvd.mul_left (Nat.dvd_of_mod_eq_zero hBTS) (i_c + 1))
  have htrip : prbTrip T BTS i_c BTL = m := by
    show ((T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL + BTS - 1) / BTS = m
    have hm' : (T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL = BTS * m := hm
    rw [hm']
    have h1 : BTS * m + BTS - 1 = BTS * m + (BTS - 1) := by omega
    rw [h1, Nat.mul_add_div hBTSpos, Nat.div_eq_of_lt (by omega), Nat.add_zero]
  rw [htrip, hm, Nat.mul_comm]

/-- **The anchored decay step, generically**: decaying the `A = i + BTS`
anchored accumulator by `2^(BTS·b_b)` (walk order) and adding the fresh
bottom block's `2^(c·b_b)`-weighted contributions re-anchors it at `i`. -/
private theorem prbDescCore_step (β : ℝ) (N i BTS : Nat) (F : Nat → ℝ)
    (hstep : i + BTS ≤ N) :
    (∑ t ∈ Finset.Ico (i + BTS) N, praW β (t - (i + BTS)) * F t)
        * Real.exp (β * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS, praW β c.val * F (i + c.val)
      = ∑ t ∈ Finset.Ico i N, praW β (t - i) * F t := by
  have hblock : (∑ t ∈ Finset.Ico i N, praW β (t - i) * F t)
      = (∑ c : Fin BTS, praW β c.val * F (i + c.val))
        + ∑ t ∈ Finset.Ico (i + BTS) N, praW β (t - i) * F t := by
    rw [prb_sum_Ico_add_block_bot i N BTS hstep]
    congr 1
    refine Finset.sum_congr rfl fun c _ => ?_
    show praW β (i + c.val - i) * F (i + c.val) = praW β c.val * F (i + c.val)
    rw [Nat.add_sub_cancel_left]
  have htail : (∑ t ∈ Finset.Ico (i + BTS) N, praW β (t - i) * F t)
      = (∑ t ∈ Finset.Ico (i + BTS) N, praW β (t - (i + BTS)) * F t)
          * Real.exp (β * ((BTS : Nat) : ℝ) * Real.log 2) := by
    rw [praW_comm, Finset.sum_mul]
    refine Finset.sum_congr rfl fun t ht => ?_
    have h1 : i + BTS ≤ t := (Finset.mem_Ico.mp ht).1
    rw [show t - i = (t - (i + BTS)) + BTS from by omega, praW_add]
    ring
  rw [hblock, htail]
  ring

/-- The walk-shaped `b_dk` desc step: the decayed anchored accumulator plus
the fresh `(2^(c·b_b)·ds)·qᵀ` block is the accumulator re-anchored at `i`. -/
private theorem prbDkDesc_step (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (i : Nat) (hstep : i + BTS ≤ prbNB T BTS) (r e : Nat) :
    prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i + BTS) r e
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS,
          (praW (prbBeta i_h) c.val
              * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
                  r (i + c.val))
            * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e (i + c.val)
      = prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i r e := by
  have hcore : prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i + BTS) r e
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS, praW (prbBeta i_h) c.val
          * (prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
              r (i + c.val)
            * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e (i + c.val))
      = prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i r e :=
    prbDescCore_step (prbBeta i_h) (prbNB T BTS) i BTS
      (fun t => prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
        * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t) hstep
  rw [← hcore]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  ring

/-- The walk-shaped `b_dv` desc step: the constant row factor `2^((BTL−r)·b_b)`
(inside `b_kd`, hence inside every `b_s` lane) factors out of the recurrence. -/
private theorem prbDvDesc_step (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (i : Nat) (hstep : i + BTS ≤ prbNB T BTS) (r p : Nat) :
    (praW (prbBeta i_h) (BTL - r)
          * prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i + BTS) r p)
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS,
          (praW (prbBeta i_h) (BTL - r)
              * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
                  r (i + c.val))
            * (prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
                  p (i + c.val)
              * praW (prbBeta i_h) c.val)
      = praW (prbBeta i_h) (BTL - r)
          * prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i r p := by
  have hcore : prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i + BTS) r p
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
      + ∑ c : Fin BTS, praW (prbBeta i_h) c.val
          * (prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
              r (i + c.val)
            * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
                p (i + c.val))
      = prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i r p :=
    prbDescCore_step (prbBeta i_h) (prbNB T BTS) i BTS
      (fun t => prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
        * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t) hstep
  rw [← hcore, mul_add]
  congr 1
  · ring
  · rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    ring

/-- **Desc-exit rescale bridge (`b_dk`)**: applying the post-loop row factor
`2^((BTL−r)·b_b) · scale` to the anchored accumulator at the loop's final
anchor `prbNB − prbTrip·BTS` yields the keep-indicator partial sum — for
`t ≥ (i_c+1)·BTL` the causal keep is identically true and the exponents fold
by `(BTL−r) + (t − (i_c+1)·BTL) = t − (i_c·BTL+r)`. -/
private theorem prbDkDesc_scale_bridge (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (r e : Nat) (hr : r < BTL) :
    prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) r e
        * (praW (prbBeta i_h) (BTL - r) * scale)
      = prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r e := by
  have hTripMul := prb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
  unfold prbDkDesc prbDkPart
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun t ht => ?_
  obtain ⟨hlo, hhi⟩ := Finset.mem_Ico.mp ht
  rcases le_or_gt ((i_c + 1) * BTL) (prbNB T BTS) with hle | hgt
  · have hA : prbNB T BTS - prbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hA] at hlo
    have hkeep : i_c * BTL + r ≤ t := by omega
    rw [if_pos hkeep, hA,
      show t - (i_c * BTL + r) = (BTL - r) + (t - (i_c + 1) * BTL) from by omega,
      praW_add]
    ring
  · have htrip0 : prbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero] at hlo
    exact absurd hhi (by omega)

/-- **Desc-exit rescale bridge (`b_dv`)**: same fold with the row factor
already outside the accumulator and the flat `· scale` applied by the walk. -/
private theorem prbDvDesc_scale_bridge (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (r p : Nat) (hr : r < BTL) :
    (praW (prbBeta i_h) (BTL - r)
          * prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) r p) * scale
      = prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r p := by
  have hTripMul := prb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
  unfold prbDvDesc prbDvPart
  rw [Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun t ht => ?_
  obtain ⟨hlo, hhi⟩ := Finset.mem_Ico.mp ht
  rcases le_or_gt ((i_c + 1) * BTL) (prbNB T BTS) with hle | hgt
  · have hA : prbNB T BTS - prbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hA] at hlo
    have hkeep : i_c * BTL + r ≤ t := by omega
    rw [if_pos hkeep, hA,
      show t - (i_c * BTL + r) = (BTL - r) + (t - (i_c + 1) * BTL) from by omega,
      praW_add]
    ring
  · have htrip0 : prbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero] at hlo
    exact absurd hhi (by omega)

/-- Diag-loop `b_dk` spec step: the decay-masked kernel summands extend the
diagonal partial sum at the top, past the fixed desc-exit term `D`; on kept
lanes the intra-block exponent `(c + off) − r` is the row-anchored
`t − (i_c·BTL + r)`. -/
private theorem prbDkDiag_step' (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (i : Nat) (D : ℝ) (r e : Nat) (hlo : i_c * BTL ≤ i) :
    (prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i_c * BTL) i r e + D)
      + ∑ c : Fin BTS,
          (prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
              r (i + c.val)
            * ((if r ≤ c.val + (i - i_c * BTL) then
                  praW (prbBeta i_h) ((c.val + (i - i_c * BTL)) - r)
                else 0) * scale))
            * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e (i + c.val)
      = prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i_c * BTL) (i + BTS)
          r e + D := by
  rw [add_right_comm]
  congr 1
  unfold prbDkPart
  rw [pra_sum_Ico_add_block (i_c * BTL) i BTS hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (r ≤ c.val + (i - i_c * BTL)) ↔ (i_c * BTL + r ≤ i + c.val) := by
    omega
  by_cases h : i_c * BTL + r ≤ i + c.val
  · rw [if_pos (hcond.mpr h), if_pos h,
      show (c.val + (i - i_c * BTL)) - r = (i + c.val) - (i_c * BTL + r) from by
        omega]
    ring
  · rw [if_neg (fun hc => h (hcond.mp hc)), if_neg h]
    ring

/-- Diag-loop `b_dv` spec step. -/
private theorem prbDvDiag_step' (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (i : Nat) (D : ℝ) (r p : Nat) (hlo : i_c * BTL ≤ i) :
    (prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i_c * BTL) i r p + D)
      + ∑ c : Fin BTS,
          (prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
              r (i + c.val)
            * ((if r ≤ c.val + (i - i_c * BTL) then
                  praW (prbBeta i_h) ((c.val + (i - i_c * BTL)) - r)
                else 0) * scale))
            * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
                p (i + c.val)
      = prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV (i_c * BTL) (i + BTS)
          r p + D := by
  rw [add_right_comm]
  congr 1
  unfold prbDvPart
  rw [pra_sum_Ico_add_block (i_c * BTL) i BTS hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (r ≤ c.val + (i - i_c * BTL)) ↔ (i_c * BTL + r ≤ i + c.val) := by
    omega
  by_cases h : i_c * BTL + r ≤ i + c.val
  · rw [if_pos (hcond.mpr h), if_pos h,
      show (c.val + (i - i_c * BTL)) - r = (i + c.val) - (i_c * BTL + r) from by
        omega]
    ring
  · rw [if_neg (fun hc => h (hcond.mp hc)), if_neg h]
    ring

/-- Fold the two-part exit sum (diag part + rescaled desc part) into
`prbDkOut`. -/
private theorem prbDkOut_split (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (r e : Nat) :
    prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) r e
      + prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r e
      = prbDkOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV r e := by
  have hTripMul := prb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
  have hout : prbDkOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV r e
      = prbDkPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (max (prbNB T BTS) ((i_c + 1) * BTL)) r e := by
    unfold prbDkOut prbDkPart
    rw [Finset.range_eq_Ico,
      ← Finset.sum_Ico_consecutive _ (Nat.zero_le (i_c * BTL))
        (le_trans (by omega) (le_max_right (prbNB T BTS) ((i_c + 1) * BTL)))]
    have hzero : (∑ t ∈ Finset.Ico 0 (i_c * BTL),
        if i_c * BTL + r ≤ t then
          praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
            * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV r t
            * prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK e t
        else 0) = 0 :=
      Finset.sum_eq_zero fun t ht => if_neg (by
        have h2 := (Finset.mem_Ico.mp ht).2
        omega)
    rw [hzero, zero_add]
  rw [hout]
  rcases le_or_gt ((i_c + 1) * BTL) (prbNB T BTS) with hle | hgt
  · have hA : prbNB T BTS - prbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hA, Nat.max_eq_left hle]
    unfold prbDkPart
    exact Finset.sum_Ico_consecutive _ (by omega) hle
  · have htrip0 : prbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero, Nat.max_eq_right (Nat.le_of_lt hgt)]
    unfold prbDkPart
    rw [Finset.Ico_self, Finset.sum_empty, add_zero]

/-- Fold the two-part exit sum into `prbDvOut`. -/
private theorem prbDvOut_split (s : BlockState) (q k v do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (i_bh i_c i_k i_v i_h : Nat) (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (r p : Nat) :
    prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) r p
      + prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r p
      = prbDvOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV r p := by
  have hTripMul := prb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
  have hout : prbDvOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV r p
      = prbDvPart s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (max (prbNB T BTS) ((i_c + 1) * BTL)) r p := by
    unfold prbDvOut prbDvPart
    rw [Finset.range_eq_Ico,
      ← Finset.sum_Ico_consecutive _ (Nat.zero_le (i_c * BTL))
        (le_trans (by omega) (le_max_right (prbNB T BTS) ((i_c + 1) * BTL)))]
    have hzero : (∑ t ∈ Finset.Ico 0 (i_c * BTL),
        if i_c * BTL + r ≤ t then
          praW (prbBeta i_h) (t - (i_c * BTL + r)) * scale
            * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK r t
            * prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV p t
        else 0) = 0 :=
      Finset.sum_eq_zero fun t ht => if_neg (by
        have h2 := (Finset.mem_Ico.mp ht).2
        omega)
    rw [hzero, zero_add]
  rw [hout]
  rcases le_or_gt ((i_c + 1) * BTL) (prbNB T BTS) with hle | hgt
  · have hA : prbNB T BTS - prbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hA, Nat.max_eq_left hle]
    unfold prbDvPart
    exact Finset.sum_Ico_consecutive _ (by omega) hle
  · have htrip0 : prbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero, Nat.max_eq_right (Nat.le_of_lt hgt)]
    unfold prbDvPart
    rw [Finset.Ico_self, Finset.sum_empty, add_zero]

/-! ## Backward value tiles and load bridges -/

/-- The fixed loaded `b_k` chunk tile. -/
noncomputable def prbKTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat) :
    Tile .real [BTL, BK] :=
  ⟨fun idx => some (prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
    idx.1.val idx.2.1.val)⟩

/-- The fixed loaded `b_v` chunk tile. -/
noncomputable def prbVTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat) :
    Tile .real [BTL, BV] :=
  ⟨fun idx => some (prbVGuarded s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
    idx.1.val idx.2.1.val)⟩

/-- The loaded transposed `b_q` block at key-column offset `i`. -/
noncomputable def prbQTile (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_k : Nat) (T K BK BTS : Nat) (i : Nat) :
    Tile .real [BK, BTS] :=
  ⟨fun idx => some (prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
    idx.1.val (i + idx.2.1.val))⟩

/-- The loaded transposed `b_do` block at key-column offset `i`. -/
noncomputable def prbDoTile (s : BlockState) (do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_v : Nat) (T V BV BTS : Nat) (i : Nat) :
    Tile .real [BV, BTS] :=
  ⟨fun idx => some (prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
    idx.1.val (i + idx.2.1.val))⟩

/-- Scalar `constNat * constNat` products (the spliced backward offsets). -/
private theorem prb_constMul_eval (s : BlockState) (a b : Nat) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat a) (Op.constNat b)) s
      = some (Tile.scalar (a * b)) := by
  rw [evalOp_mul]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Loading `p_k` at offsets `[i_c·BTL, i_k·BK]` lands on `prbKTile`. -/
private theorem prb_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k T K BTL BK : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTL, BK] name = some
      ⟨fun _ => BlockPtr.mk k (i_bh * s_qk_h) [T, K] [BTL, BK]
        [s_qk_t, s_qk_d] [i_c * BTL, i_k * BK]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] name) [0, 1])
        MaskOpt.none) sin
      = some (prbKTile s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK) := by
  rw [pra_load_bp_2d k sin name (i_bh * s_qk_h) T K BTL BK s_qk_t s_qk_d
    (i_c * BTL) (i_k * BK) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, e, u⟩ := idx
  simp only [prbKTile, prbKGuarded]
  by_cases h : i_c * BTL + r.val < T ∧ i_k * BK + e.val < K
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_v` at offsets `[i_c·BTL, i_v·BV]` lands on `prbVTile`. -/
private theorem prb_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v T V BTL BV : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTL, BV] name = some
      ⟨fun _ => BlockPtr.mk v (i_bh * s_vo_h) [T, V] [BTL, BV]
        [s_vo_t, s_vo_d] [i_c * BTL, i_v * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] name) [0, 1])
        MaskOpt.none) sin
      = some (prbVTile s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV) := by
  rw [pra_load_bp_2d v sin name (i_bh * s_vo_h) T V BTL BV s_vo_t s_vo_d
    (i_c * BTL) (i_v * BV) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, p, u⟩ := idx
  simp only [prbVTile, prbVGuarded]
  by_cases h : i_c * BTL + r.val < T ∧ i_v * BV + p.val < V
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_q` at offsets `[i_k·BK, i]` (the `(K, T)` parent) lands on
`prbQTile … i`. -/
private theorem prb_qLoad_eq (s sin : BlockState) (q : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_k T K BK BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BK, BTS] name = some
      ⟨fun _ => BlockPtr.mk q (i_bh * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [i_k * BK, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (prbQTile s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS i) := by
  rw [pra_load_bp_2d q sin name (i_bh * s_qk_h) K T BK BTS s_qk_d s_qk_t
    (i_k * BK) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, c, u⟩ := idx
  simp only [prbQTile, prbQGuarded]
  by_cases h : i_k * BK + e.val < K ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_do` at offsets `[i_v·BV, i]` (the `(V, T)` parent) lands on
`prbDoTile … i`. -/
private theorem prb_doLoad_eq (s sin : BlockState) (do_ : RegionName)
    (name : RegName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_v T V BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BV, BTS] name = some
      ⟨fun _ => BlockPtr.mk do_ (i_bh * s_vo_h) [V, T] [BV, BTS]
        [s_vo_d, s_vo_t] [i_v * BV, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (prbDoTile s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV BTS i) := by
  rw [pra_load_bp_2d do_ sin name (i_bh * s_vo_h) V T BV BTS s_vo_d s_vo_t
    (i_v * BV) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨p, c, u⟩ := idx
  simp only [prbDoTile, prbDoGuarded]
  by_cases h : i_v * BV + p.val < V ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pra_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-! ## Backward per-statement eval recipes -/

/-- The decay exponent statement lands on `prbBeta` (backward spelling, on
the spliced `i_h` constant). -/
private theorem prb_bb_eval (t : BlockState) (i_h : Nat) :
    evalOp (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
        (Op.exp2 (Op.sub .real Broadcast.nil
          (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
          (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat i_h))
            (Op.const 1.0)))))) t
      = some (Tile.scalar (some (prbBeta i_h))) := by
  simp only [evalOp, bind, Option.bind]
  rfl

/-- The `d_b = tl.math.exp2(b_b * BTS)` register (walk-order product;
`= praW β BTS` by `praW_comm`). -/
private theorem prb_db_eval (t : BlockState) (β : ℝ) (BTS : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some β))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
        (Op.natToReal (Op.constNat BTS)))) t
      = some (Tile.scalar (some (Real.exp (β * ((BTS : Nat) : ℝ)
          * Real.log 2)))) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The backward `d_h` tile (inline `tl.arange`): lane `r` holds
`2^((BTL − r)·b_b)`. -/
private theorem prb_dh_eval (t : BlockState) (β : ℝ) (BTL : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some β))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTL)
          (Op.arange BTL)))
        (Op.ref .real [] "b_b"))) t
      = some (⟨fun idx => some (praW β (BTL - idx.1.val))⟩ : Tile .real [BTL]) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The backward `d_q` tile (inline `tl.arange`): lane `c` holds `2^(c·b_b)`. -/
private theorem prb_dq_eval (t : BlockState) (β : ℝ) (BTS : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some β))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTS))
        (Op.ref .real [] "b_b"))) t
      = some (⟨fun idx => some (praW β idx.1.val)⟩ : Tile .real [BTS]) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- `b_kd = b_k * d_h[:, None]`: the row-decayed key tile. -/
private theorem prb_kd_eval (sin : BlockState) (BTL BK : Nat)
    (g : Nat → Nat → ℝ) (w : Nat → ℝ)
    (hk : sin.regs .real [BTL, BK] "b_k"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hdh : sin.regs .real [BTL] "d_h" = some ⟨fun idx => some (w idx.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_k")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))) sin
      = some ⟨fun idx : TileIndex [BTL, BK] =>
          some (g idx.1.val idx.2.1.val * w idx.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hk)
    (evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdh)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, e, u⟩ := idx
  rfl

/-- The respelled block index
`i = tl.cdiv(T, BTS)·BTS − BTS − j·BTS` from the `j` register. -/
private theorem prb_iAssign_eval (sin : BlockState) (T BTS j : Nat)
    (hj : sin.regs .nat [] "j" = some (Tile.scalar j)) :
    evalOp (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat T) (Op.constNat BTS))
                (Op.constNat 1))
              (Op.constNat BTS))
            (Op.constNat BTS))
          (Op.constNat BTS))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "j") (Op.constNat BTS))) sin
      = some (Tile.scalar (prbNB T BTS - BTS - j * BTS)) := by
  simp only [evalOp, evalOp_ref, hj, evalOp_constNat, bind, Option.bind]
  rfl

/-- The respelled descending trip count (the loop's stop expression). -/
private theorem prb_trip_eval (s : BlockState) (T BTS i_c BTL : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.div .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil (Op.constNat T) (Op.constNat BTS))
                    (Op.constNat 1))
                  (Op.constNat BTS))
                (Op.constNat BTS))
              (Op.constNat ((i_c + 1) * BTL)))
            (Op.constNat BTS))
          (Op.constNat 1))
        (Op.constNat BTS)) s
      = some (Tile.scalar (prbTrip T BTS i_c BTL)) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

/-- `b_do = b_do * d_q[None, :]`: the `d_q`-rescaled loaded `dO` block. -/
private theorem prb_doScale_eval (sin : BlockState) (BV BTS : Nat)
    (g : Nat → Nat → ℝ) (w : Nat → ℝ)
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hdq : sin.regs .real [BTS] "d_q" = some ⟨fun idx => some (w idx.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BV, BTS] "b_do")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "d_q"))) sin
      = some ⟨fun idx : TileIndex [BV, BTS] =>
          some (g idx.1.val idx.2.1.val * w idx.2.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consR (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdo)
    (evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdq)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨p, c, u⟩ := idx
  rfl

/-- `X *= d_b`: the per-iteration accumulator decay (both `b_dv` and `b_dk`). -/
private theorem prb_accDecay_eval (sin : BlockState) (name : RegName)
    (M N : Nat) (g : Nat → Nat → ℝ) (D : ℝ)
    (hx : sin.regs .real [M, N] name
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hd : sin.regs .real [] "d_b" = some (Tile.scalar (some D))) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [M, N] name)
        (Op.ref .real [] "d_b")) sin
      = some ⟨fun idx : TileIndex [M, N] =>
          some (g idx.1.val idx.2.1.val * D)⟩ := by
  rw [pra_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (by rw [evalOp_ref]; exact hx) (by rw [evalOp_ref]; exact hd)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, b, u⟩ := idx
  rfl

/-- The desc-loop `b_s = tl.dot(b_kd, b_q)`: the row factor `2^((BTL−r)·b_b)`
inside `b_kd` factors out of the `BK` contraction. -/
private theorem prb_sDesc_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k i_h : Nat)
    (T K BTL BK BTS : Nat) (i : Nat)
    (hkd : sin.regs .real [BTL, BK] "b_kd"
      = some ⟨fun idx => some (prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k
          T K BTL BK idx.1.val idx.2.1.val
        * praW (prbBeta i_h) (BTL - idx.1.val))⟩)
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some (prbQTile s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_kd")
        (Op.ref .real [BK, BTS] "b_q")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (praW (prbBeta i_h) (BTL - idx.1.val)
            * prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
                idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hkd)
    (by rw [evalOp_ref]; exact hq)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rw [pra_dot2d_elem _ _ r c
    (fun e => prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
        r.val e.val * praW (prbBeta i_h) (BTL - r.val))
    (fun e => prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
        e.val (i + c.val))
    (fun e => rfl) (fun e => rfl)]
  refine congrArg some ?_
  unfold prbSVal
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun e _ => by ring

/-- The desc-loop `b_ds = tl.dot(b_v, b_do)` on the `d_q`-rescaled `b_do`:
the column factor `2^(c·b_b)` factors out of the `BV` contraction. -/
private theorem prb_dsDesc_eval (s sin : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v i_h : Nat)
    (T V BTL BV BTS : Nat) (i : Nat)
    (hv : sin.regs .real [BTL, BV] "b_v"
      = some (prbVTile s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV))
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some ⟨fun idx => some (prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v
          T V BV idx.1.val (i + idx.2.1.val)
        * praW (prbBeta i_h) idx.2.1.val)⟩) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
        (Op.ref .real [BV, BTS] "b_do")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (praW (prbBeta i_h) idx.2.1.val
            * prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
                idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hv)
    (by rw [evalOp_ref]; exact hdo)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rw [pra_dot2d_elem _ _ r c
    (fun p => prbVGuarded s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
        r.val p.val)
    (fun p => prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
        p.val (i + c.val) * praW (prbBeta i_h) c.val)
    (fun p => rfl) (fun p => rfl)]
  refine congrArg some ?_
  unfold prbDsVal
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun p _ => by ring

/-- `b_dv += tl.dot(b_s, tl.trans(b_do))` on all-`some` tiles. -/
private theorem prb_dvAdd_eval (sin : BlockState) (BTL BTS BV : Nat)
    (g f w : Nat → Nat → ℝ)
    (hdv : sin.regs .real [BTL, BV] "b_dv"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩)
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some ⟨fun idx => some (w idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s")
          (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")))) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin BTS, f idx.1.val c.val * w idx.2.1.val c.val)⟩ := by
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")) sin
      = some (Tile.transpose []
          (⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [BV, BTS])) := by
    erw [evalOp_transpose, evalOp_ref, hdo]
    rfl
  have hdot := pra_dot_eval (Op.ref .real [BTL, BTS] "b_s")
    (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")) sin _ _
    (by rw [evalOp_ref]; exact hs) htr
  erw [pra_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdv) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  erw [pra_dot2d_elem _ _ r p (fun c => f r.val c.val) (fun c => w p.val c.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

/-- `b_dk += tl.dot(b_ds, tl.trans(b_q))` on all-`some` tiles. -/
private theorem prb_dkAdd_eval (sin : BlockState) (BTL BTS BK : Nat)
    (g f qF : Nat → Nat → ℝ)
    (hdk : sin.regs .real [BTL, BK] "b_dk"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hds : sin.regs .real [BTL, BTS] "b_ds"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩)
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some ⟨fun idx => some (qF idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_ds")
          (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")))) sin
      = some ⟨fun idx : TileIndex [BTL, BK] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin BTS, f idx.1.val c.val * qF idx.2.1.val c.val)⟩ := by
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")) sin
      = some (Tile.transpose []
          (⟨fun idx => some (qF idx.1.val idx.2.1.val)⟩ : Tile .real [BK, BTS])) := by
    erw [evalOp_transpose, evalOp_ref, hq]
    rfl
  have hdot := pra_dot_eval (Op.ref .real [BTL, BTS] "b_ds")
    (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")) sin _ _
    (by rw [evalOp_ref]; exact hds) htr
  erw [pra_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdk) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  erw [pra_dot2d_elem _ _ r e (fun c => f r.val c.val) (fun c => qF e.val c.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

/-- `b_dk *= d_h[:, None] * scale`: the post-loop row rescale. -/
private theorem prb_dkScale_eval (sin : BlockState) (BTL BK : Nat) (scale : ℝ)
    (g : Nat → Nat → ℝ) (w : Nat → ℝ)
    (hdk : sin.regs .real [BTL, BK] "b_dk"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hdh : sin.regs .real [BTL] "d_h" = some ⟨fun idx => some (w idx.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.mul .real Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))
          (Op.const scale))) sin
      = some ⟨fun idx : TileIndex [BTL, BK] =>
          some (g idx.1.val idx.2.1.val * (w idx.1.val * scale))⟩ := by
  have hexp : evalOp (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h")) sin
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx => some (w idx.1.val)⟩ : Tile .real [BTL])) := by
    erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdh]
  have hmul := pra_mulTile_eval Broadcast.scalarR _ (Op.const scale) sin _ _
    hexp (evalOp_const scale sin)
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdk) hmul]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, e, u⟩ := idx
  rfl

/-- `b_dv *= scale`: the post-loop flat rescale. -/
private theorem prb_dvScale_eval (sin : BlockState) (BTL BV : Nat) (scale : ℝ)
    (g : Nat → Nat → ℝ)
    (hdv : sin.regs .real [BTL, BV] "b_dv"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
        (Op.const scale)) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val * scale)⟩ := by
  rw [pra_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (by rw [evalOp_ref]; exact hdv) (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, p, u⟩ := idx
  rfl

private theorem prb_evalOp_le_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.le h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.le bc vx vy)) := by
  simp [evalOp]

/-- The `o_k[:, None] <= o_q[None, :]` mask at diagonal offset `off` (row `r`
kept iff `r ≤ c + off`). -/
def prbMaskTile (BTL BTS off : Nat) : Tile .bool [BTL, BTS] :=
  ⟨fun idx => decide (idx.1.val ≤ idx.2.1.val + off)⟩

private theorem prb_msLe_eval (sin : BlockState) (BTL BTS off : Nat)
    (hok : sin.regs .nat [BTL] "o_k" = some (Tile.vec fun r => r.val))
    (hoq : sin.regs .nat [BTS] "o_q" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.le ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))) sin
      = some (prbMaskTile BTL BTS off) := by
  rw [prb_evalOp_le_def]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hok,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoq]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  show decide (r.val ≤ c.val + off) = decide (r.val ≤ c.val + off)
  rfl

/-- `d_s = tl.where(m_s, exp2((o_q − o_k)·b_b), 0) * scale`: the scaled
causal decay tile at diagonal offset `off` (with the respelled subtraction
and the identity `b_b.to(tl.float32)` cast; kept lanes hold
`2^(((c+off) − r)·b_b)·scale`, masked-off lanes the hard `0·scale`). -/
private theorem prb_dsDiag_eval (sin : BlockState) (β : ℝ) (BTL BTS off : Nat)
    (scale : ℝ)
    (hm : sin.regs .bool [BTL, BTS] "m_s" = some (prbMaskTile BTL BTS off))
    (hoq : sin.regs .nat [BTS] "o_q" = some (Tile.vec fun r => r.val + off))
    (hok : sin.regs .nat [BTL] "o_k" = some (Tile.vec fun r => r.val))
    (hbb : sin.regs .real [] "b_b" = some (Tile.scalar (some β))) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.where (Op.ref .bool [BTL, BTS] "m_s")
          (Op.exp2 (Op.mul .real Broadcast.scalarR
            (Op.natToReal (Op.sub .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))))
            (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [] "b_b"))))
          (Op.broadcast (Op.const 0.0) [BTL, BTS]))
        (Op.const scale)) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some ((if idx.1.val ≤ idx.2.1.val + off
            then praW β ((idx.2.1.val + off) - idx.1.val)
            else 0) * scale)⟩ := by
  have hwhere : evalOp (Op.where (Op.ref .bool [BTL, BTS] "m_s")
        (Op.exp2 (Op.mul .real Broadcast.scalarR
          (Op.natToReal (Op.sub .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))))
          (Op.castFloat FloatDType.real FloatDType.real (Op.ref .real [] "b_b"))))
        (Op.broadcast (Op.const 0.0) [BTL, BTS])) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (if idx.1.val ≤ idx.2.1.val + off
            then praW β ((idx.2.1.val + off) - idx.1.val)
            else 0)⟩ := by
    have hbb' : sin.regs FloatDType.real.toTileDType [] "b_b"
        = some (Tile.scalar (some β)) := hbb
    rw [evalOp_where]
    simp only [evalOp.eq_def, evalOp_ref, evalOp_const, hm, hoq, hok, hbb, hbb',
      Option.bind, Option.map, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some (Tile.ext fun idx => ?_)
    obtain ⟨r, c, u⟩ := idx
    rw [Tile.select_data]
    simp only [prbMaskTile]
    by_cases h : r.val ≤ c.val + off
    · rw [if_pos (by simpa using h), if_pos h]
      rfl
    · rw [if_neg (by simpa using h), if_neg h]
      norm_num
  rw [pra_mulTile_eval Broadcast.scalarR _ _ sin _ _ hwhere
    (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rfl

/-- The diag-loop `tl.dot(b_k, b_q)` lands lanewise on `prbSVal · (i + c)`. -/
private theorem prb_sDot_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat)
    (T K BTL BK BTS : Nat) (i : Nat)
    (hk : sin.regs .real [BTL, BK] "b_k"
      = some (prbKTile s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK))
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some (prbQTile s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
        (Op.ref .real [BK, BTS] "b_q")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
            idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hk)
    (by rw [evalOp_ref]; exact hq)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rw [pra_dot2d_elem _ _ r c
    (fun e => prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
        r.val e.val)
    (fun e => prbQGuarded s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
        e.val (i + c.val))
    (fun e => rfl) (fun e => rfl)]
  rfl

/-- The diag-loop `b_s = tl.dot(b_k, b_q) * d_s` on an all-`some` decay tile. -/
private theorem prb_sDiag_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (i_bh i_c i_k : Nat)
    (T K BTL BK BTS : Nat) (i : Nat) (fd : Nat → Nat → ℝ)
    (hk : sin.regs .real [BTL, BK] "b_k"
      = some (prbKTile s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK))
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some (prbQTile s q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS i))
    (hd : sin.regs .real [BTL, BTS] "d_s"
      = some ⟨fun idx => some (fd idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
          (Op.ref .real [BK, BTS] "b_q"))
        (Op.ref .real [BTL, BTS] "d_s")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (prbSVal s q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
              idx.1.val (i + idx.2.1.val)
            * fd idx.1.val idx.2.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _
    (prb_sDot_eval s sin q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK BTS
      i hk hq)
    (by rw [evalOp_ref]; exact hd)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rfl

/-- The diag-loop `tl.dot(b_v, b_do)` (unscaled `b_do`) lands lanewise on
`prbDsVal · (i + c)`. -/
private theorem prb_dsDot_eval (s sin : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat)
    (T V BTL BV BTS : Nat) (i : Nat)
    (hv : sin.regs .real [BTL, BV] "b_v"
      = some (prbVTile s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV))
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some (prbDoTile s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
        (Op.ref .real [BV, BTS] "b_do")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
            idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pra_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hv)
    (by rw [evalOp_ref]; exact hdo)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rw [pra_dot2d_elem _ _ r c
    (fun p => prbVGuarded s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
        r.val p.val)
    (fun p => prbDoGuarded s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
        p.val (i + c.val))
    (fun p => rfl) (fun p => rfl)]
  rfl

/-- The diag-loop `b_ds = tl.dot(b_v, b_do) * d_s`. -/
private theorem prb_dsDiagMul_eval (s sin : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (i_bh i_c i_v : Nat)
    (T V BTL BV BTS : Nat) (i : Nat) (fd : Nat → Nat → ℝ)
    (hv : sin.regs .real [BTL, BV] "b_v"
      = some (prbVTile s v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV))
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some (prbDoTile s do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV BTS i))
    (hd : sin.regs .real [BTL, BTS] "d_s"
      = some ⟨fun idx => some (fd idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
          (Op.ref .real [BV, BTS] "b_do"))
        (Op.ref .real [BTL, BTS] "d_s")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (prbDsVal s v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
              idx.1.val (i + idx.2.1.val)
            * fd idx.1.val idx.2.1.val)⟩ := by
  erw [pra_mulTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _
    (prb_dsDot_eval s sin v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
      BTS i hv hdo)
    (by rw [evalOp_ref]; exact hd)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, c, u⟩ := idx
  rfl

/-- `o_q += BTS` shifts the diag arange window. -/
private theorem prb_oqAdd_eval (sin : BlockState) (BTS off : Nat)
    (hoq : sin.regs .nat [BTS] "o_q" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BTS] "o_q")
        (Op.constNat BTS)) sin
      = some (Tile.vec fun r : Fin BTS => r.val + (off + BTS)) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hoq, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨r, u⟩ := idx
  show r.val + off + BTS = r.val + (off + BTS)
  omega


/-! ## The backward streamed (descending, respelled ascending) loop -/

/-- The backward streamed-loop invariant at counter `j`: the accumulators
hold the top `j` blocks `[prbNB − j·BTS, prbNB)` of the key sweep, anchored
at `prbNB − j·BTS`, plus the loop-invariant decay registers. -/
def prbInvD (s0 : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (j : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem ∧
  j ≤ prbTrip T BTS i_c BTL ∧
  s.regs .real [] "b_b" = some (Tile.scalar (some (prbBeta i_h))) ∧
  s.regs .real [] "d_b"
    = some (Tile.scalar (some (Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ)
        * Real.log 2)))) ∧
  s.regs .real [BTL] "d_h"
    = some (⟨fun idx => some (praW (prbBeta i_h) (BTL - idx.1.val))⟩
        : Tile .real [BTL]) ∧
  s.regs .real [BTS] "d_q"
    = some (⟨fun idx => some (praW (prbBeta i_h) idx.1.val)⟩
        : Tile .real [BTS]) ∧
  s.regs .real [BTL, BK] "b_k"
    = some (prbKTile s0 k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK) ∧
  s.regs .real [BTL, BV] "b_v"
    = some (prbVTile s0 v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV) ∧
  s.regs .real [BTL, BK] "b_kd"
    = some ⟨fun idx => some (prbKGuarded s0 k s_qk_h s_qk_t s_qk_d i_bh i_c i_k
        T K BTL BK idx.1.val idx.2.1.val
      * praW (prbBeta i_h) (BTL - idx.1.val))⟩ ∧
  s.regs .real [BTL, BK] "b_dk"
    = some ⟨fun idx => some (prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
        s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (prbNB T BTS - j * BTS) idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL, BV] "b_dv"
    = some ⟨fun idx => some (praW (prbBeta i_h) (BTL - idx.1.val)
        * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - j * BTS) idx.1.val idx.2.1.val)⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The backward prologue** (statements 1–11): the decay registers
`b_b`/`d_b`/`d_h`/`d_q`, the fixed chunk pointers and loads, the zero
accumulators, and the row-decayed `b_kd` establish `prbInvD` at `0`. -/
theorem prbBwdPrologue_run (s : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) :
    ∃ sP, stepStmts
        [ Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat i_h))
                  (Op.const 1.0)))))),
          Stmt.assign .real [] "d_b"
            (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
              (Op.natToReal (Op.constNat BTS)))),
          Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, s_qk_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, s_vo_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_k"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_k") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BV] "b_v"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_v") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_dk" (Op.full [BTL, BK] (Op.const 0)),
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)),
          Stmt.assign .real [BTL] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTL)
                (Op.arange BTL)))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BK] "b_kd"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_k")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))),
          Stmt.assign .real [BTS] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTS))
              (Op.ref .real [] "b_b"))) ] s = some sP
      ∧ prbInvD s q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV 0 sP := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (prb_bb_eval s i_h))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_db_eval _ (prbBeta i_h) BTS (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval k _ _ _ _ [T, K] [BTL, BK] [s_qk_t, s_qk_d]
      (i_bh * s_qk_h) (i_c * BTL) (i_k * BK)
      (prb_constMul_eval _ i_bh s_qk_h)
      (prb_constMul_eval _ i_c BTL)
      (prb_constMul_eval _ i_k BK)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTL, BV] [s_vo_t, s_vo_d]
      (i_bh * s_vo_h) (i_c * BTL) (i_v * BV)
      (prb_constMul_eval _ i_bh s_vo_h)
      (prb_constMul_eval _ i_c BTL)
      (prb_constMul_eval _ i_v BV)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_kLoad_eq s _ k "p_k" s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
      (by rfl) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_vLoad_eq s _ v "p_v" s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
      (by rfl) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pra_zeros_eval [BTL, BK] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pra_zeros_eval [BTL, BV] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dh_eval _ (prbBeta i_h) BTL (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_kd_eval _ BTL BK
      (fun r e => prbKGuarded s k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
        r e)
      (fun r => praW (prbBeta i_h) (BTL - r))
      (by simp [prbKTile]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dq_eval _ (prbBeta i_h) BTS (by simp)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · simp
  · rw [show (⟨fun idx => some (prbDkDesc s q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - 0 * BTS) idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [prbDkDesc]]
    simp
  · rw [show (⟨fun idx => some (praW (prbBeta i_h) (BTL - idx.1.val)
          * prbDvDesc s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - 0 * BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [prbDvDesc]]
    simp

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One streamed (above-diagonal) block** of the backward loop: decay both
accumulators by `d_b` and add the fresh bottom block's contributions. -/
theorem prbBwdDesc_step (s0 sin : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (j : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hlt : j < prbTrip T BTS i_c BTL)
    (hInv : prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV j sin) :
    ∃ s', stepStmts (praBwdDescBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d
        s_vo_h s_vo_t s_vo_d T K V BTL BTS BK BV)
        (sin.setReg "j" .nat [] (Tile.scalar j)) = some s'
      ∧ prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV (j + 1) s' := by
  obtain ⟨hmem, hjle, hbb, hdb, hdh, hdq, hbk, hbv, hbkd, hbdk, hbdv⟩ := hInv
  have hTripMul := prb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hj1 : (j + 1) * BTS ≤ prbTrip T BTS i_c BTL * BTS :=
    Nat.mul_le_mul_right BTS hlt
  have hexp1 : (j + 1) * BTS = j * BTS + BTS := by ring
  have hstep : prbNB T BTS - BTS - j * BTS + BTS ≤ prbNB T BTS := by omega
  have hiadd : prbNB T BTS - BTS - j * BTS + BTS = prbNB T BTS - j * BTS := by
    omega
  have hlonew : prbNB T BTS - (j + 1) * BTS = prbNB T BTS - BTS - j * BTS := by
    omega
  unfold praBwdDescBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_iAssign_eval _ T BTS j (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval q _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (i_bh * s_qk_h) (i_k * BK) (prbNB T BTS - BTS - j * BTS)
      (prb_constMul_eval _ i_bh s_qk_h)
      (prb_constMul_eval _ i_k BK)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval do_ _ _ _ _ [V, T] [BV, BTS] [s_vo_d, s_vo_t]
      (i_bh * s_vo_h) (i_v * BV) (prbNB T BTS - BTS - j * BTS)
      (prb_constMul_eval _ i_bh s_vo_h)
      (prb_constMul_eval _ i_v BV)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_qLoad_eq s0 _ q "p_q" s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS
      (prbNB T BTS - BTS - j * BTS) (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_doLoad_eq s0 _ do_ "p_do" s_vo_h s_vo_t s_vo_d i_bh i_v T V BV BTS
      (prbNB T BTS - BTS - j * BTS) (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_doScale_eval _ BV BTS
      (fun p c => prbDoGuarded s0 do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
        p (prbNB T BTS - BTS - j * BTS + c))
      (fun c => praW (prbBeta i_h) c)
      (by simp [prbDoTile]) (by simpa using hdq)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_accDecay_eval _ "b_dv" BTL BV
      (fun r p => praW (prbBeta i_h) (BTL - r)
        * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - j * BTS) r p)
      (Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2))
      (by simpa using hbdv) (by simpa using hdb)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_sDesc_eval s0 _ q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k i_h T K BTL BK
      BTS (prbNB T BTS - BTS - j * BTS) (by simpa using hbkd) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dvAdd_eval _ BTL BTS BV
      (fun r p => (praW (prbBeta i_h) (BTL - r)
          * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - j * BTS) r p)
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2))
      (fun r c => praW (prbBeta i_h) (BTL - r)
        * prbSVal s0 q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
            r (prbNB T BTS - BTS - j * BTS + c))
      (fun p c => prbDoGuarded s0 do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
          p (prbNB T BTS - BTS - j * BTS + c)
        * praW (prbBeta i_h) c)
      (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_accDecay_eval _ "b_dk" BTL BK
      (fun r e => prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (prbNB T BTS - j * BTS) r e)
      (Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2))
      (by simpa using hbdk) (by simpa using hdb)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dsDesc_eval s0 _ v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v i_h
      T V BTL BV BTS (prbNB T BTS - BTS - j * BTS)
      (by simpa using hbv) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dkAdd_eval _ BTL BTS BK
      (fun r e => prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - j * BTS) r e
        * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2))
      (fun r c => praW (prbBeta i_h) c
        * prbDsVal s0 v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
            r (prbNB T BTS - BTS - j * BTS + c))
      (fun e c => prbQGuarded s0 q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
        e (prbNB T BTS - BTS - j * BTS + c))
      (by simp) (by simp) (by simp [prbQTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hmem
  · omega
  · simpa using hbb
  · simpa using hdb
  · simpa using hdh
  · simpa using hdq
  · simpa using hbk
  · simpa using hbv
  · simpa using hbkd
  · rw [show (⟨fun idx => some (prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - (j + 1) * BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BK])
        = ⟨fun idx => some (prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
            s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - j * BTS) idx.1.val idx.2.1.val
            * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
          + ∑ c : Fin BTS,
              (praW (prbBeta i_h) c.val
                  * prbDsVal s0 v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v
                      T V BTL BV idx.1.val
                      (prbNB T BTS - BTS - j * BTS + c.val))
                * prbQGuarded s0 q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
                    idx.2.1.val (prbNB T BTS - BTS - j * BTS + c.val))⟩
      from Tile.ext fun idx => congrArg some (by
        rw [hlonew, ← hiadd]
        exact (prbDkDesc_step s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - BTS - j * BTS) hstep idx.1.val idx.2.1.val).symm)]
    simp
  · rw [show (⟨fun idx => some (praW (prbBeta i_h) (BTL - idx.1.val)
          * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - (j + 1) * BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun idx => some ((praW (prbBeta i_h) (BTL - idx.1.val)
            * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                (prbNB T BTS - j * BTS) idx.1.val idx.2.1.val)
            * Real.exp (prbBeta i_h * ((BTS : Nat) : ℝ) * Real.log 2)
          + ∑ c : Fin BTS,
              (praW (prbBeta i_h) (BTL - idx.1.val)
                  * prbSVal s0 q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
                      idx.1.val (prbNB T BTS - BTS - j * BTS + c.val))
                * (prbDoGuarded s0 do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
                      idx.2.1.val (prbNB T BTS - BTS - j * BTS + c.val)
                  * praW (prbBeta i_h) c.val))⟩
      from Tile.ext fun idx => congrArg some (by
        rw [hlonew, ← hiadd]
        exact (prbDvDesc_step s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - BTS - j * BTS) hstep idx.1.val idx.2.1.val).symm)]
    simp

set_option maxHeartbeats 4000000 in
/-- **The full streamed loop**: the respelled descending loop consumes all
`prbTrip` above-diagonal blocks. -/
theorem prbBwdDesc_run (s0 sPre : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV 0 sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "j" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil
                  (Op.div .nat Broadcast.nil
                    (Op.sub .nat Broadcast.nil
                      (Op.add .nat Broadcast.nil (Op.constNat T)
                        (Op.constNat BTS))
                      (Op.constNat 1))
                    (Op.constNat BTS))
                  (Op.constNat BTS))
                (Op.constNat ((i_c + 1) * BTL)))
              (Op.constNat BTS))
            (Op.constNat 1))
          (Op.constNat BTS))
        (Op.constNat 1)
        (praBwdDescBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d T K V BTL BTS BK BV)) sPre = some sL
      ∧ prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV (prbTrip T BTS i_c BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "j")
      (P := fun j s => prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t
        s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV j s)
      (evalOp_constNat 0 sPre)
      (prb_trip_eval sPre T BTS i_c BTL)
      (evalOp_constNat 1 sPre)
      one_ne_zero
      hInv
      (fun j s hj hPj =>
        prbBwdDesc_step s0 s q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t
          s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV j hBTS hBTSpos
          hj hPj)
  have hfin : final = prbTrip T BTS i_c BTL := le_antisymm hP.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩


/-! ## The backward diagonal loop -/

/-- The backward diagonal-loop invariant at counter `i` (absolute key offset
within the diagonal chunk): the accumulators hold the diagonal partial sums
past the fixed rescaled desc-exit terms. -/
def prbInvG (s0 : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem ∧
  i_c * BTL ≤ i ∧ BTS ∣ (i - i_c * BTL) ∧ i ≤ (i_c + 1) * BTL ∧
  s.regs .real [] "b_b" = some (Tile.scalar (some (prbBeta i_h))) ∧
  s.regs .real [BTL, BK] "b_k"
    = some (prbKTile s0 k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK) ∧
  s.regs .real [BTL, BV] "b_v"
    = some (prbVTile s0 v s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV) ∧
  s.regs .nat [BTS] "o_q"
    = some (Tile.vec fun r => r.val + (i - i_c * BTL)) ∧
  s.regs .nat [BTL] "o_k" = some (Tile.vec fun r => r.val) ∧
  s.regs .real [BTL, BK] "b_dk"
    = some ⟨fun idx => some (prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
        s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) i idx.1.val idx.2.1.val
      + prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
          idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL, BV] "b_dv"
    = some ⟨fun idx => some (prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
        s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) i idx.1.val idx.2.1.val
      + prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
          idx.1.val idx.2.1.val)⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The mid-section**: the two post-loop rescales (`b_dk *= d_h[:,None] *
scale`, `b_dv *= scale`), the barrier no-op, and the `o_q`/`o_k` aranges
carry the desc-exit invariant into the diagonal invariant at `i = i_c·BTL`
(the rescale bridges fold the anchored accumulators into the keep-indicator
partial sums). -/
theorem prbBwdMid_run (s0 sL1 : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : prbInvD s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV
      (prbTrip T BTS i_c BTL) sL1) :
    ∃ sM, stepStmts
        [ Stmt.assign .real [BTL, BK] "b_dk"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_dk")
              (Op.mul .real Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))
                (Op.const scale))),
          Stmt.assign .real [BTL, BV] "b_dv"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
              (Op.const scale)),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL) ] sL1 = some sM
      ∧ prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV (i_c * BTL) sM := by
  obtain ⟨hmem, hjle, hbb, hdb, hdh, hdq, hbk, hbv, hbkd, hbdk, hbdv⟩ := hInv
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dkScale_eval _ BTL BK scale
      (fun r e => prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) r e)
      (fun r => praW (prbBeta i_h) (BTL - r))
      (by simpa using hbdk) (by simpa using hdh)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dvScale_eval _ BTL BV scale
      (fun r p => praW (prbBeta i_h) (BTL - r)
        * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) r p)
      (by simpa using hbdv)))]
  rw [stepStmts.cons_some (pra_ifThen_false_noop [] _)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTS _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTL _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, le_rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hmem
  · simp
  · have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    omega
  · simpa using hbb
  · simpa using hbk
  · simpa using hbv
  · rw [show ((Tile.vec fun r : Fin BTS =>
          r.val + (i_c * BTL - i_c * BTL)) : Tile .nat [BTS])
        = ((Tile.vec fun r : Fin BTS => r.val) : Tile .nat [BTS])
      from Tile.ext fun r => by
        show r.1.val + (i_c * BTL - i_c * BTL) = r.1.val
        omega]
    simp
  · simp
  · rw [show (⟨fun idx => some (prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val
        + prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun idx => some (prbDkDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
            s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) idx.1.val idx.2.1.val
          * (praW (prbBeta i_h) (BTL - idx.1.val) * scale))⟩
      from Tile.ext fun idx => congrArg some (by
        rw [show prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val = 0 from by
          simp [prbDkPart], zero_add]
        exact (prbDkDesc_scale_bridge s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV hBTS
          hBTSpos idx.1.val idx.2.1.val idx.1.isLt).symm)]
    simp
  · rw [show (⟨fun idx => some (prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val
        + prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some ((praW (prbBeta i_h) (BTL - idx.1.val)
            * prbDvDesc s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                (prbNB T BTS - prbTrip T BTS i_c BTL * BTS)
                idx.1.val idx.2.1.val) * scale)⟩
      from Tile.ext fun idx => congrArg some (by
        rw [show prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val = 0 from by
          simp [prbDvPart], zero_add]
        exact (prbDvDesc_scale_bridge s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV hBTS
          hBTSpos idx.1.val idx.2.1.val idx.1.isLt).symm)]
    simp

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 8000 in
/-- **One diagonal (causally masked, decay-weighted) block** of the backward
loop. -/
theorem prbBwdDiag_step (s0 sin : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat)
    (hBTS : BTL % BTS = 0)
    (hlt : i < (i_c + 1) * BTL)
    (hInv : prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (praBwdDiagBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d
        s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV)
        (sin.setReg "i" .nat [] (Tile.scalar i)) = some s'
      ∧ prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hmem, hlo, hdvd, hle, hbb, hbk, hbv, hoq, hok, hbdk, hbdv⟩ := hInv
  unfold praBwdDiagBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval q _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (i_bh * s_qk_h) (i_k * BK) i
      (prb_constMul_eval _ i_bh s_qk_h)
      (prb_constMul_eval _ i_k BK)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval do_ _ _ _ _ [V, T] [BV, BTS] [s_vo_d, s_vo_t]
      (i_bh * s_vo_h) (i_v * BV) i
      (prb_constMul_eval _ i_bh s_vo_h)
      (prb_constMul_eval _ i_v BV)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_qLoad_eq s0 _ q "p_q" s_qk_h s_qk_t s_qk_d i_bh i_k T K BK BTS i
      (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_doLoad_eq s0 _ do_ "p_do" s_vo_h s_vo_t s_vo_d i_bh i_v T V BV BTS i
      (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_msLe_eval _ BTL BTS (i - i_c * BTL)
      (by simpa using hok) (by simpa using hoq)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dsDiag_eval _ (prbBeta i_h) BTL BTS (i - i_c * BTL) scale
      (by simp) (by simpa using hoq) (by simpa using hok)
      (by simpa using hbb)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_sDiag_eval s0 _ q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK BTS i
      (fun r c => (if r ≤ c + (i - i_c * BTL) then
          praW (prbBeta i_h) ((c + (i - i_c * BTL)) - r) else 0) * scale)
      (by simpa using hbk) (by simp) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dsDiagMul_eval s0 _ v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v
      T V BTL BV BTS i
      (fun r c => (if r ≤ c + (i - i_c * BTL) then
          praW (prbBeta i_h) ((c + (i - i_c * BTL)) - r) else 0) * scale)
      (by simpa using hbv) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dkAdd_eval _ BTL BTS BK
      (fun r e => prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) i r e
        + prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r e)
      (fun r c => prbDsVal s0 v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL
          BV r (i + c)
        * ((if r ≤ c + (i - i_c * BTL) then
            praW (prbBeta i_h) ((c + (i - i_c * BTL)) - r) else 0) * scale))
      (fun e c => prbQGuarded s0 q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
        e (i + c))
      (by simpa using hbdk) (by simp) (by simp [prbQTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_dvAdd_eval _ BTL BTS BV
      (fun r p => prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) i r p
        + prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS) r p)
      (fun r c => prbSVal s0 q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
          r (i + c)
        * ((if r ≤ c + (i - i_c * BTL) then
            praW (prbBeta i_h) ((c + (i - i_c * BTL)) - r) else 0) * scale))
      (fun p c => prbDoGuarded s0 do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
        p (i + c))
      (by simpa using hbdv) (by simp) (by simp [prbDoTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (prb_oqAdd_eval _ BTS (i - i_c * BTL) (by simpa using hoq)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hmem
  · omega
  · have h1 : i + BTS - i_c * BTL = (i - i_c * BTL) + BTS := by omega
    rw [h1]
    exact Dvd.dvd.add hdvd dvd_rfl
  · have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    have hBTLd : BTS ∣ BTL := Nat.dvd_of_mod_eq_zero hBTS
    have h2 : i - i_c * BTL < BTL := by omega
    have h3 : (i - i_c * BTL) + BTS ≤ BTL := pra_dvd_step_le hdvd hBTLd h2
    omega
  · simpa using hbb
  · simpa using hbk
  · simpa using hbv
  · rw [show ((Tile.vec fun r : Fin BTS => r.val + (i + BTS - i_c * BTL))
          : Tile .nat [BTS])
        = ((Tile.vec fun r : Fin BTS => r.val + ((i - i_c * BTL) + BTS))
          : Tile .nat [BTS])
      from Tile.ext fun r => by
        show r.1.val + (i + BTS - i_c * BTL) = r.1.val + ((i - i_c * BTL) + BTS)
        omega]
    simp
  · simpa using hok
  · rw [show (⟨fun idx => some (prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (i + BTS) idx.1.val idx.2.1.val
        + prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun idx => some ((prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
            s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (i_c * BTL) i idx.1.val idx.2.1.val
          + prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
              idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              (prbDsVal s0 v do_ s_vo_h s_vo_t s_vo_d i_bh i_c i_v T V BTL BV
                  idx.1.val (i + c.val)
                * ((if idx.1.val ≤ c.val + (i - i_c * BTL) then
                      praW (prbBeta i_h) ((c.val + (i - i_c * BTL)) - idx.1.val)
                    else 0) * scale))
                * prbQGuarded s0 q s_qk_h s_qk_t s_qk_d i_bh i_k T K BK
                    idx.2.1.val (i + c.val))⟩
      from Tile.ext fun idx => congrArg some
        (prbDkDiag_step' s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i
          (prbDkPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)
          idx.1.val idx.2.1.val hlo).symm]
    simp
  · rw [show (⟨fun idx => some (prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d
          s_vo_h s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
          (i_c * BTL) (i + BTS) idx.1.val idx.2.1.val
        + prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some ((prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h
            s_vo_t s_vo_d i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (i_c * BTL) i idx.1.val idx.2.1.val
          + prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
              (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
              idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              (prbSVal s0 q k s_qk_h s_qk_t s_qk_d i_bh i_c i_k T K BTL BK
                  idx.1.val (i + c.val)
                * ((if idx.1.val ≤ c.val + (i - i_c * BTL) then
                      praW (prbBeta i_h) ((c.val + (i - i_c * BTL)) - idx.1.val)
                    else 0) * scale))
                * prbDoGuarded s0 do_ s_vo_h s_vo_t s_vo_d i_bh i_v T V BV
                    idx.2.1.val (i + c.val))⟩
      from Tile.ext fun idx => congrArg some
        (prbDvDiag_step' s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV i
          (prbDvPart s0 q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
            (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
            idx.1.val idx.2.1.val)
          idx.1.val idx.2.1.val hlo).symm]
    simp

set_option maxHeartbeats 4000000 in
/-- **The full diagonal loop.** -/
theorem prbBwdDiag_run (s0 sPre : BlockState) (q k v do_ : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d
      s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV (i_c * BTL) sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "i"
        (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
        (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
        (praBwdDiagBody q do_ i_bh i_k i_v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d scale T K V BTL BTS BK BV)) sPre = some sL
      ∧ prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BTS BK BV ((i_c + 1) * BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "i")
      (P := fun i s => prbInvG s0 q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t
        s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV i s)
      (prb_constMul_eval sPre i_c BTL)
      (evalOp_constNat ((i_c + 1) * BTL) sPre)
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        prbBwdDiag_step s0 s q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t
          s_qk_d s_vo_h s_vo_t s_vo_d scale T K V BTL BTS BK BV i hBTS hi hPi)
  have hfin : final = (i_c + 1) * BTL := le_antisymm hP.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩


/-! ## Backward stores and ★ main theorem -/

/-- The `dk` store address at lane `(r, e)`. -/
def prbDkOffset (i_bh i_c i_k i_v B H s_qk_h s_qk_t s_qk_d BTL BK : Nat)
    (idx : TileIndex [BTL, BK]) : Nat :=
  (i_bh + B * H * i_v) * s_qk_h + (i_c * BTL + idx.1.val) * s_qk_t
    + (i_k * BK + idx.2.1.val) * s_qk_d

/-- A `dk` store lane is *active* when it maps inside the `T × K` window. -/
def prbDkActive (i_c i_k T K BTL BK : Nat) (idx : TileIndex [BTL, BK]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_k * BK + idx.2.1.val < K

/-- The `dv` store address at lane `(r, p)`. -/
def prbDvOffset (i_bh i_c i_k i_v B H s_vo_h s_vo_t s_vo_d BTL BV : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  (i_bh + B * H * i_k) * s_vo_h + (i_c * BTL + idx.1.val) * s_vo_t
    + (i_v * BV + idx.2.1.val) * s_vo_d

/-- A `dv` store lane is *active* when it maps inside the `T × V` window. -/
def prbDvActive (i_c i_v T V BTL BV : Nat) (idx : TileIndex [BTL, BV]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_v * BV + idx.2.1.val < V

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The backward epilogue**: `p_dk`/`p_dv` makes and the two block stores,
from the diag-exit invariant, with both readbacks folded to the full-sweep
closed forms. -/
theorem prbBwdStores_run (s sL2 : BlockState) (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat) (s_qk_h s_qk_t s_vo_h s_vo_t : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv) (hσk : BK ≤ s_qk_t) (hσv : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInvF : prbInvG s q k v do_ i_bh i_c i_k i_v i_h s_qk_h s_qk_t 1 s_vo_h
      s_vo_t 1 scale T K V BTL BTS BK BV ((i_c + 1) * BTL) sL2) :
    ∃ sF, stepStmts
        [ Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.store .real [BTL, BK]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk") [0, 1])
            (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv") [0, 1])
            (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ] sL2 = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], prbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (prbDkOffset i_bh i_c i_k i_v B H s_qk_h s_qk_t 1 BTL BK idx)
            = prbDkOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k
                i_v i_h scale T K V BTL BTS BK BV idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], prbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (prbDvOffset i_bh i_c i_k i_v B H s_vo_h s_vo_t 1 BTL BV idx)
            = prbDvOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k
                i_v i_h scale T K V BTL BTS BK BV idx.1.val idx.2.1.val) := by
  obtain ⟨hFmem, hFlo, hFdvd, hFle, hFbb, hFbk, hFbv, hFoq, hFok, hFbdk,
    hFbdv⟩ := hInvF
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval dk _ _ _ _ [T, K] [BTL, BK] [s_qk_t, 1]
      ((i_bh + B * H * i_v) * s_qk_h) (i_c * BTL) (i_k * BK)
      (evalOp_constNat _ _)
      (prb_constMul_eval _ i_c BTL)
      (prb_constMul_eval _ i_k BK)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pra_makeBlockPtr_2d_eval dv _ _ _ _ [T, V] [BTL, BV] [s_vo_t, 1]
      ((i_bh + B * H * i_k) * s_vo_h) (i_c * BTL) (i_v * BV)
      (evalOp_constNat _ _)
      (prb_constMul_eval _ i_c BTL)
      (prb_constMul_eval _ i_v BV)))]
  rw [stepStmts.cons_some (praStore_step_eq _ dk "b_dk" "p_dk"
    ((i_bh + B * H * i_v) * s_qk_h) s_qk_t (i_c * BTL) (i_k * BK) T K BTL BK
    (fun r e => prbDkOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c
      i_k i_v i_h scale T K V BTL BTS BK BV r e)
    ⟨fun idx => some (prbDkPart s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) idx.1.val idx.2.1.val
      + prbDkPart s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k i_v
          i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
          idx.1.val idx.2.1.val)⟩
    (fun r e => congrArg some
      (prbDkOut_split s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k
        i_v i_h scale T K V BTL BTS BK BV hBTS hBTSpos r.val e.val))
    (by simpa using hFbdk)
    (by simp))]
  obtain ⟨hKpids, hKregs, hKread, hKother, hKuntouched⟩ := praStore_step_props
    ((sL2.setReg "p_dk" .blockPtr [BTL, BK]
        ⟨fun _ => BlockPtr.mk dk ((i_bh + B * H * i_v) * s_qk_h) [T, K]
          [BTL, BK] [s_qk_t, 1] [i_c * BTL, i_k * BK]⟩).setReg
      "p_dv" .blockPtr [BTL, BV]
        ⟨fun _ => BlockPtr.mk dv ((i_bh + B * H * i_k) * s_vo_h) [T, V]
          [BTL, BV] [s_vo_t, 1] [i_c * BTL, i_v * BV]⟩)
    dk ((i_bh + B * H * i_v) * s_qk_h) s_qk_t (i_c * BTL) (i_k * BK) T K BTL BK
    (fun r e => prbDkOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c
      i_k i_v i_h scale T K V BTL BTS BK BV r e)
    (praStoreAddr_injective _ _ _ _ BTL BK hσk)
  rw [stepStmts.cons_some (praStore_step_eq _ dv "b_dv" "p_dv"
    ((i_bh + B * H * i_k) * s_vo_h) s_vo_t (i_c * BTL) (i_v * BV) T V BTL BV
    (fun r p => prbDvOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c
      i_k i_v i_h scale T K V BTL BTS BK BV r p)
    ⟨fun idx => some (prbDvPart s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1
        i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) idx.1.val idx.2.1.val
      + prbDvPart s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k i_v
          i_h scale T K V BTL BTS BK BV
          (prbNB T BTS - prbTrip T BTS i_c BTL * BTS) (prbNB T BTS)
          idx.1.val idx.2.1.val)⟩
    (fun r p => congrArg some
      (prbDvOut_split s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c i_k
        i_v i_h scale T K V BTL BTS BK BV hBTS hBTSpos r.val p.val))
    (by rw [hKregs]; simpa using hFbdv)
    (by rw [hKregs]; simp))]
  rw [stepStmts.nil]
  obtain ⟨hVpids, hVregs, hVread, hVother, hVuntouched⟩ := praStore_step_props
    (praStoreState
      ((sL2.setReg "p_dk" .blockPtr [BTL, BK]
          ⟨fun _ => BlockPtr.mk dk ((i_bh + B * H * i_v) * s_qk_h) [T, K]
            [BTL, BK] [s_qk_t, 1] [i_c * BTL, i_k * BK]⟩).setReg
        "p_dv" .blockPtr [BTL, BV]
          ⟨fun _ => BlockPtr.mk dv ((i_bh + B * H * i_k) * s_vo_h) [T, V]
            [BTL, BV] [s_vo_t, 1] [i_c * BTL, i_v * BV]⟩)
      dk ((i_bh + B * H * i_v) * s_qk_h) s_qk_t (i_c * BTL) (i_k * BK) T K
      BTL BK
      (fun r e => prbDkOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c
        i_k i_v i_h scale T K V BTL BTS BK BV r e))
    dv ((i_bh + B * H * i_k) * s_vo_h) s_vo_t (i_c * BTL) (i_v * BV) T V BTL BV
    (fun r p => prbDvOut s q k v do_ s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 i_bh i_c
      i_k i_v i_h scale T K V BTL BTS BK BV r p)
    (praStoreAddr_injective _ _ _ _ BTL BV hσv)
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hact
    rw [hVother dk _ hDkDv]
    exact hKread idx hact
  · intro idx hact
    exact hVread idx hact

/-! ## ★ Backward main theorem -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **★ Backward dk/dv main theorem: the `dk` and `dv` stores are the genuine
retention-gradient closed forms.**

For every scalar-argument tuple `(i_bh, i_c, i_k, i_v, i_h)` the shell would
pass (universally quantified binders), executing the full backward dk/dv
surface succeeds, the `dk` block store holds
`Σ_{t ≥ i_c·BTL + r} 2^((t − (i_c·BTL+r))·b_b) · scale · ds(r,t) · qᵀ[e,t]`
and the `dv` block store holds
`Σ_{t ≥ i_c·BTL + r} 2^((t − (i_c·BTL+r))·b_b) · scale · s(r,t) · doᵀ[p,t]`
at every in-window lane — the causally kept key sweep from the diagonal row
to the streamed top `cdiv(T,BTS)·BTS` (loads beyond `T` read as zero, so
ragged tails are exact), with the per-head decay
`b_b = log2(1 − 2^(−5 − i_h))`, `s(r,t) = k[r]·qᵀ[t]` the raw retention
score, and `ds(r,t) = v[r]·doᵀ[t]` the raw output-gradient contraction.

Side conditions: `dk ≠ dv` (distinct output buffers), the host's contiguous
last-dim strides `s_qk_d = 1` / `s_vo_d = 1` with `BK ≤ s_qk_t` /
`BV ≤ s_vo_t` (store-lane injectivity), and the host's own
`assert BTL % BTS == 0`. -/
specification pra_bwd_dkv_exec_genuine
    (s : BlockState) (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv) (hSkd : s_qk_d = 1) (hSvd : s_vo_d = 1)
    (hσk : BK ≤ s_qk_t) (hσv : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pra_bwd_dkv_surface q k v do_ dk dv i_bh i_c i_k i_v i_h
        s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], prbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (prbDkOffset i_bh i_c i_k i_v B H s_qk_h s_qk_t s_qk_d BTL BK idx)
            = prbDkOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], prbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (prbDvOffset i_bh i_c i_k i_v B H s_vo_h s_vo_t s_vo_d BTL BV idx)
            = prbDvOut s q k v do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                i_bh i_c i_k i_v i_h scale T K V BTL BTS BK BV
                idx.1.val idx.2.1.val) := by
  subst hSkd
  subst hSvd
  obtain ⟨sP, hPro, hInvP⟩ := prbBwdPrologue_run s q k v do_ i_bh i_c i_k i_v
    i_h s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV
  obtain ⟨sL1, hL1, hInvDF⟩ := prbBwdDesc_run s sP q k v do_ i_bh i_c i_k i_v
    i_h s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hBTS hBTSpos
    hInvP
  obtain ⟨sM, hMid, hInvG0⟩ := prbBwdMid_run s sL1 q k v do_ i_bh i_c i_k i_v
    i_h s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hBTS hBTSpos
    hInvDF
  obtain ⟨sL2, hL2, hInvGF⟩ := prbBwdDiag_run s sM q k v do_ i_bh i_c i_k i_v
    i_h s_qk_h s_qk_t 1 s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hBTS hBTSpos
    hInvG0
  rw [exec, pra_bwd_body_eq]
  rw [show ([ Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat i_h))
                  (Op.const 1.0)))))),
          Stmt.assign .real [] "d_b"
            (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
              (Op.natToReal (Op.constNat BTS)))),
          Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_k"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_k") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BV] "b_v"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_v") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_dk" (Op.full [BTL, BK] (Op.const 0)),
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)),
          Stmt.assign .real [BTL] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTL)
                (Op.arange BTL)))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BK] "b_kd"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_k")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))),
          Stmt.assign .real [BTS] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTS))
              (Op.ref .real [] "b_b"))),
          Stmt.forRangeDyn "j" (Op.constNat 0)
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil
                      (Op.div .nat Broadcast.nil
                        (Op.sub .nat Broadcast.nil
                          (Op.add .nat Broadcast.nil (Op.constNat T) (Op.constNat BTS))
                          (Op.constNat 1))
                        (Op.constNat BTS))
                      (Op.constNat BTS))
                    (Op.constNat ((i_c + 1) * BTL)))
                  (Op.constNat BTS))
                (Op.constNat 1))
              (Op.constNat BTS))
            (Op.constNat 1)
            (praBwdDescBody q do_ i_bh i_k i_v s_qk_h s_qk_t 1
              s_vo_h s_vo_t 1 T K V BTL BTS BK BV),
          Stmt.assign .real [BTL, BK] "b_dk"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_dk")
              (Op.mul .real Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))
                (Op.const scale))),
          Stmt.assign .real [BTL, BV] "b_dv"
            (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
              (Op.const scale)),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL),
          Stmt.forRangeDyn "i"
            (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
            (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
            (praBwdDiagBody q do_ i_bh i_k i_v s_qk_h s_qk_t 1
              s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV),
          Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.store .real [BTL, BK]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk") [0, 1])
            (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv") [0, 1])
            (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ] : List Stmt)
      = [ Stmt.assign .real [] "b_b"
            (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
              (Op.exp2 (Op.sub .real Broadcast.nil
                (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
                (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat i_h))
                  (Op.const 1.0)))))),
          Stmt.assign .real [] "d_b"
            (Op.exp2 (Op.mul .real Broadcast.nil (Op.ref .real [] "b_b")
              (Op.natToReal (Op.constNat BTS)))),
          Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_qk_h))
              [T, K] [BTL, BK] [s_qk_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_vo_h))
              [T, V] [BTL, BV] [s_vo_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.assign .real [BTL, BK] "b_k"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_k") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BV] "b_v"
            (Op.load .real
              (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_v") [0, 1])
              MaskOpt.none),
          Stmt.assign .real [BTL, BK] "b_dk" (Op.full [BTL, BK] (Op.const 0)),
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)),
          Stmt.assign .real [BTL] "d_h"
            (Op.exp2 (Op.mul .real Broadcast.scalarR
              (Op.natToReal (Op.sub .nat Broadcast.scalarL (Op.constNat BTL)
                (Op.arange BTL)))
              (Op.ref .real [] "b_b"))),
          Stmt.assign .real [BTL, BK] "b_kd"
            (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.ref .real [BTL, BK] "b_k")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))),
          Stmt.assign .real [BTS] "d_q"
            (Op.exp2 (Op.mul .real Broadcast.scalarR (Op.natToReal (Op.arange BTS))
              (Op.ref .real [] "b_b"))) ]
        ++ (Stmt.forRangeDyn "j" (Op.constNat 0)
            (Op.div .nat Broadcast.nil
              (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil
                      (Op.div .nat Broadcast.nil
                        (Op.sub .nat Broadcast.nil
                          (Op.add .nat Broadcast.nil (Op.constNat T)
                            (Op.constNat BTS))
                          (Op.constNat 1))
                        (Op.constNat BTS))
                      (Op.constNat BTS))
                    (Op.constNat ((i_c + 1) * BTL)))
                  (Op.constNat BTS))
                (Op.constNat 1))
              (Op.constNat BTS))
            (Op.constNat 1)
            (praBwdDescBody q do_ i_bh i_k i_v s_qk_h s_qk_t 1
              s_vo_h s_vo_t 1 T K V BTL BTS BK BV)
          :: ([ Stmt.assign .real [BTL, BK] "b_dk"
                (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                  (Op.ref .real [BTL, BK] "b_dk")
                  (Op.mul .real Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BTL] "d_h"))
                    (Op.const scale))),
              Stmt.assign .real [BTL, BV] "b_dv"
                (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BV] "b_dv")
                  (Op.const scale)),
              Stmt.ifThen (Op.constBool Bool.false) [],
              Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
              Stmt.assign .nat [BTL] "o_k" (Op.arange BTL) ]
            ++ (Stmt.forRangeDyn "i"
                (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
                (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
                (praBwdDiagBody q do_ i_bh i_k i_v s_qk_h s_qk_t 1
                  s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV)
              :: [ Stmt.assign .blockPtr [BTL, BK] "p_dk"
                    (Op.makeBlockPtrDynOffsets dk
                      (Op.constNat ((i_bh + B * H * i_v) * s_qk_h))
                      [T, K] [BTL, BK] [s_qk_t, 1]
                      [Op.mul .nat Broadcast.nil (Op.constNat i_c)
                          (Op.constNat BTL),
                        Op.mul .nat Broadcast.nil (Op.constNat i_k)
                          (Op.constNat BK)]),
                  Stmt.assign .blockPtr [BTL, BV] "p_dv"
                    (Op.makeBlockPtrDynOffsets dv
                      (Op.constNat ((i_bh + B * H * i_k) * s_vo_h))
                      [T, V] [BTL, BV] [s_vo_t, 1]
                      [Op.mul .nat Broadcast.nil (Op.constNat i_c)
                          (Op.constNat BTL),
                        Op.mul .nat Broadcast.nil (Op.constNat i_v)
                          (Op.constNat BV)]),
                  Stmt.store .real [BTL, BK]
                    (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk")
                      [0, 1])
                    (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
                  Stmt.store .real [BTL, BV]
                    (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv")
                      [0, 1])
                    (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ])))
      from rfl]
  rw [stepStmts.append_some hPro, stepStmts.cons_some hL1,
    stepStmts.append_some hMid, stepStmts.cons_some hL2]
  obtain ⟨sF, hSt, hR1, hR2⟩ := prbBwdStores_run s sL2 q k v do_ dk dv
    i_bh i_c i_k i_v i_h s_qk_h s_qk_t s_vo_h s_vo_t scale
    B H T K V BTL BTS BK BV hDkDv hσk hσv hBTS hBTSpos hInvGF
  exact ⟨sF, hSt, hR1, hR2⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention

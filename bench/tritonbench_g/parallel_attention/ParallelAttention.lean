import VeriTile.Triton

/-!
# `parallel_attention` — strict per-kernel correctness

`parallel_attention.py` implements *rebased* (quadratic-kernel) attention:
scores are `(q·k)²` instead of `exp(q·k)`, so the causal attention output
factors into an unnormalized value sum `o` plus a separately-stored
normalizer `z` (the host divides `o / z` outside the kernel). The forward
JIT streams K/V blocks in two phases — the strictly-below-diagonal blocks
unmasked, then the diagonal block masked causally — and the backward dk/dv
helper streams Q/dO blocks in the *opposite* order: the
strictly-above-diagonal blocks first via a **descending** `-BTS` loop, then
the diagonal block masked.

## Scope

This file ports two of the four `@triton.jit` kernels:

* `parallel_attention.py`'s `parallel_rebased_fwd_kernel` — the launched
  forward JIT (program `(i_kv, i_c, i_bh)` on grid
  `(NK·NV, cdiv(T, BTL), B·H)`), and
* `_parallel_rebased_bwd_dkv` — the backward dk/dv helper JIT, verified as a
  standalone kernel over universally-quantified scalar arguments
  `i_bh, i_c, i_k, i_v, i_h` (the values the shell
  `parallel_rebased_bwd_kernel` would pass it).

The backward shell `parallel_rebased_bwd_kernel` (a prologue plus two helper
calls separated by `tl.debug_barrier()`) and the dq helper
`_parallel_rebased_bwd_dq` are not transcribed. The host launch, the
host-side `o.sum(0)` / `dq.sum(0)`-style NK/NV reductions, and the final
`o / (z + eps)` normalization are the *trusted boundary*, not proof
obligations here.

## Translation-surface blocker

Translation-surface blocker: the DSL has no cross-`@triton.jit`
function-call surface, so `_parallel_rebased_bwd_dkv` is ported as a
standalone kernel whose Python scalar arguments `i_bh, i_c, i_k, i_v, i_h`
become universally-quantified Lean binders (the shell
`parallel_rebased_bwd_kernel` and `_parallel_rebased_bwd_dq` are out of
scope), its trailing bare `return` is dropped, and its **descending** loop
`for i in range(cdiv(T, BTS)·BTS − BTS, (i_c+1)·BTL − BTS, −BTS)` is
respelled as the ascending change of variable
`for j in range(0, cdiv(cdiv(T, BTS)·BTS − (i_c+1)·BTL, BTS))` with
body-first `i = cdiv(T, BTS)·BTS − BTS − j·BTS` (the `± BTS` in Python's
`start − stop` cancels exactly in ℤ, and ℕ-truncated subtraction reproduces
Python's empty-range behaviour, so the trip counts agree for **all**
parameter values). The Lean surfaces are therefore
not line-for-line textual matches of the Python bodies, and the textual
py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port on this
marker (registered in `proof_blockers.md`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `.to(b_q.dtype)` /
`.to(p_o.dtype.element_ty)` casts are identities at `tl.float32`);
`num_warps`/`num_stages` and `tl.debug_barrier()` (an intra-program no-op
fence, transcribed verbatim) are not modeled at the scheduling level.
`boundary_check=(0, 1)` block-pointer loads zero-pad out-of-window lanes;
the specs bake that window directly into the guarded value functions, so
the headlines hold for **arbitrary ragged tails** with no divisibility
hypothesis on `T`. The only trip-count hypothesis is the host's own
`assert BTL % BTS == 0`.

The port targets `parallel_attention.py`'s `parallel_rebased_fwd_kernel`.
-/

namespace VeriTile.Bench.TritonBenchG.ParallelAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorems:** `pa_fwd_o_exec_genuine`, `pa_bwd_dkv_exec_genuine` -/

section Correct_without_Rounding

/-- Faithful transcription of `parallel_rebased_fwd_kernel`. -/
def pa_fwd_surface
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  i_kv = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  NV = tl.cdiv($(V), $(BV))
  i_k = i_kv // NV
  i_v = i_kv % NV
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
  b_z = tl.zeros([$(BTL)], dtype=tl.float32)
  for _i in range($(0), i_c * $(BTL), $(BTS)) {
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_z += tl.sum(b_s, axis=1)
    b_o = b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTL))
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
    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_s = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_z += tl.sum(b_s, axis=1)
    b_o += tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    p_k = tl.advance(p_k, [$(0), $(BTS)])
    p_v = tl.advance(p_v, [$(BTS), $(0)])
    o_k += $(BTS)
  }
  p_o = tl.make_block_ptr(base=o + (i_bh + $(B) * $(H) * i_k) * $(s_vo_h),
    shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
    offsets=(i_c * $(BTL), i_v * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  p_z = z + (i_bh + $(B) * $(H) * i_k) * $(T) + i_c * $(BTL) + tl.arange(0, $(BTL))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_z, (b_z).to(p_z.dtype.element_ty),
    mask=((i_c * $(BTL) + tl.arange(0, $(BTL))) < $(T)))
}

/-- The forward surface lowers to the algorithm layer. -/
theorem pa_fwd_surface_toAlgorithm_supported
    (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ∃ alg, (pa_fwd_surface q k v o z s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      scale B H T K V BTL BTS BK BV).toAlgorithm? = Except.ok alg := by
  simp [pa_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `_parallel_rebased_bwd_dkv`, with the helper's
scalar arguments `i_bh, i_c, i_k, i_v, i_h` as universally-quantified
binders, the descending loop spelled as its ascending change of variable
(see the preamble), and the trailing bare `return` dropped. `i_h` is unused
(rebased attention has no per-head decay); it is kept as a binder to match
the Python signature. -/
def pa_bwd_dkv_surface
    (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ComputeKernel := triton {
  p_k = tl.make_block_ptr(base=k + $(i_bh) * $(s_k_h),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_v = tl.make_block_ptr(base=v + $(i_bh) * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_dk = tl.zeros([$(BTL), $(BK)], dtype=tl.float32)
  b_dv = tl.zeros([$(BTL), $(BV)], dtype=tl.float32)
  for j in range($(0),
      tl.cdiv(tl.cdiv($(T), $(BTS)) * $(BTS) - $(((i_c + 1) * BTL : Nat)), $(BTS)),
      $(1)) {
    i = tl.cdiv($(T), $(BTS)) * $(BTS) - $(BTS) - j * $(BTS)
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    b_s = tl.dot((b_k).to(b_q.dtype), b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false) * $((scale : ℝ))
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :] * $((scale : ℝ))
    } else {
      b_ds = b_ds
    }
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
  }
  tl.debug_barrier()
  o_q = tl.arange(0, $(BTS))
  o_k = tl.arange(0, $(BTL))
  for i in range($(i_c) * $(BTL), $(((i_c + 1) * BTL : Nat)), $(BTS)) {
    p_q = tl.make_block_ptr(base=q + $(i_bh) * $(s_k_h),
      shape=($(K), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=($(i_k) * $(BK), i), block_shape=($(BK), $(BTS)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + $(i_bh) * $(s_v_h),
      shape=($(V), $(T)), strides=($(s_v_d), $(s_v_t)),
      offsets=($(i_v) * $(BV), i), block_shape=($(BV), $(BTS)), order=(0, 1))
    p_dz = dz + $(i_bh) * $(T) + i + tl.arange(0, $(BTS))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_do = (tl.load(p_do, boundary_check=([0, 1] : List Nat))).to(b_q.dtype)
    b_dz = tl.load(p_dz, mask=((i + tl.arange(0, $(BTS))) < $(T)))
    m_s = o_k[:, None] <= o_q[None, :]
    b_s = tl.dot(b_k, b_q, allow_tf32=false) * $((scale : ℝ))
    b_s2 = b_s * b_s
    b_s = tl.where(m_s, b_s, 0.0)
    b_s2 = tl.where(m_s, b_s2, 0.0)
    b_ds = tl.dot(b_v, b_do, allow_tf32=false)
    if $(i_v) == $(0) {
      b_ds += b_dz[None, :]
    } else {
      b_ds = b_ds
    }
    b_ds = tl.where(m_s, b_ds, 0.0) * $((scale : ℝ))
    b_dv += tl.dot((b_s2).to(b_q.dtype), tl.trans(b_do), allow_tf32=false)
    b_dk += tl.dot((2.0 * b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=false)
    o_q += $(BTS)
  }
  p_dk = tl.make_block_ptr(base=dk + $(((i_bh + B * H * i_v) * s_k_h : Nat)),
    shape=($(T), $(K)), strides=($(s_k_t), $(s_k_d)),
    offsets=($(i_c) * $(BTL), $(i_k) * $(BK)), block_shape=($(BTL), $(BK)), order=(1, 0))
  p_dv = tl.make_block_ptr(base=dv + $(((i_bh + B * H * i_k) * s_v_h : Nat)),
    shape=($(T), $(V)), strides=($(s_v_t), $(s_v_d)),
    offsets=($(i_c) * $(BTL), $(i_v) * $(BV)), block_shape=($(BTL), $(BV)), order=(1, 0))
  tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  tl.store(p_dv, (b_dv).to(p_dv.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The backward dk/dv surface lowers to the algorithm layer. -/
theorem pa_bwd_dkv_surface_toAlgorithm_supported
    (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    ∃ alg, (pa_bwd_dkv_surface q k v do_ dz dk dv i_bh i_c i_k i_v i_h
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale
      B H T K V BTL BTS BK BV).toAlgorithm? = Except.ok alg := by
  simp [pa_bwd_dkv_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. `tl.debug_barrier()` erases to the no-op `Stmt.ifThen (constBool
false) []`; the `if i_v == 0` gate lowers to `Stmt.ifThenElse` on
`Op.eq` of two spliced constants; `tl.trans` is `Op.transpose`; the Python
`2 *` literal is `Op.mul … Broadcast.scalarL (Op.const 2.0)`. -/

/-- The forward non-diagonal streaming body: two block-pointer loads, the
squared score, both accumulators, and the two `tl.advance`s. -/
def paFwdLoop1Body (BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .real [BK, BTS] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTS, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q") (Op.ref .real [BK, BTS] "b_k")),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTL, BTS] "b_s")),
    Stmt.assign .real [BTL] "b_z"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BTL] "b_z")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BTL, BTS] "b_s"))),
    Stmt.assign .real [BTL, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTS, BV] "b_v"))),
    Stmt.assign .blockPtr [BK, BTS] "p_k"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BK, BTS] "p_k") [(0 : Nat), BTS]),
    Stmt.assign .blockPtr [BTS, BV] "p_v"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BTS, BV] "p_v") [BTS, (0 : Nat)]) ]

/-- The forward diagonal body: the same streaming step with the causal
`o_q[:, None] >= o_k[None, :]` mask and the `o_k += BTS` carry. -/
def paFwdLoop2Body (BTL BTS BK BV : Nat) : List Stmt :=
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
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q") (Op.ref .real [BK, BTS] "b_k")),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTL, BTS] "b_s")),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.where (Op.ref .bool [BTL, BTS] "m_s") (Op.ref .real [BTL, BTS] "b_s")
        (Op.broadcast (Op.const 0.0) [BTL, BTS])),
    Stmt.assign .real [BTL] "b_z"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BTL] "b_z")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BTL, BTS] "b_s"))),
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
/-- **Forward body split (by `rfl`).** Twenty-four top-level statements. -/
theorem pa_fwd_body_eq (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    (pa_fwd_surface q k v o z s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
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
          Stmt.assign .real [BTL] "b_z" (Op.full [BTL] (Op.const 0)),
          Stmt.forRangeDyn "_i" (Op.constNat 0)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
            (Op.constNat BTS) (paFwdLoop1Body BTL BTS BK BV),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
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
            (Op.constNat BTS) (paFwdLoop2Body BTL BTS BK BV),
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
          Stmt.assign .ptr [BTL] "p_z"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase z)
              (Op.add .nat Broadcast.scalarL
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                      (Op.mul .nat Broadcast.nil
                        (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                        (Op.ref .nat [] "i_k")))
                    (Op.constNat T))
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)))
                (Op.arange BTL))),
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
            (Op.ref .real [BTL, BV] "b_o") MaskOpt.none,
          Stmt.store .real [BTL] (MemAccess.ptr (Op.ref .ptr [BTL] "p_z"))
            (Op.ref .real [BTL] "b_z")
            (MaskOpt.mask
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
                  (Op.arange BTL))
                (Op.constNat T))) ] := by
  rfl

/-- The backward descending-loop body (respelled ascending; see the
preamble): the block index `i`, three pointer makes, three loads, the
scaled score, and the three accumulations with the `i_v == 0` gate. -/
def paBwdDescBody (q do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) : List Stmt :=
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
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
        [K, T] [BK, BTS] [s_k_d, s_k_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK),
          Op.ref .nat [] "i"]),
    Stmt.assign .blockPtr [BV, BTS] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
        [V, T] [BV, BTS] [s_v_d, s_v_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV),
          Op.ref .nat [] "i"]),
    Stmt.assign .ptr [BTS] "p_dz"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase dz)
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat T))
            (Op.ref .nat [] "i"))
          (Op.arange BTS))),
    Stmt.assign .real [BK, BTS] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BV, BTS] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTS] "b_dz"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BTS] "p_dz"))
        (MaskOpt.mask
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i") (Op.arange BTS))
            (Op.constNat T)))),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
          (Op.ref .real [BK, BTS] "b_q"))
        (Op.const scale)),
    Stmt.assign .real [BTL, BTS] "b_s2"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTL, BTS] "b_s")),
    Stmt.assign .real [BTL, BV] "b_dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s2")
          (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")))),
    Stmt.assign .real [BTL, BTS] "b_ds"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
          (Op.ref .real [BV, BTS] "b_do"))
        (Op.const scale)),
    Stmt.ifThenElse
      (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat i_v) (Op.constNat 0))
      [ Stmt.assign .real [BTL, BTS] "b_ds"
          (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BTL, BTS] "b_ds")
            (Op.mul .real Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "b_dz"))
              (Op.const scale))) ]
      [ Stmt.assign .real [BTL, BTS] "b_ds" (Op.ref .real [BTL, BTS] "b_ds") ],
    Stmt.assign .real [BTL, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := [])
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real Broadcast.scalarL (Op.const 2.0)
              (Op.ref .real [BTL, BTS] "b_ds"))
            (Op.ref .real [BTL, BTS] "b_s"))
          (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")))) ]

/-- The backward diagonal body: the same step with the `o_k[:, None] <=
o_q[None, :]` causal mask (both score copies and `b_ds` masked), the
unscaled-then-rescaled `b_ds`, and the `o_q += BTS` carry. -/
def paBwdDiagBody (q do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BTS] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
        [K, T] [BK, BTS] [s_k_d, s_k_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK),
          Op.ref .nat [] "i"]),
    Stmt.assign .blockPtr [BV, BTS] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
        [V, T] [BV, BTS] [s_v_d, s_v_t]
        [Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV),
          Op.ref .nat [] "i"]),
    Stmt.assign .ptr [BTS] "p_dz"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase dz)
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat T))
            (Op.ref .nat [] "i"))
          (Op.arange BTS))),
    Stmt.assign .real [BK, BTS] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BV, BTS] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BTS] "b_dz"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BTS] "p_dz"))
        (MaskOpt.mask
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i") (Op.arange BTS))
            (Op.constNat T)))),
    Stmt.assign .bool [BTL, BTS] "m_s"
      (Op.le ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
          (Op.ref .real [BK, BTS] "b_q"))
        (Op.const scale)),
    Stmt.assign .real [BTL, BTS] "b_s2"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTL, BTS] "b_s")),
    Stmt.assign .real [BTL, BTS] "b_s"
      (Op.where (Op.ref .bool [BTL, BTS] "m_s") (Op.ref .real [BTL, BTS] "b_s")
        (Op.broadcast (Op.const 0.0) [BTL, BTS])),
    Stmt.assign .real [BTL, BTS] "b_s2"
      (Op.where (Op.ref .bool [BTL, BTS] "m_s") (Op.ref .real [BTL, BTS] "b_s2")
        (Op.broadcast (Op.const 0.0) [BTL, BTS])),
    Stmt.assign .real [BTL, BTS] "b_ds"
      (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
        (Op.ref .real [BV, BTS] "b_do")),
    Stmt.ifThenElse
      (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat i_v) (Op.constNat 0))
      [ Stmt.assign .real [BTL, BTS] "b_ds"
          (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BTL, BTS] "b_ds")
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "b_dz"))) ]
      [ Stmt.assign .real [BTL, BTS] "b_ds" (Op.ref .real [BTL, BTS] "b_ds") ],
    Stmt.assign .real [BTL, BTS] "b_ds"
      (Op.mul .real Broadcast.scalarR
        (Op.where (Op.ref .bool [BTL, BTS] "m_s") (Op.ref .real [BTL, BTS] "b_ds")
          (Op.broadcast (Op.const 0.0) [BTL, BTS]))
        (Op.const scale)),
    Stmt.assign .real [BTL, BV] "b_dv"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s2")
          (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")))),
    Stmt.assign .real [BTL, BK] "b_dk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := [])
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real Broadcast.scalarL (Op.const 2.0)
              (Op.ref .real [BTL, BTS] "b_ds"))
            (Op.ref .real [BTL, BTS] "b_s"))
          (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")))),
    Stmt.assign .nat [BTS] "o_q"
      (Op.add .nat Broadcast.scalarR (Op.ref .nat [BTS] "o_q") (Op.constNat BTS)) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Fifteen top-level statements. -/
theorem pa_bwd_body_eq (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) :
    (pa_bwd_dkv_surface q k v do_ dz dk dv i_bh i_c i_k i_v i_h
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale
        B H T K V BTL BTS BK BV).toAlgKernel.body
      = [ Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
              [T, K] [BTL, BK] [s_k_t, s_k_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
              [T, V] [BTL, BV] [s_v_t, s_v_d]
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
            (paBwdDescBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
              s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL),
          Stmt.forRangeDyn "i"
            (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
            (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
            (paBwdDiagBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
              s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV),
          Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_k_h))
              [T, K] [BTL, BK] [s_k_t, s_k_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_v_h))
              [T, V] [BTL, BV] [s_v_t, s_v_d]
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
for arbitrary ragged tails. -/

/-- `NV = tl.cdiv(V, BV)` as the prologue computes it. -/
def paNV (V BV : Nat) : Nat := (V + BV - 1) / BV

/-- `i_k = i_kv // NV`. -/
def paIk (s : BlockState) (V BV : Nat) : Nat := s.pids 0 / paNV V BV

/-- `i_v = i_kv % NV`. -/
def paIv (s : BlockState) (V BV : Nat) : Nat := s.pids 0 % paNV V BV

/-- The scaled, guarded `b_q` lane `(a, e)`: row `i_c·BTL + a` of the `(T, K)`
parent, column `i_k·BK + e`, times `scale`; `0` outside the window. -/
noncomputable def paQGuarded (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a e : Nat) : ℝ :=
  if s.pids 1 * BTL + a < T ∧ paIk s V BV * BK + e < K then
    s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + a) * s_qk_t
      + (paIk s V BV * BK + e) * s_qk_d) * scale
  else 0

/-- The guarded transposed `k` lane `(e, t)` at absolute key `t` (the `(K, T)`
parent read with strides `(s_qk_d, s_qk_t)`). -/
noncomputable def paKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV : Nat) (e t : Nat) : ℝ :=
  if paIk s V BV * BK + e < K ∧ t < T then
    s.readMem k (s.pids 2 * s_qk_h + (paIk s V BV * BK + e) * s_qk_d
      + t * s_qk_t)
  else 0

/-- The guarded `v` lane `(t, p)` at absolute key `t`. -/
noncomputable def paVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV : Nat) (t p : Nat) : ℝ :=
  if t < T ∧ paIv s V BV * BV + p < V then
    s.readMem v (s.pids 2 * s_vo_h + t * s_vo_t
      + (paIv s V BV * BV + p) * s_vo_d)
  else 0

/-- The rebased score at `(row a, key t)`: `(scale·q) · k` over the `BK`
head window. -/
noncomputable def paScore (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a t : Nat) : ℝ :=
  ∑ e : Fin BK,
    paQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a e.val
      * paKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV e.val t

/-- The unmasked accumulator after the non-diagonal loop has consumed keys
`[0, n)`: `Σ_t score² · v`. -/
noncomputable def paOAcc (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (n a p : Nat) : ℝ :=
  ∑ t ∈ Finset.range n,
    paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p

/-- The unmasked normalizer after keys `[0, n)`: `Σ_t score²`. -/
noncomputable def paZAcc (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (n a : Nat) : ℝ :=
  ∑ t ∈ Finset.range n,
    paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
      * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t

/-- The diagonal (causally masked) accumulator over keys
`[i_c·BTL, i)`: kept iff `t ≤ i_c·BTL + a`. -/
noncomputable def paODiag (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i a p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico (s.pids 1 * BTL) i,
    if t ≤ s.pids 1 * BTL + a then
      paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV t p
    else 0

/-- The diagonal normalizer over keys `[i_c·BTL, i)`. -/
noncomputable def paZDiag (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (i a : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico (s.pids 1 * BTL) i,
    if t ≤ s.pids 1 * BTL + a then
      paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
        * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a t
    else 0

/-- **The stored `o` lane** — non-diagonal sum plus the masked diagonal
block: causal rebased attention `Σ_{t ≤ i_c·BTL + a} score² · v`. -/
noncomputable def paOOut (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (a p : Nat) : ℝ :=
  paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BK BV (s.pids 1 * BTL) a p
    + paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BTL BK BV ((s.pids 1 + 1) * BTL) a p

/-- **The stored `z` lane** — the matching normalizer. -/
noncomputable def paZOut (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (a : Nat) : ℝ :=
  paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      (s.pids 1 * BTL) a
    + paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        ((s.pids 1 + 1) * BTL) a

/-- The `o` store base `(i_bh + B·H·i_k) · s_vo_h`. -/
def paOBase (s : BlockState) (B H s_vo_h V BV : Nat) : Nat :=
  (s.pids 2 + B * H * paIk s V BV) * s_vo_h

/-- The `o` store address at lane `(a, p)` (strides `(s_vo_t, 1)` after the
`s_vo_d = 1` substitution). -/
def paOOffset (s : BlockState) (B H s_vo_h s_vo_t s_vo_d V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  paOBase s B H s_vo_h V BV + (s.pids 1 * BTL + idx.1.val) * s_vo_t
    + (paIv s V BV * BV + idx.2.1.val) * s_vo_d

/-- An `o` store lane is *active* when it maps inside the `T × V` window. -/
def paOActive (s : BlockState) (T V BV BTL : Nat)
    (idx : TileIndex [BTL, BV]) : Prop :=
  s.pids 1 * BTL + idx.1.val < T ∧ paIv s V BV * BV + idx.2.1.val < V

/-- The `z` store address at lane `a`. -/
def paZOffset (s : BlockState) (B H T V BV BTL : Nat) (a : Fin BTL) : Nat :=
  (s.pids 2 + B * H * paIk s V BV) * T + s.pids 1 * BTL + a.val

/-! ## Closed-form specification (backward dk/dv)

The helper's scalar arguments are plain `Nat` binders (no pids). -/

/-- The top of the streamed key range: `cdiv(T, BTS)·BTS`. -/
def pbNB (T BTS : Nat) : Nat := (T + BTS - 1) / BTS * BTS

/-- The respelled descending trip count. -/
def pbTrip (T BTS i_c BTL : Nat) : Nat :=
  ((T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL + BTS - 1) / BTS

/-- The guarded `b_k` lane `(l, e)` of the fixed chunk block. -/
noncomputable def pbKGuarded (s : BlockState) (k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat)
    (l e : Nat) : ℝ :=
  if i_c * BTL + l < T ∧ i_k * BK + e < K then
    s.readMem k (i_bh * s_k_h + (i_c * BTL + l) * s_k_t
      + (i_k * BK + e) * s_k_d)
  else 0

/-- The guarded `b_v` lane `(l, p)` of the fixed chunk block. -/
noncomputable def pbVGuarded (s : BlockState) (v : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat)
    (l p : Nat) : ℝ :=
  if i_c * BTL + l < T ∧ i_v * BV + p < V then
    s.readMem v (i_bh * s_v_h + (i_c * BTL + l) * s_v_t
      + (i_v * BV + p) * s_v_d)
  else 0

/-- The guarded transposed `q` lane `(e, t)` at absolute key `t`. -/
noncomputable def pbQGuarded (s : BlockState) (q : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_k : Nat) (T K BK : Nat)
    (e t : Nat) : ℝ :=
  if i_k * BK + e < K ∧ t < T then
    s.readMem q (i_bh * s_k_h + (i_k * BK + e) * s_k_d + t * s_k_t)
  else 0

/-- The guarded transposed `do` lane `(p, t)` at absolute key `t`. -/
noncomputable def pbDoGuarded (s : BlockState) (do_ : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_v : Nat) (T V BV : Nat)
    (p t : Nat) : ℝ :=
  if i_v * BV + p < V ∧ t < T then
    s.readMem do_ (i_bh * s_v_h + (i_v * BV + p) * s_v_d + t * s_v_t)
  else 0

/-- The mask-guarded `dz` lane at absolute key `t` (masked-off lanes read the
clean-input `undef ≡ 0` oracle). -/
noncomputable def pbDzGuarded (s : BlockState) (dz : RegionName)
    (i_bh T : Nat) (t : Nat) : ℝ :=
  if t < T then s.readMem dz (i_bh * T + t) else 0

/-- The scaled backward score `b_s[l, t] = (b_k · b_q)·scale`. -/
noncomputable def pbSVal (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (scale : ℝ)
    (T K BTL BK : Nat) (l t : Nat) : ℝ :=
  (∑ e : Fin BK,
      pbKGuarded s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK l e.val
        * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e.val t) * scale

/-- The scaled backward dscore `b_ds[l, t] = (b_v · b_do + [i_v = 0]·dz)·scale`. -/
noncomputable def pbDsVal (s : BlockState) (v do_ dz : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (scale : ℝ)
    (T V BTL BV : Nat) (l t : Nat) : ℝ :=
  ((∑ p : Fin BV,
      pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
        * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p.val t)
    + (if i_v = 0 then pbDzGuarded s dz i_bh T t else 0)) * scale

/-- The `b_dv` accumulator over keys `[lo, hi)` with the causal keep
`i_c·BTL + l ≤ t` (identically true above the diagonal chunk). -/
noncomputable def pbDvPart (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (lo hi : Nat) (l p : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + l ≤ t then
      pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p t
    else 0

/-- The `b_dk` accumulator over keys `[lo, hi)` with the causal keep. -/
noncomputable def pbDkPart (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (lo hi : Nat) (l e : Nat) : ℝ :=
  ∑ t ∈ Finset.Ico lo hi,
    if i_c * BTL + l ≤ t then
      2 * pbDsVal s v do_ dz s_v_h s_v_t s_v_d i_bh i_c i_v scale T V BTL BV l t
        * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK l t
        * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e t
    else 0

/-- **The stored `dv` lane**: the key sweep from `i_c·BTL` up to the larger of
the streamed top `cdiv(T,BTS)·BTS` and the diagonal end `(i_c+1)·BTL` (the
kernel loads beyond `T` read as zero, so the tail summands vanish). -/
noncomputable def pbDvOut (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (l p : Nat) : ℝ :=
  pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
    scale T K V BTL BTS BK BV (i_c * BTL) (max (pbNB T BTS) ((i_c + 1) * BTL)) l p

/-- **The stored `dk` lane**: the same key sweep (up to the larger of the
streamed top and the diagonal end; the kernel loads beyond `T` read as zero)
with the gate term. -/
noncomputable def pbDkOut (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (l e : Nat) : ℝ :=
  pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
    scale T K V BTL BTS BK BV (i_c * BTL) (max (pbNB T BTS) ((i_c + 1) * BTL)) l e

/-! ## Eval recipes (local copies from the `chunk_retention` pair, plus the
`parallel_attention`-specific ops: `//`/`%` on refs, `tl.cdiv` on constants,
`>=`/`<=` masks, `tl.where`, `tl.sum(axis=1)`, block-pointer advances, and
the plain-pointer `p_z`). -/

private theorem pa_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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

private theorem pa_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
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

private theorem pa_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `NV = tl.cdiv(V, BV)` on spliced constants. -/
private theorem pa_NV_eval (s : BlockState) (V BV : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat V) (Op.constNat BV))
          (Op.constNat 1))
        (Op.constNat BV)) s
      = some (Tile.scalar (paNV V BV)) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

/-- `i_k = i_kv // NV` on refs. -/
private theorem pa_floorDiv_refs_eval (s : BlockState) (na nb : RegName)
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
private theorem pa_mod_refs_eval (s : BlockState) (na nb : RegName)
    (va vb : Nat)
    (ha : s.regs .nat [] na = some (Tile.scalar va))
    (hb : s.regs .nat [] nb = some (Tile.scalar vb)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] na)
        (Op.ref .nat [] nb)) s
      = some (Tile.scalar (va % vb)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, ha, hb, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem pa_zeros_eval (sh : TileShape) (t : BlockState) :
    evalOp (Op.full sh (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real sh) := by
  simp [evalOp_full, evalOp_const]

private theorem pa_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

private theorem pa_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
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

private theorem pa_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

private theorem pa_addTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add .real bc x y) t
      = some (Tile.bop NumericDType.real.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem pa_mulTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul .real bc x y) t
      = some (Tile.bop NumericDType.real.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- Advance a 2-D block pointer along the column axis. -/
private theorem pa_advance_col_eval (s : BlockState) (region : RegionName)
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
private theorem pa_advance_row_eval (s : BlockState) (region : RegionName)
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

/-! ## Block store machinery (same-region masked frame; the strides-`[σ, 1]`
form applies after the `s_vo_d = 1` / `s_k_d = 1` / `s_v_d = 1`
substitutions). -/

/-- One boundary-checked `[BR, BS]` block store's lane address. -/
def paStoreAddr (base σ rowOff colOff : Nat) (BR BS : Nat)
    (i : TileIndex [BR, BS]) : Nat :=
  base + (rowOff + i.1.val) * σ + (colOff + i.2.1.val) * 1

/-- The post-store state: every in-window lane written. -/
noncomputable def paStoreState (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BR, BS]).foldl
    (fun acc i => if (rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
        then acc.writeMem rg (paStoreAddr base σ rowOff colOff BR BS i)
          (f i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Block store step (eq).** -/
theorem paStore_step_eq (sin : BlockState) (rg : RegionName)
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
      = some (paStoreState sin rg base σ rowOff colOff K V BR BS f) := by
  unfold stepStmt paStoreState
  simp only [evalOp_ref, hb, hp]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [BR, BS])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, paStoreAddr]
  by_cases hbnd : rowOff + e.val < K ∧ colOff + p.val < V
  · simp only [hbnd, BlockState.writeMemTyped_real, hbf, Nat.mul_one]
    rfl
  · simp only [hbnd, decide_false, Bool.false_eq_true, if_false]

private theorem pa_block_index_inj {Q j c A B : Nat} (hA : A < Q) (hB : B < Q)
    (heq : j * Q + A = c * Q + B) : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A) hA
  have hj : (j * Q + A) / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B) / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- One block's store lanes are pairwise distinct once a `BS` segment fits
under the row stride. -/
theorem paStoreAddr_injective (base σ rowOff colOff BR BS : Nat)
    (hBSσ : BS ≤ σ) :
    Function.Injective (paStoreAddr base σ rowOff colOff BR BS) := by
  rintro ⟨e, p, u⟩ ⟨e', p', u'⟩ heq
  simp only [paStoreAddr] at heq
  have hp := p.isLt
  have hp' := p'.isLt
  have h2 : (rowOff + e.val) * σ + p.val = (rowOff + e'.val) * σ + p'.val := by
    omega
  have hlt : p.val < σ := by omega
  have hlt' : p'.val < σ := by omega
  have hjj : rowOff + e.val = rowOff + e'.val :=
    pa_block_index_inj hlt hlt' h2
  have he : e = e' := Fin.ext (by omega)
  have hpv : p = p' := Fin.ext (by
    have hσ : (rowOff + e.val) * σ = (rowOff + e'.val) * σ := by rw [hjj]
    omega)
  subst he
  subst hpv
  rfl

set_option maxHeartbeats 4000000 in
/-- **Block store readback** (mask-restricted same-region frame). -/
theorem paStore_step_props (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ)
    (hInj : Function.Injective (paStoreAddr base σ rowOff colOff BR BS)) :
    (paStoreState sin rg base σ rowOff colOff K V BR BS f).pids = sin.pids
      ∧ (paStoreState sin rg base σ rowOff colOff K V BR BS f).regs = sin.regs
      ∧ (∀ idx : TileIndex [BR, BS],
          (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
          (paStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg
              (paStoreAddr base σ rowOff colOff BR BS idx)
            = f idx.1.val idx.2.1.val)
      ∧ (∀ rg' off, rg' ≠ rg →
          (paStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg' off
            = sin.readMem rg' off)
      ∧ (∀ off, (∀ idx : TileIndex [BR, BS],
            (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
            off ≠ paStoreAddr base σ rowOff colOff BR BS idx) →
          (paStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold paStoreState
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name
    rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
      (paStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      hInj idx
    rw [h, if_pos ⟨h1, h2⟩]
  · intro rg' off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      rg (paStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin rg' off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      rg (paStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin off (fun i _ hPi => hoff i hPi)

/-! ## Value tiles and load bridges (forward) -/

private theorem pa_readMem_congr (s s0 : BlockState) (h : s.mem = s0.mem)
    (rg : RegionName) (off : Nat) : s.readMem rg off = s0.readMem rg off := by
  unfold BlockState.readMem
  rw [h]

/-- The scaled loaded `b_q` tile. -/
noncomputable def paQTile (s : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat) :
    Tile .real [BTL, BK] :=
  ⟨fun idx => some (paQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
    idx.1.val idx.2.1.val)⟩

/-- The loaded `b_k` block at key-column offset `i`: lane `(e, s') ↦ k[e, i + s']`. -/
noncomputable def paKTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV BTS : Nat) (i : Nat) :
    Tile .real [BK, BTS] :=
  ⟨fun idx => some (paKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV
    idx.1.val (i + idx.2.1.val))⟩

/-- The loaded `b_v` block at key-row offset `i`. -/
noncomputable def paVTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV BTS : Nat) (i : Nat) :
    Tile .real [BTS, BV] :=
  ⟨fun idx => some (paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV
    (i + idx.1.val) idx.2.1.val)⟩

/-- Loading `p_k` at offsets `[i_k·BK, i]` lands on `paKTile … i`. -/
private theorem pa_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K V BK BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem) (hpids : sin.pids = s.pids)
    (hreg : sin.regs .blockPtr [BK, BTS] name = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [paIk s V BV * BK, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (paKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i) := by
  rw [pa_load_bp_2d k sin name (s.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
    (paIk s V BV * BK) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, c, u⟩ := idx
  simp only [paKTile, paKGuarded]
  by_cases h : paIk s V BV * BK + e.val < K ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_v` at offsets `[i, i_v·BV]` lands on `paVTile … i`. -/
private theorem pa_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTS, BV] name = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, paIv s V BV * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTS, BV] name) [0, 1])
        MaskOpt.none) sin
      = some (paVTile s v s_vo_h s_vo_t s_vo_d T V BV BTS i) := by
  rw [pa_load_bp_2d v sin name (s.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
    i (paIv s V BV * BV) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨c, p, u⟩ := idx
  simp only [paVTile, paVGuarded]
  by_cases h : i + c.val < T ∧ paIv s V BV * BV + p.val < V
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-! ## Forward per-statement recipes -/

/-- `b_s = tl.dot(b_q, b_k)` lands lanewise on `paScore · (i + s')`. -/
private theorem pa_score_eval (s sin : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV BTS : Nat)
    (i : Nat)
    (hq : sin.regs .real [BTL, BK] "b_q"
      = some (paQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV))
    (hk : sin.regs .real [BK, BTS] "b_k"
      = some (paKTile s k s_qk_h s_qk_t s_qk_d T K V BK BV BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_q")
        (Op.ref .real [BK, BTS] "b_k")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
            idx.1.val (i + idx.2.1.val))⟩ := by
  rw [pa_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hq)
    (by rw [evalOp_ref]; exact hk)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rw [pa_dot2d_elem _ _ a c
    (fun e => paQGuarded s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      a.val e.val)
    (fun e => paKGuarded s k s_qk_h s_qk_t s_qk_d T K V BK BV
      e.val (i + c.val))
    (fun e => rfl) (fun e => rfl)]
  rfl

/-- `b_s = b_s * b_s` on an all-`some` tile squares lanewise. -/
private theorem pa_square_eval (sin : BlockState) (BTL BTS : Nat)
    (f : Nat → Nat → ℝ)
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BTS] "b_s") (Op.ref .real [BTL, BTS] "b_s")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (f idx.1.val idx.2.1.val * f idx.1.val idx.2.1.val)⟩ := by
  rw [pa_mulTile_eval _ _ _ sin _ _ (by rw [evalOp_ref]; exact hs)
    (by rw [evalOp_ref]; exact hs)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rfl

/-- `b_z += tl.sum(b_s, axis=1)` on all-`some` tiles: lanewise row sums. -/
private theorem pa_zAdd_eval (sin : BlockState) (BTL BTS : Nat)
    (g : Nat → ℝ) (f : Nat → Nat → ℝ)
    (hz : sin.regs .real [BTL] "b_z" = some ⟨fun a => some (g a.1.val)⟩)
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BTL] "b_z")
        (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BTL, BTS] "b_s"))) sin
      = some ⟨fun a : TileIndex [BTL] =>
          some (g a.1.val + ∑ c : Fin BTS, f a.1.val c.val)⟩ := by
  simp only [evalOp.eq_def, evalOp_ref, hz, hs, Option.bind, Option.map]
  refine congrArg some ?_
  ext a
  simp only [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  simp only [TileShape.insertAxisIndex, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.dropInsertedIndex, WithBot.some_eq_coe, Option.map₂, Option.bind,
    Option.map]
  rw [← WithBot.coe_sum]
  rfl

/-- `b_o (+)= tl.dot(b_s, b_v)` on all-`some` tiles: lanewise contraction. -/
private theorem pa_oAdd_eval (sin : BlockState) (BTL BTS BV : Nat)
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
    pa_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hs)
      (by rw [evalOp_ref]; exact hv)
  erw [pa_addTile_eval _ _ _ sin _ _ (by rw [evalOp_ref]; exact ho) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [pa_dot2d_elem _ _ a p (fun c => f a.val c.val) (fun c => w c.val p.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

private theorem pa_evalOp_ge_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

private theorem pa_evalOp_le_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.le h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.le bc vx vy)) := by
  simp [evalOp]

/-- The `o_q[:, None] >= o_k[None, :]` mask at diagonal offset `off`. -/
def paMaskTile (BTL BTS off : Nat) : Tile .bool [BTL, BTS] :=
  ⟨fun idx => decide (idx.2.1.val + off ≤ idx.1.val)⟩

private theorem pa_msGe_eval (sin : BlockState) (BTL BTS off : Nat)
    (hoq : sin.regs .nat [BTL] "o_q" = some (Tile.vec fun r => r.val))
    (hok : sin.regs .nat [BTS] "o_k" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_q"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_k"))) sin
      = some (paMaskTile BTL BTS off) := by
  rw [pa_evalOp_ge_def]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoq,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hok]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  show decide (a.val ≥ c.val + off) = decide (c.val + off ≤ a.val)
  rfl

/-- `b_s = tl.where(m_s, b_s, 0.0)` on all-`some` tiles: the masked lane. -/
private theorem pa_whereMask_eval (sin : BlockState) (BTL BTS off : Nat)
    (f : Nat → Nat → ℝ)
    (hm : sin.regs .bool [BTL, BTS] "m_s" = some (paMaskTile BTL BTS off))
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.where (Op.ref .bool [BTL, BTS] "m_s")
        (Op.ref .real [BTL, BTS] "b_s")
        (Op.broadcast (Op.const 0.0) [BTL, BTS])) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (if idx.2.1.val + off ≤ idx.1.val
            then f idx.1.val idx.2.1.val else 0)⟩ := by
  rw [evalOp_where]
  simp only [evalOp.eq_def, evalOp_ref, evalOp_const, hm, hs,
    Option.bind, Option.map, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rw [Tile.select_data]
  simp only [paMaskTile]
  by_cases h : c.val + off ≤ a.val
  · rw [if_pos (by simpa using h), if_pos h]
  · rw [if_neg (by simpa using h), if_neg h]
    norm_num

/-- `o_k += BTS` shifts the arange window. -/
private theorem pa_okAdd_eval (sin : BlockState) (BTS off : Nat)
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

/-! ## Block-sum split lemmas for the specs -/

private theorem pa_sum_range_add_block (n BTS : Nat) (f : Nat → ℝ) :
    ∑ t ∈ Finset.range (n + BTS), f t
      = (∑ t ∈ Finset.range n, f t) + ∑ c : Fin BTS, f (n + c.val) := by
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun c => f (n + c))]

private theorem pa_sum_Ico_add_block (lo i BTS : Nat) (hlo : lo ≤ i)
    (f : Nat → ℝ) :
    ∑ t ∈ Finset.Ico lo (i + BTS), f t
      = (∑ t ∈ Finset.Ico lo i, f t) + ∑ c : Fin BTS, f (i + c.val) := by
  rw [← Finset.sum_Ico_consecutive _ hlo (Nat.le_add_right i BTS)]
  congr 1
  rw [Finset.sum_Ico_eq_sum_range]
  simp only [Nat.add_sub_cancel_left]
  rw [Fin.sum_univ_eq_sum_range (fun c => f (i + c))]

/-! ## Spec step lemmas -/

private theorem pa_dvd_step_le {BTS i X : Nat} (hd : BTS ∣ i) (hX : BTS ∣ X)
    (h : i < X) : i + BTS ≤ X := by
  obtain ⟨a, rfl⟩ := hd
  obtain ⟨b, rfl⟩ := hX
  have hab : a < b := by
    by_contra hab
    push_neg at hab
    exact absurd (Nat.mul_le_mul_left BTS hab) (Nat.not_le.mpr h)
  calc BTS * a + BTS = BTS * (a + 1) := by ring
    _ ≤ BTS * b := Nat.mul_le_mul_left BTS hab

private theorem paOAcc_step (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a p : Nat) :
    paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BTL BK BV i a p
      + ∑ c : Fin BTS,
          paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
            * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
            * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
      = paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BK BV (i + BTS) a p := by
  unfold paOAcc
  rw [pa_sum_range_add_block]

private theorem paZAcc_step (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a : Nat) :
    paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a
      + ∑ c : Fin BTS,
          paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
            * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
      = paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV (i + BTS) a := by
  unfold paZAcc
  rw [pa_sum_range_add_block]

private theorem paODiag_step (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a p : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BTL BK BV i a p
      + ∑ c : Fin BTS,
          (if c.val + (i - s.pids 1 * BTL) ≤ a then
            paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
          else 0)
      = paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BK BV (i + BTS) a p := by
  unfold paODiag
  rw [pa_sum_Ico_add_block _ _ _ hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (c.val + (i - s.pids 1 * BTL) ≤ a) ↔ (i + c.val ≤ s.pids 1 * BTL + a) := by
    omega
  by_cases h : i + c.val ≤ s.pids 1 * BTL + a
  · rw [if_pos h, if_pos (hcond.mpr h)]
  · rw [if_neg h, if_neg (fun hc => h (hcond.mp hc))]

private theorem paZDiag_step (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a
      + ∑ c : Fin BTS,
          (if c.val + (i - s.pids 1 * BTL) ≤ a then
            paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
          else 0)
      = paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV (i + BTS) a := by
  unfold paZDiag
  rw [pa_sum_Ico_add_block _ _ _ hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (c.val + (i - s.pids 1 * BTL) ≤ a) ↔ (i + c.val ≤ s.pids 1 * BTL + a) := by
    omega
  by_cases h : i + c.val ≤ s.pids 1 * BTL + a
  · rw [if_pos h, if_pos (hcond.mpr h)]
  · rw [if_neg h, if_neg (fun hc => h (hcond.mp hc))]

/-- `b_q = (b_q * scale)` after the boundary-checked load: the scaled guarded
tile. -/
private theorem pa_qScale_eval (s sin : BlockState) (q : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ) (T K V BTL BK BV : Nat)
    (hq : sin.regs .real [BTL, BK] "b_q" = some
      ⟨fun idx : TileIndex [BTL, BK] =>
        if (s.pids 1 * BTL + idx.1.val < T ∧ paIk s V BV * BK + idx.2.1.val < K) then
          some (s.readMem q (s.pids 2 * s_qk_h + (s.pids 1 * BTL + idx.1.val) * s_qk_t
            + (paIk s V BV * BK + idx.2.1.val) * s_qk_d))
        else some 0⟩) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BTL, BK] "b_q")
        (Op.const scale)) sin
      = some (paQTile s q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) := by
  rw [pa_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (by rw [evalOp_ref]; exact hq) (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    paQTile, paQGuarded]
  by_cases h : s.pids 1 * BTL + a.val < T ∧ paIk s V BV * BK + e.val < K
  · rw [if_pos h, if_pos h]
    rfl
  · rw [if_neg h, if_neg h]
    show Option.map₂ _ (some (0 : ℝ)) (some scale) = some 0
    show some ((0 : ℝ) * scale) = some 0
    rw [zero_mul]

/-! ## The forward non-diagonal loop -/

/-- The forward loop-1 invariant at counter `i` (the absolute key offset). -/
def paInv1 (s0 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧
  BTS ∣ i ∧ i ≤ s0.pids 1 * BTL ∧
  s.regs .nat [] "i_c" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2)) ∧
  s.regs .nat [] "i_k" = some (Tile.scalar (paIk s0 V BV)) ∧
  s.regs .nat [] "i_v" = some (Tile.scalar (paIv s0 V BV)) ∧
  s.regs .real [BTL, BK] "b_q"
    = some (paQTile s0 q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) ∧
  s.regs .real [BTL, BV] "b_o"
    = some ⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale T K V BTL BK BV i idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL] "b_z"
    = some ⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
        T K V BTL BK BV i a.1.val)⟩ ∧
  s.regs .blockPtr [BK, BTS] "p_k"
    = some ⟨fun _ => BlockPtr.mk k (s0.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [paIk s0 V BV * BK, i]⟩ ∧
  s.regs .blockPtr [BTS, BV] "p_v"
    = some ⟨fun _ => BlockPtr.mk v (s0.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, paIv s0 V BV * BV]⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One non-diagonal block.** -/
theorem paFwdLoop1_step (s0 sin : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat)
    (hBTS : BTL % BTS = 0)
    (hlt : i < s0.pids 1 * BTL)
    (hInv : paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (paFwdLoop1Body BTL BTS BK BV)
        (sin.setReg "_i" .nat [] (Tile.scalar i)) = some s'
      ∧ paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hpids, hmem, hdvd, hle, hic, hibh, hik, hiv, hbq, hbo, hbz, hpk, hpv⟩ := hInv
  unfold paFwdLoop1Body
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_kLoad_eq s0 _ k "p_k" s_qk_h s_qk_t s_qk_d T K V BK BV BTS i
      (by simpa using hmem) (by simpa using hpids) (by simpa using hpk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_vLoad_eq s0 _ v "p_v" s_vo_h s_vo_t s_vo_d T V BV BTS i
      (by simpa using hmem) (by simpa using hpv)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_score_eval s0 _ q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV BTS i
      (by simpa using hbq) (by simp [paKTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_square_eval _ BTL BTS
      (fun a c => paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        a (i + c))
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_zAdd_eval _ BTL BTS
      (fun a => paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a)
      (fun a c => paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          a (i + c)
        * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c))
      (by simpa using hbz) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_oAdd_eval _ BTL BTS BV
      (fun a p => paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        scale T K V BTL BK BV i a p)
      (fun a c => paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          a (i + c)
        * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c))
      (fun c p => paVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c) p)
      (by simpa using hbo) (by simp) (by simp [paVTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_advance_col_eval _ k (s0.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
      (paIk s0 V BV * BK) i BTS "p_k" (by simp [hpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_advance_row_eval _ v (s0.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
      i (paIv s0 V BV * BV) BTS "p_v" (by simp [hpv])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · exact Dvd.dvd.add hdvd dvd_rfl
  · exact pa_dvd_step_le hdvd
      (Dvd.dvd.mul_left (Nat.dvd_of_mod_eq_zero hBTS) (s0.pids 1)) hlt
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbq
  · rw [show (⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BK BV (i + BTS)
          idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale T K V BTL BK BV i idx.1.val idx.2.1.val
          + ∑ c : Fin BTS,
              paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                  idx.1.val (i + c.val)
                * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                    idx.1.val (i + c.val)
                * paVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c.val)
                    idx.2.1.val)⟩
      from Tile.ext fun idx =>
        congrArg some (paOAcc_step s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d scale T K V BTL BK BV i BTS idx.1.val idx.2.1.val).symm]
    simp
  · rw [show (⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV (i + BTS) a.1.val)⟩ : Tile .real [BTL])
        = ⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
            T K V BTL BK BV i a.1.val
          + ∑ c : Fin BTS,
              paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                  a.1.val (i + c.val)
                * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                    a.1.val (i + c.val))⟩
      from Tile.ext fun a =>
        congrArg some (paZAcc_step s0 q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV i BTS a.1.val).symm]
    simp
  · simp
  · simp

set_option maxHeartbeats 4000000 in
/-- **The full non-diagonal loop**: from the prologue state to keys
`[0, i_c·BTL)` consumed. -/
theorem paFwdLoop1_run (s0 sPre : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BTS BK BV 0 sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "_i" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.constNat BTS) (paFwdLoop1Body BTL BTS BK BV)) sPre = some sL
      ∧ paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV (s0.pids 1 * BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "_i")
      (P := fun i s => paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale T K V BTL BTS BK BV i s)
      (evalOp_constNat 0 sPre)
      (pa_mulConst_eval sPre "i_c" (s0.pids 1) BTL hInv.2.2.2.2.1)
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        paFwdLoop1_step s0 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale T K V BTL BTS BK BV i hBTS hi hPi)
  have hfin : final = s0.pids 1 * BTL := le_antisymm hP.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-! ## The forward diagonal loop -/

/-- The forward loop-2 invariant at counter `i` (absolute key offset within
the diagonal chunk). -/
def paInv2 (s0 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ s.mem = s0.mem ∧
  s0.pids 1 * BTL ≤ i ∧ BTS ∣ (i - s0.pids 1 * BTL) ∧
  i ≤ (s0.pids 1 + 1) * BTL ∧
  s.regs .nat [] "i_c" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2)) ∧
  s.regs .nat [] "i_k" = some (Tile.scalar (paIk s0 V BV)) ∧
  s.regs .nat [] "i_v" = some (Tile.scalar (paIv s0 V BV)) ∧
  s.regs .real [BTL, BK] "b_q"
    = some (paQTile s0 q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV) ∧
  s.regs .real [BTL, BV] "b_o"
    = some ⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale T K V BTL BK BV (s0.pids 1 * BTL) idx.1.val idx.2.1.val
      + paODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BK BV i idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL] "b_z"
    = some ⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
        T K V BTL BK BV (s0.pids 1 * BTL) a.1.val
      + paZDiag s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a.1.val)⟩ ∧
  s.regs .nat [BTL] "o_q" = some (Tile.vec fun r => r.val) ∧
  s.regs .nat [BTS] "o_k"
    = some (Tile.vec fun r => r.val + (i - s0.pids 1 * BTL)) ∧
  s.regs .blockPtr [BK, BTS] "p_k"
    = some ⟨fun _ => BlockPtr.mk k (s0.pids 2 * s_qk_h) [K, T] [BK, BTS]
        [s_qk_d, s_qk_t] [paIk s0 V BV * BK, i]⟩ ∧
  s.regs .blockPtr [BTS, BV] "p_v"
    = some ⟨fun _ => BlockPtr.mk v (s0.pids 2 * s_vo_h) [T, V] [BTS, BV]
        [s_vo_t, s_vo_d] [i, paIv s0 V BV * BV]⟩

private theorem paODiag_step' (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a p : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    (paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BTL BK BV (s.pids 1 * BTL) a p
      + paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BK BV i a p)
      + ∑ c : Fin BTS,
          (if c.val + (i - s.pids 1 * BTL) ≤ a then
            paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
          else 0)
          * paVGuarded s v s_vo_h s_vo_t s_vo_d T V BV (i + c.val) p
      = paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BK BV (s.pids 1 * BTL) a p
        + paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BTL BK BV (i + BTS) a p := by
  rw [add_assoc]
  congr 1
  rw [← paODiag_step s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
    T K V BTL BK BV i BTS a p hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  by_cases h : c.val + (i - s.pids 1 * BTL) ≤ a
  · rw [if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, zero_mul]

private theorem paZDiag_step' (s : BlockState) (q k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (scale : ℝ)
    (T K V BTL BK BV : Nat) (i BTS a : Nat) (hlo : s.pids 1 * BTL ≤ i) :
    (paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        (s.pids 1 * BTL) a
      + paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a)
      + ∑ c : Fin BTS,
          (if c.val + (i - s.pids 1 * BTL) ≤ a then
            paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
              * paScore s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c.val)
          else 0)
      = paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          (s.pids 1 * BTL) a
        + paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV (i + BTS) a := by
  rw [add_assoc]
  congr 1
  exact paZDiag_step s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
    i BTS a hlo

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One diagonal block** (causally masked). -/
theorem paFwdLoop2_step (s0 sin : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) (i : Nat)
    (hBTS : BTL % BTS = 0)
    (hlt : i < (s0.pids 1 + 1) * BTL)
    (hInv : paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (paFwdLoop2Body BTL BTS BK BV)
        (sin.setReg "_i" .nat [] (Tile.scalar i)) = some s'
      ∧ paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hpids, hmem, hlo, hdvd, hle, hic, hibh, hik, hiv, hbq, hbo, hbz,
    hoq, hok, hpk, hpv⟩ := hInv
  unfold paFwdLoop2Body
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_kLoad_eq s0 _ k "p_k" s_qk_h s_qk_t s_qk_d T K V BK BV BTS i
      (by simpa using hmem) (by simpa using hpids) (by simpa using hpk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_vLoad_eq s0 _ v "p_v" s_vo_h s_vo_t s_vo_d T V BV BTS i
      (by simpa using hmem) (by simpa using hpv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_msGe_eval _ BTL BTS (i - s0.pids 1 * BTL)
      (by simpa using hoq) (by simpa using hok)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_score_eval s0 _ q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV BTS i
      (by simpa using hbq) (by simp [paKTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_square_eval _ BTL BTS
      (fun a c => paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        a (i + c))
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_whereMask_eval _ BTL BTS (i - s0.pids 1 * BTL)
      (fun a c => paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          a (i + c)
        * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_zAdd_eval _ BTL BTS
      (fun a => paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          (s0.pids 1 * BTL) a
        + paZDiag s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a)
      (fun a c => if c + (i - s0.pids 1 * BTL) ≤ a then
          paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c)
            * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c)
        else 0)
      (by simpa using hbz) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_oAdd_eval _ BTL BTS BV
      (fun a p => paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale T K V BTL BK BV (s0.pids 1 * BTL) a p
        + paODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BTL BK BV i a p)
      (fun a c => if c + (i - s0.pids 1 * BTL) ≤ a then
          paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c)
            * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a (i + c)
        else 0)
      (fun c p => paVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c) p)
      (by simpa using hbo) (by simp) (by simp [paVTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_advance_col_eval _ k (s0.pids 2 * s_qk_h) K T BK BTS s_qk_d s_qk_t
      (paIk s0 V BV * BK) i BTS "p_k" (by simp [hpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_advance_row_eval _ v (s0.pids 2 * s_vo_h) T V BTS BV s_vo_t s_vo_d
      i (paIv s0 V BV * BV) BTS "p_v" (by simp [hpv])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_okAdd_eval _ BTS (i - s0.pids 1 * BTL) (by simpa using hok)))]
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
      pa_dvd_step_le hdvd hBTLd h2
    omega
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbq
  · rw [show (⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BK BV (s0.pids 1 * BTL)
          idx.1.val idx.2.1.val
        + paODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BTL BK BV (i + BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun idx => some ((paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale T K V BTL BK BV (s0.pids 1 * BTL)
            idx.1.val idx.2.1.val
          + paODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
              T K V BTL BK BV i idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              (if c.val + (i - s0.pids 1 * BTL) ≤ idx.1.val then
                paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                    idx.1.val (i + c.val)
                  * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                      idx.1.val (i + c.val)
              else 0)
              * paVGuarded s0 v s_vo_h s_vo_t s_vo_d T V BV (i + c.val)
                  idx.2.1.val)⟩
      from Tile.ext fun idx =>
        congrArg some (paODiag_step' s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BK BV i BTS
          idx.1.val idx.2.1.val hlo).symm]
    simp
  · rw [show (⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV (s0.pids 1 * BTL) a.1.val
        + paZDiag s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV (i + BTS)
            a.1.val)⟩ : Tile .real [BTL])
        = ⟨fun a => some ((paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
            T K V BTL BK BV (s0.pids 1 * BTL) a.1.val
          + paZDiag s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV i a.1.val)
          + ∑ c : Fin BTS,
              (if c.val + (i - s0.pids 1 * BTL) ≤ a.1.val then
                paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                    a.1.val (i + c.val)
                  * paScore s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
                      a.1.val (i + c.val)
              else 0))⟩
      from Tile.ext fun a =>
        congrArg some (paZDiag_step' s0 q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV i BTS a.1.val hlo).symm]
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
theorem paFwdLoop2_run (s0 sPre : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BTS BK BV (s0.pids 1 * BTL) sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "_i"
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat 1))
          (Op.constNat BTL))
        (Op.constNat BTS) (paFwdLoop2Body BTL BTS BK BV)) sPre = some sL
      ∧ paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV ((s0.pids 1 + 1) * BTL) sL := by
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
      (P := fun i s => paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale T K V BTL BTS BK BV i s)
      (pa_mulConst_eval sPre "i_c" (s0.pids 1) BTL hic)
      hstop
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        paFwdLoop2_step s0 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          scale T K V BTL BTS BK BV i hBTS hi hPi)
  have hfin : final = (s0.pids 1 + 1) * BTL :=
    le_antisymm hP.2.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-- **The mid-section**: barrier no-op, `o_q`/`o_k` aranges, and the diagonal
pointer remakes carry `paInv1` at `i_c·BTL` into `paInv2` at `i_c·BTL`. -/
theorem paFwdMid_run (s0 sL1 : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat)
    (hInv : paInv1 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BTL BTS BK BV (s0.pids 1 * BTL) sL1) :
    ∃ sM, stepStmts
        [ Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
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
      ∧ paInv2 s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV (s0.pids 1 * BTL) sM := by
  obtain ⟨hpids, hmem, hdvd, hle, hic, hibh, hik, hiv, hbq, hbo, hbz, hpk, hpv⟩ := hInv
  rw [stepStmts.cons_some (pa_ifThen_false_noop [] sL1)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTL sL1))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTS _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (s0.pids 2 * s_qk_h) (paIk s0 V BV * BK) (s0.pids 1 * BTL)
      (pa_mulConst_eval _ "i_bh" (s0.pids 2) s_qk_h (by simpa using hibh))
      (pa_mulConst_eval _ "i_k" (paIk s0 V BV) BK (by simpa using hik))
      (pa_mulConst_eval _ "i_c" (s0.pids 1) BTL (by simpa using hic))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTS, BV] [s_vo_t, s_vo_d]
      (s0.pids 2 * s_vo_h) (s0.pids 1 * BTL) (paIv s0 V BV * BV)
      (pa_mulConst_eval _ "i_bh" (s0.pids 2) s_vo_h (by simpa using hibh))
      (pa_mulConst_eval _ "i_c" (s0.pids 1) BTL (by simpa using hic))
      (pa_mulConst_eval _ "i_v" (paIv s0 V BV) BV (by simpa using hiv))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, le_rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hpids
  · simpa using hmem
  · simp
  · have : (s0.pids 1 + 1) * BTL = s0.pids 1 * BTL + BTL := by ring
    omega
  · simpa using hic
  · simpa using hibh
  · simpa using hik
  · simpa using hiv
  · simpa using hbq
  · rw [show (⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BK BV (s0.pids 1 * BTL)
          idx.1.val idx.2.1.val
        + paODiag s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BTL BK BV (s0.pids 1 * BTL) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun idx => some (paOAcc s0 q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d scale T K V BTL BK BV (s0.pids 1 * BTL)
            idx.1.val idx.2.1.val)⟩
      from Tile.ext fun idx => by simp [paODiag]]
    simpa using hbo
  · rw [show (⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV (s0.pids 1 * BTL) a.1.val
        + paZDiag s0 q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
            (s0.pids 1 * BTL) a.1.val)⟩ : Tile .real [BTL])
        = ⟨fun a => some (paZAcc s0 q k s_qk_h s_qk_t s_qk_d scale
            T K V BTL BK BV (s0.pids 1 * BTL) a.1.val)⟩
      from Tile.ext fun a => by simp [paZDiag]]
    simpa using hbz
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
/-- **The forward prologue** (statements 1–13): program ids, `NV`/`i_k`/`i_v`,
the three block pointers, the scaled `b_q`, and the zero accumulators
establish `paInv1` at `0`. -/
theorem paFwdPrologue_run (s : BlockState) (q k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BTL BTS BK BV : Nat) :
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
          Stmt.assign .real [BTL] "b_z" (Op.full [BTL] (Op.const 0)) ] s = some sP
      ∧ paInv1 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BTL BTS BK BV 0 sP := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pa_NV_eval _ V BV))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_floorDiv_refs_eval _ "i_kv" "NV" (s.pids 0) (paNV V BV)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_mod_refs_eval _ "i_kv" "NV" (s.pids 0) (paNV V BV)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval q _ _ _ _ [T, K] [BTL, BK] [s_qk_t, s_qk_d]
      (s.pids 2 * s_qk_h) (s.pids 1 * BTL) (s.pids 0 / paNV V BV * BK)
      (pa_mulConst_eval _ "i_bh" (s.pids 2) s_qk_h (by simp))
      (pa_mulConst_eval _ "i_c" (s.pids 1) BTL (by simp))
      (pa_mulConst_eval _ "i_k" (s.pids 0 / paNV V BV) BK (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BTS] [s_qk_d, s_qk_t]
      (s.pids 2 * s_qk_h) (s.pids 0 / paNV V BV * BK) 0
      (pa_mulConst_eval _ "i_bh" (s.pids 2) s_qk_h (by simp))
      (pa_mulConst_eval _ "i_k" (s.pids 0 / paNV V BV) BK (by simp))
      (evalOp_constNat 0 _)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTS, BV] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) 0 (s.pids 0 % paNV V BV * BV)
      (pa_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simp))
      (evalOp_constNat 0 _)
      (pa_mulConst_eval _ "i_v" (s.pids 0 % paNV V BV) BV (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_load_bp_2d q _ "p_q" (s.pids 2 * s_qk_h) T K BTL BK s_qk_t s_qk_d
      (s.pids 1 * BTL) (s.pids 0 / paNV V BV * BK) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_qScale_eval s _ q s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
      (by
        have h : (⟨fun idx : TileIndex [BTL, BK] =>
            if (s.pids 1 * BTL + idx.1.val < T
                ∧ s.pids 0 / paNV V BV * BK + idx.2.1.val < K) then
              some (s.readMem q (s.pids 2 * s_qk_h
                + (s.pids 1 * BTL + idx.1.val) * s_qk_t
                + (s.pids 0 / paNV V BV * BK + idx.2.1.val) * s_qk_d))
            else some 0⟩ : Tile .real [BTL, BK])
            = ⟨fun idx : TileIndex [BTL, BK] =>
            if (s.pids 1 * BTL + idx.1.val < T
                ∧ paIk s V BV * BK + idx.2.1.val < K) then
              some (s.readMem q (s.pids 2 * s_qk_h
                + (s.pids 1 * BTL + idx.1.val) * s_qk_t
                + (paIk s V BV * BK + idx.2.1.val) * s_qk_d))
            else some 0⟩ := rfl
        rw [← h]
        simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_zeros_eval [BTL, BV] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_zeros_eval [BTL] _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, dvd_zero BTS, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · simp
  · simp
  · simp [paIk]
  · simp [paIv]
  · simp
  · rw [show (⟨fun idx => some (paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h
          s_vo_t s_vo_d scale T K V BTL BK BV 0 idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [paOAcc]]
    simp
  · rw [show (⟨fun a => some (paZAcc s q k s_qk_h s_qk_t s_qk_d scale
          T K V BTL BK BV 0 a.1.val)⟩ : Tile .real [BTL])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun a => by simp [paZAcc]]
    simp
  · simp [paIk]
  · simp [paIv]

/-! ## The `z` masked plain-pointer store -/

/-- The post-store state of the masked `z` store: lane `a` written iff
`lo + a < T`. -/
noncomputable def paZStoreState (sin : BlockState) (z : RegionName)
    (zbase lo T BTL : Nat) (g : Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BTL]).foldl
    (fun acc a => if lo + a.1.val < T
        then acc.writeMem z (zbase + a.1.val) (g a.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
/-- **`z` store step (eq).** -/
theorem paZStore_step_eq (sin : BlockState) (z : RegionName)
    (zbase lo T BTL : Nat) (g : Nat → ℝ) (bz : Tile .real [BTL])
    (hgf : ∀ a : Fin BTL, bz.data (a, PUnit.unit) = some (g a.val))
    (hb : sin.regs .real [BTL] "b_z" = some bz)
    (hp : sin.regs .ptr [BTL] "p_z"
      = some ⟨fun a => (z, zbase + a.1.val)⟩)
    (hic : sin.regs .nat [] "i_c" = some (Tile.scalar ic))
    (hlo : ic * BTL = lo) :
    stepStmt (Stmt.store .real [BTL] (MemAccess.ptr (Op.ref .ptr [BTL] "p_z"))
        (Op.ref .real [BTL] "b_z")
        (MaskOpt.mask
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
              (Op.arange BTL))
            (Op.constNat T)))) sin
      = some (paZStoreState sin z zbase lo T BTL g) := by
  have hmask : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.arange BTL))
      (Op.constNat T)) sin
      = some (⟨fun a => decide (lo + a.1.val < T)⟩ : Tile .bool [BTL]) := by
    rw [evalOp_lt]
    rw [evalOp_add]
    rw [pa_mulConst_eval sin "i_c" ic BTL hic]
    simp only [evalOp_arange, evalOp_constNat, Option.bind_eq_bind,
      Option.bind_some]
    refine congrArg some (Tile.ext fun a => ?_)
    show decide (ic * BTL + a.1.val < T) = decide (lo + a.1.val < T)
    rw [hlo]
  unfold stepStmt paZStoreState
  simp only [evalOp_ref, hb, hp, hmask]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
  refine congrArg some
    (congrArg (fun gg => List.foldl gg sin (TileShape.allIndices [BTL])) ?_)
  funext acc a
  by_cases h : lo + a.1.val < T
  · simp only [h, decide_true, if_true, BlockState.writeMemTyped_real, hgf]
    rfl
  · simp only [h, decide_false, Bool.false_eq_true, if_false]

set_option maxHeartbeats 4000000 in
/-- **`z` store readback + frames.** -/
theorem paZStore_step_props (sin : BlockState) (z : RegionName)
    (zbase lo T BTL : Nat) (g : Nat → ℝ) :
    (paZStoreState sin z zbase lo T BTL g).pids = sin.pids
      ∧ (∀ a : Fin BTL, lo + a.val < T →
          (paZStoreState sin z zbase lo T BTL g).readMem z (zbase + a.val)
            = g a.val)
      ∧ (∀ rg off, rg ≠ z →
          (paZStoreState sin z zbase lo T BTL g).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold paZStoreState
  refine ⟨?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · intro a ha
    have hInj : Function.Injective
        (fun i : TileIndex [BTL] => zbase + i.1.val) := by
      rintro ⟨x, u⟩ ⟨y, u'⟩ hxy
      have hxy' : zbase + x.val = zbase + y.val := hxy
      have hx : x = y := Fin.ext (by omega)
      cases u
      cases u'
      rw [hx]
    have h := BlockState.scatter_readback_prop_masked_nd (region := z) sin
      (fun i : TileIndex [BTL] => zbase + i.1.val)
      (fun i => g i.1.val) (fun i => lo + i.1.val < T) hInj (a, PUnit.unit)
    rw [h, if_pos ha]
  · intro rg off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      z (fun i : TileIndex [BTL] => zbase + i.1.val)
      (fun i => g i.1.val) (fun i => lo + i.1.val < T) _ sin rg off hrg

/-! ## Final store recipes -/

private theorem pa_oBase_eval (sin : BlockState) (B H s_vo_h ibh ik : Nat)
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

private theorem pa_zPtr_eval (sin : BlockState) (z : RegionName)
    (B H T BTL ibh ik ic : Nat)
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar ik))
    (hic : sin.regs .nat [] "i_c" = some (Tile.scalar ic)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase z)
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                (Op.mul .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                  (Op.ref .nat [] "i_k")))
              (Op.constNat T))
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)))
          (Op.arange BTL))) sin
      = some (⟨fun a => (z, (ibh + B * H * ik) * T + ic * BTL + a.1.val)⟩
          : Tile .ptr [BTL]) := by
  have hoff : evalOp (Op.add .nat Broadcast.scalarL
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
              (Op.ref .nat [] "i_k")))
          (Op.constNat T))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)))
      (Op.arange BTL)) sin
      = some (Tile.vec fun a : Fin BTL =>
          (ibh + B * H * ik) * T + ic * BTL + a.val) := by
    rw [evalOp_add, evalOp_add, pa_oBase_eval sin B H T ibh ik hibh hik,
      pa_mulConst_eval sin "i_c" ic BTL hic]
    simp only [evalOp_arange, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [evalOp_ptrAdd, evalOp_ptrBase, hoff]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun a => ?_)
  show (z, 0 + ((ibh + B * H * ik) * T + ic * BTL + a.1.val))
    = (z, (ibh + B * H * ik) * T + ic * BTL + a.1.val)
  rw [Nat.zero_add]

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The forward epilogue**: `p_o`/`p_z` makes and the two stores, from the
loop-2 exit invariant, with both readbacks. -/
theorem paFwdStores_run (s sL2 : BlockState) (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hOZ : o ≠ z) (hσ : BV ≤ s_vo_t)
    (hInvF : paInv2 s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      T K V BTL BTS BK BV ((s.pids 1 + 1) * BTL) sL2) :
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
          Stmt.assign .ptr [BTL] "p_z"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase z)
              (Op.add .nat Broadcast.scalarL
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil
                    (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                      (Op.mul .nat Broadcast.nil
                        (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                        (Op.ref .nat [] "i_k")))
                    (Op.constNat T))
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c")
                    (Op.constNat BTL)))
                (Op.arange BTL))),
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
            (Op.ref .real [BTL, BV] "b_o") MaskOpt.none,
          Stmt.store .real [BTL] (MemAccess.ptr (Op.ref .ptr [BTL] "p_z"))
            (Op.ref .real [BTL] "b_z")
            (MaskOpt.mask
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c")
                    (Op.constNat BTL))
                  (Op.arange BTL))
                (Op.constNat T))) ] sL2 = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], paOActive s T V BV BTL idx →
          sF.readMem o (paOOffset s B H s_vo_h s_vo_t 1 V BV BTL idx)
            = paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
                T K V BTL BK BV idx.1.val idx.2.1.val)
      ∧ (∀ a : Fin BTL, s.pids 1 * BTL + a.val < T →
          sF.readMem z (paZOffset s B H T V BV BTL a)
            = paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a.val) := by
  obtain ⟨hFpids, hFmem, hFlo, hFdvd, hFle, hFic, hFibh, hFik, hFiv, hFbq,
    hFbo, hFbz, hFoq, hFok, hFpk, hFpv⟩ := hInvF
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BTL, BV] [s_vo_t, 1]
      ((s.pids 2 + B * H * paIk s V BV) * s_vo_h) (s.pids 1 * BTL)
      (paIv s V BV * BV)
      (pa_oBase_eval _ B H s_vo_h (s.pids 2) (paIk s V BV) hFibh hFik)
      (pa_mulConst_eval _ "i_c" (s.pids 1) BTL hFic)
      (pa_mulConst_eval _ "i_v" (paIv s V BV) BV hFiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_zPtr_eval _ z B H T BTL (s.pids 2) (paIk s V BV) (s.pids 1)
      (by simpa using hFibh) (by simpa using hFik) (by simpa using hFic)))]
  rw [stepStmts.cons_some (paStore_step_eq _ o "b_o" "p_o"
    ((s.pids 2 + B * H * paIk s V BV) * s_vo_h) s_vo_t (s.pids 1 * BTL)
    (paIv s V BV * BV) T V BTL BV
    (fun a p => paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      T K V BTL BK BV a p)
    ⟨fun idx => some (paOAcc s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1
        scale T K V BTL BK BV (s.pids 1 * BTL) idx.1.val idx.2.1.val
      + paODiag s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
          T K V BTL BK BV ((s.pids 1 + 1) * BTL) idx.1.val idx.2.1.val)⟩
    (fun e p => rfl)
    (by simpa using hFbo)
    (by simp))]
  obtain ⟨hSpids, hSregs, hSread, hSother, hSuntouched⟩ := paStore_step_props
    ((sL2.setReg "p_o" .blockPtr [BTL, BV]
        ⟨fun _ => BlockPtr.mk o ((s.pids 2 + B * H * paIk s V BV) * s_vo_h)
          [T, V] [BTL, BV] [s_vo_t, 1]
          [s.pids 1 * BTL, paIv s V BV * BV]⟩).setReg "p_z" .ptr [BTL]
        ⟨fun a => (z, (s.pids 2 + B * H * paIk s V BV) * T + s.pids 1 * BTL
          + a.1.val)⟩)
    o ((s.pids 2 + B * H * paIk s V BV) * s_vo_h) s_vo_t (s.pids 1 * BTL)
    (paIv s V BV * BV) T V BTL BV
    (fun a p => paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t 1 scale
      T K V BTL BK BV a p)
    (paStoreAddr_injective _ _ _ _ BTL BV hσ)
  rw [stepStmts.cons_some (paZStore_step_eq _ z
    ((s.pids 2 + B * H * paIk s V BV) * T + s.pids 1 * BTL)
    (s.pids 1 * BTL) T BTL
    (fun a => paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a)
    ⟨fun a => some (paZAcc s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
        (s.pids 1 * BTL) a.1.val
      + paZDiag s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV
          ((s.pids 1 + 1) * BTL) a.1.val)⟩
    (fun a => rfl)
    (by rw [hSregs]; simpa using hFbz)
    (by rw [hSregs]; simp)
    (by rw [hSregs]; simpa using hFic)
    rfl)]
  rw [stepStmts.nil]
  obtain ⟨hZpids, hZread, hZother⟩ := paZStore_step_props _ z
    ((s.pids 2 + B * H * paIk s V BV) * T + s.pids 1 * BTL)
    (s.pids 1 * BTL) T BTL
    (fun a => paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a)
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hact
    rw [hZother o _ hOZ]
    exact hSread idx hact
  · intro a ha
    exact hZread a ha

/-! ## ★ Forward main theorem -/

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **★ Forward main theorem: the `o` and `z` stores are the genuine causal
rebased-attention closed forms.**

For every program `(i_kv, i_c, i_bh)` (universally quantified through
`s.pids`), executing the full forward surface succeeds, the `o` block store
holds `Σ_{t ≤ i_c·BTL + a} score(a,t)² · v[t,p]` at every in-window lane
`(a, p)`, and the `z` masked store holds the matching normalizer
`Σ_{t ≤ i_c·BTL + a} score(a,t)²` at every in-range row.

Side conditions: `o ≠ z` (the two output regions are distinct buffers),
the host's contiguous last-dim stride `s_vo_d = 1` with `BV ≤ s_vo_t`
(store-lane injectivity), and the host's own `assert BTL % BTS == 0`. The
loads are boundary-checked, so **no** divisibility hypothesis on `T` is
needed: ragged tails are exact. -/
specification pa_fwd_o_exec_genuine
    (s : BlockState) (q k v o z : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hOZ : o ≠ z) (hSd : s_vo_d = 1) (hσ : BV ≤ s_vo_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) :
    ∃ sF, exec (pa_fwd_surface q k v o z s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d scale B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BV], paOActive s T V BV BTL idx →
          sF.readMem o (paOOffset s B H s_vo_h s_vo_t s_vo_d V BV BTL idx)
            = paOOut s q k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
                T K V BTL BK BV idx.1.val idx.2.1.val)
      ∧ (∀ a : Fin BTL, s.pids 1 * BTL + a.val < T →
          sF.readMem z (paZOffset s B H T V BV BTL a)
            = paZOut s q k s_qk_h s_qk_t s_qk_d scale T K V BTL BK BV a.val) := by
  subst hSd
  obtain ⟨sP, hPro, hInvP⟩ := paFwdPrologue_run s q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV
  obtain ⟨sL1, hL1, hInv1⟩ := paFwdLoop1_run s sP q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hBTS hBTSpos hInvP
  obtain ⟨sM, hMid, hInv2⟩ := paFwdMid_run s sL1 q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hInv1
  obtain ⟨sL2, hL2, hInvF⟩ := paFwdLoop2_run s sM q k v s_qk_h s_qk_t s_qk_d
    s_vo_h s_vo_t 1 scale T K V BTL BTS BK BV hBTS hBTSpos hInv2
  rw [exec, pa_fwd_body_eq]
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
      Stmt.assign .real [BTL] "b_z" (Op.full [BTL] (Op.const 0)),
      Stmt.forRangeDyn "_i" (Op.constNat 0)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
        (Op.constNat BTS) (paFwdLoop1Body BTL BTS BK BV),
      Stmt.ifThen (Op.constBool Bool.false) [],
      Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
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
        (Op.constNat BTS) (paFwdLoop2Body BTL BTS BK BV),
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
      Stmt.assign .ptr [BTL] "p_z"
        (Op.ptrAdd Broadcast.scalarL (Op.ptrBase z)
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                    (Op.ref .nat [] "i_k")))
                (Op.constNat T))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL)))
            (Op.arange BTL))),
      Stmt.store .real [BTL, BV]
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
        (Op.ref .real [BTL, BV] "b_o") MaskOpt.none,
      Stmt.store .real [BTL] (MemAccess.ptr (Op.ref .ptr [BTL] "p_z"))
        (Op.ref .real [BTL] "b_z")
        (MaskOpt.mask
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
              (Op.arange BTL))
            (Op.constNat T))) ] : List Stmt)
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
          Stmt.assign .real [BTL] "b_z" (Op.full [BTL] (Op.const 0)) ]
        ++ (Stmt.forRangeDyn "_i" (Op.constNat 0)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BTL))
            (Op.constNat BTS) (paFwdLoop1Body BTL BTS BK BV)
          :: ([ Stmt.ifThen (Op.constBool Bool.false) [],
              Stmt.assign .nat [BTL] "o_q" (Op.arange BTL),
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
                (Op.constNat BTS) (paFwdLoop2Body BTL BTS BK BV)
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
                  Stmt.assign .ptr [BTL] "p_z"
                    (Op.ptrAdd Broadcast.scalarL (Op.ptrBase z)
                      (Op.add .nat Broadcast.scalarL
                        (Op.add .nat Broadcast.nil
                          (Op.mul .nat Broadcast.nil
                            (Op.add .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                              (Op.mul .nat Broadcast.nil
                                (Op.mul .nat Broadcast.nil (Op.constNat B) (Op.constNat H))
                                (Op.ref .nat [] "i_k")))
                            (Op.constNat T))
                          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c")
                            (Op.constNat BTL)))
                        (Op.arange BTL))),
                  Stmt.store .real [BTL, BV]
                    (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_o") [0, 1])
                    (Op.ref .real [BTL, BV] "b_o") MaskOpt.none,
                  Stmt.store .real [BTL] (MemAccess.ptr (Op.ref .ptr [BTL] "p_z"))
                    (Op.ref .real [BTL] "b_z")
                    (MaskOpt.mask
                      (Op.lt ComparableDType.nat Broadcast.scalarR
                        (Op.add .nat Broadcast.scalarL
                          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c")
                            (Op.constNat BTL))
                          (Op.arange BTL))
                        (Op.constNat T)) ) ])))
      from rfl]
  rw [stepStmts.append_some hPro, stepStmts.cons_some hL1,
    stepStmts.append_some hMid, stepStmts.cons_some hL2]
  obtain ⟨sF, hSt, hRead1, hRead2⟩ := paFwdStores_run s sL2 q k v o z
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t scale B H T K V BTL BTS BK BV
    hOZ hσ hInvF
  exact ⟨sF, hSt, hRead1, hRead2⟩

/-! ## Backward value tiles and load bridges -/

/-- The fixed loaded `b_k` chunk tile. -/
noncomputable def pbKTile (s : BlockState) (k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (T K BTL BK : Nat) :
    Tile .real [BTL, BK] :=
  ⟨fun idx => some (pbKGuarded s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK
    idx.1.val idx.2.1.val)⟩

/-- The fixed loaded `b_v` chunk tile. -/
noncomputable def pbVTile (s : BlockState) (v : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (T V BTL BV : Nat) :
    Tile .real [BTL, BV] :=
  ⟨fun idx => some (pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
    idx.1.val idx.2.1.val)⟩

/-- The loaded transposed `b_q` block at key-column offset `i`. -/
noncomputable def pbQTile (s : BlockState) (q : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_k : Nat) (T K BK BTS : Nat) (i : Nat) :
    Tile .real [BK, BTS] :=
  ⟨fun idx => some (pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK
    idx.1.val (i + idx.2.1.val))⟩

/-- The loaded transposed `b_do` block at key-column offset `i`. -/
noncomputable def pbDoTile (s : BlockState) (do_ : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_v : Nat) (T V BV BTS : Nat) (i : Nat) :
    Tile .real [BV, BTS] :=
  ⟨fun idx => some (pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
    idx.1.val (i + idx.2.1.val))⟩

/-- The masked loaded `b_dz` block at key offset `i`. -/
noncomputable def pbDzTile (s : BlockState) (dz : RegionName)
    (i_bh T BTS : Nat) (i : Nat) : Tile .real [BTS] :=
  ⟨fun c => some (pbDzGuarded s dz i_bh T (i + c.1.val))⟩

/-- Scalar `constNat * constNat` products (the spliced backward offsets). -/
private theorem pb_constMul_eval (s : BlockState) (a b : Nat) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat a) (Op.constNat b)) s
      = some (Tile.scalar (a * b)) := by
  rw [evalOp_mul]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Loading `p_k` at offsets `[i_c·BTL, i_k·BK]` lands on `pbKTile`. -/
private theorem pb_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k T K BTL BK : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTL, BK] name = some
      ⟨fun _ => BlockPtr.mk k (i_bh * s_k_h) [T, K] [BTL, BK]
        [s_k_t, s_k_d] [i_c * BTL, i_k * BK]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] name) [0, 1])
        MaskOpt.none) sin
      = some (pbKTile s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK) := by
  rw [pa_load_bp_2d k sin name (i_bh * s_k_h) T K BTL BK s_k_t s_k_d
    (i_c * BTL) (i_k * BK) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, e, u⟩ := idx
  simp only [pbKTile, pbKGuarded]
  by_cases h : i_c * BTL + l.val < T ∧ i_k * BK + e.val < K
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_v` at offsets `[i_c·BTL, i_v·BV]` lands on `pbVTile`. -/
private theorem pb_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v T V BTL BV : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BTL, BV] name = some
      ⟨fun _ => BlockPtr.mk v (i_bh * s_v_h) [T, V] [BTL, BV]
        [s_v_t, s_v_d] [i_c * BTL, i_v * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] name) [0, 1])
        MaskOpt.none) sin
      = some (pbVTile s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV) := by
  rw [pa_load_bp_2d v sin name (i_bh * s_v_h) T V BTL BV s_v_t s_v_d
    (i_c * BTL) (i_v * BV) hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, p, u⟩ := idx
  simp only [pbVTile, pbVGuarded]
  by_cases h : i_c * BTL + l.val < T ∧ i_v * BV + p.val < V
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_q` at offsets `[i_k·BK, i]` (the `(K, T)` parent) lands on
`pbQTile … i`. -/
private theorem pb_qLoad_eq (s sin : BlockState) (q : RegionName) (name : RegName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_k T K BK BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BK, BTS] name = some
      ⟨fun _ => BlockPtr.mk q (i_bh * s_k_h) [K, T] [BK, BTS]
        [s_k_d, s_k_t] [i_k * BK, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (pbQTile s q s_k_h s_k_t s_k_d i_bh i_k T K BK BTS i) := by
  rw [pa_load_bp_2d q sin name (i_bh * s_k_h) K T BK BTS s_k_d s_k_t
    (i_k * BK) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, c, u⟩ := idx
  simp only [pbQTile, pbQGuarded]
  by_cases h : i_k * BK + e.val < K ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-- Loading `p_do` at offsets `[i_v·BV, i]` (the `(V, T)` parent) lands on
`pbDoTile … i`. -/
private theorem pb_doLoad_eq (s sin : BlockState) (do_ : RegionName) (name : RegName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_v T V BV BTS : Nat) (i : Nat)
    (hmem : sin.mem = s.mem)
    (hreg : sin.regs .blockPtr [BV, BTS] name = some
      ⟨fun _ => BlockPtr.mk do_ (i_bh * s_v_h) [V, T] [BV, BTS]
        [s_v_d, s_v_t] [i_v * BV, i]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BV, BTS] name) [0, 1])
        MaskOpt.none) sin
      = some (pbDoTile s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV BTS i) := by
  rw [pa_load_bp_2d do_ sin name (i_bh * s_v_h) V T BV BTS s_v_d s_v_t
    (i_v * BV) i hreg]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨p, c, u⟩ := idx
  simp only [pbDoTile, pbDoGuarded]
  by_cases h : i_v * BV + p.val < V ∧ i + c.val < T
  · rw [if_pos h, if_pos h, pa_readMem_congr sin s hmem]
  · rw [if_neg h, if_neg h]

/-! ## Backward per-statement eval recipes -/

/-- The respelled block index
`i = tl.cdiv(T, BTS)·BTS − BTS − j·BTS` from the `j` register. -/
private theorem pb_iAssign_eval (sin : BlockState) (T BTS j : Nat)
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
      = some (Tile.scalar (pbNB T BTS - BTS - j * BTS)) := by
  simp only [evalOp, evalOp_ref, hj, evalOp_constNat, bind, Option.bind]
  rfl

/-- The respelled descending trip count (the loop's stop expression). -/
private theorem pb_trip_eval (s : BlockState) (T BTS i_c BTL : Nat) :
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
      = some (Tile.scalar (pbTrip T BTS i_c BTL)) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

/-- The `p_dz` pointer tile from the `i` register. -/
private theorem pb_dzPtr_eval (sin : BlockState) (dz : RegionName)
    (i_bh T BTS i : Nat)
    (hi : sin.regs .nat [] "i" = some (Tile.scalar i)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase dz)
        (Op.add .nat Broadcast.scalarL
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat T))
            (Op.ref .nat [] "i"))
          (Op.arange BTS))) sin
      = some (⟨fun c => (dz, i_bh * T + i + c.1.val)⟩ : Tile .ptr [BTS]) := by
  have hoff : evalOp (Op.add .nat Broadcast.scalarL
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat T))
        (Op.ref .nat [] "i"))
      (Op.arange BTS)) sin
      = some (Tile.vec fun c : Fin BTS => i_bh * T + i + c.val) := by
    rw [evalOp_add, evalOp_add, pb_constMul_eval sin i_bh T, evalOp_ref, hi]
    simp only [evalOp_arange, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [evalOp_ptrAdd, evalOp_ptrBase, hoff]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun c => ?_)
  show (dz, 0 + (i_bh * T + i + c.1.val)) = (dz, i_bh * T + i + c.1.val)
  rw [Nat.zero_add]

/-- The masked `b_dz` load: in-range lanes read the region, masked-off lanes
read the clean-input `undef ≡ 0` oracle. -/
private theorem pb_dzLoad_eq (s sin : BlockState) (dz : RegionName)
    (i_bh T BTS i : Nat)
    (hmem : sin.mem = s.mem) (hund : sin.undef = s.undef)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hp : sin.regs .ptr [BTS] "p_dz"
      = some (⟨fun c => (dz, i_bh * T + i + c.1.val)⟩ : Tile .ptr [BTS]))
    (hi : sin.regs .nat [] "i" = some (Tile.scalar i)) :
    evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BTS] "p_dz"))
        (MaskOpt.mask
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "i") (Op.arange BTS))
            (Op.constNat T)))) sin
      = some (pbDzTile s dz i_bh T BTS i) := by
  simp only [evalOp, evalOp_ref, hp, hi, evalOp_arange, evalOp_constNat, bind,
    Option.bind, Option.map]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨c, u⟩ := idx
  simp only [pbDzTile, pbDzGuarded, Tile.cop_data, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex]
  by_cases h : i + c.val < T
  · rw [if_pos h]
    simp only [Tile.scalar, Tile.vec, NumericDType.add, ComparableDType.lt,
      h, decide_true, if_true, BlockState.readMemValue_real]
    rw [pa_readMem_congr sin s hmem, Nat.add_assoc]
  · rw [if_neg h]
    simp only [Tile.scalar, Tile.vec, NumericDType.add, ComparableDType.lt,
      h, decide_false, Bool.false_eq_true, if_false, if_true]
    rw [hund, hundef]

/-- `b_s = tl.dot(b_k, b_q) * scale` lands lanewise on `pbSVal · (i + c)`. -/
private theorem pb_sScaled_eval (s sin : BlockState) (q k : RegionName)
    (s_k_h s_k_t s_k_d : Nat) (i_bh i_c i_k : Nat) (scale : ℝ)
    (T K BTL BK BTS : Nat) (i : Nat)
    (hk : sin.regs .real [BTL, BK] "b_k"
      = some (pbKTile s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK))
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some (pbQTile s q s_k_h s_k_t s_k_d i_bh i_k T K BK BTS i)) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (Op.ref .real [BTL, BK] "b_k")
          (Op.ref .real [BK, BTS] "b_q"))
        (Op.const scale)) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
            idx.1.val (i + idx.2.1.val))⟩ := by
  have hdot := pa_dot_eval (Op.ref .real [BTL, BK] "b_k")
    (Op.ref .real [BK, BTS] "b_q") sin _ _
    (by rw [evalOp_ref]; exact hk) (by rw [evalOp_ref]; exact hq)
  erw [pa_mulTile_eval Broadcast.scalarR _ _ sin _ _ hdot (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [pa_dot2d_elem _ _ l c
    (fun e => pbKGuarded s k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK
      l.val e.val)
    (fun e => pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK
      e.val (i + c.val))
    (fun e => rfl) (fun e => rfl)]
  rfl

/-- The desc-loop `b_ds = tl.dot(b_v, b_do) * scale` (scaled dot only; the
`dz` gate is folded in by `pb_gateScaled_step`). -/
private theorem pb_dsScaled_eval (s sin : BlockState) (v do_ : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat) (scale : ℝ)
    (T V BTL BV BTS : Nat) (i : Nat)
    (hv : sin.regs .real [BTL, BV] "b_v"
      = some (pbVTile s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV))
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some (pbDoTile s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV BTS i)) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
          (Op.ref .real [BV, BTS] "b_do"))
        (Op.const scale)) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some ((∑ p : Fin BV,
              pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
                idx.1.val p.val
                * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                    p.val (i + idx.2.1.val)) * scale)⟩ := by
  have hdot := pa_dot_eval (Op.ref .real [BTL, BV] "b_v")
    (Op.ref .real [BV, BTS] "b_do") sin _ _
    (by rw [evalOp_ref]; exact hv) (by rw [evalOp_ref]; exact hdo)
  erw [pa_mulTile_eval Broadcast.scalarR _ _ sin _ _ hdot (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, c, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [pa_dot2d_elem _ _ l c
    (fun p => pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
      l.val p.val)
    (fun p => pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
      p.val (i + c.val))
    (fun p => rfl) (fun p => rfl)]
  rfl

/-- The diag-loop unscaled `b_ds = tl.dot(b_v, b_do)`. -/
private theorem pb_dsDot_eval (s sin : BlockState) (v do_ : RegionName)
    (s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_v : Nat)
    (T V BTL BV BTS : Nat) (i : Nat)
    (hv : sin.regs .real [BTL, BV] "b_v"
      = some (pbVTile s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV))
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some (pbDoTile s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV BTS i)) :
    evalOp (Op.dot (batch := []) (Op.ref .real [BTL, BV] "b_v")
        (Op.ref .real [BV, BTS] "b_do")) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (∑ p : Fin BV,
            pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
              idx.1.val p.val
              * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                  p.val (i + idx.2.1.val))⟩ := by
  erw [pa_dot_eval (Op.ref .real [BTL, BV] "b_v")
    (Op.ref .real [BV, BTS] "b_do") sin _ _
    (by rw [evalOp_ref]; exact hv) (by rw [evalOp_ref]; exact hdo)]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, c, u⟩ := idx
  rw [pa_dot2d_elem _ _ l c
    (fun p => pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
      l.val p.val)
    (fun p => pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
      p.val (i + c.val))
    (fun p => rfl) (fun p => rfl)]

/-- `b_dv += tl.dot(b_s2, tl.trans(b_do))` on all-`some` tiles. -/
private theorem pb_dvAdd_eval (sin : BlockState) (BTL BTS BV : Nat)
    (g f w : Nat → Nat → ℝ)
    (hdv : sin.regs .real [BTL, BV] "b_dv"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hs2 : sin.regs .real [BTL, BTS] "b_s2"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩)
    (hdo : sin.regs .real [BV, BTS] "b_do"
      = some ⟨fun idx => some (w idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BV] "b_dv")
        (Op.dot (batch := []) (Op.ref .real [BTL, BTS] "b_s2")
          (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")))) sin
      = some ⟨fun idx : TileIndex [BTL, BV] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin BTS, f idx.1.val c.val * w idx.2.1.val c.val)⟩ := by
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")) sin
      = some (Tile.transpose []
          (⟨fun idx => some (w idx.1.val idx.2.1.val)⟩ : Tile .real [BV, BTS])) := by
    erw [evalOp_transpose, evalOp_ref, hdo]
    rfl
  have hdot := pa_dot_eval (Op.ref .real [BTL, BTS] "b_s2")
    (Op.transpose (batch := []) (Op.ref .real [BV, BTS] "b_do")) sin _ _
    (by rw [evalOp_ref]; exact hs2) htr
  erw [pa_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdv) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  erw [pa_dot2d_elem _ _ l p (fun c => f l.val c.val) (fun c => w p.val c.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

/-- `b_dk += tl.dot(2.0 * b_ds * b_s, tl.trans(b_q))` on all-`some` tiles
(the `2.0` literal is normalized to `2`). -/
private theorem pb_dkAdd_eval (sin : BlockState) (BTL BTS BK : Nat)
    (g dsF sF qF : Nat → Nat → ℝ)
    (hdk : sin.regs .real [BTL, BK] "b_dk"
      = some ⟨fun idx => some (g idx.1.val idx.2.1.val)⟩)
    (hds : sin.regs .real [BTL, BTS] "b_ds"
      = some ⟨fun idx => some (dsF idx.1.val idx.2.1.val)⟩)
    (hs : sin.regs .real [BTL, BTS] "b_s"
      = some ⟨fun idx => some (sF idx.1.val idx.2.1.val)⟩)
    (hq : sin.regs .real [BK, BTS] "b_q"
      = some ⟨fun idx => some (qF idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BTL, BK] "b_dk")
        (Op.dot (batch := [])
          (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.mul .real Broadcast.scalarL (Op.const 2.0)
              (Op.ref .real [BTL, BTS] "b_ds"))
            (Op.ref .real [BTL, BTS] "b_s"))
          (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")))) sin
      = some ⟨fun idx : TileIndex [BTL, BK] =>
          some (g idx.1.val idx.2.1.val
            + ∑ c : Fin BTS,
                2 * dsF idx.1.val c.val * sF idx.1.val c.val
                  * qF idx.2.1.val c.val)⟩ := by
  have hmul2 := pa_mulTile_eval Broadcast.scalarL (Op.const 2.0)
    (Op.ref .real [BTL, BTS] "b_ds") sin _ _ (evalOp_const 2.0 sin)
    (by rw [evalOp_ref]; exact hds)
  have hmulO := pa_mulTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ (Op.ref .real [BTL, BTS] "b_s") sin _ _ hmul2
    (by rw [evalOp_ref]; exact hs)
  have htr : evalOp (Op.transpose (batch := []) (Op.ref .real [BK, BTS] "b_q")) sin
      = some (Tile.transpose []
          (⟨fun idx => some (qF idx.1.val idx.2.1.val)⟩ : Tile .real [BK, BTS])) := by
    erw [evalOp_transpose, evalOp_ref, hq]
    rfl
  have hdot := pa_dot_eval _ _ sin _ _ hmulO htr
  erw [pa_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hdk) hdot]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨l, e, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  erw [pa_dot2d_elem _ _ l e
    (fun c => 2 * dsF l.val c.val * sF l.val c.val)
    (fun c => qF e.val c.val)
    (fun c => by
      show some ((2.0 : ℝ) * dsF l.val c.val * sF l.val c.val)
        = some (2 * dsF l.val c.val * sF l.val c.val)
      norm_num)
    (fun c => rfl)]
  rfl

/-- The `o_k[:, None] <= o_q[None, :]` mask at diagonal offset `off` (row `l`
kept iff `l ≤ c + off`). -/
def pbMaskTile (BTL BTS off : Nat) : Tile .bool [BTL, BTS] :=
  ⟨fun idx => decide (idx.1.val ≤ idx.2.1.val + off)⟩

private theorem pb_msLe_eval (sin : BlockState) (BTL BTS off : Nat)
    (hok : sin.regs .nat [BTL] "o_k" = some (Tile.vec fun r => r.val))
    (hoq : sin.regs .nat [BTS] "o_q" = some (Tile.vec fun r => r.val + off)) :
    evalOp (Op.le ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BTL] "o_k"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BTS] "o_q"))) sin
      = some (pbMaskTile BTL BTS off) := by
  rw [pa_evalOp_le_def]
  erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hok,
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoq]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  show decide (a.val ≤ c.val + off) = decide (a.val ≤ c.val + off)
  rfl

/-- `x = tl.where(m_s, x, 0.0)` on an all-`some` tile: the masked lane
(kernel-side keep `l ≤ c + off`). -/
private theorem pb_whereMask_eval (sin : BlockState) (BTL BTS off : Nat)
    (vname : RegName) (f : Nat → Nat → ℝ)
    (hm : sin.regs .bool [BTL, BTS] "m_s" = some (pbMaskTile BTL BTS off))
    (hs : sin.regs .real [BTL, BTS] vname
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.where (Op.ref .bool [BTL, BTS] "m_s")
        (Op.ref .real [BTL, BTS] vname)
        (Op.broadcast (Op.const 0.0) [BTL, BTS])) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some (if idx.1.val ≤ idx.2.1.val + off
            then f idx.1.val idx.2.1.val else 0)⟩ := by
  rw [evalOp_where]
  simp only [evalOp.eq_def, evalOp_ref, evalOp_const, hm, hs,
    Option.bind, Option.map, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨a, c, u⟩ := idx
  rw [Tile.select_data]
  simp only [pbMaskTile]
  by_cases h : a.val ≤ c.val + off
  · rw [if_pos (by simpa using h), if_pos h]
  · rw [if_neg (by simpa using h), if_neg h]
    norm_num

/-- `b_ds = tl.where(m_s, b_ds, 0.0) * scale` (the diag-loop rescale). -/
private theorem pb_whereMaskScale_eval (sin : BlockState) (BTL BTS off : Nat)
    (scale : ℝ) (f : Nat → Nat → ℝ)
    (hm : sin.regs .bool [BTL, BTS] "m_s" = some (pbMaskTile BTL BTS off))
    (hs : sin.regs .real [BTL, BTS] "b_ds"
      = some ⟨fun idx => some (f idx.1.val idx.2.1.val)⟩) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.where (Op.ref .bool [BTL, BTS] "m_s")
          (Op.ref .real [BTL, BTS] "b_ds")
          (Op.broadcast (Op.const 0.0) [BTL, BTS]))
        (Op.const scale)) sin
      = some ⟨fun idx : TileIndex [BTL, BTS] =>
          some ((if idx.1.val ≤ idx.2.1.val + off
            then f idx.1.val idx.2.1.val else 0) * scale)⟩ := by
  rw [pa_mulTile_eval Broadcast.scalarR _ _ sin _ _
    (pb_whereMask_eval sin BTL BTS off "b_ds" f hm hs) (evalOp_const scale sin)]
  refine congrArg some (Tile.ext fun idx => ?_)
  rfl

/-- `o_q += BTS` shifts the diag arange window. -/
private theorem pb_oqAdd_eval (sin : BlockState) (BTS off : Nat)
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

/-! ## The `i_v == 0` gate -/

private theorem pb_ifThenElse_step_bool (b : Bool) (cond : Op .bool [])
    (thenB elseB : List Stmt) (t : BlockState)
    (hcond : evalOp cond t = some (Tile.scalar b)) :
    stepStmt (Stmt.ifThenElse cond thenB elseB) t
      = if b then stepStmts thenB t else stepStmts elseB t := by
  have h : stepStmt (Stmt.ifThenElse cond thenB elseB) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts thenB t
            else stepStmts elseB t) := by
    unfold stepStmt
    cases evalOp cond t <;> rfl
  rw [h, hcond]
  rfl

private theorem pb_gateCond_eval (s : BlockState) (i_v : Nat) :
    evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat i_v)
      (Op.constNat 0)) s = some (Tile.scalar (decide (i_v = 0))) := by
  simp only [evalOp, evalOp_constNat, bind, Option.bind]
  rfl

/-- The desc-loop gate `if i_v == 0 { b_ds += b_dz[None, :] * scale }`:
both branches unify into a single `setReg` with the gate folded into an
`if i_v = 0` term. -/
private theorem pb_gateScaled_step (sin : BlockState) (i_v : Nat) (scale : ℝ)
    (BTL BTS : Nat) (dsF : Nat → Nat → ℝ) (dzF : Nat → ℝ)
    (hds : sin.regs .real [BTL, BTS] "b_ds"
      = some ⟨fun idx => some (dsF idx.1.val idx.2.1.val)⟩)
    (hdz : sin.regs .real [BTS] "b_dz"
      = some ⟨fun c => some (dzF c.1.val)⟩) :
    stepStmt (Stmt.ifThenElse
        (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat i_v) (Op.constNat 0))
        [ Stmt.assign .real [BTL, BTS] "b_ds"
            (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BTL, BTS] "b_ds")
              (Op.mul .real Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "b_dz"))
                (Op.const scale))) ]
        [ Stmt.assign .real [BTL, BTS] "b_ds" (Op.ref .real [BTL, BTS] "b_ds") ]) sin
      = some (sin.setReg "b_ds" .real [BTL, BTS]
          ⟨fun idx => some (dsF idx.1.val idx.2.1.val
            + (if i_v = 0 then dzF idx.2.1.val * scale else 0))⟩) := by
  rw [pb_ifThenElse_step_bool (decide (i_v = 0)) _ _ _ sin
    (pb_gateCond_eval sin i_v)]
  by_cases hv : i_v = 0
  · rw [if_pos (by simp [hv])]
    have hexpEval : evalOp (Op.expandDim ⟨0, by simp⟩
        (Op.ref .real [BTS] "b_dz")) sin
        = some (Tile.expandDim ⟨0, by simp⟩
            (⟨fun c => some (dzF c.1.val)⟩ : Tile .real [BTS])) := by
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdz]
    have hmulEval := pa_mulTile_eval Broadcast.scalarR _ (Op.const scale) sin _ _
      hexpEval (evalOp_const scale sin)
    have haddEval := pa_addTile_eval
      (Broadcast.consR (Broadcast.consSame Broadcast.nil)) _ _ sin _ _
      (by rw [evalOp_ref]; exact hds) hmulEval
    rw [stepStmts.cons_some (stepStmt_assign_eq_some haddEval), stepStmts.nil]
    refine congrArg some (congrArg (sin.setReg "b_ds" .real [BTL, BTS])
      (Tile.ext fun idx => ?_))
    obtain ⟨l, c, u⟩ := idx
    show NumericDType.real.add (some (dsF l.val c.val))
        (NumericDType.real.mul (some (dzF c.val)) (some scale))
      = some (dsF l.val c.val + (if i_v = 0 then dzF c.val * scale else 0))
    rw [if_pos hv]
    rfl
  · rw [if_neg (by simp [hv])]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .real [BTL, BTS] "b_ds") sin
          = some (⟨fun idx => some (dsF idx.1.val idx.2.1.val)⟩
            : Tile .real [BTL, BTS]) from by rw [evalOp_ref]; exact hds)),
      stepStmts.nil]
    refine congrArg some (congrArg (sin.setReg "b_ds" .real [BTL, BTS])
      (Tile.ext fun idx => ?_))
    show some (dsF idx.1.val idx.2.1.val)
      = some (dsF idx.1.val idx.2.1.val
          + (if i_v = 0 then dzF idx.2.1.val * scale else 0))
    rw [if_neg hv, add_zero]

/-- The diag-loop gate `if i_v == 0 { b_ds += b_dz[None, :] }` (unscaled). -/
private theorem pb_gateDiag_step (sin : BlockState) (i_v : Nat)
    (BTL BTS : Nat) (dsF : Nat → Nat → ℝ) (dzF : Nat → ℝ)
    (hds : sin.regs .real [BTL, BTS] "b_ds"
      = some ⟨fun idx => some (dsF idx.1.val idx.2.1.val)⟩)
    (hdz : sin.regs .real [BTS] "b_dz"
      = some ⟨fun c => some (dzF c.1.val)⟩) :
    stepStmt (Stmt.ifThenElse
        (Op.eq ComparableDType.nat Broadcast.nil (Op.constNat i_v) (Op.constNat 0))
        [ Stmt.assign .real [BTL, BTS] "b_ds"
            (Op.add .real (Broadcast.consR (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BTL, BTS] "b_ds")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BTS] "b_dz"))) ]
        [ Stmt.assign .real [BTL, BTS] "b_ds" (Op.ref .real [BTL, BTS] "b_ds") ]) sin
      = some (sin.setReg "b_ds" .real [BTL, BTS]
          ⟨fun idx => some (dsF idx.1.val idx.2.1.val
            + (if i_v = 0 then dzF idx.2.1.val else 0))⟩) := by
  rw [pb_ifThenElse_step_bool (decide (i_v = 0)) _ _ _ sin
    (pb_gateCond_eval sin i_v)]
  by_cases hv : i_v = 0
  · rw [if_pos (by simp [hv])]
    have hexpEval : evalOp (Op.expandDim ⟨0, by simp⟩
        (Op.ref .real [BTS] "b_dz")) sin
        = some (Tile.expandDim ⟨0, by simp⟩
            (⟨fun c => some (dzF c.1.val)⟩ : Tile .real [BTS])) := by
      erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hdz]
    have haddEval := pa_addTile_eval
      (Broadcast.consR (Broadcast.consSame Broadcast.nil)) _ _ sin _ _
      (by rw [evalOp_ref]; exact hds) hexpEval
    rw [stepStmts.cons_some (stepStmt_assign_eq_some haddEval), stepStmts.nil]
    refine congrArg some (congrArg (sin.setReg "b_ds" .real [BTL, BTS])
      (Tile.ext fun idx => ?_))
    obtain ⟨l, c, u⟩ := idx
    show NumericDType.real.add (some (dsF l.val c.val)) (some (dzF c.val))
      = some (dsF l.val c.val + (if i_v = 0 then dzF c.val else 0))
    rw [if_pos hv]
    rfl
  · rw [if_neg (by simp [hv])]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .real [BTL, BTS] "b_ds") sin
          = some (⟨fun idx => some (dsF idx.1.val idx.2.1.val)⟩
            : Tile .real [BTL, BTS]) from by rw [evalOp_ref]; exact hds)),
      stepStmts.nil]
    refine congrArg some (congrArg (sin.setReg "b_ds" .real [BTL, BTS])
      (Tile.ext fun idx => ?_))
    show some (dsF idx.1.val idx.2.1.val)
      = some (dsF idx.1.val idx.2.1.val
          + (if i_v = 0 then dzF idx.2.1.val else 0))
    rw [if_neg hv, add_zero]

/-! ## Backward block-sum and trip-count arithmetic -/

/-- Bottom-extension of an `Ico` sum by one `BTS` block. -/
private theorem pa_sum_Ico_add_block_bot (lo hi BTS : Nat) (h : lo + BTS ≤ hi)
    (f : Nat → ℝ) :
    ∑ t ∈ Finset.Ico lo hi, f t
      = (∑ c : Fin BTS, f (lo + c.val)) + ∑ t ∈ Finset.Ico (lo + BTS) hi, f t := by
  rw [← Finset.sum_Ico_consecutive _ (Nat.le_add_right lo BTS) h]
  congr 1
  rw [Finset.sum_Ico_eq_sum_range]
  simp only [Nat.add_sub_cancel_left]
  rw [Fin.sum_univ_eq_sum_range (fun c => f (lo + c))]

/-- Under the host's `BTL % BTS = 0`, the respelled trip count covers the
streamed range exactly: `pbTrip · BTS = pbNB − (i_c+1)·BTL`. -/
private theorem pb_trip_mul (T BTS i_c BTL : Nat) (hBTS : BTL % BTS = 0)
    (hBTSpos : 0 < BTS) :
    pbTrip T BTS i_c BTL * BTS = pbNB T BTS - (i_c + 1) * BTL := by
  obtain ⟨m, hm⟩ : BTS ∣ pbNB T BTS - (i_c + 1) * BTL :=
    Nat.dvd_sub (dvd_mul_left BTS ((T + BTS - 1) / BTS))
      (Dvd.dvd.mul_left (Nat.dvd_of_mod_eq_zero hBTS) (i_c + 1))
  have htrip : pbTrip T BTS i_c BTL = m := by
    show ((T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL + BTS - 1) / BTS = m
    have hm' : (T + BTS - 1) / BTS * BTS - (i_c + 1) * BTL = BTS * m := hm
    rw [hm']
    have h1 : BTS * m + BTS - 1 = BTS * m + (BTS - 1) := by omega
    rw [h1, Nat.mul_add_div hBTSpos, Nat.div_eq_of_lt (by omega), Nat.add_zero]
  rw [htrip, hm, Nat.mul_comm]

/-- Desc-loop `b_dk` spec step: the streamed block `[i, i+BTS)` joins the
accumulator at the **bottom**; above the diagonal chunk the causal keep is
identically true. -/
private theorem pbDkPart_step_bot (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i hi : Nat) (l e : Nat)
    (hlBTL : l < BTL)
    (hkeep : (i_c + 1) * BTL ≤ i) (hstep : i + BTS ≤ hi) :
    pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV i hi l e
      = (∑ c : Fin BTS,
          2 * pbDsVal s v do_ dz s_v_h s_v_t s_v_d i_bh i_c i_v scale T V BTL BV
              l (i + c.val)
            * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                l (i + c.val)
            * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e (i + c.val))
        + pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
            i_k i_v scale T K V BTL BTS BK BV (i + BTS) hi l e := by
  unfold pbDkPart
  rw [pa_sum_Ico_add_block_bot i hi BTS hstep]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [if_pos (by
    have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    omega)]

/-- Desc-loop `b_dv` spec step (bottom extension, keep identically true). -/
private theorem pbDvPart_step_bot (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i hi : Nat) (l p : Nat)
    (hlBTL : l < BTL)
    (hkeep : (i_c + 1) * BTL ≤ i) (hstep : i + BTS ≤ hi) :
    pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV i hi l p
      = (∑ c : Fin BTS,
          pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
              l (i + c.val)
            * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                l (i + c.val)
            * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p (i + c.val))
        + pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
            scale T K V BTL BTS BK BV (i + BTS) hi l p := by
  unfold pbDvPart
  rw [pa_sum_Ico_add_block_bot i hi BTS hstep]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [if_pos (by
    have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    omega)]

/-- Diag-loop `b_dv` spec step: masked kernel summands extend the diagonal
partial sum at the top, past the fixed desc-exit term `D`. -/
private theorem pbDvDiag_step' (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i : Nat) (D : ℝ) (l p : Nat)
    (hlo : i_c * BTL ≤ i) :
    (pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV (i_c * BTL) i l p + D)
      + ∑ c : Fin BTS,
          (if l ≤ c.val + (i - i_c * BTL) then
            pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                l (i + c.val)
              * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                  l (i + c.val)
          else 0)
          * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV p (i + c.val)
      = pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV (i_c * BTL) (i + BTS) l p + D := by
  rw [add_right_comm]
  congr 1
  unfold pbDvPart
  rw [pa_sum_Ico_add_block (i_c * BTL) i BTS hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (l ≤ c.val + (i - i_c * BTL)) ↔ (i_c * BTL + l ≤ i + c.val) := by
    omega
  by_cases h : i_c * BTL + l ≤ i + c.val
  · rw [if_pos h, if_pos (hcond.mpr h)]
  · rw [if_neg h, if_neg (fun hc => h (hcond.mp hc)), zero_mul]

/-- Diag-loop `b_dk` spec step: the doubly-masked `b_ds`/`b_s` kernel
summands realize the masked `pbDsVal`/`pbSVal` spec summands. -/
private theorem pbDkDiag_step' (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i : Nat) (D : ℝ) (l e : Nat)
    (hlo : i_c * BTL ≤ i) :
    (pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV (i_c * BTL) i l e + D)
      + ∑ c : Fin BTS,
          2 * ((if l ≤ c.val + (i - i_c * BTL) then
              (∑ p : Fin BV,
                pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
                  * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                      p.val (i + c.val))
                + (if i_v = 0 then pbDzGuarded s dz i_bh T (i + c.val) else 0)
            else 0) * scale)
            * (if l ≤ c.val + (i - i_c * BTL) then
                pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                  l (i + c.val)
              else 0)
            * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e (i + c.val)
      = pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k
          i_v scale T K V BTL BTS BK BV (i_c * BTL) (i + BTS) l e + D := by
  rw [add_right_comm]
  congr 1
  unfold pbDkPart
  rw [pa_sum_Ico_add_block (i_c * BTL) i BTS hlo]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  have hcond : (l ≤ c.val + (i - i_c * BTL)) ↔ (i_c * BTL + l ≤ i + c.val) := by
    omega
  by_cases h : i_c * BTL + l ≤ i + c.val
  · rw [if_pos (hcond.mpr h), if_pos (hcond.mpr h), if_pos h]
    rfl
  · rw [if_neg (fun hc => h (hcond.mp hc)), if_neg (fun hc => h (hcond.mp hc)),
      if_neg h]
    simp

/-- Fold the two-part exit sum (diag part + desc part) into `pbDkOut`. -/
private theorem pbDkOut_split (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (l e : Nat) :
    pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV (i_c * BTL) ((i_c + 1) * BTL) l e
      + pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k
          i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS) l e
      = pbDkOut s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k
          i_v scale T K V BTL BTS BK BV l e := by
  have hTripMul := pb_trip_mul T BTS i_c BTL hBTS hBTSpos
  unfold pbDkOut
  rcases le_or_gt ((i_c + 1) * BTL) (pbNB T BTS) with hle | hgt
  · have hlo : pbNB T BTS - pbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hlo, Nat.max_eq_left hle]
    have h1 : i_c * BTL ≤ (i_c + 1) * BTL := by
      have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
      omega
    unfold pbDkPart
    exact Finset.sum_Ico_consecutive _ h1 hle
  · have htrip0 : pbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero, Nat.max_eq_right (Nat.le_of_lt hgt)]
    unfold pbDkPart
    rw [Finset.Ico_self, Finset.sum_empty, add_zero]

/-- Fold the two-part exit sum into `pbDvOut`. -/
private theorem pbDvOut_split (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS) (l p : Nat) :
    pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV (i_c * BTL) ((i_c + 1) * BTL) l p
      + pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS) l p
      = pbDvOut s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV l p := by
  have hTripMul := pb_trip_mul T BTS i_c BTL hBTS hBTSpos
  unfold pbDvOut
  rcases le_or_gt ((i_c + 1) * BTL) (pbNB T BTS) with hle | hgt
  · have hlo : pbNB T BTS - pbTrip T BTS i_c BTL * BTS = (i_c + 1) * BTL := by
      omega
    rw [hlo, Nat.max_eq_left hle]
    have h1 : i_c * BTL ≤ (i_c + 1) * BTL := by
      have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
      omega
    unfold pbDvPart
    exact Finset.sum_Ico_consecutive _ h1 hle
  · have htrip0 : pbTrip T BTS i_c BTL * BTS = 0 := by omega
    rw [htrip0, Nat.sub_zero, Nat.max_eq_right (Nat.le_of_lt hgt)]
    unfold pbDvPart
    rw [Finset.Ico_self, Finset.sum_empty, add_zero]

/-! ## The backward streamed (descending, respelled ascending) loop -/

/-- Desc-loop `b_dk` register bridge: one bottom block joins, and the
kernel's raw gate/scale summand realizes the `pbDsVal`-based spec summand. -/
private theorem pbDkPart_desc_bridge (s : BlockState) (q k v do_ dz : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (X : Nat) (l e : Nat)
    (hlBTL : l < BTL) (hkeep : (i_c + 1) * BTL ≤ X)
    (hstep : X + BTS ≤ pbNB T BTS) :
    pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV X (pbNB T BTS) l e
      = pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k
          i_v scale T K V BTL BTS BK BV (X + BTS) (pbNB T BTS) l e
        + ∑ c : Fin BTS,
            2 * ((∑ p : Fin BV,
                pbVGuarded s v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
                  * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                      p.val (X + c.val)) * scale
              + (if i_v = 0 then pbDzGuarded s dz i_bh T (X + c.val) * scale
                else 0))
              * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                  l (X + c.val)
              * pbQGuarded s q s_k_h s_k_t s_k_d i_bh i_k T K BK e (X + c.val) := by
  rw [pbDkPart_step_bot s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
    i_bh i_c i_k i_v scale T K V BTL BTS BK BV X (pbNB T BTS) l e hlBTL hkeep
    hstep, add_comm]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  unfold pbDsVal
  rw [add_mul, ite_mul, zero_mul]

/-- Desc-loop `b_dv` register bridge: one bottom block joins. -/
private theorem pbDvPart_desc_bridge (s : BlockState) (q k do_ : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (i_bh i_c i_k i_v : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (X : Nat) (l p : Nat)
    (hlBTL : l < BTL) (hkeep : (i_c + 1) * BTL ≤ X)
    (hstep : X + BTS ≤ pbNB T BTS) :
    pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
        scale T K V BTL BTS BK BV X (pbNB T BTS) l p
      = pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV (X + BTS) (pbNB T BTS) l p
        + ∑ c : Fin BTS,
            pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                l (X + c.val)
              * pbSVal s q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                  l (X + c.val)
              * pbDoGuarded s do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                  p (X + c.val) := by
  rw [pbDvPart_step_bot s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
    i_bh i_c i_k i_v scale T K V BTL BTS BK BV X (pbNB T BTS) l p hlBTL hkeep
    hstep, add_comm]

/-- The backward streamed-loop invariant at counter `j`: the accumulators
hold the top `j` blocks `[pbNB − j·BTS, pbNB)` of the key sweep. -/
def pbInvD (s0 : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (j : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem ∧ s.undef = s0.undef ∧
  j ≤ pbTrip T BTS i_c BTL ∧
  s.regs .real [BTL, BK] "b_k"
    = some (pbKTile s0 k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK) ∧
  s.regs .real [BTL, BV] "b_v"
    = some (pbVTile s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV) ∧
  s.regs .real [BTL, BK] "b_dk"
    = some ⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h
        s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (pbNB T BTS - j * BTS) (pbNB T BTS) idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL, BV] "b_dv"
    = some ⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h
        s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (pbNB T BTS - j * BTS) (pbNB T BTS) idx.1.val idx.2.1.val)⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The backward prologue** (statements 1–6): the fixed chunk pointers and
loads plus the zero accumulators establish `pbInvD` at `0`. -/
theorem paBwdPrologue_run (s : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) :
    ∃ sP, stepStmts
        [ Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
              [T, K] [BTL, BK] [s_k_t, s_k_d]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
              [T, V] [BTL, BV] [s_v_t, s_v_d]
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
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)) ] s
        = some sP
      ∧ pbInvD s q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV 0 sP := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval k _ _ _ _ [T, K] [BTL, BK] [s_k_t, s_k_d]
      (i_bh * s_k_h) (i_c * BTL) (i_k * BK)
      (pb_constMul_eval _ i_bh s_k_h)
      (pb_constMul_eval _ i_c BTL)
      (pb_constMul_eval _ i_k BK)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BTL, BV] [s_v_t, s_v_d]
      (i_bh * s_v_h) (i_c * BTL) (i_v * BV)
      (pb_constMul_eval _ i_bh s_v_h)
      (pb_constMul_eval _ i_c BTL)
      (pb_constMul_eval _ i_v BV)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_kLoad_eq s _ k "p_k" s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK
      (by rfl) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_vLoad_eq s _ v "p_v" s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
      (by rfl) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pa_zeros_eval [BTL, BK] _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pa_zeros_eval [BTL, BV] _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, rfl, rfl, Nat.zero_le _, by simp, by simp, ?_, ?_⟩
  · rw [show (⟨fun idx => some (pbDkPart s q k v do_ dz s_k_h s_k_t s_k_d s_v_h
          s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - 0 * BTS) (pbNB T BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BK])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [pbDkPart]]
    simp
  · rw [show (⟨fun idx => some (pbDvPart s q k do_ s_k_h s_k_t s_k_d s_v_h
          s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - 0 * BTS) (pbNB T BTS) idx.1.val idx.2.1.val)⟩
          : Tile .real [BTL, BV])
        = ⟨fun _ => some (0 : ℝ)⟩
      from Tile.ext fun idx => by simp [pbDvPart]]
    simp

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One streamed (above-diagonal) block** of the backward loop. -/
theorem paBwdDesc_step (s0 sin : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (j : Nat)
    (hundef : ∀ rg off, s0.undef rg off = 0)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hlt : j < pbTrip T BTS i_c BTL)
    (hInv : pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h
      s_v_t s_v_d scale T K V BTL BTS BK BV j sin) :
    ∃ s', stepStmts (paBwdDescBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
        s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV)
        (sin.setReg "j" .nat [] (Tile.scalar j)) = some s'
      ∧ pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV (j + 1) s' := by
  obtain ⟨hmem, hund, hjle, hbk, hbv, hbdk, hbdv⟩ := hInv
  have hTripMul := pb_trip_mul T BTS i_c BTL hBTS hBTSpos
  have hj1 : (j + 1) * BTS ≤ pbTrip T BTS i_c BTL * BTS :=
    Nat.mul_le_mul_right BTS hlt
  have hexp1 : (j + 1) * BTS = j * BTS + BTS := by ring
  have hkeep : (i_c + 1) * BTL ≤ pbNB T BTS - BTS - j * BTS := by omega
  have hiadd : pbNB T BTS - BTS - j * BTS + BTS = pbNB T BTS - j * BTS := by
    omega
  have hlonew : pbNB T BTS - (j + 1) * BTS = pbNB T BTS - BTS - j * BTS := by
    omega
  unfold paBwdDescBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_iAssign_eval _ T BTS j (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval q _ _ _ _ [K, T] [BK, BTS] [s_k_d, s_k_t]
      (i_bh * s_k_h) (i_k * BK) (pbNB T BTS - BTS - j * BTS)
      (pb_constMul_eval _ i_bh s_k_h)
      (pb_constMul_eval _ i_k BK)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval do_ _ _ _ _ [V, T] [BV, BTS] [s_v_d, s_v_t]
      (i_bh * s_v_h) (i_v * BV) (pbNB T BTS - BTS - j * BTS)
      (pb_constMul_eval _ i_bh s_v_h)
      (pb_constMul_eval _ i_v BV)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dzPtr_eval _ dz i_bh T BTS (pbNB T BTS - BTS - j * BTS) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_qLoad_eq s0 _ q "p_q" s_k_h s_k_t s_k_d i_bh i_k T K BK BTS
      (pbNB T BTS - BTS - j * BTS) (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_doLoad_eq s0 _ do_ "p_do" s_v_h s_v_t s_v_d i_bh i_v T V BV BTS
      (pbNB T BTS - BTS - j * BTS) (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dzLoad_eq s0 _ dz i_bh T BTS (pbNB T BTS - BTS - j * BTS)
      (by simpa using hmem) (by simpa using hund) hundef (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_sScaled_eval s0 _ q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
      BTS (pbNB T BTS - BTS - j * BTS) (by simpa using hbk) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_square_eval _ BTL BTS
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
        l (pbNB T BTS - BTS - j * BTS + c))
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dvAdd_eval _ BTL BTS BV
      (fun l p => pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
        i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (pbNB T BTS - j * BTS) (pbNB T BTS) l p)
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
          l (pbNB T BTS - BTS - j * BTS + c)
        * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
            l (pbNB T BTS - BTS - j * BTS + c))
      (fun p c => pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
        p (pbNB T BTS - BTS - j * BTS + c))
      (by simpa using hbdv) (by simp) (by simp [pbDoTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dsScaled_eval s0 _ v do_ s_v_h s_v_t s_v_d i_bh i_c i_v scale
      T V BTL BV BTS (pbNB T BTS - BTS - j * BTS)
      (by simpa using hbv) (by simp)))]
  rw [stepStmts.cons_some
    (pb_gateScaled_step _ i_v scale BTL BTS
      (fun l c => (∑ p : Fin BV,
          pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
            * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                p.val (pbNB T BTS - BTS - j * BTS + c)) * scale)
      (fun c => pbDzGuarded s0 dz i_bh T (pbNB T BTS - BTS - j * BTS + c))
      (by simp) (by simp [pbDzTile]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dkAdd_eval _ BTL BTS BK
      (fun l e => pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
        i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (pbNB T BTS - j * BTS) (pbNB T BTS) l e)
      (fun l c => (∑ p : Fin BV,
          pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
            * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                p.val (pbNB T BTS - BTS - j * BTS + c)) * scale
        + (if i_v = 0 then
            pbDzGuarded s0 dz i_bh T (pbNB T BTS - BTS - j * BTS + c) * scale
          else 0))
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
        l (pbNB T BTS - BTS - j * BTS + c))
      (fun e c => pbQGuarded s0 q s_k_h s_k_t s_k_d i_bh i_k T K BK
        e (pbNB T BTS - BTS - j * BTS + c))
      (by simpa using hbdk) (by simp) (by simp) (by simp [pbQTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hmem
  · simpa using hund
  · omega
  · simpa using hbk
  · simpa using hbv
  · rw [show (⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - (j + 1) * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h
            s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - j * BTS) (pbNB T BTS) idx.1.val idx.2.1.val
          + ∑ c : Fin BTS,
              2 * ((∑ p : Fin BV,
                  pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
                    idx.1.val p.val
                    * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                        p.val (pbNB T BTS - BTS - j * BTS + c.val)) * scale
                + (if i_v = 0 then
                    pbDzGuarded s0 dz i_bh T
                      (pbNB T BTS - BTS - j * BTS + c.val) * scale
                  else 0))
                * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                    idx.1.val (pbNB T BTS - BTS - j * BTS + c.val)
                * pbQGuarded s0 q s_k_h s_k_t s_k_d i_bh i_k T K BK
                    idx.2.1.val (pbNB T BTS - BTS - j * BTS + c.val))⟩
      from Tile.ext fun idx => congrArg some (by
        rw [hlonew, pbDkPart_desc_bridge s0 q k v do_ dz s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - BTS - j * BTS) idx.1.val idx.2.1.val idx.1.isLt hkeep
          (by omega), hiadd])]
    simp
  · rw [show (⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h
          s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - (j + 1) * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t
            s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - j * BTS) (pbNB T BTS) idx.1.val idx.2.1.val
          + ∑ c : Fin BTS,
              pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                  idx.1.val (pbNB T BTS - BTS - j * BTS + c.val)
                * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                    idx.1.val (pbNB T BTS - BTS - j * BTS + c.val)
                * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                    idx.2.1.val (pbNB T BTS - BTS - j * BTS + c.val))⟩
      from Tile.ext fun idx => congrArg some (by
        rw [hlonew, pbDvPart_desc_bridge s0 q k do_ s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - BTS - j * BTS) idx.1.val idx.2.1.val idx.1.isLt hkeep
          (by omega), hiadd])]
    simp

set_option maxHeartbeats 4000000 in
/-- **The full streamed loop**: the respelled descending loop consumes all
`pbTrip` above-diagonal blocks. -/
theorem paBwdDesc_run (s0 sPre : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hundef : ∀ rg off, s0.undef rg off = 0)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h
      s_v_t s_v_d scale T K V BTL BTS BK BV 0 sPre) :
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
        (paBwdDescBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV)) sPre = some sL
      ∧ pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV (pbTrip T BTS i_c BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "j")
      (P := fun j s => pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t
        s_k_d s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV j s)
      (evalOp_constNat 0 sPre)
      (pb_trip_eval sPre T BTS i_c BTL)
      (evalOp_constNat 1 sPre)
      one_ne_zero
      hInv
      (fun j s hj hPj =>
        paBwdDesc_step s0 s q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV j hundef hBTS hBTSpos
          hj hPj)
  have hfin : final = pbTrip T BTS i_c BTL := le_antisymm hP.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-! ## The backward diagonal loop -/

/-- The backward diagonal-loop invariant at counter `i` (absolute key offset
within the diagonal chunk); the accumulators hold the diagonal partial sums
past the fixed desc-exit terms. -/
def pbInvG (s0 : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem ∧ s.undef = s0.undef ∧
  i_c * BTL ≤ i ∧ BTS ∣ (i - i_c * BTL) ∧ i ≤ (i_c + 1) * BTL ∧
  s.regs .real [BTL, BK] "b_k"
    = some (pbKTile s0 k s_k_h s_k_t s_k_d i_bh i_c i_k T K BTL BK) ∧
  s.regs .real [BTL, BV] "b_v"
    = some (pbVTile s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV) ∧
  s.regs .nat [BTS] "o_q"
    = some (Tile.vec fun r => r.val + (i - i_c * BTL)) ∧
  s.regs .nat [BTL] "o_k" = some (Tile.vec fun r => r.val) ∧
  s.regs .real [BTL, BK] "b_dk"
    = some ⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h
        s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (i_c * BTL) i idx.1.val idx.2.1.val
      + pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
          i_k i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩ ∧
  s.regs .real [BTL, BV] "b_dv"
    = some ⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t
        s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
        (i_c * BTL) i idx.1.val idx.2.1.val
      + pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c i_k
          i_v scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The mid-section**: barrier no-op and the `o_q`/`o_k` aranges carry the
desc-exit invariant into the diagonal invariant at `i = i_c·BTL`. -/
theorem paBwdMid_run (s0 sL1 : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hInv : pbInvD s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h
      s_v_t s_v_d scale T K V BTL BTS BK BV (pbTrip T BTS i_c BTL) sL1) :
    ∃ sM, stepStmts
        [ Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL) ] sL1 = some sM
      ∧ pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV (i_c * BTL) sM := by
  obtain ⟨hmem, hund, hjle, hbk, hbv, hbdk, hbdv⟩ := hInv
  rw [stepStmts.cons_some (pa_ifThen_false_noop [] sL1)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTS sL1))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BTL _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, le_rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hmem
  · simpa using hund
  · simp
  · have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    omega
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
  · rw [show (⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val
        + pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h
            s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩
      from Tile.ext fun idx => congrArg some (by
        rw [show pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
            i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val = 0 from by
          simp [pbDkPart], zero_add])]
    simpa using hbdk
  · rw [show (⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val
        + pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h
            s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩
      from Tile.ext fun idx => congrArg some (by
        rw [show pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
            i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (i_c * BTL) (i_c * BTL) idx.1.val idx.2.1.val = 0 from by
          simp [pbDvPart], zero_add])]
    simpa using hbdv

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 8000 in
/-- **One diagonal (causally masked) block** of the backward loop. -/
theorem paBwdDiag_step (s0 sin : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat) (i : Nat)
    (hundef : ∀ rg off, s0.undef rg off = 0)
    (hBTS : BTL % BTS = 0)
    (hlt : i < (i_c + 1) * BTL)
    (hInv : pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h
      s_v_t s_v_d scale T K V BTL BTS BK BV i sin) :
    ∃ s', stepStmts (paBwdDiagBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
        s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV)
        (sin.setReg "i" .nat [] (Tile.scalar i)) = some s'
      ∧ pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV (i + BTS) s' := by
  obtain ⟨hmem, hund, hlo, hdvd, hle, hbk, hbv, hoq, hok, hbdk, hbdv⟩ := hInv
  unfold paBwdDiagBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval q _ _ _ _ [K, T] [BK, BTS] [s_k_d, s_k_t]
      (i_bh * s_k_h) (i_k * BK) i
      (pb_constMul_eval _ i_bh s_k_h)
      (pb_constMul_eval _ i_k BK)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval do_ _ _ _ _ [V, T] [BV, BTS] [s_v_d, s_v_t]
      (i_bh * s_v_h) (i_v * BV) i
      (pb_constMul_eval _ i_bh s_v_h)
      (pb_constMul_eval _ i_v BV)
      (by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dzPtr_eval _ dz i_bh T BTS i (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_qLoad_eq s0 _ q "p_q" s_k_h s_k_t s_k_d i_bh i_k T K BK BTS i
      (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_doLoad_eq s0 _ do_ "p_do" s_v_h s_v_t s_v_d i_bh i_v T V BV BTS i
      (by simpa using hmem) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dzLoad_eq s0 _ dz i_bh T BTS i
      (by simpa using hmem) (by simpa using hund) hundef (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_msLe_eval _ BTL BTS (i - i_c * BTL)
      (by simpa using hok) (by simpa using hoq)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_sScaled_eval s0 _ q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
      BTS i (by simpa using hbk) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_square_eval _ BTL BTS
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
        l (i + c))
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_whereMask_eval _ BTL BTS (i - i_c * BTL) "b_s"
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
        l (i + c))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_whereMask_eval _ BTL BTS (i - i_c * BTL) "b_s2"
      (fun l c => pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
          l (i + c)
        * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
            l (i + c))
      (by simp) (by simp)))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dsDot_eval s0 _ v do_ s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV BTS i
      (by simpa using hbv) (by simp)))]
  rw [stepStmts.cons_some
    (pb_gateDiag_step _ i_v BTL BTS
      (fun l c => ∑ p : Fin BV,
          pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
            * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                p.val (i + c))
      (fun c => pbDzGuarded s0 dz i_bh T (i + c))
      (by simp) (by simp [pbDzTile]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_whereMaskScale_eval _ BTL BTS (i - i_c * BTL) scale
      (fun l c => (∑ p : Fin BV,
          pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
            * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                p.val (i + c))
        + (if i_v = 0 then pbDzGuarded s0 dz i_bh T (i + c) else 0))
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dvAdd_eval _ BTL BTS BV
      (fun l p => pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          i_bh i_c i_k i_v scale T K V BTL BTS BK BV (i_c * BTL) i l p
        + pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
            i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS) l p)
      (fun l c => if l ≤ c + (i - i_c * BTL) then
          pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
            l (i + c)
          * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
              l (i + c)
        else 0)
      (fun p c => pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
        p (i + c))
      (by simpa using hbdv) (by simp) (by simp [pbDoTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_dkAdd_eval _ BTL BTS BK
      (fun l e => pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          i_bh i_c i_k i_v scale T K V BTL BTS BK BV (i_c * BTL) i l e
        + pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS) l e)
      (fun l c => (if l ≤ c + (i - i_c * BTL) then
          (∑ p : Fin BV,
            pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV l p.val
              * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                  p.val (i + c))
            + (if i_v = 0 then pbDzGuarded s0 dz i_bh T (i + c) else 0)
        else 0) * scale)
      (fun l c => if l ≤ c + (i - i_c * BTL) then
          pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
            l (i + c)
        else 0)
      (fun e c => pbQGuarded s0 q s_k_h s_k_t s_k_d i_bh i_k T K BK
        e (i + c))
      (by simpa using hbdk) (by simp) (by simp) (by simp [pbQTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pb_oqAdd_eval _ BTS (i - i_c * BTL) (by simpa using hoq)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact hmem
  · exact hund
  · omega
  · have h1 : i + BTS - i_c * BTL = (i - i_c * BTL) + BTS := by omega
    rw [h1]
    exact Dvd.dvd.add hdvd dvd_rfl
  · have hexp : (i_c + 1) * BTL = i_c * BTL + BTL := by ring
    have hBTLd : BTS ∣ BTL := Nat.dvd_of_mod_eq_zero hBTS
    have h2 : i - i_c * BTL < BTL := by omega
    have h3 : (i - i_c * BTL) + BTS ≤ BTL := pa_dvd_step_le hdvd hBTLd h2
    omega
  · exact hbk
  · exact hbv
  · rw [show ((Tile.vec fun r : Fin BTS => r.val + (i + BTS - i_c * BTL))
          : Tile .nat [BTS])
        = ((Tile.vec fun r : Fin BTS => r.val + ((i - i_c * BTL) + BTS))
          : Tile .nat [BTS])
      from Tile.ext fun r => by
        show r.1.val + (i + BTS - i_c * BTL) = r.1.val + ((i - i_c * BTL) + BTS)
        omega]
    simp
  · exact hok
  · rw [show (⟨fun idx => some (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (i_c * BTL) (i + BTS) idx.1.val idx.2.1.val
        + pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BK])
        = ⟨fun idx => some ((pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h
            s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (i_c * BTL) i idx.1.val idx.2.1.val
          + pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
              i_c i_k i_v scale T K V BTL BTS BK BV
              (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
              idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              2 * ((if idx.1.val ≤ c.val + (i - i_c * BTL) then
                  (∑ p : Fin BV,
                    pbVGuarded s0 v s_v_h s_v_t s_v_d i_bh i_c i_v T V BTL BV
                      idx.1.val p.val
                      * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                          p.val (i + c.val))
                    + (if i_v = 0 then pbDzGuarded s0 dz i_bh T (i + c.val)
                      else 0)
                else 0) * scale)
                * (if idx.1.val ≤ c.val + (i - i_c * BTL) then
                    pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale
                      T K BTL BK idx.1.val (i + c.val)
                  else 0)
                * pbQGuarded s0 q s_k_h s_k_t s_k_d i_bh i_k T K BK
                    idx.2.1.val (i + c.val))⟩
      from Tile.ext fun idx => congrArg some
        (pbDkDiag_step' s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          i_bh i_c i_k i_v scale T K V BTL BTS BK BV i
          (pbDkPart s0 q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)
          idx.1.val idx.2.1.val hlo).symm]
    simp
  · rw [show (⟨fun idx => some (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
          (i_c * BTL) (i + BTS) idx.1.val idx.2.1.val
        + pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)⟩ : Tile .real [BTL, BV])
        = ⟨fun idx => some ((pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h
            s_v_t s_v_d i_bh i_c i_k i_v scale T K V BTL BTS BK BV
            (i_c * BTL) i idx.1.val idx.2.1.val
          + pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
              i_c i_k i_v scale T K V BTL BTS BK BV
              (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
              idx.1.val idx.2.1.val)
          + ∑ c : Fin BTS,
              (if idx.1.val ≤ c.val + (i - i_c * BTL) then
                pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale T K BTL BK
                    idx.1.val (i + c.val)
                  * pbSVal s0 q k s_k_h s_k_t s_k_d i_bh i_c i_k scale
                      T K BTL BK idx.1.val (i + c.val)
              else 0)
              * pbDoGuarded s0 do_ s_v_h s_v_t s_v_d i_bh i_v T V BV
                  idx.2.1.val (i + c.val))⟩
      from Tile.ext fun idx => congrArg some
        (pbDvDiag_step' s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          i_bh i_c i_k i_v scale T K V BTL BTS BK BV i
          (pbDvPart s0 q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
            i_c i_k i_v scale T K V BTL BTS BK BV
            (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
            idx.1.val idx.2.1.val)
          idx.1.val idx.2.1.val hlo).symm]
    simp

set_option maxHeartbeats 4000000 in
/-- **The full diagonal loop.** -/
theorem paBwdDiag_run (s0 sPre : BlockState) (q k v do_ dz : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat)
    (scale : ℝ) (T K V BTL BTS BK BV : Nat)
    (hundef : ∀ rg off, s0.undef rg off = 0)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInv : pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h
      s_v_t s_v_d scale T K V BTL BTS BK BV (i_c * BTL) sPre) :
    ∃ sL, stepStmt (Stmt.forRangeDyn "i"
        (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
        (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
        (paBwdDiagBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV)) sPre = some sL
      ∧ pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d s_v_h s_v_t
          s_v_d scale T K V BTL BTS BK BV ((i_c + 1) * BTL) sL := by
  obtain ⟨final, sL, hrun, hge, hP⟩ :=
    forRangeDyn_inv (idx := "i")
      (P := fun i s => pbInvG s0 q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t
        s_k_d s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV i s)
      (pb_constMul_eval sPre i_c BTL)
      (evalOp_constNat ((i_c + 1) * BTL) sPre)
      (evalOp_constNat BTS sPre)
      hBTSpos.ne'
      hInv
      (fun i s hi hPi =>
        paBwdDiag_step s0 s q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t s_k_d
          s_v_h s_v_t s_v_d scale T K V BTL BTS BK BV i hundef hBTS hi hPi)
  have hfin : final = (i_c + 1) * BTL := le_antisymm hP.2.2.2.2.1 hge
  exact ⟨sL, hrun, hfin ▸ hP⟩

/-! ## Backward stores and ★ main theorem -/

/-- The `dk` store address at lane `(l, e)`. -/
def pbDkOffset (i_bh i_c i_k i_v B H s_k_h s_k_t s_k_d BTL BK : Nat)
    (idx : TileIndex [BTL, BK]) : Nat :=
  (i_bh + B * H * i_v) * s_k_h + (i_c * BTL + idx.1.val) * s_k_t
    + (i_k * BK + idx.2.1.val) * s_k_d

/-- A `dk` store lane is *active* when it maps inside the `T × K` window. -/
def pbDkActive (i_c i_k T K BTL BK : Nat) (idx : TileIndex [BTL, BK]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_k * BK + idx.2.1.val < K

/-- The `dv` store address at lane `(l, p)`. -/
def pbDvOffset (i_bh i_c i_k i_v B H s_v_h s_v_t s_v_d BTL BV : Nat)
    (idx : TileIndex [BTL, BV]) : Nat :=
  (i_bh + B * H * i_k) * s_v_h + (i_c * BTL + idx.1.val) * s_v_t
    + (i_v * BV + idx.2.1.val) * s_v_d

/-- A `dv` store lane is *active* when it maps inside the `T × V` window. -/
def pbDvActive (i_c i_v T V BTL BV : Nat) (idx : TileIndex [BTL, BV]) : Prop :=
  i_c * BTL + idx.1.val < T ∧ i_v * BV + idx.2.1.val < V

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The backward epilogue**: `p_dk`/`p_dv` makes and the two block stores,
from the diag-exit invariant, with both readbacks folded to the full-sweep
closed forms. -/
theorem paBwdStores_run (s sL2 : BlockState) (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v : Nat) (s_k_h s_k_t s_v_h s_v_t : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv) (hσk : BK ≤ s_k_t) (hσv : BV ≤ s_v_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hInvF : pbInvG s q k v do_ dz i_bh i_c i_k i_v s_k_h s_k_t 1 s_v_h s_v_t 1
      scale T K V BTL BTS BK BV ((i_c + 1) * BTL) sL2) :
    ∃ sF, stepStmts
        [ Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_k_h))
              [T, K] [BTL, BK] [s_k_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_v_h))
              [T, V] [BTL, BV] [s_v_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.store .real [BTL, BK]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk") [0, 1])
            (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv") [0, 1])
            (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ] sL2 = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], pbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (pbDkOffset i_bh i_c i_k i_v B H s_k_h s_k_t 1 BTL BK idx)
            = pbDkOut s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k
                i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], pbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (pbDvOffset i_bh i_c i_k i_v B H s_v_h s_v_t 1 BTL BV idx)
            = pbDvOut s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k i_v
                scale T K V BTL BTS BK BV idx.1.val idx.2.1.val) := by
  obtain ⟨hFmem, hFund, hFlo, hFdvd, hFle, hFbk, hFbv, hFoq, hFok, hFbdk,
    hFbdv⟩ := hInvF
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval dk _ _ _ _ [T, K] [BTL, BK] [s_k_t, 1]
      ((i_bh + B * H * i_v) * s_k_h) (i_c * BTL) (i_k * BK)
      (evalOp_constNat _ _)
      (pb_constMul_eval _ i_c BTL)
      (pb_constMul_eval _ i_k BK)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (pa_makeBlockPtr_2d_eval dv _ _ _ _ [T, V] [BTL, BV] [s_v_t, 1]
      ((i_bh + B * H * i_k) * s_v_h) (i_c * BTL) (i_v * BV)
      (evalOp_constNat _ _)
      (pb_constMul_eval _ i_c BTL)
      (pb_constMul_eval _ i_v BV)))]
  rw [stepStmts.cons_some (paStore_step_eq _ dk "b_dk" "p_dk"
    ((i_bh + B * H * i_v) * s_k_h) s_k_t (i_c * BTL) (i_k * BK) T K BTL BK
    (fun l e => pbDkOut s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c
      i_k i_v scale T K V BTL BTS BK BV l e)
    ⟨fun idx => some (pbDkPart s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh
        i_c i_k i_v scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) idx.1.val idx.2.1.val
      + pbDkPart s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩
    (fun l e => congrArg some
      (pbDkOut_split s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k
        i_v scale T K V BTL BTS BK BV hBTS hBTSpos l.val e.val))
    (by simpa using hFbdk)
    (by simp))]
  obtain ⟨hKpids, hKregs, hKread, hKother, hKuntouched⟩ := paStore_step_props
    ((sL2.setReg "p_dk" .blockPtr [BTL, BK]
        ⟨fun _ => BlockPtr.mk dk ((i_bh + B * H * i_v) * s_k_h) [T, K]
          [BTL, BK] [s_k_t, 1] [i_c * BTL, i_k * BK]⟩).setReg
      "p_dv" .blockPtr [BTL, BV]
        ⟨fun _ => BlockPtr.mk dv ((i_bh + B * H * i_k) * s_v_h) [T, V]
          [BTL, BV] [s_v_t, 1] [i_c * BTL, i_v * BV]⟩)
    dk ((i_bh + B * H * i_v) * s_k_h) s_k_t (i_c * BTL) (i_k * BK) T K BTL BK
    (fun l e => pbDkOut s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c
      i_k i_v scale T K V BTL BTS BK BV l e)
    (paStoreAddr_injective _ _ _ _ BTL BK hσk)
  rw [stepStmts.cons_some (paStore_step_eq _ dv "b_dv" "p_dv"
    ((i_bh + B * H * i_k) * s_v_h) s_v_t (i_c * BTL) (i_v * BV) T V BTL BV
    (fun l p => pbDvOut s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k i_v
      scale T K V BTL BTS BK BV l p)
    ⟨fun idx => some (pbDvPart s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh
        i_c i_k i_v scale T K V BTL BTS BK BV
        (i_c * BTL) ((i_c + 1) * BTL) idx.1.val idx.2.1.val
      + pbDvPart s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k i_v
          scale T K V BTL BTS BK BV
          (pbNB T BTS - pbTrip T BTS i_c BTL * BTS) (pbNB T BTS)
          idx.1.val idx.2.1.val)⟩
    (fun l p => congrArg some
      (pbDvOut_split s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k
        i_v scale T K V BTL BTS BK BV hBTS hBTSpos l.val p.val))
    (by rw [hKregs]; simpa using hFbdv)
    (by rw [hKregs]; simp))]
  rw [stepStmts.nil]
  obtain ⟨hVpids, hVregs, hVread, hVother, hVuntouched⟩ := paStore_step_props
    (paStoreState
      ((sL2.setReg "p_dk" .blockPtr [BTL, BK]
          ⟨fun _ => BlockPtr.mk dk ((i_bh + B * H * i_v) * s_k_h) [T, K]
            [BTL, BK] [s_k_t, 1] [i_c * BTL, i_k * BK]⟩).setReg
        "p_dv" .blockPtr [BTL, BV]
          ⟨fun _ => BlockPtr.mk dv ((i_bh + B * H * i_k) * s_v_h) [T, V]
            [BTL, BV] [s_v_t, 1] [i_c * BTL, i_v * BV]⟩)
      dk ((i_bh + B * H * i_v) * s_k_h) s_k_t (i_c * BTL) (i_k * BK) T K BTL BK
      (fun l e => pbDkOut s q k v do_ dz s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c
        i_k i_v scale T K V BTL BTS BK BV l e))
    dv ((i_bh + B * H * i_k) * s_v_h) s_v_t (i_c * BTL) (i_v * BV) T V BTL BV
    (fun l p => pbDvOut s q k do_ s_k_h s_k_t 1 s_v_h s_v_t 1 i_bh i_c i_k i_v
      scale T K V BTL BTS BK BV l p)
    (paStoreAddr_injective _ _ _ _ BTL BV hσv)
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
causal rebased-attention gradient closed forms.**

For every scalar-argument tuple `(i_bh, i_c, i_k, i_v, i_h)` the shell would
pass (universally quantified binders; `i_h` is unused by the helper),
executing the full backward dk/dv surface succeeds, the `dk` block store
holds `Σ_{t ≥ i_c·BTL + l} 2·ds(l,t)·score(l,t)·q[e,t]` and the `dv` block
store holds `Σ_{t ≥ i_c·BTL + l} score(l,t)²·do[p,t]` at every in-window
lane — the causal key sweep from the diagonal row to the streamed top
`cdiv(T,BTS)·BTS` (loads beyond `T` read as zero, so ragged tails are
exact), with the `i_v = 0` gate folding the normalizer gradient `dz` into
`ds`.

Side conditions: `dk ≠ dv` (distinct output buffers), the host's contiguous
last-dim strides `s_k_d = 1` / `s_v_d = 1` with `BK ≤ s_k_t` / `BV ≤ s_v_t`
(store-lane injectivity), the host's own `assert BTL % BTS == 0`, and a
clean-input state (`undef ≡ 0`, the masked `dz` load's off-lanes). -/
specification pa_bwd_dkv_exec_genuine
    (s : BlockState) (q k v do_ dz dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat)
    (hDkDv : dk ≠ dv)
    (hSkd : s_k_d = 1) (hSvd : s_v_d = 1)
    (hσk : BK ≤ s_k_t) (hσv : BV ≤ s_v_t)
    (hBTS : BTL % BTS = 0) (hBTSpos : 0 < BTS)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    ∃ sF, exec (pa_bwd_dkv_surface q k v do_ dz dk dv i_bh i_c i_k i_v i_h
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale
        B H T K V BTL BTS BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BTL, BK], pbDkActive i_c i_k T K BTL BK idx →
          sF.readMem dk
              (pbDkOffset i_bh i_c i_k i_v B H s_k_h s_k_t s_k_d BTL BK idx)
            = pbDkOut s q k v do_ dz s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh
                i_c i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val)
      ∧ (∀ idx : TileIndex [BTL, BV], pbDvActive i_c i_v T V BTL BV idx →
          sF.readMem dv
              (pbDvOffset i_bh i_c i_k i_v B H s_v_h s_v_t s_v_d BTL BV idx)
            = pbDvOut s q k do_ s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d i_bh i_c
                i_k i_v scale T K V BTL BTS BK BV idx.1.val idx.2.1.val) := by
  subst hSkd
  subst hSvd
  obtain ⟨sP, hPro, hInvP⟩ := paBwdPrologue_run s q k v do_ dz i_bh i_c i_k i_v
    s_k_h s_k_t 1 s_v_h s_v_t 1 scale T K V BTL BTS BK BV
  obtain ⟨sL1, hL1, hInvDF⟩ := paBwdDesc_run s sP q k v do_ dz i_bh i_c i_k i_v
    s_k_h s_k_t 1 s_v_h s_v_t 1 scale T K V BTL BTS BK BV hundef hBTS hBTSpos
    hInvP
  obtain ⟨sM, hMid, hInvG0⟩ := paBwdMid_run s sL1 q k v do_ dz i_bh i_c i_k i_v
    s_k_h s_k_t 1 s_v_h s_v_t 1 scale T K V BTL BTS BK BV hInvDF
  obtain ⟨sL2, hL2, hInvGF⟩ := paBwdDiag_run s sM q k v do_ dz i_bh i_c i_k i_v
    s_k_h s_k_t 1 s_v_h s_v_t 1 scale T K V BTL BTS BK BV hundef hBTS hBTSpos
    hInvG0
  rw [exec, pa_bwd_body_eq]
  rw [show ([ Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
              [T, K] [BTL, BK] [s_k_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
              [T, V] [BTL, BV] [s_v_t, 1]
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
          Stmt.forRangeDyn "j" (Op.constNat 0)
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
            (paBwdDescBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t 1
              s_v_h s_v_t 1 scale T K V BTL BTS BK BV),
          Stmt.ifThen (Op.constBool Bool.false) [],
          Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
          Stmt.assign .nat [BTL] "o_k" (Op.arange BTL),
          Stmt.forRangeDyn "i"
            (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
            (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
            (paBwdDiagBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t 1
              s_v_h s_v_t 1 scale T K V BTL BTS BK BV),
          Stmt.assign .blockPtr [BTL, BK] "p_dk"
            (Op.makeBlockPtrDynOffsets dk
              (Op.constNat ((i_bh + B * H * i_v) * s_k_h))
              [T, K] [BTL, BK] [s_k_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_dv"
            (Op.makeBlockPtrDynOffsets dv
              (Op.constNat ((i_bh + B * H * i_k) * s_v_h))
              [T, V] [BTL, BV] [s_v_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_v) (Op.constNat BV)]),
          Stmt.store .real [BTL, BK]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BK] "p_dk") [0, 1])
            (Op.ref .real [BTL, BK] "b_dk") MaskOpt.none,
          Stmt.store .real [BTL, BV]
            (MemAccess.blockPtr (Op.ref .blockPtr [BTL, BV] "p_dv") [0, 1])
            (Op.ref .real [BTL, BV] "b_dv") MaskOpt.none ] : List Stmt)
      = [ Stmt.assign .blockPtr [BTL, BK] "p_k"
            (Op.makeBlockPtrDynOffsets k
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_k_h))
              [T, K] [BTL, BK] [s_k_t, 1]
              [Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL),
                Op.mul .nat Broadcast.nil (Op.constNat i_k) (Op.constNat BK)]),
          Stmt.assign .blockPtr [BTL, BV] "p_v"
            (Op.makeBlockPtrDynOffsets v
              (Op.mul .nat Broadcast.nil (Op.constNat i_bh) (Op.constNat s_v_h))
              [T, V] [BTL, BV] [s_v_t, 1]
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
          Stmt.assign .real [BTL, BV] "b_dv" (Op.full [BTL, BV] (Op.const 0)) ]
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
            (paBwdDescBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t 1
              s_v_h s_v_t 1 scale T K V BTL BTS BK BV)
          :: ([ Stmt.ifThen (Op.constBool Bool.false) [],
              Stmt.assign .nat [BTS] "o_q" (Op.arange BTS),
              Stmt.assign .nat [BTL] "o_k" (Op.arange BTL) ]
            ++ (Stmt.forRangeDyn "i"
                (Op.mul .nat Broadcast.nil (Op.constNat i_c) (Op.constNat BTL))
                (Op.constNat ((i_c + 1) * BTL)) (Op.constNat BTS)
                (paBwdDiagBody q do_ dz i_bh i_c i_k i_v s_k_h s_k_t 1
                  s_v_h s_v_t 1 scale T K V BTL BTS BK BV)
              :: [ Stmt.assign .blockPtr [BTL, BK] "p_dk"
                    (Op.makeBlockPtrDynOffsets dk
                      (Op.constNat ((i_bh + B * H * i_v) * s_k_h))
                      [T, K] [BTL, BK] [s_k_t, 1]
                      [Op.mul .nat Broadcast.nil (Op.constNat i_c)
                          (Op.constNat BTL),
                        Op.mul .nat Broadcast.nil (Op.constNat i_k)
                          (Op.constNat BK)]),
                  Stmt.assign .blockPtr [BTL, BV] "p_dv"
                    (Op.makeBlockPtrDynOffsets dv
                      (Op.constNat ((i_bh + B * H * i_k) * s_v_h))
                      [T, V] [BTL, BV] [s_v_t, 1]
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
  obtain ⟨sF, hSt, hR1, hR2⟩ := paBwdStores_run s sL2 q k v do_ dz dk dv
    i_bh i_c i_k i_v s_k_h s_k_t s_v_h s_v_t scale B H T K V BTL BTS BK BV
    hDkDv hσk hσv hBTS hBTSpos hInvGF
  exact ⟨sF, hSt, hR1, hR2⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ParallelAttention

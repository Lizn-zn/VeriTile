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

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by
`rfl`. New lowerings this pair adds over `chunk_linear_attn`: `tl.toReal` →
`Op.natToReal`, `tl.math.exp2`/`log2` → `Op.exp2`/`Op.log2`, `%` →
`Op.mod IntegralDType.nat`, and the boundary gate `==`/`!=`/`and` →
`Op.eq`/`Op.ne`/`Op.boolAnd`. -/

/-- The `USE_INITIAL_STATE` branch (identical to `chunk_linear_attn`'s). -/
def crhInitBranch (h0 : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_h0"
      (Op.makeBlockPtrDynOffsets h0
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.castFloat FloatDType.real FloatDType.real
        (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1])
          MaskOpt.none)) ]

/-- The ragged-last-chunk gate body: rebind `d_b`/`d_i` to the `T % BT` length. -/
def crhBoundaryBranch (T BT : Nat) : List Stmt :=
  [ Stmt.assign .real [] "d_b"
      (Op.exp2 (Op.mul .real Broadcast.nil
        (Op.natToReal (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T)
          (Op.constNat BT)))
        (Op.ref .real [] "b_b"))),
    Stmt.assign .real [BT] "d_i"
      (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.sub .nat Broadcast.scalarR
          (Op.sub .nat Broadcast.scalarL
            (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT))
            (Op.ref .nat [BT] "o_i"))
          (Op.constNat 1)))
        (Op.ref .real [] "b_b"))) ]

/-- The forward chunk body: block pointers, the state **store**, the loads, the
ragged-boundary gate, and the decayed recurrence. -/
def crhFwdBody (k v h : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_k") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.ifThen
      (Op.boolAnd Broadcast.nil
        (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_t")
          (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1)))
        (Op.ne ComparableDType.nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT))
          (Op.constNat 0)))
      (crhBoundaryBranch T BT),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real Broadcast.scalarL (Op.ref .real [] "d_b")
          (Op.ref .real [BK, BV] "b_h"))
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BT, BV] "b_v")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i"))))) ]

/-- The `STORE_FINAL_STATE` branch (identical to `chunk_linear_attn`'s). -/
def crhFinalBranch (ht : RegionName) (K V BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BK, BV] "p_ht"
      (Op.makeBlockPtrDynOffsets ht
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat K))
          (Op.constNat V))
        [K, V] [BK, BV] [V, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_ht") [0, 1])
      (Op.ref .real [BK, BV] "b_h") MaskOpt.none ]

/-- The decay prologue shared by both kernels: `i_h`, `b_b`, `o_i`, and the
standard (`BT`-length) `d_b`. -/
def crhDecayPrologue (H BT : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "i_bh")
        (Op.constNat H)),
    Stmt.assign .real [] "b_b"
      (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
        (Op.exp2 (Op.sub .real Broadcast.nil
          (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
          (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
            (Op.const 1.0)))))),
    Stmt.assign .nat [BT] "o_i" (Op.arange BT),
    Stmt.assign .real [] "d_b"
      (Op.exp2 (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat BT))
        (Op.ref .real [] "b_b"))) ]

set_option maxRecDepth 8000 in
/-- **Forward body split (by `rfl`).** Twelve top-level statements. -/
theorem crh_fwd_body_eq (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (UIS SFS : Bool) :
    (crh_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t H T K V BT BK BV NT UIS SFS).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2) ]
        ++ crhDecayPrologue H BT
        ++ [ Stmt.assign .real [BT] "d_i"
              (Op.exp2 (Op.mul .real Broadcast.scalarR
                (Op.natToReal (Op.sub .nat Broadcast.scalarR
                  (Op.sub .nat Broadcast.scalarL (Op.constNat BT)
                    (Op.ref .nat [BT] "o_i"))
                  (Op.constNat 1)))
                (Op.ref .real [] "b_b"))),
            Stmt.assign .real [BK, BV] "b_h" (Op.full [BK, BV] (Op.const 0)),
            Stmt.ifThen (Op.constBool UIS) (crhInitBranch h0 K V BK BV),
            Stmt.forRange "i_t" 0 NT 1
              (crhFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
                s_h_t T K V BT BK BV NT),
            Stmt.ifThen (Op.constBool SFS) (crhFinalBranch ht K V BK BV) ] := by
  rfl

/-- The backward chunk body: the descending index, three block pointers, three
loads (the `dh` one dead), and the accumulation. -/
def crhBwdBody (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT NT : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")),
    Stmt.assign .blockPtr [BT, BT] "p_o"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BT] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BT] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BT] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BT] "p_h"
      (Op.makeBlockPtrDynOffsets dh
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BT, BT] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BT)]),
    Stmt.assign .real [BT, BT] "b_o"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_o") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_h"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_h") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_dh"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_dh")
        (Op.dot (batch := [])
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BT, BT] "b_o")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i")))
          (Op.ref .real [BT, BT] "b_v"))) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Fourteen top-level statements. -/
theorem crh_bwd_body_eq (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT NT : Nat) :
    (crh_bwd_dh_surface v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t
        H T K V BT NT).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2) ]
        ++ crhDecayPrologue H BT
        ++ [ Stmt.assign .real [BT] "d_i"
              (Op.exp2 (Op.mul .real Broadcast.scalarR
                (Op.natToReal (Op.add .nat Broadcast.scalarR
                  (Op.ref .nat [BT] "o_i") (Op.constNat 1)))
                (Op.ref .real [] "b_b"))),
            Stmt.assign .real [BT, BT] "b_dh" (Op.full [BT, BT] (Op.const 0)),
            Stmt.assign .nat [] "i_t" (Op.constNat 0),
            Stmt.forRange "j" 0 NT 1
              (crhBwdBody v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT NT),
            Stmt.assign .real [BT, BT] "b_dh"
              (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BT] "b_dh")
                (Op.ref .real [] "d_b")),
            Stmt.assign .blockPtr [BT, BT] "p_dh"
              (Op.makeBlockPtrDynOffsets dh
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh")
                    (Op.constNat s_h_h))
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k")
                      (Op.constNat K))
                    (Op.constNat V)))
                [K, V] [BT, BT] [s_h_t, 1]
                [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BT),
                  Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
            Stmt.store .real [BT, BT]
              (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_dh") [0, 1])
              (Op.ref .real [BT, BT] "b_dh") MaskOpt.none ] := by
  rfl

/-! ## Closed-form specification

The decay values mirror the walk exactly: `exp2 x = Real.exp (x·log 2)` and
`log2 y = Real.log y / log 2` are the semantics' projections. -/

/-- The per-head decay exponent `b_b = log2(1 - 2^(-5 - i_h))`, exactly as the
walk computes it. -/
noncomputable def crhBeta (s : BlockState) (H : Nat) : ℝ :=
  Real.log ((1.0 : ℝ) - Real.exp ((((0.0 : ℝ) - 5.0)
      - ((s.pids 2 % H : Nat) : ℝ) * 1.0) * Real.log 2)) / Real.log 2

/-- Chunk `t`'s effective length: `T % BT` on a ragged last chunk, else `BT`. -/
def crhLen (T BT NT t : Nat) : Nat :=
  if t = NT - 1 ∧ T % BT ≠ 0 then T % BT else BT

/-- The inter-chunk decay factor `d_b(t) = 2^(len(t)·b_b)`. -/
noncomputable def crhDb (s : BlockState) (H T BT NT t : Nat) : ℝ :=
  Real.exp (((crhLen T BT NT t : Nat) : ℝ) * crhBeta s H * Real.log 2)

/-- The intra-chunk decay weights `d_i(t)[c] = 2^((len(t) - c - 1)·b_b)` — with
the `.nat`-truncated tail lanes (see the preamble; unobservable through the
boundary-checked `v`). -/
noncomputable def crhDi (s : BlockState) (H T BT NT t c : Nat) : ℝ :=
  Real.exp (((crhLen T BT NT t - c - 1 : Nat) : ℝ) * crhBeta s H * Real.log 2)

/-- The backward weights `d_i[c] = 2^((c + 1)·b_b)` (never rebound). -/
noncomputable def crhDiBwd (s : BlockState) (H c : Nat) : ℝ :=
  Real.exp (((c + 1 : Nat) : ℝ) * crhBeta s H * Real.log 2)

/-- The backward outer factor `d_b = 2^(BT·b_b)` (never rebound). -/
noncomputable def crhDbBwd (s : BlockState) (H BT : Nat) : ℝ :=
  Real.exp (((BT : Nat) : ℝ) * crhBeta s H * Real.log 2)

/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
noncomputable def crhKElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)

/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
noncomputable def crhVElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)

/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
noncomputable def crhH0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)

/-- The guarded `k` lane, as `b_k` holds it. -/
noncomputable def crhKGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    crhKElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0

/-- The guarded `v` (or `do`) lane, as `b_v` / `b_o` holds it. -/
noncomputable def crhVGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    crhVElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0

/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
noncomputable def crhH0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    crhH0Elem s h0 K V BK BV e p
  else 0

/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`
before chunk `t` runs: the decayed recurrence
`H_{t+1} = d_b(t)·H_t + Σ_c k[e,c]·(v[c,p]·d_i(t)[c])` seeded with the gated
`h0`. Recursive (not a power closed form): the ragged last chunk gives the
final step its own decay length. -/
noncomputable def crhState (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) : Nat → Nat → Nat → ℝ
  | 0 => fun e p => if UIS then crhH0Guarded s h0 K V BK BV e p else 0
  | t + 1 => fun e p =>
      crhDb s H T BT NT t
          * crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
              H T K V BT BK BV NT t e p
        + ∑ c : Fin BT,
            crhKGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
              * (crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p
                  * crhDi s H T BT NT t c.val)

/-- One backward chunk's contribution at lane `(x, y)`:
`Σ_r (do[x, r]·d_i[x]) · v[r, y]` — the scrambled contraction the upstream
kernel actually performs (both operands are `[BT, BT]` blocks of the `(T, V)`
parents; see the preamble). -/
noncomputable def crhBContrib (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT : Nat) (t x y : Nat) : ℝ :=
  ∑ r : Fin BT,
    (crhVGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BT t x r.val
        * crhDiBwd s H x)
      * crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BT t r.val y

/-- The backward accumulator after `c` descending iterations: every chunk from
`NT - c` up (the same indicator-sum carry as `chunk_linear_attn`). -/
noncomputable def crhDhAcc (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (c x y : Nat) : ℝ :=
  ∑ u : Fin NT, if NT - c ≤ u.val then
    crhBContrib s v do_ s_vo_h s_vo_t s_vo_d H T V BT u.val x y
  else 0

/-- **The stored backward value**: `d_b · Σ_{all chunks} contrib`. -/
noncomputable def crhDhOut (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (x y : Nat) : ℝ :=
  crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT NT x y * crhDbBwd s H BT

/-- The `h` / `ht` chunk-store address at lane `(e, p)` (same layout as
`chunk_linear_attn`). -/
def crhHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- The `ht` store address at lane `(e, p)`. -/
def crhHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- The backward single-store address at lane `(x, y)`: base
`i_bh·s_h_h + i_k·K·V`, offsets `(i_v·BT, i_t·BT)` with the post-loop
`i_t = 0`. -/
def crhDhOffset (s : BlockState) (s_h_h s_h_t K V BT : Nat)
    (idx : TileIndex [BT, BT]) : Nat :=
  s.pids 2 * s_h_h + s.pids 0 * K * V + (s.pids 1 * BT + idx.1.val) * s_h_t
    + (0 * BT + idx.2.1.val) * 1

/-- A forward state lane is *active* when it maps inside the `K × V` window. -/
def crhActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V

/-- A backward store lane is *active* under its own (scrambled) window. -/
def crhDhActive (s : BlockState) (K V BT : Nat)
    (idx : TileIndex [BT, BT]) : Prop :=
  s.pids 1 * BT + idx.1.val < K ∧ 0 * BT + idx.2.1.val < V

/-! ## Eval recipes (local copies + the decay family) -/

private theorem crh_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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

private theorem crh_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
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

private theorem crh_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem crh_mulMulConst_eval (s : BlockState) (name : RegName)
    (val cB cC : Nat) (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat cB))
        (Op.constNat cC)) s
      = some (Tile.scalar (val * cB * cC)) := by
  rw [evalOp_mul, crh_mulConst_eval s name val cB hr]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem crh_hBase_eval (s : BlockState) (s_h_h K V : Nat) (ibh it : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar it)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
          (Op.constNat V))) s
      = some (Tile.scalar (ibh * s_h_h + it * K * V)) := by
  rw [evalOp_add, crh_mulConst_eval s "i_bh" ibh s_h_h hibh, evalOp_mul,
    crh_mulConst_eval s "i_t" it K hit]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The `dh` single-store base `i_bh * s_h_h + i_k * K * V`. -/
private theorem crh_dhBase_eval (s : BlockState) (s_h_h K V : Nat) (ibh ik : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hik : s.regs .nat [] "i_k" = some (Tile.scalar ik)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat K))
          (Op.constNat V))) s
      = some (Tile.scalar (ibh * s_h_h + ik * K * V)) := by
  rw [evalOp_add, crh_mulConst_eval s "i_bh" ibh s_h_h hibh, evalOp_mul,
    crh_mulConst_eval s "i_k" ik K hik]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem crh_itIdx_eval (t : BlockState) (NT c : Nat)
    (hj : t.regs .nat [] "j" = some (Tile.scalar c)) :
    evalOp (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")) t
      = some (Tile.scalar (NT - 1 - c)) := by
  have hx : evalOp (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1)) t
      = some (Tile.scalar (NT - 1)) := by
    rw [evalOp_sub]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [evalOp_sub, hx]
  simp only [evalOp_ref, hj, Option.bind_eq_bind, Option.bind_some]
  rfl

private theorem crh_castFloat_eval {sh : TileShape} (x : Op .real sh)
    (t : BlockState) (vx : Tile .real sh) (hx : evalOp x t = some vx) :
    @evalOp TileDType.real sh (Op.castFloat FloatDType.real FloatDType.real x) t
      = some vx := by
  show @evalOp FloatDType.real.toTileDType sh
    (Op.castFloat FloatDType.real FloatDType.real x) t = some vx
  rw [evalOp_castFloat,
    show @evalOp FloatDType.real.toTileDType sh x t = some vx from hx]
  rfl

private theorem crh_zeros_eval (BR BS : Nat) (t : BlockState) :
    evalOp (Op.full [BR, BS] (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BR, BS]) := by
  simp [evalOp_full, evalOp_const]

private theorem crh_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

private theorem crh_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ))
    = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

private theorem crh_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
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
  exact crh_withBot_sum_some _

private theorem crh_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

private theorem crh_ifThen_false_noop (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.false) body) X = some X := by
  simp [stepStmt, evalOp]

private theorem crh_ifThen_true_run (body : List Stmt) (X : BlockState) :
    stepStmt (Stmt.ifThen (Op.constBool Bool.true) body) X = stepStmts body X := by
  simp [stepStmt, evalOp]

private theorem crh_addTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add .real bc x y) t
      = some (Tile.bop NumericDType.real.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- Scalar `name % H`. -/
private theorem crh_mod_eval (s : BlockState) (name : RegName) (val H : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] name)
        (Op.constNat H)) s
      = some (Tile.scalar (val % H)) := by
  simp only [evalOp, evalOp_ref, hr, evalOp_constNat, bind, Option.bind]
  rfl

/-- The decay exponent statement lands on `crhBeta`. -/
private theorem crh_bb_eval (s t : BlockState) (H : Nat)
    (hih : t.regs .nat [] "i_h" = some (Tile.scalar (s.pids 2 % H))) :
    evalOp (Op.log2 (Op.sub .real Broadcast.nil (Op.const 1.0)
        (Op.exp2 (Op.sub .real Broadcast.nil
          (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 5.0))
          (Op.mul .real Broadcast.nil (Op.natToReal (Op.ref .nat [] "i_h"))
            (Op.const 1.0)))))) t
      = some (Tile.scalar (some (crhBeta s H))) := by
  simp only [evalOp, evalOp_ref, hih, bind, Option.bind]
  rfl

/-- The standard (`BT`-length) `d_b` statement lands on `crhDbBwd`. -/
private theorem crh_db_eval (s t : BlockState) (H BT : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.nil (Op.natToReal (Op.constNat BT))
        (Op.ref .real [] "b_b"))) t
      = some (Tile.scalar (some (crhDbBwd s H BT))) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The boundary `d_b` statement lands on `crhDb` at the ragged length. -/
private theorem crh_dbBoundary_eval (s t : BlockState) (H T BT : Nat)
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.nil
        (Op.natToReal (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T)
          (Op.constNat BT)))
        (Op.ref .real [] "b_b"))) t
      = some (Tile.scalar (some (Real.exp (((T % BT : Nat) : ℝ) * crhBeta s H
          * Real.log 2)))) := by
  simp only [evalOp, evalOp_ref, hbb, bind, Option.bind]
  rfl

/-- The `o_i` arange tile. -/
def crhOiTile (BT : Nat) : Tile .nat [BT] :=
  Tile.vec fun i : Fin BT => (i.val : Nat)

/-- The forward standard `d_i` tile: lane `c` holds `2^((BT - c - 1)·b_b)`. -/
noncomputable def crhDiStdTile (s : BlockState) (H BT : Nat) : Tile .real [BT] :=
  ⟨fun idx => some (Real.exp (((BT - idx.1.val - 1 : Nat) : ℝ) * crhBeta s H
    * Real.log 2))⟩

/-- The forward standard `d_i` statement lands on `crhDiStdTile`. -/
private theorem crh_diStd_eval (s t : BlockState) (H BT : Nat)
    (hoi : t.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.sub .nat Broadcast.scalarR
          (Op.sub .nat Broadcast.scalarL (Op.constNat BT) (Op.ref .nat [BT] "o_i"))
          (Op.constNat 1)))
        (Op.ref .real [] "b_b"))) t
      = some (crhDiStdTile s H BT) := by
  simp only [evalOp, evalOp_ref, hoi, hbb, bind, Option.bind]
  rfl

/-- The boundary `d_i` tile: lane `c` holds `2^((T%BT - c - 1)·b_b)` (nat-sub). -/
noncomputable def crhDiBoundaryTile (s : BlockState) (H T BT : Nat) :
    Tile .real [BT] :=
  ⟨fun idx => some (Real.exp (((T % BT - idx.1.val - 1 : Nat) : ℝ) * crhBeta s H
    * Real.log 2))⟩

/-- The boundary `d_i` statement lands on `crhDiBoundaryTile`. -/
private theorem crh_diBoundary_eval (s t : BlockState) (H T BT : Nat)
    (hoi : t.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.sub .nat Broadcast.scalarR
          (Op.sub .nat Broadcast.scalarL
            (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT))
            (Op.ref .nat [BT] "o_i"))
          (Op.constNat 1)))
        (Op.ref .real [] "b_b"))) t
      = some (crhDiBoundaryTile s H T BT) := by
  simp only [evalOp, evalOp_ref, hoi, hbb, bind, Option.bind]
  rfl

/-- The backward `d_i` tile: lane `c` holds `2^((c + 1)·b_b)`. -/
noncomputable def crhDiBwdTile (s : BlockState) (H BT : Nat) : Tile .real [BT] :=
  ⟨fun idx => some (crhDiBwd s H idx.1.val)⟩

/-- The backward `d_i` statement lands on `crhDiBwdTile`. -/
private theorem crh_diBwd_eval (s t : BlockState) (H BT : Nat)
    (hoi : t.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : t.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))) :
    evalOp (Op.exp2 (Op.mul .real Broadcast.scalarR
        (Op.natToReal (Op.add .nat Broadcast.scalarR (Op.ref .nat [BT] "o_i")
          (Op.constNat 1)))
        (Op.ref .real [] "b_b"))) t
      = some (crhDiBwdTile s H BT) := by
  simp only [evalOp, evalOp_ref, hoi, hbb, bind, Option.bind]
  rfl

/-- The ragged-boundary gate condition evaluates to the decidable conjunction. -/
private theorem crh_cond_eval (t : BlockState) (T BT NT i : Nat)
    (hit : t.regs .nat [] "i_t" = some (Tile.scalar i)) :
    evalOp (Op.boolAnd Broadcast.nil
        (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_t")
          (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1)))
        (Op.ne ComparableDType.nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT))
          (Op.constNat 0))) t
      = some (Tile.scalar (decide (i = NT - 1) && decide (T % BT ≠ 0))) := by
  simp only [evalOp, evalOp_ref, hit, bind, Option.bind]
  rfl

private theorem crh_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkRetention

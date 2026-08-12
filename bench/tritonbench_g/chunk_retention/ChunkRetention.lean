import VeriTile.Triton

/-!
# `chunk_retention` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_retention_fwd_kernel_h` (the file's first kernel), the output
forward `fwd_kernel_o`, the state-recurrence backward `bwd_kernel_dh`, and the
fused gradient backward `bwd_kernel_dqkv`. This file covers the two
**state-recurrence** kernels — `chunk_retention.py`'s
`chunk_retention_fwd_kernel_h` (the file's first kernel) and its descending
mirror `chunk_retention_bwd_kernel_dh` — the retention (decayed) siblings of
the ported `chunk_linear_attn` pair:

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

## Proof map

```
crh_fwd_h_surface / crh_bwd_dh_surface   faithful DSL transcriptions
├─ crh_fwd_body_eq / crh_bwd_body_eq     statement-list splits (rfl)
├─ crhState (RECURSIVE) / crhDhOut       per-chunk decayed spec / scrambled sum
├─ crhPreDecay_run                       shared pids + decay prologue walk
├─ crhBoundaryGate_run                   ragged gate → crhDb t / crhDiFullTile t
├─ crhStore_step_eq / _props             offset-parameterized [BR,BS] block store
├─ crhFwdBody_step / crhBwdBody_step     one chunk (store+gate+advance / loads)
├─ crhFwdLoop_run / crhBwdLoop_run       `forRange_inv`; fwd carries the
│                                        conditional register clause c < NT
├─ ★ crh_fwd_h_exec_genuine              h[·,·,t] = decayed pre-chunk state,
│                                        ∀ t < NT; ht flush under the gate
└─ ★ crh_bwd_dh_exec_genuine             dh block = d_b · Σ (do ⊙ d_i)·v
```

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

/-! ## The generic state-block store (offset-parameterized)

Serves four stores: `h[·,·,t]` per chunk and `ht` (offsets `(i_k·BK, i_v·BV)`),
and `dh` once (offsets `(i_v·BT, i_t·BT)` with the post-loop `i_t = 0`). -/

/-- The store address at lane `(e, p)`: `base + (rowOff+e)·σ + (colOff+p)`. -/
def crhStoreAddr (base σ rowOff colOff : Nat) (BR BS : Nat)
    (idx : TileIndex [BR, BS]) : Nat :=
  base + (rowOff + idx.1.val) * σ + (colOff + idx.2.1.val) * 1

theorem crhHOffset_eq_storeAddr (s : BlockState) (s_h_h s_h_t K V BK BV t : Nat)
    (idx : TileIndex [BK, BV]) :
    crhHOffset s s_h_h s_h_t K V BK BV t idx
      = crhStoreAddr (s.pids 2 * s_h_h + t * K * V) s_h_t
          (s.pids 0 * BK) (s.pids 1 * BV) BK BV idx := rfl

theorem crhHtOffset_eq_storeAddr (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) :
    crhHtOffset s K V BK BV idx
      = crhStoreAddr (s.pids 2 * K * V) V
          (s.pids 0 * BK) (s.pids 1 * BV) BK BV idx := rfl

theorem crhDhOffset_eq_storeAddr (s : BlockState) (s_h_h s_h_t K V BT : Nat)
    (idx : TileIndex [BT, BT]) :
    crhDhOffset s s_h_h s_h_t K V BT idx
      = crhStoreAddr (s.pids 2 * s_h_h + s.pids 0 * K * V) s_h_t
          (s.pids 1 * BT) (0 * BT) BT BT idx := rfl

/-- The explicit post-store state: the masked lane-by-lane scatter of cell fn
`f` over input state `sin`. -/
noncomputable def crhStoreState (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BR, BS]).foldl
    (fun acc i => if (rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
        then acc.writeMem rg (crhStoreAddr base σ rowOff colOff BR BS i)
          (f i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **State-block store step (eq).** -/
theorem crhStore_step_eq (sin : BlockState) (rg : RegionName)
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
      = some (crhStoreState sin rg base σ rowOff colOff K V BR BS f) := by
  unfold stepStmt crhStoreState
  simp only [evalOp_ref, hb, hp]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [BR, BS])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, crhStoreAddr]
  by_cases hbnd : rowOff + e.val < K ∧ colOff + p.val < V
  · simp only [hbnd, BlockState.writeMemTyped_real, hbf, Nat.mul_one]
    rfl
  · simp only [hbnd, decide_false, Bool.false_eq_true, if_false]

/-- If two `Q`-blocks with in-block offsets `A, B < Q` collide, the block
indices agree. -/
private theorem crh_block_index_inj {Q j c A B : Nat} (hA : A < Q) (hB : B < Q)
    (heq : j * Q + A = c * Q + B) : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A) hA
  have hj : (j * Q + A) / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B) / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- One block's store lanes are pairwise distinct once a `BS` row segment fits
under the row stride. -/
theorem crhStoreAddr_injective (base σ rowOff colOff BR BS : Nat)
    (hBSσ : BS ≤ σ) :
    Function.Injective (crhStoreAddr base σ rowOff colOff BR BS) := by
  rintro ⟨e, p, u⟩ ⟨e', p', u'⟩ heq
  simp only [crhStoreAddr] at heq
  have hp := p.isLt
  have hp' := p'.isLt
  have h2 : (rowOff + e.val) * σ + p.val = (rowOff + e'.val) * σ + p'.val := by
    omega
  have hlt : p.val < σ := by omega
  have hlt' : p'.val < σ := by omega
  have hjj : rowOff + e.val = rowOff + e'.val :=
    crh_block_index_inj hlt hlt' h2
  have he : e = e' := Fin.ext (by omega)
  have hpv : p = p' := Fin.ext (by
    have hσ : (rowOff + e.val) * σ = (rowOff + e'.val) * σ := by rw [hjj]
    omega)
  subst he
  subst hpv
  rfl

set_option maxHeartbeats 4000000 in
/-- **State-block store readback** (mask-restricted same-region frame). -/
theorem crhStore_step_props (sin : BlockState) (rg : RegionName)
    (base σ rowOff colOff K V BR BS : Nat) (f : Nat → Nat → ℝ)
    (hInj : Function.Injective (crhStoreAddr base σ rowOff colOff BR BS)) :
    (crhStoreState sin rg base σ rowOff colOff K V BR BS f).pids = sin.pids
      ∧ (crhStoreState sin rg base σ rowOff colOff K V BR BS f).regs = sin.regs
      ∧ (∀ idx : TileIndex [BR, BS],
          (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
          (crhStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg
              (crhStoreAddr base σ rowOff colOff BR BS idx)
            = f idx.1.val idx.2.1.val)
      ∧ (∀ rg' off, rg' ≠ rg →
          (crhStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg' off
            = sin.readMem rg' off)
      ∧ (∀ off, (∀ idx : TileIndex [BR, BS],
            (rowOff + idx.1.val < K ∧ colOff + idx.2.1.val < V) →
            off ≠ crhStoreAddr base σ rowOff colOff BR BS idx) →
          (crhStoreState sin rg base σ rowOff colOff K V BR BS f).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold crhStoreState
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name
    rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
      (crhStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      hInj idx
    rw [h, if_pos ⟨h1, h2⟩]
  · intro rg' off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      rg (crhStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin rg' off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      rg (crhStoreAddr base σ rowOff colOff BR BS) (fun i => f i.1.val i.2.1.val)
      (fun i => rowOff + i.1.val < K ∧ colOff + i.2.1.val < V)
      _ sin off (fun i _ hPi => hoff i hPi)

/-- Distinct chunks write disjoint `K·V` blocks of `h`: active lanes of chunk
`j` never collide with active lanes of chunk `c ≠ j`. -/
theorem crhHOffset_chunk_disjoint (s : BlockState)
    (s_h_h s_h_t K V BK BV : Nat) (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (j c : Nat) (hjc : j ≠ c) (idxj idxc : TileIndex [BK, BV])
    (hja : crhActive s K V BK BV idxj) (hca : crhActive s K V BK BV idxc) :
    crhHOffset s s_h_h s_h_t K V BK BV j idxj
      ≠ crhHOffset s s_h_h s_h_t K V BK BV c idxc := by
  obtain ⟨ej, pj, _⟩ := idxj
  obtain ⟨ec, pc, _⟩ := idxc
  obtain ⟨hjK, hjV⟩ := hja
  obtain ⟨hcK, hcV⟩ := hca
  simp only [crhHOffset]
  intro heq
  apply hjc
  have hbound : ∀ R C : Nat, R < K → C < V → R * s_h_t + C < K * V := by
    intro R C hR hC
    have h1 : R * s_h_t ≤ (K - 1) * s_h_t := Nat.mul_le_mul_right s_h_t (by omega)
    omega
  have hjm : j * (K * V) = j * K * V := by rw [Nat.mul_assoc]
  have hcm : c * (K * V) = c * K * V := by rw [Nat.mul_assoc]
  have heq2 : j * (K * V) + ((s.pids 0 * BK + ej.val) * s_h_t + (s.pids 1 * BV + pj.val))
      = c * (K * V) + ((s.pids 0 * BK + ec.val) * s_h_t + (s.pids 1 * BV + pc.val)) := by
    omega
  exact crh_block_index_inj
    (hbound _ _ hjK hjV) (hbound _ _ hcK hcV) heq2

/-! ## Value tiles, load bridges, and spec recurrences -/

/-- The loaded `b_k` tile of chunk `t`: cell `(e, c)`. -/
noncomputable def crhBkTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat) : Tile .real [BK, BT] :=
  ⟨fun idx => some (crhKGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t
    idx.2.1.val idx.1.val)⟩

/-- The loaded `b_v` / `b_o` tile of chunk `t`: cell `(c, p)`. -/
noncomputable def crhBvTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t
    idx.1.val idx.2.1.val)⟩

/-- The seeded `b_h` tile: cell `(e, p)`. -/
noncomputable def crhH0Tile (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (crhH0Guarded s h0 K V BK BV idx.1.val idx.2.1.val)⟩

private theorem crh_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem k off = s.readMem k off)
    (hpk : sin.regs .blockPtr [BK, BT] name = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [s.pids 0 * BK, t * BT]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] name) [0, 1]) MaskOpt.none) sin
      = some (crhBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t) := by
  rw [crh_load_bp_2d k sin name (s.pids 2 * s_qk_h) K T BK BT s_qk_d s_qk_t
    (s.pids 0 * BK) (t * BT) hpk]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, c, u⟩ := idx
  simp only [crhBkTile, crhKGuarded, crhKElem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ t * BT + c.val < T
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

private theorem crh_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem v off = s.readMem v off)
    (hpv : sin.regs .blockPtr [BT, BV] name = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [t * BT, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] name) [0, 1]) MaskOpt.none) sin
      = some (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t) := by
  rw [crh_load_bp_2d v sin name (s.pids 2 * s_vo_h) T V BT BV s_vo_t s_vo_d
    (t * BT) (s.pids 1 * BV) hpv]
  refine congrArg some ?_
  ext idx
  obtain ⟨c, p, u⟩ := idx
  simp only [crhBvTile, crhVGuarded, crhVElem, hmem]
  by_cases hb : t * BT + c.val < T ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

private theorem crh_h0Load_eq (s sin : BlockState) (h0 : RegionName)
    (K V BK BV : Nat)
    (hmem : ∀ off, sin.readMem h0 off = s.readMem h0 off)
    (hph0 : sin.regs .blockPtr [BK, BV] "p_h0" = some
      ⟨fun _ => BlockPtr.mk h0 (s.pids 2 * K * V) [K, V] [BK, BV] [V, 1]
        [s.pids 0 * BK, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1]) MaskOpt.none) sin
      = some (crhH0Tile s h0 K V BK BV) := by
  rw [crh_load_bp_2d h0 sin "p_h0" (s.pids 2 * K * V) K V BK BV V 1
    (s.pids 0 * BK) (s.pids 1 * BV) hph0]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, p, u⟩ := idx
  simp only [crhH0Tile, crhH0Guarded, crhH0Elem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-- The post-gate `d_i` tile at chunk `t`: lane `c` holds `crhDi t c`. -/
noncomputable def crhDiFullTile (s : BlockState) (H T BT NT t : Nat) :
    Tile .real [BT] :=
  ⟨fun idx => some (crhDi s H T BT NT t idx.1.val)⟩

/-- Off the ragged last chunk the decay factor is the standard one. -/
theorem crhDb_std (s : BlockState) (H T BT NT t : Nat)
    (h : ¬(t = NT - 1 ∧ T % BT ≠ 0)) :
    crhDb s H T BT NT t = crhDbBwd s H BT := by
  unfold crhDb crhDbBwd crhLen
  rw [if_neg h]

/-- Off the ragged last chunk the weight tile is the standard one. -/
theorem crhDiFull_std (s : BlockState) (H T BT NT t : Nat)
    (h : ¬(t = NT - 1 ∧ T % BT ≠ 0)) :
    crhDiFullTile s H T BT NT t = crhDiStdTile s H BT := by
  refine Tile.ext fun idx => ?_
  simp only [crhDiFullTile, crhDiStdTile, crhDi, crhLen, if_neg h]

/-- On the ragged last chunk they are the boundary values. -/
theorem crhDb_boundary (s : BlockState) (H T BT NT t : Nat)
    (h : t = NT - 1 ∧ T % BT ≠ 0) :
    crhDb s H T BT NT t
      = Real.exp (((T % BT : Nat) : ℝ) * crhBeta s H * Real.log 2) := by
  unfold crhDb crhLen
  rw [if_pos h]

theorem crhDiFull_boundary (s : BlockState) (H T BT NT t : Nat)
    (h : t = NT - 1 ∧ T % BT ≠ 0) :
    crhDiFullTile s H T BT NT t = crhDiBoundaryTile s H T BT := by
  refine Tile.ext fun idx => ?_
  simp only [crhDiFullTile, crhDiBoundaryTile, crhDi, crhLen, if_pos h]

theorem crhDhAcc_zero (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (x y : Nat) :
    crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT 0 x y = 0 := by
  refine Finset.sum_eq_zero fun u _ => ?_
  rw [if_neg]
  have := u.isLt
  omega

theorem crhDhAcc_succ (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (c x y : Nat)
    (hc : c < NT) :
    crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT (c + 1) x y
      = crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT c x y
        + crhBContrib s v do_ s_vo_h s_vo_t s_vo_d H T V BT (NT - 1 - c) x y := by
  classical
  unfold crhDhAcc
  set B : Nat → ℝ := fun uv =>
    crhBContrib s v do_ s_vo_h s_vo_t s_vo_d H T V BT uv x y with hB
  have hpt : ∀ u : Fin NT,
      (if NT - (c + 1) ≤ u.val then B u.val else 0)
        = (if NT - c ≤ u.val then B u.val else 0)
          + (if u = (⟨NT - 1 - c, by omega⟩ : Fin NT) then B u.val else 0) := by
    intro u
    by_cases h1 : NT - c ≤ u.val
    · rw [if_pos (by omega), if_pos h1, if_neg (by
        intro hu
        have : u.val = NT - 1 - c := by rw [hu]
        omega), add_zero]
    · by_cases h2 : u = (⟨NT - 1 - c, by omega⟩ : Fin NT)
      · have hval : u.val = NT - 1 - c := by rw [h2]
        rw [if_pos (by omega), if_neg h1, if_pos h2, zero_add]
      · have hval : u.val ≠ NT - 1 - c := by
          intro hv
          exact h2 (Fin.ext hv)
        rw [if_neg (by omega), if_neg h1, if_neg h2, add_zero]
  rw [Finset.sum_congr rfl (fun u _ => hpt u), Finset.sum_add_distrib,
    Finset.sum_ite_eq' Finset.univ (⟨NT - 1 - c, by omega⟩ : Fin NT)
      (fun u => B u.val), if_pos (Finset.mem_univ _)]

/-! ## The shared walk pieces -/

private theorem crh_mulTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul .real bc x y) t
      = some (Tile.bop NumericDType.real.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

set_option maxHeartbeats 1000000 in
/-- **The shared prologue**: the three program ids and the decay prologue
(`i_h`, `b_b`, `o_i`, the standard `d_b`). -/
private theorem crhPreDecay_run (s : BlockState) (H BT : Nat) :
    ∃ sP, (∀ rest : List Stmt, stepStmts
        ([ Stmt.assign .nat [] "i_k" (Op.programId 0),
           Stmt.assign .nat [] "i_v" (Op.programId 1),
           Stmt.assign .nat [] "i_bh" (Op.programId 2) ]
          ++ crhDecayPrologue H BT ++ rest) s = stepStmts rest sP)
      ∧ sP.pids = s.pids
      ∧ sP.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ sP.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ sP.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ sP.regs .nat [BT] "o_i" = some (crhOiTile BT)
      ∧ sP.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))
      ∧ sP.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT)))
      ∧ (∀ rg off, sP.readMem rg off = s.readMem rg off) := by
  refine ⟨((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
        "i_v" .nat [] (Tile.scalar (s.pids 1))).setReg
        "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
        "i_h" .nat [] (Tile.scalar (s.pids 2 % H))).setReg
        "b_b" .real [] (Tile.scalar (some (crhBeta s H)))).setReg
        "o_i" .nat [BT] (crhOiTile BT)).setReg
        "d_b" .real [] (Tile.scalar (some (crhDbBwd s H BT))),
    fun rest => ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun rg off => ?_⟩
  · rw [show ([ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2) ]
          ++ crhDecayPrologue H BT ++ rest)
        = Stmt.assign .nat [] "i_k" (Op.programId 0)
          :: Stmt.assign .nat [] "i_v" (Op.programId 1)
          :: Stmt.assign .nat [] "i_bh" (Op.programId 2)
          :: (crhDecayPrologue H BT ++ rest) from rfl]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
    show stepStmts
      (Stmt.assign .nat [] "i_h" _ :: Stmt.assign .real [] "b_b" _
        :: Stmt.assign .nat [BT] "o_i" _ :: Stmt.assign .real [] "d_b" _ :: rest) _
      = _
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (crh_mod_eval _ "i_bh" (s.pids 2) H (by simp))),
      stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_bb_eval s _ H (by simp))),
      stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.arange BT) _ = some (crhOiTile BT) from by
          rw [evalOp_arange]
          rfl)),
      stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_db_eval s _ H BT (by simp)))]
    rfl
  · rw [BlockState.setReg_pids, BlockState.setReg_pids, BlockState.setReg_pids,
      BlockState.setReg_pids, BlockState.setReg_pids, BlockState.setReg_pids,
      BlockState.setReg_pids]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_same]
  · rw [BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem, BlockState.setReg_readMem,
      BlockState.setReg_readMem]

/-- **The ragged-boundary gate.** Whatever the branch decides, the registers
land on the chunk-`i` decay values `crhDb i` / `crhDiFullTile i`. -/
private theorem crhBoundaryGate_run (s tS : BlockState) (H T BT NT i : Nat)
    (hit : tS.regs .nat [] "i_t" = some (Tile.scalar i))
    (hoi : tS.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : tS.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H))))
    (hdb : tS.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT))))
    (hdi : tS.regs .real [BT] "d_i" = some (crhDiStdTile s H BT)) :
    ∃ s', stepStmt (Stmt.ifThen
        (Op.boolAnd Broadcast.nil
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_t")
            (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1)))
          (Op.ne ComparableDType.nat Broadcast.nil
            (Op.mod IntegralDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT))
            (Op.constNat 0)))
        (crhBoundaryBranch T BT)) tS = some s'
      ∧ s'.pids = tS.pids
      ∧ (∀ rg off, s'.readMem rg off = tS.readMem rg off)
      ∧ (∀ {dt : TileDType} {sh : TileShape} (nm : RegName),
          nm ≠ "d_b" → nm ≠ "d_i" → s'.regs dt sh nm = tS.regs dt sh nm)
      ∧ s'.regs .real [] "d_b" = some (Tile.scalar (some (crhDb s H T BT NT i)))
      ∧ s'.regs .real [BT] "d_i" = some (crhDiFullTile s H T BT NT i) := by
  rw [crh_ifThen_step, crh_cond_eval tS T BT NT i hit,
    show ((some (Tile.scalar (decide (i = NT - 1) && decide (T % BT ≠ 0))
        : Tile .bool [])).bind
        (fun c => if c.data PUnit.unit then stepStmts (crhBoundaryBranch T BT) tS
          else some tS)
      = (if (decide (i = NT - 1) && decide (T % BT ≠ 0)) = Bool.true then
          stepStmts (crhBoundaryBranch T BT) tS else some tS)) from rfl]
  by_cases hb : i = NT - 1 ∧ T % BT ≠ 0
  · rw [if_pos (by simp [hb.1, hb.2])]
    unfold crhBoundaryBranch
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_dbBoundary_eval s tS H T BT hbb)),
      stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_diBoundary_eval s _ H T BT (by simpa using hoi) (by simpa using hbb))),
      stepStmts.nil]
    refine ⟨_, rfl, by simp, fun rg off => by simp, ?_, ?_, ?_⟩
    · intro dt sh nm hnm1 hnm2
      rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnm2,
        BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnm1]
    · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        BlockState.setReg_same, crhDb_boundary s H T BT NT i hb]
    · rw [BlockState.setReg_same, crhDiFull_boundary s H T BT NT i hb]
  · rw [if_neg (by
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      exact hb)]
    refine ⟨tS, rfl, rfl, fun _ _ => rfl, ?_, ?_, ?_⟩
    · intro dt sh nm _ _
      rfl
    · rw [hdb, crhDb_std s H T BT NT i hb]
    · rw [hdi, crhDiFull_std s H T BT NT i hb]

/-- The carried forward state tile at chunk `t`. -/
noncomputable def crhHTile (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) (t : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
    s_vo_d H T K V BT BK BV NT t idx.1.val idx.2.1.val)⟩

/-- The decayed recurrence statement advances the state tile by one chunk —
definitionally, since `crhState` is recursive. -/
private theorem crh_fwd_advance_eval (s sin : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) (t : Nat)
    (hdb : sin.regs .real [] "d_b"
      = some (Tile.scalar (some (crhDb s H T BT NT t))))
    (hdi : sin.regs .real [BT] "d_i" = some (crhDiFullTile s H T BT NT t))
    (hbh : sin.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT t))
    (hbk : sin.regs .real [BK, BT] "b_k"
      = some (crhBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t))
    (hbv : sin.regs .real [BT, BV] "b_v"
      = some (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real Broadcast.scalarL (Op.ref .real [] "d_b")
          (Op.ref .real [BK, BV] "b_h"))
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BT, BV] "b_v")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i"))))) sin
      = some (crhHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          H T K V BT BK BV NT (t + 1)) := by
  have hxe : evalOp (Op.mul .real Broadcast.scalarL (Op.ref .real [] "d_b")
      (Op.ref .real [BK, BV] "b_h")) sin
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarL
        (Tile.scalar (some (crhDb s H T BT NT t)))
        (crhHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          H T K V BT BK BV NT t)) :=
    crh_mulTile_eval Broadcast.scalarL _ _ sin _ _
      (by rw [evalOp_ref]; exact hdb) (by rw [evalOp_ref]; exact hbh)
  have hwv : evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BT, BV] "b_v")
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i"))) sin
      = some (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t)
        (Tile.expandDim ⟨1, by simp⟩ (crhDiFullTile s H T BT NT t))) :=
    crh_mulTile_eval (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      _ _ sin _ _
      (by rw [evalOp_ref]; exact hbv)
      (by
        erw [evalOp_expandDim]
        simp only [evalOp_ref, hdi, Option.bind_eq_bind, Option.bind_some]
        rfl)
  have hye : evalOp (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BV] "b_v")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i")))) sin
      = some (Tile.dot [] (crhBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t)
        (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t)
          (Tile.expandDim ⟨1, by simp⟩ (crhDiFullTile s H T BT NT t)))) :=
    crh_dot_eval _ _ sin _ _ (by rw [evalOp_ref]; exact hbk) hwv
  erw [crh_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ hxe hye]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨e, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, crhHTile]
  rw [crh_dot2d_elem _ _ e p
    (fun c => crhKGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e.val)
    (fun c => crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p.val
      * crhDi s H T BT NT t c.val)
    (fun c => rfl) (fun c => rfl)]
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One forward chunk.** Stores the carried state at chunk `t`'s block of
`h`, runs the ragged-boundary gate, then advances by the decayed recurrence. -/
theorem crhFwdBody_step (s sin : BlockState) (k v h h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (t : Nat)
    (hHk : h ≠ k) (hHv : h ≠ v) (hσ : BV ≤ s_h_t)
    (hmemK : ∀ off, sin.readMem k off = s.readMem k off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit : sin.regs .nat [] "i_t" = some (Tile.scalar t))
    (hoi : sin.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : sin.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H))))
    (hdb : sin.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT))))
    (hdi : sin.regs .real [BT] "d_i" = some (crhDiStdTile s H BT))
    (hbh : sin.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT t)) :
    ∃ s', stepStmts (crhFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t T K V BT BK BV NT) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .nat [BT] "o_i" = some (crhOiTile BT)
      ∧ s'.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))
      ∧ s'.regs .real [] "d_b" = some (Tile.scalar (some (crhDb s H T BT NT t)))
      ∧ s'.regs .real [BT] "d_i" = some (crhDiFullTile s H T BT NT t)
      ∧ s'.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT (t + 1))
      ∧ (∀ rg off, rg ≠ h → s'.readMem rg off = sin.readMem rg off)
      ∧ (∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
          s'.readMem h (crhHOffset s s_h_h s_h_t K V BK BV t idx)
            = crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                H T K V BT BK BV NT t idx.1.val idx.2.1.val)
      ∧ (∀ off, (∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
            off ≠ crhHOffset s s_h_h s_h_t K V BK BV t idx) →
          s'.readMem h off = sin.readMem h off) := by
  unfold crhFwdBody
  -- p_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval k sin _ _ _ [K, T] [BK, BT] [s_qk_d, s_qk_t]
      (s.pids 2 * s_qk_h) (s.pids 0 * BK) (t * BT)
      (crh_mulConst_eval sin "i_bh" (s.pids 2) s_qk_h hibh)
      (crh_mulConst_eval sin "i_k" (s.pids 0) BK hik)
      (crh_mulConst_eval sin "i_t" t BT hit)))]
  -- p_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BT, BV] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) (t * BT) (s.pids 1 * BV)
      (crh_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simpa using hibh))
      (crh_mulConst_eval _ "i_t" t BT (by simpa using hit))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- p_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (s.pids 2 * s_h_h + t * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
      (crh_hBase_eval _ s_h_h K V (s.pids 2) t (by simpa using hibh)
        (by simpa using hit))
      (crh_mulConst_eval _ "i_k" (s.pids 0) BK (by simpa using hik))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BV (by simpa using hiv))))]
  -- the state store into h's chunk-t block
  rw [stepStmts.cons_some (crhStore_step_eq _ h "b_h" "p_h"
    (s.pids 2 * s_h_h + t * K * V) s_h_t (s.pids 0 * BK) (s.pids 1 * BV) K V BK BV
    (fun e p => crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      H T K V BT BK BV NT t e p)
    (crhHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      H T K V BT BK BV NT t)
    (fun e p => rfl)
    (by simpa using hbh)
    (by simp))]
  obtain ⟨hSpids, hSregs, hSact, hSoth, hSoff⟩ :=
    crhStore_step_props
      (((sin.setReg "p_k" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, t * BT]⟩ : Tile .blockPtr [BK, BT])).setReg
        "p_v" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [t * BT, s.pids 1 * BV]⟩ : Tile .blockPtr [BT, BV])).setReg
        "p_h" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk h (s.pids 2 * s_h_h + t * K * V) [K, V] [BK, BV]
            [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
      h (s.pids 2 * s_h_h + t * K * V) s_h_t (s.pids 0 * BK) (s.pids 1 * BV) K V BK BV
      (fun e p => crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        H T K V BT BK BV NT t e p)
      (crhStoreAddr_injective (s.pids 2 * s_h_h + t * K * V) s_h_t
        (s.pids 0 * BK) (s.pids 1 * BV) BK BV hσ)
  set sSt := crhStoreState
      (((sin.setReg "p_k" .blockPtr [BK, BT]
          (⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT]
            [s_qk_d, s_qk_t] [s.pids 0 * BK, t * BT]⟩ : Tile .blockPtr [BK, BT])).setReg
        "p_v" .blockPtr [BT, BV]
          (⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV]
            [s_vo_t, s_vo_d] [t * BT, s.pids 1 * BV]⟩ : Tile .blockPtr [BT, BV])).setReg
        "p_h" .blockPtr [BK, BV]
          (⟨fun _ => BlockPtr.mk h (s.pids 2 * s_h_h + t * K * V) [K, V] [BK, BV]
            [s_h_t, 1] [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
      h (s.pids 2 * s_h_h + t * K * V) s_h_t (s.pids 0 * BK) (s.pids 1 * BV) K V BK BV
      (fun e p => crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        H T K V BT BK BV NT t e p) with hsSt
  -- b_k load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_kLoad_eq s sSt k "p_k" s_qk_h s_qk_t s_qk_d T K BT BK t
      (fun off => by
        rw [hSoth k off (Ne.symm hHk)]
        simpa using hmemK off)
      (by rw [hSregs]; simp)))]
  -- b_v load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_vLoad_eq s _ v "p_v" s_vo_h s_vo_t s_vo_d T V BT BV t
      (fun off => by
        rw [BlockState.setReg_readMem, hSoth v off (Ne.symm hHv)]
        simpa using hmemV off)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simp)))]
  -- the ragged-boundary gate
  obtain ⟨sG, hgstep, hgpids, hgmem, hgregs, hgdb, hgdi⟩ :=
    crhBoundaryGate_run s
      ((sSt.setReg "b_k" .real [BK, BT]
          (crhBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t)).setReg
        "b_v" .real [BT, BV] (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t))
      H T BT NT t
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hit)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hoi)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hbb)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hdb)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hdi)
  rw [stepStmts.cons_some hgstep]
  -- b_h advance
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_fwd_advance_eval s sG k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
      s_vo_d H T K V BT BK BV NT t hgdb hgdi
      (by
        rw [hgregs "b_h" (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
        simpa using hbh)
      (by
        rw [hgregs "b_k" (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])
      (by
        rw [hgregs "b_v" (by decide) (by decide), BlockState.setReg_same])))]
  rw [stepStmts.nil]
  have hgmem' : ∀ rg off, sG.readMem rg off = sSt.readMem rg off := by
    intro rg off
    rw [hgmem rg off, BlockState.setReg_readMem, BlockState.setReg_readMem]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    rw [BlockState.setReg_pids, hgpids, BlockState.setReg_pids,
      BlockState.setReg_pids, hSpids, BlockState.setReg_pids,
      BlockState.setReg_pids, BlockState.setReg_pids]
  · -- i_k
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hgregs "i_k" (by decide) (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hik
  · -- i_v
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hgregs "i_v" (by decide) (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hiv
  · -- i_bh
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hgregs "i_bh" (by decide) (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hibh
  · -- o_i
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hgregs "o_i" (by decide) (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hoi
  · -- b_b
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hgregs "b_b" (by decide) (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hSregs]
    simpa using hbb
  · -- d_b
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hgdb
  · -- d_i
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hgdi
  · -- b_h
    rw [BlockState.setReg_same]
  · -- other regions unchanged
    intro rg off hrg
    rw [BlockState.setReg_readMem, hgmem' rg off, hSoth rg off hrg]
    simp
  · -- chunk-t active readback
    intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    rw [BlockState.setReg_readMem, hgmem' h _, crhHOffset_eq_storeAddr]
    exact hSact idx ⟨h1, h2⟩
  · -- h off the active block unchanged
    intro off hoff
    rw [BlockState.setReg_readMem, hgmem' h off,
      hSoff off (fun idx hidx => by
        rw [← crhHOffset_eq_storeAddr]
        exact hoff idx hidx)]
    simp

set_option maxHeartbeats 4000000 in
/-- **The collapsed forward chunk loop.** -/
theorem crhFwdLoop_run (s sPre : BlockState) (k v h h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat)
    (hHk : h ≠ k) (hHv : h ≠ v) (hσ : BV ≤ s_h_t)
    (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (hpids : sPre.pids = s.pids)
    (hik : sPre.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sPre.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sPre.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hoi : sPre.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : sPre.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H))))
    (hdb : sPre.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT))))
    (hdi : sPre.regs .real [BT] "d_i" = some (crhDiStdTile s H BT))
    (hbh : sPre.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT 0))
    (hmem : ∀ rg off, sPre.readMem rg off = s.readMem rg off) :
    ∃ sL, stepStmt (Stmt.forRange "i_t" 0 NT 1
        (crhFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
          T K V BT BK BV NT)) sPre = some sL
      ∧ sL.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ sL.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ sL.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ sL.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT NT)
      ∧ (∀ rg off, rg ≠ h → sL.readMem rg off = s.readMem rg off)
      ∧ (∀ j (idx : TileIndex [BK, BV]), j < NT → crhActive s K V BK BV idx →
          sL.readMem h (crhHOffset s s_h_h s_h_t K V BK BV j idx)
            = crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                H T K V BT BK BV NT j idx.1.val idx.2.1.val) := by
  have hres := forRange_inv (idx := "i_t") (start := 0) (stop := NT) (step := 1)
    (body := crhFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
      s_h_t T K V BT BK BV NT)
    (P := fun c tS =>
      c ≤ NT
      ∧ tS.pids = s.pids
      ∧ tS.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ tS.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ tS.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ tS.regs .nat [BT] "o_i" = some (crhOiTile BT)
      ∧ tS.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))
      ∧ (c < NT →
          tS.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT)))
          ∧ tS.regs .real [BT] "d_i" = some (crhDiStdTile s H BT))
      ∧ tS.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT c)
      ∧ (∀ rg off, rg ≠ h → tS.readMem rg off = s.readMem rg off)
      ∧ (∀ j (idx : TileIndex [BK, BV]), j < c → crhActive s K V BK BV idx →
          tS.readMem h (crhHOffset s s_h_h s_h_t K V BK BV j idx)
            = crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                H T K V BT BK BV NT j idx.1.val idx.2.1.val))
    (s_init := sPre) (by decide)
    ⟨Nat.zero_le _, hpids, hik, hiv, hibh, hoi, hbb, fun _ => ⟨hdb, hdi⟩, hbh,
      fun rg off _ => hmem rg off,
      fun j idx hj _ => absurd hj (Nat.not_lt_zero j)⟩
    (by
      intro i tS hi hP
      obtain ⟨hle, hPpids, hPik, hPiv, hPibh, hPoi, hPbb, hPd, hPbh, hPoth,
        hPhist⟩ := hP
      obtain ⟨hPdb, hPdi⟩ := hPd hi
      obtain ⟨s', hstep, hspids, hsik, hsiv, hsibh, hsoi, hsbb, hsdb, hsdi,
        hsbh, hsoth, hsact, hsoff⟩ :=
        crhFwdBody_step s (tS.setReg "i_t" .nat [] (Tile.scalar i)) k v h h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
          H T K V BT BK BV NT i hHk hHv hσ
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth k off (Ne.symm hHk))
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPoth v off (Ne.symm hHv))
          (by simpa using hPik)
          (by simpa using hPiv)
          (by simpa using hPibh)
          (by simp)
          (by simpa using hPoi)
          (by simpa using hPbb)
          (by simpa using hPdb)
          (by simpa using hPdi)
          (by simpa using hPbh)
      refine ⟨s', hstep, by omega,
        by rw [hspids, BlockState.setReg_pids]; exact hPpids,
        hsik, hsiv, hsibh, hsoi, hsbb, ?_, hsbh, ?_, ?_⟩
      · -- d_b / d_i standard again whenever another iteration remains
        intro hlt
        have hnb : ¬(i = NT - 1 ∧ T % BT ≠ 0) := by
          intro hb
          omega
        constructor
        · rw [hsdb, crhDb_std s H T BT NT i hnb]
        · rw [hsdi, crhDiFull_std s H T BT NT i hnb]
      · intro rg off hrg
        rw [hsoth rg off hrg, BlockState.setReg_readMem]
        exact hPoth rg off hrg
      · intro j idx hj hidx
        rcases Nat.lt_or_ge j i with hji | hji
        · rw [hsoff (crhHOffset s s_h_h s_h_t K V BK BV j idx)
            (fun idx' hidx' => crhHOffset_chunk_disjoint s s_h_h s_h_t K V BK BV
              hFit j i (by omega) idx idx' hidx hidx'),
            BlockState.setReg_readMem]
          exact hPhist j idx hji hidx
        · have hji' : j = i := by omega
          subst hji'
          exact hsact idx hidx)
  obtain ⟨final, sL, hstep, hfin, hle, _, hLik, hLiv, hLibh, _, _, _, hLbh,
    hLoth, hLhist⟩ := hres
  have hfinal : final = NT := Nat.le_antisymm hle hfin
  subst hfinal
  exact ⟨sL, hstep, hLik, hLiv, hLibh, hLbh, hLoth, hLhist⟩

/-- **The `USE_INITIAL_STATE` gate.** -/
theorem crhInitBranch_run (s s0 : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat)
    (hpids : s0.pids = s.pids)
    (hik : s0.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : s0.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : s0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hoi : s0.regs .nat [BT] "o_i" = some (crhOiTile BT))
    (hbb : s0.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H))))
    (hdb : s0.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT))))
    (hdi : s0.regs .real [BT] "d_i" = some (crhDiStdTile s H BT))
    (hbh : s0.regs .real [BK, BV] "b_h"
      = some (⟨fun _ => some 0⟩ : Tile .real [BK, BV]))
    (hmem : ∀ off, s0.readMem h0 off = s.readMem h0 off) :
    ∃ s1, stepStmt (Stmt.ifThen (Op.constBool UIS) (crhInitBranch h0 K V BK BV)) s0
        = some s1
      ∧ s1.pids = s.pids
      ∧ (∀ rg off, s1.readMem rg off = s0.readMem rg off)
      ∧ s1.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s1.regs .nat [BT] "o_i" = some (crhOiTile BT)
      ∧ s1.regs .real [] "b_b" = some (Tile.scalar (some (crhBeta s H)))
      ∧ s1.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT)))
      ∧ s1.regs .real [BT] "d_i" = some (crhDiStdTile s H BT)
      ∧ s1.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
          s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT 0) := by
  cases UIS
  · refine ⟨s0, crh_ifThen_false_noop _ _, hpids, fun _ _ => rfl,
      hik, hiv, hibh, hoi, hbb, hdb, hdi, ?_⟩
    rw [hbh]
    refine congrArg some (Tile.ext fun idx => ?_)
    simp [crhHTile, crhState]
  ·
      rw [crh_ifThen_true_run]
      unfold crhInitBranch
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_makeBlockPtr_2d_eval h0 s0 _ _ _ [K, V] [BK, BV] [V, 1]
          (s.pids 2 * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
          (crh_mulMulConst_eval s0 "i_bh" (s.pids 2) K V hibh)
          (crh_mulConst_eval s0 "i_k" (s.pids 0) BK hik)
          (crh_mulConst_eval s0 "i_v" (s.pids 1) BV hiv)))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_castFloat_eval _ _ _
          (crh_h0Load_eq s _ h0 K V BK BV
            (fun off => by simpa using hmem off)
            (by simp))))]
      rw [stepStmts.nil]
      refine ⟨_, rfl, by simp [hpids], fun rg off => by simp, ?_, ?_, ?_, ?_,
        ?_, ?_, ?_, ?_⟩
      · simpa using hik
      · simpa using hiv
      · simpa using hibh
      · simpa using hoi
      · simpa using hbb
      · simpa using hdb
      · simpa using hdi
      · rw [BlockState.setReg_same]
        refine congrArg some (Tile.ext fun idx => ?_)
        simp [crhH0Tile, crhHTile, crhState]

/-- **The `STORE_FINAL_STATE` gate.** -/
theorem crhFinalBranch_run (s sL : BlockState) (k v h0 ht : RegionName)
    (UIS SFS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (H T K V BT BK BV NT : Nat) (hBVV : BV ≤ V)
    (hik : sL.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sL.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sL.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hbh : sL.regs .real [BK, BV] "b_h" = some (crhHTile s k v h0 UIS
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT NT)) :
    ∃ sF, stepStmt (Stmt.ifThen (Op.constBool SFS) (crhFinalBranch ht K V BK BV)) sL
        = some sF
      ∧ (∀ rg off, rg ≠ ht → sF.readMem rg off = sL.readMem rg off)
      ∧ (SFS = Bool.true → ∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
          sF.readMem ht (crhHtOffset s K V BK BV idx)
            = crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
                H T K V BT BK BV NT NT idx.1.val idx.2.1.val) := by
  cases SFS
  · exact ⟨sL, crh_ifThen_false_noop _ _, fun _ _ _ => rfl,
      fun hSFS => nomatch hSFS⟩
  ·
      rw [crh_ifThen_true_run]
      unfold crhFinalBranch
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (crh_makeBlockPtr_2d_eval ht sL _ _ _ [K, V] [BK, BV] [V, 1]
          (s.pids 2 * K * V) (s.pids 0 * BK) (s.pids 1 * BV)
          (crh_mulMulConst_eval sL "i_bh" (s.pids 2) K V hibh)
          (crh_mulConst_eval sL "i_k" (s.pids 0) BK hik)
          (crh_mulConst_eval sL "i_v" (s.pids 1) BV hiv)))]
      rw [stepStmts.cons_some (crhStore_step_eq _ ht "b_h" "p_ht"
        (s.pids 2 * K * V) V (s.pids 0 * BK) (s.pids 1 * BV) K V BK BV
        (fun e p => crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
          s_vo_d H T K V BT BK BV NT NT e p)
        (crhHTile s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          H T K V BT BK BV NT NT)
        (fun e p => rfl)
        (by simpa using hbh)
        (by simp))]
      rw [stepStmts.nil]
      obtain ⟨hFpids, hFregs, hFact, hFoth, hFoff⟩ :=
        crhStore_step_props
          (sL.setReg "p_ht" .blockPtr [BK, BV]
            (⟨fun _ => BlockPtr.mk ht (s.pids 2 * K * V) [K, V] [BK, BV] [V, 1]
              [s.pids 0 * BK, s.pids 1 * BV]⟩ : Tile .blockPtr [BK, BV]))
          ht (s.pids 2 * K * V) V (s.pids 0 * BK) (s.pids 1 * BV) K V BK BV
          (fun e p => crhState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
            s_vo_d H T K V BT BK BV NT NT e p)
          (crhStoreAddr_injective (s.pids 2 * K * V) V
            (s.pids 0 * BK) (s.pids 1 * BV) BK BV hBVV)
      refine ⟨_, rfl, ?_, ?_⟩
      · intro rg off hrg
        rw [hFoth rg off hrg]
        simp
      · intro _ idx hidx
        obtain ⟨h1, h2⟩ := hidx
        rw [crhHtOffset_eq_storeAddr]
        exact hFact idx ⟨h1, h2⟩

set_option maxHeartbeats 1000000 in
/-- **Genuine, dimension-general correctness** of the file's first kernel,
`chunk_retention_fwd_kernel_h`. For every launch state the kernel runs to
completion; every active lane of every chunk block `h[·,·,t]` (`t < NT`) holds
the **pre-chunk** state `crhState t` of the decayed recurrence
`H_{t+1} = d_b(t)·H_t + k_tᵀ·(v_t ⊙ d_i(t))` — including the ragged last
chunk, whose in-loop `d_b`/`d_i` rebind gives the final step its own decay
length `T % BT` — and, when `STORE_FINAL_STATE` is set, `ht` holds
`crhState NT`. One theorem covers all four gate configurations; every
dimension, stride, the head count `H`, and the chunk count `NT` stay
symbolic. -/
specification crh_fwd_h_exec_genuine
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (s : BlockState)
    (hHk : h ≠ k) (hHv : h ≠ v) (hHtH : ht ≠ h)
    (hσ : BV ≤ s_h_t) (hFit : (K - 1) * s_h_t + V ≤ K * V) (hBVV : BV ≤ V) :
    ∃ sF, exec (crh_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t
        s_vo_d s_h_h s_h_t H T K V BT BK BV NT
        USE_INITIAL_STATE STORE_FINAL_STATE).toAlgKernel s = some sF
      ∧ (∀ t (idx : TileIndex [BK, BV]), t < NT → crhActive s K V BK BV idx →
          sF.readMem h (crhHOffset s s_h_h s_h_t K V BK BV t idx)
            = crhState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT t idx.1.val idx.2.1.val)
      ∧ (STORE_FINAL_STATE = Bool.true →
          ∀ idx : TileIndex [BK, BV], crhActive s K V BK BV idx →
          sF.readMem ht (crhHtOffset s K V BK BV idx)
            = crhState s k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
                s_vo_t s_vo_d H T K V BT BK BV NT NT idx.1.val idx.2.1.val) := by
  obtain ⟨sP, hpre, hPpids, hPik, hPiv, hPibh, hPoi, hPbb, hPdb, hPmem⟩ :=
    crhPreDecay_run s H BT
  rw [exec, crh_fwd_body_eq, hpre]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_diStd_eval s sP H BT hPoi hPbb))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (crh_zeros_eval BK BV _))]
  obtain ⟨s1, hinit, h1pids, h1mem, h1ik, h1iv, h1ibh, h1oi, h1bb, h1db, h1di,
    h1bh⟩ :=
    crhInitBranch_run s
      ((sP.setReg "d_i" .real [BT] (crhDiStdTile s H BT)).setReg
        "b_h" .real [BK, BV] (⟨fun _ => some 0⟩ : Tile .real [BK, BV]))
      k v h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      H T K V BT BK BV NT
      (by simp [hPpids])
      (by simpa using hPik)
      (by simpa using hPiv)
      (by simpa using hPibh)
      (by simpa using hPoi)
      (by simpa using hPbb)
      (by simpa using hPdb)
      (by simp)
      (by simp)
      (fun off => by simpa using hPmem h0 off)
  rw [stepStmts.cons_some hinit]
  obtain ⟨sL, hloop, hLik, hLiv, hLibh, hLbh, hLoth, hLhist⟩ :=
    crhFwdLoop_run s s1 k v h h0 USE_INITIAL_STATE s_qk_h s_qk_t s_qk_d s_vo_h
      s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT hHk hHv hσ hFit
      h1pids h1ik h1iv h1ibh h1oi h1bb h1db h1di h1bh
      (fun rg off => by
        rw [h1mem rg off]
        simpa using hPmem rg off)
  rw [stepStmts.cons_some hloop]
  obtain ⟨sF, hfin, hFoth, hFht⟩ :=
    crhFinalBranch_run s sL k v h0 ht USE_INITIAL_STATE STORE_FINAL_STATE
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d H T K V BT BK BV NT hBVV
      hLik hLiv hLibh hLbh
  rw [stepStmts.cons_some hfin, stepStmts.nil]
  refine ⟨sF, rfl, ?_, hFht⟩
  intro t idx htNT hidx
  rw [hFoth h _ (Ne.symm hHtH)]
  exact hLhist t idx htNT hidx

/-! ## The backward walk -/

/-- The backward accumulator tile after `c` descending iterations. -/
noncomputable def crhDhAccTile (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (c : Nat) :
    Tile .real [BT, BT] :=
  ⟨fun idx => some (crhDhAcc s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT c
    idx.1.val idx.2.1.val)⟩

/-- The backward accumulation advances the indicator carry by one chunk. -/
private theorem crh_bwd_advance_eval (s sin : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) (c : Nat) (hc : c < NT)
    (hbdh : sin.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
      s_vo_h s_vo_t s_vo_d H T V BT NT c))
    (hbo : sin.regs .real [BT, BT] "b_o"
      = some (crhBvTile s do_ s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)))
    (hbv : sin.regs .real [BT, BT] "b_v"
      = some (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)))
    (hdi : sin.regs .real [BT] "d_i" = some (crhDiBwdTile s H BT)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_dh")
        (Op.dot (batch := [])
          (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [BT, BT] "b_o")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i")))
          (Op.ref .real [BT, BT] "b_v"))) sin
      = some (crhDhAccTile s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT (c + 1)) := by
  have hwo : evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ref .real [BT, BT] "b_o")
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i"))) sin
      = some (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (crhBvTile s do_ s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c))
        (Tile.expandDim ⟨1, by simp⟩ (crhDiBwdTile s H BT))) :=
    crh_mulTile_eval (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      _ _ sin _ _
      (by rw [evalOp_ref]; exact hbo)
      (by
        erw [evalOp_expandDim]
        simp only [evalOp_ref, hdi, Option.bind_eq_bind, Option.bind_some]
        rfl)
  have hye : evalOp (Op.dot (batch := [])
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BT] "b_o")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "d_i")))
      (Op.ref .real [BT, BT] "b_v")) sin
      = some (Tile.dot []
        (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (crhBvTile s do_ s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c))
          (Tile.expandDim ⟨1, by simp⟩ (crhDiBwdTile s H BT)))
        (crhBvTile s v s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c))) :=
    crh_dot_eval _ _ sin _ _ hwo (by rw [evalOp_ref]; exact hbv)
  erw [crh_addTile_eval (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    _ _ sin _ _ (by rw [evalOp_ref]; exact hbdh) hye]
  refine congrArg some (Tile.ext fun idx => ?_)
  obtain ⟨x, y, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    crhDhAccTile, NumericDType.add, WithBot.realAdd]
  rw [crh_dot2d_elem _ _ x y
    (fun r => crhVGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)
      x.val r.val * crhDiBwd s H x.val)
    (fun r => crhVGuarded s v s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)
      r.val y.val)
    (fun r => rfl) (fun r => rfl)]
  rw [show ∀ a b : ℝ, Option.map₂ (· + ·) (some a : WithBot ℝ) (some b) = some (a + b)
    from fun _ _ => rfl]
  refine congrArg some ?_
  rw [crhDhAcc_succ s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT c x.val y.val hc]
  rfl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **One backward chunk** (ascending counter `c`, descending chunk `NT-1-c`):
three pointer makes, three loads (the `dh` one dead), and the weighted
accumulation. No store — memory is untouched. -/
theorem crhBwdBody_step (s sin : BlockState) (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) (c : Nat) (hc : c < NT)
    (hmemDo : ∀ off, sin.readMem do_ off = s.readMem do_ off)
    (hmemV : ∀ off, sin.readMem v off = s.readMem v off)
    (hik : sin.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sin.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sin.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hj : sin.regs .nat [] "j" = some (Tile.scalar c))
    (hdi : sin.regs .real [BT] "d_i" = some (crhDiBwdTile s H BT))
    (hbdh : sin.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
      s_vo_h s_vo_t s_vo_d H T V BT NT c)) :
    ∃ s', stepStmts (crhBwdBody v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t
        T K V BT NT) sin = some s'
      ∧ s'.pids = sin.pids
      ∧ (∀ rg off, s'.readMem rg off = sin.readMem rg off)
      ∧ s'.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s'.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ s'.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s'.regs .nat [] "i_t" = some (Tile.scalar (NT - 1 - c))
      ∧ s'.regs .real [BT] "d_i" = some (crhDiBwdTile s H BT)
      ∧ s'.regs .real [] "d_b" = sin.regs .real [] "d_b"
      ∧ s'.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
          s_vo_h s_vo_t s_vo_d H T V BT NT (c + 1)) := by
  unfold crhBwdBody
  -- i_t = NT - 1 - j
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (crh_itIdx_eval sin NT c hj))]
  -- p_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval do_ _ _ _ _ [T, V] [BT, BT] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) ((NT - 1 - c) * BT) (s.pids 1 * BT)
      (crh_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simpa using hibh))
      (crh_mulConst_eval _ "i_t" (NT - 1 - c) BT (by simp))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BT (by simpa using hiv))))]
  -- p_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BT, BT] [s_vo_t, s_vo_d]
      (s.pids 2 * s_vo_h) ((NT - 1 - c) * BT) (s.pids 1 * BT)
      (crh_mulConst_eval _ "i_bh" (s.pids 2) s_vo_h (by simpa using hibh))
      (crh_mulConst_eval _ "i_t" (NT - 1 - c) BT (by simp))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BT (by simpa using hiv))))]
  -- p_h (the dead dh pointer)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval dh _ _ _ _ [K, V] [BT, BT] [s_h_t, 1]
      (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) (s.pids 0 * BT) (s.pids 1 * BT)
      (crh_hBase_eval _ s_h_h K V (s.pids 2) (NT - 1 - c) (by simpa using hibh)
        (by simp))
      (crh_mulConst_eval _ "i_k" (s.pids 0) BT (by simpa using hik))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BT (by simpa using hiv))))]
  -- b_o load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_vLoad_eq s _ do_ "p_o" s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)
      (fun off => by
        simp only [BlockState.setReg_readMem]
        exact hmemDo off)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])))]
  -- b_v load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_vLoad_eq s _ v "p_v" s_vo_h s_vo_t s_vo_d T V BT BT (NT - 1 - c)
      (fun off => by
        simp only [BlockState.setReg_readMem]
        exact hmemV off)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])))]
  -- b_h dead load (the raw guarded tile)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_load_bp_2d dh _ "p_h" (s.pids 2 * s_h_h + (NT - 1 - c) * K * V) K V
      BT BT s_h_t 1 (s.pids 0 * BT) (s.pids 1 * BT)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])))]
  -- the accumulation
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_bwd_advance_eval s _ v do_ s_vo_h s_vo_t s_vo_d H T V BT NT c hc
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        simpa using hbdh)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same])
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        simpa using hdi)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]
  · intro rg off
    simp only [BlockState.setReg_readMem]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hik
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hiv
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hibh
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_same]
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hdi
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
  · rw [BlockState.setReg_same]

set_option maxHeartbeats 4000000 in
/-- **The collapsed backward loop.** Memory is untouched; the accumulator ends
at `crhDhAcc NT` and the loop register `i_t` ends at `0`. -/
theorem crhBwdLoop_run (s sPre : BlockState) (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT NT : Nat)
    (hpids : sPre.pids = s.pids)
    (hik : sPre.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (hiv : sPre.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1)))
    (hibh : sPre.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hit : sPre.regs .nat [] "i_t" = some (Tile.scalar 0))
    (hdb : sPre.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT))))
    (hdi : sPre.regs .real [BT] "d_i" = some (crhDiBwdTile s H BT))
    (hbdh : sPre.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
      s_vo_h s_vo_t s_vo_d H T V BT NT 0))
    (hmem : ∀ rg off, sPre.readMem rg off = s.readMem rg off) :
    ∃ sL, stepStmt (Stmt.forRange "j" 0 NT 1
        (crhBwdBody v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT NT)) sPre
        = some sL
      ∧ sL.pids = s.pids
      ∧ sL.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ sL.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ sL.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ sL.regs .nat [] "i_t" = some (Tile.scalar 0)
      ∧ sL.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT)))
      ∧ sL.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
          s_vo_h s_vo_t s_vo_d H T V BT NT NT)
      ∧ (∀ rg off, sL.readMem rg off = s.readMem rg off) := by
  have hres := forRange_inv (idx := "j") (start := 0) (stop := NT) (step := 1)
    (body := crhBwdBody v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t T K V BT NT)
    (P := fun c tS =>
      c ≤ NT
      ∧ tS.pids = s.pids
      ∧ tS.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ tS.regs .nat [] "i_v" = some (Tile.scalar (s.pids 1))
      ∧ tS.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ tS.regs .nat [] "i_t"
          = some (Tile.scalar (if c = 0 then 0 else NT - c))
      ∧ tS.regs .real [] "d_b" = some (Tile.scalar (some (crhDbBwd s H BT)))
      ∧ tS.regs .real [BT] "d_i" = some (crhDiBwdTile s H BT)
      ∧ tS.regs .real [BT, BT] "b_dh" = some (crhDhAccTile s v do_
          s_vo_h s_vo_t s_vo_d H T V BT NT c)
      ∧ (∀ rg off, tS.readMem rg off = s.readMem rg off))
    (s_init := sPre) (by decide)
    ⟨Nat.zero_le _, hpids, hik, hiv, hibh, by simpa using hit, hdb, hdi, hbdh,
      fun rg off => hmem rg off⟩
    (by
      intro i tS hi hP
      obtain ⟨hle, hPpids, hPik, hPiv, hPibh, hPit, hPdb, hPdi, hPbdh,
        hPmem⟩ := hP
      obtain ⟨s', hstep, hspids, hsmem, hsik, hsiv, hsibh, hsit, hsdi, hsdb,
        hsbdh⟩ :=
        crhBwdBody_step s (tS.setReg "j" .nat [] (Tile.scalar i)) v do_ dh
          s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT NT i hi
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPmem do_ off)
          (fun off => by
            rw [BlockState.setReg_readMem]
            exact hPmem v off)
          (by simpa using hPik)
          (by simpa using hPiv)
          (by simpa using hPibh)
          (by simp)
          (by simpa using hPdi)
          (by simpa using hPbdh)
      refine ⟨s', hstep, by omega,
        by rw [hspids, BlockState.setReg_pids]; exact hPpids,
        hsik, hsiv, hsibh, ?_, ?_, hsdi, hsbdh, ?_⟩
      · rw [hsit]
        refine congrArg some (congrArg Tile.scalar ?_)
        rw [if_neg (Nat.succ_ne_zero i)]
        show NT - 1 - i = NT - (i + 1)
        omega
      · rw [hsdb]
        simpa using hPdb
      · intro rg off
        rw [hsmem rg off, BlockState.setReg_readMem]
        exact hPmem rg off)
  obtain ⟨final, sL, hstep, hfin, hle, hLpids, hLik, hLiv, hLibh, hLit, hLdb,
    _, hLbdh, hLmem⟩ := hres
  have hfinal : final = NT := Nat.le_antisymm hle hfin
  rw [hfinal] at hLit hLbdh
  refine ⟨sL, hstep, hLpids, hLik, hLiv, hLibh, ?_, hLdb, hLbdh, hLmem⟩
  rw [hLit]
  refine congrArg some (congrArg Tile.scalar ?_)
  show (if NT = 0 then 0 else NT - NT) = 0
  by_cases h0 : NT = 0
  · rw [if_pos h0]
  · rw [if_neg h0]
    omega

/-- The scaled output tile: cell `(x, y)` is `crhDhOut`. -/
noncomputable def crhDhOutTile (s : BlockState) (v do_ : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (H T V BT NT : Nat) : Tile .real [BT, BT] :=
  ⟨fun idx => some (crhDhOut s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT
    idx.1.val idx.2.1.val)⟩

set_option maxHeartbeats 1000000 in
/-- **Genuine correctness** of `chunk_retention_bwd_kernel_dh`, exactly as the
upstream kernel computes (see the preamble's faithfulness notes: the square
shape is forced by its `tl.dot`, the contraction is the scrambled one, the
final store reuses the post-loop `i_t = 0`, and the `dh` loads are dead).
Every active lane of the single `dh` store block reads back
`crhDhOut = d_b · Σ_{all chunks} (do ⊙ d_i)·v`. -/
specification crh_bwd_dh_exec_genuine
    (v do_ dh : RegionName)
    (s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (H T K V BT NT : Nat) (s : BlockState)
    (hσ : BT ≤ s_h_t) :
    ∃ sF, exec (crh_bwd_dh_surface v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t
        H T K V BT NT).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BT, BT], crhDhActive s K V BT idx →
          sF.readMem dh (crhDhOffset s s_h_h s_h_t K V BT idx)
            = crhDhOut s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT
                idx.1.val idx.2.1.val := by
  obtain ⟨sP, hpre, hPpids, hPik, hPiv, hPibh, hPoi, hPbb, hPdb, hPmem⟩ :=
    crhPreDecay_run s H BT
  rw [exec, crh_bwd_body_eq, hpre]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_diBwd_eval s sP H BT hPoi hPbb))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (crh_zeros_eval BT BT _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _))]
  obtain ⟨sL, hloop, hLpids, hLik, hLiv, hLibh, hLit, hLdb, hLbdh, hLmem⟩ :=
    crhBwdLoop_run s
      (((sP.setReg "d_i" .real [BT] (crhDiBwdTile s H BT)).setReg
          "b_dh" .real [BT, BT] (⟨fun _ => some 0⟩ : Tile .real [BT, BT])).setReg
        "i_t" .nat [] (Tile.scalar 0))
      v do_ dh s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT NT
      (by simp [hPpids])
      (by simpa using hPik)
      (by simpa using hPiv)
      (by simpa using hPibh)
      (by simp)
      (by simpa using hPdb)
      (by simp)
      (by
        rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
          BlockState.setReg_same]
        refine congrArg some (Tile.ext fun idx => ?_)
        simp [crhDhAccTile, crhDhAcc_zero])
      (fun rg off => by simpa using hPmem rg off)
  rw [stepStmts.cons_some hloop]
  -- b_dh *= d_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BT] "b_dh")
        (Op.ref .real [] "d_b")) sL
      = some (crhDhOutTile s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT) from by
      rw [crh_mulTile_eval Broadcast.scalarR _ _ sL
        (crhDhAccTile s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT NT)
        (Tile.scalar (some (crhDbBwd s H BT)))
        (by rw [evalOp_ref]; exact hLbdh)
        (by rw [evalOp_ref]; exact hLdb)]
      refine congrArg some (Tile.ext fun idx => ?_)
      obtain ⟨x, y, u⟩ := idx
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        crhDhAccTile, crhDhOutTile, crhDhOut]
      rfl))]
  -- p_dh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (crh_makeBlockPtr_2d_eval dh _ _ _ _ [K, V] [BT, BT] [s_h_t, 1]
      (s.pids 2 * s_h_h + s.pids 0 * K * V) (s.pids 1 * BT) (0 * BT)
      (crh_dhBase_eval _ s_h_h K V (s.pids 2) (s.pids 0)
        (by simpa using hLibh) (by simpa using hLik))
      (crh_mulConst_eval _ "i_v" (s.pids 1) BT (by simpa using hLiv))
      (crh_mulConst_eval _ "i_t" 0 BT (by simpa using hLit))))]
  -- the single store
  rw [stepStmts.cons_some (crhStore_step_eq _ dh "b_dh" "p_dh"
    (s.pids 2 * s_h_h + s.pids 0 * K * V) s_h_t (s.pids 1 * BT) (0 * BT) K V BT BT
    (fun x y => crhDhOut s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT x y)
    (crhDhOutTile s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT)
    (fun x y => rfl)
    (by simp)
    (by rw [BlockState.setReg_same]))]
  rw [stepStmts.nil]
  obtain ⟨hFpids, hFregs, hFact, hFoth, hFoff⟩ :=
    crhStore_step_props
      (((sL.setReg "b_dh" .real [BT, BT]
          (crhDhOutTile s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT)).setReg
        "p_dh" .blockPtr [BT, BT]
          (⟨fun _ => BlockPtr.mk dh (s.pids 2 * s_h_h + s.pids 0 * K * V)
            [K, V] [BT, BT] [s_h_t, 1] [s.pids 1 * BT, 0 * BT]⟩ :
              Tile .blockPtr [BT, BT])))
      dh (s.pids 2 * s_h_h + s.pids 0 * K * V) s_h_t (s.pids 1 * BT) (0 * BT)
      K V BT BT
      (fun x y => crhDhOut s v do_ s_vo_h s_vo_t s_vo_d H T V BT NT x y)
      (crhStoreAddr_injective (s.pids 2 * s_h_h + s.pids 0 * K * V) s_h_t
        (s.pids 1 * BT) (0 * BT) BT BT hσ)
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  obtain ⟨h1, h2⟩ := hidx
  rw [crhDhOffset_eq_storeAddr]
  exact hFact idx ⟨h1, h2⟩


end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkRetention

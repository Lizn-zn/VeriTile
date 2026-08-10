import VeriTile.Triton

/-!
# `chunk_gla_fwd` — strict per-kernel correctness

`chunk_gla_fwd.py` holds **five** `@triton.jit` kernels: four that build the
intra-chunk attention matrix `A` (`_A_kernel_intra_sub_inter`,
`_A_kernel_intra_sub_intra`, `_A_kernel_intra_sub_intra_split`,
`_A_kernel_intra_sub_intra_merge`) and one that consumes it to produce the output.
**This file covers the output kernel, `chunk_gla_fwd_kernel_o`.** Covering a subset
of a multi-kernel file is the established shape here — `triton_linear_activation`
and `kv_cache_filling` each carry two headlines against five `@triton.jit` kernels.

`chunk_gla_fwd_kernel_o` is the gated-linear-attention chunked forward output step.
One program owns one `(i_v, i_t, i_bh)` block and computes

```
O[block] = Σ_{i_k} (scale · Q[block, i_k] ⊙ exp(G[block, i_k])) · H[i_k, block]
             + tril(A[block]) · V[block]
```

— the inter-chunk term accumulated over the K blocks with a per-element gate, plus
the intra-chunk term from the causally-masked `A` that the other four kernels built.

It is the gated sibling of the already-ported `chunk_gla_simple`
(`chunk_simple_gla_fwd_kernel_o`), and the block-pointer layout for `q` / `h` / `v` /
`o` is the same. Two differences drive everything below: the gate `g` here is a
**2-D `[T, K]` tensor** loaded per K block rather than a `[T]` vector applied once,
and `A` is **read from memory** rather than accumulated in registers.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the 3-D grid,
the host-computed `BT`/`BK`/`BV`, and the four `A`-building kernels that fill the `A`
region) is the *trusted boundary*. Every dimension, stride and the `scale` stays a
symbolic parameter. The `.to(...)` dtype round-trips erase to the identity at the
algorithm layer.

## Faithfulness note worth flagging

The source guards the inter-chunk accumulation with `if i_k >= 0:`, which is
**vacuously true** — `i_k` is a loop index over `range(tl.cdiv(K, BK))`. It is
transcribed as a real `Stmt.ifThen` on `Op.ge`, because it is in the source; reading
it as an unconditional accumulation would be a simplification rather than a
transcription, and the guard is observable in the statement list.

Two spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, both surface syntax
rather than semantics:

* integer literals inside index arithmetic are written `$(n)`, since a bare literal
  is inferred `.real` by the DSL's expression typing;
* `boundary_check=(0, 1)` is written `boundary_check=([0, 1] : List Nat)`, the
  spelling the DSL's `tl.make_block_ptr` loads take.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd

open VeriTile.Triton

section Correct_without_Rounding

def chunk_gla_fwd_o_surface
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
    b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    if i_k >= $(0) {
      b_o += tl.dot(b_qg, (b_h).to(b_qg.dtype))
    }
  }
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
    shape=($(T), $(BT)), strides=($(BT), $(1)),
    offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
  b_A = (tl.where(m_s, b_A, 0.0)).to(b_v.dtype)
  b_o += tl.dot(b_A, b_v, allow_tf32=false)
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

theorem chunk_gla_fwd_o_surface_toAlgorithm_supported
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ∃ alg, (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale T K V BT BK BV).toAlgorithm? = Except.ok alg := by
  simp [chunk_gla_fwd_o_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`
rather than assumed. Lowerings worth naming because they are not guessable from the
source text:

* `(x * scale)` with `scale : ℝ` a launch constant is `Op.mul .real Broadcast.scalarR`
  — the **tile is on the left**;
* `tl.exp(e)` is postfix `Op.exp`, and `.to(dtype)` **inside** an expression erases
  entirely (it is only a statement when the source writes it as one);
* `tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]` keeps both `tl.arange`
  calls — there is no shared register — and broadcasts with `consR (consL nil)`;
* `tl.where(m, x, 0.)` puts the scalar `other` through `Op.broadcast`;
* the `fp32` accumulator wrapper (`ComputeOp.alg ComputeDType.fp32 …`) is what
  `toAlgKernel` strips, so it does not appear here. -/

/-- The five statements before the K loop: the three program ids, the causal mask,
and the zeroed accumulator. -/
def cgfPreLoop (BT BV : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_v" (Op.programId 0),
    Stmt.assign .nat [] "i_t" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .bool [BT, BT] "m_s"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.arange BT))
        (Op.expandDim ⟨0, by simp⟩ (Op.arange BT))),
    Stmt.assign .real [BT, BV] "b_o" (Op.full [BT, BV] (Op.const 0)) ]

/-- The K-loop body: three block pointers, the scaled `q` load, the gate load, the
gated `q`, the `h` load, and the `i_k >= 0` guarded accumulation.

`if i_k >= 0` is **vacuously true** — `i_k` is a `Nat` — but it is in the source, so
it is transcribed as a real `Stmt.ifThen`. Reading it as an unconditional
accumulation would be a simplification, not a transcription. -/
def cgfLoopBody (g h q : RegionName) (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BT, BK] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h))
        [T, K] [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BT, BK] "p_g"
      (Op.makeBlockPtrDynOffsets g
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h))
        [T, K] [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
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
    Stmt.assign .real [BT, BK] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BK] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BK] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BT, BK] "b_q") (Op.const scale)),
    Stmt.assign .real [BT, BK] "b_g"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BK] "p_g") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BK] "b_qg"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_q") (Op.exp (Op.ref .real [BT, BK] "b_g"))),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1])
        MaskOpt.none),
    Stmt.ifThen
      (Op.ge ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 0))
      [ Stmt.assign .real [BT, BV] "b_o"
          (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Op.ref .real [BT, BV] "b_o")
            (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_qg")
              (Op.ref .real [BK, BV] "b_h"))) ] ]

/-- The tail: the `v` / `o` / `A` block pointers, the `v` and `A` loads, the causal
masking of `A`, the second `tl.dot`, and the store. -/
def cgfPostLoop (v o A : RegionName) (s_v_h s_v_t T V BT BV : Nat) : List Stmt :=
  [ Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h))
        [T, V] [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BT, BV] "p_o"
      (Op.makeBlockPtrDynOffsets o
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h))
        [T, V] [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BT, BT] "p_A"
      (Op.makeBlockPtrDynOffsets A
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
          (Op.constNat BT))
        [T, BT] [BT, BT] [BT, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.constNat 0]),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_A"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_A") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BT, BT] "b_A"
      (Op.where (Op.ref .bool [BT, BT] "m_s") (Op.ref .real [BT, BT] "b_A")
        (Op.broadcast (Op.const 0.0) [BT, BT])),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_A")
          (Op.ref .real [BT, BV] "b_v"))),
    Stmt.store .real [BT, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_o") [0, 1])
      (Op.ref .real [BT, BV] "b_o") MaskOpt.none ]

set_option maxRecDepth 20000 in
/-- **Full body split (by `rfl`).** The lowered surface is exactly
`cgfPreLoop ++ [forRangeDyn "i_k" 0 (cdiv K BK) 1 cgfLoopBody] ++ cgfPostLoop`
— 14 top-level statements, every one checked against the macro output. -/
theorem cgf_body_eq (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
        scale T K V BT BK BV).toAlgKernel.body
      = cgfPreLoop BT BV
        ++ [Stmt.forRangeDyn "i_k" (Op.constNat 0)
              (Op.div .nat Broadcast.nil
                (Op.sub .nat Broadcast.nil
                  (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
                  (Op.constNat 1))
                (Op.constNat BK))
              (Op.constNat 1)
              (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)]
        ++ cgfPostLoop v o A s_v_h s_v_t T V BT BV := by
  rfl

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkGlaFwd

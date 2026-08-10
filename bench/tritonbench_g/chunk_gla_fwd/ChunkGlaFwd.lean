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

/-! ## Closed-form specification

Every accessor is a raw read at the address its block pointer computes; the
boundary check lives in the *guarded* accessors below, matching the semantics of a
`boundary_check=(0, 1)` load, which yields `0` outside the parent shape.

The number of K blocks is `numKB K BK = cdiv(K, BK)`, and the spec sums over **all**
of them — this port does not assume the launcher's single-block regime, unlike the
ported `chunk_gla_simple`, whose headline carries `K = BK`. That is affordable here
because the kernel recomputes its three block pointers from `i_k` on every iteration
instead of advancing them, so the loop invariant has nothing to carry but the partial
sum. -/

/-- `cdiv(K, BK)` — the K-block trip count. -/
def numKB (K BK : Nat) : Nat := (K + BK - 1) / BK

/-- Global row (time) index of tile lane `i`: `i_t · BT + i`. -/
def tIndex (s : BlockState) (BT : Nat) (i : Nat) : Nat := s.pids 1 * BT + i

/-- Global key index of lane `e` in K block `kb`: `kb · BK + e`. -/
def kIndex (BK kb e : Nat) : Nat := kb * BK + e

/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
def vIndex (s : BlockState) (BV : Nat) (p : Nat) : Nat := s.pids 0 * BV + p

/-- `q[i, kIndex kb e]`, at the address `p_q` computes. -/
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT BK : Nat)
    (kb : Nat) (i e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + tIndex s BT i * s_k_t + kIndex BK kb e * 1)

/-- `g[i, kIndex kb e]` — the **2-D** gate, on `q`'s own layout (same base stride
`s_k_h`, same parent shape `(T, K)`). This is what separates this kernel from
`chunk_gla_simple`, whose gate is a `[T]` vector read once after the loop. -/
noncomputable def gElem (s : BlockState) (g : RegionName) (s_k_h s_k_t BT BK : Nat)
    (kb : Nat) (i e : Nat) : ℝ :=
  s.readMem g (s.pids 2 * s_k_h + tIndex s BT i * s_k_t + kIndex BK kb e * 1)

/-- `h[kIndex kb e, vIndex p]`, the chunk state at base `h + i_bh·s_h_h + i_t·K·V`. -/
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (kb BK : Nat) (e p : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + kIndex BK kb e * s_h_t
    + vIndex s BV p * 1)

/-- `v[j, vIndex p]`. -/
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + tIndex s BT j * s_v_t + vIndex s BV p * 1)

/-- `A[i, j]`, the intra-chunk attention matrix the other four kernels built, at base
`A + i_bh·T·BT` with parent shape `(T, BT)` and strides `(BT, 1)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName) (T BT : Nat)
    (i j : Nat) : ℝ :=
  s.readMem A (s.pids 2 * T * BT + tIndex s BT i * BT + j * 1)

/-! ### The guarded forms — what a `boundary_check=(0, 1)` load actually delivers -/

/-- The gated query lane, as the loop body's `b_qg` holds it:
`(q · scale) · exp(g)`, or `0` outside the `(T, K)` parent window. `q` and `g` share
one block-pointer geometry, so one guard covers both — and an out-of-bounds lane is
`0` regardless of the gate, since the load returns `0` and `0 · exp(0) = 0`. -/
noncomputable def qgElem (s : BlockState) (q g : RegionName)
    (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BK : Nat) (kb i e : Nat) : ℝ :=
  if tIndex s BT i < T ∧ kIndex BK kb e < K then
    qElem s q s_k_h s_k_t BT BK kb i e * scale
      * Real.exp (gElem s g s_k_h s_k_t BT BK kb i e)
  else 0

/-- The chunk-state lane, as `b_h` holds it. -/
noncomputable def hGuarded (s : BlockState) (h : RegionName)
    (s_h_h s_h_t : Nat) (K V BV : Nat) (kb BK e p : Nat) : ℝ :=
  if kIndex BK kb e < K ∧ vIndex s BV p < V then
    hElem s h s_h_h s_h_t K V BV kb BK e p
  else 0

/-- The value lane, as `b_v` holds it. -/
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_v_h s_v_t : Nat) (T V BT BV : Nat) (j p : Nat) : ℝ :=
  if tIndex s BT j < T ∧ vIndex s BV p < V then
    vElem s v s_v_h s_v_t BT BV j p
  else 0

/-- The intra-chunk matrix lane, as `b_A` holds it after `tl.where(m_s, b_A, 0.)`:
the causal mask `i ≥ j` on top of the row boundary check. The **column** check is
vacuous — `p_A`'s parent shape is `(T, BT)` and the lane index `j` is a `Fin BT` — so
only the row guard bites. -/
noncomputable def aMasked (s : BlockState) (A : RegionName) (T BT : Nat)
    (i j : Nat) : ℝ :=
  if j ≤ i then (if tIndex s BT i < T then aElem s A T BT i j else 0) else 0

/-- **The stored value.** Lane `(i, p)` of `b_o` after the K loop and the intra-chunk
`tl.dot`: the inter-chunk sum over every K block, plus the causally-masked `A · V`. -/
noncomputable def cgfOutput (s : BlockState) (q v g h A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (i p : Nat) : ℝ :=
  (∑ kb : Fin (numKB K BK), ∑ e : Fin BK,
      qgElem s q g s_k_h s_k_t scale T K BT BK kb.val i e.val
        * hGuarded s h s_h_h s_h_t K V BV kb.val BK e.val p)
    + ∑ j : Fin BT,
        aMasked s A T BT i j.val * vGuarded s v s_v_h s_v_t T V BT BV j.val p

/-- The output store address for lane `(i, p)` — `p_o` shares `p_v`'s layout. -/
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1.val * s_v_t + vIndex s BV idx.2.1.val * 1

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

/-! ## Per-statement eval recipes

Local by necessity — bench files never import each other, so the block-pointer
recipes proven for `chunk_gla_simple` are re-derived here under a `cgf_` prefix. -/

/-- **2D `makeBlockPtrDynOffsets` eval recipe.** Given the base / two offset ops
evaluate to scalars, the block-pointer op evaluates to the constant tile
`load_bp_2d_ref` consumes. -/
private theorem cgf_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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

/-- No-mask 2D block-pointer load through a bound register: lane `(i,j)` reads the
genuine memory cell when in-bounds, else `0`. -/
private theorem cgf_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
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

/-- Scalar `name * c`. -/
private theorem cgf_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The `p_h` base: `i_bh * s_h_h + i_t * K * V`, all scalar `nat` steps. -/
private theorem cgf_hBase_eval (s : BlockState) (s_h_h K V : Nat) (ibh it : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar it)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
          (Op.constNat V))) s
      = some (Tile.scalar (ibh * s_h_h + it * K * V)) := by
  rw [evalOp_add, cgf_mulConst_eval s "i_bh" ibh s_h_h hibh, evalOp_mul,
    cgf_mulConst_eval s "i_t" it K hit]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The `p_A` base: `i_bh * T * BT`. -/
private theorem cgf_aBase_eval (s : BlockState) (T BT : Nat) (ibh : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
        (Op.constNat BT)) s
      = some (Tile.scalar (ibh * T * BT)) := by
  rw [evalOp_mul, cgf_mulConst_eval s "i_bh" ibh T hibh]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Tile `*` with both operand values known — the library's `evalOp_mul` is a `do`
block, not a `rw` target once the operands are known. -/
private theorem cgf_mulTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul .real bc x y) t
      = some (Tile.bop NumericDType.real.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

/-- Tile `+`. -/
private theorem cgf_addTile_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (t : BlockState)
    (vx : Tile .real a) (vy : Tile .real b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add .real bc x y) t
      = some (Tile.bop NumericDType.real.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

/-- `tl.exp`. -/
private theorem cgf_exp_eval {sh : TileShape} (x : Op .real sh) (t : BlockState)
    (vx : Tile .real sh) (hx : evalOp x t = some vx) :
    evalOp (Op.exp x) t = some (Tile.uop WithBot.realExp vx) := by
  simp only [evalOp, hx]
  rfl

/-- `tl.where(m, x, other)`. -/
private theorem cgf_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

/-- Scalar fill (the `where`'s broadcast `0.0`). -/
private theorem cgf_broadcast_eval {dtype : TileDType} (e : Op dtype [])
    (sh : TileShape) (t : BlockState) (v : Tile dtype [])
    (hv : evalOp e t = some v) :
    evalOp (Op.broadcast e sh) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  simp only [evalOp, hv]
  rfl

/-- Evaluation unfolding for the `≥` comparison op. -/
private theorem cgf_ge_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- The loop guard `i_k >= 0` on `nat` scalars is `true`. -/
private theorem cgf_geGuard_eval (t : BlockState) (kb : Nat)
    (hk : t.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    evalOp (Op.ge ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_k")
        (Op.constNat 0)) t
      = some (Tile.scalar Bool.true) := by
  simp only [evalOp, evalOp_ref, hk, evalOp_constNat]
  rfl

/-- `Stmt.ifThen`'s step equation (well-founded recursion, so named). -/
private theorem cgf_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- A `WithBot ℝ` sum of `some`s is `some` of the real sum. -/
private theorem cgf_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ))
    = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- 2D dot element collapse for all-`some` operands. -/
private theorem cgf_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
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
  exact cgf_withBot_sum_some _

/-- `tl.dot` at rank 2, `erw`-only shapes. -/
private theorem cgf_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-! ## The loop body's value tiles -/

/-- `m_s` as the prologue leaves it: `true` exactly on the causal (lower) triangle. -/
def cgfMsTile (BT : Nat) : Tile .bool [BT, BT] :=
  ⟨fun idx => decide (idx.2.1.val ≤ idx.1.val)⟩

/-- The `m_s` statement lands on `cgfMsTile`. -/
private theorem cgf_ms_eval (t : BlockState) (BT : Nat) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.arange BT))
        (Op.expandDim ⟨0, by simp⟩ (Op.arange BT))) t
      = some (cgfMsTile BT) := by
  have h1N : evalOp (Op.expandDim (shape := [BT]) ⟨1, by simp⟩ (Op.arange BT)) t
      = some (Tile.expandDim (shape := [BT]) ⟨1, by simp⟩
          (Tile.vec fun i : Fin BT => (i.val : Nat))) := by
    rw [evalOp_expandDim, evalOp_arange]
    rfl
  -- restate at the literal shape so `rw` can hit the do-block's bind head
  have h1 : @evalOp .nat [BT, 1]
        (Op.expandDim (shape := [BT]) ⟨1, by simp⟩ (Op.arange BT)) t
      = some (Tile.expandDim (shape := [BT]) ⟨1, by simp⟩
          (Tile.vec fun i : Fin BT => (i.val : Nat))) := h1N
  have h0N : evalOp (Op.expandDim (shape := [BT]) ⟨0, by simp⟩ (Op.arange BT)) t
      = some (Tile.expandDim (shape := [BT]) ⟨0, by simp⟩
          (Tile.vec fun i : Fin BT => (i.val : Nat))) := by
    rw [evalOp_expandDim, evalOp_arange]
    rfl
  have h0 : @evalOp .nat [1, BT]
        (Op.expandDim (shape := [BT]) ⟨0, by simp⟩ (Op.arange BT)) t
      = some (Tile.expandDim (shape := [BT]) ⟨0, by simp⟩
          (Tile.vec fun i : Fin BT => (i.val : Nat))) := h0N
  rw [cgf_ge_def, h1, h0]
  show some _ = some (cgfMsTile BT)
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨i, j, u⟩ := idx
  simp only [cgfMsTile, Tile.cop_data, Tile.expandDim_data, Tile.vec,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex]
  show ComparableDType.nat.ge _ _ = _
  simp only [ComparableDType.ge]

/-- The raw `b_q` load at K block `kb` (scale and gate not yet applied). -/
noncomputable def cgfQRaw (s : BlockState) (q : RegionName)
    (s_k_h s_k_t T K BT BK kb : Nat) : Tile .real [BT, BK] :=
  ⟨fun idx => if tIndex s BT idx.1.val < T ∧ kIndex BK kb idx.2.1.val < K then
      some (qElem s q s_k_h s_k_t BT BK kb idx.1.val idx.2.1.val)
    else some 0⟩

/-- The raw `b_g` load — same geometry, `g` region. -/
noncomputable def cgfGRaw (s : BlockState) (g : RegionName)
    (s_k_h s_k_t T K BT BK kb : Nat) : Tile .real [BT, BK] :=
  ⟨fun idx => if tIndex s BT idx.1.val < T ∧ kIndex BK kb idx.2.1.val < K then
      some (gElem s g s_k_h s_k_t BT BK kb idx.1.val idx.2.1.val)
    else some 0⟩

/-- `b_qg` as statement 7 leaves it: the guarded, scaled, gated query lane. -/
noncomputable def cgfQgTile (s : BlockState) (q g : RegionName)
    (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BK : Nat) (kb : Nat) :
    Tile .real [BT, BK] :=
  ⟨fun idx => some (qgElem s q g s_k_h s_k_t scale T K BT BK kb
      idx.1.val idx.2.1.val)⟩

/-- `b_h` as statement 8 leaves it. -/
noncomputable def cgfHTile (s : BlockState) (h : RegionName)
    (s_h_h s_h_t K V BV : Nat) (kb BK : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (hGuarded s h s_h_h s_h_t K V BV kb BK idx.1.val idx.2.1.val)⟩

/-- `b_o` after `i` K blocks: the partial inter-chunk sum. -/
noncomputable def cgfAccTile (s : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
    (i : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (∑ kb : Fin i, ∑ e : Fin BK,
      qgElem s q g s_k_h s_k_t scale T K BT BK kb.val idx.1.val e.val
        * hGuarded s h s_h_h s_h_t K V BV kb.val BK e.val idx.2.1.val)⟩

/-- At `i = 0` the accumulator is the zero tile `tl.zeros` produces. -/
theorem cgfAccTile_zero (s : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat) :
    cgfAccTile s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV 0
      = (⟨fun _ => some 0⟩ : Tile .real [BT, BV]) := by
  apply Tile.ext
  intro idx
  simp [cgfAccTile]

/-- **The accumulator statement.** `b_o += tl.dot(b_qg, b_h)` extends the partial
sum by one K block. -/
theorem cgfAccTile_dot_succ (s : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV i : Nat) :
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (cgfAccTile s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i)
        (Tile.dot [] (cgfQgTile s q g s_k_h s_k_t scale T K BT BK i)
          (cgfHTile s h s_h_h s_h_t K V BV i BK))
      = cgfAccTile s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV (i + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, cgfAccTile,
    NumericDType.add, WithBot.realAdd]
  rw [cgf_dot2d_elem _ _ r p
    (fun e => qgElem s q g s_k_h s_k_t scale T K BT BK i r.val e.val)
    (fun e => hGuarded s h s_h_h s_h_t K V BV i BK e.val p.val)
    (fun e => rfl) (fun e => rfl)]
  rw [show ∀ x y : ℝ, Option.map₂ (· + ·) (some x : WithBot ℝ) (some y) = some (x + y)
    from fun _ _ => rfl]
  exact congrArg some (Fin.sum_univ_castSucc
    (f := fun kb : Fin (i + 1) => ∑ e : Fin BK,
      qgElem s q g s_k_h s_k_t scale T K BT BK kb.val r.val e.val
        * hGuarded s h s_h_h s_h_t K V BV kb.val BK e.val p.val)).symm

/-! ## The K-loop invariant

Deliberately small. The kernel recomputes `p_q` / `p_g` / `p_h` from `i_k` on every
iteration instead of advancing them, so nothing about pointers survives an
iteration — the invariant carries the launch memory, the block coordinates, the
causal mask, and the partial sum, and that is all. `i ≤ numKB` is included because
`forRangeDyn_inv` concludes only `stop ≤ final`; the two together pin
`final = numKB K BK`. -/

/-- `setReg` leaves memory alone, at function level (a deep tower's `t.mem = s.mem`
overruns `whnf` as a single `rfl`; nine cheap rewrites do not). -/
private theorem cgf_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- The state carried across K blocks. -/
noncomputable def cgfInv (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
    (i : Nat) (t : BlockState) : Prop :=
  i ≤ numKB K BK
  ∧ t.mem = s0.mem
  ∧ t.pids = s0.pids
  ∧ t.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 0))
  ∧ t.regs .nat [] "i_t" = some (Tile.scalar (s0.pids 1))
  ∧ t.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2))
  ∧ t.regs .bool [BT, BT] "m_s" = some (cgfMsTile BT)
  ∧ t.regs .real [BT, BV] "b_o"
      = some (cgfAccTile s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i)

/-- The loop combinator writes `i_k` before each iteration; `cgfInv` constrains no
register of that name, so it survives the write. -/
theorem cgfInv_setReg_k (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV i j : Nat)
    (t : BlockState)
    (hinv : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i t) :
    cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i
      (t.setReg "i_k" .nat [] (Tile.scalar j)) := by
  obtain ⟨hle, hmem, hpids, hiv, hit, hibh, hms, hbo⟩ := hinv
  exact ⟨hle, hmem, hpids, by simpa using hiv, by simpa using hit,
    by simpa using hibh, by simpa using hms, by simpa using hbo⟩

/-! ### The three loads, bridged to the named tiles -/

/-- The `b_q` load at K block `kb`, on the launch state's memory: the raw guarded
tile (the scale and gate have not been applied yet). -/
private theorem cgf_qLoad_eq (s0 : BlockState) (q : RegionName) (t : BlockState)
    (s_k_h s_k_t T K BT BK kb : Nat)
    (hmem : t.mem = s0.mem)
    (hpq : t.regs .blockPtr [BT, BK] "p_q" = some
      ⟨fun _ => BlockPtr.mk q (s0.pids 2 * s_k_h) [T, K] [BT, BK] [s_k_t, 1]
        [s0.pids 1 * BT, kb * BK]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BK] "p_q") [0, 1]) MaskOpt.none) t
      = some (cgfQRaw s0 q s_k_h s_k_t T K BT BK kb) := by
  rw [cgf_load_bp_2d q t "p_q" (s0.pids 2 * s_k_h) T K BT BK s_k_t 1
    (s0.pids 1 * BT) (kb * BK) hpq]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [cgfQRaw, qElem, tIndex, kIndex, BlockState.readMem, hmem]

/-- The `b_g` load — same geometry as `b_q`, on the `g` region. -/
private theorem cgf_gLoad_eq (s0 : BlockState) (g : RegionName) (t : BlockState)
    (s_k_h s_k_t T K BT BK kb : Nat)
    (hmem : t.mem = s0.mem)
    (hpg : t.regs .blockPtr [BT, BK] "p_g" = some
      ⟨fun _ => BlockPtr.mk g (s0.pids 2 * s_k_h) [T, K] [BT, BK] [s_k_t, 1]
        [s0.pids 1 * BT, kb * BK]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BK] "p_g") [0, 1]) MaskOpt.none) t
      = some (cgfGRaw s0 g s_k_h s_k_t T K BT BK kb) := by
  rw [cgf_load_bp_2d g t "p_g" (s0.pids 2 * s_k_h) T K BT BK s_k_t 1
    (s0.pids 1 * BT) (kb * BK) hpg]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [cgfGRaw, gElem, tIndex, kIndex, BlockState.readMem, hmem]

/-- The `b_h` load lands directly on `cgfHTile`. -/
private theorem cgf_hLoad_eq (s0 : BlockState) (h : RegionName) (t : BlockState)
    (s_h_h s_h_t K V BK BV kb : Nat)
    (hmem : t.mem = s0.mem)
    (hph : t.regs .blockPtr [BK, BV] "p_h" = some
      ⟨fun _ => BlockPtr.mk h (s0.pids 2 * s_h_h + s0.pids 1 * K * V) [K, V] [BK, BV]
        [s_h_t, 1] [kb * BK, s0.pids 0 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1]) MaskOpt.none) t
      = some (cgfHTile s0 h s_h_h s_h_t K V BV kb BK) := by
  rw [cgf_load_bp_2d h t "p_h" (s0.pids 2 * s_h_h + s0.pids 1 * K * V) K V BK BV
    s_h_t 1 (kb * BK) (s0.pids 0 * BV) hph]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [cgfHTile, hGuarded, hElem, kIndex, vIndex, BlockState.readMem, hmem]
  split <;> rfl

/-- **The gated-query composition.** Statements 5 and 7 turn the raw `b_q` into
`cgfQgTile`: multiply by the scalar `scale`, then by `exp(b_g)` lane-wise. The two
raw tiles share one guard, and an out-of-bounds lane is `0` on both sides
(`0 · scale · exp 0 = 0`). -/
private theorem cgf_qgTile_eq (s0 : BlockState) (q g : RegionName)
    (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BK kb : Nat) :
    Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (cgfQRaw s0 q s_k_h s_k_t T K BT BK kb) (Tile.scalar (some scale)))
        (Tile.uop WithBot.realExp (cgfGRaw s0 g s_k_h s_k_t T K BT BK kb))
      = cgfQgTile s0 q g s_k_h s_k_t scale T K BT BK kb := by
  apply Tile.ext
  intro idx
  simp only [cgfQgTile, qgElem, cgfQRaw, cgfGRaw, Tile.bop_data, Tile.uop,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul,
    WithBot.realMul, WithBot.realExp]
  split
  · rfl
  · show Option.map₂ _ (Option.map₂ _ (some 0) _) _ = _
    norm_num

/-- **Statement 9.** The `if i_k >= 0` guard is true on the `nat` channel, so the
statement is the single accumulation, and it lands on the `bop`-of-`dot` value the
successor lemma consumes. -/
private theorem cgf_ifThen_acc_run (t : BlockState) (kb BT BK BV : Nat)
    (vqg : Tile .real [BT, BK]) (vh : Tile .real [BK, BV])
    (vacc : Tile .real [BT, BV])
    (hk : t.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hqg : t.regs .real [BT, BK] "b_qg" = some vqg)
    (hh : t.regs .real [BK, BV] "b_h" = some vh)
    (hacc : t.regs .real [BT, BV] "b_o" = some vacc) :
    stepStmt (Stmt.ifThen
        (Op.ge ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat 0))
        [ Stmt.assign .real [BT, BV] "b_o"
            (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.ref .real [BT, BV] "b_o")
              (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_qg")
                (Op.ref .real [BK, BV] "b_h"))) ]) t
      = some (t.setReg "b_o" .real [BT, BV]
          (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            vacc (Tile.dot [] vqg vh))) := by
  rw [cgf_ifThen_step, cgf_geGuard_eval t kb hk]
  show stepStmts _ t = _
  -- `erw`: `Op.dot`'s output shape is `[] ++ [BT, BV]`, invisible to `rw`
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_addTile_eval _ _ _ t vacc (Tile.dot [] vqg vh)
      (by rw [evalOp_ref]; exact hacc)
      (cgf_dot_eval _ _ t vqg vh (by rw [evalOp_ref]; exact hqg)
        (by rw [evalOp_ref]; exact hh))))]
  rw [stepStmts.nil]
  rfl


/-! ### The K step -/

theorem cgfLoopBody_run (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV i : Nat)
    (t : BlockState)
    (hnext : i + 1 ≤ numKB K BK)
    (hk : t.regs .nat [] "i_k" = some (Tile.scalar i))
    (hinv : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i t) :
    ∃ t', stepStmts (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)
          t = some t'
      ∧ cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV (i + 1) t' := by
  obtain ⟨-, hmem, hpids, hiv, hit, hibh, hms, hbo⟩ := hinv
  unfold cgfLoopBody
  -- 1-3. the three block pointers, rebuilt from `i_k`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval q t _ _ _ [T, K] [BT, BK] [s_k_t, 1]
      (s0.pids 2 * s_k_h) (s0.pids 1 * BT) (i * BK)
      (cgf_mulConst_eval t "i_bh" (s0.pids 2) s_k_h hibh)
      (cgf_mulConst_eval t "i_t" (s0.pids 1) BT hit)
      (cgf_mulConst_eval t "i_k" i BK hk)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval g _ _ _ _ [T, K] [BT, BK] [s_k_t, 1]
      (s0.pids 2 * s_k_h) (s0.pids 1 * BT) (i * BK)
      (cgf_mulConst_eval _ "i_bh" (s0.pids 2) s_k_h (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (cgf_mulConst_eval _ "i_k" i BK (by simpa using hk))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (s0.pids 2 * s_h_h + s0.pids 1 * K * V) (i * BK) (s0.pids 0 * BV)
      (cgf_hBase_eval _ s_h_h K V (s0.pids 2) (s0.pids 1)
        (by simpa using hibh) (by simpa using hit))
      (cgf_mulConst_eval _ "i_k" i BK (by simpa using hk))
      (cgf_mulConst_eval _ "i_v" (s0.pids 0) BV (by simpa using hiv))))]
  -- 4. `b_q = tl.load(p_q, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_qLoad_eq s0 q _ s_k_h s_k_t T K BT BK i
      (by simpa [cgf_setReg_mem] using hmem) (by simp)))]
  -- 5. `b_q = b_q * scale`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_mulTile_eval Broadcast.scalarR _ _ _
      (cgfQRaw s0 q s_k_h s_k_t T K BT BK i) (Tile.scalar (some scale))
      (by rw [evalOp_ref]; simp) (evalOp_const scale _)))]
  -- 6. `b_g = tl.load(p_g, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_gLoad_eq s0 g _ s_k_h s_k_t T K BT BK i
      (by simpa [cgf_setReg_mem] using hmem) (by simp)))]
  -- 7. `b_qg = b_q * tl.exp(b_g)` — lands on `cgfQgTile` via the composition
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BK] "b_q") (Op.exp (Op.ref .real [BT, BK] "b_g"))) _
      = some (cgfQgTile s0 q g s_k_h s_k_t scale T K BT BK i) from by
      rw [← cgf_qgTile_eq s0 q g s_k_h s_k_t scale T K BT BK i]
      exact cgf_mulTile_eval _ _ _ _
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (cgfQRaw s0 q s_k_h s_k_t T K BT BK i) (Tile.scalar (some scale)))
        (Tile.uop WithBot.realExp (cgfGRaw s0 g s_k_h s_k_t T K BT BK i))
        (by rw [evalOp_ref]; simp)
        (cgf_exp_eval _ _ (cgfGRaw s0 g s_k_h s_k_t T K BT BK i)
          (by rw [evalOp_ref]; simp))))]
  -- 8. `b_h = tl.load(p_h, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_hLoad_eq s0 h _ s_h_h s_h_t K V BK BV i
      (by simpa [cgf_setReg_mem] using hmem) (by simp)))]
  -- 9. the vacuously-true `if i_k >= 0`, then `b_o += tl.dot(b_qg, b_h)`
  rw [stepStmts.cons_some (cgf_ifThen_acc_run _ i BT BK BV
    (cgfQgTile s0 q g s_k_h s_k_t scale T K BT BK i)
    (cgfHTile s0 h s_h_h s_h_t K V BV i BK)
    (cgfAccTile s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i)
    (by simpa using hk) (by simp) (by simp) (by simpa using hbo))]
  rw [stepStmts.nil]
  rw [cgfAccTile_dot_succ s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [cgf_setReg_mem] using hmem
  · simpa using hpids
  · simpa using hiv
  · simpa using hit
  · simpa using hibh
  · simpa using hms
  · simp

/-! ### Collapsing the K loop -/

theorem cgfLoop_collapse (s0 : BlockState) (q g h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
    (t : BlockState)
    (h0 : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV 0 t) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "i_k" (Op.constNat 0)
          (Op.div .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
              (Op.constNat 1))
            (Op.constNat BK))
          (Op.constNat 1)
          (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)) t
        = some sF
      ∧ cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV
          (numKB K BK) sF := by
  have hstop : evalOp (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
      (Op.constNat BK)) t = some (Tile.scalar (numKB K BK)) := by
    rw [evalOp_div, evalOp_sub, evalOp_add]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  obtain ⟨final, sF, hrun, hfinal, hP⟩ :=
    forRangeDyn_inv (idx := "i_k") (start := 0) (stop := numKB K BK) (step := 1)
      (P := fun i u => cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale
        T K V BT BK BV i u)
      (evalOp_constNat _ _) hstop (evalOp_constNat _ _) one_ne_zero h0
      (fun i u hi hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          cgfLoopBody_run s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV i _
            (by omega) (by simp)
            (cgfInv_setReg_k s0 q g h s_k_h s_k_t s_h_h s_h_t scale
              T K V BT BK BV i i u hinv)
        exact ⟨s', hs', hinv'⟩)
  have hEq : final = numKB K BK := le_antisymm hP.1 hfinal
  subst hEq
  exact ⟨sF, hrun, hP⟩

/-! ## The tail

Three block pointers, the `v` and `A` loads, the causal masking of `A`, the second
`tl.dot`, and the boundary-checked store. -/

/-- `b_v` as its load leaves it. -/
noncomputable def cgfVTile (s : BlockState) (v : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (vGuarded s v s_v_h s_v_t T V BT BV idx.1.val idx.2.1.val)⟩

/-- `b_A` after the causal `tl.where`. -/
noncomputable def cgfATile (s : BlockState) (A : RegionName) (T BT : Nat) :
    Tile .real [BT, BT] :=
  ⟨fun idx => some (aMasked s A T BT idx.1.val idx.2.1.val)⟩

/-- The `b_v` load. -/
private theorem cgf_vLoad_eq (s0 : BlockState) (v : RegionName) (t : BlockState)
    (s_v_h s_v_t T V BT BV : Nat)
    (hmem : t.mem = s0.mem)
    (hpv : t.regs .blockPtr [BT, BV] "p_v" = some
      ⟨fun _ => BlockPtr.mk v (s0.pids 2 * s_v_h) [T, V] [BT, BV] [s_v_t, 1]
        [s0.pids 1 * BT, s0.pids 0 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1]) MaskOpt.none) t
      = some (cgfVTile s0 v s_v_h s_v_t T V BT BV) := by
  rw [cgf_load_bp_2d v t "p_v" (s0.pids 2 * s_v_h) T V BT BV s_v_t 1
    (s0.pids 1 * BT) (s0.pids 0 * BV) hpv]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp only [cgfVTile, vGuarded, vElem, tIndex, vIndex, BlockState.readMem, hmem]
  split <;> rfl

/-- The raw `b_A` load. The column check `0 + j < BT` is vacuous (`j : Fin BT`),
which is why the raw tile below carries only the row guard. -/
private theorem cgf_aLoad_eq (s0 : BlockState) (A : RegionName) (t : BlockState)
    (T BT : Nat)
    (hmem : t.mem = s0.mem)
    (hpA : t.regs .blockPtr [BT, BT] "p_A" = some
      ⟨fun _ => BlockPtr.mk A (s0.pids 2 * T * BT) [T, BT] [BT, BT] [BT, 1]
        [s0.pids 1 * BT, 0]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BT] "p_A") [0, 1]) MaskOpt.none) t
      = some ⟨fun idx : TileIndex [BT, BT] =>
          if tIndex s0 BT idx.1.val < T then
            some (aElem s0 A T BT idx.1.val idx.2.1.val)
          else some 0⟩ := by
  rw [cgf_load_bp_2d A t "p_A" (s0.pids 2 * T * BT) T BT BT BT BT 1
    (s0.pids 1 * BT) 0 hpA]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨i, j, u⟩ := idx
  simp only [aElem, tIndex, BlockState.readMem, hmem, Nat.zero_add, j.isLt, and_true]

/-- The causal `tl.where` lands on `cgfATile`. -/
private theorem cgf_aWhere_eq (s0 : BlockState) (A : RegionName) (T BT : Nat) :
    Tile.select (cgfMsTile BT)
        (⟨fun idx : TileIndex [BT, BT] =>
          if tIndex s0 BT idx.1.val < T then
            some (aElem s0 A T BT idx.1.val idx.2.1.val)
          else some 0⟩ : Tile .real [BT, BT])
        (⟨fun _ => some 0.0⟩ : Tile .real [BT, BT])
      = cgfATile s0 A T BT := by
  apply Tile.ext
  intro idx
  simp only [cgfATile, aMasked, cgfMsTile, Tile.select_data, decide_eq_true_eq]
  by_cases hji : idx.2.1.val ≤ idx.1.val
  · simp only [hji, if_pos]
    split <;> rfl
  · simp only [hji, if_false]
    norm_num

/-- The final output tile: the loop's full accumulator plus the intra-chunk dot. -/
noncomputable def cgfOutTile (s : BlockState) (q v g h A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale T K V BT BK BV idx.1.val idx.2.1.val)⟩

/-- **The intra-chunk accumulation.** `b_o += tl.dot(b_A, b_v)` on top of the full
inter-chunk sum is exactly `cgfOutTile`. -/
private theorem cgf_outTile_eq (s0 : BlockState) (q v g h A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) :
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (cgfAccTile s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV
          (numKB K BK))
        (Tile.dot [] (cgfATile s0 A T BT) (cgfVTile s0 v s_v_h s_v_t T V BT BV))
      = cgfOutTile s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
          T K V BT BK BV := by
  apply Tile.ext
  intro idx
  obtain ⟨r, p, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, cgfAccTile,
    cgfOutTile, cgfOutput, NumericDType.add, WithBot.realAdd]
  rw [cgf_dot2d_elem _ _ r p
    (fun j => aMasked s0 A T BT r.val j.val)
    (fun j => vGuarded s0 v s_v_h s_v_t T V BT BV j.val p.val)
    (fun j => rfl) (fun j => rfl)]
  rfl

/-! ## The store and its readback -/

/-- Post-store state of the boundary-checked output store. -/
noncomputable def cgfStoreState (o : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s0 : BlockState) (f : TileIndex [BT, BV] → ℝ) (t : BlockState) : BlockState :=
  (TileShape.allIndices [BT, BV]).foldl
    (fun acc i =>
      if tIndex s0 BT i.1.val < T ∧ vIndex s0 BV i.2.1.val < V then
        acc.writeMem o (outOffset s0 s_v_h s_v_t BT BV i) (f i)
      else acc) t

private theorem cgf_store_eq (o : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s0 t : BlockState) (vt : Tile .real [BT, BV]) (f : TileIndex [BT, BV] → ℝ)
    (hfv : ∀ i, vt.data i = some (f i))
    (hpo : t.regs .blockPtr [BT, BV] "p_o" = some
      ⟨fun _ => BlockPtr.mk o (s0.pids 2 * s_v_h) [T, V] [BT, BV] [s_v_t, 1]
        [s0.pids 1 * BT, s0.pids 0 * BV]⟩)
    (hv : t.regs .real [BT, BV] "b_o" = some vt) :
    stepStmt (Stmt.store .real [BT, BV]
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_o") [0, 1])
        (Op.ref .real [BT, BV] "b_o") MaskOpt.none) t
      = some (cgfStoreState o s_v_h s_v_t T V BT BV s0 f t) := by
  unfold stepStmt cgfStoreState
  simp only [evalOp_ref, hv, hpo]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BT, BV])) ?_)
  funext acc i
  obtain ⟨r, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, outOffset, tIndex, vIndex]
  by_cases hb : s0.pids 1 * BT + r.val < T ∧ s0.pids 0 * BV + p.val < V
  · simp only [hb, and_self, decide_true, if_true, BlockState.writeMemTyped_real, hfv]
    rfl
  · simp only [hb, decide_false, Bool.false_eq_true, if_false]

/-- **Tail statement 6.** `b_A = tl.where(m_s, b_A, 0.)` on a state whose two
registers are known lands on `cgfATile`. -/
private theorem cgf_aWhere_run (s0 : BlockState) (A : RegionName) (T BT : Nat)
    (t : BlockState)
    (hms : t.regs .bool [BT, BT] "m_s" = some (cgfMsTile BT))
    (hba : t.regs .real [BT, BT] "b_A" = some
      ⟨fun idx : TileIndex [BT, BT] =>
        if tIndex s0 BT idx.1.val < T then
          some (aElem s0 A T BT idx.1.val idx.2.1.val)
        else some 0⟩) :
    stepStmt (Stmt.assign .real [BT, BT] "b_A"
        (Op.where (Op.ref .bool [BT, BT] "m_s") (Op.ref .real [BT, BT] "b_A")
          (Op.broadcast (Op.const 0.0) [BT, BT]))) t
      = some (t.setReg "b_A" .real [BT, BT] (cgfATile s0 A T BT)) := by
  refine stepStmt_assign_eq_some ?_
  rw [← cgf_aWhere_eq s0 A T BT]
  exact cgf_where_eval _ _ _ _ (cgfMsTile BT)
    ((⟨fun idx : TileIndex [BT, BT] =>
      if tIndex s0 BT idx.1.val < T then
        some (aElem s0 A T BT idx.1.val idx.2.1.val)
      else some 0⟩ : Tile .real [BT, BT]))
    ((⟨fun _ => some 0.0⟩ : Tile .real [BT, BT]))
    (by rw [evalOp_ref]; exact hms)
    (by rw [evalOp_ref]; exact hba)
    (show evalOp ((Op.const 0.0).broadcast [BT, BT]) t
        = some (⟨fun _ => some 0.0⟩ : Tile .real [BT, BT]) from
      cgf_broadcast_eval (Op.const 0.0) [BT, BT] t (Tile.scalar (some 0.0))
        (evalOp_const 0.0 t))

/-- **Tail statement 7.** `b_o += tl.dot(b_A, b_v)` with all three registers known
lands on `cgfOutTile`. -/
private theorem cgf_finalAcc_run (s0 : BlockState) (q v g h A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t : BlockState)
    (hbo : t.regs .real [BT, BV] "b_o" = some
      (cgfAccTile s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV (numKB K BK)))
    (hba : t.regs .real [BT, BT] "b_A" = some (cgfATile s0 A T BT))
    (hbv : t.regs .real [BT, BV] "b_v" = some (cgfVTile s0 v s_v_h s_v_t T V BT BV)) :
    stepStmt (Stmt.assign .real [BT, BV] "b_o"
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BT, BV] "b_o")
          (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_A")
            (Op.ref .real [BT, BV] "b_v")))) t
      = some (t.setReg "b_o" .real [BT, BV]
          (cgfOutTile s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
            T K V BT BK BV)) := by
  refine stepStmt_assign_eq_some ?_
  rw [← cgf_outTile_eq s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
    T K V BT BK BV]
  exact cgf_addTile_eval _ _ _ _
    (cgfAccTile s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV (numKB K BK))
    (Tile.dot [] (cgfATile s0 A T BT) (cgfVTile s0 v s_v_h s_v_t T V BT BV))
    (by rw [evalOp_ref]; exact hbo)
    (cgf_dot_eval _ _ _ (cgfATile s0 A T BT) (cgfVTile s0 v s_v_h s_v_t T V BT BV)
      (by rw [evalOp_ref]; exact hba) (by rw [evalOp_ref]; exact hbv))

/-! ## The tail walk and the main theorem -/

theorem cgfPostLoop_run (s0 : BlockState) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s0 s_v_h s_v_t BT BV idx))
    (hinv : cgfInv s0 q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV
      (numKB K BK) t) :
    ∃ sF, stepStmts (cgfPostLoop v o A s_v_h s_v_t T V BT BV) t = some sF
      ∧ ∀ idx : TileIndex [BT, BV],
          (tIndex s0 BT idx.1.val < T ∧ vIndex s0 BV idx.2.1.val < V) →
          sF.readMem o (outOffset s0 s_v_h s_v_t BT BV idx)
            = cgfOutput s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV idx.1.val idx.2.1.val := by
  obtain ⟨-, hmem, hpids, hiv, hit, hibh, hms, hbo⟩ := hinv
  unfold cgfPostLoop
  -- 1-3. the three block pointers
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval v t _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval t "i_bh" (s0.pids 2) s_v_h hibh)
      (cgf_mulConst_eval t "i_t" (s0.pids 1) BT hit)
      (cgf_mulConst_eval t "i_v" (s0.pids 0) BV hiv)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BT, BV] [s_v_t, 1]
      (s0.pids 2 * s_v_h) (s0.pids 1 * BT) (s0.pids 0 * BV)
      (cgf_mulConst_eval _ "i_bh" (s0.pids 2) s_v_h (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (cgf_mulConst_eval _ "i_v" (s0.pids 0) BV (by simpa using hiv))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_makeBlockPtr_2d_eval A _ _ _ _ [T, BT] [BT, BT] [BT, 1]
      (s0.pids 2 * T * BT) (s0.pids 1 * BT) 0
      (cgf_aBase_eval _ T BT (s0.pids 2) (by simpa using hibh))
      (cgf_mulConst_eval _ "i_t" (s0.pids 1) BT (by simpa using hit))
      (evalOp_constNat 0 _)))]
  -- 4. `b_v = tl.load(p_v, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_vLoad_eq s0 v _ s_v_h s_v_t T V BT BV
      (by simpa [cgf_setReg_mem] using hmem) (by simp)))]
  -- 5. `b_A = tl.load(p_A, boundary_check=(0, 1))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cgf_aLoad_eq s0 A _ T BT (by simpa [cgf_setReg_mem] using hmem) (by simp)))]
  -- 6. `b_A = tl.where(m_s, b_A, 0.)`
  rw [stepStmts.cons_some (cgf_aWhere_run s0 A T BT _ (by simpa using hms) (by simp))]
  -- 7. `b_o += tl.dot(b_A, b_v)`
  rw [stepStmts.cons_some (cgf_finalAcc_run s0 q v g h A s_k_h s_k_t s_v_h s_v_t
    s_h_h s_h_t scale T K V BT BK BV _ (by simpa using hbo) (by simp) (by simp))]
  -- 8. the store
  rw [stepStmts.cons_some
    (cgf_store_eq o s_v_h s_v_t T V BT BV s0 _
      (cgfOutTile s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV)
      (fun idx => cgfOutput s0 q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        T K V BT BK BV idx.1.val idx.2.1.val)
      (fun idx => rfl) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hactive
  unfold cgfStoreState
  exact BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ idx
    (by simpa [tIndex, vIndex] using hactive)
    (fun kk _ heq => hInj heq)

/-- The prologue: three program ids, the causal mask, and the zeroed accumulator,
as named facts on an opaque state (a `by simp` against a metavariable state cannot
be postponed, so the headline consumes this instead of inline rewrites). -/
theorem cgfPreLoop_run (s : BlockState) (BT BV : Nat) :
    ∃ t, stepStmts (cgfPreLoop BT BV) s = some t
      ∧ t.mem = s.mem
      ∧ t.pids = s.pids
      ∧ t.regs .nat [] "i_v" = some (Tile.scalar (s.pids 0))
      ∧ t.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1))
      ∧ t.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ t.regs .bool [BT, BT] "m_s" = some (cgfMsTile BT)
      ∧ t.regs .real [BT, BV] "b_o" = some (⟨fun _ => some 0⟩ : Tile .real [BT, BV]) := by
  unfold cgfPreLoop
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (cgf_ms_eval _ BT))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BT, BV] (Op.const 0)) _
        = some (⟨fun _ => some 0⟩ : Tile .real [BT, BV]) from by
      rw [evalOp_full, evalOp_const]
      rfl))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [cgf_setReg_mem]

/-! ## Main theorem -/

set_option maxHeartbeats 1000000 in
/-- **Genuine, dimension-general correctness** of `chunk_gla_fwd_kernel_o`. For
every launch state, the kernel runs to completion and every in-window output lane
holds `cgfOutput`: the inter-chunk sum over **all** `cdiv(K, BK)` K blocks of the
scaled, gated query against the chunk state, plus the causally-masked intra-chunk
`A · V` term.

`hInj` says distinct output lanes get distinct `o` addresses — the standard
row-major side condition, exactly as in the ported sibling `chunk_gla_simple`.
Unlike that sibling, there is **no** `K = BK` hypothesis: the K loop is verified in
full multi-block generality. -/
specification chunk_gla_fwd_o_exec_genuine
    (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ∃ sF, exec (chunk_gla_fwd_o_surface q v g h o A s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BT, BV],
          (tIndex s BT idx.1.val < T ∧ vIndex s BV idx.2.1.val < V) →
          sF.readMem o (outOffset s s_v_h s_v_t BT BV idx)
            = cgfOutput s q v g h A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
                T K V BT BK BV idx.1.val idx.2.1.val := by
  rw [exec, cgf_body_eq, List.append_assoc]
  -- prologue
  obtain ⟨t0, hpre, h0mem, h0pids, h0iv, h0it, h0ibh, h0ms, h0bo⟩ :=
    cgfPreLoop_run s BT BV
  rw [stepStmts.append_some hpre]
  rw [show [Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
            (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)]
      ++ cgfPostLoop v o A s_v_h s_v_t T V BT BV
    = Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK))
            (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (cgfLoopBody g h q s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV)
      :: cgfPostLoop v o A s_v_h s_v_t T V BT BV from rfl]
  -- the collapsed K loop
  obtain ⟨t1, hloop, hinv1⟩ :=
    cgfLoop_collapse s q g h s_k_h s_k_t s_h_h s_h_t scale T K V BT BK BV t0
      ⟨Nat.zero_le _, h0mem, h0pids, h0iv, h0it, h0ibh, h0ms,
        by rw [h0bo, cgfAccTile_zero]⟩
  rw [stepStmts.cons_some hloop]
  -- the tail
  obtain ⟨sF, hpost, hout⟩ :=
    cgfPostLoop_run s q v g h o A s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      T K V BT BK BV t1 hInj hinv1
  exact ⟨sF, hpost, hout⟩

end Correct_without_Rounding


end VeriTile.Bench.TritonBenchG.ChunkGlaFwd

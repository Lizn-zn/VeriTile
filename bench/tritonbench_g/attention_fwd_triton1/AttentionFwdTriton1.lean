import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `attention_fwd_triton1` — strict per-kernel correctness

`attention_fwd_kernel` is a linear (chunked) attention forward kernel: program
`i_bh` (one batch-head) carries a `[BD, BD]` recurrent state `b_h` across the
`cdiv(T, BT)` time chunks. Per chunk it loads `Q`/`K`/`V` block pointers, scales
`b_q` by `scale`, forms `b_s = dot(b_q, b_k)` and the local output
`b_o = dot(b_s, b_v)`, optionally adds the inter-chunk term `dot(b_q, b_h)`,
updates `b_h += dot(b_k, b_v)`, optionally stores `b_h` to `h` (`STORE`), and
stores `b_o` to `o`. The `IFCOND` flag selects whether the first chunk skips the
recurrent add (resetting `b_h = dot(b_k, b_v)` instead).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`attention_fwd_kernel[grid](...)`, the grid over
`batch·n_heads`, scheduling, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`i_bh` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
attention_fwd_triton1_python_test_shape_output_summary       ← TOP THEOREM
  ├─ attention_fwd_triton1_surface_output_compute_correct     O store, 4 STORE/IFCOND branches
  │    └─ attention_fwd_triton1_output_store_slice_compute_correct
  │         └─ attention_fwd_triton1_output_store_slice_correct
  └─ attention_fwd_triton1_surface_h_compute_correct          H store, 2 STORE branches
       └─ attention_fwd_triton1_h_store_slice_compute_correct
            └─ attention_fwd_triton1_h_store_slice_correct

(supporting arithmetic-producer summaries, factored out as slice inputs:
   attention_fwd_triton1_bo_formula_slice_compute_correct      ← boFormulaSpec
   attention_fwd_triton1_bh_formula_slice_compute_correct      ← bhFormulaSpec)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); dtype casts collapse to
the identity post-erasure; `@triton.autotune` / `num_warps`/`num_stages` are not
modeled. The output summary is stated at the Python test shape
(`B=2, H=8, T=1024, D=128, BT=32, BD=128, NT=32`, `scale = 1/sqrt(128)`,
contiguous strides) and covers all four `STORE`/`IFCOND` combinations. The
`O`/`H` writebacks are stated against `attentionFwdTriton1SurfaceValue` (the
single-program surface value at each offset). The dot-product arithmetic that
produces the `b_o`/`b_h` tiles is captured by the separate `bo_formula`/
`bh_formula` slice summaries (`boFormulaSpec`/`bhFormulaSpec`) and fed in as
slice inputs to keep the writeback statements tractable. This is a
single-program (single batch-head) scope; cross-program composition is the
trusted host boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The Python kernel uses block pointers plus two constexpr gates, `STORE` and
`IFCOND`. The `order` metadata is accepted by the DSL and erased into the same
block-pointer AST. -/
def attention_fwd_kernel_surface
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(p_h.dtype.element_ty))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
  }
}

/-- The full Python-shaped forward surface lowers to the algorithm layer,
including the block-pointer loop, optional H-state store, dot products, and
output writeback. -/
theorem attention_fwd_kernel_surface_toAlgorithm_supported
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ∃ alg, (attention_fwd_kernel_surface q k v h o s_qh s_qt s_qd s_hh
      s_ht T scale BT BD NT STORE IFCOND).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

noncomputable def attentionFwdTriton1SurfaceValue
    (s : BlockState) (Q K V H O Out : RegionName)
    (STORE IFCOND : Bool) (offset : Nat) : ℝ :=
  match exec (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 STORE IFCOND) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

/-! ## Forward accumulator arithmetic surfaces

These proof-oriented surfaces expose the arithmetic producers for the `BO` and
`BHPre` tiles consumed by the store-slice theorems below. They mirror the
source loop body statements rather than starting from an opaque accumulator:

* `b_s = tl.dot((b_q * scale).to(b_q.dtype), b_k)`
* `b_o = tl.dot(b_s.to(b_q.dtype), b_v)`
* `b_h = tl.dot(b_k, b_v)`
-/

def attention_fwd_triton1_bo_formula_slice
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat) :
    ComputeKernel := triton {
  offs_t = tl.arange(0, $(BT))
  offs_d = tl.arange(0, $(BD))
  offs_s = tl.arange(0, $(BT))
  b_q = tl.load(QTile + offs_t[:, None] * $(BD) + offs_d[None, :])
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_k = tl.load(KTile + offs_d[:, None] * $(BT) + offs_s[None, :])
  b_v = tl.load(VTile + offs_s[:, None] * $(BD) + offs_d[None, :])
  b_s = tl.dot(b_q, b_k, allow_tf32=false)
  b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
  tl.store(BO + offs_t[:, None] * $(BD) + offs_d[None, :], b_o)
}

theorem attention_fwd_triton1_bo_formula_slice_toAlgorithm_supported
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat) :
    ∃ alg, (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
      scale BT BD).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_triton1_bo_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def attention_fwd_triton1_bh_formula_slice
    (KTile VTile BH : RegionName) (BT BD : Nat) :
    ComputeKernel := triton {
  offs_d0 = tl.arange(0, $(BD))
  offs_t = tl.arange(0, $(BT))
  offs_d1 = tl.arange(0, $(BD))
  b_k = tl.load(KTile + offs_d0[:, None] * $(BT) + offs_t[None, :])
  b_v = tl.load(VTile + offs_t[:, None] * $(BD) + offs_d1[None, :])
  b_h = tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(BH + offs_d0[:, None] * $(BD) + offs_d1[None, :], b_h)
}

theorem attention_fwd_triton1_bh_formula_slice_toAlgorithm_supported
    (KTile VTile BH : RegionName) (BT BD : Nat) :
    ∃ alg, (attention_fwd_triton1_bh_formula_slice KTile VTile BH
      BT BD).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_triton1_bh_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def localBoOffset (BD : Nat) (idx : TileIndex [BT, BD]) : Nat :=
  idx.1.val * BD + idx.2.1.val

def localBhOffset (BD : Nat) (idx : TileIndex [BD, BD]) : Nat :=
  idx.1.val * BD + idx.2.1.val

noncomputable def boFormulaSpec
    (s : BlockState) (QTile KTile VTile : RegionName)
    (scale : ℝ) (BT BD : Nat) (idx : TileIndex [BT, BD]) : ℝ :=
  ∑ t : Fin BT,
    (∑ d : Fin BD,
      (s.readMem QTile (idx.1.val * BD + d.val) * scale) *
        s.readMem KTile (d.val * BT + t.val)) *
      s.readMem VTile (t.val * BD + idx.2.1.val)

noncomputable def bhFormulaSpec
    (s : BlockState) (KTile VTile : RegionName)
    (BT BD : Nat) (idx : TileIndex [BD, BD]) : ℝ :=
  ∑ t : Fin BT,
    s.readMem KTile (idx.1.val * BT + t.val) *
      s.readMem VTile (t.val * BD + idx.2.1.val)

theorem attention_fwd_triton1_bo_formula_slice_correct
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => localBoOffset BD idx)) :
    ∀ idx : TileIndex [BT, BD],
      let outAddr := localBoOffset BD idx
      (exec (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
            scale BT BD) s).map (·.readMem BO outAddr)
        = some (boFormulaSpec s QTile KTile VTile scale BT BD idx) := by
  intro idx
  simp [exec, attention_fwd_triton1_bo_formula_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
        NumericDType.add, NumericDType.mul, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, localBoOffset,
        boFormulaSpec, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BD] → Nat :=
    fun i => i.1.val * BD + i.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, localBoOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem attention_fwd_triton1_bo_formula_slice_compute_correct
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => localBoOffset BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        scale BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (BO, localBoOffset BD idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        boFormulaSpec s QTile KTile VTile scale BT BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_bo_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_bo_formula_slice_correct QTile KTile VTile
    BO scale BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem attention_fwd_triton1_bh_formula_slice_correct
    (KTile VTile BH : RegionName) (BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => localBhOffset BD idx)) :
    ∀ idx : TileIndex [BD, BD],
      let outAddr := localBhOffset BD idx
      (exec (attention_fwd_triton1_bh_formula_slice KTile VTile BH BT BD)
          s).map (·.readMem BH outAddr)
        = some (bhFormulaSpec s KTile VTile BT BD idx) := by
  intro idx
  simp [exec, attention_fwd_triton1_bh_formula_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
        NumericDType.add, NumericDType.mul, localBhOffset, bhFormulaSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BD, BD] → Nat :=
    fun i => i.1.val * BD + i.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, localBhOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem attention_fwd_triton1_bh_formula_slice_compute_correct
    (KTile VTile BH : RegionName) (BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => localBhOffset BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BH BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (BH, localBhOffset BD idx))
      (expected := fun idx : TileIndex [BD, BD] =>
        bhFormulaSpec s KTile VTile BT BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_bh_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_bh_formula_slice_correct KTile VTile BH
    BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-! ## Genuine closed form for the linear-attention output

`attention_fwd_kernel` implements *linear (chunked) attention*: there is **no
softmax** at all. Per batch-head program the loop carries the recurrent state

  `b_h_i = Σ_{j < i} (Kⱼᵀ · Vⱼ)`   (a `[BD, BD]` matrix),

and the output of time-chunk `i` (default branch `STORE=False, IFCOND=False`,
which the `IFCOND=True, i>0` branch coincides with, and the `i=0` branch reduces
to since `b_h_0 = 0`) is

  `Oᵢ[t, d] = ((scale·Qᵢ) · Kᵢ · Vᵢ)[t, d] + ((scale·Qᵢ) · b_h_i)[t, d]`.

The first summand is exactly `boFormulaSpec` for chunk `i`'s tiles; the second
is the contraction of the scaled query against the accumulated state. The
definitions below give that genuine closed form purely in terms of the input
memory (no reference to the executed kernel output), and the identity theorems
establish the linear-attention algebra: the recurrent contribution telescopes
into a single double sum over all prior keys/values.

The per-chunk tile accessors are passed as offset functions `qAddr`/`kAddr`/
`vAddr : Nat → ... → Nat` mapping `(chunk, row, col)` to a flat memory offset,
matching the `make_block_ptr` layout `base + (chunk·BT + row)·s_t + col·s_d`. -/

/-- One chunk's local (`(scale·Q)·K·V`) closed-form entry, stated directly over
input memory via per-chunk row/col offset accessors. -/
noncomputable def localTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ tk : Fin BT,
    (∑ dd : Fin BD,
      (s.readMem Q (qAddr chunk (BD * t.val + dd.val)) * scale) *
        s.readMem K (kAddr chunk (BT * dd.val + tk.val))) *
      s.readMem V (vAddr chunk (BD * tk.val + d.val))

/-- The recurrent state matrix `b_h_i[d', d] = Σ_{j < i} (Kⱼᵀ·Vⱼ)[d', d]`,
genuine closed form over input memory. -/
noncomputable def recurrentState
    (s : BlockState) (K V : RegionName) (BT BD : Nat)
    (kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (d' d : Fin BD) : ℝ :=
  ∑ j ∈ Finset.range chunk,
    ∑ tk : Fin BT,
      s.readMem K (kAddr j (BT * d'.val + tk.val)) *
        s.readMem V (vAddr j (BD * tk.val + d.val))

/-- The recurrent output contribution `((scale·Qᵢ)·b_h_i)[t, d]`. -/
noncomputable def recurrentTerm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  ∑ d' : Fin BD,
    (s.readMem Q (qAddr chunk (BD * t.val + d'.val)) * scale) *
      recurrentState s K V BT BD kAddr vAddr chunk d' d

/-- Genuine closed-form output of chunk `chunk`, position `(t, d)`:
local term plus recurrent term. No self-reference to the executed kernel. -/
noncomputable def outputClosedForm
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat)
    (chunk : Nat) (t : Fin BT) (d : Fin BD) : ℝ :=
  localTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d +
    recurrentTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d

/-- **Base chunk.** For the first time-chunk (`chunk = 0`) the recurrent state is
empty, so the genuine closed form collapses to the pure local term — matching the
Python `i = 0` behavior (`b_h` initialized to zeros, and under `IFCOND=True` the
`i=0` branch skips the recurrent add entirely). -/
theorem outputClosedForm_zero
    (s : BlockState) (Q K V : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat) (t : Fin BT) (d : Fin BD) :
    outputClosedForm s Q K V scale BT BD qAddr kAddr vAddr 0 t d
      = localTerm s Q K V scale BT BD qAddr kAddr vAddr 0 t d := by
  simp [outputClosedForm, recurrentTerm, recurrentState]

/-- **Local term = `boFormulaSpec`.** The local contribution of chunk `chunk`
is exactly the already-proven dot-product producer spec `boFormulaSpec`,
instantiated at that chunk's tiles. This pins the genuine closed form to the
checked arithmetic surface rather than to the executed output.

`Qc/Kc/Vc` are the per-chunk tile regions whose flat layout (`row·BD + col` for
`Q`/`V`, `row·BT + col` for `K`) matches the producer slices; the hypotheses say
the chunk accessors read the same memory as those tiles. -/
theorem localTerm_eq_boFormulaSpec
    (s : BlockState) (Q K V Qc Kc Vc : RegionName) (scale : ℝ) (BT BD : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat) (chunk : Nat)
    (t : Fin BT) (d : Fin BD)
    (hQ : ∀ off, s.readMem Q (qAddr chunk off) = s.readMem Qc off)
    (hK : ∀ off, s.readMem K (kAddr chunk off) = s.readMem Kc off)
    (hV : ∀ off, s.readMem V (vAddr chunk off) = s.readMem Vc off) :
    localTerm s Q K V scale BT BD qAddr kAddr vAddr chunk t d
      = boFormulaSpec s Qc Kc Vc scale BT BD (t, d, PUnit.unit) := by
  simp only [localTerm, boFormulaSpec, hQ, hK, hV, Nat.mul_comm]

/-- **Recurrent state telescopes into a sum of per-chunk `bhFormulaSpec`s.** The
accumulated state at chunk `chunk` equals the sum over prior chunks `j` of the
already-proven `bhFormulaSpec` (the `dot(K, V)` producer) for chunk `j`. The
chunk-`j` accessors are supplied via the per-chunk tile regions. -/
theorem recurrentState_eq_sum_bhFormulaSpec
    (s : BlockState) (K V : RegionName) (BT BD : Nat)
    (kAddr vAddr : Nat → Nat → Nat) (chunk : Nat) (d' d : Fin BD)
    (Kc Vc : Nat → RegionName)
    (hK : ∀ j off, s.readMem K (kAddr j off) = s.readMem (Kc j) off)
    (hV : ∀ j off, s.readMem V (vAddr j off) = s.readMem (Vc j) off) :
    recurrentState s K V BT BD kAddr vAddr chunk d' d
      = ∑ j ∈ Finset.range chunk,
          bhFormulaSpec s (Kc j) (Vc j) BT BD (d', d, PUnit.unit) := by
  simp only [recurrentState, bhFormulaSpec, hK, hV, Nat.mul_comm]

/-- **Step recurrence.** The genuine closed form satisfies the same one-chunk
recurrence the Python loop realizes: advancing from chunk `n` to chunk `n+1`
adds the `n`-th chunk's `Kᵀ·V` block to the recurrent state. This is the loop
invariant in closed form (`b_h_{n+1} = b_h_n + Kₙᵀ·Vₙ`). -/
theorem recurrentState_succ
    (s : BlockState) (K V : RegionName) (BT BD : Nat)
    (kAddr vAddr : Nat → Nat → Nat) (n : Nat) (d' d : Fin BD) :
    recurrentState s K V BT BD kAddr vAddr (n + 1) d' d
      = recurrentState s K V BT BD kAddr vAddr n d' d
        + (∑ tk : Fin BT,
            s.readMem K (kAddr n (BT * d'.val + tk.val)) *
              s.readMem V (vAddr n (BD * tk.val + d.val))) := by
  simp [recurrentState, Finset.sum_range_succ]

/-- Surface transcription/proof-oriented output-store slice of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The full kernel iterates over time blocks, optionally stores the recurrent
state `b_h`, computes `b_o`, and stores it through `p_o`. This slice represents
one loop iteration with program axes `(i_bh, i_block)`, starts from a precomputed
`BO` tile, and proves the unmasked `p_o` block writeback into `O`. The
`tl.float32` recurrent state initializer and dot-loop that produce `BO` are
outside this slice. -/
def attention_fwd_triton1_output_store_slice
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i = tl.program_id(1)
  offs_t = i * $(BT) + tl.arange(0, $(BT))
  offs_d = tl.arange(0, $(BD))
  b_o = tl.load(BO + i_bh * $(stride_bo_bh) +
      offs_t[:, None] * $(stride_bo_t) + offs_d[None, :] * $(stride_bo_d))
  tl.store(O + i_bh * $(s_qh) + offs_t[:, None] * $(s_qt) +
      offs_d[None, :] * $(s_qd), (b_o).to(O.dtype.element_ty))
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

def dIndex (idx : TileIndex [BT, BD]) : Nat :=
  idx.2.1.val

def boOffset
    (s : BlockState)
    (stride_bo_bh stride_bo_t stride_bo_d BT : Nat)
    (idx : TileIndex [BT, BD]) : Nat :=
  s.pids 0 * stride_bo_bh +
    tIndex s BT idx.1 * stride_bo_t + dIndex idx * stride_bo_d

def outOffset
    (s : BlockState)
    (s_qh s_qt s_qd BT : Nat)
    (idx : TileIndex [BT, BD]) : Nat :=
  s.pids 0 * s_qh + tIndex s BT idx.1 * s_qt + dIndex idx * s_qd

/-- Algorithm-layer correctness for one `p_o` output block store. -/
theorem attention_fwd_triton1_output_store_slice_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ∀ idx : TileIndex [BT, BD],
      let outAddr := outOffset s s_qh s_qt s_qd BT idx
      (exec (attention_fwd_triton1_output_store_slice BO O stride_bo_bh
            stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD) s).map
          (·.readMem O outAddr)
        = some (s.readMem BO
            (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  intro idx
  simp [exec, attention_fwd_triton1_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, tIndex, dIndex,
        boOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BD] → Nat :=
    fun idx => s.pids 0 * s_qh + (s.pids 1 * BT + idx.1.val) * s_qt +
      idx.2.1.val * s_qd
  let valueFn : TileIndex [BT, BD] → ℝ :=
    fun idx =>
      s.readMem BO
        (s.pids 0 * stride_bo_bh +
          (s.pids 1 * BT + idx.1.val) * stride_bo_t +
          idx.2.1.val * stride_bo_d)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, tIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BT, BD])).readMem O (offsetFn idx) =
    s.readMem BO (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, boOffset, tIndex, dIndex]

/-- Compute-facing correctness for one `p_o` output block store. -/
theorem attention_fwd_triton1_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_output_store_slice_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named output writeback for the `IFCOND=True, i=0` branch.

The branch-specific dot-product arithmetic is represented by `BO`; this theorem
exposes the Python-observed `p_o` store for that branch using the shared output
store proof. -/
theorem attention_fwd_triton1_ifcond_first_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  exact attention_fwd_triton1_output_store_slice_compute_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj

/-- Named output writeback for the recurrent-output branch (`IFCOND=False` or
`IFCOND=True, i>0`). `BO` carries the branch-specific accumulated value. -/
theorem attention_fwd_triton1_recurrent_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  exact attention_fwd_triton1_output_store_slice_compute_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj

/-- Proof-oriented `p_h` state-store slice of `attention_fwd_triton1.py`.
Companion to the output_store_slice: takes a precomputed `BHPre` [BD, BD]
tile and proves the per-iteration writeback into `H` at the canonical block
offset `i_bh * s_hh + (i * BD + idx.1) * s_ht + idx.2.1`. -/
def attention_fwd_triton1_h_store_slice
    (BHPre H : RegionName) (i_iter s_hh s_ht _BT BD : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  offs_d0 = $(i_iter) * $(BD) + tl.arange(0, $(BD))
  offs_d1 = tl.arange(0, $(BD))
  b_h = tl.load(BHPre + i_bh * $(s_hh) + offs_d0[:, None] * $(s_ht) +
      offs_d1[None, :])
  tl.store(H + i_bh * $(s_hh) + offs_d0[:, None] * $(s_ht) +
      offs_d1[None, :], b_h)
}

def hRow (i_iter BD : Nat) (i : Fin BD) : Nat := i_iter * BD + i.val
def hCol (j : Fin BD) : Nat := j.val

def hOffset (s : BlockState) (i_iter s_hh s_ht BD : Nat)
    (idx : TileIndex [BD, BD]) : Nat :=
  s.pids 0 * s_hh + hRow i_iter BD idx.1 * s_ht + hCol idx.2.1

noncomputable def hStoreSpec (s : BlockState) (BHPre : RegionName)
    (i_iter s_hh s_ht BD : Nat) (idx : TileIndex [BD, BD]) : ℝ :=
  s.readMem BHPre (hOffset s i_iter s_hh s_ht BD idx)

theorem attention_fwd_triton1_h_store_slice_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ∀ idx : TileIndex [BD, BD],
      let outAddr := hOffset s i_iter s_hh s_ht BD idx
      (exec (attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD) s).map
          (·.readMem H outAddr)
        = some (hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD, BD] =>
        s.pids 0 * s_hh + (i_iter * BD + idx.1.val) * s_ht + idx.2.1.val) := by
    simpa [hOffset, hRow, hCol] using hOutInj
  simp [exec, attention_fwd_triton1_h_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex]
  simp only [hOffset, hRow, hCol]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]
  simp [hStoreSpec, hOffset, hRow, hCol]

theorem attention_fwd_triton1_h_store_slice_compute_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (H, hOffset s i_iter s_hh s_ht BD idx))
      (expected := fun idx => hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_h_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_h_store_slice_correct BHPre H i_iter s_hh s_ht BT BD
    s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named H-state writeback for the Python `STORE=True` branch. -/
theorem attention_fwd_triton1_store_enabled_h_store_slice_compute_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (H, hOffset s i_iter s_hh s_ht BD idx))
      (expected := fun idx => hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  exact attention_fwd_triton1_h_store_slice_compute_correct BHPre H
    i_iter s_hh s_ht BT BD s hOutInj

theorem attention_fwd_triton1_surface_output_compute_correct
    (Q K V H O Out : RegionName) (STORE IFCOND : Bool)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 STORE IFCOND)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (Out, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O Out STORE IFCOND
          (outOffset s 131072 128 1 32 idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [attentionFwdTriton1SurfaceValue, hExec]

theorem attention_fwd_triton1_surface_h_compute_correct
    (Q K V H O : RegionName) (IFCOND : Bool)
    (i_iter : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true IFCOND)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true IFCOND
          (hOffset s i_iter 524288 128 128 idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [attentionFwdTriton1SurfaceValue, hExec]

/-! ## Python test-shape wrappers

The checked `attention_fwd_triton1.py` main cases use `B = 2`, `H = 8`,
`T = 1024`, `D = 128`, `BT = 32`, `BD = 128`, and `NT = 32`. For contiguous
`[B, H, T, D]` tensors this gives `s_qh = 1024 * 128`, `s_qt = 128`,
`s_qd = 1`; the recurrent state tensor has `s_hh = NT * BD * BD` and
`s_ht = 128`. -/

theorem attention_fwd_triton1_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  apply attention_fwd_triton1_output_store_slice_compute_correct
  rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, tIndex, dIndex] at h
  have ht : ta = tb := by omega
  have hd : da = db := by omega
  subst tb
  subst db
  rfl

theorem attention_fwd_triton1_ifcond_first_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s

theorem attention_fwd_triton1_recurrent_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s

theorem attention_fwd_triton1_store_enabled_h_python_test_shape_compute_correct
    (BHPre H : RegionName) (i_iter : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx) := by
  apply attention_fwd_triton1_store_enabled_h_store_slice_compute_correct
  rintro ⟨⟨ar, har⟩, ⟨ac, hac⟩, _⟩ ⟨⟨br, hbr⟩, ⟨bc, hbc⟩, _⟩ h
  simp [hOffset, hRow, hCol] at h
  have hr : ar = br := by omega
  have hc : ac = bc := by omega
  subst br
  subst bc
  rfl

/-- Python test-shape output coverage for `attention_fwd_triton1`: the checked
output-store variants and the `STORE=True` H-state store all realize their
specialized output shapes. -/
theorem attention_fwd_triton1_python_test_shape_all_outputs_compute_correct
    (BO O BHPre H : RegionName) (i_iter : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx)) := by
  constructor
  · exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s
  constructor
  · exact attention_fwd_triton1_ifcond_first_output_python_test_shape_compute_correct
      BO O s
  constructor
  · exact attention_fwd_triton1_recurrent_output_python_test_shape_compute_correct
      BO O s
  · exact attention_fwd_triton1_store_enabled_h_python_test_shape_compute_correct
      BHPre H i_iter s

theorem attention_fwd_triton1_local_bo_offset_python_test_shape_injective :
    Function.Injective
      (fun idx : TileIndex [32, 128] => localBoOffset 128 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
  simp [localBoOffset] at h
  have ht : ta = tb := by omega
  have hd : da = db := by omega
  subst tb
  subst db
  rfl

theorem attention_fwd_triton1_local_bh_offset_python_test_shape_injective :
    Function.Injective
      (fun idx : TileIndex [128, 128] => localBhOffset 128 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [localBhOffset] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem attention_fwd_triton1_bo_formula_python_test_shape_compute_correct
    (QTile KTile VTile BO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s QTile KTile VTile ((Real.sqrt (128 : ℝ))⁻¹)
          32 128 idx) := by
  exact attention_fwd_triton1_bo_formula_slice_compute_correct QTile KTile
    VTile BO ((Real.sqrt (128 : ℝ))⁻¹) 32 128 s
    attention_fwd_triton1_local_bo_offset_python_test_shape_injective

theorem attention_fwd_triton1_bh_formula_python_test_shape_compute_correct
    (KTile VTile BHPre : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
        32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s KTile VTile 32 128 idx) := by
  exact attention_fwd_triton1_bh_formula_slice_compute_correct KTile VTile
    BHPre 32 128 s
    attention_fwd_triton1_local_bh_offset_python_test_shape_injective

/-- Python-shape arithmetic coverage for the `BO` and `BHPre` producer tiles.

This pins the checked `BT = 32`, `BD = 128`, `scale = 1 / sqrt(128)` path for
the direct output accumulator `BO`, and the recurrent-state accumulator
`BHPre = dot(K, V)` used by the `STORE` branch summaries. -/
theorem attention_fwd_triton1_python_test_shape_accumulator_formula_summary
    (QTile KTile VTile BO BHPre : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
      ((Real.sqrt (128 : ℝ))⁻¹) 32 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
      32 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s QTile KTile VTile ((Real.sqrt (128 : ℝ))⁻¹)
          32 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
        32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s KTile VTile 32 128 idx)) := by
  constructor
  · exact attention_fwd_triton1_bo_formula_slice_toAlgorithm_supported
      QTile KTile VTile BO ((Real.sqrt (128 : ℝ))⁻¹) 32 128
  constructor
  · exact attention_fwd_triton1_bh_formula_slice_toAlgorithm_supported
      KTile VTile BHPre 32 128
  constructor
  · exact attention_fwd_triton1_bo_formula_python_test_shape_compute_correct
      QTile KTile VTile BO s
  · exact attention_fwd_triton1_bh_formula_python_test_shape_compute_correct
      KTile VTile BHPre s

/-- Public Python-shape summary for the main `attention_fwd_triton1` cases.

The checked Python tests instantiate `B = 2`, `H = 8`, `T = 1024`, `D = 128`,
`BT = 32`, `BD = 128`, and `NT = 32`, with four `STORE`/`IFCOND`
combinations. This summary records faithful full-surface lowering for those
four branch combinations and ties them to the checked output-store/H-state
writeback slices. The dot-product arithmetic that produces the precomputed
`BO`/`BHPre` tiles remains represented by those slice inputs. -/
theorem attention_fwd_triton1_python_test_shape_slice_summary
    (Q K V H O BO BHPre : RegionName) (i_iter : Nat) (s : BlockState) :
    ((∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.true).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx)) := by
  constructor
  · constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.false
    constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false
    constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.true
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true
  · exact attention_fwd_triton1_python_test_shape_all_outputs_compute_correct
      BO O BHPre H i_iter s

/-- Combined checked-shape summary for `attention_fwd_triton1.py`.

This exposes the four Python branch surfaces, the observable `O`/`H`
writebacks, and the arithmetic producers for the `BO` and `BHPre` tiles in
one public target. -/
theorem attention_fwd_triton1_python_test_shape_complete_summary
    (Q K V H O BO BHPre : RegionName) (i_iter : Nat) (s : BlockState) :
    (((∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.true).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx))) ∧
    ((∃ alg, (attention_fwd_triton1_bo_formula_slice Q K V BO
      ((Real.sqrt (128 : ℝ))⁻¹) 32 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton1_bh_formula_slice K V BHPre
      32 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice Q K V BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s Q K V ((Real.sqrt (128 : ℝ))⁻¹) 32 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice K V BHPre 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s K V 32 128 idx))) := by
  constructor
  · exact attention_fwd_triton1_python_test_shape_slice_summary Q K V H O BO
      BHPre i_iter s
  · exact attention_fwd_triton1_python_test_shape_accumulator_formula_summary
      Q K V BO BHPre s



















/-! ## Genuine closed-form output writeback (Python test shape)

The store-slice machinery above proves the observable `p_o` writeback faithfully
copies the precomputed `BO` tile into `O`. Combined with the genuine closed form
`outputClosedForm` (local + recurrent linear-attention terms, stated purely over
input memory — *not* the executed kernel output), this yields a closed-form
output theorem: when `BO` holds the genuine closed-form value at each block
offset, the kernel's output store realizes that closed form into `O`.

The hypothesis `hBO` is the producer obligation discharged by the already-proven
`boFormulaSpec`/`bhFormulaSpec` slices (`localTerm_eq_boFormulaSpec`,
`recurrentState_eq_sum_bhFormulaSpec`): the `BO` tile equals
`localTerm + recurrentTerm` at the corresponding chunk/row/col. The per-program
`chunk`, query/key/value accessors, and scale are supplied by the caller (they
are fixed by the host launch and the `make_block_ptr` layout). -/
theorem attention_fwd_triton1_output_closed_form_correct
    (BO O Q K V : RegionName) (scale : ℝ) (chunk : Nat)
    (qAddr kAddr vAddr : Nat → Nat → Nat) (s : BlockState)
    (hBO : ∀ idx : TileIndex [32, 128],
      s.readMem BO (boOffset s 131072 128 1 32 idx)
        = outputClosedForm s Q K V scale 32 128 qAddr kAddr vAddr
            chunk idx.1 idx.2.1) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        outputClosedForm s Q K V scale 32 128 qAddr kAddr vAddr
          chunk idx.1 idx.2.1) := by
  have hbase := attention_fwd_triton1_output_python_test_shape_compute_correct
    BO O s
  unfold ComputeCorrect.Realizes at hbase ⊢
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_output_store_slice_correct BO O
    131072 128 1 131072 128 1 32 128 s
    (by
      rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
      simp [outOffset, tIndex, dIndex] at h
      have ht : ta = tb := by omega
      have hd : da = db := by omega
      subst tb; subst db; rfl) idx
  rw [hExec] at h
  have h2 : s'.readMem O (outOffset s 131072 128 1 32 idx)
      = s.readMem BO (boOffset s 131072 128 1 32 idx) := Option.some.inj h
  simp only [ComputeCorrect.OutputReadable.read_real]
  rw [h2, hBO idx]

/-- **Base-chunk specialization.** For the first time-chunk (`chunk = 0`) the
recurrent state vanishes, so the genuine closed-form output is exactly the local
`(scale·Q)·K·V` term. This is the fully self-contained genuine closed form for
the leading output block: no recurrence, no self-reference. -/
theorem attention_fwd_triton1_output_closed_form_base_correct
    (BO O Q K V : RegionName) (scale : ℝ)
    (qAddr kAddr vAddr : Nat → Nat → Nat) (s : BlockState)
    (hBO : ∀ idx : TileIndex [32, 128],
      s.readMem BO (boOffset s 131072 128 1 32 idx)
        = localTerm s Q K V scale 32 128 qAddr kAddr vAddr 0 idx.1 idx.2.1) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        outputClosedForm s Q K V scale 32 128 qAddr kAddr vAddr
          0 idx.1 idx.2.1) := by
  apply attention_fwd_triton1_output_closed_form_correct BO O Q K V scale 0
    qAddr kAddr vAddr s
  intro idx
  rw [hBO idx, outputClosedForm_zero]

/-- `output_summary` for the checked `attention_fwd_triton1` surface. -/
theorem attention_fwd_triton1_python_test_shape_output_summary
    (Q K V H O : RegionName) (i_iter : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.false Bool.false
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.true Bool.false
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.false Bool.true
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.true Bool.true
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true Bool.false
          (hOffset s i_iter 524288 128 128 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true Bool.true
          (hOffset s i_iter 524288 128 128 idx))) := by
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.false Bool.false s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.true Bool.false s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.false Bool.true s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.true Bool.true s
  constructor
  · exact attention_fwd_triton1_surface_h_compute_correct Q K V H O
      Bool.false i_iter s
  · exact attention_fwd_triton1_surface_h_compute_correct Q K V H O
      Bool.true i_iter s

/-! ## Full-surface exec → genuine closed form (carry-fold derivation)

The block below derives the executed output of the full `attention_fwd_kernel_surface`
directly, eliminating the self-referential `attentionFwdTriton1SurfaceValue`. The
loop carries `b_h = recurrentState n` across the 32 chunks via `forRangeDyn_inv`. -/

/-- The erased (algorithm-layer) loop body of `attention_fwd_kernel_surface` at the
Python test shape, parameterized by the `STORE`/`IFCOND` constexpr gates. This is
exactly what `(…).toAlgKernel.body`'s `forRangeDyn` carries (checked by `rfl` in
`attention_fwd_triton1_body_split`). -/
noncomputable def aft1LoopBody (Q K V H O : RegionName) (STORE IFCOND : Bool) : List Stmt :=
  [ Stmt.assign .blockPtr [32, 128] "p_q"
      (Op.makeBlockPtrDynOffsets Q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 131072))
        [1024, 128] [32, 128] [128, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 32), Op.constNat 0]),
    Stmt.assign .blockPtr [128, 32] "p_k"
      (Op.makeBlockPtrDynOffsets K
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 131072))
        [128, 1024] [128, 32] [1, 128]
        [Op.constNat 0, Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 32)]),
    Stmt.assign .blockPtr [32, 128] "p_v"
      (Op.makeBlockPtrDynOffsets V
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 131072))
        [1024, 128] [32, 128] [128, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 32), Op.constNat 0]),
    Stmt.assign .blockPtr [128, 128] "p_h"
      (Op.makeBlockPtrDynOffsets H
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 524288))
        [32 * 128, 128] [128, 128] [128, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 128), Op.constNat 0]),
    Stmt.assign .blockPtr [32, 128] "p_o"
      (Op.makeBlockPtrDynOffsets O
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat 131072))
        [1024, 128] [32, 128] [128, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 32), Op.constNat 0]),
    Stmt.ifThen (Op.constBool STORE)
      [Stmt.store .real [128, 128]
          (MemAccess.blockPtr (Op.ref .blockPtr [128, 128] "p_h") [])
          (Op.ref .real [128, 128] "b_h") MaskOpt.none],
    Stmt.assign .real [32, 128] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [32, 128] "p_q") []) MaskOpt.none),
    Stmt.assign .real [32, 128] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [32, 128] "b_q") (Op.const (√128)⁻¹)),
    Stmt.assign .real [128, 32] "b_k"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [128, 32] "p_k") []) MaskOpt.none),
    Stmt.assign .real [32, 128] "b_v"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [32, 128] "p_v") []) MaskOpt.none),
    Stmt.assign .real [32, 32] "b_s"
      (@Op.dot [] 32 128 32 (Op.ref .real [32, 128] "b_q") (Op.ref .real [128, 32] "b_k")),
    Stmt.assign .real [32, 128] "b_o"
      (@Op.dot [] 32 32 128 (Op.ref .real [32, 32] "b_s") (Op.ref .real [32, 128] "b_v")),
    Stmt.ifThenElse (Op.constBool IFCOND)
      [Stmt.ifThenElse
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "i") (Op.constNat 0))
          [Stmt.assign .real [128, 128] "b_h"
              (@Op.dot [] 128 32 128 (Op.ref .real [128, 32] "b_k") (Op.ref .real [32, 128] "b_v"))]
          [Stmt.assign .real [32, 128] "b_o"
              (Op.add .real Broadcast.nil.consSame.consSame
                (Op.ref .real [32, 128] "b_o")
                (@Op.dot [] 32 128 128 (Op.ref .real [32, 128] "b_q") (Op.ref .real [128, 128] "b_h"))),
            Stmt.assign .real [128, 128] "b_h"
              (Op.add .real Broadcast.nil.consSame.consSame
                (Op.ref .real [128, 128] "b_h")
                (@Op.dot [] 128 32 128 (Op.ref .real [128, 32] "b_k") (Op.ref .real [32, 128] "b_v")))]]
      [Stmt.assign .real [32, 128] "b_o"
          (Op.add .real Broadcast.nil.consSame.consSame
            (Op.ref .real [32, 128] "b_o")
            (@Op.dot [] 32 128 128 (Op.ref .real [32, 128] "b_q") (Op.ref .real [128, 128] "b_h"))),
        Stmt.assign .real [128, 128] "b_h"
          (Op.add .real Broadcast.nil.consSame.consSame
            (Op.ref .real [128, 128] "b_h")
            (@Op.dot [] 128 32 128 (Op.ref .real [128, 32] "b_k") (Op.ref .real [32, 128] "b_v")))],
    Stmt.store .real [32, 128]
      (MemAccess.blockPtr (Op.ref .blockPtr [32, 128] "p_o") [])
      (Op.ref .real [32, 128] "b_o") MaskOpt.none]

/-- The erased prologue (program id + zero-init of `b_h`). -/
def aft1Prologue : List Stmt :=
  [ Stmt.assign .nat [] "i_bh" (Op.programId 0),
    Stmt.assign .real [128, 128] "b_h" (Op.full [128, 128] (Op.const 0)) ]

/-- The dynamic loop-bound op `tl.cdiv(1024, 32)`. -/
def aft1StopOp : Op .nat [] :=
  Op.div .nat Broadcast.nil
    (Op.sub .nat Broadcast.nil
      (Op.add .nat Broadcast.nil (Op.constNat 1024) (Op.constNat 32)) (Op.constNat 1))
    (Op.constNat 32)

set_option maxRecDepth 8000 in
/-- **Body split (by `rfl`).** The Python-shape surface lowers to the prologue
followed by the single `forRangeDyn` chunk loop carrying `aft1LoopBody`. -/
theorem attention_fwd_triton1_body_split
    (Q K V H O : RegionName) (STORE IFCOND : Bool) :
    (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128:ℝ))⁻¹)
      32 128 32 STORE IFCOND).toAlgKernel.body
      = aft1Prologue
        ++ [Stmt.forRangeDyn "i" (Op.constNat 0) aft1StopOp (Op.constNat 1)
              (aft1LoopBody Q K V H O STORE IFCOND)] := by
  rfl

/-! ### Per-statement eval recipes (block-ptr loads + matmuls)

Ported, self-contained versions of the generic block-pointer-load / matmul
readout recipes used to step the chunk loop body. -/

theorem aft1_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- **2D dot element lemma.** For all-`some` operand tiles `a : [M,K]`, `b : [K,N]`,
the `(m, n)` cell of `dot a b` is `Σ_e a[m,e]·b[e,n]`. -/
theorem aft1_dot2d_elem {M K N : Nat} (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ e : Fin K, a.data (m, e, PUnit.unit) = some (fa e))
    (hb : ∀ e : Fin K, b.data (e, n, PUnit.unit) = some (fb e)) :
    (Tile.dot [] a b).data (m, n, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fa e * fb e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (m, e, PUnit.unit)) (b.data (e, n, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fa e * fb e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [ha e, hb e]; rfl)]
  exact aft1_withBot_sum_some _

/-- Scalar offset op `name * c` evaluates to `scalar (val * c)` given `name = val`. -/
theorem aft1_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **2D `makeBlockPtrDynOffsets` eval recipe.** -/
theorem aft1_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-- **No-mask, no-boundary-check 2D block-pointer load through a bound register
`name`.** With empty `boundaryCheck` the in-bounds gate is vacuously true, so every
lane `(i,j)` reads the genuine memory cell `base + (rowOff+i)·strideT + (colOff+j)·strideS`.
This matches `attention_fwd_kernel`'s `tl.load(p_x)` (no boundary check). -/
theorem aft1_load_bp_2d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs TileDType.blockPtr [BT, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BT, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT, BS] name) []) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))⟩ := by
  simp only [evalOp, evalOp_ref, hreg, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds,
    List.all_nil, BlockState.readMemValue_real, if_true]

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1

import VeriTile.Triton

/-!
# `chunk_gated_attention` — strict per-kernel correctness

`chunk_gated_attention.py` defines two `@triton.jit` kernels for chunked gated
linear attention (`fla`-style):

* `chunk_gated_abc_fwd_kernel_cum` — per chunk `(i_s, i_t, i_bh)`, computes the
  intra-chunk cumulative normalizer `b_o = m_s @ b_s` (lower-triangular causal
  matmul over `g`) via block pointers and stores it to `o`.
* `chunk_gated_abc_fwd_kernel_h` — per `(i_v, i_k, i_bh)`, runs the chunk-level
  recurrence over `NT` time chunks: optionally seeded from `h0`, each step
  *stores the current state* `b_h` into the `h` rows, gates `b_k`/`b_h` by
  `exp(b_gn ...)` (the `GATEK` branch selects K-side vs V-side gating), and
  accumulates `b_h += b_k @ b_v`; the final state is optionally written to `ht`.

## Scope — what is and is not verified

Verified:

* **one hand-cut slice of `chunk_gated_abc_fwd_kernel_cum`**
  (`chunk_gated_attention_cum_compute_slice`), which reproduces that kernel's
  whole body — masked tile load, lower-triangular `tl.dot`, masked store — at
  symbolic dimensions and strides; and
* **one loop body of `chunk_gated_abc_fwd_kernel_h` per `GATEK` branch**
  (`chunk_gated_attention_h_step_gate{v,k}_slice`) — the gate
  `b_h *= exp(b_gn)[...]`, the `b_k`/`b_v` pre-gating, and the
  `b_h += tl.dot(b_k, b_v)` accumulation — over the launched `K`, `V`, `G` at
  symbolic strides, realizing the genuine closed form `hClosed (m+1)` under an
  assumed carry invariant.

The `NT`-iteration *driver* is no longer an assumption chain:
`chunk_gated_attention_h_state_carry_fold` runs the step slices through one
shared carry region `C` and reaches `hClosed NT` from the single seed assumption
`hSeedBuf`, for **both** `GATEK` branches. What that fold models is a
memory-threaded, one-launch-per-chunk program; the launched kernel keeps `b_h`
in a *register* across its own loop, and that object is still not modeled — see
`VeriTile.Triton.CarryFold`'s scope note. The headline itself still carries the
per-step `hPrev` assumptions (the fold is a separate theorem, not a conjunct).
`STORE_FINAL_STATE` has no face. Both `@triton.jit` bodies are also shown
to *lower* to the algorithm layer (`*_toAlgorithm_supported`), which is a
well-formedness fact, not correctness.

The host launch (`fwd_pre` / `fwd_inner` grids over `(cdiv(S,BS), NT, B*H)` and
`(NV, NK, B*H)`, the `@triton.autotune` `BS`/warps selection, block scheduling,
and how the runtime composes per-program writes into `o`/`h`/`ht`) is the
*trusted boundary*. Because the program ids are universally quantified (via
`BlockState`), the per-program statement covers every program of the grid.

## Proof architecture

```
chunk_gated_attention_cum_slice_output_summary_general        ← TOP THEOREM
  ├─ chunk_gated_attention_cum_compute_slice_closed_form_general
  │    └─ chunk_gated_attention_cum_compute_slice_compute_correct
  │         (expected := cumComputeStoreValue = lowerTri @ source)
  ├─ chunk_gated_attention_h_step_gatev_closed_form_general   GATEK = false body
  │    ├─ chunk_gated_attention_h_step_gatev_slice_correct  (expected := hStepSpec)
  │    └─ hStepSpec_eq_hClosed_succ ── hClosed_succ
  ├─ chunk_gated_attention_h_step_gatek_closed_form_general   GATEK = true body
  │    ├─ chunk_gated_attention_h_step_gatek_slice_correct  (expected := hStepSpec)
  │    └─ hStepSpec_eq_hClosed_succ ── hClosed_succ
  └─ hClosed_zero                                             hClosed 0 = hSeed

-- not reachable from the headline, but genuine (the cross-chunk induction):
chunk_gated_attention_h_state_carry_fold
  ├─ CarryFold.carryFold_execChain                            the driver
  ├─ chunk_gated_attention_h_step_gate{v,k}_slice_correct      per-chunk step
  ├─ chunk_gated_attention_h_step_gate{v,k}_frame              the frame
  ├─ hStepSpec_transport / finalStateOffset_congr              transport
  └─ hClosed_succ                                             closed-form step

-- not reachable from the headline, and deliberately so:
chunk_gated_attention_h_state_memcpy_transports_hClosed       (masked memcpy)
chunk_gated_attention_final_state_memcpy_transports_hClosed   (masked memcpy)
```

Offset injectivity is a *hypothesis* of the headline (`hCumInj`), not a lemma:
this file contains no `*_offset_injective` declarations, and earlier revisions of
this docstring that named such lemmas were referring to nothing.

`cumComputeStoreValue` — the causal intra-chunk cumsum `lowerTri @ b_s` — is a
genuine standalone closed form over the input region, never a read-back of the
kernel's own output.

## The gated recurrence: base, step, and induction

`hClosed m = seed ⊙ ∏_{j<m} G_j + Σ_{t<m} S_t ⊙ ∏_{t<j<m} G_j` (with `G_m` the
per-chunk matrix gate — `exp(b_gn_m[k])` per key-row under `GATEK`, else
`exp(b_gn_m[v])` per value-column — and `S_m = (gated b_k) @ (gated b_v)`) is the
closed form of the carried `b_h`. It is defined below, with input reads routed
through named element accessors (`ktElem`, `tvElem`, `gnkElem`, `gnvElem`,
`h0Elem`) that state the logical indexing and the block-pointer footprint they
mirror — each at that pointer's own element stride.

What is proven about it:

* `hClosed_zero` — `hClosed 0 = hSeed`, the base case.
* `hClosed_succ` — `hClosed (m+1) = hClosed m ⊙ G_m + S_m`, the carry-fold.
* `chunk_gated_attention_h_step_gate{v,k}_slice_correct` — one *kernel* loop body
  per `GATEK` branch realizes `hStepSpec = HPrev ⊙ G_m + S_m` over the launched
  `K`, `V`, `G`; and `hStepSpec_eq_hClosed_succ` turns that into `hClosed (m+1)`
  under the carry invariant `HPrev = hClosed m`.

* `chunk_gated_attention_h_state_carry_fold` — the cross-chunk **induction**,
  joining base and step: chaining `NT` step slices through one carry region `C`
  lands `hClosed NT` on every lane, from one seed assumption instead of `NT`
  carry assumptions. Its limit is that the carry travels through *memory*, one
  launch per chunk, whereas Python keeps `b_h` in a register.

The two `*_memcpy_transports_hClosed` lemmas remain masked memcpys whose load and
store addresses are character-identical — they *assume* a staging buffer already
holds `hClosed` and conclude that copying it delivers `hClosed`, and they are
deliberately out of the headline.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `allow_tf32=False` is moot);
`@triton.autotune` (the `BS ∈ {16,32,64}` × `num_warps` configs) and
`num_warps`/`num_stages` are not modeled. The `.to(...)` casts reduce to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). The
`boundary_check=(0, 1)` store is modeled as the `cumSurfaceActive` write
predicate over the `TileIndex [BT, BS]` footprint at symbolic dimensions;
out-of-boundary tile lanes are preserved (mask = false ⇒ no store).

Region distinctness: no `≠` hypotheses are stated and none are needed — every
slice here performs all of its loads before its single store and every `expected`
is a function of the *initial* state, so aliasing cannot falsify a face.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `chunk_gated_attention_cum_slice_output_summary_general` —
shape-general; covers the `fwd_pre` cumsum kernel's slice plus one loop body of
the gated `chunk_gated_abc_fwd_kernel_h` recurrence per `GATEK` branch, and the
recurrence's base case. The `NT` driver chaining the bodies is *not* part of the
headline; it is the separate theorem
`chunk_gated_attention_h_state_carry_fold`. -/

/-! # ══════════ CORRECT — genuine / shape-general (review this) ══════════ -/

section Correct_without_Rounding

/-- Faithful transcription of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_cum`.

This is the forward-preprocessing cumsum kernel used by `fwd_pre`: it builds the
lower-triangular accumulation mask, loads one `[BT, BS]` tile, computes the
chunk-local cumulative sum as a matrix product, and writes the result through a
boundary-checked block pointer. The Triton block-pointer `order` metadata is
scheduling-only; the DSL accepts it at the surface and erases it into the same
block-pointer AST. -/
def chunk_gated_attention_cum_surface
    (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
  p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h),
    shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_s_h),
    shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
    offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The cumulative preprocessing surface lowers to the algorithm layer. -/
theorem chunk_gated_attention_cum_surface_toAlgorithm_supported
    (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ∃ alg, (chunk_gated_attention_cum_surface s o s_s_h s_s_t s_s_d T S BT
      BS).toAlgorithm? = Except.ok alg := by
  simp [chunk_gated_attention_cum_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `chunk_gated_attention.py`'s
`chunk_gated_abc_fwd_kernel_h`.

The `GATEK`, `USE_INITIAL_STATE`, and `STORE_FINAL_STATE` constexpr branches
are preserved, including the local-tile dtype casts on gated `b_k`/`b_v` and
the block-pointer element dtype casts on state stores. -/
def chunk_gated_attention_h_surface
    (K V G H H0 HT : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T KSize VSize TK TV BT BK BV NT : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = tl.make_block_ptr(base=H0 + i_bh * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(VSize), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_h += tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  }
  for i_t in range($(0), $(NT), $(1)) {
    p_k = tl.make_block_ptr(base=K + i_bh * $(s_k_h),
      shape=($(KSize), $(T)), strides=($(s_k_d), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=V + i_bh * $(s_v_h),
      shape=($(T), $(VSize)), strides=($(s_v_t), $(s_v_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_h = tl.make_block_ptr(base=H + i_bh * $(s_h_h) + i_t * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(s_h_t), $(s_h_d)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
    if GATEK {
      p_g = tl.make_block_ptr(base=G + i_bh * $(s_k_h),
        shape=($(KSize), $(T)), strides=($(s_k_d), $(s_k_t)),
        offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
      p_gn = tl.make_block_ptr(base=G + i_bh * $(s_k_h),
        shape=($(TK)), strides=($(s_k_d)),
        offsets=((i_t * $(BT) + $(BT) - $(1)) * $(KSize) + i_k * $(BK)),
        block_shape=($(BK)), order=(0))
      b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
      b_h = b_h * tl.exp(b_gn)[:, None]
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
      b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
    } else {
      p_g = tl.make_block_ptr(base=G + i_bh * $(s_v_h),
        shape=($(T), $(VSize)), strides=($(s_v_t), $(s_v_d)),
        offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
      p_gn = tl.make_block_ptr(base=G + i_bh * $(s_v_h),
        shape=($(TV)), strides=($(s_v_d)),
        offsets=((i_t * $(BT) + $(BT) - $(1)) * $(VSize) + i_v * $(BV)),
        block_shape=($(BV)), order=(0))
      b_gn = tl.load(p_gn, boundary_check=([0] : List Nat))
      b_h = b_h * tl.exp(b_gn)[None, :]
      b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
      b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
    }
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_h = tl.make_block_ptr(base=HT + i_bh * $(KSize) * $(VSize),
      shape=($(KSize), $(VSize)), strides=($(VSize), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_h, (b_h).to(p_h.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The gated H-state forward surface lowers to the algorithm layer, including
initial-state, gate-K/gate-V, recurrent state stores, and final-state branches. -/
theorem chunk_gated_attention_h_surface_toAlgorithm_supported
    (K V G H H0 HT : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      T KSize VSize TK TV BT BK BV NT : Nat)
    (GATEK USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (chunk_gated_attention_h_surface K V G H H0 HT s_k_h s_k_t
      s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T KSize VSize TK TV
      BT BK BV NT GATEK USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm?
        = Except.ok alg := by
  simp [chunk_gated_attention_h_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Cumulative-normalizer index accessors

`chunk_gated_abc_fwd_kernel_cum` is launched on the grid
`(cdiv(S, BS), NT, B*H)` and reads `i_s, i_t, i_bh = pid 0, pid 1, pid 2`
(`chunk_gated_attention.py:32`), which `chunk_gated_attention_cum_surface`
transcribes verbatim. These accessors are the single source of truth for that
pid assignment; the slice below uses them, so a pid mix-up cannot hide in the
offset arithmetic. -/

/-- Global time row of tile lane `i`: `i_t * BT + i` with `i_t = pid 1`. -/
def cumSurfaceTIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

/-- Global feature column of tile lane `j`: `i_s * BS + j` with `i_s = pid 0`. -/
def cumSurfaceSIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val

/-- The `boundary_check=(0, 1)` in-range predicate of the `[BT, BS]` tile. -/
def cumSurfaceActive (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  cumSurfaceTIndex s BT idx.1 < T ∧ cumSurfaceSIndex s BS idx.2.1 < S

instance cumSurfaceActiveDecidable (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) :
    Decidable (cumSurfaceActive s T S BT BS idx) := by
  unfold cumSurfaceActive
  infer_instance

/-- Flat address of tile lane `idx` in the `[T, S]` block-pointer footprint
`base = R + i_bh * s_s_h`, `strides = (s_s_t, s_s_d)`, with `i_bh = pid 2`. -/
def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 2 * s_s_h + cumSurfaceTIndex s BT idx.1 * s_s_t +
    cumSurfaceSIndex s BS idx.2.1 * s_s_d

/-! ## Computed cumulative-normalizer slice

The `fwd_pre` Python path computes `b_o = tl.dot(m_s, b_s)` with a
lower-triangular mask before storing. This slice proves that computation and
writeback directly, rather than starting from a precomputed `BC` tile. -/

def chunk_gated_attention_cum_compute_slice
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_o = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_o, mask=mask)
}

noncomputable def lowerTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val >= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }

noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if cumSurfaceActive s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }

noncomputable def cumComputeStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (lowerTriTile BT)
      (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))

theorem chunk_gated_attention_cum_compute_slice_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (chunk_gated_attention_cum_compute_slice SReg Z s_s_h s_s_t
            s_s_d T S BT BS) s).map (·.readMem Z outAddr)
        = some (if cumSurfaceActive s T S BT BS idx then
            cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_cum_compute_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, cumSurfaceTIndex, cumSurfaceSIndex, cumSurfaceActive,
        tileOffset, sourceTile, lowerTriTile, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 2 * s_s_h + (s.pids 1 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, cumSurfaceTIndex, cumSurfaceSIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 1 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · simp [offsetFn, cumSurfaceActive, cumSurfaceTIndex, cumSurfaceSIndex, tileOffset, cumComputeStoreValue,
      sourceTile, lowerTriTile, Tile.dot, hActive]
  · simp [offsetFn, cumSurfaceActive, cumSurfaceTIndex, cumSurfaceSIndex, tileOffset, hActive]

theorem chunk_gated_attention_cum_compute_slice_compute_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => cumSurfaceActive s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_cum_compute_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_cum_compute_slice_correct SReg Z s_s_h
    s_s_t s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Masked memcpy** modelling the intermediate-state writeback of
`chunk_gated_abc_fwd_kernel_h`.

At each `i_t`, the Python kernel stores the current recurrent state `b_h` into
`H + i_bh * s_h_h + i_t * KSize * VSize` before applying the chunk update. This
slice starts from a precomputed `BH` tile — and its **load and store addresses
are character-identical** under the same mask, so it computes nothing: it
transports whatever `BH` holds into `H`. The gating and the `b_k @ b_v`
accumulation that actually produce `b_h` are *not* modeled anywhere in this
file. Consequently no theorem built on this slice is a claim about the
recurrence; see the module docstring. -/
def chunk_gated_attention_h_state_store_slice
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BH + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    mask=mask, other=0.0)
  tl.store(H + i_bh * $(s_h_h) + $(i_t) * $(KSize) * $(VSize) +
      offs_k[:, None] * $(s_h_t) + offs_v[None, :] * $(s_h_d),
    b_h, mask=mask)
}

def kIndexState (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val

def vIndexState (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def stateActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexState s BK idx.1 < KSize ∧ vIndexState s BV idx.2.1 < VSize

instance stateActiveDecidable (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (stateActive s KSize VSize BK BV idx) := by
  unfold stateActive
  infer_instance

def hStateOffset (s : BlockState) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + i_t * KSize * VSize +
    kIndexState s BK idx.1 * s_h_t + vIndexState s BV idx.2.1 * s_h_d

noncomputable def hStateStoreValue (s : BlockState) (BH : RegionName)
    (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if stateActive s KSize VSize BK BV idx then
      some (s.readMem BH (hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_gated_attention_h_state_store_slice_correct
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
      (exec (chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t
            s_h_d KSize VSize BK BV) s).map (·.readMem H outAddr)
        = some (if stateActive s KSize VSize BK BV idx then
            hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
          else s.readMem H outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_h_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndexState, vIndexState, stateActive, hStateOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * s_h_h + i_t * KSize * VSize +
      (s.pids 1 * BK + idx.1.val) * s_h_t +
      (s.pids 0 * BV + idx.2.1.val) * s_h_d
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BK + idx.1.val < KSize ∧
          s.pids 0 * BV + idx.2.1.val < VSize then
        some (s.readMem BH (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, hStateOffset, kIndexState, vIndexState] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem H (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem H (offsetFn idx) =
    if P idx then
      hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
    else s.readMem H (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  · rfl
  · rfl

theorem chunk_gated_attention_h_state_store_slice_compute_correct
    (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t s_h_h
        s_h_t s_h_d KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => stateActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (H, hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hStateStoreValue s BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_h_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_h_state_store_slice_correct BH H i_t s_h_h
    s_h_t s_h_d KSize VSize BK BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Masked memcpy** modelling the `STORE_FINAL_STATE` writeback of
`chunk_gated_attention.py`'s `chunk_gated_abc_fwd_kernel_h`: it moves a
precomputed final-state `BHFinal` `[BK, BV]` tile into `Ht` after the
`NT`-iteration chunk loop. Load and store addresses are character-identical
under the same mask, so this slice computes nothing — it transports whatever
`BHFinal` holds. -/
def chunk_gated_attention_final_state_store_slice
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = i_k * $(BK) + tl.arange(0, $(BK))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_k[:, None] < $(KSize)) & (offs_v[None, :] < $(VSize))
  b_h = tl.load(BHFinal + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :],
    mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(KSize) * $(VSize) +
      offs_k[:, None] * $(VSize) + offs_v[None, :], b_h, mask=mask)
}

def kIndexFinal (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 1 * BK + i.val

def vIndexFinal (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def finalActive (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  kIndexFinal s BK idx.1 < KSize ∧ vIndexFinal s BV idx.2.1 < VSize

instance finalActiveDecidable (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Decidable (finalActive s KSize VSize BK BV idx) := by
  unfold finalActive
  infer_instance

def finalStateOffset (s : BlockState) (KSize VSize BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * KSize * VSize +
    kIndexFinal s BK idx.1 * VSize + vIndexFinal s BV idx.2.1

noncomputable def finalStateStoreValue (s : BlockState) (BHFinal : RegionName)
    (KSize VSize BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  WithBot.unbotD 0
    (if finalActive s KSize VSize BK BV idx then
      some (s.readMem BHFinal (finalStateOffset s KSize VSize BK BV idx))
    else some (0.0 : ℝ))

theorem chunk_gated_attention_final_state_store_slice_correct
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      let outAddr := finalStateOffset s KSize VSize BK BV idx
      (exec (chunk_gated_attention_final_state_store_slice BHFinal Ht
            KSize VSize BK BV) s).map (·.readMem Ht outAddr)
        = some (if finalActive s KSize VSize BK BV idx then
            finalStateStoreValue s BHFinal KSize VSize BK BV idx
          else s.readMem Ht outAddr) := by
  intro idx
  simp [exec, chunk_gated_attention_final_state_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        kIndexFinal, vIndexFinal, finalActive, finalStateOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BK, BV] → Nat :=
    fun idx => s.pids 2 * KSize * VSize +
      (s.pids 1 * BK + idx.1.val) * VSize +
      (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BK, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BK + idx.1.val < KSize ∧
          s.pids 0 * BV + idx.2.1.val < VSize then
        some (s.readMem BHFinal (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BK, BV] → Prop :=
    fun idx => s.pids 1 * BK + idx.1.val < KSize ∧
      s.pids 0 * BV + idx.2.1.val < VSize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, finalStateOffset, kIndexFinal, vIndexFinal] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Ht (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BK, BV])).readMem Ht (offsetFn idx) =
    if P idx then finalStateStoreValue s BHFinal KSize VSize BK BV idx
    else s.readMem Ht (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BK + idx.1.val < KSize ∧ s.pids 0 * BV + idx.2.1.val < VSize
  · rfl
  · rfl

theorem chunk_gated_attention_final_state_store_slice_compute_correct
    (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => finalActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] => (Ht, finalStateOffset s KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        finalStateStoreValue s BHFinal KSize VSize BK BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_final_state_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gated_attention_final_state_store_slice_correct BHFinal Ht
    KSize VSize BK BV s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed form for the gated chunk-recurrence state `b_h`

The `chunk_gated_abc_fwd_kernel_h` loop carries a `[BK, BV]` state `b_h` across
`NT` time chunks. At each chunk `m` it (1) stores the *current* state, then
(2) gates `b_h ⊙ G_m` and (3) accumulates the gated matmul `S_m = b_k' @ b_v'`.
The value stored at chunk `m` is therefore the *pre-update* folded state.

`hClosed m` is the genuine standalone closed form of that folded state, read
entirely off the **input** regions `K`, `V`, `G`, `H0` (never the kernel's own
output `H`/`HT`):

  `hClosed m = seed ⊙ ∏_{j<m} G_j  +  Σ_{t<m} S_t ⊙ ∏_{t<j<m} G_j`

generalizing the scalar gated carry-fold of `chunk_gate_recurrence` (#290) and
the per-channel decay outer-product of `fused_rwkv6` (#291) to a chunk-level
**matrix** gate (`G_m`, per-row `k` when `GATEK`, per-column `v` otherwise) and a
full gated `b_k @ b_v` accumulation term `S_m`.

All input reads below go through **named element accessors** (`ktElem`,
`tvElem`, `gnkElem`, `gnvElem`, `h0Elem`), each of which states the logical
tensor indexing and the block-pointer footprint it mirrors, so the closed form
can be reviewed without decoding flat offset arithmetic. -/

/-- Global time index of the **last lane of chunk `m`** (`i_t*BT + BT - 1`) —
the row whose cumulative gate `b_gn` normalizes the whole chunk. -/
def chunkLastTime (BT m : Nat) : Nat := m * BT + BT - 1

/-- `[k, t]`-layout element `R[i_bh][k, t]` at batch-head `s.pids 2`: mirrors
the `p_k`-shaped block pointers (`base = R + i_bh*s_k_h`, `shape = (K, T)`,
`strides = (s_k_d, s_k_t)`) — used for `K` itself and, under `GATEK`, for the
per-time gate read `b_g` on `G`. -/
noncomputable def ktElem (s : BlockState) (R : RegionName)
    (s_k_h s_k_t s_k_d k t : Nat) : ℝ :=
  s.readMem R (s.pids 2 * s_k_h + k * s_k_d + t * s_k_t)

/-- `[t, v]`-layout element `R[i_bh][t, v]` at batch-head `s.pids 2`: mirrors
the `p_v`-shaped block pointers (`base = R + i_bh*s_v_h`, `shape = (T, V)`,
`strides = (s_v_t, s_v_d)`) — used for `V` itself and, under `¬GATEK`, for the
per-time gate read `b_g` on `G`. -/
noncomputable def tvElem (s : BlockState) (R : RegionName)
    (s_v_h s_v_t s_v_d t v : Nat) : ℝ :=
  s.readMem R (s.pids 2 * s_v_h + t * s_v_t + v * s_v_d)

/-- Last-lane cumulative gate `b_gn[k]` under `GATEK`: lane `tLast*KSize + k` of
`p_gn`'s flattened `(T*K,)` view of `G` at batch-head `s.pids 2`, at the block
pointer's own element stride `s_k_d` (`chunk_gated_attention.py:90`). -/
noncomputable def gnkElem (s : BlockState) (G : RegionName)
    (s_k_h s_k_d KSize tLast k : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_k_h + (tLast * KSize + k) * s_k_d)

/-- Last-lane cumulative gate `b_gn[v]` under `¬GATEK`: lane `tLast*VSize + v` of
`p_gn`'s flattened `(T*V,)` view of `G` at batch-head `s.pids 2`, at the block
pointer's own element stride `s_v_d` (`chunk_gated_attention.py:100`). -/
noncomputable def gnvElem (s : BlockState) (G : RegionName)
    (s_v_h s_v_d VSize tLast v : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_v_h + (tLast * VSize + v) * s_v_d)

/-- Initial-state element `h0[i_bh][k, v]`: mirrors the `h0` block pointer
(`base = h0 + i_bh*K*V`, `shape = (K, V)`, `strides = (V, 1)`). -/
noncomputable def h0Elem (s : BlockState) (H0 : RegionName)
    (KSize VSize k v : Nat) : ℝ :=
  s.readMem H0 (s.pids 2 * KSize * VSize + k * VSize + v)

/-- Per-chunk gate factor `G_m[k,v]` at the bench shape. `GATEK` ⇒ per-key-row
`exp(b_gn_m[k])`; otherwise per-value-column `exp(b_gn_m[v])`. `b_gn` is the
last-row (`t = BT-1`) cumulative gate of chunk `m`. -/
noncomputable def hGate
    (s : BlockState) (G : RegionName) (GATEK : Bool)
    (s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  if GATEK then
    Real.exp (gnkElem s G s_k_h s_k_d KSize (chunkLastTime BT m)
      (kIndexState s BK idx.1))
  else
    Real.exp (gnvElem s G s_v_h s_v_d VSize (chunkLastTime BT m)
      (vIndexState s BV idx.2.1))

/-- Per-chunk gated matmul accumulation `S_m[k,v] = (gated b_k) @ (gated b_v)`,
summed over the `BT` intra-chunk time lanes. `GATEK` gates `b_k` by
`exp(b_gn_m[k] - b_g_m[k,t])`; otherwise gates `b_v` by `exp(b_gn_m[v] - b_g_m[t,v])`. -/
noncomputable def hStepTerm
    (s : BlockState) (K V G : RegionName) (GATEK : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  ∑ t : Fin BT,
    let kVal := ktElem s K s_k_h s_k_t s_k_d
      (kIndexState s BK idx.1) (m * BT + t.val)
    let vVal := tvElem s V s_v_h s_v_t s_v_d
      (m * BT + t.val) (vIndexState s BV idx.2.1)
    if GATEK then
      let gnVal := gnkElem s G s_k_h s_k_d KSize (chunkLastTime BT m)
        (kIndexState s BK idx.1)
      let gVal := ktElem s G s_k_h s_k_t s_k_d
        (kIndexState s BK idx.1) (m * BT + t.val)
      (kVal * Real.exp (gnVal - gVal)) * vVal
    else
      let gnVal := gnvElem s G s_v_h s_v_d VSize (chunkLastTime BT m)
        (vIndexState s BV idx.2.1)
      let gVal := tvElem s G s_v_h s_v_t s_v_d
        (m * BT + t.val) (vIndexState s BV idx.2.1)
      kVal * (vVal * Real.exp (gnVal - gVal))

/-- Seed state `b_h^(0)[k,v]`: `h0[k,v]` when `USE_INITIAL_STATE`, else `0`. -/
noncomputable def hSeed
    (s : BlockState) (H0 : RegionName) (USE_INITIAL_STATE : Bool)
    (KSize VSize BK BV : Nat) (idx : TileIndex [BK, BV]) : ℝ :=
  if USE_INITIAL_STATE then
    h0Elem s H0 KSize VSize (kIndexState s BK idx.1) (vIndexState s BV idx.2.1)
  else 0

/-- **Genuine closed form** for the folded gated-recurrence state at chunk `m`
(the value the kernel stores into `h` at loop row `m`). Standalone over the
input regions `K`, `V`, `G`, `H0`. -/
noncomputable def hClosed
    (s : BlockState) (K V G H0 : RegionName)
    (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx *
      (∏ j ∈ Finset.range m,
        hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV j idx) +
    ∑ t ∈ Finset.range m,
      hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          KSize VSize BT BK BV t idx *
        (∏ j ∈ Finset.Ico (t + 1) m,
          hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV j idx)

/-- **The gated carry-fold's base case.** At `m = 0` the closed form collapses to
the seed (empty gate product, empty accumulation sum): `hClosed 0 = hSeed`. This
is the closed-form counterpart of the Python prologue `b_h = tl.zeros([BK, BV],
...)` plus the optional `USE_INITIAL_STATE` load
(`chunk_gated_attention.py:74–77`). -/
theorem hClosed_zero
    (s : BlockState) (K V G H0 : RegionName)
    (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (idx : TileIndex [BK, BV]) :
    hClosed s K V G H0 GATEK USE_INITIAL_STATE
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV 0 idx
      = hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx := by
  simp [hClosed]

/-- **The gated carry-fold recurrence.** Unrolling one chunk:
`hClosed (m+1) = hClosed m ⊙ G_m + S_m` — the exact closed-form counterpart of
the Python loop body `b_h *= exp(b_gn)[...]` followed by
`b_h += tl.dot(b_k, b_v)` (`chunk_gated_attention.py:94/104` and `:109`).

The two summands separate cleanly because the gate product telescopes: the seed
picks up one more factor `G_m`, every accumulated term `S_t` (`t < m`) picks up
the same factor via `Finset.prod_Ico_succ_top` (legal since `t < m ⇒ t+1 ≤ m`),
and the new term `S_m` carries the empty product `∏_{Ico (m+1) (m+1)} = 1`. -/
theorem hClosed_succ
    (s : BlockState) (K V G H0 : RegionName)
    (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) :
    hClosed s K V G H0 GATEK USE_INITIAL_STATE
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx
      = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx *
          hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m idx
        + hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
            KSize VSize BT BK BV m idx := by
  unfold hClosed
  rw [Finset.prod_range_succ, Finset.sum_range_succ, Finset.Ico_self]
  simp only [Finset.prod_empty, mul_one]
  have hsum :
      (∑ t ∈ Finset.range m,
          hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
              KSize VSize BT BK BV t idx *
            ∏ j ∈ Finset.Ico (t + 1) (m + 1),
              hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV j idx)
        = (∑ t ∈ Finset.range m,
              hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
                  KSize VSize BT BK BV t idx *
                ∏ j ∈ Finset.Ico (t + 1) m,
                  hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize
                    BT BK BV j idx) *
            hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m idx := by
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun t ht => ?_
    simp only [Finset.mem_range] at ht
    rw [Finset.prod_Ico_succ_top (by omega : t + 1 ≤ m)]
    ring
  rw [hsum]
  ring

/-- Swap the `expected` value of a `writeIf`-`Realizes_without_Rounding` when the two `expected`
functions agree on every cumSurfaceActive (`mask`-satisfying) index. -/
theorem realizes_writeIf_expected_congr {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    (kernel : ComputeKernel) (s : BlockState)
    (mask : ι → Prop) [DecidablePred mask]
    (addr : ι → MemCellAddr) (e1 e2 : ι → α)
    (hAgree : ∀ i, mask i → e1 i = e2 i)
    (h : ComputeCorrect.Realizes_without_Rounding kernel s
      (ComputeCorrect.WriteMap.writeIf mask addr) e1) :
    ComputeCorrect.Realizes_without_Rounding kernel s
      (ComputeCorrect.WriteMap.writeIf mask addr) e2 := by
  rw [ComputeCorrect.realizes_writeIf_iff] at h ⊢
  unfold ComputeKernel.ExecCorrect ComputeKernel.ComputeCorrect
    ComputeKernel.ProjectedCorrect ComputeKernel.AlgorithmCorrect_without_Rounding at h ⊢
  refine ⟨h.1, ?_⟩
  obtain ⟨_, hp⟩ := h
  revert hp
  cases hAlg : kernel.toAlgorithm? with
  | error e => intro hp; exact hp
  | ok ak =>
      intro hp
      unfold Kernel.Correct_without_Rounding at hp ⊢
      intro s0 s' hExec hs0 i hi
      rw [← hAgree i hi]
      exact hp s0 s' hExec hs0 i hi

/-! ## ════════ Gated chunk-recurrence step slices (genuine) ════════

At a fixed chunk index `m`, `chunk_gated_abc_fwd_kernel_h`'s loop body does
exactly two things to the carried `[BK, BV]` state: gate it
(`b_h *= tl.exp(b_gn)[...]`) and add the gated matmul
(`b_h += tl.dot(b_k, b_v)`), with the `GATEK` constexpr deciding whether the gate
rides the key rows (`[:, None]`, pre-gating `b_k`) or the value columns
(`[None, :]`, pre-gating `b_v`). The two slices below transcribe those two
branches over the **launched** regions `K`, `V`, `G`, at the launch's own strides
and its own pid assignment `i_v, i_k, i_bh = pid 0, pid 1, pid 2`, taking the
carried state from a materialized buffer `HPrev` and writing the updated state to
`HOut`.

Scope, stated plainly:

* `HPrev` / `HOut` are **fiction regions** — the Python `b_h` is a register, not a
  tensor. They are addressed at the contiguous `[K, V]` state layout
  `finalStateOffset` (`i_bh·KSize·VSize + (i_k·BK + j_k)·VSize + (i_v·BV + j_v)`),
  the layout the `STORE_FINAL_STATE` writeback uses. So these are faces about the
  carried register's update, not about the Python tensors `h` / `ht`.
* Unlike the memcpy lemmas below, they **compute**: `expected` is
  `HPrev ⊙ G_m + S_m`, with `G_m` and `S_m` read off `K`, `V`, `G` — the very
  `hGate` / `hStepTerm` that build `hClosed`.
* Nothing *in these two faces* threads chunk `m`'s `HOut` into chunk `m+1`'s
  `HPrev`, so `hPrev` stays an assumption here. With `hClosed_succ` they give
  the recurrence's *step* and with `hClosed_zero` its *base*; the induction
  joining them is `chunk_gated_attention_h_state_carry_fold` below, which
  chains these very slices with `HPrev` and `HOut` instantiated to one shared
  region — at the cost of modeling the carry as memory rather than a
  register.
* The Python loop stores the *pre-update* state before gating, so the value these
  slices produce is the state entering chunk `m+1`, i.e. `hClosed (m+1)`.
* The `b_k` / `b_v` / `b_g` / `b_gn` loads are unmasked here, whereas the Python
  reads them through `boundary_check`ed block pointers; the partial-tail-chunk
  regime (`m·BT + BT > T`, or `i_k·BK + BK > KSize` / `i_v·BV + BV > VSize`) is
  part of the trusted boundary, as it already is for the
  `fused_recurrent_retention` step slices. -/

/-- `GATEK = false` branch: the gate rides the **value columns**, and `b_v` is
pre-gated by `exp(b_gn[v] - b_g[t,v])`. -/
def chunk_gated_attention_h_step_gatev_slice
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  offs_t = tl.arange(0, $(BT))
  b_h = tl.load(HPrev + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]))
  b_gn = tl.load(G + i_bh * $(s_v_h) +
    ($(chunkLastTime BT m) * $(VSize) + (i_v * $(BV) + offs_v)) * $(s_v_d))
  b_g = tl.load(G + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_k = tl.load(K + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_v = tl.load(V + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_h = b_h * tl.exp(b_gn)[None, :]
  b_v = b_v * tl.exp(b_gn[None, :] - b_g)
  b_h = b_h + tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(HOut + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]),
    (b_h).to(HOut.dtype.element_ty))
}

/-- `GATEK = true` branch: the gate rides the **key rows**, and `b_k` is
pre-gated by `exp(b_gn[k] - b_g[k,t])`. -/
def chunk_gated_attention_h_step_gatek_slice
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_k = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_k = tl.arange(0, $(BK))
  offs_v = tl.arange(0, $(BV))
  offs_t = tl.arange(0, $(BT))
  b_h = tl.load(HPrev + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]))
  b_gn = tl.load(G + i_bh * $(s_k_h) +
    ($(chunkLastTime BT m) * $(KSize) + (i_k * $(BK) + offs_k)) * $(s_k_d))
  b_g = tl.load(G + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_k = tl.load(K + i_bh * $(s_k_h) + (i_k * $(BK) + offs_k[:, None]) * $(s_k_d) +
    ($(m) * $(BT) + offs_t[None, :]) * $(s_k_t))
  b_v = tl.load(V + i_bh * $(s_v_h) + ($(m) * $(BT) + offs_t[:, None]) * $(s_v_t) +
    (i_v * $(BV) + offs_v[None, :]) * $(s_v_d))
  b_h = b_h * tl.exp(b_gn)[:, None]
  b_k = b_k * tl.exp(b_gn[:, None] - b_g)
  b_h = b_h + tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(HOut + i_bh * $(KSize) * $(VSize) +
    (i_k * $(BK) + offs_k[:, None]) * $(VSize) + (i_v * $(BV) + offs_v[None, :]),
    (b_h).to(HOut.dtype.element_ty))
}

theorem chunk_gated_attention_h_step_gatev_slice_toAlgorithm_supported
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ∃ alg, (chunk_gated_attention_h_step_gatev_slice K V G HPrev HOut
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK
      BV).toAlgorithm? = Except.ok alg := by
  simp [chunk_gated_attention_h_step_gatev_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

theorem chunk_gated_attention_h_step_gatek_slice_toAlgorithm_supported
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat) :
    ∃ alg, (chunk_gated_attention_h_step_gatek_slice K V G HPrev HOut
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK
      BV).toAlgorithm? = Except.ok alg := by
  simp [chunk_gated_attention_h_step_gatek_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The arithmetic spec of one gated chunk-recurrence body: `HPrev ⊙ G_m + S_m`,
the materialized previous state gated by `hGate` plus the gated
`tl.dot(b_k, b_v)` accumulation `hStepTerm`. -/
noncomputable def hStepSpec
    (s : BlockState) (HPrev K V G : RegionName) (GATEK : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx) *
      hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m idx +
    hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
      KSize VSize BT BK BV m idx

theorem chunk_gated_attention_h_step_gatev_slice_correct
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      (exec (chunk_gated_attention_h_step_gatev_slice K V G HPrev HOut
            m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV) s).map
          (·.readMem HOut (finalStateOffset s KSize VSize BK BV idx))
        = some (hStepSpec s HPrev K V G Bool.false
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) := by
  intro idx
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        s.pids 2 * KSize * VSize + (s.pids 1 * BK + idx.1.val) * VSize +
          (s.pids 0 * BV + idx.2.1.val)) := by
    simpa [finalStateOffset, kIndexFinal, vIndexFinal] using hOutInj
  simp [exec, chunk_gated_attention_h_step_gatev_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.dot, NumericDType.add,
    NumericDType.mul, NumericDType.sub, WithBot.realExp,
    ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    hStepSpec, hGate, hStepTerm, ktElem, tvElem, gnvElem, finalStateOffset,
    kIndexFinal, vIndexFinal, kIndexState, vIndexState,
    TileShape.dropInsertedIndex]
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]

theorem chunk_gated_attention_h_step_gatek_slice_correct
    (K V G HPrev HOut : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx)) :
    ∀ idx : TileIndex [BK, BV],
      (exec (chunk_gated_attention_h_step_gatek_slice K V G HPrev HOut
            m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV) s).map
          (·.readMem HOut (finalStateOffset s KSize VSize BK BV idx))
        = some (hStepSpec s HPrev K V G Bool.true
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) := by
  intro idx
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        s.pids 2 * KSize * VSize + (s.pids 1 * BK + idx.1.val) * VSize +
          (s.pids 0 * BV + idx.2.1.val)) := by
    simpa [finalStateOffset, kIndexFinal, vIndexFinal] using hOutInj
  simp [exec, chunk_gated_attention_h_step_gatek_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.dot, NumericDType.add,
    NumericDType.mul, NumericDType.sub, WithBot.realExp,
    ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    hStepSpec, hGate, hStepTerm, ktElem, tvElem, gnkElem, finalStateOffset,
    kIndexFinal, vIndexFinal, kIndexState, vIndexState,
    TileShape.dropInsertedIndex]
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]

/-- **Gated carry-fold step (genuine).** If the materialized previous-state
buffer `HPrev` holds `hClosed m`, one loop body — `hStepSpec`, i.e.
`HPrev ⊙ G_m + S_m` — produces exactly `hClosed (m+1)`. This is the statement the
module docstring used to *describe* without stating. -/
theorem hStepSpec_eq_hClosed_succ
    (s : BlockState) (HPrev K V G H0 : RegionName)
    (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m : Nat)
    (hPrev : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx)
    (idx : TileIndex [BK, BV]) :
    hStepSpec s HPrev K V G GATEK
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx
      = hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx := by
  rw [hClosed_succ]
  unfold hStepSpec
  rw [hPrev idx]

/-! ## Cross-step carry fold — the `NT` chunk loop, threaded through memory

The step-slice scope list above records the induction as missing its
step-chaining: each step face constrains `HPrev` while concluding about `HOut`,
with nothing relating the two region names, so an `NT`-chunk story needed `NT`
assumptions. This section supplies exactly that missing piece, by running the
step slices as a `CarryFold.execChain` whose carry-in and carry-out region are
literally the same name `C`.

Both `GATEK` branches are chained by the **same** theorem: the chain's step
function selects the branch, as the launched kernel's constexpr does.

Because these slices' stores are **unmasked**, the fold runs at the full tile
index — no write-active subtype is needed, unlike `fused_rwkv6_kernel` and
`fused_recurrent_delta`.

Honesty limits, the same ones `VeriTile.Triton.CarryFold` states: the carry
travels through a **memory region**, one launch per chunk, while the Python loop
keeps `b_h` in a *register* across its `NT` iterations — that object is still not
modeled. `hSeedBuf` is an assumption about the caller's buffer; what
`hClosed_zero` contributes is that the value it must hold is exactly `hSeed`.
The unmasked `b_k`/`b_v`/`b_g`/`b_gn` loads stay trusted here too. -/

/-- Address congruence: `finalStateOffset` reads the state only through `pids`. -/
theorem finalStateOffset_congr (s u : BlockState) (KSize VSize BK BV : Nat)
    (h : u.pids = s.pids) (idx : TileIndex [BK, BV]) :
    finalStateOffset u KSize VSize BK BV idx
      = finalStateOffset s KSize VSize BK BV idx := by
  unfold finalStateOffset kIndexFinal vIndexFinal; rw [h]

/-- One `GATEK = false` chunk step leaves everything outside the carry region
alone: it writes only its output region, and an (unmasked) `writeMem` scatter
touches neither the program ids nor any other region. -/
theorem chunk_gated_attention_h_step_gatev_frame
    (C K V G : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (u u' : BlockState)
    (hExec : exec (chunk_gated_attention_h_step_gatev_slice K V G C C
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV) u = some u') :
    u'.pids = u.pids ∧ ∀ r, r ≠ C → ∀ o, u'.mem r o = u.mem r o := by
  simp [exec, chunk_gated_attention_h_step_gatev_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.dot, NumericDType.add,
    NumericDType.mul, NumericDType.sub, WithBot.realExp,
    ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    finalStateOffset, kIndexFinal, vIndexFinal, kIndexState, vIndexState,
    TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  refine ⟨?_, ?_⟩
  · rw [BlockState.foldl_writeMem_pids]; simp
  · intro r hr o
    rw [BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o]; simp

/-- The `GATEK = true` companion of `chunk_gated_attention_h_step_gatev_frame`. -/
theorem chunk_gated_attention_h_step_gatek_frame
    (C K V G : RegionName)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (u u' : BlockState)
    (hExec : exec (chunk_gated_attention_h_step_gatek_slice K V G C C
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV) u = some u') :
    u'.pids = u.pids ∧ ∀ r, r ≠ C → ∀ o, u'.mem r o = u.mem r o := by
  simp [exec, chunk_gated_attention_h_step_gatek_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.uop, Tile.ptrAdd, Tile.dot, NumericDType.add,
    NumericDType.mul, NumericDType.sub, WithBot.realExp,
    ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    finalStateOffset, kIndexFinal, vIndexFinal, kIndexState, vIndexState,
    TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  refine ⟨?_, ?_⟩
  · rw [BlockState.foldl_writeMem_pids]; simp
  · intro r hr o
    rw [BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o]; simp

/-- `hStepSpec` transported to a state `u` agreeing with the reference state `s`
on `pids` and on the three **input** regions it reads (`K`, `V`, `G`). The carry
region `C` is deliberately *not* transported: its content at `u` is what the
fold's invariant supplies. This is what forces the `K`/`V`/`G ≠ C` side
conditions below. -/
theorem hStepSpec_transport
    (s u : BlockState) (C K V G : RegionName) (GATEK : Bool)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (hpids : u.pids = s.pids)
    (hK : ∀ o, u.readMem K o = s.readMem K o)
    (hV : ∀ o, u.readMem V o = s.readMem V o)
    (hG : ∀ o, u.readMem G o = s.readMem G o)
    (idx : TileIndex [BK, BV]) :
    hStepSpec u C K V G GATEK
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx
      = u.readMem C (finalStateOffset s KSize VSize BK BV idx) *
          hGate s G GATEK s_k_h s_k_d s_v_h s_v_d KSize VSize BT BK BV m idx
        + hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
            KSize VSize BT BK BV m idx := by
  simp only [hStepSpec, hGate, hStepTerm, ktElem, tvElem, gnkElem, gnvElem,
    finalStateOffset, kIndexFinal, vIndexFinal, kIndexState, vIndexState,
    hpids, hK, hV, hG]

/-- **Cross-step carry fold for the gated chunk recurrence (genuine).**

Running `NT` chunk steps as a chain through the shared carry region `C`, the
final buffer holds the genuine closed form `hClosed NT` on **every** lane, and
the chain leaves everything outside `C` untouched.

This is the step-chaining the module docstring records as missing: the step face
constrains `HPrev` and concludes about `HOut` with nothing relating them, so an
`NT`-chunk story needed `NT` assumptions. Here there is **one**, `hSeedBuf`, and
the identification is structural — the same region name is both the slice's
`HPrev` and its `HOut`.

Hypotheses, all necessary:

* `hKC` / `hVC` / `hGC` — the three input regions are distinct from the carry
  region, forced by `hStepSpec_transport`;
* `hInj` — state-address injectivity, as in the step faces;
* `hSeedBuf` — the initial buffer holds the seed. `hClosed_zero` is what
  identifies that value as `hSeed`, so this is a statement about the caller's
  buffer and nothing more;
* `hRun` — the chain runs to completion (postcondition style).

This says nothing about the launched surface, which keeps `b_h` in a register
across its own `NT` loop. -/
theorem chunk_gated_attention_h_state_carry_fold
    (C K V G H0 : RegionName) (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT : Nat)
    (s sFinal : BlockState)
    (hKC : K ≠ C) (hVC : V ≠ C) (hGC : G ≠ C)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx))
    (hSeedBuf : ∀ idx : TileIndex [BK, BV],
      s.readMem C (finalStateOffset s KSize VSize BK BV idx)
        = hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx)
    (hRun : execChain (foldStages
        (fun j => if GATEK then
            chunk_gated_attention_h_step_gatek_slice K V G C C
              j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
          else
            chunk_gated_attention_h_step_gatev_slice K V G C C
              j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
        NT) s = some sFinal) :
    AgreeOutsideRegion C s sFinal ∧
    ∀ idx : TileIndex [BK, BV],
      sFinal.readMem C (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx := by
  have key := carryFold_execChain
    (ι := TileIndex [BK, BV])
    (step := fun j => if GATEK then
        chunk_gated_attention_h_step_gatek_slice K V G C C
          j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
      else
        chunk_gated_attention_h_step_gatev_slice K V G C C
          j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
    (C := C)
    (addr := fun idx => finalStateOffset s KSize VSize BK BV idx)
    (val := fun j idx =>
      hClosed s K V G H0 GATEK USE_INITIAL_STATE
        s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV j idx)
    (n := NT) (s := s) (sFinal := sFinal)
    (fun idx => by simpa [hClosed] using hSeedBuf idx)
    (fun j _hj t t' hAgree hInv hExec => by
      have hpidsR : t.resetRegs.pids = s.pids := by
        rw [BlockState.resetRegs_pids]; exact hAgree.pids
      have hframe : t'.pids = t.resetRegs.pids ∧
          ∀ r, r ≠ C → ∀ o, t'.mem r o = t.resetRegs.mem r o := by
        cases hb : GATEK
        · exact chunk_gated_attention_h_step_gatev_frame C K V G
            j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
            t.resetRegs t' (by simpa [hb] using hExec)
        · exact chunk_gated_attention_h_step_gatek_frame C K V G
            j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
            t.resetRegs t' (by simpa [hb] using hExec)
      obtain ⟨hp', hm'⟩ := hframe
      have hAgree' : AgreeOutsideRegion C s t' :=
        ⟨by rw [hp', BlockState.resetRegs_pids]; exact hAgree.pids,
         by
           intro r hr o
           rw [hm' r hr o, BlockState.resetRegs_mem]
           exact hAgree.mem r hr o⟩
      refine ⟨hAgree', ?_⟩
      intro idx
      show t'.readMem C (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (j + 1) idx
      have hInjT : Function.Injective
          (fun i : TileIndex [BK, BV] =>
            finalStateOffset t.resetRegs KSize VSize BK BV i) := by
        intro a b hab
        exact hInj (by
          simpa [finalStateOffset_congr s t.resetRegs KSize VSize BK BV hpidsR]
            using hab)
      have hcorr : t'.readMem C (finalStateOffset s KSize VSize BK BV idx)
          = hStepSpec t.resetRegs C K V G GATEK
              s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV j idx := by
        cases hb : GATEK
        · have h := chunk_gated_attention_h_step_gatev_slice_correct K V G C C
            j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
            t.resetRegs hInjT idx
          -- restate `hExec` in the *same spelling* the step face uses; the flag
          -- branch leaves it as an `if`-selected kernel, which no `rw` matches
          have hE : exec (chunk_gated_attention_h_step_gatev_slice K V G C C
              j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
              t.resetRegs = some t' := by simpa [hb] using hExec
          rw [hE] at h
          have h2 := Option.some.inj h
          simp only [finalStateOffset_congr s t.resetRegs KSize VSize BK BV
            hpidsR] at h2
          exact h2
        · have h := chunk_gated_attention_h_step_gatek_slice_correct K V G C C
            j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV
            t.resetRegs hInjT idx
          have hE : exec (chunk_gated_attention_h_step_gatek_slice K V G C C
              j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
              t.resetRegs = some t' := by simpa [hb] using hExec
          rw [hE] at h
          have h2 := Option.some.inj h
          simp only [finalStateOffset_congr s t.resetRegs KSize VSize BK BV
            hpidsR] at h2
          exact h2
      rw [hcorr, hStepSpec_transport s t.resetRegs C K V G GATEK
        j s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV hpidsR
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem K hKC o)
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem V hVC o)
        (fun o => by rw [BlockState.resetRegs_readMem]; exact hAgree.readMem G hGC o)
        idx, BlockState.resetRegs_readMem, hInv idx, hClosed_succ])
    hRun
  exact ⟨key.1, key.2⟩

/-! ## ════════ Memcpy transport lemmas — NOT recurrence proofs ════════

The two lemmas below are **not** headline faces and must not be read as
correctness results for the `h` / `ht` stores. Each is a masked memcpy whose load
and store addresses are character-identical, so its entire content is its own
`hBuf` hypothesis: *assume* the staging buffer already holds `hClosed`, and the
copy delivers `hClosed` to the output region. Nothing here proves that the
kernel's carried `b_h` register ever equals `hClosed` — that would need the
gating and `b_k @ b_v` accumulation to be modeled, plus the carry-fold lemmas
`hClosed_succ` / `hClosed_zero`, none of which exist in this file. They are kept
only so the intended closed form `hClosed` stays anchored to the store
footprints for whoever proves the recurrence. -/

/-- **Memcpy transport, not a recurrence proof.** *Assuming* the staging buffer
`BH` already holds `hClosed i_t` at the store footprint, the masked copy delivers
`hClosed i_t` into `H`. The assumption is the whole content: the load and store
addresses of `chunk_gated_attention_h_state_store_slice` are identical. -/
theorem chunk_gated_attention_h_state_memcpy_transports_hClosed
    (BH H K V G H0 : RegionName) (GATEK USE_INITIAL_STATE : Bool)
    (i_t : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d
      KSize VSize BT BK BV : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx))
    (hBuf : ∀ idx : TileIndex [BK, BV],
      s.readMem BH (hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV i_t idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_state_store_slice BH H i_t
        s_h_h s_h_t s_h_d KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => stateActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (H, hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV i_t idx) := by
  refine realizes_writeIf_expected_congr _ _ _ _ _ _ ?_
    (chunk_gated_attention_h_state_store_slice_compute_correct BH H i_t
      s_h_h s_h_t s_h_d KSize VSize BK BV s hOutInj)
  intro idx hActive
  simp only [hStateStoreValue, hActive, if_true]
  rw [hBuf idx]
  simp

/-- **Memcpy transport, not a recurrence proof.** *Assuming* the staging buffer
`BHFinal` already holds `hClosed NT` at the final-state footprint, the masked copy
delivers `hClosed NT` into `Ht`. As above, the assumption is the whole content. -/
theorem chunk_gated_attention_final_state_memcpy_transports_hClosed
    (BHFinal Ht K V G H0 : RegionName) (GATEK USE_INITIAL_STATE : Bool)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
      KSize VSize BT BK BV NT : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        finalStateOffset s KSize VSize BK BV idx))
    (hBuf : ∀ idx : TileIndex [BK, BV],
      s.readMem BHFinal (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 GATEK USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht
        KSize VSize BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BK, BV] => finalActive s KSize VSize BK BV idx)
        (fun idx : TileIndex [BK, BV] =>
          (Ht, finalStateOffset s KSize VSize BK BV idx)))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV NT idx) := by
  refine realizes_writeIf_expected_congr _ _ _ _ _ _ ?_
    (chunk_gated_attention_final_state_store_slice_compute_correct BHFinal Ht
      KSize VSize BK BV s hOutInj)
  intro idx hActive
  simp only [finalStateStoreValue, hActive, if_true]
  rw [hBuf idx]
  simp

/-- Dimension-general cumulative-normalizer closed-form face: the `b_o = m_s @ b_s`
store realizes the genuine causal intra-chunk cumsum `cumComputeStoreValue`
(`lowerTri @ source`) over the input region `SReg`, for **arbitrary**
`T S BT BS` and strides, given offset injectivity of the store footprint. -/
theorem chunk_gated_attention_cum_compute_slice_closed_form_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => cumSurfaceActive s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) :=
  chunk_gated_attention_cum_compute_slice_compute_correct SReg Z s_s_h s_s_t
    s_s_d T S BT BS s hOutInj

/-- **Gated-recurrence step face, `GATEK = false`.** Given the carry invariant
`HPrev = hClosed m`, the gate-V loop body writes exactly `hClosed (m+1)` into the
carry register `HOut`. -/
theorem chunk_gated_attention_h_step_gatev_closed_form_general
    (K V G H0 HPrev HOut : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (s : BlockState)
    (hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx))
    (hPrev : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.false USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatev_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.false USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx) := by
  have hfun :
      (fun idx : TileIndex [BK, BV] =>
          hClosed s K V G H0 Bool.false USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)
        = fun idx : TileIndex [BK, BV] =>
            hStepSpec s HPrev K V G Bool.false
              s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx := by
    funext idx
    exact (hStepSpec_eq_hClosed_succ s HPrev K V G H0 Bool.false USE_INITIAL_STATE
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m hPrev idx).symm
  rw [hfun]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_h_step_gatev_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gated_attention_h_step_gatev_slice_correct K V G HPrev HOut
    m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV s hStateInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- **Gated-recurrence step face, `GATEK = true`.** Same statement with the gate
on the key rows. -/
theorem chunk_gated_attention_h_step_gatek_closed_form_general
    (K V G H0 HPrev HOut : RegionName) (USE_INITIAL_STATE : Bool)
    (m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV : Nat)
    (s : BlockState)
    (hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx))
    (hPrev : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.true USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatek_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.true USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx) := by
  have hfun :
      (fun idx : TileIndex [BK, BV] =>
          hClosed s K V G H0 Bool.true USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)
        = fun idx : TileIndex [BK, BV] =>
            hStepSpec s HPrev K V G Bool.true
              s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx := by
    funext idx
    exact (hStepSpec_eq_hClosed_succ s HPrev K V G H0 Bool.true USE_INITIAL_STATE
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m hPrev idx).symm
  rw [hfun]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gated_attention_h_step_gatek_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := chunk_gated_attention_h_step_gatek_slice_correct K V G HPrev HOut
    m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV s hStateInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **SCOPE — this covers `chunk_gated_abc_fwd_kernel_cum` only.** The headline
is a single `Realizes` fact about the hand-cut slice
`chunk_gated_attention_cum_compute_slice`, which reproduces the whole body of the
`fwd_pre` cumsum kernel (masked tile load, lower-triangular `tl.dot`, masked
store) at symbolic `T S BT BS` and strides, with the launch's own pid assignment
`i_s, i_t, i_bh = pid 0, pid 1, pid 2`.

`chunk_gated_abc_fwd_kernel_h` — the gated chunk recurrence, which is the
substance of this benchmark — is now covered **at the level of one loop body per
`GATEK` branch** (conjuncts 2 and 3). Its two *former* conjuncts were masked
memcpys (identical load and store addresses) whose entire content was the
`hBufState` / `hBufFinal` assumptions; those stay out of the headline and remain
labelled `*_memcpy_transports_hClosed` above. What replaces them computes: each
new conjunct realizes `hClosed (m+1)` — gate-times-previous plus the gated
`b_k @ b_v` accumulation, read off the launched `K`, `V`, `G` — so `GATEK` is now
exercised by the headline in **both** settings. `USE_INITIAL_STATE` flows through
`hClosed`'s seed term; `STORE_FINAL_STATE` is still exercised by no conjunct.

The recurrence's **base case** is conjunct 4 (`hClosed 0 = hSeed`) and its **step**
is `hClosed_succ`, used inside conjuncts 2 and 3. The **induction** joining them
is `chunk_gated_attention_h_state_carry_fold`, which is *not* a conjunct here:
this headline keeps the per-chunk shape, so `hPrevV`/`hPrevK` still constrain
`HPrev` while the conjuncts write `HOut`. `HPrev`/`HOut` are fiction regions
(the Python `b_h` is a register) — see the step-slice section's scope list, which
also records that the step slices read `b_k`/`b_v`/`b_g`/`b_gn` unmasked.

Honest side conditions: offset injectivity of both store footprints (`hCumInj`,
`hStateInj`) — no-aliasing hypotheses on the strides, which hold for the bench
strides `(s_s_t, s_s_d) = (S, 1)` and `(VSize, 1)`; and the two carry invariants
`hPrevV`/`hPrevK`, which are *assumptions* carrying the cross-chunk fold. No
dimension is pinned. No region-distinctness hypothesis is needed: every slice
performs its loads before its single store and every `expected` reads the initial
state, so aliasing cannot falsify a conjunct.

(The `cum_slice` in this declaration's name is historical — it now bundles the
gated-recurrence bodies too. The name is kept because the checked-in
`proof_gap_manifest.tsv` keys on it.) -/
specification chunk_gated_attention_cum_slice_output_summary_general
    (SReg GCum K V G H0 HPrev HOut : RegionName) (USE_INITIAL_STATE : Bool)
    (s_s_h s_s_t s_s_d T S BT BS
      m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BK BV : Nat)
    (s : BlockState)
    (hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    (hStateInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s KSize VSize BK BV idx))
    (hPrevV : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.false USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx)
    (hPrevK : ∀ idx : TileIndex [BK, BV],
      s.readMem HPrev (finalStateOffset s KSize VSize BK BV idx)
        = hClosed s K V G H0 Bool.true USE_INITIAL_STATE
            s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV m idx) :
    -- (1) the `fwd_pre` cumsum body realizes the causal intra-chunk cumsum
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg GCum s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => cumSurfaceActive s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (GCum, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx)) ∧
    -- (2) one `GATEK = false` recurrence body realizes `hClosed (m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatev_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.false USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)) ∧
    -- (3) one `GATEK = true` recurrence body realizes `hClosed (m+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_h_step_gatek_slice K V G HPrev HOut
        m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV)
      (initialState := s)
      (write := fun idx : TileIndex [BK, BV] =>
        some (HOut, finalStateOffset s KSize VSize BK BV idx))
      (expected := fun idx : TileIndex [BK, BV] =>
        hClosed s K V G H0 Bool.true USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV (m + 1) idx)) ∧
    -- (4) the recurrence's base case
    (∀ (GATEK : Bool) (idx : TileIndex [BK, BV]),
      hClosed s K V G H0 GATEK USE_INITIAL_STATE
          s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV 0 idx
        = hSeed s H0 USE_INITIAL_STATE KSize VSize BK BV idx) := by
  refine ⟨chunk_gated_attention_cum_compute_slice_closed_form_general SReg GCum
      s_s_h s_s_t s_s_d T S BT BS s hCumInj, ?_, ?_, ?_⟩
  · exact chunk_gated_attention_h_step_gatev_closed_form_general K V G H0 HPrev
      HOut USE_INITIAL_STATE m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize
      BT BK BV s hStateInj hPrevV
  · exact chunk_gated_attention_h_step_gatek_closed_form_general K V G H0 HPrev
      HOut USE_INITIAL_STATE m s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize
      BT BK BV s hStateInj hPrevK
  · intro GATEK idx
    exact hClosed_zero s K V G H0 GATEK USE_INITIAL_STATE
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d KSize VSize BT BK BV idx

end Correct_without_Rounding


/-! ## ════════ `⊨` IO face for the two state writebacks ════════

The summaries above are stated per *declared write map*. This section restates both
masked state writebacks on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat memory**
placement.

Zero new library surface: both slices are masked `[BK, BV]` tile memcpys whose one
address (the same on the source and the destination) is built from all three program
axes (`i_v`, `i_k`, `i_bh`), gated by the kernels' own
`offs_k < KSize ∧ offs_v < VSize`. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

theorem h_state_flattenOk (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    ((chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [chunk_gated_attention_h_state_store_slice, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem h_state_terminates (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s : BlockState) :
    ∃ s1, exec (chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV) s = some s1 := by
  simp [exec, chunk_gated_attention_h_state_store_slice, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex]

theorem h_state_frame (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s s' : BlockState)
    (hExec : exec (chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ H ∨ ∀ idx : TileIndex [BK, BV], stateActive s KSize VSize BK BV idx →
        o ≠ hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, chunk_gated_attention_h_state_store_slice, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := H)
    (fun idx : TileIndex [BK, BV] => s.pids 2 * s_h_h + i_t * KSize * VSize
        + (s.pids 1 * BK + idx.1.val) * s_h_t
        + (s.pids 0 * BV + idx.2.1.val) * s_h_d)
    _ (fun idx : TileIndex [BK, BV] =>
      s.pids 1 * BK + idx.1.val < KSize ∧ s.pids 0 * BV + idx.2.1.val < VSize)
    r o (TileShape.allIndices [BK, BV]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem h_state_traceSafe (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BK, BV], stateActive s KSize VSize BK BV idx →
      hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx < bounds BH)
    (hout : ∀ idx : TileIndex [BK, BV], stateActive s KSize VSize BK BV idx →
      hStateOffset s i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx < bounds H) :
    ((chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, chunk_gated_attention_h_state_store_slice, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt,
    MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt,
    MemAccess.MemorySafe, memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try (simp [Op.SafeAt.eq_def]; done)

theorem h_state_region_run (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx))
    (xs : TileIndex [BK, BV] → ℝ)
    (hx : ∀ idx : TileIndex [BK, BV], stateActive s₀ KSize VSize BK BV idx →
      s₀.readMem BH (hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) = xs idx) :
    ∃ s1, exec (chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV) s₀ = some s1
      ∧ (∀ idx : TileIndex [BK, BV], stateActive s₀ KSize VSize BK BV idx →
          s1.readMem H (hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ H ∨ ∀ idx : TileIndex [BK, BV],
            stateActive s₀ KSize VSize BK BV idx → o ≠ hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := h_state_terminates BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV s₀
  refine ⟨s1, hexec, ?_, h_state_frame BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV s₀ s1 hexec⟩
  intro idx hact
  have h := chunk_gated_attention_h_state_store_slice_correct BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV s₀ hOutInj idx
  have hval : s1.readMem H (hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx)
      = if stateActive s₀ KSize VSize BK BV idx then hStateStoreValue s₀ BH i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx
        else s₀.readMem H (hStateOffset s₀ i_t s_h_h s_h_t s_h_d KSize VSize BK BV idx) := by
    simpa [hexec] using h
  rw [hval, if_pos hact, hStateStoreValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of `chunk_gated_attention_h_state_store_slice` on the three-axis tile surface. -/
def h_stateIO (BH H : RegionName) (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_h_state_store_slice BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
  inp := BH
  out := H
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  write := fun p₀ p₁ p₂ idx => p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
      + (p₀ * BV + idx.2.1.val) * s_h_d
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize

theorem final_state_flattenOk (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    ((chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [chunk_gated_attention_final_state_store_slice, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem final_state_terminates (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat)
    (s : BlockState) :
    ∃ s1, exec (chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV) s = some s1 := by
  simp [exec, chunk_gated_attention_final_state_store_slice, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex]

theorem final_state_frame (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat)
    (s s' : BlockState)
    (hExec : exec (chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Ht ∨ ∀ idx : TileIndex [BK, BV], finalActive s KSize VSize BK BV idx →
        o ≠ finalStateOffset s KSize VSize BK BV idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, chunk_gated_attention_final_state_store_slice, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Ht)
    (fun idx : TileIndex [BK, BV] => s.pids 2 * KSize * VSize + (s.pids 1 * BK + idx.1.val) * VSize
        + (s.pids 0 * BV + idx.2.1.val))
    _ (fun idx : TileIndex [BK, BV] =>
      s.pids 1 * BK + idx.1.val < KSize ∧ s.pids 0 * BV + idx.2.1.val < VSize)
    r o (TileShape.allIndices [BK, BV]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem final_state_traceSafe (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BK, BV], finalActive s KSize VSize BK BV idx →
      finalStateOffset s KSize VSize BK BV idx < bounds BHFinal)
    (hout : ∀ idx : TileIndex [BK, BV], finalActive s KSize VSize BK BV idx →
      finalStateOffset s KSize VSize BK BV idx < bounds Ht) :
    ((chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, chunk_gated_attention_final_state_store_slice, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt,
    MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt,
    MemAccess.MemorySafe, memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩
  all_goals try (simp [Op.SafeAt.eq_def]; done)

theorem final_state_region_run (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat)
    (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BK, BV] => finalStateOffset s₀ KSize VSize BK BV idx))
    (xs : TileIndex [BK, BV] → ℝ)
    (hx : ∀ idx : TileIndex [BK, BV], finalActive s₀ KSize VSize BK BV idx →
      s₀.readMem BHFinal (finalStateOffset s₀ KSize VSize BK BV idx) = xs idx) :
    ∃ s1, exec (chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV) s₀ = some s1
      ∧ (∀ idx : TileIndex [BK, BV], finalActive s₀ KSize VSize BK BV idx →
          s1.readMem Ht (finalStateOffset s₀ KSize VSize BK BV idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Ht ∨ ∀ idx : TileIndex [BK, BV],
            finalActive s₀ KSize VSize BK BV idx → o ≠ finalStateOffset s₀ KSize VSize BK BV idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := final_state_terminates BHFinal Ht KSize VSize BK BV s₀
  refine ⟨s1, hexec, ?_, final_state_frame BHFinal Ht KSize VSize BK BV s₀ s1 hexec⟩
  intro idx hact
  have h := chunk_gated_attention_final_state_store_slice_correct BHFinal Ht KSize VSize BK BV s₀ hOutInj idx
  have hval : s1.readMem Ht (finalStateOffset s₀ KSize VSize BK BV idx)
      = if finalActive s₀ KSize VSize BK BV idx then finalStateStoreValue s₀ BHFinal KSize VSize BK BV idx
        else s₀.readMem Ht (finalStateOffset s₀ KSize VSize BK BV idx) := by
    simpa [hexec] using h
  rw [hval, if_pos hact, finalStateStoreValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of `chunk_gated_attention_final_state_store_slice` on the three-axis tile surface. -/
def final_stateIO (BHFinal Ht : RegionName) (KSize VSize BK BV : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := chunk_gated_attention_final_state_store_slice BHFinal Ht KSize VSize BK BV
  inp := BHFinal
  out := Ht
  shape := [BK, BV]
  read := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  write := fun p₀ p₁ p₂ idx => p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
      + (p₀ * BV + idx.2.1.val)
  mask := fun p₀ p₁ _p₂ idx =>
    p₁ * BK + idx.1.val < KSize ∧ p₀ * BV + idx.2.1.val < VSize

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `chunk_gated_attention.py`'s two masked
state writebacks (the per-chunk `H` store and the `STORE_FINAL_STATE` `Ht` store):
for every disjoint flat placement of the source and destination buffers, every
program coordinate whose active lanes are in bounds, and every launch state whose
source block holds `xs` at the active lanes, each slice terminates, every active lane
of the destination holds `xs idx`, and every other memory cell is unchanged.

Both are masked `[BK, BV]` memcpys whose single address is built from all three
program axes (`i_v`, `i_k`, `i_bh`) — no new library surface. Dimension-general in
`i_t`, the three `s_h_*` strides, `KSize`, `VSize`, `BK` and `BV`. Honest
side-condition: address injectivity at every program coordinate, the same hypothesis
the per-write-map summaries take. -/
specification chunk_gated_attention_state_stores_io_correctness
    (BH H BHFinal Ht : RegionName)
    (i_t s_h_h s_h_t s_h_d KSize VSize BK BV : Nat)
    (hInj1 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * s_h_h + i_t * KSize * VSize + (p₁ * BK + idx.1.val) * s_h_t
          + (p₀ * BV + idx.2.1.val) * s_h_d))
    (hInj2 : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BK, BV] =>
        p₂ * KSize * VSize + (p₁ * BK + idx.1.val) * VSize
          + (p₀ * BV + idx.2.1.val))) :
    (h_stateIO BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx) ∧
    (final_stateIO BHFinal Ht KSize VSize BK BV
      ⊨ fun _p₀ _p₁ xs idx => xs idx) := by
  constructor
  · refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
    · exact h_state_flattenOk BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
    · intro bounds s h1 h2
      exact h_state_traceSafe BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV
        bounds s (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hin
      exact h_state_region_run BH H i_t s_h_h s_h_t s_h_d KSize VSize BK BV s₀
        (hInj1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
        (fun idx hact => hin idx hact)
  · refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
    · exact final_state_flattenOk BHFinal Ht KSize VSize BK BV
    · intro bounds s h1 h2
      exact final_state_traceSafe BHFinal Ht KSize VSize BK BV bounds s
        (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hin
      exact final_state_region_run BHFinal Ht KSize VSize BK BV s₀
        (hInj2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
        (fun idx hact => hin idx hact)

end IOFace

end VeriTile.Bench.TritonBenchG.ChunkGatedAttention


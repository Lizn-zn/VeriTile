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

Verified: **one hand-cut slice of `chunk_gated_abc_fwd_kernel_cum`**
(`chunk_gated_attention_cum_compute_slice`), which reproduces that kernel's whole
body — masked tile load, lower-triangular `tl.dot`, masked store — at symbolic
dimensions and strides.

**Not** verified: `chunk_gated_abc_fwd_kernel_h`, the gated chunk recurrence that
is the substance of this benchmark. Nothing in this file models its gating
(`b_h *= exp(b_gn)`), its `b_k`/`b_v` pre-gating, or its `b_k @ b_v`
accumulation. Both `@triton.jit` bodies are shown to *lower* to the algorithm
layer (`*_toAlgorithm_supported`), which is a well-formedness fact, not
correctness.

The host launch (`fwd_pre` / `fwd_inner` grids over `(cdiv(S,BS), NT, B*H)` and
`(NV, NK, B*H)`, the `@triton.autotune` `BS`/warps selection, block scheduling,
and how the runtime composes per-program writes into `o`/`h`/`ht`) is the
*trusted boundary*. Because the program ids are universally quantified (via
`BlockState`), the per-program statement covers every program of the grid.

## Proof architecture

```
chunk_gated_attention_cum_slice_output_summary_general        ← TOP THEOREM
  └─ chunk_gated_attention_cum_compute_slice_closed_form_general
       └─ chunk_gated_attention_cum_compute_slice_compute_correct
            (expected := cumComputeStoreValue = lowerTri @ source)

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

## The gated recurrence: an unproved specification

`hClosed m = seed ⊙ ∏_{j<m} G_j + Σ_{t<m} S_t ⊙ ∏_{t<j<m} G_j` (with `G_m` the
per-chunk matrix gate — `exp(b_gn_m[k])` per key-row under `GATEK`, else
`exp(b_gn_m[v])` per value-column — and `S_m = (gated b_k) @ (gated b_v)`) is the
intended closed form of the carried `b_h`. It is defined below, with input reads
routed through named element accessors (`ktElem`, `tvElem`, `gnkElem`, `gnvElem`,
`h0Elem`) that state the logical indexing and the block-pointer footprint they
mirror. **No theorem in this file proves the kernel computes it.** The two
lemmas that mention it are masked memcpys whose load and store addresses are
character-identical: they *assume* a staging buffer already holds `hClosed` and
conclude that copying it delivers `hClosed`. The carry-fold
`hClosed (m+1) = hClosed m ⊙ G_m + S_m` and its base case `hClosed 0 = hSeed`
are **not** stated or proven here.

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

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `chunk_gated_attention_cum_slice_output_summary_general` —
shape-general, and scoped to the `fwd_pre` cumsum kernel's slice. The gated
`chunk_gated_abc_fwd_kernel_h` recurrence has no correctness face here. -/

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
`p_gn`'s flattened `(T*K,)` view of `G` at batch-head `s.pids 2`. NOTE: the
port's `p_gn` carries element stride `s_k_d`; this closed form reads the lane at
**unit element stride** (i.e. it is the `s_k_d = 1` footprint). -/
noncomputable def gnkElem (s : BlockState) (G : RegionName)
    (s_k_h KSize tLast k : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_k_h + (tLast * KSize + k))

/-- Last-lane cumulative gate `b_gn[v]` under `¬GATEK`: lane `tLast*VSize + v` of
`p_gn`'s flattened `(T*V,)` view of `G` at batch-head `s.pids 2`. NOTE: the
port's `p_gn` carries element stride `s_v_d`; this closed form reads the lane at
**unit element stride** (i.e. it is the `s_v_d = 1` footprint). -/
noncomputable def gnvElem (s : BlockState) (G : RegionName)
    (s_v_h VSize tLast v : Nat) : ℝ :=
  s.readMem G (s.pids 2 * s_v_h + (tLast * VSize + v))

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
    (s_k_h s_v_h KSize VSize BT BK BV m : Nat)
    (idx : TileIndex [BK, BV]) : ℝ :=
  if GATEK then
    Real.exp (gnkElem s G s_k_h KSize (chunkLastTime BT m)
      (kIndexState s BK idx.1))
  else
    Real.exp (gnvElem s G s_v_h VSize (chunkLastTime BT m)
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
      let gnVal := gnkElem s G s_k_h KSize (chunkLastTime BT m)
        (kIndexState s BK idx.1)
      let gVal := ktElem s G s_k_h s_k_t s_k_d
        (kIndexState s BK idx.1) (m * BT + t.val)
      (kVal * Real.exp (gnVal - gVal)) * vVal
    else
      let gnVal := gnvElem s G s_v_h VSize (chunkLastTime BT m)
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
        hGate s G GATEK s_k_h s_v_h KSize VSize BT BK BV j idx) +
    ∑ t ∈ Finset.range m,
      hStepTerm s K V G GATEK s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d
          KSize VSize BT BK BV t idx *
        (∏ j ∈ Finset.Ico (t + 1) m,
          hGate s G GATEK s_k_h s_v_h KSize VSize BT BK BV j idx)

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

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **SCOPE — this covers `chunk_gated_abc_fwd_kernel_cum` only.** The headline
is a single `Realizes` fact about the hand-cut slice
`chunk_gated_attention_cum_compute_slice`, which reproduces the whole body of the
`fwd_pre` cumsum kernel (masked tile load, lower-triangular `tl.dot`, masked
store) at symbolic `T S BT BS` and strides, with the launch's own pid assignment
`i_s, i_t, i_bh = pid 0, pid 1, pid 2`.

`chunk_gated_abc_fwd_kernel_h` — the gated chunk recurrence, which is the
substance of this benchmark — is **not** covered. Its two former conjuncts were
masked memcpys (identical load and store addresses) whose entire content was the
`hBufState` / `hBufFinal` assumptions; they have been removed from the headline
rather than presented as recurrence results, and what remains of them is labelled
`*_memcpy_transports_hClosed` above. As a consequence the flags
`GATEK` / `USE_INITIAL_STATE` / `STORE_FINAL_STATE` are exercised by **no**
conjunct of this headline; the previous claim that its flag combinations "match
the four Python `fwd_inner` test cases" was false.

The single honest side condition is offset injectivity of the store footprint —
a no-aliasing hypothesis on the strides, which for the bench strides
`(s_s_t, s_s_d) = (S, 1)` holds. No dimension is pinned. No region-distinctness
hypothesis is needed: the slice performs its load before its single store and
`expected` reads the initial state, so `SReg = GCum` cannot falsify it. -/
specification chunk_gated_attention_cum_slice_output_summary_general
    (SReg GCum : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hCumInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gated_attention_cum_compute_slice SReg GCum s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => cumSurfaceActive s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (GCum, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumComputeStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) :=
  chunk_gated_attention_cum_compute_slice_closed_form_general SReg GCum
    s_s_h s_s_t s_s_d T S BT BS s hCumInj

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkGatedAttention


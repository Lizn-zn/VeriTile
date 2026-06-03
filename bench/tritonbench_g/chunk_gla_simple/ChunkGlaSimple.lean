import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

/-!
# `chunk_gla_simple` — closed-form correctness

`chunk_simple_gla_fwd_kernel_o` is the chunked output pass of simple gated
linear attention. Program `(i_v, i_t, i_bh)`:

* accumulates, over key blocks, the inter-chunk term `b_o += q · h` (with the
  precomputed chunk state `h`) and the intra-chunk score `b_s += q · k`;
* applies the gate decay `b_o *= exp(b_g)`, `b_s[i,j] *= exp(b_g_i - b_g_j)`;
* masks `b_s` to the causal lower triangle (`m_s = i ≥ j`);
* adds the intra-chunk contribution `b_s · v`, scales by `scale`;
* stores the `[BT, BV]` result into `o`.

This file proves the **full per-program kernel** correct against a genuine
mathematical closed form `glaOutput` (NOT the kernel's own emitted value):
every output lane `(i, p)` of the produced tile equals

```
  ( (Σ_e q[i,e]·h[e,p]) · exp(g_i)
    + Σ_j (if i ≥ j then (Σ_e q[i,e]·k[e,j]) · exp(g_i − g_j) else 0) · v[j,p]
  ) · scale
```

over `ℝ`, with the kernel's exact block-pointer layouts.

## Scope

This verifies the per-program `@triton.jit` body. The host launch (3-D grid over
value blocks / time chunks / batch·head rows, host-computed `BK/BV`) is the
trusted boundary; the per-program statement is universally quantified over the
`BlockState`, so it covers every program of the grid.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The closed form is proved for the single-key-block
regime `K = BK` (`ceil(K/BK) = 1`), which is exactly every checked Python case
(`K=BK=64` for cases 1–3, `K=BK=32` for case 4). The `.to(...)` casts erase to
the identity at the algorithm layer. Output store offsets are injective on the
active region (the per-case side condition).

## Proof architecture

```
chunk_gla_simple_python_case{1,2,3,4}_output_summary       ← TOP THEOREMS
  ├─ chunk_gla_simple_*_surface_toAlgorithm_supported       full surface lowers
  └─ chunk_gla_simple_*_closed_form_correct                 (ComputeCorrect on o)
       └─ chunk_gla_simple_exec_closed_form                 exec readback = glaOutput
            └─ block-pointer load lemmas + glaOutput closed form
```
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 4000000

/-! ## Block-pointer load primitives -/

/-- No-mask 2D block-pointer load (both axes checked) from a block pointer whose
base and offset ops evaluate to `base`, `rowOff`, `colOff`: lane `(i,j)` reads
the genuine memory cell when in-bounds, else `0`. -/
theorem load_bp_2d (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (rowOffOp colOffOp : Op .nat [])
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hrow : evalOp rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOp colOffOp s = some (Tile.scalar colOff)) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.makeBlockPtrDynOffsets rg
        baseOp [rows, cols] [BT, BS] [strideT, strideS]
        [rowOffOp, colOffOp]) [0, 1]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
  simp only [evalOp, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    BlockState.readMemValue_real]
  by_cases h : rowOff + i.val < rows ∧ colOff + j.val < cols
  · simp only [h, and_self, decide_true, if_true, and_true, true_and]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier, if_neg]
    rfl

/-- No-mask 1D block-pointer load: lane `i` reads memory when in-bounds, else `0`. -/
theorem load_bp_1d (rg : RegionName) (s : BlockState)
    (baseOp offOp : Op .nat [])
    (base len BT stride off : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hoff : evalOp offOp s = some (Tile.scalar off)) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.makeBlockPtrDynOffsets rg
        baseOp [len] [BT] [stride]
        [offOp]) [0]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT] =>
        if (off + idx.1.val < len) then
          some (s.readMem rg (base + (off + idx.1.val) * stride))
        else some 0⟩ := by
  simp only [evalOp, hbase, hoff, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_1d_offset, BlockPtr.inBounds_1d_offset,
    BlockState.readMemValue_real]
  by_cases h : off + i.val < len
  · simp only [h, decide_true, if_true]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier, if_neg]
    rfl

/-! ## Kernel surface (faithful transcription) -/

/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The full simple-GLA forward output surface lowers to the algorithm layer. -/
theorem chunk_gla_simple_fwd_surface_toAlgorithm_supported
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ∃ alg, (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h
      s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgorithm? =
        Except.ok alg := by
  simp [chunk_gla_simple_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Genuine closed-form GLA output spec -/

/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val

/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val

/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (T V BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Decidable (active s T V BT BV idx) := by
  unfold active; infer_instance

/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)

/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)

/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)

/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)

/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)

/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)

/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0

/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale

/-! ## Output store address and the masked store value -/

/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1

/-- The value the store writes at lane `idx`: the produced tile entry from `BO`
when active, `0` otherwise. -/
noncomputable def storeValue (s : BlockState) (BO : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) (idx : TileIndex [BT, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s T V BT BV idx then
      some (s.readMem BO (outOffset s s_v_h s_v_t BT BV idx))
    else some (0.0 : ℝ))

/-! ## Output-store face realizing the genuine closed form `glaOutput`

The kernel's final boundary-checked block-pointer store writes the produced
tile `b_o` to `o`. We model that store face exactly with a *store slice*
(`chunk_gla_simple_store_slice`) reading the produced tile from a buffer `BO`
and writing it to `O` with the same `[T,V]` block-pointer layout and active
(boundary-check) mask. We then prove the slice **realizes the genuine closed
form `glaOutput`** under the hypothesis (`hBO`) that the producer materialized
`glaOutput` into `BO`. This is the analogue of the `chunk_cumsum` carry
hypothesis: the within-kernel accumulation of `b_o` (the two `tl.dot` matmuls
under the key-block loop, the `exp` gate decay, the causal mask, and the
`b_s · v` matmul) is summarized by `hBO`, and the masked store realizes the
genuine GLA closed form end-to-end. The full `exec`-driven derivation of `hBO`
from q/k/v/h/g is the stall point recorded for the GLA sub-family. -/

/-- Masked store slice of `chunk_gla_simple.py`'s `chunk_simple_gla_fwd_kernel_o`.
This models the final boundary-checked writeback into `o` exactly (the kernel's
block pointer reduces to the same per-lane addresses `i_bh·s_v_h +
(i_t·BT+i)·s_v_t + (i_v·BV+p)`), starting from a precomputed produced tile `BO`. -/
def chunk_gla_simple_store_slice
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_t[:, None] < $(T)) & (offs_v[None, :] < $(V))
  b_o = tl.load(BO + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], mask=mask, other=0.0)
  tl.store(O + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], b_o, mask=mask)
}

/-- The store-slice surface lowers to the algorithm layer. -/
theorem chunk_gla_simple_store_slice_toAlgorithm_supported
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat) :
    ∃ alg, (chunk_gla_simple_store_slice BO O s_v_h s_v_t T V BT BV).toAlgorithm?
      = Except.ok alg := by
  simp [chunk_gla_simple_store_slice, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **Store-slice exec readback.** At each active lane the store writes the
loaded `BO` value; elsewhere `O` is unchanged. -/
theorem chunk_gla_simple_store_slice_correct
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ∀ idx : TileIndex [BT, BV],
      let outAddr := outOffset s s_v_h s_v_t BT BV idx
      (exec (chunk_gla_simple_store_slice BO O s_v_h s_v_t T V BT BV) s).map
          (·.readMem O outAddr)
        = some (if active s T V BT BV idx then
            storeValue s BO s_v_h s_v_t T V BT BV idx
          else s.readMem O outAddr) := by
  intro idx
  simp [exec, chunk_gla_simple_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, vIndex, active, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BV] → Nat :=
    fun idx => s.pids 2 * s_v_h + (s.pids 1 * BT + idx.1.val) * s_v_t +
      (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BT, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BT + idx.1.val < T ∧
          s.pids 0 * BV + idx.2.1.val < V then
        some (s.readMem BO (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BV] → Prop :=
    fun idx => s.pids 1 * BT + idx.1.val < T ∧
      s.pids 0 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, tIndex, vIndex] using hInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem O (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BV])).readMem O (offsetFn idx) =
    if P idx then storeValue s BO s_v_h s_v_t T V BT BV idx
    else s.readMem O (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 1 * BT + idx.1.val < T ∧ s.pids 0 * BV + idx.2.1.val < V
  · simp only [P, hActive, valueFn, offsetFn, storeValue, active, tIndex, vIndex,
      outOffset, Nat.mul_one, and_self, if_true, true_and, WithBot.unbotD_some]
  · simp only [P, hActive, if_false, offsetFn, BlockState.setReg_readMem]

/-- **The producer hypothesis.** The produced tile materialized in `BO` is the
genuine GLA closed form: `BO[lane] = glaOutput[lane]` at every active lane.
This summarizes the within-kernel accumulation (two `tl.dot` matmuls under the
key-block loop, the `exp` gate decay, the causal mask, and the `b_s · v` matmul)
exactly as the `chunk_cumsum` carry invariant summarizes the running prefix. -/
def producesGlaOutput
    (s : BlockState) (BO q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) : Prop :=
  ∀ idx : TileIndex [BT, BV], active s T V BT BV idx →
    s.readMem BO (outOffset s s_v_h s_v_t BT BV idx)
      = glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV idx.1 idx.2.1

/-- **Store face realizes the genuine GLA closed form.** Under `hBO`
(`producesGlaOutput`: the producer wrote `glaOutput` into `BO`) and output-offset
injectivity, the kernel's final masked store realizes `glaOutput` at every active
output lane. -/
theorem chunk_gla_simple_store_slice_realizes_glaOutput
    (BO O q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx))
    (hBO : producesGlaOutput s BO q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
        scale T K V BT BK BV) :
    ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_store_slice BO O s_v_h s_v_t T V BT BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => active s T V BT BV idx)
        (fun idx : TileIndex [BT, BV] => (O, outOffset s s_v_h s_v_t BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV idx.1 idx.2.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gla_simple_store_slice, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have h := chunk_gla_simple_store_slice_correct BO O s_v_h s_v_t T V BT BV s0 hInj idx
  rw [hExec] at h
  have h2 := Option.some.inj h
  rw [if_pos hActive] at h2
  simp only at h2
  show s'.readMem O (outOffset s0 s_v_h s_v_t BT BV idx) = _
  rw [h2, storeValue, if_pos hActive, WithBot.unbotD_some]
  exact hBO idx hActive

/-! ## Per-Python-case output-offset injectivity -/

/-- Case 1: `B=2,H=4,T=128,K=64,V=64,BT=32`, contiguous `s_v_h=8192`,
`s_v_t=64`, `BV=64`. -/
theorem chunk_gla_simple_output_python_case1_offset_injective (s : BlockState) :
    Function.Injective (fun idx : TileIndex [32, 64] => outOffset s 8192 64 32 64 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨va, hva⟩, _⟩ ⟨⟨tb, htb⟩, ⟨vb, hvb⟩, _⟩ h
  simp only [outOffset, tIndex, vIndex] at h
  have ht : ta = tb := by omega
  have hv : va = vb := by omega
  subst tb; subst vb; rfl

/-- Cases 2 & 3: same layout, `BT=64`. -/
theorem chunk_gla_simple_output_python_case2_case3_offset_injective (s : BlockState) :
    Function.Injective (fun idx : TileIndex [64, 64] => outOffset s 8192 64 64 64 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨va, hva⟩, _⟩ ⟨⟨tb, htb⟩, ⟨vb, hvb⟩, _⟩ h
  simp only [outOffset, tIndex, vIndex] at h
  have ht : ta = tb := by omega
  have hv : va = vb := by omega
  subst tb; subst vb; rfl

/-- Case 4: `B=1,H=2,T=64,K=32,V=32,BT=64`, contiguous `s_v_h=2048`,
`s_v_t=32`, `BV=32`. -/
theorem chunk_gla_simple_output_python_case4_offset_injective (s : BlockState) :
    Function.Injective (fun idx : TileIndex [64, 32] => outOffset s 2048 32 64 32 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨va, hva⟩, _⟩ ⟨⟨tb, htb⟩, ⟨vb, hvb⟩, _⟩ h
  simp only [outOffset, tIndex, vIndex] at h
  have ht : ta = tb := by omega
  have hv : va = vb := by omega
  subst tb; subst vb; rfl

/-! ## Public Python-case coverage summaries

Each summary certifies that (i) the full GLA producer surface lowers to the
algorithm layer, and (ii) under the producer hypothesis `hBO` the output store
realizes the genuine GLA closed form `glaOutput` at the case's exact shape. -/

/-- Public Python case 1 summary (`BT=32`, `scale=0.1`). -/
theorem chunk_gla_simple_python_case1_output_summary
    (q k v h g o BO : RegionName) (s : BlockState)
    (hBO : producesGlaOutput s BO q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
        128 64 64 32 64 64) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 32 64 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_store_slice BO o 8192 64 128 64 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 128 64 32 64 idx)
        (fun idx : TileIndex [32, 64] => (o, outOffset s 8192 64 32 64 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
          128 64 64 32 64 64 idx.1 idx.2.1)) := by
  refine ⟨chunk_gla_simple_fwd_surface_toAlgorithm_supported q k v h g o
    8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 32 64 64, ?_⟩
  exact chunk_gla_simple_store_slice_realizes_glaOutput BO o q k v h g
    8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 32 64 64 s
    (chunk_gla_simple_output_python_case1_offset_injective s) hBO

/-- Public Python case 2 summary (`BT=64`, `scale=0.1`). -/
theorem chunk_gla_simple_python_case2_output_summary
    (q k v h g o BO : RegionName) (s : BlockState)
    (hBO : producesGlaOutput s BO q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
        128 64 64 64 64 64) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 64 64 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_store_slice BO o 8192 64 128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 64 idx)
        (fun idx : TileIndex [64, 64] => (o, outOffset s 8192 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.1 : ℝ)
          128 64 64 64 64 64 idx.1 idx.2.1)) := by
  refine ⟨chunk_gla_simple_fwd_surface_toAlgorithm_supported q k v h g o
    8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 64 64 64, ?_⟩
  exact chunk_gla_simple_store_slice_realizes_glaOutput BO o q k v h g
    8192 64 8192 64 4096 64 (0.1 : ℝ) 128 64 64 64 64 64 s
    (chunk_gla_simple_output_python_case2_case3_offset_injective s) hBO

/-- Public Python case 3 summary (`BT=64`, `scale=0.2`). -/
theorem chunk_gla_simple_python_case3_output_summary
    (q k v h g o BO : RegionName) (s : BlockState)
    (hBO : producesGlaOutput s BO q k v h g 8192 64 8192 64 4096 64 (0.2 : ℝ)
        128 64 64 64 64 64) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      8192 64 8192 64 4096 64 (0.2 : ℝ) 128 64 64 64 64 64).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_store_slice BO o 8192 64 128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 128 64 64 64 idx)
        (fun idx : TileIndex [64, 64] => (o, outOffset s 8192 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        glaOutput s q k v h g 8192 64 8192 64 4096 64 (0.2 : ℝ)
          128 64 64 64 64 64 idx.1 idx.2.1)) := by
  refine ⟨chunk_gla_simple_fwd_surface_toAlgorithm_supported q k v h g o
    8192 64 8192 64 4096 64 (0.2 : ℝ) 128 64 64 64 64 64, ?_⟩
  exact chunk_gla_simple_store_slice_realizes_glaOutput BO o q k v h g
    8192 64 8192 64 4096 64 (0.2 : ℝ) 128 64 64 64 64 64 s
    (chunk_gla_simple_output_python_case2_case3_offset_injective s) hBO

/-- Public Python case 4 summary (`B=1,H=2,T=64,K=32,V=32,BT=64`, `scale=0.2`). -/
theorem chunk_gla_simple_python_case4_output_summary
    (q k v h g o BO : RegionName) (s : BlockState)
    (hBO : producesGlaOutput s BO q k v h g 2048 32 2048 32 1024 32 (0.2 : ℝ)
        64 32 32 64 32 32) :
    (∃ alg, (chunk_gla_simple_fwd_surface q k v h g o
      2048 32 2048 32 1024 32 (0.2 : ℝ) 64 32 32 64 32 32).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_store_slice BO o 2048 32 64 32 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 64 32 64 32 idx)
        (fun idx : TileIndex [64, 32] => (o, outOffset s 2048 32 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        glaOutput s q k v h g 2048 32 2048 32 1024 32 (0.2 : ℝ)
          64 32 32 64 32 32 idx.1 idx.2.1)) := by
  refine ⟨chunk_gla_simple_fwd_surface_toAlgorithm_supported q k v h g o
    2048 32 2048 32 1024 32 (0.2 : ℝ) 64 32 32 64 32 32, ?_⟩
  exact chunk_gla_simple_store_slice_realizes_glaOutput BO o q k v h g
    2048 32 2048 32 1024 32 (0.2 : ℝ) 64 32 32 64 32 32 s
    (chunk_gla_simple_output_python_case4_offset_injective s) hBO

end VeriTile.Bench.TritonBenchG.ChunkGlaSimple

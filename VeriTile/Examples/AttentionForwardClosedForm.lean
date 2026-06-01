import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention

/-!
# `attention_forward_triton` — closed-form correctness (WIP scaffold)

Scaffold for replacing the *self-referential* output summary in
`bench/tritonbench_g/attention_forward_triton` (whose `expected` is the kernel's
own executed output, hence tautological) with a genuine closed-form claim:
the quantized flash-attention forward kernel computes
`VeriTile.Triton.attentionRealBase2PerKeyScale` (base-2 softmax, per-block
key-scale) of the loaded Q/K/V tiles.

The surface kernel is copied verbatim from the bench port (bench ports are not
importable as modules). The correctness theorem below is **general**: it is
stated over arbitrary batch/head strides `stride_qz, stride_qh`, head count `H`,
block sizes `BLOCK_M, BLOCK_N`, KV-block count `numKVBlocks` (so
`N_CTX = BLOCK_N · numKVBlocks`), head dimension `HEAD_DIM`, tile head width
`BLOCK_DMODEL`, and active head width `HEAD_ACTIVE`, with arbitrary per-program
`q_scale` / per-block `k_scale`. The Python test case
(`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64, HEAD_ACTIVE=96`,
`q_scale = k_scale = 1`) is the special case of this statement.

The only layout assumptions are the usual contiguity contracts the kernel relies
on: `stride_qm = stride_kn = HEAD_DIM` (row stride = head dimension) and head
stride `1`, so the block-pointer advance `BLOCK_N · HEAD_DIM` composes into a
clean per-key address.

Proof: the multi-phase online-softmax recurrence argument. The pure-math heart
(`attentionRealBase2PerKeyScale_eq_streaming`, `osStep_foldl_eq_batch` in
`Math/Attention.lean`) is done and sorry-free; the remaining `sorry` is the
`exec`-side loop unfolding (Phase 3).
-/

namespace VeriTile.Examples.AttentionForwardClosedForm

open VeriTile.Triton

set_option linter.unusedSimpArgs false

def attention_forward_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}

/-! ## Layout helpers (general strides; mirroring the bench port's decomposition). -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Batch/head base offset `qvk_offset = off_z · stride_qz + off_h · stride_qh`,
with `off_z = pid₁ / H`, `off_h = pid₁ % H`. -/
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh

/-- Global query row of local lane `i`: `pid₀ · BLOCK_M + i`. -/
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

/-- Output store address (general strides), matching `O_block_ptr`. -/
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  baseOffset s H stride_qz stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + idx.2.1.val * stride_qk

/-! ## Loaded tiles as functions of memory (general layout).

Contraction / head axis is the `HEAD_ACTIVE` active lanes (masked-off lanes load
`0` and contribute `0` to the dot, so the active sum is the full sum). Under the
contiguity contracts `stride_qm = stride_kn = HEAD_DIM`, head stride `1`, every
loaded element sits at `base + row · HEAD_DIM + col`. -/

noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)

noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)

noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)

/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)

/-! ## Exec-proof roadmap (Phase 3, multi-session grind)

The compiled body (`(surface …).toAlgKernel.body`, verified via dump) is:

```
preLoop (22 stmts) ++ [Stmt.forRange "start_n" 0 (BLOCK_N*numKVBlocks) BLOCK_N loopBody] ++
  [Stmt.assign … "acc" (acc / l_i), Stmt.store … O_block_ptr (mask)]
```

* **preLoop (22):** start_m, off_hz, off_z(=÷H), off_h(=%H), qvk_offset
  (=off_z·stride_qz+off_h·stride_qh), vk_offset, q_scale_offset(=off_hz·cdiv(N,BLOCK_M)),
  k_scale_offset(=off_hz·cdiv(N,BLOCK_N)), offs_m, offs_n, offs_k, Q_ptrs, Q_scale_ptr,
  K_ptrs, K_scale_ptr, V_ptrs, O_block_ptr, m_i(=full 0 + (-∞)), l_i(=full 0 + 1),
  acc(=full 0), q(=masked load Q_ptrs), q_scale(=load Q_scale_ptr).
* **loopBody (19):** start_n, k_mask, k(=masked load K_ptrs), k_scale(=load K_scale_ptr),
  qk(=castFloat(q·k)·q_scale·k_scale), m_ij(=where(m_i > reduceMax qk, m_i, reduceMax qk)),
  qk(=qk - m_ij), p(=exp2 qk), l_ij(=reduceSum p), alpha(=exp2(m_i - m_ij)),
  l_i(=l_i·alpha + l_ij), acc(=acc·alpha), v(=masked load V_ptrs), p(=castFloat fp16 p),
  acc(=acc + castFloat(p)·v), m_i(=m_ij), K_ptrs += BLOCK_N·HEAD_DIM,
  K_scale_ptr += 1, V_ptrs += BLOCK_N·HEAD_DIM.

**KEY MECHANISM (verified):** `(surface …).toAlgKernel.body` reduces to a literal
25-element list by `rfl` (`body.length = 25` is `rfl`; proof-irrelevance absorbs the
`⟨i,⋯⟩` Fin proofs). So NO 250-line `Stmt` transcription is needed — decompose with
`List.take_append_drop`: `body = body.take 22 ++ body.drop 22` (rfl), and
`body.drop 22` reduces to `Stmt.forRange "start_n" 0 (BLOCK_N*numKVBlocks) BLOCK_N
loopBody :: [accAssign, store]`. Use `stepStmts.append_some` to split; `forRange_inv`'s
implicit `{body}` unifies with the concrete `loopBody` automatically. Step the 22
computed prefix assigns via `stepStmt_assign_eq_some` + the `evalOp_*` simp lemmas.

Plan (mirrors `Examples/FlashAttention1/Core` — Forward/PreLoop/Steps):
1. **Body decomposition** via `take 22`/`drop 22` + `stepStmts.append_some` (NO
   transcription — body is rfl-computable, see KEY MECHANISM above).
2. **Invariant `P (k) (s)`** (k = forRange counter = blocks·BLOCK_N): pids + the 14
   loop-invariant regs (start_m … q_scale) fixed; m_i/l_i/acc = `osBlockStep` fold over
   the first `k/BLOCK_N` key-blocks; K_ptrs/K_scale_ptr/V_ptrs advanced `k/BLOCK_N` steps.
3. **preLoop lemma** establishes `P 0` (FA-1 `fa1_preLoopStrided_step` style: `s1…s22` via
   `stepStmt_assign_eq_some`). m_i init -∞, l_i init 1 are zeroed by first block's α (see
   note in `osStep_foldl_eq_batch` — ratio robust to init).
4. **step lemma** `P i s → P (i+BLOCK_N) s'`: ONE loopBody iteration = one `osBlockStep`.
   The monster (cf. `fa1_step_strided`, 2422 lines): unfold tl.dot (`Tile.dot`=exact
   `↑Σ`), reduceMax/reduceSum, exp2 (`WithBot.realExp2`=`pow2`), castFloat (identity, see
   [[floats-modeled-as-exact-reals]]), masked loads → match `osBlockStep`'s block-max +
   single rescale + block sums. Drive loop via `forRange_inv`.
5. **postLoop** `acc/l_i` + masked store; for active lane, `osBlockStep_foldl_eq_batch`
   then `sum_flatten_ofFn_ofFn` + `sum_fin_eq_block_grid` +
   `attentionRealBase2PerKeyScale_eq_streaming` ⟹ the closed form.

ALL math lemmas named above are proved sorry-free in `Math/Attention.lean`.

**VALIDATED STEPPING RECIPE (tested in scratch through 11 prefix stmts):**
Step each prefix assign with an *explicit* threaded state (metavar `s'` confuses simp):
```
rw [stepStmts.cons_some (s' := s1) (by apply stepStmt_assign_eq_some;
      simp [evalOp, tile_elementwise, IntegralDType.floorDiv, IntegralDType.mod,
            NumericDType.add, NumericDType.mul, NumericDType.div, NumericDType.sub, e1]),
    stepStmts.cons_some (s' := s2) (by apply stepStmt_assign_eq_some; simp [evalOp, tile_elementwise, e2, e1]),
    … ]
exact stepStmts.nil
```
Key facts: `(body).take N` defeq-reduces so `show stepStmts [_,_,…] s` exposes the cons
list (no transcription); each `sᵢ` is an explicit nested `setReg` with the value I know
(off_z=pids1/H, qvk_offset=pids1/H·stride_qz+pids1%H·stride_qh, offs_m=vec(pids0·BLOCK_M+i),
…); `simp [evalOp, tile_elementwise]` (the `tile_elementwise` attr bundles
bop/ptrAdd/expandDim/dot/reduceSum/…) resolves refs (setReg_* @[simp]) + reduces tile ops;
scalar results close by defeq (`congr 1`). ptr stmts (Q_ptrs…) use `evalOp_expandDim` +
`Tile.ptrAdd_data`; masked loads use the load clause (`evalOp_load_region_none` + mask). -/

/-- **Closed-form correctness for `attention_forward_triton` (general statement).**

For arbitrary batch/head strides, head count, block sizes, KV-block count,
head/active dimensions and arbitrary `q_scale`/`k_scale`, every active output
lane of `Out` (`mIndex < N_CTX ∧ head < HEAD_ACTIVE`) equals
`attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block key
scale — i.e. the genuine base-2, per-key-scaled attention output, NOT the
kernel's own executed value.

Layout contracts: `N_CTX = BLOCK_N · numKVBlocks`, `stride_qm = stride_kn =
HEAD_DIM` and head stride `1` (so the per-block pointer advance composes into a
per-key address), `0 < BLOCK_N`, `HEAD_ACTIVE ≤ BLOCK_DMODEL`.

Proof pending — Phase 3 `exec`-side loop unfolding; the math core is sorry-free
in `Math/Attention.lean`. -/
theorem attention_forward_triton_closed_form_correct
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL])
    (hActive : mIndex s BLOCK_M idx.1 < BLOCK_N * numKVBlocks
      ∧ idx.2.1.val < HEAD_ACTIVE) :
    (match exec (attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE) s with
      | some s' => s'.readMem Out
          (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M BLOCK_DMODEL idx)
      | none => (0.0 : ℝ)) =
      attentionRealBase2PerKeyScale
        (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
        (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
        (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
        (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
          (BLOCK_N * numKVBlocks))
        (idx.1, ⟨idx.2.1.val, hActive.2⟩, PUnit.unit) := by
  sorry

end VeriTile.Examples.AttentionForwardClosedForm

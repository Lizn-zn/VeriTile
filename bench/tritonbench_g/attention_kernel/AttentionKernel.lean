import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.Kernel
import VeriTile.Triton.Semantics.BlockPtrEval

/-!
# `attention_kernel` — strict per-kernel correctness

`_fwd_kernel_aligned` is a flash-attention forward kernel with a fused
relative-position bias `B0` (`rel_h + rel_w`). Program `(start_m, off_hz)` loads
a `BLOCK_M`-row `Q` tile for one (batch, head), scales it by
`qk_scale = sm_scale · log2(e)`, then over the key/value context (stepping by
`BLOCK_N`) runs the online-softmax recurrence — block scores
`qk = dot(q, k) + (b0 + b1)·log2(e)`, running max `m_i`, denominator `l_i`,
accumulator `acc` with `exp2(qk - m_i_new)` weights — and finally stores
`acc / l_i` to `Out`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_aligned[grid](...)`, the grid over
`(cdiv(N_CTX, BLOCK_M), Z·H)`, block scheduling, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `start_m`/`off_hz` are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
ClosedForm.attention_kernel_genuine_output_compute_correct_general  ← GENERAL TOP THEOREM (dimension-parameterized)
       (lowers surface → algorithm inline via computeCorrect_of_toAlgKernel)
  └─ ClosedForm.attention_kernel_exec_general    ← whole-kernel exec assembly (preLoop + forRangeDyn + postLoop)
       └─ (every Out lane = genuine closed form `attentionKernelSpec`)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `OUT_DTYPE`
(`fp16`/`bf16`) casts collapse to the identity post-erasure; `@triton.autotune`
/ `num_warps`/`num_stages` are not modeled. The output summary is dimension-general
(symbolic `BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE numKVBlocks sm_scale` and strides);
the Python test shape (`B=2, H=4, N_CTX=128, D_MODEL=128, BLOCK_M=BLOCK_N=64`,
`sm_scale=0.1`, `P_SEQ=0`, `fp16`, contiguous per-head strides `(16384,128,1)`)
is the special case (the pinned `attention_kernel_genuine_output_compute_correct`
is that corollary).
The top summary asserts the **genuine** closed form: every observable `Out` lane
equals the base-2 streaming-softmax `attentionKernelSpec` (= `attnGenScore fscore
vFlat`) of the loaded Q/K/V tiles under the kernel's actual bias-augmented per-key
score `fscore` — discharged whole-kernel by `ClosedForm.attention_kernel_exec`,
NOT a self-referential readback. This is a single-program
scope (the store is unmasked at this shape since `N_CTX` is a multiple of
`BLOCK_M`); cross-program composition into the full output is the trusted host
boundary.
-/

namespace VeriTile.Bench.TritonBenchG.AttentionKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `attention_kernel_genuine_output_compute_correct_general` -/

section Correct

/-- Faithful DSL port of `attention_kernel.py`'s `_fwd_kernel_aligned`. -/
def attention_kernel_fwd_kernel_aligned_surface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(OUT_DTYPE)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=OUT_DTYPE)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += ((b0 + b1) * 1.44269504)

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(OUT_DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(OUT_DTYPE))
}

/-- The full aligned attention-kernel surface lowers to the algorithm layer. -/
theorem attention_kernel_fwd_kernel_aligned_surface_toAlgorithm_supported
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ∃ alg, (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
      stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
      stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
      BLOCK_M BLOCK_N out_dtype).toAlgorithm? = Except.ok alg := by
  simp [attention_kernel_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def surfaceOutOffset
    (s : BlockState)
    (stride_qh stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 1 * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + kIndex idx * stride_on

/-! ## Genuine closed-form correctness — reusable foundation

The section below builds the **genuine** closed-form claim that the public
summary asserts: the kernel computes the base-2 streaming softmax
`attnGenScore fscore V` of the loaded Q/K/V tiles, with the kernel's actual
per-key score `fscore` (scaled dot plus the additive relative-position bias
`(b0 + b1)·log2 e`) — not the kernel's own executed `Out` readback.

The pure-math heart (`VeriTile.Triton.attnGenScore`, `closed_form_g`,
`attnGenScore_eq_streaming` in `Math/Attention.lean`) and the block-pointer
`evalOp` reduction lemmas (`Semantics/BlockPtrEval.lean`) are complete and
sorry-free; this section supplies the kernel-specific `evalOp` reductions
(`makeBlockPtrDyn`/`makeBlockPtr`-with-dynamic-row-offset and the K/V/Q
block-pointer loads at their resolved contiguous addresses) plus the genuine
score/tile specification `fscore`/`qRaw`/`kFlat`/`vFlat`/`b0Val`/`b1Val`. The
`exec`-side assembly (preLoop base case, the 15-statement loop-body
invariant step over `forRangeDyn`, the `acc /= l_i` + block-pointer-store
epilogue, and the `ComputeCorrect.Realizes` bench bridge) is built in-file,
mirroring `VeriTile.Examples.AttentionForwardClosedForm`, swapping in the
generalized `mPg`/`lPgK`/`oPg` invariant, the `forRangeDyn` loop driver, and
these block-ptr lemmas. -/

namespace ClosedForm

open VeriTile.Triton

/-- **`makeBlockPtrDyn` eval** (the `K`/`V` block pointers — static offsets,
dynamic base): evaluates the base-offset op and packages the constant
`BlockPtr` tile. -/
theorem makeBlockPtrDyn_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides offsets : List Nat) (s : BlockState) (base : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base)) :
    evalOp (.makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := offsets }⟩) := by
  simp only [evalOp, hb, Option.bind]
  rfl

/-- **`makeBlockPtrDynOffsets` (dynamic row offset, literal `0` column) eval**
(the `Q`/`O` block pointers): packages the constant `BlockPtr` tile with the
resolved row offset. -/
theorem makeBlockPtr_rowcol_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape) (strides : List Nat)
    (rowOp : Op .nat []) (s : BlockState) (base rowOff : Nat)
    (hb : evalOp baseOffset s = some (Tile.scalar base))
    (hr : evalOp rowOp s = some (Tile.scalar rowOff)) :
    evalOp (.makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides
        [rowOp, Op.constNat 0]) s
      = some (⟨fun _ : TileIndex blockShape =>
          { region := region, baseOffset := base, parentShape := parentShape,
            blockShape := blockShape, strides := strides, offsets := [rowOff, 0] }⟩) := by
  rw [makeBlockPtr2_eval]
  simp only [hb, hr, evalOp, Option.bind, List.mapM, List.mapM.loop, Tile.scalar]
  rfl

/-- **`tl.advance [0, d]` of a `[rowOff=0, colOff]` block pointer** (the `K`
pointer step): advances the column offset by `d`. -/
theorem advance_col_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS colOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, colOff] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [(0:Nat), d]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [0, colOff + d] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

/-- **`tl.advance [d, 0]` of a `[rowOff, colOff=0]` block pointer** (the `V`
pointer step): advances the row offset by `d`. -/
theorem advance_row_eval (s : BlockState) (region : RegionName)
    (base rows cols BT BS strideT strideS rowOff d : Nat) (name : RegName)
    (hkp : s.regs .blockPtr [BT, BS] name = some
      (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [rowOff, 0] }⟩)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BT, BS] name) [d, (0:Nat)]) s
      = some (⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS], offsets := [rowOff + d, 0] }⟩) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp, hkp, Option.bind]
  refine congrArg some ?_
  ext i
  simp [BlockPtr.advance_2d_offsets]

/-! ### Genuine kernel specification

The loaded Q/K/V/B0 tiles read from the *input* state `s0`, and the kernel's
genuine per-key score `fscore` (used as the `score` argument of the generalized
streaming-softmax spec `VeriTile.Triton.attnGenScore`). -/

/-- Loaded (pre-scale) Q tile: row `r`, head lane `e`. -/
noncomputable def qRaw (s0 : BlockState) (Q : RegionName)
    (q_offset BLOCK_M HEAD_DIM : Nat) (start_m : Nat) :
    TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  fun (r, e, _) => s0.readMem Q (q_offset + (start_m * BLOCK_M + r.val) * HEAD_DIM + e.val)

/-- Loaded K tile as a flat per-key function over `[HEAD_DIM, N_CTX]`. -/
noncomputable def kFlat (s0 : BlockState) (K : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    Fin HEAD_DIM → Fin N_CTX → ℝ :=
  fun e j => s0.readMem K (kv_offset + e.val + j.val * HEAD_DIM)

/-- Loaded V tile, flat per-key over `[N_CTX, HEAD_DIM]`. -/
noncomputable def vFlat (s0 : BlockState) (V : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    TileIndex [N_CTX, HEAD_DIM] → ℝ :=
  fun (j, d, _) => s0.readMem V (kv_offset + j.val * HEAD_DIM + d.val)

/-- Per-row, per-block-column bias `b0` read (`c = j / BLOCK_N`). -/
noncomputable def b0Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (c : Nat) : ℝ :=
  s0.readMem B0 (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + c)

/-- Per-row, per-lane bias `b1` read at lane `jL` (`jL = j % BLOCK_N`). -/
noncomputable def b1Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (jL : Nat) : ℝ :=
  s0.readMem B0
    (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + (jL % BIAS_LAST_SIZE + BIAS_LAST_SIZE))

/-- **Genuine per-key score** `fscore r j` of `_fwd_kernel_aligned`:
`qk_scale·(Σ_e Q[r,e]·K[e,j]) + (b0[r, j/BN] + b1[r, j%BN])·log2 e`,
with `qk_scale = sm_scale · log2 e` already folded into the pre-scaled `q`. This
is the `score` argument of `VeriTile.Triton.attnGenScore`, whose batch base-2
softmax `(Σ 2^fscore · V) / (Σ 2^fscore)` is the kernel's closed form (see
`closed_form_g`). -/
noncomputable def fscore (s0 : BlockState) (Q K B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat)
    (r : Fin BLOCK_M) (j : Fin N_CTX) : ℝ :=
  (sm_scale * 1.44269504) *
      Finset.univ.sum (fun e : Fin HEAD_DIM =>
        qRaw s0 Q q_offset BLOCK_M HEAD_DIM start_m (r, e, PUnit.unit)
          * kFlat s0 K kv_offset HEAD_DIM N_CTX e j)
    + (b0Val s0 B0 b_offset BLOCK_M stride_b0m start_m r (j.val / BLOCK_N)
        + b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE start_m r (j.val % BLOCK_N))
      * 1.44269504

/-- **Closed-form target.** The kernel's genuine specification is
`attnGenScore (fscore …) (vFlat …)`: the base-2 streaming softmax over the
genuine per-key score `fscore` (scaled dot + additive bias). The generalized
math lemma `closed_form_g` establishes that the running `oPg / lPg` recurrence
the kernel's loop maintains converges to exactly this value after all
`numKVBlocks` key blocks — which is what the remaining `exec`-side assembly
discharges. -/
noncomputable def attentionKernelSpec (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat) : TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  attnGenScore
    (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m start_m)
    (vFlat s0 V kv_offset HEAD_DIM N_CTX)

/-- **Closed-form readout.** After all `numKVBlocks` key blocks the kernel's
running `oPg / lPg` ratio (which the loop invariant maintains) is exactly the
genuine spec `attentionKernelSpec` (= `attnGenScore fscore vFlat`). Direct
corollary of the generalized math lemma `closed_form_g`. -/
theorem attentionKernelSpec_eq_ratio (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM BIAS_LAST_SIZE stride_b0m nB : Nat)
    (start_m : Nat) (hBN : 0 < BLOCK_N) (hnB : 1 ≤ nB)
    (i : Fin BLOCK_M) (d : Fin HEAD_DIM) :
    let score := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m start_m
    let V' := vFlat s0 V kv_offset HEAD_DIM (BLOCK_N * nB)
    oPg score V' i d nB / lPg score i nB
      = attentionKernelSpec s0 Q K V B0 sm_scale q_offset kv_offset b_offset
          BLOCK_M BLOCK_N HEAD_DIM (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m start_m (i, d, PUnit.unit) :=
  closed_form_g _ _ hBN hnB i d

/-- **Kernel-faithful running denominator.** Same recurrence as the math `lPg`
but seeded `0` (the kernel inits `l_i = 0`, whereas the generalized `lPg` seeds
`1`). This is the value the loop invariant must carry in the `l_i` register. -/
noncomputable def lPgK {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
      if h : c + 1 ≤ nB then
        alphaPg score i c * lPgK score i c +
          Finset.univ.sum (fun a : Fin BN =>
            pow2 (score i (gkey BN nB c (by omega) a) - mRg score i (c + 1)))
      else lPgK score i c

/-- The seed difference is killed by `alphaPg 0 = 0`: for `1 ≤ c ≤ nB` the
kernel's seed-`0` denominator agrees with the math `lPg` (seed `1`). So the
final `c = nB ≥ 1` readout is unaffected by the kernel's `l_i = 0` initialization,
and `closed_form_g` / `attentionKernelSpec_eq_ratio` apply unchanged. -/
theorem lPgK_eq_lPg {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN * nB) → ℝ) (i : Fin Mq) :
    ∀ c, 1 ≤ c → c ≤ nB → lPgK score i c = lPg score i c := by
  intro c hc1 hc
  induction c, hc1 using Nat.le_induction with
  | base => simp only [lPgK, lPg, dif_pos hc, alphaPg_zero score i, zero_mul]
  | succ c hc1 ih =>
    have ihc := ih (by omega)
    simp only [lPgK, lPg, dif_pos hc, ihc]

/-! ### Generic streaming-softmax register bridges

The per-cell facts that the loop body's tile arithmetic (over the generic per-key
`score`) realizes the `mPg`/`lPgK`/`oPg` `_succ` recurrence. Mirrors the
`mij_eq`/`alpha_eq`/`li_eq`/`acc_eq` bridges of
`VeriTile.Examples.AttentionForwardClosedForm`, generic over `score` (the kernel's
`fscore`), so they apply unchanged to the banked invariant. -/

/-- `WithBot.realExp2` is total: never `⊥`. -/
theorem realExp2_eq_some_unbotD (z : WithBot ℝ) :
    WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
  cases z <;> rfl

theorem realExp2_unbotD_coe (r : ℝ) :
    (WithBot.realExp2 ((r : ℝ) : WithBot ℝ)).unbotD 0 = pow2 r := by
  simp [pow2, mul_comm]

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- The `p · v` dot row (contraction over all `BLOCK_N` keys). -/
theorem dot_pv (BM BN BD : Nat) (p : Tile .real [BM,BN]) (v : Tile .real [BN,BD])
    (r : Fin BM) (d : Fin BD) (fp fv : Fin BN → ℝ)
    (hp : ∀ jL : Fin BN, p.data (r, jL, PUnit.unit) = some (fp jL))
    (hv : ∀ jL : Fin BN, v.data (jL, d, PUnit.unit) = some (fv jL)) :
    (Tile.dot [] p v).data (r, d, PUnit.unit) = some (Finset.univ.sum fun jL : Fin BN => fp jL * fv jL) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (p.data (r, k, PUnit.unit)) (v.data (k, d, PUnit.unit))))
      = @Finset.sum (Fin BN) (WithBot ℝ) _ Finset.univ (fun k => (some (fp k * fv k) : WithBot ℝ))
      from Finset.sum_congr rfl (fun k _ => by rw [hp k, hv k]; rfl)]
  exact withBot_sum_some _

/-- A `some`-valued exp2 reduction at one cell. -/
theorem exp2_some {M N : Nat} (h : Fin M → Fin N → ℝ) (x : Tile .real [M, N])
    (i : Fin M) (j : Fin N) (hx : x.data (i, j, PUnit.unit) = some (h i j)) :
    (Tile.uop WithBot.realExp2 x).data (i, j, PUnit.unit) = some (pow2 (h i j)) := by
  show WithBot.realExp2 _ = _
  rw [hx, WithBot.realExp2_some]
  refine congrArg some ?_
  rw [show pow2 (h i j) = Real.exp (Real.log 2 * (h i j)) from rfl, mul_comm]

/-- `reduceMaxDrop` of a row whose cells are `g jL` = `Finset.sup` of the `g`. -/
theorem reduceMaxDrop_data_row (BM BN : Nat) (hBN : 0 < BN) (qk : Tile .real [BM, BN])
    (rmaxT : Tile .real [BM]) (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qk = some rmaxT)
    (r : Fin BM) (g : Fin BN → WithBot ℝ) (hqk : ∀ jL : Fin BN, qk.data (r, jL, PUnit.unit) = g jL) :
    rmaxT.data (r, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BM,BN] (⟨1, by simp⟩ : Fin [BM,BN].length) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- **`m_ij = mPg(c+1)`** (generic). -/
theorem mijg_eq {Mq : Nat} (BN nB c : Nat) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i rmaxT : Tile .real [Mq]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hrmax : rmaxT.data (r, PUnit.unit)
        = Finset.univ.sup (fun a : Fin BN => ((score r (gkey BN nB c hc a) : ℝ) : WithBot ℝ))) :
    (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) m_i rmaxT) m_i rmaxT).data (r, PUnit.unit)
      = mPg BN nB score r (c+1) := by
  rw [Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmi, hrmax]
  rw [mPg, dif_pos (by omega : c + 1 ≤ nB)]
  by_cases h : mPg BN nB score r c ≤ Finset.univ.sup (fun a : Fin BN => ((score r (gkey BN nB c hc a) : ℝ) : WithBot ℝ))
  · rw [if_neg (by simp [not_lt.mpr h]), max_eq_right h]
  · rw [if_pos (by simpa using not_le.mp h), max_eq_left (le_of_lt (not_le.mp h))]

/-- `mPg (c+1) ≠ ⊥` whenever `mPg c ≠ ⊥` (running max is monotone). -/
theorem mPg_succ_ne_bot {Mq : Nat} {BN nB : Nat}
    (score : Fin Mq → Fin (BN*nB) → ℝ) (r : Fin Mq) (c : Nat)
    (hc : mPg BN nB score r c ≠ ⊥) : mPg BN nB score r (c+1) ≠ ⊥ := by
  rw [mPg]
  by_cases h : c + 1 ≤ nB
  · rw [dif_pos h]
    intro hbot; rw [max_eq_bot] at hbot; exact hc hbot.1
  · rw [dif_neg h]; exact hc

/-- **`alpha = some(alphaPg c)`** (generic). -/
theorem alphag_eq {Mq : Nat} (BN nB c : Nat)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i m_ij : Tile .real [Mq]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hmij : m_ij.data (r, PUnit.unit) = mPg BN nB score r (c+1)) :
    (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)).data (r, PUnit.unit)
      = some (alphaPg score r c) := by
  show WithBot.realExp2 _ = _
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmi, hmij, NumericDType.sub]
  unfold alphaPg
  cases hm : mPg BN nB score r c with
  | bot => rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
  | coe a =>
    show WithBot.realExp2 (WithBot.realSub (↑a) (mPg BN nB score r (c+1))) = some (pow2 (a - mRg score r (c+1)))
    rw [show mRg score r (c+1) = (mPg BN nB score r (c+1)).unbotD 0 from rfl]
    have hne : mPg BN nB score r (c+1) ≠ ⊥ := mPg_succ_ne_bot score r c (by rw [hm]; exact WithBot.coe_ne_bot)
    cases hm1 : mPg BN nB score r (c+1) with
    | bot => exact absurd hm1 hne
    | coe b =>
      rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
      refine congrArg some ?_
      rw [show pow2 (a - b) = Real.exp (Real.log 2 * (a - b)) from rfl, mul_comm]

set_option maxHeartbeats 1000000 in
/-- **`l_i = lPgK(c+1)`** (generic). -/
theorem lig_eq {Mq : Nat} (BN nB c : Nat) (hBN : 0 < BN) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ)
    (m_i m_ij l_i : Tile .real [Mq]) (qk : Tile .real [Mq, BN]) (r : Fin Mq)
    (hmi : m_i.data (r, PUnit.unit) = mPg BN nB score r c)
    (hmij : m_ij.data (r, PUnit.unit) = mPg BN nB score r (c+1))
    (hli : l_i.data (r, PUnit.unit) = some (lPgK score r c))
    (hqk : ∀ a : Fin BN, qk.data (r, a, PUnit.unit) = some (score r (gkey BN nB c hc a))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
       (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) l_i
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) m_i m_ij)))
       (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [Mq,BN].length)
         (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qk
           (Tile.expandDim (⟨1, by simp⟩ : Fin [Mq].length.succ) m_ij))))).data (r, PUnit.unit)
      = some (lPgK score r (c+1)) := by
  have hmijReal : m_ij.data (r, PUnit.unit) = some (mRg score r (c+1)) := by
    rw [hmij]
    rw [mPg_eq_coe score hBN r (c+1) (by omega) (by omega)]; rfl
  have halpha := alphag_eq BN nB c score m_i m_ij r hmi hmij
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [halpha, hli]
  simp only [Tile.reduceSumDrop_data, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, TileShape.dropInsertedIndex,
    TileShape.insertAxisIndex, hqk, hmijReal, NumericDType.sub, WithBot.realSub,
    Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
  simp only [withBot_sum_some]
  rw [lPgK, dif_pos (by omega : c + 1 ≤ nB)]
  simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [mul_comm (lPgK score r c)]
  congr 1
  exact Finset.sum_congr rfl (fun k _ => by rw [pow2, mul_comm])

set_option maxHeartbeats 1000000 in
/-- **`acc = oPg(c+1)`** (generic). -/
theorem accg_eq {Mq Dh : Nat} (BN nB c : Nat) (hc : c < nB)
    (score : Fin Mq → Fin (BN*nB) → ℝ) (V : TileIndex [BN*nB, Dh] → ℝ)
    (acc : Tile .real [Mq,Dh]) (alpha : Tile .real [Mq]) (p : Tile .real [Mq,BN]) (v : Tile .real [BN,Dh])
    (r : Fin Mq) (d : Fin Dh)
    (hacc : acc.data (r, d, PUnit.unit) = some (oPg score V r d c))
    (halpha : alpha.data (r, PUnit.unit) = some (alphaPg score r c))
    (hp : ∀ a : Fin BN, p.data (r, a, PUnit.unit) = some (pow2 (score r (gkey BN nB c hc a) - mRg score r (c+1))))
    (hv : ∀ a : Fin BN, v.data (a, d, PUnit.unit) = some (V (gkey BN nB c hc a, d, PUnit.unit))) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
       (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acc
         (Tile.expandDim (⟨1, by simp⟩ : Fin [Mq].length.succ) alpha))
       (Tile.dot [] p v)).data (r, d, PUnit.unit)
      = some (oPg score V r d (c+1)) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, halpha]
  rw [dot_pv Mq BN Dh p v r d (fun a => pow2 (score r (gkey BN nB c hc a) - mRg score r (c+1)))
      (fun a => V (gkey BN nB c hc a, d, PUnit.unit)) hp hv]
  rw [hacc]
  rw [oPg, dif_pos (by omega : c + 1 ≤ nB)]
  simp only [NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
    Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [mul_comm (oPg score V r d c)]

/-! ### Loop invariant skeleton

The `forRangeDyn` loop-invariant `Prop` carried across key blocks. The counter `i`
ranges over `0, BLOCK_N, 2·BLOCK_N, …` (`c = i / BLOCK_N` blocks consumed); the
streaming registers `m_i`/`l_i`/`acc` equal the generalized partials
`mPg`/`lPgK`/`oPg` of the genuine score `fscore` and value tile `vFlat` over the
first `c` blocks (kernel seed `l_i = 0`, hence `lPgK` not `lPg`), and the K/V
block pointers have advanced `c` column/row steps of `BLOCK_N`. The loop-invariant
registers (`q` pre-scaled+loaded, `b1`, `b_offset`, `start_m`/`off_hz`) are fixed.
`preLoop` establishes `attnKernelInvariant … 0`, `attn_step` advances it by one
block, and `attn_postLoop` reads the closed form off `… numKVBlocks` via
`attentionKernelSpec_eq_ratio`. -/
noncomputable def attnKernelInvariant
    (s0 : BlockState) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM P_SEQ BIAS_LAST_SIZE stride_b0m stride_kk stride_kn
      stride_vk stride_vn numKVBlocks : Nat)
    (start_m : Nat) (i : Nat) (s : BlockState) : Prop :=
  let nB := numKVBlocks; let c := i / BLOCK_N; let N_CTX := BLOCK_N * numKVBlocks
  let score := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
    BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m start_m
  let V' := vFlat s0 V kv_offset HEAD_DIM N_CTX
  s.pids = s0.pids ∧ i = c * BLOCK_N ∧ c ≤ nB ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] => mPg BLOCK_N nB score r.1 c⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] => ((lPgK score r.1 c : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, HEAD_DIM] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, HEAD_DIM] =>
        ((oPg score V' idx.1 idx.2.1 c : ℝ) : WithBot ℝ)⟩) ∧
  (s.regs .blockPtr [HEAD_DIM, BLOCK_N] "K_block_ptr" = some
      (⟨fun _ : TileIndex [HEAD_DIM, BLOCK_N] =>
        { region := K, baseOffset := kv_offset,
          parentShape := [HEAD_DIM, N_CTX + P_SEQ], blockShape := [HEAD_DIM, BLOCK_N],
          strides := [stride_kk, stride_kn], offsets := [0, c * BLOCK_N] }⟩)) ∧
  (s.regs .blockPtr [BLOCK_N, HEAD_DIM] "V_block_ptr" = some
      (⟨fun _ : TileIndex [BLOCK_N, HEAD_DIM] =>
        { region := V, baseOffset := kv_offset,
          parentShape := [N_CTX + P_SEQ, HEAD_DIM], blockShape := [BLOCK_N, HEAD_DIM],
          strides := [stride_vk, stride_vn], offsets := [c * BLOCK_N, 0] }⟩)) ∧
  -- loop-invariant (fixed) registers the body reads
  (s.regs .fp16 [BLOCK_M, HEAD_DIM] "q" = some ⟨fun idx : TileIndex [BLOCK_M, HEAD_DIM] =>
      FloatDType.real.cast FloatDType.fp16
        (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD_DIM start_m idx))⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_N] "b1" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE start_m idx.1 idx.2.1.val)⟩) ∧
  (s.regs .nat [] "b_offset" = some (Tile.scalar b_offset)) ∧
  (s.regs .nat [] "start_m" = some (Tile.scalar start_m)) ∧
  (s.regs .nat [] "q_offset" = some (Tile.scalar q_offset)) ∧
  (s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-! ### preLoop prefix (deterministic prologue → invariant base seeds)

The first ten prologue statements are deterministic (no loads): the four scalar
offsets, the three block pointers, and the `m_i`/`l_i`/`acc` initializers. They
step with the validated recipe (`stepStmts.cons_some (stepStmt_assign_eq_some
…)` per statement, threading register reads through the accumulated `setReg`
chain) using the block-pointer eval lemmas above. The result is the base-case
register seeding for the loop invariant: `m_i = ⊥` (= `mPg … 0`),
`l_i = 0` (the kernel's seed; `lPgK … 0`), `acc = 0` (= `oPg … 0`), and the
`K`/`V` block pointers at their initial `[0,0]` offsets. -/

/-- Scalar `ref · constNat` eval (the `q_offset` / `kv_offset` / `start_m·BLOCK_M`
offsets). -/
theorem evalOp_mul_ref_const (s : BlockState) (name : RegName) (a c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar a)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (a * c)) := by
  rw [evalOp_mul, evalOp_ref, hr, evalOp_constNat]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop_data, NumericDType.mul]
  rfl

set_option maxHeartbeats 1000000 in
/-- **preLoop prefix** (prologue statements 0–9): steps the deterministic
offset/pointer/init prologue to a state carrying the invariant base seeds. -/
theorem preLoop_prefix (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (out_dtype : FloatDType) (s : BlockState) :
    ∃ s10, stepStmts ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
        Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N
        out_dtype).toAlgKernel.body.take 10) s = some s10
      ∧ s10.pids = s.pids
      ∧ s10.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s10.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s10.regs .nat [] "q_offset" = some (Tile.scalar (s.pids 1 * stride_qh))
      ∧ s10.regs .nat [] "kv_offset" = some (Tile.scalar (s.pids 1 * stride_kh))
      ∧ s10.regs .blockPtr [BLOCK_M, BLOCK_DMODEL] "Q_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
            { region := Q, baseOffset := s.pids 1 * stride_qh,
              parentShape := [N_CTX, BLOCK_DMODEL], blockShape := [BLOCK_M, BLOCK_DMODEL],
              strides := [stride_qm, stride_qk], offsets := [s.pids 0 * BLOCK_M, 0] }⟩)
      ∧ s10.regs .blockPtr [BLOCK_DMODEL, BLOCK_N] "K_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            { region := K, baseOffset := s.pids 1 * stride_kh,
              parentShape := [BLOCK_DMODEL, N_CTX + P_SEQ], blockShape := [BLOCK_DMODEL, BLOCK_N],
              strides := [stride_kk, stride_kn], offsets := [0, 0] }⟩)
      ∧ s10.regs .blockPtr [BLOCK_N, BLOCK_DMODEL] "V_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            { region := V, baseOffset := s.pids 1 * stride_kh,
              parentShape := [N_CTX + P_SEQ, BLOCK_DMODEL], blockShape := [BLOCK_N, BLOCK_DMODEL],
              strides := [stride_vk, stride_vn], offsets := [0, 0] }⟩)
      ∧ s10.regs .real [BLOCK_M] "m_i" = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩)
      ∧ s10.regs .real [BLOCK_M] "l_i" = some (⟨fun _ : TileIndex [BLOCK_M] => some (0:ℝ)⟩)
      ∧ s10.regs .real [BLOCK_M, BLOCK_DMODEL] "acc"
          = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0:ℝ)⟩)
      ∧ s10.undef = s.undef
      ∧ s10.mem = s.mem := by
  rw [show ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h stride_b0m
        Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N
        out_dtype).toAlgKernel.body.take 10)
      = [ Stmt.assign .nat [] "start_m" (Op.programId 0),
          Stmt.assign .nat [] "off_hz" (Op.programId 1),
          Stmt.assign .nat [] "q_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat stride_qh)),
          Stmt.assign .nat [] "kv_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat stride_kh)),
          Stmt.assign .blockPtr [BLOCK_M, BLOCK_DMODEL] "Q_block_ptr"
            (Op.makeBlockPtrDynOffsets Q (Op.ref .nat [] "q_offset") [N_CTX, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL]
              [stride_qm, stride_qk] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M), Op.constNat 0]),
          Stmt.assign .blockPtr [BLOCK_DMODEL, BLOCK_N] "K_block_ptr"
            (Op.makeBlockPtrDyn K (Op.ref .nat [] "kv_offset") [BLOCK_DMODEL, N_CTX + P_SEQ] [BLOCK_DMODEL, BLOCK_N]
              [stride_kk, stride_kn] [0, 0]),
          Stmt.assign .blockPtr [BLOCK_N, BLOCK_DMODEL] "V_block_ptr"
            (Op.makeBlockPtrDyn V (Op.ref .nat [] "kv_offset") [N_CTX + P_SEQ, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL]
              [stride_vk, stride_vn] [0, 0]),
          Stmt.assign .real [BLOCK_M] "m_i" (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
          Stmt.assign .real [BLOCK_M] "l_i" (Op.full [BLOCK_M] (Op.const 0)),
          Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) ] from rfl]
  have hmi : ∀ s' : BlockState, evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) s'
      = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) := by
    intro s'
    simp only [evalOp_add, evalOp_full, evalOp_negInf, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext r; simp only [Tile.bop_data, NumericDType.add]; rfl
  have hli : ∀ s' : BlockState, evalOp (Op.full [BLOCK_M] (Op.const 0)) s'
      = some (⟨fun _ : TileIndex [BLOCK_M] => some (0:ℝ)⟩ : Tile .real [BLOCK_M]) := by
    intro s'; simp [evalOp_full, evalOp_const, Option.bind]
  have hacc : ∀ s' : BlockState, evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) s'
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => some (0:ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) := by
    intro s'; simp [evalOp_full, evalOp_const, Option.bind]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (s := s) (by rw [evalOp_programId])),
    stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (evalOp_mul_ref_const _ "off_hz" (s.pids 1) stride_qh (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (evalOp_mul_ref_const _ "off_hz" (s.pids 1) stride_kh (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_rowcol_eval Q _ [N_CTX, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL] [stride_qm, stride_qk] _ _
        (s.pids 1 * stride_qh) (s.pids 0 * BLOCK_M)
        (by simp [BlockState.setReg]) (evalOp_mul_ref_const _ "start_m" (s.pids 0) BLOCK_M (by simp [BlockState.setReg])))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtrDyn_eval K _ [BLOCK_DMODEL, N_CTX + P_SEQ] [BLOCK_DMODEL, BLOCK_N] [stride_kk, stride_kn] [0,0] _
        (s.pids 1 * stride_kh) (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtrDyn_eval V _ [N_CTX + P_SEQ, BLOCK_DMODEL] [BLOCK_N, BLOCK_DMODEL] [stride_vk, stride_vn] [0,0] _
        (s.pids 1 * stride_kh) (by simp [BlockState.setReg]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (hmi _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (hli _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (hacc _)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [BlockState.setReg, BlockState.setReg_same, BlockState.setReg_ne_name]

/-! ### preLoop tail loads (statements 10–18 eval helpers)

The prologue's final block (statements 10–18: `qk_scale`, the Q block-pointer
load + scale + `OUT_DTYPE` cast, the `lo`/`hi` constants, the bias arange/offset
prep, and the `b1` region load) needs three eval reductions beyond the
deterministic-prologue recipe used by `preLoop_prefix`:

* the two generalized canonical-axis `expandDim` reductions over an arbitrary
  `nat` operand (`evalOp_expandDim_one_nat'`/`evalOp_expandDim_zero_nat'`) — the
  generic `evalOp_expandDim` cannot `rw`/`simp`-match the `triton{}`-elaborated
  surface because its result shape `TileShape.insertAxis [M] 1 1` is only
  defeq-not-syntactic to `[M, 1]`; these state the concrete-shape form (the
  `b1`/`b0` offset uses `expandDim` of a *computed* `mul`/`add`, not a bare
  register, so the existing register-only specializations don't apply);
* the `b1` (and identically-shaped `b0`) relative-position-bias region load
  (`load_b1_eval`): reads `B0` at the resolved contiguous per-lane address
  `b_offset + (start_m·BLOCK_M + r)·stride_b0m + (jL % BIAS_LAST_SIZE +
  BIAS_LAST_SIZE)` — i.e. exactly `b1Val`'s address (the `b0` block-column read
  `… + start_n // BLOCK_N` is the `floorDiv` variant). -/

/-- Generalized canonical axis-1 `expandDim` eval over an arbitrary `nat`
operand (explicit `[M, 1]` result shape). -/
@[simp] theorem evalOp_expandDim_one_nat' {M : Nat} (e : Op .nat [M]) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ e) s =
      (evalOp e s).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  conv_lhs => unfold evalOp
  cases evalOp e s with
  | none => rfl
  | some v =>
      simp only [Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_; ext i; simp [Tile.expandDim]

/-- Generalized canonical axis-0 `expandDim` eval over an arbitrary `nat`
operand (explicit `[1, D]` result shape). -/
@[simp] theorem evalOp_expandDim_zero_nat' {D : Nat} (e : Op .nat [D]) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ e) s =
      (evalOp e s).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  conv_lhs => unfold evalOp
  cases evalOp e s with
  | none => rfl
  | some v =>
      simp only [Option.bind_some, Option.bind_eq_bind]
      refine congrArg some ?_; ext i; simp [Tile.expandDim]

/-- **`b1` relative-position-bias region load** (statement 18; the `b0` per-block
read is the `floorDiv` analogue). Resolves the `B0 + b_offset + (start_m·BLOCK_M
+ b_ptr_offsets_m)·stride_b0m[:,None] + b_ptr_offsets_n_1[None,:]` pointer
arithmetic to the per-lane `readMem` at `b_offset + (start_m·BLOCK_M + r)·
stride_b0m + (jL % BIAS_LAST_SIZE + BIAS_LAST_SIZE)` (= `b1Val`'s address). -/
theorem load_b1_eval (s : BlockState) (B0 : RegionName)
    (BLOCK_M BLOCK_N stride_b0m BIAS_LAST_SIZE smbm boff : Nat)
    (hbo : s.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hm : s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val)))
    (hn1 : s.regs .nat [BLOCK_N] "b_ptr_offsets_n_1"
      = some (Tile.vec (fun jL : Fin BLOCK_N => jL.val % BIAS_LAST_SIZE + BIAS_LAST_SIZE))) :
    evalOp (Op.load .real (.region B0
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "b_ptr_offsets_n_1"))))
        .none) s
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some (s.readMem B0
            (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m
              + (idx.2.1.val % BIAS_LAST_SIZE + BIAS_LAST_SIZE)))⟩) := by
  rw [evalOp_load_region_none]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat',
    evalOp_expandDim_zero_nat', hbo, hsm, hm, hn1, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, Tile.vec,
    NumericDType.add, NumericDType.mul, BlockState.readMemValue_real, Region.cast_id]

/-! ## exec-side loop-body assembly (Python shape `BLOCK_M=BLOCK_N=64`, `HEAD_DIM=128`)

The 15-statement `forRangeDyn` body, stepped one block at a time. Mirrors the
`AttentionForwardClosedForm`/`flash_attn` assembly: block-ptr K/V loads, the
bias-augmented `qk = fscore` cell, the online-softmax `m_i`/`alpha`/`p`/`acc`/`l_i`
updates, and the K/V block-pointer advances — non-causal (no masks). -/

/-- `evalOp` of `tl.math.exp2`. -/
theorem evalOp_exp2' {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- The lowered 15-statement `forRangeDyn` body of the Python-shape kernel. -/
def attnLoopBody (B0 : RegionName) : List Stmt :=
  [ Stmt.assign .real [128, 64] "k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [128, 64] "K_block_ptr") []) .none),
    Stmt.assign .real [64, 128] "v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 128] "V_block_ptr") []) .none),
    Stmt.assign .fp16 [64, 64] "qk" (Op.full [64, 64] (Op.castFloat .real .fp16 (Op.const 0))),
    Stmt.assign .real [64, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 64] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q"))
          (Op.ref .real [128, 64] "k"))),
    Stmt.assign .real [64, 1] "b0"
      (Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 64))
                (Op.ref .nat [64] "b_ptr_offsets_m")) (Op.constNat 128))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))))
        .none),
    Stmt.assign .real [64, 64] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 64] "qk")
        (Op.mul .real Broadcast.scalarR
          (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.ref .real [64, 1] "b0") (Op.ref .real [64, 64] "b1")) (Op.const 1.44269504))),
    Stmt.assign .real [64] "m_i_new"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [64] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "qk")))
        (Op.ref .real [64] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "qk"))),
    Stmt.assign .real [64] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [64] "m_i")
        (Op.ref .real [64] "m_i_new"))),
    Stmt.assign .real [64, 64] "p"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 64] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "m_i_new")))),
    Stmt.assign .real [64, 128] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "alpha"))),
    Stmt.assign .real [64, 128] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 128] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [64, 64] "p"))) (Op.ref .real [64, 128] "v"))),
    Stmt.assign .real [64] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [64] "l_i")
          (Op.ref .real [64] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [64, 64].length) Bool.false (Op.ref .real [64, 64] "p"))),
    Stmt.assign .real [64] "m_i" (Op.ref .real [64] "m_i_new"),
    Stmt.assign .blockPtr [128, 64] "K_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [128, 64] "K_block_ptr") [(0:Nat), (64:Nat)]),
    Stmt.assign .blockPtr [64, 128] "V_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [64, 128] "V_block_ptr") [(64:Nat), (0:Nat)]) ]

/-- The lowered 15-statement `forRangeDyn` body, dimension-parameterized over
`BLOCK_M BLOCK_N HEAD_DIM stride_b0m`. Specializes to `attnLoopBody` at the
Python test shape (`64 64 128 128`). -/
def attnLoopBodyG (B0 : RegionName)
    (BLOCK_M BLOCK_N HEAD_DIM stride_b0m : Nat) : List Stmt :=
  [ Stmt.assign .real [HEAD_DIM, BLOCK_N] "k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [HEAD_DIM, BLOCK_N] "K_block_ptr") []) .none),
    Stmt.assign .real [BLOCK_N, HEAD_DIM] "v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_N, HEAD_DIM] "V_block_ptr") []) .none),
    Stmt.assign .fp16 [BLOCK_M, BLOCK_N] "qk" (Op.full [BLOCK_M, BLOCK_N] (Op.castFloat .real .fp16 (Op.const 0))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, HEAD_DIM] "q"))
          (Op.ref .real [HEAD_DIM, BLOCK_N] "k"))),
    Stmt.assign .real [BLOCK_M, 1] "b0"
      (Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N))))
        .none),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.mul .real Broadcast.scalarR
          (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.ref .real [BLOCK_M, 1] "b0") (Op.ref .real [BLOCK_M, BLOCK_N] "b1")) (Op.const 1.44269504))),
    Stmt.assign .real [BLOCK_M] "m_i_new"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))),
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BLOCK_M] "m_i")
        (Op.ref .real [BLOCK_M] "m_i_new"))),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_i_new")))),
    Stmt.assign .real [BLOCK_M, HEAD_DIM] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, HEAD_DIM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "alpha"))),
    Stmt.assign .real [BLOCK_M, HEAD_DIM] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, HEAD_DIM] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p"))) (Op.ref .real [BLOCK_N, HEAD_DIM] "v"))),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BLOCK_M] "l_i")
          (Op.ref .real [BLOCK_M] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p"))),
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_i_new"),
    Stmt.assign .blockPtr [HEAD_DIM, BLOCK_N] "K_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [HEAD_DIM, BLOCK_N] "K_block_ptr") [(0:Nat), BLOCK_N]),
    Stmt.assign .blockPtr [BLOCK_N, HEAD_DIM] "V_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BLOCK_N, HEAD_DIM] "V_block_ptr") [BLOCK_N, (0:Nat)]) ]

/-- The Python-shape lowered body `drop 19` is the `forRangeDyn` loop over
`attnLoopBody`, then the 3 post-loop statements. By `rfl`. -/
theorem attnLoopBody_check (Q K V B0 Out : RegionName) :
    (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.drop 19
      = Stmt.forRangeDyn "start_n" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat 64) (attnLoopBody B0)
        :: (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
            16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
            FloatDType.fp16).toAlgKernel.body.drop 20 := by
  rfl

/-- `ComputeOp.toAlgorithm?` of an `.alg` op (the structural erase). -/
@[simp] theorem computeOp_toAlgorithm?_alg {dtype : ComputeDType} {shape : TileShape}
    (e : Op dtype.eraseDType shape) :
    ComputeOp.toAlgorithm? (ComputeOp.alg dtype e) = Except.ok e := rfl

/-- `ComputeExpr.toAlgorithm?` of `compute (alg ..)`. -/
@[simp] theorem computeExpr_toAlgorithm?_compute_alg {dtype : ComputeDType} {shape : TileShape}
    (e : Op dtype.eraseDType shape) :
    (ComputeExpr.compute (ComputeOp.alg dtype e)).toAlgorithm? = Except.ok e := rfl

set_option maxRecDepth 100000 in
/-- **General** `drop 19` of the dimension-parameterized surface = `forRangeDyn`
over `attnLoopBodyG`, then `drop 20`. By `rfl` (statement structure is
dim-independent). -/
theorem attnLoopBodyG_check (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
        stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
        BLOCK_M BLOCK_N FloatDType.fp16).toAlgKernel.body.drop 19
      = Stmt.forRangeDyn "start_n" (Op.ref .nat [] "lo") (Op.ref .nat [] "hi")
          (Op.constNat BLOCK_N) (attnLoopBodyG B0 BLOCK_M BLOCK_N BLOCK_DMODEL stride_b0m)
        :: (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
            stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
            stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
            stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
            BLOCK_M BLOCK_N FloatDType.fp16).toAlgKernel.body.drop 20 := by
  simp only [attention_kernel_fwd_kernel_aligned_surface, ComputeKernel.toAlgKernel_mk,
    ComputeStmt.listToAlgorithm?_cons_assign_alg, ComputeStmt.listToAlgorithm?_cons_assign_compute,
    ComputeStmt.listToAlgorithm?_cons_forRangeDyn, ComputeStmt.listToAlgorithm?_cons_store_alg,
    ComputeStmt.listToAlgorithm?_nil, computeExpr_toAlgorithm?_compute_alg,
    ComputeExpr.toAlgorithm?_compute_full_alg, ComputeExpr.toAlgorithm?_alg,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?_full_alg, computeOp_toAlgorithm?_alg,
    List.drop_succ_cons, List.drop_zero, attnLoopBodyG]
  rfl

/-- The lowered 3-statement post-loop (`acc /= l_i[:, None]`, `O_block_ptr`,
masked-free block store to `Out`). -/
def attnPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .real [64, 128] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "l_i"))),
    Stmt.assign .blockPtr [64, 128] "O_block_ptr"
      (Op.makeBlockPtrDynOffsets Out (Op.ref .nat [] "q_offset") [128, 128] [64, 128]
        [128, 1] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 64), Op.constNat 0]),
    Stmt.store .fp16 [64, 128] (.blockPtr (Op.ref .blockPtr [64, 128] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [64, 128] "acc")) .none ]

/-- The Python-shape lowered body `drop 20` is exactly `attnPostLoop`. By `rfl`. -/
theorem attnPostLoop_check (Q K V B0 Out : RegionName) :
    (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.drop 20 = attnPostLoop Out := by
  rfl

/-- The lowered 3-statement post-loop, dimension-parameterized over
`BLOCK_M BLOCK_DMODEL N_CTX stride_om stride_on`. Specializes to `attnPostLoop`
at the Python test shape. -/
def attnPostLoopG (Out : RegionName)
    (BLOCK_M BLOCK_DMODEL N_CTX stride_om stride_on : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))),
    Stmt.assign .blockPtr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.makeBlockPtrDynOffsets Out (Op.ref .nat [] "q_offset") [N_CTX, BLOCK_DMODEL] [BLOCK_M, BLOCK_DMODEL]
        [stride_om, stride_on] [Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M), Op.constNat 0]),
    Stmt.store .fp16 [BLOCK_M, BLOCK_DMODEL] (.blockPtr (Op.ref .blockPtr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")) .none ]

/-- **General** `drop 20` of the dimension-parameterized surface = `attnPostLoopG`. -/
theorem attnPostLoopG_check (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
        stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
        BLOCK_M BLOCK_N FloatDType.fp16).toAlgKernel.body.drop 20
      = attnPostLoopG Out BLOCK_M BLOCK_DMODEL N_CTX stride_om stride_on := by
  simp only [attention_kernel_fwd_kernel_aligned_surface, ComputeKernel.toAlgKernel_mk,
    ComputeStmt.listToAlgorithm?_cons_assign_alg, ComputeStmt.listToAlgorithm?_cons_assign_compute,
    ComputeStmt.listToAlgorithm?_cons_forRangeDyn, ComputeStmt.listToAlgorithm?_cons_store_alg,
    ComputeStmt.listToAlgorithm?_nil, computeExpr_toAlgorithm?_compute_alg,
    ComputeExpr.toAlgorithm?_compute_full_alg, ComputeExpr.toAlgorithm?_alg,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?_full_alg, computeOp_toAlgorithm?_alg,
    List.drop_succ_cons, List.drop_zero, attnPostLoopG]
  rfl

/-! ### Loop-body op-eval recipes (Python shape) -/

/-- **`qk = fp16-zeros + dot(q, k)`** (L2-L3 collapsed): the fp16 `tl.zeros` is
`0`, so after the dot-add `qk` is `castFloat(0) + dot(castFloat q, k)`. -/
theorem qk_dot_op_eval (s : BlockState) (qtile : Tile .fp16 [64, 128]) (ktile : Tile .real [128, 64])
    (hqk : s.regs .fp16 [64, 64] "qk" = some ⟨fun _ : TileIndex [64, 64] =>
        FloatDType.real.cast FloatDType.fp16 (some 0)⟩)
    (hq : s.regs .fp16 [64, 128] "q" = some qtile)
    (hk : s.regs .real [128, 64] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 64] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q"))
          (Op.ref .real [128, 64] "k"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [64, 128]) ktile)) := by
  have hczA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 64] "qk")) s
      = some (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [64, 64]) := by
    rw [evalOp_castFloat]; simp [hqk]
  have hcz : @evalOp TileDType.real [64, 64] (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 64] "qk")) s
      = some (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [64, 64]) := hczA
  have hcbA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [64, 128]) := by
    rw [evalOp_castFloat]; simp [hq]
  have hcb2 : @evalOp TileDType.real [64, 128] (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [64, 128]) := hcbA
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q")) (Op.ref .real [128, 64] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [64, 128]) ktile) := by
    rw [evalOp_dot]; simp [hcb2, hk]
  have hdotN2 : @evalOp TileDType.real [64, 64]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q")) (Op.ref .real [128, 64] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [64, 128]) ktile) := hdotN
  rw [evalOp_add, hcz, hdotN2]; rfl

/-- **General** `qk = fp16-zeros + dot(q, k)` (dimension-parameterized). -/
theorem qk_dot_op_evalG (s : BlockState) (BM BN HEAD : Nat)
    (qtile : Tile .fp16 [BM, HEAD]) (ktile : Tile .real [HEAD, BN])
    (hqk : s.regs .fp16 [BM, BN] "qk" = some ⟨fun _ : TileIndex [BM, BN] =>
        FloatDType.real.cast FloatDType.fp16 (some 0)⟩)
    (hq : s.regs .fp16 [BM, HEAD] "q" = some qtile)
    (hk : s.regs .real [HEAD, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "qk"))
        (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, HEAD] "q"))
          (Op.ref .real [HEAD, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [BM, BN] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, HEAD]) ktile)) := by
  have hczA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "qk")) s
      = some (⟨fun _ : TileIndex [BM, BN] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [BM, BN]) := by
    rw [evalOp_castFloat]; simp [hqk]
  have hcz : @evalOp TileDType.real [BM, BN] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, BN] "qk")) s
      = some (⟨fun _ : TileIndex [BM, BN] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩ : Tile .real [BM, BN]) := hczA
  have hcbA : evalOp (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, HEAD] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, HEAD]) := by
    rw [evalOp_castFloat]; simp [hq]
  have hcb2 : @evalOp TileDType.real [BM, HEAD] (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, HEAD] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, HEAD]) := hcbA
  have hdotN : evalOp (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, HEAD] "q")) (Op.ref .real [HEAD, BN] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, HEAD]) ktile) := by
    rw [evalOp_dot]; simp [hcb2, hk]
  have hdotN2 : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BM, HEAD] "q")) (Op.ref .real [HEAD, BN] "k")) s
      = some (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real (qtile.data i)⟩ : Tile .real [BM, HEAD]) ktile) := hdotN
  rw [evalOp_add, hcz, hdotN2]; rfl

/-- **`qk += (b0 + b1)·log2e`** (L5): adds the additive relative-position bias. -/
theorem qk_bias_op_eval (s : BlockState) (qktile : Tile .real [64, 64])
    (b0tile : Tile .real [64, 1]) (b1tile : Tile .real [64, 64])
    (hqk : s.regs .real [64, 64] "qk" = some qktile)
    (hb0 : s.regs .real [64, 1] "b0" = some b0tile)
    (hb1 : s.regs .real [64, 64] "b1" = some b1tile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [64, 64] "qk")
        (Op.mul .real Broadcast.scalarR
          (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.ref .real [64, 1] "b0") (Op.ref .real [64, 64] "b1")) (Op.const 1.44269504))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qktile
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0tile b1tile)
            (Tile.scalar (some 1.44269504)))) := by
  rw [evalOp_add]
  simp only [evalOp_mul, evalOp_add, evalOp_const, evalOp_ref, hqk, hb0, hb1,
    Option.bind_eq_bind, Option.bind_some]

/-- **General** `qk += (b0 + b1)·log2e` (dimension-parameterized). -/
theorem qk_bias_op_evalG (s : BlockState) (BM BN : Nat) (qktile : Tile .real [BM, BN])
    (b0tile : Tile .real [BM, 1]) (b1tile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hb0 : s.regs .real [BM, 1] "b0" = some b0tile)
    (hb1 : s.regs .real [BM, BN] "b1" = some b1tile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.mul .real Broadcast.scalarR
          (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.ref .real [BM, 1] "b0") (Op.ref .real [BM, BN] "b1")) (Op.const 1.44269504))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qktile
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0tile b1tile)
            (Tile.scalar (some 1.44269504)))) := by
  rw [evalOp_add]
  simp only [evalOp_mul, evalOp_add, evalOp_const, evalOp_ref, hqk, hb0, hb1,
    Option.bind_eq_bind, Option.bind_some]

/-- `evalOp` helper for `Op.floorDiv` (no dedicated `@[simp]` form). -/
theorem evalOp_floorDiv' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

/-- **`b0` per-block bias load** (L4): resolves the `B0 + b_offset + (...)[:, None]
+ start_n // BLOCK_N` pointer to the per-row read at `b_offset + (start_m·BLOCK_M
+ r)·stride_b0m + start_n // BLOCK_N` (= `b0Val`'s address at `c = start_n /
BLOCK_N`). -/
theorem load_b0_eval (s : BlockState) (B0 : RegionName)
    (BLOCK_M stride_b0m smbm boff snv : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (hbo : s.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar snv))
    (hm : s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))) :
    evalOp ((Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))))
        .none : Op .real [BLOCK_M, 1])) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          some (s.readMem B0
            (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / 64))⟩ : Tile .real [BLOCK_M, 1]) := by
  rw [evalOp_load_region_none]
  have hoff : evalOp ((Op.add .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
          (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
              (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
        (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 64))) : Op .nat [BLOCK_M, 1]) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / 64⟩ : Tile .nat [BLOCK_M, 1]) := by
    conv_lhs => unfold evalOp
    -- outer add scalarR: arg1 = add scalarL, arg2 = floorDiv
    conv_lhs => unfold evalOp  -- arg1 add scalarL
    conv_lhs => unfold evalOp  -- expandDim
    conv_lhs => unfold evalOp  -- inner mul scalarR
    conv_lhs => unfold evalOp  -- inner add scalarL
    conv_lhs => unfold evalOp  -- inner mul nil
    rw [evalOp_ref, hsm]
    simp only [evalOp_ref, evalOp_constNat, hbo, hsn, hm, Option.bind, Option.map]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul,
      IntegralDType.floorDiv]
  rw [hoff]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [BlockState.readMemValue_real, Region.cast_id]

/-- **General** `b0` per-block bias load (dimension-parameterized over `BLOCK_N`). -/
theorem load_b0_evalG (s : BlockState) (B0 : RegionName)
    (BLOCK_M BLOCK_N stride_b0m smbm boff snv : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (hbo : s.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : s.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar snv))
    (hm : s.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))) :
    evalOp ((Op.load .real (.region B0
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N))))
        .none : Op .real [BLOCK_M, 1])) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          some (s.readMem B0
            (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / BLOCK_N))⟩ : Tile .real [BLOCK_M, 1]) := by
  rw [evalOp_load_region_none]
  have hoff : evalOp ((Op.add .nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
          (Op.expandDim ⟨1, hax⟩ (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
              (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
        (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N))) : Op .nat [BLOCK_M, 1]) s
      = some (⟨fun idx : TileIndex [BLOCK_M, 1] =>
          boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + snv / BLOCK_N⟩ : Tile .nat [BLOCK_M, 1]) := by
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    conv_lhs => unfold evalOp
    rw [evalOp_ref, hsm]
    simp only [evalOp_ref, evalOp_constNat, hbo, hsn, hm, Option.bind, Option.map]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul,
      IntegralDType.floorDiv]
  rw [hoff]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [BlockState.readMemValue_real, Region.cast_id]

/-! ### Loop-body execution chain (`attnLoopBody_steps`)

Steps the 15-statement `attnLoopBody` one block at a time, exposing the symbolic
`qk`/`m_i_new`/`alpha`/`p`/`acc`/`l_i` tiles so the banked per-cell bridges
(`mijg_eq`/`alphag_eq`/`lig_eq`/`accg_eq`) advance the invariant by one block.
Mirrors `flashLoopBody_steps`, but the kernel's `qk = dot + (b0+b1)·log2e` bias
augmentation replaces flash's causal mask, and the registers seed `lPgK`
(`l_i = 0`). -/

/-- **`m_i = m_i_new`** op-eval recipe (plain register copy). -/
theorem ak_ref_op_eval (s : BlockState) {dtype : TileDType} {shape : TileShape}
    (name : RegName) (v : Tile dtype shape) (h : s.regs dtype shape name = some v) :
    evalOp (Op.ref dtype shape name) s = some v := by rw [evalOp_ref, h]

/-- **`m_i_new = maximum(m_i, tl.max(qk, 1))`** op-eval recipe. -/
theorem ak_mnew_op_eval (s : BlockState) (BM BN : Nat)
    (mtile : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmax : s.regs .real [BM] "m_i" = some mtile)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM,BN].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")))
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) := by
  have hrmaxN : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm
  have hrmax : @evalOp TileDType.real [BM]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "qk")) s = some rmaxT := hrmaxN
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmax, hrmax, Option.bind_eq_bind, Option.bind_some]

/-- **`alpha = tl.math.exp2(m_i - m_i_new)`** op-eval recipe. -/
theorem ak_alpha_op_eval (s : BlockState) (BM : Nat) (mtile mnewtile : Tile .real [BM])
    (hmax : s.regs .real [BM] "m_i" = some mtile) (hmnew : s.regs .real [BM] "m_i_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewtile)) := by
  rw [evalOp_exp2', evalOp_sub]; simp [evalOp_ref, hmax, hmnew]

/-- **`p = tl.math.exp2(qk - m_i_new[:, None])`** op-eval recipe. -/
theorem ak_p_op_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mnewtile : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile) (hmnew : s.regs .real [BM] "m_i_new" = some mnewtile) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, BN] "qk")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")))) s
      = some (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qktile
          (Tile.expandDim ⟨1, hax⟩ mnewtile))) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")) s
      = some (Tile.expandDim ⟨1, hax⟩ mnewtile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmnew
  rw [evalOp_exp2', evalOp_sub]
  simp only [evalOp_ref, hqk, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`acc *= alpha[:, None]`** op-eval recipe. -/
theorem ak_accmul_op_eval (s : BlockState) (BM DIM : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM, DIM]) (atile : Tile .real [BM])
    (hacc : s.regs .real [BM, DIM] "acc" = some acctile) (ha : s.regs .real [BM] "alpha" = some atile) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil)) (Op.ref .real [BM, DIM] "acc")
        (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile
          (Tile.expandDim ⟨1, hax⟩ atile)) := by
  have hexp2 : @evalOp TileDType.real [BM, 1] (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ atile) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ ha
  rw [evalOp_mul]; simp only [evalOp_ref, hacc, hexp2, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`acc += tl.dot((p).to(fp16), v)`** op-eval recipe (fp16 round-trip identity). -/
theorem ak_accadd_op_eval (s : BlockState) (BM BN DIM : Nat)
    (acc1tile : Tile .real [BM, DIM]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, DIM])
    (hacc : s.regs .real [BM, DIM] "acc" = some acc1tile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, DIM] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) (Op.ref .real [BM, DIM] "acc")
        (Op.dot (batch := []) (Op.castFloat .fp16 .real
          (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc1tile
          (Tile.dot [] ptile vtile)) := by
  have hcastInner : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ : Tile .fp16 [BM, BN]) := by
    rw [evalOp_castFloat]; simp [evalOp_ref, hp]
  have hpf16 : evalOp (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) s = some ptile := by
    rw [evalOp_castFloat, hcastInner]
    refine congrArg some ?_; ext i; simp [FloatDType.cast]
  have hpf16' : @evalOp TileDType.real [BM, BN]
      (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) s = some ptile := hpf16
  have hdotN : evalOp
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    rw [evalOp_dot]; simp [hpf16', evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, DIM]
      (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.castFloat .real .fp16 (Op.ref .real [BM, BN] "p"))) (Op.ref .real [BN, DIM] "v")) s
      = some (Tile.dot [] ptile vtile) := hdotN
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

/-- **`l_i = l_i * alpha + tl.sum(p, 1)`** op-eval recipe. -/
theorem ak_li_op_eval (s : BlockState) (BM BN : Nat)
    (ltile atile : Tile .real [BM]) (ptile : Tile .real [BM, BN])
    (hl : s.regs .real [BM] "l_i" = some ltile) (ha : s.regs .real [BM] "alpha" = some atile)
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile atile)
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile)) := by
  have hsumN0 : evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := by
    rw [evalOp_reduceSum]; simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl
  have hsumN : @evalOp TileDType.real [BM]
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BM,BN].length) Bool.false (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM,BN].length) ptile) := hsumN0
  rw [evalOp_add]
  simp only [evalOp_mul, evalOp_ref, hl, ha, hsumN, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
theorem attnLoopBody_steps (B0 : RegionName) (sin : BlockState) (SN : Nat)
    (gm : Fin 64 → Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow kcols vrows : Nat)
    (boff smbm : Nat)
    (qtile : Tile .fp16 [64, 128]) (mtile ltile : Tile .real [64])
    (acctile : Tile .real [64, 128]) (ktile : Tile .real [128, 64]) (vtile : Tile .real [64, 128])
    (b1tile : Tile .real [64, 64])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hbo : sin.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hm : sin.regs .nat [64] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin 64 => r.val)))
    (hmax : sin.regs .real [64] "m_i" = some mtile)
    (hl : sin.regs .real [64] "l_i" = some ltile)
    (hacc : sin.regs .real [64, 128] "acc" = some acctile)
    (hq : sin.regs .fp16 [64, 128] "q" = some qtile)
    (hb1 : sin.regs .real [64, 64] "b1" = some b1tile)
    (hKp : sin.regs .blockPtr [128, 64] "K_block_ptr" = some
      (⟨fun _ : TileIndex [128, 64] =>
        { region := Kreg, baseOffset := kbase, parentShape := [128, kcols],
          blockShape := [128, 64], strides := [1, 128], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [64, 128] "V_block_ptr" = some
      (⟨fun _ : TileIndex [64, 128] =>
        { region := Vreg, baseOffset := vbase, parentShape := [vrows, 128],
          blockShape := [64, 128], strides := [128, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [128, 64],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * 128)))
    (hvload : ∀ idx : TileIndex [64, 128],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * 128 + idx.2.1.val * 1)))
    (qoffV : Nat) (hqo : sin.regs .nat [] "q_offset" = some (Tile.scalar qoffV))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (attnLoopBody B0) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .fp16 [64, 128] "q" = some qtile
      ∧ sF.regs .real [64, 64] "b1" = some b1tile
      ∧ sF.regs .nat [] "b_offset" = some (Tile.scalar boff)
      ∧ sF.regs .nat [] "start_m" = some (Tile.scalar smbm)
      ∧ sF.regs .nat [] "q_offset" = some (Tile.scalar qoffV)
      ∧ sF.regs .nat [64] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin 64 => r.val))
      ∧ sF.regs .blockPtr [128, 64] "K_block_ptr" = some
          (⟨fun _ : TileIndex [128, 64] =>
            { region := Kreg, baseOffset := kbase, parentShape := [128, kcols],
              blockShape := [128, 64], strides := [1, 128], offsets := [0, kcol + 64] }⟩)
      ∧ sF.regs .blockPtr [64, 128] "V_block_ptr" = some
          (⟨fun _ : TileIndex [64, 128] =>
            { region := Vreg, baseOffset := vbase, parentShape := [vrows, 128],
              blockShape := [64, 128], strides := [128, 1], offsets := [vrow + 64, 0] }⟩)
      ∧ ∃ (qkT : Tile .real [64, 64]) (rmaxT mnewT alphaT : Tile .real [64])
            (pT : Tile .real [64, 64]),
          (∀ i : Fin 64, ∀ j : Fin 64,
            qkT.data (i, j, PUnit.unit)
              = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
                    (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [64, 128]) ktile))
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                      (⟨fun idx : TileIndex [64, 1] =>
                        some (sin.readMem B0 (boff + (smbm * 64 + idx.1.val) * 128 + SN / 64))⟩)
                      b1tile)
                    (Tile.scalar (some 1.44269504)))).data (i, j, PUnit.unit))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some rmaxT
          ∧ mnewT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT))
          ∧ sF.regs .real [64] "m_i" = some mnewT
          ∧ sF.regs .real [64, 128] "acc" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
              (Tile.dot [] pT vtile))
          ∧ sF.regs .real [64] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
              (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [64, 64].length) pT)) := by
  -- the qk tile after L2–L5 (fp16-zeros + dot + bias)
  set b0T : Tile .real [64, 1] :=
    ⟨fun idx : TileIndex [64, 1] => some (sin.readMem B0 (boff + (smbm * 64 + idx.1.val) * 128 + SN / 64))⟩ with hb0T
  set qkdotT : Tile .real [64, 64] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
      (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [64, 128]) ktile) with hqkdotT
  set qkT : Tile .real [64, 64] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile)
        (Tile.scalar (some 1.44269504))) with hqkT
  have hqkData : ∀ i : Fin 64, ∀ j : Fin 64, qkT.data (i, j, PUnit.unit)
      = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile)
            (Tile.scalar (some 1.44269504)))).data (i, j, PUnit.unit) := fun _ _ => rfl
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [64, 64].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [64, 64] (⟨1, by simp⟩ : Fin [64, 64].length) from by decide)]⟩
  set mnewT : Tile .real [64] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmnew
  set alphaT : Tile .real [64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT) with halpha
  set pT : Tile .real [64, 64] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT)) with hpT
  unfold attnLoopBody
  -- L0: k = load K_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [128, 64] "K_block_ptr") []) .none) sin
        = some ktile from by
      rw [load_blockPtr_K_eval Kreg kbase 128 (kcols) 128 64 1 128 kcol
        (Op.ref .blockPtr [128, 64] "K_block_ptr") sin (by rw [evalOp_ref]; simp [hKp])]
      refine congrArg some ?_; ext idx; rw [hkload idx]))]
  -- L1: v = load V_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 128] "V_block_ptr") []) .none) _
        = some vtile from by
      rw [load_blockPtr_Q_eval Vreg vbase (vrows) 128 64 128 128 1 vrow
        (Op.ref .blockPtr [64, 128] "V_block_ptr") _ (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hVp])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem]; rw [hvload idx]))]
  -- L2: qk = fp16 zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show @evalOp TileDType.fp16 [64, 64] (Op.full [64, 64] (Op.castFloat .real .fp16 (Op.const 0)))
          ((sin.setReg "k" .real [128, 64] ktile).setReg "v" .real [64, 128] vtile)
        = some (⟨fun _ : TileIndex [64, 64] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩ : Tile .fp16 [64, 64]) from by
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      simp only [Option.bind_eq_bind]
      rfl))]
  -- L3: qk += dot(q, k)  →  qkdotT
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 64] "qk"))
          (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [64, 128] "q"))
            (Op.ref .real [128, 64] "k"))) _ = some qkdotT from by
      rw [qk_dot_op_eval _ qtile ktile (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]))]
  -- L4: b0 = load B0[..]  →  b0T
  rw [stepStmts.cons_some
    (stepStmt_assign_eq_some (v := b0T)
      (by
        rw [load_b0_eval
          ((((sin.setReg "k" .real [128, 64] ktile).setReg "v" .real [64, 128] vtile).setReg "qk"
              .fp16 [64, 64] (⟨fun _ : TileIndex [64, 64] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩)).setReg
            "qk" .real [64, 64] qkdotT)
          B0 64 128 smbm boff SN attnLoopBody._proof_1
          (by simp [BlockState.setReg_ne_name, hbo]) (by simp [BlockState.setReg_ne_name, hsm])
          (by simp [BlockState.setReg_ne_name, hsn]) (by simp [BlockState.setReg_ne_name, hm])]
        refine congrArg some ?_; ext idx
        simp only [hb0T, BlockState.setReg_readMem] : _ = some b0T))]
  -- L5: qk += (b0 + b1)·log2e  →  qkT
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [64, 64] "qk")
          (Op.mul .real Broadcast.scalarR
            (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
              (Op.ref .real [64, 1] "b0") (Op.ref .real [64, 64] "b1")) (Op.const 1.44269504))) _
        = some qkT from by
      rw [qk_bias_op_eval _ qkdotT b0T b1tile (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hb1])]))]
  -- L6: m_i_new = maximum(m_i, reduceMax qk 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_mnew_op_eval _ 64 64 mtile qkT rmaxT
      (by simp [BlockState.setReg_ne_name, hmax]) (by rw [BlockState.setReg_same]) hrm))]
  -- L7: alpha = exp2(m_i - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_alpha_op_eval _ 64 mtile mnewT
      (by simp [BlockState.setReg_ne_name, hmax]) (by simp [BlockState.setReg_same, hmnew])))]
  -- L8: p = exp2(qk - m_i_new[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_p_op_eval _ 64 64 (by simp) qkT mnewT
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name, hmnew])))]
  -- L9: acc *= alpha[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accmul_op_eval _ 64 128 (by simp) acctile alphaT
      (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_same, halpha])))]
  -- L10: acc += dot((p).to fp16).to real, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accadd_op_eval _ 64 64 128
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile
      (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hpT])
      (by simp [BlockState.setReg_ne_name])))]
  -- L11: l_i = l_i * alpha + sum p 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_li_op_eval _ 64 64 ltile alphaT pT
      (by simp [BlockState.setReg_ne_name, hl]) (by simp [BlockState.setReg_ne_name, halpha])
      (by simp [BlockState.setReg_ne_name, hpT])))]
  -- L12: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_ref_op_eval _ "m_i_new" mnewT (by rfl)))]
  -- L13: K_block_ptr = advance(K_block_ptr, [0, 64])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_col_eval _ Kreg kbase 128 (kcols) 128 64 1 128 kcol 64 "K_block_ptr"
      (by simp [BlockState.setReg_ne_name, hKp])))]
  -- L14: V_block_ptr = advance(V_block_ptr, [64, 0])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_row_eval _ Vreg vbase (vrows) 128 64 128 128 1 vrow 64 "V_block_ptr"
      (by simp [BlockState.setReg_ne_name, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mnewT, alphaT, pT,
    hqkData, hrm, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hb1]
  · simp [BlockState.setReg_ne_name, hbo]
  · simp [BlockState.setReg_ne_name, hsm]
  · simp [BlockState.setReg_ne_name, hqo]
  · simp [BlockState.setReg_ne_name, hm]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **General** loop-body execution chain (dimension-parameterized over
`BLOCK_M BLOCK_N HEAD`, with contiguous K/V layout `stride_kk=1, stride_kn=HEAD,
stride_vk=HEAD, stride_vn=1` baked by `kFlat`/`vFlat`). -/
theorem attnLoopBody_stepsG (B0 : RegionName) (BLOCK_M BLOCK_N HEAD stride_b0m : Nat)
    (hBN : 0 < BLOCK_N) (sin : BlockState) (SN : Nat)
    (Kreg Vreg : RegionName) (kbase vbase kcol vrow kcols vrows : Nat)
    (boff smbm : Nat)
    (qtile : Tile .fp16 [BLOCK_M, HEAD]) (mtile ltile : Tile .real [BLOCK_M])
    (acctile : Tile .real [BLOCK_M, HEAD]) (ktile : Tile .real [HEAD, BLOCK_N]) (vtile : Tile .real [BLOCK_N, HEAD])
    (b1tile : Tile .real [BLOCK_M, BLOCK_N])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hbo : sin.regs .nat [] "b_offset" = some (Tile.scalar boff))
    (hsm : sin.regs .nat [] "start_m" = some (Tile.scalar smbm))
    (hm : sin.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val)))
    (hmax : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hl : sin.regs .real [BLOCK_M] "l_i" = some ltile)
    (hacc : sin.regs .real [BLOCK_M, HEAD] "acc" = some acctile)
    (hq : sin.regs .fp16 [BLOCK_M, HEAD] "q" = some qtile)
    (hb1 : sin.regs .real [BLOCK_M, BLOCK_N] "b1" = some b1tile)
    (hKp : sin.regs .blockPtr [HEAD, BLOCK_N] "K_block_ptr" = some
      (⟨fun _ : TileIndex [HEAD, BLOCK_N] =>
        { region := Kreg, baseOffset := kbase, parentShape := [HEAD, kcols],
          blockShape := [HEAD, BLOCK_N], strides := [1, HEAD], offsets := [0, kcol] }⟩))
    (hVp : sin.regs .blockPtr [BLOCK_N, HEAD] "V_block_ptr" = some
      (⟨fun _ : TileIndex [BLOCK_N, HEAD] =>
        { region := Vreg, baseOffset := vbase, parentShape := [vrows, HEAD],
          blockShape := [BLOCK_N, HEAD], strides := [HEAD, 1], offsets := [vrow, 0] }⟩))
    (hkload : ∀ idx : TileIndex [HEAD, BLOCK_N],
      ktile.data idx = some (sin.readMem Kreg (kbase + idx.1.val * 1 + (kcol + idx.2.1.val) * HEAD)))
    (hvload : ∀ idx : TileIndex [BLOCK_N, HEAD],
      vtile.data idx = some (sin.readMem Vreg (vbase + (vrow + idx.1.val) * HEAD + idx.2.1.val * 1)))
    (qoffV : Nat) (hqo : sin.regs .nat [] "q_offset" = some (Tile.scalar qoffV))
    (hundef : ∀ rg o, sin.undef rg o = 0) :
    ∃ sF, stepStmts (attnLoopBodyG B0 BLOCK_M BLOCK_N HEAD stride_b0m) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = 0)
      ∧ sF.regs .fp16 [BLOCK_M, HEAD] "q" = some qtile
      ∧ sF.regs .real [BLOCK_M, BLOCK_N] "b1" = some b1tile
      ∧ sF.regs .nat [] "b_offset" = some (Tile.scalar boff)
      ∧ sF.regs .nat [] "start_m" = some (Tile.scalar smbm)
      ∧ sF.regs .nat [] "q_offset" = some (Tile.scalar qoffV)
      ∧ sF.regs .nat [BLOCK_M] "b_ptr_offsets_m" = some (Tile.vec (fun r : Fin BLOCK_M => r.val))
      ∧ sF.regs .blockPtr [HEAD, BLOCK_N] "K_block_ptr" = some
          (⟨fun _ : TileIndex [HEAD, BLOCK_N] =>
            { region := Kreg, baseOffset := kbase, parentShape := [HEAD, kcols],
              blockShape := [HEAD, BLOCK_N], strides := [1, HEAD], offsets := [0, kcol + BLOCK_N] }⟩)
      ∧ sF.regs .blockPtr [BLOCK_N, HEAD] "V_block_ptr" = some
          (⟨fun _ : TileIndex [BLOCK_N, HEAD] =>
            { region := Vreg, baseOffset := vbase, parentShape := [vrows, HEAD],
              blockShape := [BLOCK_N, HEAD], strides := [HEAD, 1], offsets := [vrow + BLOCK_N, 0] }⟩)
      ∧ ∃ (qkT : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT mnewT alphaT : Tile .real [BLOCK_M])
            (pT : Tile .real [BLOCK_M, BLOCK_N]),
          (∀ i : Fin BLOCK_M, ∀ j : Fin BLOCK_N,
            qkT.data (i, j, PUnit.unit)
              = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
                    (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [BLOCK_M, HEAD]) ktile))
                  (Tile.bop NumericDType.real.mul Broadcast.scalarR
                    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                      (⟨fun idx : TileIndex [BLOCK_M, 1] =>
                        some (sin.readMem B0 (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + SN / BLOCK_N))⟩)
                      b1tile)
                    (Tile.scalar (some 1.44269504)))).data (i, j, PUnit.unit))
          ∧ Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some rmaxT
          ∧ mnewT = Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT
          ∧ alphaT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT)
          ∧ pT = Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT))
          ∧ sF.regs .real [BLOCK_M] "m_i" = some mnewT
          ∧ sF.regs .real [BLOCK_M, HEAD] "acc" = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
              (Tile.dot [] pT vtile))
          ∧ sF.regs .real [BLOCK_M] "l_i" = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
              (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) ltile alphaT)
              (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT)) := by
  set b0T : Tile .real [BLOCK_M, 1] :=
    ⟨fun idx : TileIndex [BLOCK_M, 1] => some (sin.readMem B0 (boff + (smbm * BLOCK_M + idx.1.val) * stride_b0m + SN / BLOCK_N))⟩ with hb0T
  set qkdotT : Tile .real [BLOCK_M, BLOCK_N] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
      (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [BLOCK_M, HEAD]) ktile) with hqkdotT
  set qkT : Tile .real [BLOCK_M, BLOCK_N] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile)
        (Tile.scalar (some 1.44269504))) with hqkT
  have hqkData : ∀ i : Fin BLOCK_M, ∀ j : Fin BLOCK_N, qkT.data (i, j, PUnit.unit)
      = (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkdotT
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil)) b0T b1tile)
            (Tile.scalar (some 1.44269504)))).data (i, j, PUnit.unit) := fun _ _ => rfl
  obtain ⟨rmaxT, hrm⟩ : ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkT = some t :=
    ⟨_, by unfold Tile.reduceMaxDrop; rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]⟩
  set mnewT : Tile .real [BLOCK_M] := Tile.select (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT with hmnew
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mnewT) with halpha
  set pT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkT (Tile.expandDim ⟨1, by simp⟩ mnewT)) with hpT
  unfold attnLoopBodyG
  -- L0: k = load K_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [HEAD, BLOCK_N] "K_block_ptr") []) .none) sin
        = some ktile from by
      rw [load_blockPtr_K_eval Kreg kbase HEAD (kcols) HEAD BLOCK_N 1 HEAD kcol
        (Op.ref .blockPtr [HEAD, BLOCK_N] "K_block_ptr") sin (by rw [evalOp_ref]; simp [hKp])]
      refine congrArg some ?_; ext idx; rw [hkload idx]))]
  -- L1: v = load V_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_N, HEAD] "V_block_ptr") []) .none) _
        = some vtile from by
      rw [load_blockPtr_Q_eval Vreg vbase (vrows) HEAD BLOCK_N HEAD HEAD 1 vrow
        (Op.ref .blockPtr [BLOCK_N, HEAD] "V_block_ptr") _ (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hVp])]
      refine congrArg some ?_; ext idx
      simp only [BlockState.setReg_readMem]; rw [hvload idx]))]
  -- L2: qk = fp16 zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show @evalOp TileDType.fp16 [BLOCK_M, BLOCK_N] (Op.full [BLOCK_M, BLOCK_N] (Op.castFloat .real .fp16 (Op.const 0)))
          ((sin.setReg "k" .real [HEAD, BLOCK_N] ktile).setReg "v" .real [BLOCK_N, HEAD] vtile)
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩ : Tile .fp16 [BLOCK_M, BLOCK_N]) from by
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      conv_lhs => unfold evalOp
      simp only [Option.bind_eq_bind]
      rfl))]
  -- L3: qk += dot(q, k)  →  qkdotT
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "qk"))
          (Op.dot (batch := []) (Op.castFloat .fp16 .real (Op.ref .fp16 [BLOCK_M, HEAD] "q"))
            (Op.ref .real [HEAD, BLOCK_N] "k"))) _ = some qkdotT from by
      rw [qk_dot_op_evalG _ BLOCK_M BLOCK_N HEAD qtile ktile (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]))]
  -- L4: b0 = load B0[..]  →  b0T
  rw [stepStmts.cons_some
    (stepStmt_assign_eq_some (v := b0T)
      (by
        rw [load_b0_evalG
          ((((sin.setReg "k" .real [HEAD, BLOCK_N] ktile).setReg "v" .real [BLOCK_N, HEAD] vtile).setReg "qk"
              .fp16 [BLOCK_M, BLOCK_N] (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => FloatDType.real.cast FloatDType.fp16 (some 0)⟩)).setReg
            "qk" .real [BLOCK_M, BLOCK_N] qkdotT)
          B0 BLOCK_M BLOCK_N stride_b0m smbm boff SN (by simp)
          (by simp [BlockState.setReg_ne_name, hbo]) (by simp [BlockState.setReg_ne_name, hsm])
          (by simp [BlockState.setReg_ne_name, hsn]) (by simp [BlockState.setReg_ne_name, hm])]
        refine congrArg some ?_; ext idx
        simp only [hb0T, BlockState.setReg_readMem] : _ = some b0T))]
  -- L5: qk += (b0 + b1)·log2e  →  qkT
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
          (Op.mul .real Broadcast.scalarR
            (Op.add .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
              (Op.ref .real [BLOCK_M, 1] "b0") (Op.ref .real [BLOCK_M, BLOCK_N] "b1")) (Op.const 1.44269504))) _
        = some qkT from by
      rw [qk_bias_op_evalG _ BLOCK_M BLOCK_N qkdotT b0T b1tile (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hb1])]))]
  -- L6: m_i_new = maximum(m_i, reduceMax qk 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_mnew_op_eval _ BLOCK_M BLOCK_N mtile qkT rmaxT
      (by simp [BlockState.setReg_ne_name, hmax]) (by rw [BlockState.setReg_same]) hrm))]
  -- L7: alpha = exp2(m_i - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_alpha_op_eval _ BLOCK_M mtile mnewT
      (by simp [BlockState.setReg_ne_name, hmax]) (by simp [BlockState.setReg_same, hmnew])))]
  -- L8: p = exp2(qk - m_i_new[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_p_op_eval _ BLOCK_M BLOCK_N (by simp) qkT mnewT
      (by simp [BlockState.setReg_ne_name]) (by simp [BlockState.setReg_ne_name, hmnew])))]
  -- L9: acc *= alpha[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accmul_op_eval _ BLOCK_M HEAD (by simp) acctile alphaT
      (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_same, halpha])))]
  -- L10: acc += dot((p).to fp16).to real, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_accadd_op_eval _ BLOCK_M BLOCK_N HEAD
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile
      (by rw [BlockState.setReg_same]) (by simp [BlockState.setReg_ne_name, hpT])
      (by simp [BlockState.setReg_ne_name])))]
  -- L11: l_i = l_i * alpha + sum p 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_li_op_eval _ BLOCK_M BLOCK_N ltile alphaT pT
      (by simp [BlockState.setReg_ne_name, hl]) (by simp [BlockState.setReg_ne_name, halpha])
      (by simp [BlockState.setReg_ne_name, hpT])))]
  -- L12: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ak_ref_op_eval _ "m_i_new" mnewT (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; rw [hmnew])))]
  -- L13: K_block_ptr = advance(K_block_ptr, [0, BLOCK_N])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_col_eval _ Kreg kbase HEAD (kcols) HEAD BLOCK_N 1 HEAD kcol BLOCK_N "K_block_ptr"
      (by simp [BlockState.setReg_ne_name, hKp])))]
  -- L14: V_block_ptr = advance(V_block_ptr, [BLOCK_N, 0])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (advance_row_eval _ Vreg vbase (vrows) HEAD BLOCK_N HEAD HEAD 1 vrow BLOCK_N "V_block_ptr"
      (by simp [BlockState.setReg_ne_name, hVp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, qkT, rmaxT, mnewT, alphaT, pT,
    hqkData, hrm, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef, hundef]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hb1]
  · simp [BlockState.setReg_ne_name, hbo]
  · simp [BlockState.setReg_ne_name, hsm]
  · simp [BlockState.setReg_ne_name, hqo]
  · simp [BlockState.setReg_ne_name, hm]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]

/-- The lowered statements 10–18 of the Python-shape prologue (after the
deterministic `preLoop_prefix`): qk_scale, the Q load + scale + fp16 cast, the
`lo`/`hi` constants, the bias arange/offset prep, and the `b1` region load. -/
def attnPreLoopTail (B0 : RegionName) : List Stmt :=
  [ Stmt.assign .real [] "qk_scale" (Op.mul .real Broadcast.nil (Op.const 0.1) (Op.const 1.44269504)),
    Stmt.assign .real [64, 128] "q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 128] "Q_block_ptr") []) .none),
    Stmt.assign .fp16 [64, 128] "q"
      (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [64, 128] "q") (Op.ref .real [] "qk_scale"))),
    Stmt.assign .nat [] "lo" (Op.constNat 0),
    Stmt.assign .nat [] "hi" (Op.constNat 128),
    Stmt.assign .nat [64] "b_ptr_offsets_m" (Op.arange 64),
    Stmt.assign .nat [] "b_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat 16384)),
    Stmt.assign .nat [64] "b_ptr_offsets_n_1"
      (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange 64) (Op.constNat 64)) (Op.constNat 64)),
    Stmt.assign .real [64, 64] "b1"
      (Op.load .real (.region B0
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 64))
                (Op.ref .nat [64] "b_ptr_offsets_m")) (Op.constNat 128))))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [64] "b_ptr_offsets_n_1"))))
        .none) ]

theorem attnPreLoopTail_check (Q K V B0 Out : RegionName) :
    ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.take 19).drop 10 = attnPreLoopTail B0 := by
  rfl

/-- The lowered prologue tail (statements 10–18), dimension-parameterized over
`sm_scale BLOCK_M BLOCK_N HEAD_DIM N_CTX P_SEQ BIAS_LAST_SIZE stride_b0h stride_b0m`. -/
def attnPreLoopTailG (B0 : RegionName) (sm_scale : ℝ)
    (BLOCK_M BLOCK_N HEAD_DIM N_CTX P_SEQ BIAS_LAST_SIZE stride_b0h stride_b0m : Nat) : List Stmt :=
  [ Stmt.assign .real [] "qk_scale" (Op.mul .real Broadcast.nil (Op.const sm_scale) (Op.const 1.44269504)),
    Stmt.assign .real [BLOCK_M, HEAD_DIM] "q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_M, HEAD_DIM] "Q_block_ptr") []) .none),
    Stmt.assign .fp16 [BLOCK_M, HEAD_DIM] "q"
      (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_M, HEAD_DIM] "q") (Op.ref .real [] "qk_scale"))),
    Stmt.assign .nat [] "lo" (Op.constNat 0),
    Stmt.assign .nat [] "hi" (Op.constNat (N_CTX + P_SEQ)),
    Stmt.assign .nat [BLOCK_M] "b_ptr_offsets_m" (Op.arange BLOCK_M),
    Stmt.assign .nat [] "b_offset" (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat stride_b0h)),
    Stmt.assign .nat [BLOCK_N] "b_ptr_offsets_n_1"
      (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange BLOCK_N) (Op.constNat BIAS_LAST_SIZE)) (Op.constNat BIAS_LAST_SIZE)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "b1"
      (Op.load .real (.region B0
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "b_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
                (Op.ref .nat [BLOCK_M] "b_ptr_offsets_m")) (Op.constNat stride_b0m))))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "b_ptr_offsets_n_1"))))
        .none) ]

theorem attnPreLoopTailG_check (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk stride_vh
        stride_vk stride_vn stride_oh stride_om stride_on stride_b0h
        stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL
        BLOCK_M BLOCK_N FloatDType.fp16).toAlgKernel.body.take 19).drop 10
      = attnPreLoopTailG B0 sm_scale BLOCK_M BLOCK_N BLOCK_DMODEL N_CTX P_SEQ BIAS_LAST_SIZE stride_b0h stride_b0m := by
  simp only [attention_kernel_fwd_kernel_aligned_surface, ComputeKernel.toAlgKernel_mk,
    ComputeStmt.listToAlgorithm?_cons_assign_alg, ComputeStmt.listToAlgorithm?_cons_assign_compute,
    ComputeStmt.listToAlgorithm?_cons_forRangeDyn, ComputeStmt.listToAlgorithm?_cons_store_alg,
    ComputeStmt.listToAlgorithm?_nil, computeExpr_toAlgorithm?_compute_alg,
    ComputeExpr.toAlgorithm?_compute_full_alg, ComputeExpr.toAlgorithm?_alg,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?_full_alg, computeOp_toAlgorithm?_alg,
    List.take_succ_cons, List.take_zero, List.drop_succ_cons, List.drop_zero, attnPreLoopTailG]
  rfl

/-- **`q = (q * qk_scale).to fp16`** op-eval recipe (the pre-scale + fp16 cast). -/
theorem ak_qscale_op_eval (s : BlockState) (qtile : Tile .real [64, 128]) (sc : ℝ)
    (hq : s.regs .real [64, 128] "q" = some qtile)
    (hqs : s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [64, 128] "q") (Op.ref .real [] "qk_scale"))) s
      = some (⟨fun idx : TileIndex [64, 128] =>
          FloatDType.real.cast FloatDType.fp16 ((qtile.data idx).bind (fun x => some (x * sc)))⟩ : Tile .fp16 [64, 128]) := by
  have hmul0 : evalOp
        (Op.mul .real Broadcast.scalarR (Op.ref .real [64, 128] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := by
    rw [evalOp_mul]; simp only [evalOp_ref, hq, hqs, Option.bind_eq_bind, Option.bind_some]
  have hmul : @evalOp FloatDType.real.toTileDType [64, 128]
        (Op.mul .real Broadcast.scalarR (Op.ref .real [64, 128] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := hmul0
  rw [evalOp_castFloat, hmul]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- **General** `q = (q · qk_scale).to fp16` (dimension-parameterized). -/
theorem ak_qscale_op_evalG (s : BlockState) (BM HEAD : Nat) (qtile : Tile .real [BM, HEAD]) (sc : ℝ)
    (hq : s.regs .real [BM, HEAD] "q" = some qtile)
    (hqs : s.regs .real [] "qk_scale" = some (Tile.scalar (some sc))) :
    evalOp (Op.castFloat .real .fp16
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, HEAD] "q") (Op.ref .real [] "qk_scale"))) s
      = some (⟨fun idx : TileIndex [BM, HEAD] =>
          FloatDType.real.cast FloatDType.fp16 ((qtile.data idx).bind (fun x => some (x * sc)))⟩ : Tile .fp16 [BM, HEAD]) := by
  have hmul0 : evalOp
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, HEAD] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := by
    rw [evalOp_mul]; simp only [evalOp_ref, hq, hqs, Option.bind_eq_bind, Option.bind_some]
  have hmul : @evalOp FloatDType.real.toTileDType [BM, HEAD]
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, HEAD] "q") (Op.ref .real [] "qk_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qtile (Tile.scalar (some sc))) := hmul0
  rw [evalOp_castFloat, hmul]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-- `evalOp` helper for `Op.mod` (no dedicated `@[simp]` form). -/
theorem evalOp_mod' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- **`b_ptr_offsets_n_1 = (arange % BIAS) + BIAS`** op-eval recipe. -/
theorem ak_bn1_op_eval (s : BlockState) :
    evalOp (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange 64) (Op.constNat 64)) (Op.constNat 64)) s
      = some (Tile.vec (fun jL : Fin 64 => jL.val % 64 + 64)) := by
  rw [evalOp_add, evalOp_mod']
  simp only [evalOp_arange, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, IntegralDType.nat_mod]

/-- **General** `b_ptr_offsets_n_1 = (arange % BIAS) + BIAS` (parameterized). -/
theorem ak_bn1_op_evalG (s : BlockState) (BLOCK_N BIAS_LAST_SIZE : Nat) :
    evalOp (Op.add .nat Broadcast.scalarR
        (Op.mod .nat Broadcast.scalarR (Op.arange BLOCK_N) (Op.constNat BIAS_LAST_SIZE)) (Op.constNat BIAS_LAST_SIZE)) s
      = some (Tile.vec (fun jL : Fin BLOCK_N => jL.val % BIAS_LAST_SIZE + BIAS_LAST_SIZE)) := by
  rw [evalOp_add, evalOp_mod']
  simp only [evalOp_arange, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, IntegralDType.nat_mod]

/-! ### preLoop — full 19-statement prologue → `attnKernelInvariant … 0` -/

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
theorem preLoop (Q K V B0 Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.take 19) s = some s0
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar 128)
      ∧ attnKernelInvariant s Q K V B0 Out 0.1
          (s.pids 1 * 16384) (s.pids 1 * 16384) (s.pids 1 * 16384)
          64 64 128 0 64 128 1 128 128 1 2 (s.pids 0) 0 s0 := by
  -- decompose body.take 19 = take 10 ++ attnPreLoopTail
  rw [show (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.take 19
      = (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
          16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
          FloatDType.fp16).toAlgKernel.body.take 10 ++ attnPreLoopTail B0 from by
    conv_lhs => rw [← List.take_append_drop 10
      ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
          16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
          FloatDType.fp16).toAlgKernel.body.take 19)]
    rw [List.take_take, attnPreLoopTail_check Q K V B0 Out]
    norm_num]
  -- step the prefix (0-9)
  obtain ⟨s10, hpre, hpids, hstartm, hoffhz, hqoff, hkvoff, hQp, hKp, hVp, hmi, hli, hacc, hundef10, hmem10⟩ :=
    preLoop_prefix Q K V B0 Out 0.1 16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64 FloatDType.fp16 s
  rw [stepStmts.append_some hpre]
  unfold attnPreLoopTail
  -- T0: qk_scale = 0.1 * 1.44269504
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.nil (Op.const 0.1) (Op.const 1.44269504)) s10
        = some (Tile.scalar (some (0.1 * 1.44269504))) from by
      rw [evalOp_mul]; simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- T1: q = load Q_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [64, 128] "Q_block_ptr") []) .none)
          (s10.setReg "qk_scale" .real [] (Tile.scalar (some (0.1 * 1.44269504))))
        = some (⟨fun idx : TileIndex [64, 128] =>
            some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (0.1 * 1.44269504)))).readMem Q
              (s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1))⟩
            : Tile .real [64, 128]) from
      load_blockPtr_Q_eval Q (s.pids 1 * 16384) 128 128 64 128 128 1 (s.pids 0 * 64)
        (Op.ref .blockPtr [64, 128] "Q_block_ptr") _
        (by rw [evalOp_ref]; rw [show (s10.setReg "qk_scale" .real [] (Tile.scalar (some (0.1 * 1.44269504)))).regs .blockPtr [64, 128] "Q_block_ptr" = s10.regs .blockPtr [64, 128] "Q_block_ptr" from by simp [BlockState.setReg_ne_name]]; exact hQp)))]
  -- T2: q = (q * qk_scale).to fp16
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [64, 128] "q") (Op.ref .real [] "qk_scale"))) _
        = some (⟨fun idx : TileIndex [64, 128] =>
            FloatDType.real.cast FloatDType.fp16
              (some (0.1 * 1.44269504 * qRaw s Q (s.pids 1 * 16384) 64 128 (s.pids 0) idx))⟩ : Tile .fp16 [64, 128]) from by
      rw [ak_qscale_op_eval _
        (⟨fun idx : TileIndex [64, 128] =>
          some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (0.1 * 1.44269504)))).readMem Q
            (s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1))⟩ : Tile .real [64, 128])
        (0.1 * 1.44269504)
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
      refine congrArg some ?_; ext idx
      simp only [Option.bind, Option.map, qRaw]
      have hrm : (s10.setReg "qk_scale" .real [] (Tile.scalar (some (0.1 * 1.44269504)))).readMem Q
            (s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1)
          = s.readMem Q (s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val) := by
        rw [BlockState.setReg_readMem]
        unfold BlockState.readMem; rw [hmem10]; congr 1; ring
      rw [hrm]; ring_nf))]
  -- T3: lo = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _))]
  -- T4: hi = 128
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 128 _))]
  -- T5: b_ptr_offsets_m = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange 64 _))]
  -- T6: b_offset = off_hz * 16384
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (evalOp_mul_ref_const _ "off_hz" (s.pids 1) 16384
      (by simp [BlockState.setReg_ne_name, hoffhz])))]
  -- T7: b_ptr_offsets_n_1 = (arange % 64) + 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ak_bn1_op_eval _))]
  -- T8: b1 = load B0[..]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_b1_eval _ B0 64 64 128 64 (s.pids 0) (s.pids 1 * 16384)
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, hstartm])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  -- the invariant at i = 0
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [BlockState.setReg_pids]; exact hpids
  · -- i = c * BLOCK_N
    rfl
  · -- c ≤ nB
    norm_num
  · -- m_i = mPg 0 = ⊥
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hmi]; rfl
  · -- l_i = lPgK 0 = 0
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hli]; rfl
  · -- acc = oPg 0 = 0
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
    rw [hacc]; rfl
  · -- K block ptr offset [0, 0]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hKp]
  · -- V block ptr offset [0, 0]
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hVp]
  · -- q = scaled fp16 cast
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, FloatDType.toTileDType_fp16]
  · -- b1 = b1Val
    rw [BlockState.setReg_same]
    refine congrArg some ?_; ext idx
    simp only [b1Val, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem10]
  · -- b_offset
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · -- start_m
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hstartm
  · -- q_offset
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hqoff
  · -- b_ptr_offsets_m
    simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · -- undef = 0
    intro rg o
    show s10.undef rg o = 0
    rw [hundef10]; exact hundef rg o
  · -- mem = s.mem
    show s10.mem = s.mem
    exact hmem10

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **General** `preLoop` (dimension-parameterized). Steps the 19-statement
prologue to `attnKernelInvariant … 0`. Contiguous Q/K/V layout
(`stride_qm=HEAD, stride_qk=1, stride_kk=1, stride_kn=HEAD, stride_vk=HEAD,
stride_vn=1`) per `qRaw`/`kFlat`/`vFlat`. -/
theorem preLoopG (Q K V B0 Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h
      BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16).toAlgKernel.body.take 19) s = some s0
      ∧ s0.regs .nat [] "lo" = some (Tile.scalar 0)
      ∧ s0.regs .nat [] "hi" = some (Tile.scalar (BLOCK_N * nB + P_SEQ))
      ∧ attnKernelInvariant s Q K V B0 Out sm_scale
          (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h)
          BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m 1 HEAD HEAD 1 nB (s.pids 0) 0 s0 := by
  rw [show (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16).toAlgKernel.body.take 19
      = (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
          stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
          stride_b0h stride_b0m 2 4 (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
          FloatDType.fp16).toAlgKernel.body.take 10
        ++ attnPreLoopTailG B0 sm_scale BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE stride_b0h stride_b0m from by
    conv_lhs => rw [← List.take_append_drop 10
      ((attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
          stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
          stride_b0h stride_b0m 2 4 (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
          FloatDType.fp16).toAlgKernel.body.take 19)]
    rw [List.take_take, attnPreLoopTailG_check Q K V B0 Out]
    norm_num]
  obtain ⟨s10, hpre, hpids, hstartm, hoffhz, hqoff, hkvoff, hQp, hKp, hVp, hmi, hli, hacc, hundef10, hmem10⟩ :=
    preLoop_prefix Q K V B0 Out sm_scale stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
      stride_b0h stride_b0m 2 4 (BLOCK_N * nB) P_SEQ BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N FloatDType.fp16 s
  rw [stepStmts.append_some hpre]
  unfold attnPreLoopTailG
  -- T0: qk_scale = sm_scale * 1.44269504
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.nil (Op.const sm_scale) (Op.const 1.44269504)) s10
        = some (Tile.scalar (some (sm_scale * 1.44269504))) from by
      rw [evalOp_mul]; simp only [evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (congrArg Tile.scalar ?_)
      simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]))]
  -- T1: q = load Q_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (.blockPtr (Op.ref .blockPtr [BLOCK_M, HEAD] "Q_block_ptr") []) .none)
          (s10.setReg "qk_scale" .real [] (Tile.scalar (some (sm_scale * 1.44269504))))
        = some (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
            some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (sm_scale * 1.44269504)))).readMem Q
              (s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1))⟩
            : Tile .real [BLOCK_M, HEAD]) from
      load_blockPtr_Q_eval Q (s.pids 1 * stride_qh) (BLOCK_N * nB) HEAD BLOCK_M HEAD HEAD 1 (s.pids 0 * BLOCK_M)
        (Op.ref .blockPtr [BLOCK_M, HEAD] "Q_block_ptr") _
        (by rw [evalOp_ref]; rw [show (s10.setReg "qk_scale" .real [] (Tile.scalar (some (sm_scale * 1.44269504)))).regs .blockPtr [BLOCK_M, HEAD] "Q_block_ptr" = s10.regs .blockPtr [BLOCK_M, HEAD] "Q_block_ptr" from by simp [BlockState.setReg_ne_name]]; exact hQp)))]
  -- T2: q = (q * qk_scale).to fp16
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat .real .fp16 (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_M, HEAD] "q") (Op.ref .real [] "qk_scale"))) _
        = some (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
            FloatDType.real.cast FloatDType.fp16
              (some (sm_scale * 1.44269504 * qRaw s Q (s.pids 1 * stride_qh) BLOCK_M HEAD (s.pids 0) idx))⟩ : Tile .fp16 [BLOCK_M, HEAD]) from by
      rw [ak_qscale_op_evalG _ BLOCK_M HEAD
        (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
          some ((s10.setReg "qk_scale" .real [] (Tile.scalar (some (sm_scale * 1.44269504)))).readMem Q
            (s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1))⟩ : Tile .real [BLOCK_M, HEAD])
        (sm_scale * 1.44269504)
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])]
      refine congrArg some ?_; ext idx
      simp only [Option.bind, Option.map, qRaw]
      have hrm : (s10.setReg "qk_scale" .real [] (Tile.scalar (some (sm_scale * 1.44269504)))).readMem Q
            (s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1)
          = s.readMem Q (s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val) := by
        rw [BlockState.setReg_readMem]
        unfold BlockState.readMem; rw [hmem10]; congr 1; ring
      rw [hrm]; ring_nf))]
  -- T3: lo = 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _))]
  -- T4: hi = N_CTX + P_SEQ
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat (BLOCK_N * nB + P_SEQ) _))]
  -- T5: b_ptr_offsets_m = arange BLOCK_M
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BLOCK_M _))]
  -- T6: b_offset = off_hz * stride_b0h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (evalOp_mul_ref_const _ "off_hz" (s.pids 1) stride_b0h
      (by simp [BlockState.setReg_ne_name, hoffhz])))]
  -- T7: b_ptr_offsets_n_1 = (arange % BIAS) + BIAS
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ak_bn1_op_evalG _ BLOCK_N BIAS_LAST_SIZE))]
  -- T8: b1 = load B0[..]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_b1_eval _ B0 BLOCK_M BLOCK_N stride_b0m BIAS_LAST_SIZE (s.pids 0) (s.pids 1 * stride_b0h)
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, hstartm])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
      (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  have hzd : (0:Nat) / BLOCK_N = 0 := Nat.zero_div _
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [BlockState.setReg_pids]; exact hpids
  · rw [hzd, Nat.zero_mul]
  · rw [hzd]; omega
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hzd]
    rw [hmi]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hzd]
    rw [hli]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hzd]
    rw [hacc]; rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hzd, Nat.zero_mul, hKp]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, hzd, Nat.zero_mul, hVp]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, and_self, and_true, true_and, FloatDType.toTileDType_fp16]
  · rw [BlockState.setReg_same]
    refine congrArg some ?_; ext idx
    simp only [b1Val, BlockState.setReg_readMem]
    unfold BlockState.readMem; rw [hmem10]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hstartm
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]; exact hqoff
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq, and_self, and_true, true_and]
  · intro rg o
    show s10.undef rg o = 0
    rw [hundef10]; exact hundef rg o
  · show s10.mem = s.mem
    exact hmem10

/-! ### M3 — `attn_qk_cell`: the body's `qk` cell = `fscore` at the global key

The crux. The loop body's symbolic `qk` tile (from `attnLoopBody_steps`:
`fp16-zeros + dot(q, k) + (b0 + b1)·log2e`) evaluated at row `i`, lane `jL`, equals
the genuine per-key score `fscore` at the global key `gkey 64 nB c jL`
(`= c·64 + jL`). The Fin plumbing keeps `jL : Fin 64` (never coerced to ℕ in the
key) so the `b0`/`b1` block/lane reindex (`j/64 = c`, `j%64 = jL`) lines up; all
casts collapse to identity (modeling boundary). -/
set_option maxHeartbeats 1000000 in
theorem attn_qk_cell (s0 : BlockState) (Q K B0 : RegionName) (sm_scale : ℝ)
    (q_offset kv_offset b_offset : Nat) (nB c bcol : Nat) (hc : c < nB)
    (qtile : Tile .fp16 [64, 128]) (ktile : Tile .real [128, 64])
    (b1tile : Tile .real [64, 64])
    (hbcol : bcol = c)
    (hq : qtile = ⟨fun idx : TileIndex [64, 128] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset 64 128 (s0.pids 0) idx))⟩)
    (hk : ∀ idx : TileIndex [128, 64],
        ktile.data idx = some (s0.readMem K (kv_offset + idx.1.val * 1 + (c * 64 + idx.2.1.val) * 128)))
    (hb1 : b1tile = ⟨fun idx : TileIndex [64, 64] =>
        some (b1Val s0 B0 b_offset 64 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
    (i : Fin 64) (jL : Fin 64) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [64, 64] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [64, 128]) ktile))
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (⟨fun idx : TileIndex [64, 1] =>
              some (s0.readMem B0 (b_offset + (s0.pids 0 * 64 + idx.1.val) * 128 + bcol))⟩)
            b1tile)
          (Tile.scalar (some 1.44269504)))).data (i, jL, PUnit.unit)
      = some (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
          64 64 128 (64 * nB) 64 128 (s0.pids 0) i (gkey 64 nB c hc jL)) := by
  -- the dot row = scale · Σ_e qRaw · kFlat
  have hdot : (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [64, 128]) ktile).data (i, jL, PUnit.unit)
      = some (sm_scale * 1.44269504 * Finset.univ.sum (fun e : Fin 128 =>
          qRaw s0 Q q_offset 64 128 (s0.pids 0) (i, e, PUnit.unit)
            * kFlat s0 K kv_offset 128 (64 * nB) e (gkey 64 nB c hc jL))) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·)
            ((⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [64, 128]).data (i, e, PUnit.unit))
            (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ
          (fun e => (some (sm_scale * 1.44269504 * (qRaw s0 Q q_offset 64 128 (s0.pids 0) (i, e, PUnit.unit)
              * kFlat s0 K kv_offset 128 (64 * nB) e (gkey 64 nB c hc jL))) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          simp only [hq, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            FloatDType.real_toWithBot, Option.map₂]
          rw [hk (e, jL, PUnit.unit)]
          simp only [Option.bind, Option.map]
          refine congrArg some ?_
          simp only [qRaw, kFlat, gkey]
          rw [show kv_offset + e.val * 1 + (c * 64 + jL.val) * 128
                = kv_offset + e.val + (c * 64 + jL.val) * 128 from by ring]
          ring)]
    rw [withBot_sum_some]
    refine congrArg some ?_
    rw [Finset.mul_sum]
  -- assemble: cells
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    NumericDType.add, NumericDType.mul, hb1, hdot, b1Val, b0Val]
  -- fp16-zeros cell = some 0
  simp only [FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  simp only [fscore, qRaw, kFlat, b0Val, b1Val, gkey, hbcol]
  have hjLlt := jL.isLt
  rw [show (c * 64 + jL.val) / 64 = c from by
    rw [Nat.mul_comm c 64, Nat.mul_add_div (by norm_num : (0:Nat) < 64), Nat.div_eq_of_lt jL.isLt]
    omega]
  rw [show (c * 64 + jL.val) % 64 = jL.val from by
    rw [Nat.mul_comm c 64, Nat.mul_add_mod, Nat.mod_eq_of_lt jL.isLt]]
  ring

set_option maxHeartbeats 1000000 in
/-- **General** `attn_qk_cell` (dimension-parameterized over `BLOCK_M BLOCK_N HEAD
BIAS_LAST_SIZE stride_b0m`, with `0 < BLOCK_N`). -/
theorem attn_qk_cellG (s0 : BlockState) (Q K B0 : RegionName) (sm_scale : ℝ)
    (BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m : Nat) (hBN : 0 < BLOCK_N)
    (q_offset kv_offset b_offset : Nat) (nB c bcol : Nat) (hc : c < nB)
    (qtile : Tile .fp16 [BLOCK_M, HEAD]) (ktile : Tile .real [HEAD, BLOCK_N])
    (b1tile : Tile .real [BLOCK_M, BLOCK_N])
    (hbcol : bcol = c)
    (hq : qtile = ⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) idx))⟩)
    (hk : ∀ idx : TileIndex [HEAD, BLOCK_N],
        ktile.data idx = some (s0.readMem K (kv_offset + idx.1.val * 1 + (c * BLOCK_N + idx.2.1.val) * HEAD)))
    (hb1 : b1tile = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE (s0.pids 0) idx.1 idx.2.1.val)⟩)
    (i : Fin BLOCK_M) (jL : Fin BLOCK_N) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (some 0))⟩)
          (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [BLOCK_M, HEAD]) ktile))
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (⟨fun idx : TileIndex [BLOCK_M, 1] =>
              some (s0.readMem B0 (b_offset + (s0.pids 0 * BLOCK_M + idx.1.val) * stride_b0m + bcol))⟩)
            b1tile)
          (Tile.scalar (some 1.44269504)))).data (i, jL, PUnit.unit)
      = some (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
          BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s0.pids 0) i (gkey BLOCK_N nB c hc jL)) := by
  have hdot : (Tile.dot [] (⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [BLOCK_M, HEAD]) ktile).data (i, jL, PUnit.unit)
      = some (sm_scale * 1.44269504 * Finset.univ.sum (fun e : Fin HEAD =>
          qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) (i, e, PUnit.unit)
            * kFlat s0 K kv_offset HEAD (BLOCK_N * nB) e (gkey BLOCK_N nB c hc jL))) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin HEAD) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·)
            ((⟨fun a => FloatDType.fp16.cast FloatDType.real (qtile.data a)⟩ : Tile .real [BLOCK_M, HEAD]).data (i, e, PUnit.unit))
            (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin HEAD) (WithBot ℝ) _ Finset.univ
          (fun e => (some (sm_scale * 1.44269504 * (qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) (i, e, PUnit.unit)
              * kFlat s0 K kv_offset HEAD (BLOCK_N * nB) e (gkey BLOCK_N nB c hc jL))) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          simp only [hq, FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            FloatDType.real_toWithBot, Option.map₂]
          rw [hk (e, jL, PUnit.unit)]
          simp only [Option.bind, Option.map]
          refine congrArg some ?_
          simp only [qRaw, kFlat, gkey]
          rw [show kv_offset + e.val * 1 + (c * BLOCK_N + jL.val) * HEAD
                = kv_offset + e.val + (c * BLOCK_N + jL.val) * HEAD from by ring]
          ring)]
    rw [withBot_sum_some]
    refine congrArg some ?_
    rw [Finset.mul_sum]
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar,
    NumericDType.add, NumericDType.mul, hb1, hdot, b1Val, b0Val]
  simp only [FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  simp only [fscore, qRaw, kFlat, b0Val, b1Val, gkey, hbcol]
  have hjLlt := jL.isLt
  rw [show (c * BLOCK_N + jL.val) / BLOCK_N = c from by
    rw [Nat.mul_comm c BLOCK_N, Nat.mul_add_div hBN, Nat.div_eq_of_lt jL.isLt]
    omega]
  rw [show (c * BLOCK_N + jL.val) % BLOCK_N = jL.val from by
    rw [Nat.mul_comm c BLOCK_N, Nat.mul_add_mod, Nat.mod_eq_of_lt jL.isLt]]
  ring

/-! ### M3 — `attn_step`: the loop body advances the invariant by one block -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
theorem attn_step (Q K V B0 Out : RegionName) (s0 : BlockState)
    (nB : Nat) (i : Nat) (s : BlockState) (hilt : i < 64 * nB)
    (hinv : attnKernelInvariant s0 Q K V B0 Out 0.1
        (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
        64 64 128 0 64 128 1 128 128 1 nB (s0.pids 0) i s) :
    ∃ s', stepStmts (attnLoopBody B0) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnKernelInvariant s0 Q K V B0 Out 0.1
          (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
          64 64 128 0 64 128 1 128 128 1 nB (s0.pids 0) (i + 64) s' := by
  have hBN : (0:Nat) < 64 := by norm_num
  have hc : i / 64 < nB := (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + 64) / 64 = i / 64 + 1 := Nat.add_div_right i hBN
  simp only [attnKernelInvariant] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 0.1 (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
    64 64 128 (64 * nB) 64 128 (s0.pids 0) with hscdef
  set V' := vFlat s0 V (s0.pids 1 * 16384) 128 (64 * nB) with hVpdef
  -- memory of the body-input state = s0
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar i)).readMem R o = s0.readMem R o := by
    intro R o; simp only [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  -- the running registers carried into the body (named to avoid simp recursion)
  have hmi' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [64] "m_i"
      = some (⟨fun r : TileIndex [64] => mPg 64 nB sc r.1 (i / 64)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi
  have hli' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [64] "l_i"
      = some (⟨fun r : TileIndex [64] => ((lPgK sc r.1 (i / 64) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli
  have hacc' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [64, 128] "acc"
      = some (⟨fun idx : TileIndex [64, 128] => ((oPg sc V' idx.1 idx.2.1 (i / 64) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
  have hKp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [128, 64] "K_block_ptr"
      = some (⟨fun _ : TileIndex [128, 64] =>
        { region := K, baseOffset := s0.pids 1 * 16384, parentShape := [128, 64 * nB + 0],
          blockShape := [128, 64], strides := [1, 128], offsets := [0, i / 64 * 64] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hKp]
  have hVp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [64, 128] "V_block_ptr"
      = some (⟨fun _ : TileIndex [64, 128] =>
        { region := V, baseOffset := s0.pids 1 * 16384, parentShape := [64 * nB + 0, 128],
          blockShape := [64, 128], strides := [128, 1], offsets := [i / 64 * 64, 0] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hVp]
  have hq' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .fp16 [64, 128] "q"
      = some (⟨fun idx : TileIndex [64, 128] =>
        FloatDType.real.cast FloatDType.fp16
          (some (0.1 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 16384) 64 128 (s0.pids 0) idx))⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  have hb1' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [64, 64] "b1"
      = some (⟨fun idx : TileIndex [64, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 16384) 64 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hb1
  -- run the body chain
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hb1F, hboF, hsmF, hqoF, hmF, hKpF, hVpF,
      qkT, rmaxT, mnewT, alphaT, pT, hqkData, hrm, hmnewd, halphad, hpTd, hm_iF, haccF, hl_iF⟩ :=
    attnLoopBody_steps B0 (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      (fun r : Fin 64 => r.val)
      K V (s0.pids 1 * 16384) (s0.pids 1 * 16384) (i / 64 * 64) (i / 64 * 64) (64 * nB + 0) (64 * nB + 0)
      (s0.pids 1 * 16384) (s0.pids 0)
      (⟨fun idx : TileIndex [64, 128] =>
        FloatDType.real.cast FloatDType.fp16
          (some (0.1 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 16384) 64 128 (s0.pids 0) idx))⟩)
      (⟨fun r : TileIndex [64] => mPg 64 nB sc r.1 (i / 64)⟩)
      (⟨fun r : TileIndex [64] => ((lPgK sc r.1 (i / 64) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [64, 128] => ((oPg sc V' idx.1 idx.2.1 (i / 64) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [128, 64] =>
        some (s0.readMem K (s0.pids 1 * 16384 + idx.1.val * 1 + (i / 64 * 64 + idx.2.1.val) * 128))⟩)
      (⟨fun idx : TileIndex [64, 128] =>
        some (s0.readMem V (s0.pids 1 * 16384 + (i / 64 * 64 + idx.1.val) * 128 + idx.2.1.val * 1))⟩)
      (⟨fun idx : TileIndex [64, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 16384) 64 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbo)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsm)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hm)
      hmi' hli' hacc' hq' hb1' hKp' hVp'
      (fun idx => congrArg some (hrmem K _).symm) (fun idx => congrArg some (hrmem V _).symm)
      (s0.pids 1 * 16384) (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqo)
      (by intro rg o; simp [BlockState.setReg_undef, hundef])
  refine ⟨sF, hchain, ?_⟩
  -- the per-cell qk bridge
  have hqk : ∀ (r : Fin 64) (jL : Fin 64),
      qkT.data (r, jL, PUnit.unit) = some (sc r (gkey 64 nB (i / 64) hc jL)) := by
    intro r jL
    rw [hqkData r jL]
    have := attn_qk_cell s0 Q K B0 0.1 (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
      nB (i / 64) (i / 64) hc
      (⟨fun idx : TileIndex [64, 128] =>
        FloatDType.real.cast FloatDType.fp16
          (some (0.1 * 1.44269504 * qRaw s0 Q (s0.pids 1 * 16384) 64 128 (s0.pids 0) idx))⟩)
      (⟨fun idx : TileIndex [128, 64] =>
        some (s0.readMem K (s0.pids 1 * 16384 + idx.1.val * 1 + (i / 64 * 64 + idx.2.1.val) * 128))⟩)
      (⟨fun idx : TileIndex [64, 64] =>
        some (b1Val s0 B0 (s0.pids 1 * 16384) 64 128 64 (s0.pids 0) idx.1 idx.2.1.val)⟩)
      rfl rfl (fun idx => rfl) rfl r jL
    simp only [hscdef]
    simp only [hrmem] at this ⊢
    exact this
  -- m_ij bridge
  have hmij_eq : ∀ idx : TileIndex [64], mnewT.data idx
      = mPg 64 nB sc idx.1 (i / 64 + 1) := by
    intro idx
    rw [hmnewd]
    refine mijg_eq 64 nB (i / 64) hc sc _ rmaxT idx.1 ?_ ?_
    · rfl
    · exact reduceMaxDrop_data_row 64 64 hBN _ rmaxT hrm idx.1 _ (fun jL => hqk idx.1 jL)
  simp only [attnKernelInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF]; exact hpids
  · rw [hc1, Nat.add_one_mul, ← hieq]
  · rw [hc1]; omega
  · -- m_i = mPg (c+1)
    rw [hm_iF, hc1]; refine congrArg some ?_; ext idx; exact hmij_eq idx
  · -- l_i = lPgK (c+1)
    rw [hl_iF, hc1]; refine congrArg some ?_; ext idx
    rw [halphad, hpTd]
    exact lig_eq 64 nB (i / 64) hBN hc sc _ mnewT _ qkT idx.1 rfl (hmij_eq idx) rfl (fun jL => hqk idx.1 jL)
  · -- acc = oPg (c+1)
    rw [haccF, hc1]; refine congrArg some ?_
    ext idx
    have hv : ∀ jL : Fin 64,
        (⟨fun idx : TileIndex [64, 128] =>
          some (s0.readMem V (s0.pids 1 * 16384 + (i / 64 * 64 + idx.1.val) * 128 + idx.2.1.val * 1))⟩
          : Tile .real [64, 128]).data (jL, idx.2.1, PUnit.unit)
          = some (V' (gkey 64 nB (i / 64) hc jL, idx.2.1, PUnit.unit)) := by
      intro jL
      refine congrArg some ?_
      simp only [hVpdef, vFlat, gkey]
      rw [show s0.pids 1 * 16384 + (i / 64 * 64 + jL.val) * 128 + idx.2.1.val * 1
            = s0.pids 1 * 16384 + (i / 64 * 64 + jL.val) * 128 + idx.2.1.val from by ring]
    refine accg_eq 64 nB (i / 64) hc sc V' _ alphaT pT _ idx.1 idx.2.1 rfl ?_ ?_ hv
    · rw [halphad]
      exact alphag_eq 64 nB (i / 64) sc _ mnewT idx.1 rfl (hmij_eq (idx.1, PUnit.unit))
    · intro jL
      rw [hpTd]
      refine exp2_some (fun a b => sc a (gkey 64 nB (i / 64) hc b) - mRg sc a (i / 64 + 1)) _ idx.1 jL ?_
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk idx.1 jL,
        hmij_eq (idx.1, PUnit.unit),
        mPg_eq_coe sc hBN idx.1 (i / 64 + 1) (Nat.le_add_left 1 (i / 64)) hc,
        NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]
  · -- K block ptr advances
    rw [hKpF, hc1, show i / 64 * 64 + 64 = (i / 64 + 1) * 64 from by ring]
  · -- V block ptr advances
    rw [hVpF, hc1, show i / 64 * 64 + 64 = (i / 64 + 1) * 64 from by ring]
  · rw [hqF]
  · rw [hb1F]
  · rw [hboF]
  · rw [hsmF]
  · rw [hqoF]
  · rw [hmF]
  · exact hundefF
  · rw [hmemF]; exact hmem

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General** `attn_step` (dimension-parameterized). Advances the invariant by
one key block of `BLOCK_N` keys. Contiguous K/V layout (`stride_kk=1,
stride_kn=HEAD, stride_vk=HEAD, stride_vn=1`) per `kFlat`/`vFlat`. -/
theorem attn_stepG (Q K V B0 Out : RegionName) (s0 : BlockState) (sm_scale : ℝ)
    (BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m : Nat) (hBN : 0 < BLOCK_N)
    (q_offset kv_offset b_offset : Nat)
    (nB : Nat) (i : Nat) (s : BlockState) (hilt : i < BLOCK_N * nB)
    (hinv : attnKernelInvariant s0 Q K V B0 Out sm_scale
        q_offset kv_offset b_offset
        BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m 1 HEAD HEAD 1 nB (s0.pids 0) i s) :
    ∃ s', stepStmts (attnLoopBodyG B0 BLOCK_M BLOCK_N HEAD stride_b0m) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ attnKernelInvariant s0 Q K V B0 Out sm_scale
          q_offset kv_offset b_offset
          BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m 1 HEAD HEAD 1 nB (s0.pids 0) (i + BLOCK_N) s' := by
  have hc : i / BLOCK_N < nB := (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_N) / BLOCK_N = i / BLOCK_N + 1 := Nat.add_div_right i hBN
  simp only [attnKernelInvariant] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
    BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s0.pids 0) with hscdef
  set V' := vFlat s0 V kv_offset HEAD (BLOCK_N * nB) with hVpdef
  have hrmem : ∀ (R : RegionName) (o : Nat),
      (s.setReg "start_n" .nat [] (Tile.scalar i)).readMem R o = s0.readMem R o := by
    intro R o; simp only [BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hmi' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [BLOCK_M] "m_i"
      = some (⟨fun r : TileIndex [BLOCK_M] => mPg BLOCK_N nB sc r.1 (i / BLOCK_N)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hmi
  have hli' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [BLOCK_M] "l_i"
      = some (⟨fun r : TileIndex [BLOCK_M] => ((lPgK sc r.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hli
  have hacc' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [BLOCK_M, HEAD] "acc"
      = some (⟨fun idx : TileIndex [BLOCK_M, HEAD] => ((oPg sc V' idx.1 idx.2.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hacc
  have hKp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [HEAD, BLOCK_N] "K_block_ptr"
      = some (⟨fun _ : TileIndex [HEAD, BLOCK_N] =>
        { region := K, baseOffset := kv_offset, parentShape := [HEAD, BLOCK_N * nB + P_SEQ],
          blockShape := [HEAD, BLOCK_N], strides := [1, HEAD], offsets := [0, i / BLOCK_N * BLOCK_N] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hKp]
  have hVp' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .blockPtr [BLOCK_N, HEAD] "V_block_ptr"
      = some (⟨fun _ : TileIndex [BLOCK_N, HEAD] =>
        { region := V, baseOffset := kv_offset, parentShape := [BLOCK_N * nB + P_SEQ, HEAD],
          blockShape := [BLOCK_N, HEAD], strides := [HEAD, 1], offsets := [i / BLOCK_N * BLOCK_N, 0] }⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hVp]
  have hq' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .fp16 [BLOCK_M, HEAD] "q"
      = some (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) idx))⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hq
  have hb1' : (s.setReg "start_n" .nat [] (Tile.scalar i)).regs .real [BLOCK_M, BLOCK_N] "b1"
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE (s0.pids 0) idx.1 idx.2.1.val)⟩) := by
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hb1
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF, hqF, hb1F, hboF, hsmF, hqoF, hmF, hKpF, hVpF,
      qkT, rmaxT, mnewT, alphaT, pT, hqkData, hrm, hmnewd, halphad, hpTd, hm_iF, haccF, hl_iF⟩ :=
    attnLoopBody_stepsG B0 BLOCK_M BLOCK_N HEAD stride_b0m hBN (s.setReg "start_n" .nat [] (Tile.scalar i)) i
      K V kv_offset kv_offset (i / BLOCK_N * BLOCK_N) (i / BLOCK_N * BLOCK_N) (BLOCK_N * nB + P_SEQ) (BLOCK_N * nB + P_SEQ)
      b_offset (s0.pids 0)
      (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) idx))⟩)
      (⟨fun r : TileIndex [BLOCK_M] => mPg BLOCK_N nB sc r.1 (i / BLOCK_N)⟩)
      (⟨fun r : TileIndex [BLOCK_M] => ((lPgK sc r.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [BLOCK_M, HEAD] => ((oPg sc V' idx.1 idx.2.1 (i / BLOCK_N) : ℝ) : WithBot ℝ)⟩)
      (⟨fun idx : TileIndex [HEAD, BLOCK_N] =>
        some (s0.readMem K (kv_offset + idx.1.val * 1 + (i / BLOCK_N * BLOCK_N + idx.2.1.val) * HEAD))⟩)
      (⟨fun idx : TileIndex [BLOCK_N, HEAD] =>
        some (s0.readMem V (kv_offset + (i / BLOCK_N * BLOCK_N + idx.1.val) * HEAD + idx.2.1.val * 1))⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE (s0.pids 0) idx.1 idx.2.1.val)⟩)
      (by rw [BlockState.setReg_same])
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hbo)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hsm)
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hm)
      hmi' hli' hacc' hq' hb1' hKp' hVp'
      (fun idx => congrArg some (hrmem K _).symm) (fun idx => congrArg some (hrmem V _).symm)
      q_offset (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hqo)
      (by intro rg o; simp [BlockState.setReg_undef, hundef])
  refine ⟨sF, hchain, ?_⟩
  have hqk : ∀ (r : Fin BLOCK_M) (jL : Fin BLOCK_N),
      qkT.data (r, jL, PUnit.unit) = some (sc r (gkey BLOCK_N nB (i / BLOCK_N) hc jL)) := by
    intro r jL
    rw [hqkData r jL]
    have := attn_qk_cellG s0 Q K B0 sm_scale BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m hBN
      q_offset kv_offset b_offset
      nB (i / BLOCK_N) (i / BLOCK_N) hc
      (⟨fun idx : TileIndex [BLOCK_M, HEAD] =>
        FloatDType.real.cast FloatDType.fp16
          (some (sm_scale * 1.44269504 * qRaw s0 Q q_offset BLOCK_M HEAD (s0.pids 0) idx))⟩)
      (⟨fun idx : TileIndex [HEAD, BLOCK_N] =>
        some (s0.readMem K (kv_offset + idx.1.val * 1 + (i / BLOCK_N * BLOCK_N + idx.2.1.val) * HEAD))⟩)
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE (s0.pids 0) idx.1 idx.2.1.val)⟩)
      rfl rfl (fun idx => rfl) rfl r jL
    simp only [hscdef]
    simp only [hrmem] at this ⊢
    exact this
  have hmij_eq : ∀ idx : TileIndex [BLOCK_M], mnewT.data idx
      = mPg BLOCK_N nB sc idx.1 (i / BLOCK_N + 1) := by
    intro idx
    rw [hmnewd]
    refine mijg_eq BLOCK_N nB (i / BLOCK_N) hc sc _ rmaxT idx.1 ?_ ?_
    · rfl
    · exact reduceMaxDrop_data_row BLOCK_M BLOCK_N hBN _ rmaxT hrm idx.1 _ (fun jL => hqk idx.1 jL)
  simp only [attnKernelInvariant]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF]; exact hpids
  · rw [hc1, Nat.add_one_mul, ← hieq]
  · rw [hc1]; omega
  · rw [hm_iF, hc1]; refine congrArg some ?_; ext idx; exact hmij_eq idx
  · rw [hl_iF, hc1]; refine congrArg some ?_; ext idx
    rw [halphad, hpTd]
    exact lig_eq BLOCK_N nB (i / BLOCK_N) hBN hc sc _ mnewT _ qkT idx.1 rfl (hmij_eq idx) rfl (fun jL => hqk idx.1 jL)
  · rw [haccF, hc1]; refine congrArg some ?_
    ext idx
    have hv : ∀ jL : Fin BLOCK_N,
        (⟨fun idx : TileIndex [BLOCK_N, HEAD] =>
          some (s0.readMem V (kv_offset + (i / BLOCK_N * BLOCK_N + idx.1.val) * HEAD + idx.2.1.val * 1))⟩
          : Tile .real [BLOCK_N, HEAD]).data (jL, idx.2.1, PUnit.unit)
          = some (V' (gkey BLOCK_N nB (i / BLOCK_N) hc jL, idx.2.1, PUnit.unit)) := by
      intro jL
      refine congrArg some ?_
      simp only [hVpdef, vFlat, gkey]
      rw [show kv_offset + (i / BLOCK_N * BLOCK_N + jL.val) * HEAD + idx.2.1.val * 1
            = kv_offset + (i / BLOCK_N * BLOCK_N + jL.val) * HEAD + idx.2.1.val from by ring]
    refine accg_eq BLOCK_N nB (i / BLOCK_N) hc sc V' _ alphaT pT _ idx.1 idx.2.1 rfl ?_ ?_ hv
    · rw [halphad]
      exact alphag_eq BLOCK_N nB (i / BLOCK_N) sc _ mnewT idx.1 rfl (hmij_eq (idx.1, PUnit.unit))
    · intro jL
      rw [hpTd]
      refine exp2_some (fun a b => sc a (gkey BLOCK_N nB (i / BLOCK_N) hc b) - mRg sc a (i / BLOCK_N + 1)) _ idx.1 jL ?_
      simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex, hqk idx.1 jL,
        hmij_eq (idx.1, PUnit.unit),
        mPg_eq_coe sc hBN idx.1 (i / BLOCK_N + 1) (Nat.le_add_left 1 (i / BLOCK_N)) hc,
        NumericDType.sub, WithBot.realSub, Option.map₂, Option.bind, Option.map]
  · rw [hKpF, hc1, show i / BLOCK_N * BLOCK_N + BLOCK_N = (i / BLOCK_N + 1) * BLOCK_N from by ring]
  · rw [hVpF, hc1, show i / BLOCK_N * BLOCK_N + BLOCK_N = (i / BLOCK_N + 1) * BLOCK_N from by ring]
  · rw [hqF]
  · rw [hb1F]
  · rw [hboF]
  · rw [hsmF]
  · rw [hqoF]
  · rw [hmF]
  · exact hundefF
  · rw [hmemF]; exact hmem

/-! ### M3 — `attn_postLoop`: `acc /= l_i` + block store = `attentionKernelSpec`

After the streaming loop (invariant at `c = nB`), the `acc /= l_i[:, None]` rescale
and the (unmasked) block-pointer store write the normalized accumulator
`oPg / lPgK = oPg / lPg = attentionKernelSpec` to `Out` at every lane. The store is a
scatter over an injective block-pointer offset map (`row·128 + col` injective since
`col < 128`); `lPgK_eq_lPg` kills the kernel's `l_i = 0` seed (`1 ≤ nB`), and
`attentionKernelSpec_eq_ratio` reads off the closed form. -/

/-- A fp16 `writeMemTyped` foldl over offsets all `≠ o` preserves `mem … o`. -/
theorem foldl_writeMemTyped_fp16_preserves' {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, offsetFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k)) s).mem region o)
      = s.mem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
    have htl : ∀ k ∈ tl, offsetFn k ≠ o := fun k hk => h k (List.mem_cons_of_mem hd hk)
    rw [List.foldl_cons, ih _ htl]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    change (if region = region ∧ o = offsetFn hd then _ else s.mem region o) = s.mem region o
    rw [if_neg]; rintro ⟨_, h_eq⟩; exact hhd h_eq.symm

/-- Fp16 block-store readback: an injective fp16 scatter reads back its stored
cell as the fp16 `MemCell`. (Local mirror of the flash O-store readback.) -/
theorem scatter_memcell_fp16_nd' {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i))
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq; have hki : k = i := h_inj heq; subst hki; exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves' offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else _) = _
  rw [if_pos ⟨rfl, rfl⟩]

set_option maxHeartbeats 1000000 in
theorem attn_postLoop (Q K V B0 Out : RegionName) (s0 : BlockState)
    (nB : Nat) (hnB : 1 ≤ nB) (st : BlockState)
    (hinv : attnKernelInvariant s0 Q K V B0 Out 0.1
        (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
        64 64 128 0 64 128 1 128 128 1 nB (s0.pids 0) (64 * nB) st) :
    ∃ sfin, stepStmts (attnPostLoop Out) st = some sfin
      ∧ ∀ idx : TileIndex [64, 128],
          sfin.mem Out (s0.pids 1 * 16384 + (s0.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (attentionKernelSpec s0 Q K V B0 0.1
                  (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
                  64 64 128 (64 * nB) 64 128 (s0.pids 0) idx))) := by
  have hBN : (0:Nat) < 64 := by norm_num
  have hcnB : (64 * nB) / 64 = nB := by rw [Nat.mul_comm, Nat.mul_div_cancel _ hBN]
  simp only [attnKernelInvariant, hcnB] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 0.1 (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
    64 64 128 (64 * nB) 64 128 (s0.pids 0) with hscdef
  set V' := vFlat s0 V (s0.pids 1 * 16384) 128 (64 * nB) with hVpdef
  -- div result tile (acc / l_i[:, None]) = oPg / lPgK
  set acc' : Tile .real [64, 128] :=
    ⟨fun idx : TileIndex [64, 128] => ((oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB : ℝ) : WithBot ℝ)⟩
    with hacc'def
  unfold attnPostLoop
  -- L0: acc = acc / l_i[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [64, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "l_i"))) st = some acc' from by
      have hexp : @evalOp TileDType.real [64, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [64] "l_i")) st
          = some (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [64] => ((lPgK sc r.1 nB : ℝ) : WithBot ℝ)⟩ : Tile .real [64])) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
        Option.map₂, Option.bind, Option.map, hacc'def]
      rfl))]
  set st1 := st.setReg "acc" .real [64, 128] acc' with hst1
  -- L1: O_block_ptr = makeBlockPtr(Out + q_offset, offsets=(start_m*64, 0))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_rowcol_eval Out (Op.ref .nat [] "q_offset") [128, 128] [64, 128] [128, 1]
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 64)) st1
      (s0.pids 1 * 16384) (s0.pids 0 * 64)
      (by rw [evalOp_ref]; simp only [hst1, BlockState.setReg_ne_name]; exact hqo)
      (evalOp_mul_ref_const st1 "start_m" (s0.pids 0) 64
        (by simp only [hst1, BlockState.setReg_ne_name]; exact hsm))))]
  set st2 := st1.setReg "O_block_ptr" .blockPtr [64, 128]
      ⟨fun _ : TileIndex [64, 128] =>
        { region := Out, baseOffset := s0.pids 1 * 16384, parentShape := [128, 128],
          blockShape := [64, 128], strides := [128, 1], offsets := [s0.pids 0 * 64, 0] }⟩
    with hst2
  -- store value / offset functions
  set oValFn : TileIndex [64, 128] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16
      (some (oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB)) with hoValFn
  set oOffFn : TileIndex [64, 128] → Nat :=
    fun idx => s0.pids 1 * 16384 + (s0.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1 with hoOffFn
  have hOpref : @evalOp TileDType.blockPtr [64, 128] (Op.ref .blockPtr [64, 128] "O_block_ptr") st2
      = some (⟨fun _ : TileIndex [64, 128] =>
          { region := Out, baseOffset := s0.pids 1 * 16384, parentShape := [128, 128],
            blockShape := [64, 128], strides := [128, 1], offsets := [s0.pids 0 * 64, 0] }⟩
          : Tile .blockPtr [64, 128]) := by rw [evalOp_ref, hst2, BlockState.setReg_same]
  have haccref : @evalOp TileDType.real [64, 128] (Op.ref .real [64, 128] "acc") st2 = some acc' := by
    rw [evalOp_ref, hst2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hst1,
      BlockState.setReg_same]
  have hval : evalOp (Op.castFloat .real .fp16 (Op.ref .real [64, 128] "acc")) st2
        = some (⟨oValFn⟩ : Tile .fp16 [64, 128]) := by
    rw [evalOp_castFloat]; erw [haccref]; rfl
  -- L2: store O via O_block_ptr (acc.to fp16)
  have hstore : stepStmt (Stmt.store .fp16 [64, 128]
      (MemAccess.blockPtr (Op.ref .blockPtr [64, 128] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [64, 128] "acc")) MaskOpt.none) st2
      = some ((TileShape.allIndices [64, 128]).foldl
          (fun acc idx => acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx)) st2) := by
    unfold stepStmt
    erw [hval]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    erw [hOpref]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ st2 ?_
    intro acc idx _
    simp only [TileShape.blockPtr_inBounds_nil_index, Bool.and_true, Bool.true_and,
      TileShape.blockPtr_address_2d_row_offset_index, hoOffFn, if_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinjO : Function.Injective oOffFn := by
    rintro ⟨⟨a, ha⟩, ⟨b, hb⟩, _⟩ ⟨⟨c, hc⟩, ⟨d, hd⟩, _⟩ heq
    simp only [hoOffFn] at heq
    have hac : a = c ∧ b = d := by
      set P := s0.pids 0 * 64 with hP
      omega
    obtain ⟨rfl, rfl⟩ := hac
    rfl
  rw [show s0.pids 1 * 16384 + (s0.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1 = oOffFn idx from rfl]
  rw [scatter_memcell_fp16_nd' (region := Out) st2 oOffFn oValFn hinjO idx]
  refine congrArg (MemCell.of .fp16) ?_
  rw [hoValFn]
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot,
    FloatDType.toWithBot, WithBot.unbotD_coe]
  refine congrArg (FloatDType.real.cast FloatDType.fp16) ?_
  refine congrArg some ?_
  rw [lPgK_eq_lPg sc idx.1 nB hnB le_rfl]
  exact attentionKernelSpec_eq_ratio s0 Q K V B0 0.1
    (s0.pids 1 * 16384) (s0.pids 1 * 16384) (s0.pids 1 * 16384)
    64 64 128 64 128 nB (s0.pids 0) hBN hnB idx.1 idx.2.1

set_option maxHeartbeats 1000000 in
/-- **General** `attn_postLoop` (dimension-parameterized). The `acc /= l_i` rescale
and block store write the closed form `attentionKernelSpec` to `Out` at every lane.
Contiguous Out layout (`stride_om=HEAD, stride_on=1`). -/
theorem attn_postLoopG (Q K V B0 Out : RegionName) (s0 : BlockState) (sm_scale : ℝ)
    (BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m q_offset kv_offset b_offset : Nat)
    (hBN : 0 < BLOCK_M) (hKN : 0 < BLOCK_N) (hHD : 0 < HEAD)
    (nB : Nat) (hnB : 1 ≤ nB) (st : BlockState)
    (hinv : attnKernelInvariant s0 Q K V B0 Out sm_scale
        q_offset kv_offset b_offset
        BLOCK_M BLOCK_N HEAD P_SEQ BIAS_LAST_SIZE stride_b0m 1 HEAD HEAD 1 nB (s0.pids 0) (BLOCK_N * nB) st) :
    ∃ sfin, stepStmts (attnPostLoopG Out BLOCK_M HEAD (BLOCK_N * nB) HEAD 1) st = some sfin
      ∧ ∀ idx : TileIndex [BLOCK_M, HEAD],
          sfin.mem Out (q_offset + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (attentionKernelSpec s0 Q K V B0 sm_scale
                  q_offset kv_offset b_offset
                  BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s0.pids 0) idx))) := by
  have hcnB : (BLOCK_N * nB) / BLOCK_N = nB := by rw [Nat.mul_comm, Nat.mul_div_cancel _ hKN]
  have hieqcn : BLOCK_N * nB = BLOCK_N * nB / BLOCK_N * BLOCK_N := by rw [hcnB, Nat.mul_comm]
  simp only [attnKernelInvariant, hcnB] at hinv
  obtain ⟨hpids, hieq, hcle, hmi, hli, hacc, hKp, hVp, hq, hb1, hbo, hsm, hqo, hm, hundef, hmem⟩ := hinv
  set sc := fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
    BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s0.pids 0) with hscdef
  set V' := vFlat s0 V kv_offset HEAD (BLOCK_N * nB) with hVpdef
  set acc' : Tile .real [BLOCK_M, HEAD] :=
    ⟨fun idx : TileIndex [BLOCK_M, HEAD] => ((oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB : ℝ) : WithBot ℝ)⟩
    with hacc'def
  unfold attnPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, HEAD] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))) st = some acc' from by
      have hexp : @evalOp TileDType.real [BLOCK_M, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i")) st
          = some (Tile.expandDim ⟨1, by simp⟩
              (⟨fun r : TileIndex [BLOCK_M] => ((lPgK sc r.1 nB : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M])) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
        Option.map₂, Option.bind, Option.map, hacc'def]
      rfl))]
  set st1 := st.setReg "acc" .real [BLOCK_M, HEAD] acc' with hst1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_rowcol_eval Out (Op.ref .nat [] "q_offset") [BLOCK_N * nB, HEAD] [BLOCK_M, HEAD] [HEAD, 1]
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) st1
      q_offset (s0.pids 0 * BLOCK_M)
      (by rw [evalOp_ref]; simp only [hst1, BlockState.setReg_ne_name]; exact hqo)
      (evalOp_mul_ref_const st1 "start_m" (s0.pids 0) BLOCK_M
        (by simp only [hst1, BlockState.setReg_ne_name]; exact hsm))))]
  set st2 := st1.setReg "O_block_ptr" .blockPtr [BLOCK_M, HEAD]
      ⟨fun _ : TileIndex [BLOCK_M, HEAD] =>
        { region := Out, baseOffset := q_offset, parentShape := [BLOCK_N * nB, HEAD],
          blockShape := [BLOCK_M, HEAD], strides := [HEAD, 1], offsets := [s0.pids 0 * BLOCK_M, 0] }⟩
    with hst2
  set oValFn : TileIndex [BLOCK_M, HEAD] → TileCarrier TileDType.fp16 :=
    fun idx => FloatDType.real.cast FloatDType.fp16
      (some (oPg sc V' idx.1 idx.2.1 nB / lPgK sc idx.1 nB)) with hoValFn
  set oOffFn : TileIndex [BLOCK_M, HEAD] → Nat :=
    fun idx => q_offset + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1 with hoOffFn
  have hOpref : @evalOp TileDType.blockPtr [BLOCK_M, HEAD] (Op.ref .blockPtr [BLOCK_M, HEAD] "O_block_ptr") st2
      = some (⟨fun _ : TileIndex [BLOCK_M, HEAD] =>
          { region := Out, baseOffset := q_offset, parentShape := [BLOCK_N * nB, HEAD],
            blockShape := [BLOCK_M, HEAD], strides := [HEAD, 1], offsets := [s0.pids 0 * BLOCK_M, 0] }⟩
          : Tile .blockPtr [BLOCK_M, HEAD]) := by rw [evalOp_ref, hst2, BlockState.setReg_same]
  have haccref : @evalOp TileDType.real [BLOCK_M, HEAD] (Op.ref .real [BLOCK_M, HEAD] "acc") st2 = some acc' := by
    rw [evalOp_ref, hst2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hst1,
      BlockState.setReg_same]
  have hval : evalOp (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, HEAD] "acc")) st2
        = some (⟨oValFn⟩ : Tile .fp16 [BLOCK_M, HEAD]) := by
    rw [evalOp_castFloat]; erw [haccref]; rfl
  have hstore : stepStmt (Stmt.store .fp16 [BLOCK_M, HEAD]
      (MemAccess.blockPtr (Op.ref .blockPtr [BLOCK_M, HEAD] "O_block_ptr") [])
      (Op.castFloat .real .fp16 (Op.ref .real [BLOCK_M, HEAD] "acc")) MaskOpt.none) st2
      = some ((TileShape.allIndices [BLOCK_M, HEAD]).foldl
          (fun acc idx => acc.writeMemTyped .fp16 Out (oOffFn idx) (oValFn idx)) st2) := by
    unfold stepStmt
    erw [hval]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    erw [hOpref]
    simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some]
    refine congrArg some ?_
    refine List.foldl_ext _ _ st2 ?_
    intro acc idx _
    simp only [TileShape.blockPtr_inBounds_nil_index, Bool.and_true, Bool.true_and,
      TileShape.blockPtr_address_2d_row_offset_index, hoOffFn, if_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinjO : Function.Injective oOffFn := by
    rintro ⟨⟨a, ha⟩, ⟨b, hb⟩, _⟩ ⟨⟨c, hc⟩, ⟨d, hd⟩, _⟩ heq
    simp only [hoOffFn] at heq
    have hbHD : b < HEAD := hb
    have hdHD : d < HEAD := hd
    set P := s0.pids 0 * BLOCK_M with hP
    have heq2 : (P + a) * HEAD + b = (P + c) * HEAD + d := by
      have : q_offset + (P + a) * HEAD + b * 1 = q_offset + (P + c) * HEAD + d * 1 := heq
      omega
    have hbd : b = d := by
      have e1 : ((P + a) * HEAD + b) % HEAD = b := by
        rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hbHD]
      have e2 : ((P + c) * HEAD + d) % HEAD = d := by
        rw [Nat.mul_add_mod', Nat.mod_eq_of_lt hdHD]
      rw [← e1, ← e2, heq2]
    subst hbd
    have hac : a = c := by
      have hmm : (P + a) * HEAD = (P + c) * HEAD := by omega
      have := Nat.eq_of_mul_eq_mul_right hHD hmm
      omega
    simp only [hoOffFn, hac]
  rw [show q_offset + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1 = oOffFn idx from rfl]
  rw [scatter_memcell_fp16_nd' (region := Out) st2 oOffFn oValFn hinjO idx]
  refine congrArg (MemCell.of .fp16) ?_
  rw [hoValFn]
  simp only [FloatDType.cast, FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot,
    FloatDType.toWithBot, WithBot.unbotD_coe]
  refine congrArg (FloatDType.real.cast FloatDType.fp16) ?_
  refine congrArg some ?_
  rw [lPgK_eq_lPg sc idx.1 nB hnB le_rfl]
  exact attentionKernelSpec_eq_ratio s0 Q K V B0 sm_scale
    q_offset kv_offset b_offset
    BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB (s0.pids 0) hKN hnB idx.1 idx.2.1

/-! ## M4 — top theorem: genuine streaming-softmax closed-form correctness

The whole-kernel `Out` store realizes the genuine base-2 streaming-softmax closed
form `attentionKernelSpec` (= `attnGenScore fscore vFlat` of the loaded Q/K/V/B0
tiles, with the kernel's actual per-key score `fscore` — scaled dot + additive
relative-position bias), NOT the kernel's own executed value. Composed from
`preLoop` (`P 0`) + `forRangeDyn_inv` over `attn_step` + `attn_postLoop`. -/

set_option maxHeartbeats 1000000 in
/-- **`attention_kernel.py` whole-kernel exec assembly.** Steps the entire faithful
surface (prologue + `forRangeDyn` loop + epilogue) and reads off the genuine
closed form at every `Out` lane. The loop runs exactly `nB = 2` blocks at the
Python test shape (`N_CTX = 128 = 64·2`). -/
theorem attention_kernel_exec (Q K V B0 Out : RegionName) (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [64, 128],
          sF.mem Out (s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (attentionKernelSpec s Q K V B0 0.1
                  (s.pids 1 * 16384) (s.pids 1 * 16384) (s.pids 1 * 16384)
                  64 64 128 (64 * 2) 64 128 (s.pids 0) idx))) := by
  -- decompose body = take 19 ++ drop 19, with take 19 = preLoop prologue
  rw [← List.take_append_drop 19 (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body]
  -- preLoop
  obtain ⟨s0, hpre, hlo, hhi, hinv0⟩ := preLoop Q K V B0 Out s hundef
  rw [stepStmts.append_some hpre]
  -- drop 19 = forRangeDyn :: drop 20 (= attnPostLoop)
  rw [attnLoopBody_check Q K V B0 Out, show (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out 0.1
        16384 128 1 16384 128 1 16384 128 1 16384 128 1 16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16).toAlgKernel.body.drop 20 = attnPostLoop Out from attnPostLoop_check Q K V B0 Out]
  -- run the forRangeDyn loop via forRangeDyn_inv with P = attnKernelInvariant
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.ref .nat [] "lo") (stopOp := Op.ref .nat [] "hi")
      (stepOp := Op.constNat 64)
      (P := fun i st => attnKernelInvariant s Q K V B0 Out 0.1
        (s.pids 1 * 16384) (s.pids 1 * 16384) (s.pids 1 * 16384)
        64 64 128 0 64 128 1 128 128 1 2 (s.pids 0) i st)
      (s_init := s0)
      (by rw [evalOp_ref, hlo])
      (by rw [evalOp_ref, hhi])
      (by rw [evalOp_constNat])
      (by norm_num)
      hinv0
      (fun i st hi hP => attn_step Q K V B0 Out s 2 i st (by omega) hP)
  rw [stepStmts.cons_some hloop]
  -- at loop exit, counter `final` is the first multiple of 64 ≥ 128, i.e. 128 = 64*2
  have hfinal : final = 64 * 2 := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  subst hfinal
  -- postLoop
  obtain ⟨sF, hpostStep, hO⟩ := attn_postLoop Q K V B0 Out s 2 (by norm_num) sL hinvL
  exact ⟨sF, hpostStep, hO⟩

set_option maxHeartbeats 1000000 in
/-- **General** whole-kernel exec assembly (dimension-parameterized). Steps the
entire faithful surface and reads off the genuine closed form at every `Out` lane.
The loop runs exactly `nB = numKVBlocks` blocks (`N_CTX = BLOCK_N·nB`, `P_SEQ = 0`),
with contiguous Q/K/V/Out layout. -/
theorem attention_kernel_exec_general (Q K V B0 Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hKN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hHD : 0 < HEAD) (hnB : 1 ≤ nB)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [BLOCK_M, HEAD],
          sF.mem Out (s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (attentionKernelSpec s Q K V B0 sm_scale
                  (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h)
                  BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s.pids 0) idx))) := by
  rw [← List.take_append_drop 19 (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16).toAlgKernel.body]
  obtain ⟨s0, hpre, hlo, hhi, hinv0⟩ := preLoopG Q K V B0 Out s sm_scale
    stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD 0 BIAS_LAST_SIZE stride_b0m nB hundef
  rw [stepStmts.append_some hpre]
  rw [attnLoopBodyG_check Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N,
    show (attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16).toAlgKernel.body.drop 20
      = attnPostLoopG Out BLOCK_M HEAD (BLOCK_N * nB) HEAD 1 from by
      have := attnPostLoopG_check Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
      simpa using this]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRangeDyn_inv (idx := "start_n")
      (startOp := Op.ref .nat [] "lo") (stopOp := Op.ref .nat [] "hi")
      (stepOp := Op.constNat BLOCK_N)
      (P := fun i st => attnKernelInvariant s Q K V B0 Out sm_scale
        (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h)
        BLOCK_M BLOCK_N HEAD 0 BIAS_LAST_SIZE stride_b0m 1 HEAD HEAD 1 nB (s.pids 0) i st)
      (s_init := s0)
      (by rw [evalOp_ref, hlo])
      (by rw [evalOp_ref, hhi])
      (by rw [evalOp_constNat])
      (by omega)
      hinv0
      (fun i st hi hP => attn_stepG Q K V B0 Out s sm_scale BLOCK_M BLOCK_N HEAD 0 BIAS_LAST_SIZE stride_b0m hKN
        (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h) nB i st (by omega) hP)
  rw [stepStmts.cons_some hloop]
  have hfinal : final = BLOCK_N * nB := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    rw [Nat.add_zero] at hfin
    -- hmod : final = final / BLOCK_N * BLOCK_N ; hle : final / BLOCK_N ≤ nB ; hfin : BLOCK_N * nB ≤ final
    have hub : final ≤ BLOCK_N * nB := by
      calc final = final / BLOCK_N * BLOCK_N := hmod
        _ ≤ nB * BLOCK_N := Nat.mul_le_mul_right _ hle
        _ = BLOCK_N * nB := Nat.mul_comm _ _
    omega
  subst hfinal
  obtain ⟨sF, hpostStep, hO⟩ := attn_postLoopG Q K V B0 Out s sm_scale BLOCK_M BLOCK_N HEAD 0 BIAS_LAST_SIZE stride_b0m
    (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h) hBM hKN hHD nB hnB sL hinvL
  exact ⟨sF, hpostStep, hO⟩

set_option maxHeartbeats 1000000 in
/-- **Genuine closed-form `Out`-store correctness for `attention_kernel.py`.**
Every output lane of `_fwd_kernel_aligned` holds the closed-form base-2
streaming-softmax attention `attentionKernelSpec` (= `attnGenScore fscore vFlat`)
of the loaded Q/K/V tiles under the kernel's genuine bias-augmented per-key score
`fscore` — NOT the kernel's own executed output. At the checked Python test shape. -/
theorem attention_kernel_genuine_output_compute_correct
    (Q K V B0 Out : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out
        0.1 16384 128 1 16384 128 1 16384 128 1 16384 128 1
        16384 128 2 4 128 0 64 128 128 64 64
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [64, 128] =>
        some (Out, surfaceOutOffset s 16384 128 1 64 idx))
      (expected := fun idx : TileIndex [64, 128] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (attentionKernelSpec s Q K V B0 0.1
            (s.pids 1 * 16384) (s.pids 1 * 16384) (s.pids 1 * 16384)
            64 64 128 (64 * 2) 64 128 (s.pids 0) idx)))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  obtain ⟨sF, hstep, hO⟩ := attention_kernel_exec Q K V B0 Out s hundef
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_memcell]
  rw [show surfaceOutOffset s 16384 128 1 64 idx
        = s.pids 1 * 16384 + (s.pids 0 * 64 + idx.1.val) * 128 + idx.2.1.val * 1 from by
    simp [surfaceOutOffset, mIndex, kIndex]]
  exact hO idx

set_option maxHeartbeats 1000000 in
/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General genuine closed-form `Out`-store correctness** (dimension-parameterized).
Every output lane of `_fwd_kernel_aligned` holds the closed-form base-2
streaming-softmax attention `attentionKernelSpec` (= `attnGenScore fscore vFlat`)
under the kernel's genuine bias-augmented per-key score `fscore` — NOT the
kernel's own executed output. Genuinely general over
`BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE numKVBlocks sm_scale` and the head/bias
strides (`P_SEQ = 0`, contiguous Q/K/V/Out layout per `qRaw`/`kFlat`/`vFlat`). -/
theorem attention_kernel_genuine_output_compute_correct_general
    (Q K V B0 Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hKN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hHD : 0 < HEAD) (hnB : 1 ≤ nB)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, HEAD] =>
        some (Out, surfaceOutOffset s stride_qh HEAD 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, HEAD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (attentionKernelSpec s Q K V B0 sm_scale
            (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h)
            BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s.pids 0) idx)))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_kernel_fwd_kernel_aligned_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  obtain ⟨sF, hstep, hO⟩ := attention_kernel_exec_general Q K V B0 Out s sm_scale
    stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB hKN hBM hHD hnB hundef
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_memcell]
  rw [show surfaceOutOffset s stride_qh HEAD 1 BLOCK_M idx
        = s.pids 1 * stride_qh + (s.pids 0 * BLOCK_M + idx.1.val) * HEAD + idx.2.1.val * 1 from by
    simp [surfaceOutOffset, mIndex, kIndex]]
  exact hO idx

end ClosedForm

end Correct

end VeriTile.Bench.TritonBenchG.AttentionKernel

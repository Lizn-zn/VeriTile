import VeriTile.Triton

/-!
# `chunk_linear_attn` — strict per-kernel correctness

The upstream file holds **four** `@triton.jit` kernels: the state-recurrence
forward `chunk_linear_attn_fwd_kernel_h`, the output forward
`chunk_linear_attn_fwd_kernel_o`, the state-recurrence backward
`chunk_linear_attn_bwd_kernel_dh`, and the fused gradient backward
`chunk_linear_attn_bwd_kernel_dqkv`. This file covers the two **state-recurrence**
kernels — the file's first kernel, `chunk_linear_attn.py`'s
`chunk_linear_attn_fwd_kernel_h`, and its descending mirror `bwd_kernel_dh` —
which share one proof skeleton: a `[BK, BV]` running state is **stored to memory
at every chunk** and then advanced by one `tl.dot`.

```
fwd_h:  h[·,·,t] = H_t,   H_0 = (h0 or 0),  H_{t+1} = H_t + k_tᵀ · v_t
        (and ht = H_NT when STORE_FINAL_STATE)
bwd_dh: dh[·,·,t] = D_t,  D_NT = 0,         D_{t-1} = D_t + (scale·q_t) · do_t
```

The store-then-accumulate order matters and is carried by the spec: chunk `t`
receives the state *before* chunk `t`'s own contribution.

## The descending loop, spelled ascending

`bwd_kernel_dh` iterates `for i_t in range(NT - 1, -1, -1)`. The DSL's `forRange`
counts up (`stepForRangeAux` advances while `cur < stop`), so the faithful surface
spells the change of variable explicitly:

    for j in range(0, NT, 1) { i_t = NT - $(1) - j; … }

— the identical iteration sequence, with the descending index computed from the
ascending counter as the body's first statement. This is the established
respelling (`triton_linear_activation`'s "descending K-loop as antiquoted
ascending trip count", `completion_audit.md`), and it is observable in the
surface rather than hidden in a spec.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the host launch (the 3-D
grid `(NK, NV, B*H)` and the host-computed `NT`) is the *trusted boundary*. Every
dimension and stride stays a symbolic parameter, as do the two `constexpr` gates
`USE_INITIAL_STATE` / `STORE_FINAL_STATE` — one theorem covers all four
configurations. The `.to(...)` dtype round-trips erase to the identity at the
algorithm layer.

Spelling notes, per `bench/MAIN_THEOREM_CONVENTIONS.md`, all surface syntax
rather than semantics: integer literals inside index arithmetic are written
`$(n)`; `boundary_check=(0, 1)` is written `boundary_check=([0, 1] : List Nat)`.
-/

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn

open VeriTile.Triton

section Correct_without_Rounding

/-- Faithful transcription of `chunk_linear_attn_fwd_kernel_h`. -/
def cla_fwd_h_surface
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
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
    b_h += tl.dot(b_k, b_v, allow_tf32=false)
  }
  if STORE_FINAL_STATE {
    p_ht = tl.make_block_ptr(base=ht + i_bh * $(K) * $(V),
      shape=($(K), $(V)), strides=($(V), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), boundary_check=([0, 1] : List Nat))
  }
}

/-- The forward state surface lowers to the algorithm layer. -/
theorem cla_fwd_h_surface_toAlgorithm_supported
    (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ∃ alg, (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t T K V BT BK BV NT
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg := by
  simp [cla_fwd_h_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Faithful transcription of `chunk_linear_attn_bwd_kernel_dh`, with the
descending loop spelled as its ascending change of variable (see the preamble). -/
def cla_bwd_dh_surface
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_v = tl.program_id(1)
  i_bh = tl.program_id(2)
  b_dh = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
  for j in range($(0), $(NT), $(1)) {
    i_t = $(NT) - $(1) - j
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qk_h),
      shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_do = tl.make_block_ptr(base=do_ + i_bh * $(s_vo_h),
      shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
      offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
    p_dh = tl.make_block_ptr(base=dh + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    tl.store(p_dh, (b_dh).to(p_dh.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_q = (b_q * $(scale)).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
    b_dh += tl.dot(b_q, (b_do).to(b_q.dtype), allow_tf32=false)
  }
}

/-- The backward state surface lowers to the algorithm layer. -/
theorem cla_bwd_dh_surface_toAlgorithm_supported
    (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    ∃ alg, (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
      s_h_h s_h_t scale T K V BT BK BV NT).toAlgorithm? = Except.ok alg := by
  simp [cla_bwd_dh_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition

The algorithm-lowered statement lists, checked against the macro output by `rfl`.
One lowering worth naming: an assignment-position `.to(tl.float32)` on a load
lowers to a real `Op.castFloat FloatDType.real FloatDType.real` (not to erasure —
only *expression-interior* casts erase), whose `.real → .real` semantics is the
identity on lanes. -/

/-- The `USE_INITIAL_STATE` branch: the `h0` block pointer and the seeded load. -/
def claInitBranch (h0 : RegionName) (K V BK BV : Nat) : List Stmt :=
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

/-- The forward chunk body: three block pointers, the state **store**, the `k` and
`v` loads, and the recurrence `b_h += tl.dot(b_k, b_v)`. The store-before-load
order is the whole content of the spec's "state *before* chunk `t`". -/
def claFwdBody (k v h : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV : Nat) : List Stmt :=
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
    Stmt.assign .real [BK, BV] "b_h"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_h")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_k")
          (Op.ref .real [BT, BV] "b_v"))) ]

/-- The `STORE_FINAL_STATE` branch: the `ht` block pointer and the final store. -/
def claFinalBranch (ht : RegionName) (K V BK BV : Nat) : List Stmt :=
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

set_option maxRecDepth 8000 in
/-- **Forward body split (by `rfl`).** Seven top-level statements. -/
theorem cla_fwd_body_eq (k v h h0 ht : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (T K V BT BK BV NT : Nat) (UIS SFS : Bool) :
    (cla_fwd_h_surface k v h h0 ht s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t T K V BT BK BV NT UIS SFS).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_h" (Op.full [BK, BV] (Op.const 0)),
          Stmt.ifThen (Op.constBool UIS) (claInitBranch h0 K V BK BV),
          Stmt.forRange "i_t" 0 NT 1
            (claFwdBody k v h s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
              T K V BT BK BV),
          Stmt.ifThen (Op.constBool SFS) (claFinalBranch ht K V BK BV) ] := by
  rfl

/-- The backward chunk body: the descending index as its first statement, then the
same store-then-accumulate skeleton over `q` / `do`. -/
def claBwdBody (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "i_t"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat NT) (Op.constNat 1))
        (Op.ref .nat [] "j")),
    Stmt.assign .blockPtr [BK, BT] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
        [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BT, BV] "p_do"
      (Op.makeBlockPtrDynOffsets do_
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_vo_h))
        [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .blockPtr [BK, BV] "p_dh"
      (Op.makeBlockPtrDynOffsets dh
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
            (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.store .real [BK, BV]
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_dh") [0, 1])
      (Op.ref .real [BK, BV] "b_dh") MaskOpt.none,
    Stmt.assign .real [BK, BT] "b_q"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] "p_q") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BT] "b_q"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BK, BT] "b_q") (Op.const scale)),
    Stmt.assign .real [BT, BV] "b_do"
      (Op.load .real (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] "p_do") [0, 1])
        MaskOpt.none),
    Stmt.assign .real [BK, BV] "b_dh"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BK, BV] "b_dh")
        (Op.dot (batch := []) (Op.ref .real [BK, BT] "b_q")
          (Op.ref .real [BT, BV] "b_do"))) ]

set_option maxRecDepth 8000 in
/-- **Backward body split (by `rfl`).** Five top-level statements. -/
theorem cla_bwd_body_eq (q do_ dh : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat) :
    (cla_bwd_dh_surface q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        s_h_h s_h_t scale T K V BT BK BV NT).toAlgKernel.body
      = [ Stmt.assign .nat [] "i_k" (Op.programId 0),
          Stmt.assign .nat [] "i_v" (Op.programId 1),
          Stmt.assign .nat [] "i_bh" (Op.programId 2),
          Stmt.assign .real [BK, BV] "b_dh" (Op.full [BK, BV] (Op.const 0)),
          Stmt.forRange "j" 0 NT 1
            (claBwdBody q do_ dh s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h
              s_h_t scale T K V BT BK BV NT) ] := by
  rfl

/-! ## Closed-form specification

All accessors read the **launch** state's memory at the addresses the block
pointers compute, with the boundary checks factored into guarded forms. The chunk
index is `t`, the K lane `e`, the T lane `c`, the V lane `p`. -/

/-- `k[i_k·BK + e, t·BT + c]` (parent `(K, T)`, strides `(s_qk_d, s_qk_t)`). -/
noncomputable def kElem (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d BT BK : Nat) (t c e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_qk_h + (s.pids 0 * BK + e) * s_qk_d
    + (t * BT + c) * s_qk_t)

/-- `v[t·BT + c, i_v·BV + p]` (parent `(T, V)`, strides `(s_vo_t, s_vo_d)`). -/
noncomputable def vElem (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d BT BV : Nat) (t c p : Nat) : ℝ :=
  s.readMem v (s.pids 2 * s_vo_h + (t * BT + c) * s_vo_t
    + (s.pids 1 * BV + p) * s_vo_d)

/-- `h0[i_k·BK + e, i_v·BV + p]` (parent `(K, V)`, strides `(V, 1)`). -/
noncomputable def h0Elem (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  s.readMem h0 (s.pids 2 * K * V + (s.pids 0 * BK + e) * V
    + (s.pids 1 * BV + p) * 1)

/-- The guarded `k` lane, as `b_k` holds it. -/
noncomputable def kGuarded (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d : Nat) (T K BT BK : Nat) (t c e : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ t * BT + c < T then
    kElem s k s_qk_h s_qk_t s_qk_d BT BK t c e
  else 0

/-- The guarded `v` lane, as `b_v` holds it. -/
noncomputable def vGuarded (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d : Nat) (T V BT BV : Nat) (t c p : Nat) : ℝ :=
  if t * BT + c < T ∧ s.pids 1 * BV + p < V then
    vElem s v s_vo_h s_vo_t s_vo_d BT BV t c p
  else 0

/-- The guarded initial state, as the `USE_INITIAL_STATE` branch leaves `b_h`. -/
noncomputable def h0Guarded (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) (e p : Nat) : ℝ :=
  if s.pids 0 * BK + e < K ∧ s.pids 1 * BV + p < V then
    h0Elem s h0 K V BK BV e p
  else 0

/-- One chunk's contribution to state lane `(e, p)`: `Σ_c k[e, c] · v[c, p]`. -/
noncomputable def claContrib (s : BlockState) (k v : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t c.val e
      * vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The forward state at chunk `t`** — what the kernel stores into `h[·,·,t]`:
the (gated) initial state plus every chunk *before* `t`. -/
noncomputable def claHState (s : BlockState) (k v h0 : RegionName)
    (UIS : Bool) (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  (if UIS then h0Guarded s h0 K V BK BV e p else 0)
    + ∑ u : Fin t, claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV u.val e p

/-- One chunk's contribution to the backward state lane `(e, p)`:
`Σ_c (scale · q)[e, c] · do[c, p]` — `q` shares `k`'s layout, `do` shares `v`'s. -/
noncomputable def claBContrib (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV : Nat) (t e p : Nat) : ℝ :=
  ∑ c : Fin BT,
    (kGuarded s q s_qk_h s_qk_t s_qk_d T K BT BK t c.val e * scale)
      * vGuarded s do_ s_vo_h s_vo_t s_vo_d T V BT BV t c.val p

/-- **The backward state at chunk `t`** — what the kernel stores into `dh[·,·,t]`:
every chunk *after* `t` (the loop descends, and each chunk is stored before it is
accumulated). -/
noncomputable def claDhState (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (t e p : Nat) : ℝ :=
  ∑ u : Fin NT, if t < u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0

/-- The `h` / `dh` chunk-`t` store address for lane `(e, p)`. -/
def claHOffset (s : BlockState) (s_h_h s_h_t K V BK BV : Nat) (t : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * s_h_h + t * K * V + (s.pids 0 * BK + idx.1.val) * s_h_t
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- The `ht` store address for lane `(e, p)`. -/
def claHtOffset (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  s.pids 2 * K * V + (s.pids 0 * BK + idx.1.val) * V
    + (s.pids 1 * BV + idx.2.1.val) * 1

/-- A state lane is *active* when it maps inside the `K × V` window. -/
def claActive (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Prop :=
  s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BV + idx.2.1.val < V

/-! ## Eval recipes

Local copies (bench files never import each other), plus the two lowerings this
pair adds: the `.real → .real` `castFloat` identity and the descending-index
arithmetic `NT - 1 - j`. -/

private theorem cla_makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
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
private theorem cla_load_bp_2d (rg : RegionName) (s : BlockState) (name : RegName)
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
private theorem cla_mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- Scalar `(name * cB) * cC` — the `h0` / `ht` base `i_bh * K * V`. -/
private theorem cla_mulMulConst_eval (s : BlockState) (name : RegName)
    (val cB cC : Nat) (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat cB))
        (Op.constNat cC)) s
      = some (Tile.scalar (val * cB * cC)) := by
  rw [evalOp_mul, cla_mulConst_eval s name val cB hr]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The `p_h` / `p_dh` base: `i_bh * s_h_h + i_t * K * V`. -/
private theorem cla_hBase_eval (s : BlockState) (s_h_h K V : Nat) (ibh it : Nat)
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar ibh))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar it)) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
          (Op.constNat V))) s
      = some (Tile.scalar (ibh * s_h_h + it * K * V)) := by
  rw [evalOp_add, cla_mulConst_eval s "i_bh" ibh s_h_h hibh, evalOp_mul,
    cla_mulConst_eval s "i_t" it K hit]
  simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- The descending change of variable: `i_t = NT - 1 - j` on `nat` scalars. -/
private theorem cla_itIdx_eval (t : BlockState) (NT c : Nat)
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

/-- The assignment-position `.to(tl.float32)` on an already-`.real` operand: a
genuine `Op.castFloat .real .real` node whose semantics is the identity. -/
private theorem cla_castFloat_eval {sh : TileShape} (x : Op .real sh)
    (t : BlockState) (vx : Tile .real sh) (hx : evalOp x t = some vx) :
    evalOp (Op.castFloat FloatDType.real FloatDType.real x) t = some vx := by
  rw [evalOp_castFloat,
    show @evalOp FloatDType.real.toTileDType sh x t = some vx from hx]
  rfl

/-- `tl.zeros([BK, BV], dtype=tl.float32)`. -/
private theorem cla_zeros_eval (BK BV : Nat) (t : BlockState) :
    evalOp (Op.full [BK, BV] (Op.const 0)) t
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [BK, BV]) := by
  simp [evalOp_full, evalOp_const]

/-- `tl.dot` at rank 2, `erw`-only shapes. -/
private theorem cla_dot_eval {M K N : Nat} (x : Op .real [M, K]) (y : Op .real [K, N])
    (t : BlockState) (vx : Tile .real [M, K]) (vy : Tile .real [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dot (batch := []) x y) t = some (Tile.dot [] vx vy) := by
  erw [evalOp_dot, hx, hy]
  rfl

/-- A `WithBot ℝ` sum of `some`s is `some` of the real sum. -/
private theorem cla_withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ))
    = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- 2D dot element collapse for all-`some` operands. -/
private theorem cla_dot2d_elem {M K N : Nat} (a : Tile .real [M, K])
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
  exact cla_withBot_sum_some _

/-- `Stmt.ifThen`'s step equation (well-founded recursion, so named). -/
private theorem cla_ifThen_step (cond : Op .bool []) (body : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThen cond body) t
      = (evalOp cond t).bind
          (fun c => if c.data PUnit.unit then stepStmts body t else some t) := by
  unfold stepStmt
  cases evalOp cond t <;> rfl

/-- `setReg` leaves memory alone, at function level. -/
private theorem cla_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-! ## Value tiles and load bridges

The four loads (`k`/`v` forward; `q`/`do` backward reuse the same two bridges —
`q` shares `k`'s layout and `do` shares `v`'s, with only the region name and
launch strides substituted). -/

/-- The loaded `b_k` tile of chunk `t`: cell `(e, c)` holds the guarded `k` lane. -/
noncomputable def claBkTile (s : BlockState) (k : RegionName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat) : Tile .real [BK, BT] :=
  ⟨fun idx => some (kGuarded s k s_qk_h s_qk_t s_qk_d T K BT BK t
    idx.2.1.val idx.1.val)⟩

/-- The loaded `b_v` tile of chunk `t`: cell `(c, p)` holds the guarded `v` lane. -/
noncomputable def claBvTile (s : BlockState) (v : RegionName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat) : Tile .real [BT, BV] :=
  ⟨fun idx => some (vGuarded s v s_vo_h s_vo_t s_vo_d T V BT BV t
    idx.1.val idx.2.1.val)⟩

/-- `b_k` (or `b_q`) load lands on `claBkTile` (memory matched to launch `s`). -/
private theorem cla_kLoad_eq (s sin : BlockState) (k : RegionName) (name : RegName)
    (s_qk_h s_qk_t s_qk_d T K BT BK : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem k off = s.readMem k off)
    (hpk : sin.regs .blockPtr [BK, BT] name = some
      ⟨fun _ => BlockPtr.mk k (s.pids 2 * s_qk_h) [K, T] [BK, BT] [s_qk_d, s_qk_t]
        [s.pids 0 * BK, t * BT]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BT] name) [0, 1]) MaskOpt.none) sin
      = some (claBkTile s k s_qk_h s_qk_t s_qk_d T K BT BK t) := by
  rw [cla_load_bp_2d k sin name (s.pids 2 * s_qk_h) K T BK BT s_qk_d s_qk_t
    (s.pids 0 * BK) (t * BT) hpk]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, c, u⟩ := idx
  simp only [claBkTile, kGuarded, kElem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ t * BT + c.val < T
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-- `b_v` (or `b_do`) load lands on `claBvTile`. -/
private theorem cla_vLoad_eq (s sin : BlockState) (v : RegionName) (name : RegName)
    (s_vo_h s_vo_t s_vo_d T V BT BV : Nat) (t : Nat)
    (hmem : ∀ off, sin.readMem v off = s.readMem v off)
    (hpv : sin.regs .blockPtr [BT, BV] name = some
      ⟨fun _ => BlockPtr.mk v (s.pids 2 * s_vo_h) [T, V] [BT, BV] [s_vo_t, s_vo_d]
        [t * BT, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT, BV] name) [0, 1]) MaskOpt.none) sin
      = some (claBvTile s v s_vo_h s_vo_t s_vo_d T V BT BV t) := by
  rw [cla_load_bp_2d v sin name (s.pids 2 * s_vo_h) T V BT BV s_vo_t s_vo_d
    (t * BT) (s.pids 1 * BV) hpv]
  refine congrArg some ?_
  ext idx
  obtain ⟨c, p, u⟩ := idx
  simp only [claBvTile, vGuarded, vElem, hmem]
  by_cases hb : t * BT + c.val < T ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-- The seeded `b_h` tile: cell `(e, p)` holds the guarded `h0` lane. -/
noncomputable def claH0Tile (s : BlockState) (h0 : RegionName)
    (K V BK BV : Nat) : Tile .real [BK, BV] :=
  ⟨fun idx => some (h0Guarded s h0 K V BK BV idx.1.val idx.2.1.val)⟩

/-- The `h0` load lands on `claH0Tile`. -/
private theorem cla_h0Load_eq (s sin : BlockState) (h0 : RegionName)
    (K V BK BV : Nat)
    (hmem : ∀ off, sin.readMem h0 off = s.readMem h0 off)
    (hph0 : sin.regs .blockPtr [BK, BV] "p_h0" = some
      ⟨fun _ => BlockPtr.mk h0 (s.pids 2 * K * V) [K, V] [BK, BV] [V, 1]
        [s.pids 0 * BK, s.pids 1 * BV]⟩) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] "p_h0") [0, 1]) MaskOpt.none) sin
      = some (claH0Tile s h0 K V BK BV) := by
  rw [cla_load_bp_2d h0 sin "p_h0" (s.pids 2 * K * V) K V BK BV V 1
    (s.pids 0 * BK) (s.pids 1 * BV) hph0]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, p, u⟩ := idx
  simp only [claH0Tile, h0Guarded, h0Elem, hmem]
  by_cases hb : s.pids 0 * BK + e.val < K ∧ s.pids 1 * BV + p.val < V
  · rw [if_pos hb, if_pos hb]
  · rw [if_neg hb, if_neg hb]

/-! ## The generic state-block store

All three stores (`h[·,·,t]` per chunk, `ht` once, `dh[·,·,t]` per chunk) are the
same shape: a `[BK, BV]` block of a `(K, V)` parent at strides `(σ, 1)` and
offsets `(i_k·BK, i_v·BV)`, differing only in base and stride, so one lemma pair
serves all three. -/

/-- The state-block store address at lane `(e, p)`: `base + row·σ + col`. -/
def claStoreAddr (s : BlockState) (base σ BK BV : Nat)
    (idx : TileIndex [BK, BV]) : Nat :=
  base + (s.pids 0 * BK + idx.1.val) * σ + (s.pids 1 * BV + idx.2.1.val) * 1

theorem claHOffset_eq_storeAddr (s : BlockState) (s_h_h s_h_t K V BK BV t : Nat)
    (idx : TileIndex [BK, BV]) :
    claHOffset s s_h_h s_h_t K V BK BV t idx
      = claStoreAddr s (s.pids 2 * s_h_h + t * K * V) s_h_t BK BV idx := rfl

theorem claHtOffset_eq_storeAddr (s : BlockState) (K V BK BV : Nat)
    (idx : TileIndex [BK, BV]) :
    claHtOffset s K V BK BV idx
      = claStoreAddr s (s.pids 2 * K * V) V BK BV idx := rfl

/-- The explicit post-store state: the masked lane-by-lane scatter of cell fn `f`
over input state `sin` (value and mask built from launch `s`). -/
noncomputable def claStoreState (s sin : BlockState) (rg : RegionName)
    (base σ K V BK BV : Nat) (f : Nat → Nat → ℝ) : BlockState :=
  (TileShape.allIndices [BK, BV]).foldl
    (fun acc i => if (s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
        then acc.writeMem rg (claStoreAddr s base σ BK BV i)
          (f i.1.val i.2.1.val) else acc) sin

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **State-block store step (eq).** Stepping the block-ptr store of tile `bT`
(cell fn `f`) through pointer register `pname` yields `claStoreState`. -/
theorem claStore_step_eq (s sin : BlockState) (rg : RegionName)
    (bname pname : RegName) (base σ K V BK BV : Nat)
    (f : Nat → Nat → ℝ) (bT : Tile .real [BK, BV])
    (hbf : ∀ e p, bT.data (e, p, PUnit.unit) = some (f e.val p.val))
    (hb : sin.regs .real [BK, BV] bname = some bT)
    (hp : sin.regs .blockPtr [BK, BV] pname = some
      ⟨fun _ => BlockPtr.mk rg base [K, V] [BK, BV] [σ, 1]
        [s.pids 0 * BK, s.pids 1 * BV]⟩) :
    stepStmt (Stmt.store .real [BK, BV]
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BV] pname) [0, 1])
        (Op.ref .real [BK, BV] bname) MaskOpt.none) sin
      = some (claStoreState s sin rg base σ K V BK BV f) := by
  unfold stepStmt claStoreState
  simp only [evalOp_ref, hb, hp]
  refine congrArg some
    (congrArg (fun g => List.foldl g sin (TileShape.allIndices [BK, BV])) ?_)
  funext acc i
  obtain ⟨e, p, u⟩ := i
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets, Bool.true_and, claStoreAddr]
  by_cases hbnd : s.pids 0 * BK + e.val < K ∧ s.pids 1 * BV + p.val < V
  · simp only [hbnd, BlockState.writeMemTyped_real, hbf, Nat.mul_one]
    rfl
  · simp only [hbnd, decide_false, Bool.false_eq_true, if_false]

/-- If two `Q`-blocks with in-block offsets `A, B < Q` collide, the block indices
agree. -/
private theorem cla_block_index_inj {Q j c A B : Nat} (hA : A < Q) (hB : B < Q)
    (heq : j * Q + A = c * Q + B) : j = c := by
  have hQ : 0 < Q := Nat.lt_of_le_of_lt (Nat.zero_le A) hA
  have hj : (j * Q + A) / Q = j := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.add_zero]
  have hc : (c * Q + B) / Q = c := by
    rw [Nat.mul_comm, Nat.mul_add_div hQ, Nat.div_eq_of_lt hB, Nat.add_zero]
  rw [← hj, heq, hc]

/-- The store addresses of one block are pairwise distinct once a whole `BV` row
segment fits under the row stride. -/
theorem claStoreAddr_injective (s : BlockState) (base σ BK BV : Nat)
    (hBVσ : BV ≤ σ) :
    Function.Injective (claStoreAddr s base σ BK BV) := by
  rintro ⟨e, p, u⟩ ⟨e', p', u'⟩ heq
  simp only [claStoreAddr] at heq
  have hp := p.isLt
  have hp' := p'.isLt
  have h2 : (s.pids 0 * BK + e.val) * σ + p.val
      = (s.pids 0 * BK + e'.val) * σ + p'.val := by omega
  have hlt : p.val < σ := by omega
  have hlt' : p'.val < σ := by omega
  have hjj : s.pids 0 * BK + e.val = s.pids 0 * BK + e'.val :=
    cla_block_index_inj hlt hlt' h2
  have he : e = e' := Fin.ext (by omega)
  have hpv : p = p' := Fin.ext (by
    have hσ : (s.pids 0 * BK + e.val) * σ = (s.pids 0 * BK + e'.val) * σ := by
      rw [hjj]
    omega)
  subst he
  subst hpv
  rfl

set_option maxHeartbeats 4000000 in
/-- **State-block store readback.** `claStoreState` writes `f` at every active
lane, leaves other regions alone, and leaves same-region offsets alone when they
avoid every *active*-lane address (mask-restricted, so no pid pinning needed). -/
theorem claStore_step_props (s sin : BlockState) (rg : RegionName)
    (base σ K V BK BV : Nat) (f : Nat → Nat → ℝ)
    (hInj : Function.Injective (claStoreAddr s base σ BK BV)) :
    (claStoreState s sin rg base σ K V BK BV f).pids = sin.pids
      ∧ (claStoreState s sin rg base σ K V BK BV f).regs = sin.regs
      ∧ (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg
              (claStoreAddr s base σ BK BV idx)
            = f idx.1.val idx.2.1.val)
      ∧ (∀ rg' off, rg' ≠ rg →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg' off
            = sin.readMem rg' off)
      ∧ (∀ off, (∀ idx : TileIndex [BK, BV], claActive s K V BK BV idx →
            off ≠ claStoreAddr s base σ BK BV idx) →
          (claStoreState s sin rg base σ K V BK BV f).readMem rg off
            = sin.readMem rg off) := by
  classical
  unfold claStoreState
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [BlockState.foldl_writeMem_prop_masked_pids]
  · funext dtype shape name
    rw [BlockState.foldl_writeMem_prop_masked_regs]
  · intro idx hidx
    obtain ⟨h1, h2⟩ := hidx
    have h := BlockState.scatter_readback_prop_masked_nd (region := rg) sin
      (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      hInj idx
    rw [h, if_pos ⟨h1, h2⟩]
  · intro rg' off hrg
    exact BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      rg (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      _ sin rg' off hrg
  · intro off hoff
    exact BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
      rg (claStoreAddr s base σ BK BV) (fun i => f i.1.val i.2.1.val)
      (fun i => s.pids 0 * BK + i.1.val < K ∧ s.pids 1 * BV + i.2.1.val < V)
      _ sin off (fun i _ hPi => hoff i hPi)

/-! ### Cross-chunk block disjointness -/

/-- Distinct chunks write disjoint `K*V` blocks of `h` / `dh`: active lanes of
chunk `j` never collide with active lanes of chunk `c ≠ j`. The block-fit
hypothesis `(K-1)·s_h_t + V ≤ K·V` is an equality under the launcher's
contiguous `s_h_t = V`. -/
theorem claHOffset_chunk_disjoint (s : BlockState)
    (s_h_h s_h_t K V BK BV : Nat) (hFit : (K - 1) * s_h_t + V ≤ K * V)
    (j c : Nat) (hjc : j ≠ c) (idxj idxc : TileIndex [BK, BV])
    (hja : claActive s K V BK BV idxj) (hca : claActive s K V BK BV idxc) :
    claHOffset s s_h_h s_h_t K V BK BV j idxj
      ≠ claHOffset s s_h_h s_h_t K V BK BV c idxc := by
  obtain ⟨ej, pj, _⟩ := idxj
  obtain ⟨ec, pc, _⟩ := idxc
  obtain ⟨hjK, hjV⟩ := hja
  obtain ⟨hcK, hcV⟩ := hca
  simp only [claHOffset]
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
  exact cla_block_index_inj
    (by have := hbound _ _ hjK hjV; omega)
    (by have := hbound _ _ hcK hcV; omega)
    heq2

/-! ## Spec-layer recurrences -/

theorem claHState_zero (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (e p : Nat) :
    claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV 0 e p
      = (if UIS then h0Guarded s h0 K V BK BV e p else 0) := by
  simp [claHState]

theorem claHState_succ (s : BlockState) (k v h0 : RegionName) (UIS : Bool)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat)
    (T K V BT BK BV : Nat) (t e p : Nat) :
    claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
        T K V BT BK BV (t + 1) e p
      = claHState s k v h0 UIS s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
          T K V BT BK BV t e p
        + claContrib s k v s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d
            T K V BT BK BV t e p := by
  simp [claHState, Fin.sum_univ_castSucc, add_assoc]

/-- The backward running state after `c` descending iterations: every chunk from
`NT - c` up. Stored (before advancing) at chunk `NT - 1 - c`. -/
noncomputable def claDhCarry (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) : ℝ :=
  ∑ u : Fin NT, if NT - c ≤ u.val then
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV u.val e p
  else 0

theorem claDhCarry_zero (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (e p : Nat) :
    claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT 0 e p = 0 := by
  refine Finset.sum_eq_zero fun u _ => ?_
  rw [if_neg]
  have := u.isLt
  omega

theorem claDhCarry_succ (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) (hc : c < NT) :
    claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT (c + 1) e p
      = claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BT BK BV NT c e p
        + claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
            T K V BT BK BV (NT - 1 - c) e p := by
  classical
  unfold claDhCarry
  set B : Nat → ℝ := fun uv =>
    claBContrib s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
      T K V BT BK BV uv e p with hB
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

/-- What the backward kernel stores at chunk `NT - 1 - c` is exactly the spec's
`claDhState` there: the strictly-later chunks. -/
theorem claDhState_eq_carry (s : BlockState) (q do_ : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (T K V BT BK BV NT : Nat) (c e p : Nat) (hc : c < NT) :
    claDhState s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
        T K V BT BK BV NT (NT - 1 - c) e p
      = claDhCarry s q do_ s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d scale
          T K V BT BK BV NT c e p := by
  unfold claDhState claDhCarry
  exact Finset.sum_congr rfl fun u _ => if_congr (by omega) rfl rfl

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkLinearAttn

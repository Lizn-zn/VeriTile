import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `bmm_chunk_bwd` — closed-form batched-matmul-backward correctness

`_bmm_chunk_bwd_kernel` is the chunked batched matmul backward of Mamba-style
SSMs: for each `(m, n)` program in a `(batch, chunk·group)` grid it accumulates
`acc += tl.dot(dout, a)` over `cdiv(chunk_size_limit, BLOCK_SIZE_CS)` chunk-size
blocks (with per-block tail masking and a `chunk_size`/`K` row/col mask on the
loads), optionally adds a residual tile (`HAS_RESIDUAL`), downcasts to the `db`
dtype, and stores the `BLOCK_SIZE_M×BLOCK_SIZE_N` gradient tile to `db` under a
`(m<chunk_size_limit)&(n<K)` mask.

This file proves the **full kernel** for the genuine batched-matmul-backward
configuration (`HAS_RESIDUAL = false` — Python test case 1) correct against a
genuine mathematical reference: every active output cell
`Db[batch, chunk, head, i, j]` of the computed tile equals
`Σ_{cs < CSL} Dout[batch,chunk,head, i, cs] · A[batch,chunk,head, cs, j]`
over `ℝ`, where `CSL = BLOCK_SIZE_CS · numCSBlocks` is the contracted
chunk-size dimension. This is NOT the kernel's own emitted value — it is the
independent closed-form per-program `Σ_cs Dout·A` batched-matmul-backward
reference (`dB = doutᵀ-contracted-with-a`), derived from the loaded `Dout`/`A`
tiles under the kernel's own strided batch/chunk/head/row/col addressing.

The contraction in `tl.dot(dout, a)` runs over the chunk-size axis (`offs_cs`):
`dout` is the `[BLOCK_SIZE_M, BLOCK_SIZE_CS]` tile (row = output row `m` with
stride `stride_dout_csize_n`, col = chunk position `cs` with stride
`stride_dout_csize_m`); `a` is the `[BLOCK_SIZE_CS, BLOCK_SIZE_N]` tile (row =
chunk position `cs` with stride `stride_a_seqlen`, col = output col `n` with
stride `stride_ak`).

## Proof architecture

```
bmm_chunk_bwd_closed_form_correct                 ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ bbwd_exec_closed_form                        ← exec-side closed form (every active cell = ∑_cs Dout·A)
       ├─ bbwd_preLoop      (P 0: acc = 0, dout/a pointers seeded with batch offset)
       ├─ bbwd_step         (one CS-block: acc += tl.dot advances the partial sum)
       ├─ bbwd_loop         (forRangeDyn drives the CS-loop via forRangeAux_inv)
       └─ bbwd_postLoop     (final masked store = the closed form)
```

`bmm_chunk_bwd_surface_toAlgorithm_supported` (and the two per-case lowerings)
additionally witness that the **fully general** surface — with optional residual
add and the destination dtype cast — lowers to the algorithm layer for every
constexpr combination.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the runtime `dot_dtype` and
`db_ptr.dtype.element_ty` casts collapse to the real carrier).
`@triton.autotune` / `num_warps` / `num_stages` are not modeled. The host launch
(the 3D grid over m·n tiles / batch / chunk·groups, and how per-program writes
compose into one buffer) is the trusted boundary; the per-program statement is
universally quantified over `s`, so it covers every program of the grid. The
contracted bound `chunk_size_limit = min(chunk_size, seqlen - pid_c·chunk_size)`
is supplied to the genuine surface as the precomputed Nat `CSL = BCS·numCSBlocks`
(the runtime `min`). The layout contract is the kernel's own strided pointer
arithmetic:
`dout[i,cs]` at `Dout + batchOff_dout + (PM·BM+i)·stride_dout_csize_n + cs·stride_dout_csize_m`,
`a[cs,j]` at `A + batchOff_a + cs·stride_a_seqlen + (PN·BN+j)·stride_ak`,
`db[i,j]` at `Db + batchOff_db + (PM·BM+i)·stride_db_seqlen + (PN·BN+j)·stride_db_k`,
where `batchOff_a/db = pid_b·stride_batch + pid_c·chunk_size·stride_seqlen +
pid_h·stride_head` and `batchOff_dout = pid_b·stride_dout_batch +
pid_c·stride_dout_chunk + pid_h·stride_dout_head`.

Preconditions for the general theorem: `0 < BLOCK_SIZE_CS` (so the CS-tail load
mask is all-true given `CSL = BCS·numCSBlocks`); all tile rows/cols in-bounds
(`PM·BM+i < CSL ≤ chunk_size`, `PN·BN+j < K`, making both load masks and the
store mask all-true); output-address injectivity; clean initial `undef`.
-/

namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## Fully general surface (all constexpr configurations) -/

/-- Faithful transcription of `bmm_chunk_bwd.py`'s `_bmm_chunk_bwd_kernel`.

This covers both `HAS_RESIDUAL=false` and `HAS_RESIDUAL=true`. The source
kernel's runtime `dot_dtype` cast is preserved as a surface dtype annotation on
the `dout`/`a` loads; the destination `db_ptr.dtype.element_ty` cast collapses to
the real carrier. The contracted `chunk_size_limit` is supplied as the explicit
`chunk_size_limit` parameter (the precomputed `min(chunk_size, seqlen -
pid_c·chunk_size)`). -/
def bmm_chunk_bwd_surface
    (A Dout db_ptr Res : RegionName)
    (chunk_size chunk_size_limit K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType)
    (HAS_RESIDUAL : Bool) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n

  A += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  Dout += pid_b * $(stride_dout_batch) +
    pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
  dout_ptrs = Dout +
    offs_m[:, None] * $(stride_dout_csize_n) +
    offs_cs[None, :] * $(stride_dout_csize_m)
  a_ptrs = A +
    offs_cs[:, None] * $(stride_a_seqlen) +
    offs_n[None, :] * $(stride_ak)

  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for cs in range($(0), tl.cdiv($(chunk_size_limit), $(BLOCK_SIZE_CS)), $(1)) {
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_cs[None, :] < $(chunk_size_limit) - cs * $(BLOCK_SIZE_CS)),
      other=0.0).to(dot_dtype)
    a = tl.load(a_ptrs,
      mask=(offs_cs[:, None] < $(chunk_size_limit) - cs * $(BLOCK_SIZE_CS)) &
        (offs_n[None, :] < $(K)), other=0.0).to(dot_dtype)
    acc += tl.dot(dout, a)
    dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
    a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
  }

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  if HAS_RESIDUAL {
    Res += pid_b * $(stride_res_batch) +
      pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
    res_ptrs = Res +
      offs_m[:, None] * $(stride_res_seqlen) +
      offs_n[None, :] * $(stride_res_k)
    res = tl.load(res_ptrs, mask=(offs_m[:, None] < $(chunk_size_limit)) &
      (offs_n[None, :] < $(K))).to(tl.float32)
    acc += res
  }

  db = (acc).to(db_ptr.dtype.element_ty)
  db_ptr += pid_b * $(stride_db_batch) +
    pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
  db_ptrs = db_ptr + offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k)
  tl.store(db_ptrs, db, mask=(offs_m[:, None] < $(chunk_size_limit)) &
    (offs_n[None, :] < $(K)))
}

/-- The full BMM chunk backward surface lowers to the algorithm layer, including
the residual add and the destination dtype cast. -/
theorem bmm_chunk_bwd_surface_toAlgorithm_supported
    (A Dout db_ptr Res : RegionName)
    (chunk_size chunk_size_limit K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      stride_res_batch stride_res_seqlen stride_res_head stride_res_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat)
    (dot_dtype : TileDType)
    (HAS_RESIDUAL : Bool) :
    ∃ alg, (bmm_chunk_bwd_surface A Dout db_ptr Res chunk_size chunk_size_limit K
      ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m
      stride_dout_csize_n stride_db_batch stride_db_seqlen stride_db_head
      stride_db_k stride_res_batch stride_res_seqlen stride_res_head
      stride_res_k BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS dot_dtype
      HAS_RESIDUAL).toAlgorithm? = Except.ok alg := by
  simp [bmm_chunk_bwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Genuine batched-matmul-backward surface (`HAS_RESIDUAL = false`)

This is the faithful transcription of `_bmm_chunk_bwd_kernel` for the pure
batched-matmul-backward configuration (Python test case 1): no residual add.
Since `db_ptr.dtype.element_ty` collapses to the real carrier the `db` store is
the real-valued accumulator. -/

/-- The genuine batched-matmul-backward surface: chunked `acc += tl.dot(dout, a)`
with the kernel's batch/chunk/head pointer offsets, CS-block dot loop, per-block
CS-tail and `chunk_size`/`K` row/col load masks, and the masked output store. -/
def bbwd_matmul_surface
    (A Dout Db : RegionName)
    (chunk_size CSL K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  Dout += pid_b * $(stride_dout_batch) +
    pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
  A += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
  dout_ptrs = Dout + offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m)
  a_ptrs = A + offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for cs in range($(0), tl.cdiv($(CSL), $(BLOCK_SIZE_CS)), $(1)) {
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_cs[None, :] < $(CSL) - cs * $(BLOCK_SIZE_CS)), other=0.0)
    a = tl.load(a_ptrs, mask=(offs_cs[:, None] < $(CSL) - cs * $(BLOCK_SIZE_CS)) &
      (offs_n[None, :] < $(K)), other=0.0)
    acc += tl.dot(dout, a)
    dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
    a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  db = (acc).to(Db.dtype.element_ty)
  Db += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
  db_ptrs = Db + $(stride_db_seqlen) * offs_m[:, None] + offs_n[None, :] * $(stride_db_k)
  tl.store(db_ptrs, db, mask=(offs_m[:, None] < $(CSL)) &
    (offs_n[None, :] < $(K)))
}

/-- The genuine batched-matmul-backward surface lowers to the algorithm layer. -/
theorem bbwd_matmul_surface_toAlgorithm_supported
    (A Dout Db : RegionName)
    (chunk_size CSL K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ∃ alg, (bbwd_matmul_surface A Dout Db chunk_size CSL K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS).toAlgorithm? = Except.ok alg := by
  simp [bbwd_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers -/

theorem evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

theorem evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

theorem evalOp_ptrAdd {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-! ## Index/value abbreviations for the batched-matmul-backward spec -/

/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
def cdiv (a b : Nat) : Nat := (a + b - 1) / b

/-- The kernel's `num_pid_n = cdiv(K, BLOCK_SIZE_N)`. -/
def numPidN (K BN : Nat) : Nat := cdiv K BN

/-- `pid_c = pid_ch // ngroups` (program axis 2 split by `ngroups`). -/
def pidC (PCH ngroups : Nat) : Nat := PCH / ngroups

/-- `pid_h = pid_ch - pid_c · ngroups`. -/
def pidH (PCH ngroups : Nat) : Nat := PCH - PCH / ngroups * ngroups

/-- `pid_m = program_id(0) // num_pid_n`. -/
def pidM (P0 K BN : Nat) : Nat := P0 / numPidN K BN

/-- `pid_n = program_id(0) % num_pid_n`. -/
def pidN (P0 K BN : Nat) : Nat := P0 % numPidN K BN

/-- Global chunk row of tile lane `i`: `PM · BM + i`. -/
def rowIndex (PM BM : Nat) (i : Fin BM) : Nat := PM * BM + i.val

/-- Global output col of tile lane `j`: `PN · BN + j`. -/
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val

/-- The kernel's `a`/`db` batch+chunk+head base offset (with `chunk_size`):
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head

/-- The kernel's `dout` batch+chunk+head base offset (no `chunk_size` multiply):
`pid_b·stride_dout_batch + pid_c·stride_dout_chunk + pid_h·stride_dout_head`. -/
def doutOff (PB PC PH SDB SDC SDH : Nat) : Nat :=
  PB * SDB + PC * SDC + PH * SDH

/-- `Dout[i, cs] = readMem Dout (doutOff + (PM·BM+i)·stride_dout_csize_n + cs·stride_dout_csize_m)`. -/
noncomputable def doutElem (s : BlockState) (Dout : RegionName)
    (PB PC PH PM BM SDB SDC SDH SDN SDM : Nat) (i : Fin BM) (cs : Nat) : ℝ :=
  s.readMem Dout (doutOff PB PC PH SDB SDC SDH + rowIndex PM BM i * SDN + cs * SDM)

/-- `A[cs, j] = readMem A (batchOff_a + cs·stride_a_seqlen + (PN·BN+j)·stride_ak)`. -/
noncomputable def aElem (s : BlockState) (A : RegionName)
    (PB PC PH PN BN chunk_size SAB SAS SAH SAK : Nat) (j : Fin BN) (cs : Nat) : ℝ :=
  s.readMem A (batchOff PB PC PH chunk_size SAB SAS SAH + cs * SAS + colIndex PN BN j * SAK)

/-- **Genuine batched-matmul-backward spec**:
`Db[i,j] = Σ_{cs < BLOCK_CS·numCSBlocks} Dout[i,cs] · A[cs,j]`. -/
noncomputable def bbwdSpec (s : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK
      BLOCK_CS numCSBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_CS * numCSBlocks)).sum
    (fun cs => doutElem s Dout PB PC PH PM BM SDB SDC SDH SDN SDM i cs
      * aElem s A PB PC PH PN BN chunk_size SAB SAS SAH SAK j cs)

/-- Partial accumulator after `c` CS-blocks: `Σ_{cs < c·BLOCK_CS} Dout·A`. -/
noncomputable def accPartial (s : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK
      BLOCK_CS : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_CS)).sum
    (fun cs => doutElem s Dout PB PC PH PM BM SDB SDC SDH SDN SDM i cs
      * aElem s A PB PC PH PN BN chunk_size SAB SAS SAH SAK j cs)

/-- One-block step of the partial accumulator. -/
theorem accPartial_succ (s : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK
      BLOCK_CS : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) :
    accPartial s Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BLOCK_CS i j (c + 1)
      = accPartial s Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BLOCK_CS i j c
        + (Finset.univ.sum fun e : Fin BLOCK_CS =>
            doutElem s Dout PB PC PH PM BM SDB SDC SDH SDN SDM i (c * BLOCK_CS + e.val)
              * aElem s A PB PC PH PN BN chunk_size SAB SAS SAH SAK j (c * BLOCK_CS + e.val)) := by
  unfold accPartial
  have h : (c + 1) * BLOCK_CS = c * BLOCK_CS + BLOCK_CS := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e => doutElem s Dout PB PC PH PM BM SDB SDC SDH SDN SDM i (c * BLOCK_CS + e)
        * aElem s A PB PC PH PN BN chunk_size SAB SAS SAH SAK j (c * BLOCK_CS + e)]

/-! ## Pointer / load / dot eval lemmas -/

/-- The scalar `dout` base-pointer add `Dout + doutOff` (no `chunk_size` multiply)
evaluates to the scalar pointer tile `(Dout, doutOff)`. -/
theorem doutptr_base_eval (s : BlockState) (Dout : RegionName) (PB PC PH SDB SDC SDH : Nat)
    (hpb : s.regs .nat [] "pid_b" = some (Tile.scalar PB))
    (hpc : s.regs .nat [] "pid_c" = some (Tile.scalar PC))
    (hph : s.regs .nat [] "pid_h" = some (Tile.scalar PH)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Dout)
      (Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SDB))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SDC)))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SDH)))) s
      = some (Tile.scalar (Dout.cast, doutOff PB PC PH SDB SDC SDH)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hpb, hpc, hph,
    Option.bind, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, doutOff]
  ring_nf

/-- The scalar `a` base-pointer add `A + batchOff` (with `chunk_size` multiply)
evaluates to the scalar pointer tile `(A, batchOff)`. -/
theorem aptr_base_eval (s : BlockState) (A : RegionName) (PB PC PH chunk_size SAB SAS SAH : Nat)
    (hpb : s.regs .nat [] "pid_b" = some (Tile.scalar PB))
    (hpc : s.regs .nat [] "pid_c" = some (Tile.scalar PC))
    (hph : s.regs .nat [] "pid_h" = some (Tile.scalar PH)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase A)
      (Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SAB))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SAS)))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SAH)))) s
      = some (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hpb, hpc, hph,
    Option.bind, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, batchOff]
  ring_nf

/-- `dout_ptrs` eval from a scalar `Dout` register: cell `(i,e) = (Dout, base + offs_m i · SDN + e · SDM)`. -/
theorem doutptrs_eval (s : BlockState) (Dout : RegionName) (M CS SDN SDM base : Nat) (gm : Fin M → Nat)
    (hbase : s.regs .ptr [] "Dout" = some (Tile.scalar (Dout.cast, base)))
    (hm : s.regs .nat [M] "offs_m" = some (Tile.vec gm))
    (hcs : s.regs .nat [CS] "offs_cs" = some (Tile.vec (fun e : Fin CS => e.val))) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Dout")
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m")) (Op.constNat SDN)))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [CS] "offs_cs")) (Op.constNat SDM))) s
      = some (⟨fun idx : TileIndex [M, CS] => (Dout.cast, base + gm idx.1 * SDN + idx.2.1.val * SDM)⟩ : Tile .ptr [M, CS]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hm, hcs, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `a_ptrs` eval from a scalar `A` register: cell `(e,j) = (A, base + e · SAS + offs_n j · SAK)`. -/
theorem aptrs_eval (s : BlockState) (A : RegionName) (CS N SAS SAK base : Nat) (gn : Fin N → Nat)
    (hbase : s.regs .ptr [] "A" = some (Tile.scalar (A.cast, base)))
    (hcs : s.regs .nat [CS] "offs_cs" = some (Tile.vec (fun e : Fin CS => e.val)))
    (hn : s.regs .nat [N] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "A")
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [CS] "offs_cs")) (Op.constNat SAS)))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offs_n")) (Op.constNat SAK))) s
      = some (⟨fun idx : TileIndex [CS, N] => (A.cast, base + idx.1.val * SAS + gn idx.2.1 * SAK)⟩ : Tile .ptr [CS, N]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hcs, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `acc` init eval: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `dout_ptrs += BLOCK_SIZE_CS · stride_dout_csize_m` eval (scalar add via `scalarR`). -/
theorem doutptr_adv_eval (s : BlockState) (M CS BCS SDM : Nat) (dp : Tile .ptr [M, CS])
    (hx : s.regs .ptr [M, CS] "dout_ptrs" = some dp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, CS] "dout_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BCS) (Op.constNat SDM))) s
      = some (Tile.ptrAdd Broadcast.scalarR dp (Tile.scalar (BCS * SDM))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hx, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `a_ptrs += BLOCK_SIZE_CS · stride_a_seqlen` eval. -/
theorem aptr_adv_eval (s : BlockState) (CS N BCS SAS : Nat) (ap : Tile .ptr [CS, N])
    (hy : s.regs .ptr [CS, N] "a_ptrs" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [CS, N] "a_ptrs")
      (Op.mul .nat Broadcast.nil (Op.constNat BCS) (Op.constNat SAS))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BCS * SAS))) := by
  rw [evalOp_ptrAdd]
  simp [evalOp_ref, hy, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- The dot of two all-`some` loaded tiles, lane `(i,j)`, equals
`some (Σ_e fx e · fy e)`. -/
theorem dot_da (M CS N : Nat) (x : Tile .real [M, CS]) (y : Tile .real [CS, N])
    (i : Fin M) (j : Fin N) (fx : Fin CS → ℝ) (fy : Fin CS → ℝ)
    (hx : ∀ e : Fin CS, x.data (i, e, PUnit.unit) = some (fx e))
    (hy : ∀ e : Fin CS, y.data (e, j, PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin CS => fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin CS) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (x.data (i, e, PUnit.unit)) (y.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin CS) (WithBot ℝ) _ Finset.univ (fun e => (some (fx e * fy e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (Finset.univ.sum fun e => ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

/-- **`acc = acc + tl.dot(dout, a)` statement eval.** -/
theorem accdot_op_eval (M CS N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, CS]) (yt : Tile .real [CS, N])
    (hz : st.regs .real [M, N] "acc" = some zt)
    (hx : st.regs .real [M, CS] "dout" = some xt)
    (hy : st.regs .real [CS, N] "a" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "acc")
        (Op.dot (batch := []) (Op.ref .real [M, CS] "dout") (Op.ref .real [CS, N] "a"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, CS] "dout")
        (Op.ref .real [CS, N] "a")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem accadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-- A `.ptr` masked load with `.maskOther`, when every mask lane is `true`,
reads `readMem` at each pointer (clean `undef`). -/
theorem load_ptr_maskOther_alltrue {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (otherOp : Op .real shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (mtile : Tile .bool shape) (otile : Tile .real shape)
    (hp : evalOp ptrOp s = some ptrs)
    (hm : evalOp maskOp s = some mtile)
    (ho : evalOp otherOp s = some otile)
    (hall : ∀ i, mtile.data i = Bool.true) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp otherOp)) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  unfold evalOp
  simp only [hp, hm, ho, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [hall i, if_true, BlockState.readMemValue_real]

/-- The constant `other = 0.0` broadcast tile. -/
theorem other_broadcast_eval (s : BlockState) (shape : TileShape) :
    evalOp ((Op.const 0.0).broadcast shape) s = some (⟨fun _ : TileIndex shape => some (0.0 : ℝ)⟩ : Tile .real shape) := by
  simp [evalOp, evalOp_const, Option.bind]

/-- The dout-load mask `(offs_m < chunk_size) & (offs_cs < CSL - cs·BCS)`: every
lane is `true` when each tile row is `< chunk_size` and `c·BCS + e < CSL`. -/
theorem doutmask_alltrue (s : BlockState) (BM BCS chunk_size CSL c : Nat) (gm : Fin BM → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hcs : s.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)))
    (hcc : s.regs .nat [] "cs" = some (Tile.scalar c))
    (hmlt : ∀ i : Fin BM, gm i < chunk_size)
    (hlt : ∀ e : Fin BCS, e.val < CSL - c * BCS) :
    ∃ mtile : Tile .bool [BM, BCS],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.constNat chunk_size))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BCS] "offs_cs"))
          (Op.sub .nat Broadcast.nil (Op.constNat CSL)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cs") (Op.constNat BCS))))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hm, hcs, hcc, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, ComparableDType.lt,
    NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨by simpa using hmlt _, by simpa using hlt _⟩

/-- The a-load mask `(offs_cs < CSL - cs·BCS) & (offs_n < K)`: every lane is
`true` under the same conditions plus `gn j < K`. -/
theorem amask_alltrue (s : BlockState) (BN BCS K CSL c : Nat) (gn : Fin BN → Nat)
    (hcs : s.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn))
    (hcc : s.regs .nat [] "cs" = some (Tile.scalar c))
    (hnlt : ∀ j : Fin BN, gn j < K)
    (hlt : ∀ e : Fin BCS, e.val < CSL - c * BCS) :
    ∃ mtile : Tile .bool [BCS, BN],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BCS] "offs_cs"))
          (Op.sub .nat Broadcast.nil (Op.constNat CSL)
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cs") (Op.constNat BCS))))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat K))) s = some mtile
      ∧ ∀ i, mtile.data i = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hcs, hn, hcc, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro i
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, ComparableDType.lt,
    NumericDType.sub, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨by simpa using hlt _, by simpa using hnlt _⟩

/-! ## Prologue: the scalar pids and index vectors -/

/-- **preLoop scalars** (statements 0–6): the pid derivation. Steps to a state
where `pid_b/pid_c/pid_h/pid_m/pid_n` hold their derived values. -/
theorem preLoop_scalars (s : BlockState) (K ngroups BM BN BCS : Nat) :
    ∃ s7,
      stepStmts
        [ Stmt.assign .nat [] "pid_b" (Op.programId 1),
          Stmt.assign .nat [] "pid_ch" (Op.programId 2),
          Stmt.assign .nat [] "pid_c"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid_ch") (Op.constNat ngroups)),
          Stmt.assign .nat [] "pid_h"
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "pid_ch")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat ngroups))),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")) ] s = some s7
      ∧ s7.pids = s.pids
      ∧ s7.regs .nat [] "pid_b" = some (Tile.scalar (s.pids 1))
      ∧ s7.regs .nat [] "pid_c" = some (Tile.scalar (pidC (s.pids 2) ngroups))
      ∧ s7.regs .nat [] "pid_h" = some (Tile.scalar (pidH (s.pids 2) ngroups))
      ∧ s7.regs .nat [] "pid_m" = some (Tile.scalar (pidM (s.pids 0) K BN))
      ∧ s7.regs .nat [] "pid_n" = some (Tile.scalar (pidN (s.pids 0) K BN))
      ∧ s7.undef = s.undef
      ∧ s7.mem = s.mem := by
  simp only [pidC, pidH, pidM, pidN, numPidN, cdiv]
  simp [stepStmts, stepStmt, evalOp_floorDiv, evalOp_mod, Option.bind, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.div,
    NumericDType.sub, IntegralDType.floorDiv, IntegralDType.mod]

/-! ## Loop body and invariant -/

/-- The 5-statement CS-loop body, transcribed. -/
def bbwdLoopBody (BM BN BCS chunk_size CSL K SDM SAS : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BCS] "dout"
      (Op.load .real (.ptr (Op.ref .ptr [BM, BCS] "dout_ptrs"))
        (.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
              (Op.constNat chunk_size))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BCS] "offs_cs"))
              (Op.sub .nat Broadcast.nil (Op.constNat CSL)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cs") (Op.constNat BCS)))))
          ((Op.const 0.0).broadcast [BM, BCS]))),
    Stmt.assign .real [BCS, BN] "a"
      (Op.load .real (.ptr (Op.ref .ptr [BCS, BN] "a_ptrs"))
        (.maskOther
          (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BCS] "offs_cs"))
              (Op.sub .nat Broadcast.nil (Op.constNat CSL)
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cs") (Op.constNat BCS))))
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
              (Op.constNat K)))
          ((Op.const 0.0).broadcast [BCS, BN]))),
    Stmt.assign .real [BM, BN] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BCS] "dout") (Op.ref .real [BCS, BN] "a"))),
    Stmt.assign .ptr [BM, BCS] "dout_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BCS] "dout_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BCS) (Op.constNat SDM))),
    Stmt.assign .ptr [BCS, BN] "a_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BCS, BN] "a_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BCS) (Op.constNat SAS))) ]

/-- **Loop invariant** (counter `i = c`, the CS-block index).

After `c` CS-blocks: program ids and `mem`/`undef` fixed; `pid_*` and
`offs_m/offs_n/offs_cs` seeded; `Dout`/`A` base pointers seeded; `acc` equals the
partial accumulator `accPartial … c`; and `dout_ptrs`/`a_ptrs` advanced by `c`
blocks. -/
noncomputable def bbwdInvariant
    (Dout A : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BLOCK_CS numCSBlocks : Nat)
    (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids ∧ c ≤ numCSBlocks ∧
  (s.regs .real [BM, BN] "acc" = some ⟨fun idx : TileIndex [BM, BN] =>
      some (accPartial s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BLOCK_CS idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [] "pid_b" = some (Tile.scalar PB)) ∧
  (s.regs .nat [] "pid_c" = some (Tile.scalar PC)) ∧
  (s.regs .nat [] "pid_h" = some (Tile.scalar PH)) ∧
  (s.regs .nat [] "pid_m" = some (Tile.scalar PM)) ∧
  (s.regs .nat [] "pid_n" = some (Tile.scalar PN)) ∧
  (s.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val))) ∧
  (s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val))) ∧
  (s.regs .nat [BLOCK_CS] "offs_cs" = some (Tile.vec (fun e : Fin BLOCK_CS => e.val))) ∧
  (s.regs .ptr [BM, BLOCK_CS] "dout_ptrs" = some ⟨fun idx : TileIndex [BM, BLOCK_CS] =>
      (Dout.cast, doutOff PB PC PH SDB SDC SDH + (PM * BM + idx.1.val) * SDN + idx.2.1.val * SDM + c * BLOCK_CS * SDM)⟩) ∧
  (s.regs .ptr [BLOCK_CS, BN] "a_ptrs" = some ⟨fun idx : TileIndex [BLOCK_CS, BN] =>
      (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + idx.1.val * SAS + (PN * BN + idx.2.1.val) * SAK + c * BLOCK_CS * SAS)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1000000 in
/-- **preLoop** (statements 0–14): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `bbwdInvariant … 0` — the base case
(`acc = 0`, pointers seeded with the batch offset). -/
theorem bbwd_preLoop (Dout A Db : RegionName) (s : BlockState)
    (chunk_size _CSL K ngroups SDB SDC SDH SDN SDM SAB SAS SAH SAK SOB SOS SOH SOK BM BN BCS numCSBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS).toAlgKernel.body.take 15) s = some s'
      ∧ bbwdInvariant Dout A s (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN)
          chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks 0 s' := by
  obtain ⟨s7, h7, hpids, hpb, hpc, hph, hpm, hpn, huf, hmem⟩ :=
    preLoop_scalars s K ngroups BM BN BCS
  set PB := s.pids 1 with hPB
  set PC := pidC (s.pids 2) ngroups with hPC
  set PH := pidH (s.pids 2) ngroups with hPH
  set PM := pidM (s.pids 0) K BN with hPM
  set PN := pidN (s.pids 0) K BN with hPN
  rw [show ((bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS).toAlgKernel.body.take 15)
      = [ Stmt.assign .nat [] "pid_b" (Op.programId 1),
          Stmt.assign .nat [] "pid_ch" (Op.programId 2),
          Stmt.assign .nat [] "pid_c"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "pid_ch") (Op.constNat ngroups)),
          Stmt.assign .nat [] "pid_h"
            (Op.sub .nat Broadcast.nil (Op.ref .nat [] "pid_ch")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat ngroups))),
          Stmt.assign .nat [] "num_pid_n"
            (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BN)) (Op.constNat 1)) (Op.constNat BN)),
          Stmt.assign .nat [] "pid_m"
            (Op.floorDiv .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")),
          Stmt.assign .nat [] "pid_n"
            (Op.mod .nat Broadcast.nil (Op.programId 0) (Op.ref .nat [] "num_pid_n")) ]
      ++ [ Stmt.assign .ptr [] "Dout"
            (Op.ptrAdd Broadcast.nil (Op.ptrBase Dout)
              (Op.add .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SDB))
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat SDC)))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SDH)))),
          Stmt.assign .ptr [] "A"
            (Op.ptrAdd Broadcast.nil (Op.ptrBase A)
              (Op.add .nat Broadcast.nil
                (Op.add .nat Broadcast.nil
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SAB))
                  (Op.mul .nat Broadcast.nil
                    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SAS)))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SAH)))),
          Stmt.assign .nat [BM] "offs_m"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
          Stmt.assign .nat [BN] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
          Stmt.assign .nat [BCS] "offs_cs" (Op.arange BCS),
          Stmt.assign .ptr [BM, BCS] "dout_ptrs"
            (Op.ptrAdd Broadcast.nil.consL.consR
              (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Dout")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat SDN)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BCS] "offs_cs")) (Op.constNat SDM))),
          Stmt.assign .ptr [BCS, BN] "a_ptrs"
            (Op.ptrAdd Broadcast.nil.consL.consR
              (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "A")
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BCS] "offs_cs")) (Op.constNat SAS)))
              (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SAK))),
          Stmt.assign .real [BM, BN] "acc" (Op.full [BM, BN] (Op.const 0)) ] from rfl]
  -- step the pid prologue
  rw [stepStmts.append_some h7]
  -- Dout base
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (doutptr_base_eval s7 Dout PB PC PH SDB SDC SDH hpb hpc hph))]
  set s8 := s7.setReg "Dout" .ptr [] (Tile.scalar (Dout.cast, doutOff PB PC PH SDB SDC SDH)) with hs8
  have hpb8 : s8.regs .nat [] "pid_b" = some (Tile.scalar PB) := by simp [hs8, hpb]
  have hpc8 : s8.regs .nat [] "pid_c" = some (Tile.scalar PC) := by simp [hs8, hpc]
  have hph8 : s8.regs .nat [] "pid_h" = some (Tile.scalar PH) := by simp [hs8, hph]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (aptr_base_eval s8 A PB PC PH chunk_size SAB SAS SAH hpb8 hpc8 hph8))]
  set s9 := s8.setReg "A" .ptr [] (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) with hs9
  -- offs_m
  have hpm9 : s9.regs .nat [] "pid_m" = some (Tile.scalar PM) := by simp [hs9, hs8, hpm]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s9
      = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm9, evalOp_constNat, evalOp_arange, Option.bind, Option.bind_some]
      refine congrArg some ?_; ext i
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  set s10 := s9.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => PM * BM + i.val)) with hs10
  -- offs_n
  have hpn10 : s10.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs10, hs9, hs8, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s10
      = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn10, evalOp_constNat, evalOp_arange, Option.bind, Option.bind_some]
      refine congrArg some ?_; ext j
      simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  set s11 := s10.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => PN * BN + j.val)) with hs11
  -- offs_cs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BCS) s11 = some (Tile.vec (fun e : Fin BCS => e.val)) by simp [evalOp_arange]))]
  set s12 := s11.setReg "offs_cs" .nat [BCS] (Tile.vec (fun e : Fin BCS => e.val)) with hs12
  -- dout_ptrs
  have hdbase12 : s12.regs .ptr [] "Dout" = some (Tile.scalar (Dout.cast, doutOff PB PC PH SDB SDC SDH)) := by
    simp [hs12, hs11, hs10, hs9, hs8]
  have hm12 : s12.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hs12, hs11, hs10]
  have hcs12 : s12.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)) := by simp [hs12]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (doutptrs_eval s12 Dout BM BCS SDN SDM (doutOff PB PC PH SDB SDC SDH) (fun i => PM * BM + i.val) hdbase12 hm12 hcs12))]
  set s13 := s12.setReg "dout_ptrs" .ptr [BM, BCS]
    (⟨fun idx : TileIndex [BM, BCS] => (Dout.cast, doutOff PB PC PH SDB SDC SDH + (PM * BM + idx.1.val) * SDN + idx.2.1.val * SDM)⟩) with hs13
  -- a_ptrs
  have habase13 : s13.regs .ptr [] "A" = some (Tile.scalar (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH)) := by
    simp [hs13, hs12, hs11, hs10, hs9]
  have hcs13 : s13.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)) := by simp [hs13, hs12]
  have hn13 : s13.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hs13, hs12, hs11]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptrs_eval s13 A BCS BN SAS SAK (batchOff PB PC PH chunk_size SAB SAS SAH) (fun j => PN * BN + j.val) habase13 hcs13 hn13))]
  set s14 := s13.setReg "a_ptrs" .ptr [BCS, BN]
    (⟨fun idx : TileIndex [BCS, BN] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + idx.1.val * SAS + (PN * BN + idx.2.1.val) * SAK)⟩) with hs14
  -- acc init
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval s14 BM BN)), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp [hs14, hs13, hs12, hs11, hs10, hs9, hs8, hpids], by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- acc = accPartial … 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    simp only [accPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpb
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpc
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hph
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpm
  · simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hpn
  · simp only [hs14, hs13, hs12, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hm12
  · simp only [hs14, hs13, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hn13
  · simp only [hs14, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hcs13
  · -- dout_ptrs (c = 0)
    simp only [hs14, hs13, BlockState.setReg_same, BlockState.setReg_ne_name,
      BlockState.setReg_ne_shape, BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq, Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · -- a_ptrs (c = 0)
    simp only [hs14, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_shape,
      BlockState.setReg_ne_dtype, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq,
      Option.some.injEq]
    ext idx <;> simp [Nat.zero_mul]
  · intro rg o; simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_undef]; exact huf ▸ hundef rg o
  · show _ = s.mem
    simp only [hs14, hs13, hs12, hs11, hs10, hs9, hs8, BlockState.setReg_mem]; exact hmem

set_option maxHeartbeats 4000000 in
/-- **Step lemma**: one CS-loop body iteration advances the invariant by one
block (`acc += tl.dot(dout, a)` adds the `c`-th block's dot to the partial
accumulator; the `dout`/`a` pointers advance one step). Under
`CSL = BCS · numCSBlocks` and `PM·BM+i < chunk_size`, `PN·BN+j < K` the per-block
load masks are all satisfied. -/
theorem bbwd_step (Dout A : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks : Nat) (hBCS : 0 < BCS)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < K)
    (c : Nat) (s : BlockState) (hclt : c < numCSBlocks)
    (hinv : bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks c s) :
    ∃ s', stepStmts (bbwdLoopBody BM BN BCS chunk_size (BCS * numCSBlocks) K SDM SAS)
        (s.setReg "cs" .nat [] (Tile.scalar c)) = some s'
      ∧ bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks (c + 1) s' := by
  set CSL := BCS * numCSBlocks with hCSLdef
  simp only [bbwdInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpb, hpc, hph, hpm, hpn, hmm, hnn, hcs, hdp, hap, hundef, hmem⟩ := hinv
  have hlt : ∀ e : Fin BCS, e.val < CSL - c * BCS := by
    intro e
    have hcCSL : c * BCS + BCS ≤ CSL := by
      rw [hCSLdef]; calc c * BCS + BCS = (c + 1) * BCS := by ring
        _ ≤ numCSBlocks * BCS := Nat.mul_le_mul_right _ hclt
        _ = BCS * numCSBlocks := Nat.mul_comm _ _
    omega
  set dpT : Tile .ptr [BM, BCS] :=
    ⟨fun idx : TileIndex [BM, BCS] => (Dout.cast, doutOff PB PC PH SDB SDC SDH + (PM * BM + idx.1.val) * SDN + idx.2.1.val * SDM + c * BCS * SDM)⟩ with hdpT
  set apT : Tile .ptr [BCS, BN] :=
    ⟨fun idx : TileIndex [BCS, BN] => (A.cast, batchOff PB PC PH chunk_size SAB SAS SAH + idx.1.val * SAS + (PN * BN + idx.2.1.val) * SAK + c * BCS * SAS)⟩ with hapT
  set zT : Tile .real [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => some (accPartial s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS idx.1 idx.2.1 c)⟩ with hzT
  set sk := s.setReg "cs" .nat [] (Tile.scalar c) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hdpk : sk.regs .ptr [BM, BCS] "dout_ptrs" = some dpT := by simp [hsk, hdp, hdpT]
  have hapk : sk.regs .ptr [BCS, BN] "a_ptrs" = some apT := by simp [hsk, hap, hapT]
  have hzk : sk.regs .real [BM, BN] "acc" = some zT := by simp [hsk, hz, hzT]
  have hmk : sk.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => PM * BM + i.val)) := by simp [hsk, hmm]
  have hnk : sk.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk, hnn]
  have hcsk : sk.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)) := by simp [hsk, hcs]
  have hcck : sk.regs .nat [] "cs" = some (Tile.scalar c) := by simp [hsk]
  set dsub : Tile .real [BM, BCS] :=
    ⟨fun idx => some (sk.readMem (dpT.data idx).1 (dpT.data idx).2)⟩ with hdsub
  set sk1 := sk.setReg "dout" .real [BM, BCS] dsub with hsk1
  set asub : Tile .real [BCS, BN] :=
    ⟨fun idx => some (sk1.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set sk2 := sk1.setReg "a" .real [BCS, BN] asub with hsk2
  have hcsk1 : sk1.regs .nat [BCS] "offs_cs" = some (Tile.vec (fun e : Fin BCS => e.val)) := by simp [hsk1, hcsk]
  have hnk1 : sk1.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => PN * BN + j.val)) := by simp [hsk1, hnk]
  have hcck1 : sk1.regs .nat [] "cs" = some (Tile.scalar c) := by simp [hsk1, hcck]
  obtain ⟨dmt, hdm_eval, hdm_true⟩ := doutmask_alltrue sk BM BCS chunk_size CSL c _ hmk hcsk hcck hmlt hlt
  obtain ⟨amt, ham_eval, ham_true⟩ := amask_alltrue sk1 BN BCS K CSL c _ hcsk1 hnk1 hcck1 hnlt hlt
  unfold bbwdLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BM, BCS] "dout_ptrs") _ _ sk dpT dmt _
          (by rw [evalOp_ref]; simp [hdpk]) hdm_eval (other_broadcast_eval sk [BM, BCS]) hdm_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_alltrue (Op.ref .ptr [BCS, BN] "a_ptrs") _ _ sk1 apT amt _
          (by rw [evalOp_ref]; simp [hsk1, hapk]) ham_eval (other_broadcast_eval sk1 [BCS, BN]) ham_true))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BM BCS BN sk2 zT dsub asub
          (by simp [hsk2, hsk1, hzk])
          (by simp [hsk2, hsk1])
          (by simp [hsk2])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (doutptr_adv_eval _ BM BCS BCS SDM dpT (by simp [hsk2, hsk1, hdpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BCS BN BCS SAS apT (by simp [hsk2, hsk1, hapk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [bbwdInvariant]
  refine ⟨by simp [hsk2, hsk1, hsk, hpids], by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- acc = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx
    have hds : ∀ e : Fin BCS, dsub.data (idx.1, e, PUnit.unit)
        = some (doutElem s0 Dout PB PC PH PM BM SDB SDC SDH SDN SDM idx.1 (c * BCS + e.val)) := by
      intro e
      simp only [hdsub, hdpT, hrmem, doutElem, rowIndex, doutOff]
      congr 2
      ring
    have hrmem1 : ∀ (R : RegionName) (o : Nat), sk1.readMem R o = s0.readMem R o := by
      intro R o; rw [hsk1, BlockState.setReg_readMem]; exact hrmem R o
    have has : ∀ e : Fin BCS, asub.data (e, idx.2.1, PUnit.unit)
        = some (aElem s0 A PB PC PH PN BN chunk_size SAB SAS SAH SAK idx.2.1 (c * BCS + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem1, aElem, colIndex, batchOff]
      congr 2
      ring
    rw [accadd_eval BM BN zT (Tile.dot [] dsub asub) idx.1 idx.2.1
        (accPartial s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BCS =>
          doutElem s0 Dout PB PC PH PM BM SDB SDC SDH SDN SDM idx.1 (c * BCS + e.val)
            * aElem s0 A PB PC PH PN BN chunk_size SAB SAS SAH SAK idx.2.1 (c * BCS + e.val))
        (by rw [hzT])
        (dot_da BM BCS BN dsub asub idx.1 idx.2.1 _ _ hds has)]
    show some _ = some (accPartial s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk2, hsk1, hsk, hpb]
  · simp [hsk2, hsk1, hsk, hpc]
  · simp [hsk2, hsk1, hsk, hph]
  · simp [hsk2, hsk1, hsk, hpm]
  · simp [hsk2, hsk1, hsk, hpn]
  · simp [hsk2, hsk1, hsk, hmm]
  · simp [hsk2, hsk1, hsk, hnn]
  · simp [hsk2, hsk1, hsk, hcs]
  · -- dout_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hdpT, NumericDType.add]
    ring
  · -- a_ptrs advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_
    ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.scalar, hapT, NumericDType.add]
    ring
  · intro rg o; simp [hsk2, hsk1, hsk, hundef]
  · show _ = s0.mem; rw [← hmem]; rfl

/-- The dynamic CS-loop bound resolves: `cdiv CSL BCS = numCSBlocks` and the loop
runs `numCSBlocks` iterations, advancing `bbwdInvariant` from `0` to
`numCSBlocks`. -/
theorem bbwd_loop (Dout A : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks K : Nat) (hBCS : 0 < BCS)
    (hmlt : ∀ i : Fin BM, PM * BM + i.val < chunk_size)
    (hnlt : ∀ j : Fin BN, PN * BN + j.val < K)
    (s : BlockState)
    (hP0 : bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks 0 s) :
    ∃ sLoop, stepStmt (Stmt.forRangeDyn "cs" (Op.constNat 0)
        (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat (BCS * numCSBlocks)) (Op.constNat BCS)) (Op.constNat 1)) (Op.constNat BCS))
        (Op.constNat 1) (bbwdLoopBody BM BN BCS chunk_size (BCS * numCSBlocks) K SDM SAS)) s = some sLoop
      ∧ bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks numCSBlocks sLoop := by
  have hcdiv : (BCS * numCSBlocks + BCS - 1) / BCS = numCSBlocks := by
    have he : BCS * numCSBlocks + BCS - 1 = (BCS - 1) + BCS * numCSBlocks := by omega
    rw [he, Nat.add_mul_div_left _ _ hBCS, Nat.div_eq_of_lt (by omega), Nat.zero_add]
  have hresolve : stepStmt (Stmt.forRangeDyn "cs" (Op.constNat 0)
      (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat (BCS * numCSBlocks)) (Op.constNat BCS)) (Op.constNat 1)) (Op.constNat BCS))
      (Op.constNat 1) (bbwdLoopBody BM BN BCS chunk_size (BCS * numCSBlocks) K SDM SAS)) s
      = stepForRangeAux "cs" 0 numCSBlocks 1 (bbwdLoopBody BM BN BCS chunk_size (BCS * numCSBlocks) K SDM SAS) s := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Tile.scalar, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, NumericDType.sub,
      NumericDType.add, Tile.data, Option.bind_some, bind]
    rw [hcdiv]
  rw [hresolve]
  obtain ⟨final, sLoop, haux, hfinal, hPfinal⟩ :=
    forRangeAux_inv (idx := "cs") (stop := numCSBlocks) (step := 1)
      (P := bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks)
      (by norm_num)
      (fun i st hlt hinv => bbwd_step Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks hBCS hmlt hnlt i st hlt hinv)
      0 s hP0
  have hfinalEq : final = numCSBlocks := by
    simp only [bbwdInvariant] at hPfinal
    omega
  subst hfinalEq
  exact ⟨sLoop, haux, hPfinal⟩

/-! ## Post-loop: the masked output store -/

/-- The 6 post-loop statements: refresh `offs_m`/`offs_n`, the `db` real copy,
the `Db` base-pointer advance, the output pointers, and the masked store. -/
def bbwdPostBody (Db : RegionName) (chunk_size CSL K SOB SOS SOH SOK BM BN : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)),
    Stmt.assign .real [BM, BN] "db" (Op.ref .real [BM, BN] "acc"),
    Stmt.assign .ptr [] "Db"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Db)
        (Op.add .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SOS)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))),
    Stmt.assign .ptr [BM, BN] "db_ptrs"
      (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Db")
          (Op.mul .nat Broadcast.scalarL (Op.constNat SOS) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SOK))),
    Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "db_ptrs")) (Op.ref .real [BM, BN] "db")
      (.mask (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
          (Op.constNat CSL))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))
          (Op.constNat K)))) ]

/-- The kernel body decomposes as prefix (15) ++ [CS-loop, post-statements]. By `rfl`. -/
theorem bbwd_body_split (A Dout Db : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS numCSBlocks K : Nat) :
    (bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS).toAlgKernel.body
      = (bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
          SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS).toAlgKernel.body.take 15
        ++ (Stmt.forRangeDyn "cs" (Op.constNat 0)
              (Op.div .nat Broadcast.nil (Op.sub .nat Broadcast.nil
                (Op.add .nat Broadcast.nil (Op.constNat (BCS * numCSBlocks)) (Op.constNat BCS)) (Op.constNat 1)) (Op.constNat BCS))
              (Op.constNat 1) (bbwdLoopBody BM BN BCS chunk_size (BCS * numCSBlocks) K SDM SAS)
            :: bbwdPostBody Db chunk_size (BCS * numCSBlocks) K SOB SOS SOH SOK BM BN) := by
  rfl

/-- The kernel's output offset for tile lane `idx`. -/
def dbOffset (PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  PB * SOB + PC * chunk_size * SOS + PH * SOH + SOS * rowIndex PM BM idx.1 + colIndex PN BN idx.2.1 * SOK

/-- The genuine output cell `Σ_cs Dout·A` as a real `MemCell`. -/
noncomputable def dbCell (s0 : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks : Nat)
    (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue
    (some (bbwdSpec s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx.1 idx.2.1))))

private theorem foldl_writeMemTyped_real_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.real)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil => intro s _h; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = Bool.true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change
          (if region = region ∧ o = offsetFn hd then
            MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by intro hsame; exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

private theorem scatter_memcell_real_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.real)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .real region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_real_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_real_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-- `offs_m` eval refresh. -/
theorem offs_m_eval (s : BlockState) (PM BM : Nat)
    (hpm : s.regs .nat [] "pid_m" = some (Tile.scalar PM)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM)) (Op.arange BM)) s
      = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpm, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_; ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, rowIndex]

/-- `offs_n` eval refresh. -/
theorem offs_n_eval (s : BlockState) (PN BN : Nat)
    (hpn : s.regs .nat [] "pid_n" = some (Tile.scalar PN)) :
    evalOp (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)) (Op.arange BN)) s
      = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hpn, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_; ext j
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, colIndex]

/-- `Db` base eval: `Db + pid_b·SOB + pid_c·chunk_size·SOS + pid_h·SOH`. -/
theorem dbptr_base_eval (s : BlockState) (Db : RegionName) (PB PC PH chunk_size SOB SOS SOH : Nat)
    (hpb : s.regs .nat [] "pid_b" = some (Tile.scalar PB))
    (hpc : s.regs .nat [] "pid_c" = some (Tile.scalar PC))
    (hph : s.regs .nat [] "pid_h" = some (Tile.scalar PH)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Db)
      (Op.add .nat Broadcast.nil
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_b") (Op.constNat SOB))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_c") (Op.constNat chunk_size)) (Op.constNat SOS)))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_h") (Op.constNat SOH)))) s
      = some (Tile.scalar (Db.cast, PB * SOB + PC * chunk_size * SOS + PH * SOH)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hpb, hpc, hph,
    Option.bind, Option.bind_some, Option.bind_eq_bind]
  refine congrArg some ?_
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]
  ring_nf

/-- `db_ptrs` eval: cell `(i,j) = (Db, base + SOS·offs_m i + offs_n j·SOK)`. -/
theorem dbptrs_eval (s : BlockState) (Db : RegionName) (BM BN SOS SOK base : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hbase : s.regs .ptr [] "Db" = some (Tile.scalar (Db.cast, base)))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn)) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Db")
        (Op.mul .nat Broadcast.scalarL (Op.constNat SOS) (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))))
      (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat SOK))) s
      = some (⟨fun idx : TileIndex [BM, BN] => (Db.cast, base + SOS * gm idx.1 + gn idx.2.1 * SOK)⟩ : Tile .ptr [BM, BN]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd]
  simp only [evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hbase, hm, hn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- The output store mask `(offs_m < CSL) & (offs_n < K)`: all-`true` when in-bounds. -/
theorem dbmask_alltrue (s : BlockState) (CSL K BM BN : Nat) (gm : Fin BM → Nat) (gn : Fin BN → Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec gn))
    (hmlt : ∀ i, gm i < CSL) (hnlt : ∀ j, gn j < K) :
    ∃ mtile : Tile .bool [BM, BN],
      evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat CSL))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat K))) s
        = some mtile
      ∧ ∀ idx : TileIndex [BM, BN], mtile.data idx = Bool.true := by
  refine ⟨_, by
    simp only [evalOp, evalOp.eq_def, hm, hn, Option.bind, Option.bind_some, Option.bind_eq_bind,
      Tile.expandDim, pure, Option.some.injEq]; rfl, ?_⟩
  intro idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt, TileShape.dropInsertedIndex, Bool.and_eq_true,
    decide_eq_true_eq]
  exact ⟨by simpa using hmlt _, by simpa using hnlt _⟩

/-- The masked real store reduces to the masked scatter foldl. -/
theorem store_eval (BM BN : Nat) (st : BlockState)
    (oT : Tile .real [BM, BN]) (opT : Tile .ptr [BM, BN]) (mT : Tile .bool [BM, BN])
    (CSL K : Nat)
    (ho : st.regs .real [BM, BN] "db" = some oT)
    (hop : st.regs .ptr [BM, BN] "db_ptrs" = some opT)
    (hmeval : evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat CSL))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat K))) st = some mT) :
    stepStmt (Stmt.store .real [BM, BN] (.ptr (Op.ref .ptr [BM, BN] "db_ptrs")) (Op.ref .real [BM, BN] "db")
        (.mask (Op.boolAnd Broadcast.nil.consL.consR
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat CSL))
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.constNat K))))) st
      = some ((TileShape.allIndices [BM, BN]).foldl
          (fun acc idx =>
            if mT.data idx then acc.writeMemTyped .real (opT.data idx).1 (opT.data idx).2 (oT.data idx) else acc) st) := by
  simp only [stepStmt, evalOp_ref, ho, hop, hmeval, bind, Option.bind_some]
  refine congrArg some (List.foldl_ext _ _ st (fun acc idx _ => ?_))
  by_cases hb : mT.data idx
  · simp only [hb, if_true]
  · simp only [hb, if_false, Bool.false_eq_true]

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numCSBlocks` blocks, the masked store
writes the genuine value `Σ_cs Dout·A` at every in-bounds output lane. -/
theorem bbwd_postLoop (Dout A Db : RegionName) (s0 : BlockState)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK SOB SOS SOH SOK BCS numCSBlocks K : Nat)
    (hInj : Function.Injective (dbOffset PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex PM BM i < BCS * numCSBlocks)
    (hnlt : ∀ j : Fin BN, colIndex PN BN j < K)
    (st : BlockState)
    (hinv : bbwdInvariant Dout A s0 PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks numCSBlocks st) :
    ∃ sfin, stepStmts (bbwdPostBody Db chunk_size (BCS * numCSBlocks) K SOB SOS SOH SOK BM BN) st = some sfin
      ∧ ∀ idx : TileIndex [BM, BN],
          sfin.mem Db (dbOffset PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN idx)
            = dbCell s0 Dout A PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx := by
  simp only [bbwdInvariant] at hinv
  obtain ⟨hpids, hcle, hz, hpb, hpc, hph, hpm, hpn, hmm, hnn, hcs, hdp, hap, hundef, hmem⟩ := hinv
  set g : TileIndex [BM, BN] → ℝ :=
    fun idx => bbwdSpec s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx.1 idx.2.1 with hg
  have hzspec : st.regs .real [BM, BN] "acc" = some ⟨fun idx => some (g idx)⟩ := by
    rw [hz]; refine congrArg some ?_; ext idx; simp only [hg, bbwdSpec, accPartial, Nat.mul_comm numCSBlocks BCS]
  unfold bbwdPostBody
  -- step 1: offs_m
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_m_eval st PM BM hpm))]
  set s1 := st.setReg "offs_m" .nat [BM] (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) with hs1
  -- step 2: offs_n
  have hpn1 : s1.regs .nat [] "pid_n" = some (Tile.scalar PN) := by simp [hs1, hpn]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (offs_n_eval s1 PN BN hpn1))]
  set s2 := s1.setReg "offs_n" .nat [BN] (Tile.vec (fun j : Fin BN => colIndex PN BN j)) with hs2
  -- step 3: db = acc
  have hacc2 : s2.regs .real [BM, BN] "acc" = some ⟨fun idx => some (g idx)⟩ := by simp [hs2, hs1, hzspec]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BM, BN] "acc") s2 = some ⟨fun idx => some (g idx)⟩ by rw [evalOp_ref, hacc2]))]
  set s3 := s2.setReg "db" .real [BM, BN] (⟨fun idx => some (g idx)⟩ : Tile .real [BM, BN]) with hs3
  -- step 4: Db base
  have hpb3 : s3.regs .nat [] "pid_b" = some (Tile.scalar PB) := by simp [hs3, hs2, hs1, hpb]
  have hpc3 : s3.regs .nat [] "pid_c" = some (Tile.scalar PC) := by simp [hs3, hs2, hs1, hpc]
  have hph3 : s3.regs .nat [] "pid_h" = some (Tile.scalar PH) := by simp [hs3, hs2, hs1, hph]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (dbptr_base_eval s3 Db PB PC PH chunk_size SOB SOS SOH hpb3 hpc3 hph3))]
  set s4 := s3.setReg "Db" .ptr [] (Tile.scalar (Db.cast, PB * SOB + PC * chunk_size * SOS + PH * SOH)) with hs4
  -- step 5: db_ptrs
  have hobase4 : s4.regs .ptr [] "Db" = some (Tile.scalar (Db.cast, PB * SOB + PC * chunk_size * SOS + PH * SOH)) := by simp [hs4]
  have hm4 : s4.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs4, hs3, hs2, hs1]
  have hn4 : s4.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs4, hs3, hs2]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (dbptrs_eval s4 Db BM BN SOS SOK (PB * SOB + PC * chunk_size * SOS + PH * SOH) (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hobase4 hm4 hn4))]
  set opT : Tile .ptr [BM, BN] :=
    ⟨fun idx : TileIndex [BM, BN] => (Db.cast, (PB * SOB + PC * chunk_size * SOS + PH * SOH) + SOS * rowIndex PM BM idx.1 + colIndex PN BN idx.2.1 * SOK)⟩ with hopT
  set s5 := s4.setReg "db_ptrs" .ptr [BM, BN] opT with hs5
  -- step 6: masked store
  have hm5 : s5.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => rowIndex PM BM i)) := by simp [hs5, hm4]
  have hn5 : s5.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => colIndex PN BN j)) := by simp [hs5, hn4]
  obtain ⟨mT, hmask_eval, hmask_true⟩ :=
    dbmask_alltrue s5 (BCS * numCSBlocks) K BM BN (fun i => rowIndex PM BM i) (fun j => colIndex PN BN j) hm5 hn5 hmlt hnlt
  set oT : Tile .real [BM, BN] := (⟨fun idx => some (g idx)⟩ : Tile .real [BM, BN]) with hoT
  have hout5 : s5.regs .real [BM, BN] "db" = some oT := by simp [hs5, hs4, hs3, hs2, hoT]
  have hop5 : s5.regs .ptr [BM, BN] "db_ptrs" = some opT := by simp [hs5]
  rw [stepStmts.cons_some (store_eval BM BN s5 oT opT mT (BCS * numCSBlocks) K hout5 hop5 hmask_eval), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hstep_eq :
      (fun (acc : BlockState) k =>
        if mT.data k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc)
        =
      (fun (acc : BlockState) k =>
        if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc) := by
    funext acc k; rw [hmask_true k]; simp
  rw [hstep_eq]
  have hDbRegion : ∀ k : TileIndex [BM, BN], (opT.data k).1 = Db := by intro k; simp [hopT]
  rw [show
      ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real (opT.data k).1 (opT.data k).2 (oT.data k) else acc) s5)
      = ((TileShape.allIndices [BM, BN]).foldl
        (fun acc k => if (fun _ : TileIndex [BM, BN] => True) k then acc.writeMemTyped .real Db ((opT.data k).2) (oT.data k) else acc) s5)
      from List.foldl_ext _ _ s5 (fun acc k _ => by rw [hDbRegion k])]
  have hooff : ∀ k : TileIndex [BM, BN], (opT.data k).2 = dbOffset PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN k := by
    intro k; simp only [hopT, dbOffset]
  have hoffInj : Function.Injective (fun idx : TileIndex [BM, BN] => (opT.data idx).2) := by
    intro a b hab; apply hInj
    rw [← hooff a, ← hooff b]; exact hab
  rw [← hooff idx]
  rw [scatter_memcell_real_prop_masked_nd (region := Db) (s := s5)
    (offsetFn := fun idx : TileIndex [BM, BN] => (opT.data idx).2)
    (valueFn := fun idx => oT.data idx)
    (P := fun _ => True) hoffInj idx]
  simp only [if_pos trivial, dbCell, hoT, hg]

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: `preLoop` + CS-loop (`bbwd_loop`) + `postLoop`
compose into the full `exec`. Every in-bounds output lane equals `Σ_cs Dout·A`. -/
theorem bbwd_exec_closed_form (A Dout Db : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS numCSBlocks K : Nat) (hBCS : 0 < BCS)
    (hInj : Function.Injective (dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < BCS * numCSBlocks)
    (hmlt' : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) K BN) BN j < K)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BM, BN]) :
    (match exec (bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS) s with
      | some s' => s'.mem Db (dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN idx)
      | none => (0 : MemCell)) =
      dbCell s Dout A (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
        (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN)
        chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx := by
  set PB := s.pids 1 with hPB
  set PC := pidC (s.pids 2) ngroups with hPC
  set PH := pidH (s.pids 2) ngroups with hPH
  set PM := pidM (s.pids 0) K BN with hPM
  set PN := pidN (s.pids 0) K BN with hPN
  obtain ⟨s0, hpre_eq, hP0⟩ := bbwd_preLoop Dout A Db s chunk_size (BCS * numCSBlocks) K ngroups
    SDB SDC SDH SDN SDM SAB SAS SAH SAK SOB SOS SOH SOK BM BN BCS numCSBlocks hundef
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    bbwd_loop Dout A s PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks K hBCS hmlt' hnlt s0 hP0
  obtain ⟨sfin, hTail, hpost⟩ :=
    bbwd_postLoop Dout A Db s PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK SOB SOS SOH SOK BCS numCSBlocks K hInj hmlt hnlt sLoop hPLoop
  have hexec : exec (bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
      SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS) s = some sfin := by
    rw [exec, bbwd_body_split A Dout Db chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS numCSBlocks K,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  rw [hexec]
  exact hpost idx

/-- **Closed-form batched-matmul-backward correctness for `bmm_chunk_bwd`
(general statement).**

For arbitrary `chunk_size`, `K`, `ngroups`, tile dims `BM`/`BN`, strides,
CS-block size `BCS`, and CS-block count `numCSBlocks` (so the contracted bound
`CSL = BCS · numCSBlocks`), every in-bounds output cell of the computed `BM × BN`
gradient tile equals the genuine batched matrix-product backward
`Σ_{cs < BCS·numCSBlocks} Dout[i,cs] · A[cs,j]` over ℝ — NOT the kernel's own
executed value — under the kernel's batch/chunk/head/row/col addressing.

`PB/PC/PH/PM/PN` are the kernel's own derived program coordinates. Preconditions:
`0 < BCS`; all tile rows/cols in-bounds (`PM·BM+i < CSL` and `< chunk_size`,
`PN·BN+j < K`), making the load and store masks all-true; output-address
injectivity; clean initial `undef`. -/
theorem bmm_chunk_bwd_closed_form_correct
    (A Dout Db : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS numCSBlocks K : Nat) (hBCS : 0 < BCS)
    (hInj : Function.Injective (dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < BCS * numCSBlocks)
    (hmlt' : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) K BN) BN j < K)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Db, dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        dbCell s Dout A (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN)
          chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bbwd_matmul_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx
  have hmain := bbwd_exec_closed_form A Dout Db s0 chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN
    SOB SOS SOH SOK BM BN BCS numCSBlocks K hBCS hInj hmlt hmlt' hnlt hundef idx
  rw [hExec] at hmain
  exact hmain

/-! ## Public Python test-shape summary (genuine spec)

Python test case 1 (ungrouped, no residual) is the pure batched-matmul-backward
configuration. Layout: `a = (2,128,64)` fp16, `dout = (2,4,32,32)` fp16,
`out = empty_like(a) = (2,128,64)`. So `chunk_size = 32`, `K = 64`, `ngroups = 1`,
`seqlen = 128`, `nchunks = 4`. The contracted dimension is the chunk size: with
`BLOCK_SIZE_CS = 32`, `numCSBlocks = 1` so `CSL = 32`.

Strides (row-major):
* `a`: `(8192, 64, _, 1)` → `stride_a_batch=8192, stride_a_seqlen=64, stride_ak=1`.
* `dout = (2,4,32,32)`: strides `(4096, 1024, 32, 1)` →
  `stride_dout_batch=4096, stride_dout_chunk=1024, stride_dout_csize_m=32,
   stride_dout_csize_n=1`.
* `out = (2,128,64)`: `(8192, 64, _, 1)` →
  `stride_db_batch=8192, stride_db_seqlen=64, stride_db_k=1`. -/

/-- **Python case 1 summary** (ungrouped, no residual: `chunk_size=32`, `K=64`,
`ngroups=1`; `BLOCK_SIZE_CS=32`, `numCSBlocks=1`, `CSL=32`): the genuine batched
matmul-backward surface lowers to the algorithm layer and realizes the
per-program matrix-product backward `Σ_{cs<32} Dout[i,cs]·A[cs,j]` on every
in-bounds output lane. -/
theorem bmm_chunk_bwd_python_case1_summary
    (A Dout Db : RegionName) (s : BlockState) (BM BN : Nat)
    (hInj : Function.Injective (dbOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
      32 8192 64 0 1 (pidM (s.pids 0) 64 BN) (pidN (s.pids 0) 64 BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) 64 BN) BM i < 32)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) 64 BN) BN j < 64)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bbwd_matmul_surface A Dout Db 32 32 64 1 8192 64 0 1
        4096 1024 0 32 1 8192 64 0 1 BM BN 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bbwd_matmul_surface A Dout Db 32 32 64 1 8192 64 0 1
        4096 1024 0 32 1 8192 64 0 1 BM BN 32)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Db, dbOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          32 8192 64 0 1 (pidM (s.pids 0) 64 BN) (pidN (s.pids 0) 64 BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        dbCell s Dout A (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          (pidM (s.pids 0) 64 BN) (pidN (s.pids 0) 64 BN)
          32 BM BN 4096 1024 0 1 32 8192 64 0 1 32 1 idx) := by
  refine ⟨bbwd_matmul_surface_toAlgorithm_supported A Dout Db 32 32 64 1 8192 64 0 1 4096 1024 0 32 1 8192 64 0 1 BM BN 32, ?_⟩
  exact bmm_chunk_bwd_closed_form_correct A Dout Db s 32 1 8192 64 0 1 4096 1024 0 32 1 8192 64 0 1 BM BN 32 1 64
    (by norm_num) hInj hmlt hmlt hnlt hundef

end VeriTile.Bench.TritonBenchG.BmmChunkBwd

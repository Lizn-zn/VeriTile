# Spec sheet — `bench/tritonbench_g/attention_forward_triton/AttentionForwardTriton.lean`

**Python source:** `bench/tritonbench_g/attention_forward_triton/attention_forward_triton.py`

## Public theorem: `attention_forward_triton_final_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the attention-forward final output store. -/
```
</details>

**Statement:**
```lean
theorem attention_forward_triton_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_forward_triton_final_store_slice Acc Out H N_CTX
        HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            BLOCK_M idx))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)`

**Closed-form spec defs (transitive):** `outOffset`, `attention_forward_triton_final_store_slice`, `active`, `accOffset`, `offZ`, `offH`, `mIndex`, `kIndex`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>attention_forward_triton_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `attention_forward_triton.py`'s
`_attn_fwd`.

The full kernel computes a tiled attention accumulator with Q/K/V block loads,
quantization scales, and a streaming softmax reduction. This slice starts after
`acc = acc / l_i[:, None]` with a precomputed `Acc` tile and proves the final
masked writeback into `Out`. It preserves the source program-id decomposition
and the source store mask `(offs_m < N_CTX) & (offs_k < 96)`. The source forms
`O_block_ptr` with the Q strides, so this slice names those as the output store
strides. The inner `tl.float32` accumulator and `p.to(tl.float16)` dot-input
cast are outside this slice. -/
```
```lean
def attention_forward_triton_final_store_slice
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

## Also present (pinned special-case summaries)
- `attention_forward_triton_closed_form_correct`

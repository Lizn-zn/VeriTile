# Spec sheet — `bench/tritonbench_g/destindex_copy/DestindexCopy.lean`

**Python source:** `bench/tritonbench_g/destindex_copy/destindex_copy.py`

## Public theorem: `destindex_copy_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: the dual dest-indexed KV scatter implements the pure copy
`(xs, ys)` on its metadata IO signature — for every disjoint flat placement of
the buffers, every program id whose declared cells/lanes are in bounds, and
every launch state pinning the loaded destination row to `m₁` and the two source
tiles to `xs`/`ys`, the translated pointer kernel terminates, every lane of the
**`m₁`-indexed** `O_nope` / `O_rope` rows holds `xs j` / `ys j`, and every other
memory cell is unchanged.

Side conditions (required for truth, not convenience): `hNopeInj` / `hRopeInj`,
the tile part of each output address map is injective — without them the
unmasked scatter is last-writer-wins and the per-lane readback is false; both
are `Dest_loc`-independent (the loaded row only shifts every address by the
constant `m₁ · stride_o_*_bs`). `hRegion : O_nope ≠ O_rope`, so the `O_nope`
readback survives the later `O_rope` store.

Proof: `MetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
metadata triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification destindex_copy_correctness
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (hRegion : O_nope ≠ O_rope)
    (hNopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_NOPE => stride_o_nope_d * j.val))
    (hRopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_ROPE => stride_o_rope_d * j.val)) :
    destindexCopyIO KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
      ⊨ fun _ _ _ xs ys => (xs, ys)
```

**Assumptions / layout contracts:**
- `hRegion : O_nope ≠ O_rope`
- `hNopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_NOPE => stride_o_nope_d * j.val)`
- `hRopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_ROPE => stride_o_rope_d * j.val)`

**Closed-form spec defs (transitive):** `destindexCopyIO`, `fwd_kernel_destindex_copy_kv`

<details><summary><code>destindexCopyIO</code></summary>

```
/-- `_fwd_kernel_destindex_copy_kv`'s metadata-genre **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline (`MetaMasked2DKernelIO₂ₓ₂`,
the plain metadata genre with one `.nat` slot and two independent data tiles):

* `mbuf1 = Dest_loc`, `mwin1 = pid₀` — the `.nat` metadata slot: program
  `cur_index = pid₀` reads cell `pid₀` of `Dest_loc`, yielding the destination
  row `m₁`;
* `in1 = KV_nope → out1 = O_nope`, `in2 = KV_rope → out2 = O_rope` — the two
  data channels, `B1 = BLOCK_DMODEL_NOPE` / `B2 = BLOCK_DMODEL_ROPE` lanes;
* `read1`/`read2` — lane `j` reads
  `pid₀·stride_kv_*_bs + stride_kv_*_d·j` (the program's own source row);
* `write1`/`write2` — lane `j` writes `m₁·stride_o_*_bs + stride_o_*_d·j`: the
  address **eats the loaded slot value**, which is the whole point of the
  scatter;
* masks default to always-True (the copy loads and stores are unmasked).

The slot cell, windows, and masks are declared, not parsed from the kernel; the
headline **proves** the kernel's actual slot load, addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the declared cells. -/
```
```lean
def destindexCopyIO
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    MetaMasked2DKernelIO₂ₓ₂ where
  kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num
    BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
  mbuf1 := Dest_loc
  in1 := KV_nope
  in2 := KV_rope
  out1 := O_nope
  out2 := O_rope
  B1 := BLOCK_DMODEL_NOPE
  B2 := BLOCK_DMODEL_ROPE
  mwin1 := fun pid₀ _ => pid₀
  read1 := fun pid₀ _ _ j => pid₀ * stride_kv_nope_bs + stride_kv_nope_d * j.val
  read2 := fun pid₀ _ _ j => pid₀ * stride_kv_rope_bs + stride_kv_rope_d * j.val
  write1 := fun _ _ m₁ j => m₁ * stride_o_nope_bs + stride_o_nope_d * j.val
  write2 := fun _ _ m₁ j => m₁ * stride_o_rope_bs + stride_o_rope_d * j.val
```
</details>

<details><summary><code>fwd_kernel_destindex_copy_kv</code></summary>

```
/-- Faithful transcription of `destindex_copy.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL_*: tl.constexpr` -> Lean `Nat` parameters.
- The unused Python head-count and head-stride arguments are retained at the
  theorem boundary, matching the original kernel signature. -/
```
```lean
def fwd_kernel_destindex_copy_kv
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs _stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs _stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs _stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs _stride_o_rope_h stride_o_rope_d
      _kv_nope_head_num _kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_d_nope = tl.arange(0, $(BLOCK_DMODEL_NOPE))
  offs_d_rope = tl.arange(0, $(BLOCK_DMODEL_ROPE))
  dest_index = tl.load(Dest_loc + cur_index)

  kv_nope_ptrs = KV_nope + cur_index * $(stride_kv_nope_bs) +
    $(stride_kv_nope_d) * offs_d_nope[None, :]
  kv_rope_ptrs = KV_rope + cur_index * $(stride_kv_rope_bs) +
    $(stride_kv_rope_d) * offs_d_rope[None, :]

  o_nope_ptrs = O_nope + dest_index * $(stride_o_nope_bs) +
    $(stride_o_nope_d) * offs_d_nope[None, :]
  o_rope_ptrs = O_rope + dest_index * $(stride_o_rope_bs) +
    $(stride_o_rope_d) * offs_d_rope[None, :]

  kv_nope = tl.load(kv_nope_ptrs)
  kv_rope = tl.load(kv_rope_ptrs)

  tl.store(o_nope_ptrs, kv_nope)
  tl.store(o_rope_ptrs, kv_rope)
}
```
</details>

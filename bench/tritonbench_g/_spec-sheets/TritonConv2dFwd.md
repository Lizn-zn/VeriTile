# Spec sheet — `bench/tritonbench_g/triton_conv2d_fwd/TritonConv2dFwd.lean`

**Python source:** `bench/tritonbench_g/triton_conv2d_fwd/triton_conv2d_fwd.py`

## Public theorem: `conv2d_output_summary`

<details><summary>docstring</summary>

```
/-- The full conv2d forward surface lowers to the algorithm layer and realizes
the genuine convolution `convSpec` on every active output lane. The convolution
dot-accumulator, the per-block padding/boundary masking, and the masked
writeback are all proven; only the host launch / scheduling is trusted. -/
```
</details>

**Statement:**
```lean
theorem conv2d_output_summary
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (s : BlockState) (hBIN : 0 < BIN) (hundef : ∀ rg o, s.undef rg o = 0)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks)
    (hOutInj : Function.Injective (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups))) :
    (∃ alg, (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBIN : 0 < BIN`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hIGD : in_feat_dim / groups = BIN * numCBlocks`

**Closed-form spec defs (transitive):** `outputOffset`, `conv2d_forward_surface`, `batchIdx`, `featIdx`, `heightIdx`, `widthIdx`, `bhIdx`, `bhwIdx`

<details><summary><code>outputOffset</code></summary>

```
/-- The output write address for tile lane `(i,j)`. -/
```
```lean
def outputOffset (s0 : BlockState) (BHW OF OH OW OBS OOFS OHS OWS OGD : Nat)
    (idx : TileIndex [BHW, OF]) : Nat :=
  OBS * batchIdx s0 OH OW BHW idx.1 +
    OOFS * (s0.pids 2 * OGD + featIdx s0 OF idx.2.1) +
    OHS * heightIdx s0 OH OW BHW idx.1 + OWS * widthIdx s0 OW BHW idx.1
```
</details>

<details><summary><code>conv2d_forward_surface</code></summary>

```
/-- Faithful transcription of `triton_conv2d_fwd.py`'s `conv2d_forward_kernel`. -/
```
```lean
def conv2d_forward_surface
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel := triton {
  batch_height_width_pid = tl.program_id(0)
  out_feat_pid = tl.program_id(1)
  group_pid = tl.program_id(2)
  in_group_dim = $(in_feat_dim) // $(groups)
  out_group_dim = $(out_feat_dim) // $(groups)
  batch_height_width_offset =
    batch_height_width_pid * $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH) +
      tl.arange(0, $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
  batch_height_offset = batch_height_width_offset // $(out_width)
  batch_offset = batch_height_offset // $(out_height)
  output_feat_offset = out_feat_pid * $(BLOCK_SIZE_OUT_FEAT) +
    tl.arange(0, $(BLOCK_SIZE_OUT_FEAT))
  output_height_offset = batch_height_offset % $(out_height)
  output_width_offset = batch_height_width_offset % $(out_width)
    Input +=
      ($(input_batch_stride) * batch_offset +
        $(input_in_feat_stride) * group_pid * in_group_dim)[:, None]
    Weight +=
      ($(weight_out_feat_stride) * output_feat_offset +
        $(weight_out_feat_stride) * group_pid * out_group_dim)[None, :]
  accum = tl.zeros([$(BLOCK_SIZE_BATCH_HEIGHT_WIDTH), $(BLOCK_SIZE_OUT_FEAT)], dtype=tl.float32)
  for h in range($(0), $(kernel_height), $(1)) {
    for w in range($(0), $(kernel_width), $(1)) {
      for c in range($(0), in_group_dim, $(BLOCK_SIZE_IN_FEAT)) {
        input_feat_offset = c + tl.arange(0, $(BLOCK_SIZE_IN_FEAT))
        input_height_offset = h - $((padding_height : Int)) +
          $(stride_height) * output_height_offset
        input_width_offset = w - $((padding_width : Int)) +
          $(stride_width) * output_width_offset
          curr_input_pointer = Input +
            ($(input_in_feat_stride) * input_feat_offset)[None, :] +
            ($(input_height_stride) * input_height_offset)[:, None] +
            ($(input_width_stride) * input_width_offset)[:, None]
          curr_weight_pointer = Weight +
            ($(weight_in_feat_stride) * input_feat_offset)[:, None] +
            $(weight_height_stride) * h + $(weight_width_stride) * w
        input_mask = (batch_offset[:, None] < $(batch_dim)) &
          (input_feat_offset[None, :] < in_group_dim) &
          ($((0 : Int)) <= input_height_offset[:, None]) &
          (input_height_offset[:, None] < $(in_height)) &
          ($((0 : Int)) <= input_width_offset[:, None]) &
          (input_width_offset[:, None] < $(in_width))
        weight_mask = (input_feat_offset[:, None] < in_group_dim) &
          (output_feat_offset[None, :] < out_group_dim)
          input_block = tl.load(curr_input_pointer, mask=input_mask)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask)
        if _fp16 {
          input_block = (input_block).to(tl.float16)
          weight_block = (weight_block).to(tl.float16)
        }
        accum += tl.dot(input_block, weight_block, allow_tf32=_tf32)
      }
    }
  }
    Output += $(output_batch_stride) * batch_offset[:, None] +
      $(output_out_feat_stride) * (group_pid * out_group_dim + output_feat_offset)[None, :] +
      $(output_height_stride) * output_height_offset[:, None] +
      $(output_width_stride) * output_width_offset[:, None]
    output_mask = (batch_offset[:, None] < $(batch_dim)) &
      (output_feat_offset[None, :] < out_group_dim) &
      (output_height_offset[:, None] < $(out_height)) &
      (output_width_offset[:, None] < $(out_width))
    tl.store(Output, accum, mask=output_mask)
}
```
</details>

<details><summary><code>batchIdx</code></summary>

```
/-- `batch_offset = bhIdx // out_height`. -/
```
```lean
def batchIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i / OH
```
</details>

<details><summary><code>featIdx</code></summary>

```
/-- `output_feat_offset = pid1 · OF + j`. -/
```
```lean
def featIdx (s0 : BlockState) (OF : Nat) (j : Fin OF) : Nat := s0.pids 1 * OF + j.val
```
</details>

<details><summary><code>heightIdx</code></summary>

```
/-- `output_height_offset = bhIdx % out_height`. -/
```
```lean
def heightIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i % OH
```
</details>

<details><summary><code>widthIdx</code></summary>

```
/-- `output_width_offset = bhwIdx % out_width`. -/
```
```lean
def widthIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i % OW
```
</details>

<details><summary><code>bhIdx</code></summary>

```
/-- `batch_height_offset = bhwIdx // out_width`. -/
```
```lean
def bhIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i / OW
```
</details>

<details><summary><code>bhwIdx</code></summary>

```
/-- Flattened BHW index of lane `i`: `pid0 · BHW + i`. -/
```
```lean
def bhwIdx (s0 : BlockState) (BHW : Nat) (i : Fin BHW) : Nat := s0.pids 0 * BHW + i.val
```
</details>

## Also present (pinned special-case summaries)
- `conv2d_closed_form_correct`

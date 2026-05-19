---
title: "GPU 内存模型范围"
---

本文档说明 VeriTile 在谈到 "GPU memory" 时具体指什么。它是受支持 Triton 子集
的 functional-correctness 模型,而不是 CUDA 硬件的微架构模型。

## 已建模的层

| GPU 概念 | VeriTile 模型 | 备注 |
| --- | --- | --- |
| Global memory / HBM | `BlockState.mem : RegionName -> Nat -> MemCell` | `tl.load` 和 `tl.store` 的目标。`RegionName` 区分有名 buffer;`Nat` 是 cell offset,不是 byte 地址。proof-facing 的 Real contract 用 `readMem` / `writeMem`。 |
| Block-local 变量 | `BlockState.regs : RegFile` | 由赋值产生的逻辑 Triton SSA / register value。这不是物理 CUDA register file。 |
| 寻址元数据 | `TensorView`、`Offset.strided`、`InputAt` | 把逻辑 tensor 连接到 global-memory offset,用于 theorem 陈述和证明内部。 |

theorem-facing 的主要 surface 是:

```lean
TensorView.loaded s view tensor
TensorView.observe sf view idx
```

`TensorView` 是元数据:它记录一个 region、base offset、和每个 axis 的 stride。
它不存数据。数据存在 `BlockState.mem` 里,而现有 Real 值 theorem 通过
`BlockState.readMem` 来观测它。

## 寻址模型

VeriTile 当前支持 strided-affine tensor view:

```text
addr = base + sum_i idx_i * stride_i
```

这覆盖了当前 FA-1 4D view 以及常规连续/strided 示例使用的 layout。当某个
helper 还没有打包成 `TensorView` 时,底层证明仍可使用 `InputAt` 配合任意
offset 函数。

部分建模:

- 针对 `RegionName × Nat` 的一等公民 pointer value。
- `tl.make_block_ptr` / `tl.advance` 的 Triton 风格 block pointer value,
  以及带 zero padding / store skip 的 checked block-pointer load/store。

尚未建模:

- pointer cast、pointer comparison,以及 `RegionName` 等价之外的 pointer
  alias analysis。
- 顺序 lane 语义之外的硬件 / TMA block-pointer 行为。
- 完整 paged-KV 证明基础设施。存储模型现已支持核心的 indirect-addressing
  pattern——typed index load 喂入 pointer arithmetic——而 `IndirectView` 把
  read-only view 层打包好。bound、alias / page-ownership、以及 paged FA-1
  等价证明仍是后续 consumer 工作;见 issue #42。
- 丰富的 signed/unsigned 整数 dtype lattice。当前 typed HBM 模型支持
  `.nat`(`tl.uint8/uint16/uint32/uint64`)和数学 `.int`
  (`tl.int8/int16/int32/int64`),足够 index / block-table cell 使用,
  但不包含 bit-width 相关的整数语义。

## 顺序一致性

在单个 symbolic program instance 内部,内存是顺序一致的:

- 一个 `tl.store` 更新 `BlockState.mem`;
- 后继的 `tl.load`,只要 `RegionName` 和 offset 相同,就能观测到该更新;
- masked store 不修改被 mask 掉的 lane;
- 当 `other=None` 时,masked load 在 mask 掉的 lane 上使用
  `BlockState.undef`。

这是当前 per-program 证明的合适抽象。整网格执行只作为 theorem surface
建模:`GridIndex` 实例化 `BlockState.pids`,`Kernel.ForAllPrograms` /
`ForAllProgramsSome` 在 ND grid 上量化每个 program instance。VeriTile 还没建模
sequential 或 concurrent 的 launch executor、global memory merge、overlapping
write、race、atomic 或 scheduling。

Layer-2a frame reasoning 作为谓词级证明 contract 建模:
`WriteFootprint := (RegionName × Nat) → Prop` 和 `BlockState.WriteWithin`
表示一次 single-program execution 只修改了所给 footprint 内的 cell。Layer-2b
增加了 `Kernel.mergeFrames`:在显式 per-program `Kernel.ExecFrame` 上做
extensional、disjoint 的整网格 merge。这仍不是 concurrent / interleaved
executor;overlapping write、atomic、scheduling、barrier、async 和 shared
memory 仍在模型范围之外。

## 未建模

下列硬件层和性能效应有意不在当前语义 contract 内:

- Shared memory / SMEM 分配和 bank conflict。
- L1/L2 cache、cache modifier、eviction policy、volatile memory 行为,以及
  coalescing。
- 物理 register 分配、register pressure、spilling、occupancy、warp、lane,
  或 scheduling。
- Tensor Core / WGMMA 指令行为和 mixed-precision accumulation。
- Async copy、TMA、barrier、fence,或跨 program 同步。
- Atomic 操作和跨 block 的 memory race。

这些缺省意味着 VeriTile 为单个 symbolic Triton program instance 证明实数值的
functional correctness。它不证明性能性质,也不证明 CUDA memory-system fidelity。

## 扩展路径

当前模型刻意保持精简。已落地的 memory-proof 层有:

- **Memory safety / bounds (#48):** 在当前 typed storage 层之上的 active-lane
  region-bounds contract。
- **Write footprint / frame (#60):** 用于 single-program frame reasoning 的
  谓词级 `WriteFootprint` 和 `BlockState.WriteWithin` contract。
- **Disjoint whole-grid composition (#49):** `Kernel.mergeFrames` 在 write
  footprint 两两 disjoint 时合并显式 per-program `Kernel.ExecFrame`。

下一层 proof-ergonomics:

- **结构化 footprint 提取 (#61):** `WriteFootprint.tileImage`、
  `activeTileImage` 和 address-image helper 从常见的 direct、masked、checked
  block-pointer 以及普通 store pattern 中导出谓词 footprint。
- **Unrelated-frame helper (#62):** 便利引理,证明 single-program 或 grid
  footprint 之外的 cell 或整个 region 保持不变。

更长期的扩展点:

- **Paged KV / indirect addressing (#42):** 把当前 `IndirectView`
  smoke/proof surface 扩展为 paged-attention 专用的逻辑 view 和
  consumer 端等价 theorem。
- **Async 和并发 (#12):** `ConcurrencySemantics.md` 定义了 shared-memory
  state、barrier、atomic、async/TMA,以及显式 scheduling 或 trace 模型的
  边界。
- **浮点 fidelity (#11):** 在需要时把 `R` 抽象替换或细化为 IEEE /
  mixed-precision 语义。

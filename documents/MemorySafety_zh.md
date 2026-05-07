# Layer-1 内存 bounds safety

`VeriTile.Triton.Memory.Bounds` 定义了 Triton 内存操作的轻量 bounds-safety 层:

```lean
abbrev RegionBounds := RegionName -> Nat

def Op.MemorySafe (bounds : RegionBounds) : Op dtype shape -> Prop
def Stmt.MemorySafe (bounds : RegionBounds) : Stmt -> Prop
def Kernel.MemorySafe (bounds : RegionBounds) (k : Kernel) : Prop
def ComputeKernel.MemorySafe (bounds : RegionBounds) (ck : ComputeKernel) : Prop
```

公开 predicate 不依赖具体 state,但内存操作内部对所有 `BlockState`
做量化。这是必要的,因为 mask 和 first-class pointer 都是普通表达式,
其 active lane 和动态地址只有在求值后才能知道。

## Active lanes

load 与 store 只在能执行内存访问的 lane 上做检查。

- `MaskOpt.none`:所有 lane 都是 active 的。
- `MaskOpt.mask m`:只有 `m` 求值为 `true` 的 lane 是 active 的。
- `MaskOpt.maskOther m other`:地址安全性沿用与 `mask` 相同的 active
  lane;inactive lane 取 `other`,不读内存。
- Block-pointer `boundary_check`:`BlockPtr.inBounds` 检查为 false
  的 lane 平凡安全。checked load 返回 zero/默认 padding,checked store
  跳过该 lane 的写入。

这与 `EvalOp.lean` 和 `Step.lean` 中的 sequential 操作语义一致:
inactive lane 不会发出需要 region bound 的内存读写。

## Addressing 形式

predicate 覆盖 AST 中的三种内存寻址形式:

- 直接 region offset,`MemAccess.region region off`。
- First-class pointer 值,`MemAccess.ptr ptr`,通过对求值后的
  `(RegionName, Nat)` pointer lane 上的 operational
  `Op.PointerAddressesSafeOn` predicate 表达。
- Block pointer,`MemAccess.blockPtr ptr boundaryCheck`,checked 维度
  越界的 lane 不参与 bound 义务。

pointer-register provenance 在这里有意不解决。`VeriTile.Triton.Memory.Typing`
中的可选 checker 之后可以为动态 pointer-address 安全性提供充分条件;
bounds 这一层保持纯语义且可组合。

## Non-Goals

这一层不建模也不证明:

- Race freedom 或跨 program 内存组合。
- Frame theorem 或 ownership/permission accounting。
- Aliasing 或 page ownership。
- Atomic、barrier、async copy、shared memory,或 scheduling 语义。
- 硬件特定的内存层次行为。

这些属于并发 roadmap 的更高层;参见
`documents/ConcurrencySemantics.md`。`Kernel.MemorySafe` 仅是 per-program
active-lane region-bounds contract。

Layer 2a 单独在 `VeriTile.Triton.Memory.Frame` 中实现。它增加了
predicate 级 write footprint 和 single-program execution 的
`BlockState.WriteWithin` frame contract。Layer 2b 放在
`VeriTile.Triton.Launch.Composition`:在 write footprint 两两不相交时
合并显式的 per-program `Kernel.ExecFrame`。结构化 footprint 抽取与
proof automation 放在 `VeriTile.Triton.Memory.Footprint`(#61)。
这一层把 `WriteFootprint := MemCellAddr -> Prop` 作为语义接口,
并增加诸如 `WriteFootprint.tileImage`、`activeTileImage` 以及
block-pointer address-image helper 等 smart constructor。

Unrelated-memory preservation helper 放在同一个 frame stack 上(#62)。
在手动展开 `WriteWithin` 或 `GridWriteFootprint` 之前,
请先用 `BlockState.WriteWithin.mem_eq_of_not_written`、
`Kernel.ExecFrame.mem_eq_of_region_not_written`、
`Kernel.ExecWritesWithin.mem_eq_of_region_not_written`,以及
`Kernel.mergeFrames_mem_eq_of_region_not_written`。

## 示例

`VeriTile/Examples/MemorySafety.lean` 包含若干代表性证明:

- `straightLineCopy_memorySafe`:直接 region+offset load/store。
- `maskedTailAdd_memorySafe`:masked inactive lane 平凡安全。
- `blockPtrBoundary_memorySafe`:`boundary_check` 排除 checked
  block-pointer 越界 lane。

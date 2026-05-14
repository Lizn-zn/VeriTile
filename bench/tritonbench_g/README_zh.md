# TritonBench-G v1 port

VeriTile 对 [TritonBench-G v1](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) 的 port 工作区 —— 184 个 GitHub-scraped 真实 Triton kernel,作为 TritonBench(ACL 2025 Findings)的 headline channel 发布。

本目录每个 kernel 一个子目录。每个子目录打包 **上游 Python 源码 + VeriTile DSL port + 一个 per-kernel README**,所以单个 port 完整地住在一个文件夹里。

## 布局

```
bench/tritonbench_g/
├── README.md                       (本文件)
├── tritonbench_coverage.md         (静态覆盖分析,184 kernel)
└── <kernel_name>/
    ├── README.md                   per-kernel notes(状态、坑、TODO)
    ├── <kernel_name>.py            上游 Python 源码(pinned,见 Provenance)
    └── <KernelName>.lean           VeriTile DSL port;namespace `VeriTile.Bench.TritonBenchG.<KernelName>`
```

Lean 文件名是目录名的 **CamelCase 形式**(例如 `vector_addition/` 包含
`VectorAddition.lean`)。namespace 镜像它 —— 没有 `.Port` 后缀或其他填充。

## 状态解读

一个 port 经过三个阶段,在 per-kernel `README.md` 跟踪:

1. **DSL port** —— `<KernelName>.lean` 是上游 `.py` kernel 的
   **忠实 1:1 转写** 到 `triton { ... }` 语法。仅允许机械的 Lean 语法
   修改见 [`review_criteria_zh.md`](./review_criteria_zh.md)。如果 port 用了
   尚未 land 的 DSL surface,可能编译不过 —— 编译失败正是 DSL surface 需要
   扩展的预期信号。**当前编译通过:141 / 141 个 port pair;184 个工作目录中
   另外 43 个还是 README-only scaffold,不计为已完成 port。**
2. **Spec** —— 写出 kernel 预期输出的 Real-valued 数学规范。
3. **Verification** —— 证明 `ComputeCorrect.Realizes` /
   `ComputeRefine.Realizes` theorem,并登记到
   `scripts/kernel-manifest.tsv`。

阶段 1 是逐字转写契约;到达阶段 3(verification)既需要 DSL gap 关闭,
也需要 proof 落地。

## 当前审计状态

当前 sweep 见 [`completion_audit.md`](./completion_audit.md)。
`bench/check_ports.sh` 编译所有 Python/Lean port pair,当前报告
`TritonBench-G ports: 141 ok, 0 fail`。placeholder proof 扫描
`rg -n "True := by|trivial|sorry|admit" bench/tritonbench_g -g '*.lean'`
当前无匹配。

当前没有显式的 algorithm-layer `hAlg` blocker。后续如果新增 proof blocker,
应登记到 [`proof_blockers.md`](./proof_blockers.md)。

## 构建

这些 port 故意 **不在** `lakefile.toml` 主库 glob 中。它们与上游源码
一起放在 benchmark 工作区。要编译它们:

```bash
# 当前所有已 port 的 kernel
bench/check_ports.sh

# 当前 TritonBench-G sweep 的机械审计 gate
bench/audit_tritonbench_g.sh

# 按 kernel name 子集
bench/check_ports.sh vector_addition softmax_triton1
```

脚本对每个 `<KernelName>.lean` 独立跑 `lake env lean`,报告 per-kernel
pass/fail,任何失败时退出非零(CI-friendly)。
审计脚本在 port-build gate 外还检查 Python/Lean 数量、placeholder proof、
correctness surface、已编译 port 的 README 状态、`.to(tl.float32)` 覆盖、
Lean-only `tl.load(..., dtype=...)`、`keep_dims` 替换、`+=` 覆盖、
`rsqrt` 保留、Lean-only `tl.where`、`tl.*(...)` 调用集合和顺序、kernel
控制流计数,以及顶层 statement lhs 顺序。它是机械 gate;逐行忠实性仍以
[`review_criteria_zh.md`](./review_criteria_zh.md) 为准,未完成证明仍见
[`proof_blockers.md`](./proof_blockers.md)。

## Provenance

| 导入日期 | 上游 commit | Kernel | 备注 |
|---|---|---|---|
| 2026-05-06 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 15(Tier 1)| 初始 DSL port;暂无 spec / theorem |
| 2026-05-13 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 141 port pair | 当前已审计 port 集;剩余 proof obligation 见 `completion_audit.md` |

### 对 vendored `.py` 文件的本地修改

vendored `.py` 文件 **不是** 与上游严格逐字节相同。下列修改在所有
已导入文件上本地应用:

- **每个 `@triton.jit` kernel 签名上的输入类型标注。** Pointer 参数标注
  `tl.tensor`,运行时 int 标量 `tl.int32`,运行时 float 标量 `tl.float32`。
  上游的 `tl.constexpr` 标注原样保留。这些标注是 JIT 等价的(Triton 在
  编译时忽略非 `constexpr` Python type hint),所以 kernel 行为不变 ——
  它们存在纯粹作为 in-source 文档,把 Python 签名与 Lean port 依赖的
  类型信息对齐。

导入新批次时:

1. 在表中 pin 你 fetch 的上游 commit。
2. 确保上游 LICENSE 没有变更(当前 **Apache-2.0**)。
3. 在每个 `.py` 加 per-file attribution header(见下面的 [Licensing](#licensing))。
4. 给每个 `@triton.jit` 签名加上面描述的输入类型标注。

## Licensing

上游 `thunlp/TritonBench` 使用 **Apache-2.0** license。VeriTile 是 MIT
license;Apache-2.0 → MIT vendoring 在 attribution 之下被允许。

每个 vendored `.py` 应携带类似的 attribution header:

```python
# Source: thunlp/TritonBench@<commit-hash>
#   data/TritonBench_G_v1/<filename>.py
# Upstream license: Apache-2.0 (see https://github.com/thunlp/TritonBench/blob/main/LICENSE)
```

初始导入未带这些 header(commit `eab9b81`);回填是 open work。

## 添加 kernel

1. 从上游 `data/TritonBench_G_v1/` 选一个文件。
2. 在 [`tritonbench_coverage.md`](./tritonbench_coverage.md) 交叉对照
   它的判定。优先 `OK`;次 `Soft`;只有相关能力扩展已 land 时才碰 `Hard`。
3. 创建 `bench/tritonbench_g/<kernel_name>/`。
4. 把上游 `.py` 放进去,带上面的 attribution header。
5. 加 `<KernelName>.lean` 包含 DSL port。用
   `namespace VeriTile.Bench.TritonBenchG.<KernelName>`,导入
   `VeriTile.Triton.Core` + `VeriTile.Triton.DSL`。
6. 用 `bench/check_ports.sh <kernel_name>` 验证。
7. (阶段 2/3)写好 spec / proof 后,加一行 `scripts/kernel-manifest.tsv`,
   `source = tritonbench:<filename>.py`、`source_ref = <upstream-commit>`。
8. 如果从新上游 commit 导入,更新上面的 Provenance 表。

## 另见

- [`tritonbench_coverage.md`](./tritonbench_coverage.md) —— 全 184 个 kernel 的静态覆盖分类
- [`../README.md`](../README_zh.md) —— 整体 benchmark 政策
- [`../check_ports.sh`](../check_ports.sh) —— port 构建脚本
- [`../../documents/KernelManifest.md`](../../documents/KernelManifest.md) —— manifest schema(在阶段 3 用)
- [`../../documents/TheoremSurfaces.md`](../../documents/TheoremSurfaces.md) —— verification theorem 命名规约

---
title: "Kernel Manifest"
---

`scripts/kernel-manifest.tsv` 是 artifact 在 per-kernel 维度的权威注册表。
它替代了之前扁平的 theorem 与 examples 列表。

每条非注释行使用以下 tab 分隔的 schema:

```text
id	file	theorem	kind	status	source	source_ref	config	label	notes
```

## 字段

- `id`:稳定的机器可读标识符。必须唯一。
- `file`:包含该定理的 Lean 源文件。
- `theorem`:由 artifact 检查保证的公开定理符号。
- `kind`:定理类别。
- `status`:验证状态。
- `source`:kernel/源出处,例如 `internal`、`tutorial`、
  `paper:<name>` 或 `tritonbench:<path>`。
- `source_ref`:source commit、URL、paper anchor,不适用时填 `-`。
- `config`:重要的静态配置,例如 `BLOCK_N=128` 或
  `S=T=D=1,numIters=0`。
- `label`:简短的人类可读名字。
- `notes`:简化、丢弃的特性,或其他 caveat。

允许的 `kind` 值:

```text
correct refine math launch trace safety frame
```

允许的 `status` 值:

```text
proven projected test-gap blocked smoke
```

## 添加条目

新增公开 example 或 benchmark 定理时:

1. 普通 example 用公开的 `ComputeCorrect.*` 或 `ComputeRefine.*` 定理
   surface 增加 Lean 定理。详见
   [`CorrectnessSurfaces.md`](/VeriTile/proofs/correctness-surfaces/)。
2. 在 `scripts/kernel-manifest.tsv` 增加一行。
3. 把 source 和静态 config 记录得足够精确,使该 port 可复现。
4. 把简化或证明 scope 限制写进 `notes`。
5. 跑 `scripts/check-artifact.sh`。

artifact checker 会校验:每个 manifest 的 file 都存在、每个 theorem
符号都存在、id 唯一、`kind` / `status` 使用允许的词汇。

## 与 TritonBench-G 的关系

未来的 TritonBench-G 端口里,`source` 应标识 benchmark 条目,`source_ref`
应固定 upstream commit 或 URL。`config` 记录所选的 `BLOCK_*` / dtype /
静态 meta-parameter。`notes` 记录被丢弃或简化的特性,例如 dropout、
不支持的 async 路径,或暂缓的 IEEE compute 语义。

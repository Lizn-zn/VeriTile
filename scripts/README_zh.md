# 自动证明脚本

`scripts/prove.sh` 调用 Claude Code 的 `/lean4:autoprove` 搜索证明，
由 [Lean 官方 comparator](https://github.com/leanprover/comparator) 判定结果。

## 安装与使用

需要 Linux Landlock、可用的 systemd 用户服务、Go 1.24+、本项目 Lean
工具链，以及 Claude Code 的 `lean4` 插件。

```bash
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"
lake build VeriTile VeriTileFull
scripts/prove.sh path/to/Task.lean --theorem MyKernel.correctness \
  --max-cycles 20 --prompt "Try induction on n first."
```

文件必须位于项目内。路径和定理名是占位示例，请替换为实际任务。
**必须通过 `--theorem` 指定完整定理名**；可以重复此参数，检查多个定理。
默认最多 5 个证明周期。只对指定定理及其依赖作判定。

安装脚本固定使用适配 Lean 4.29.0 的官方 comparator 提交
`2a00b30df5e9173e70c4e4ec669fdf03da3163b9`、其 lockfile 中的 lean4export，
以及 landrun 提交 `5283024a2f49b28046c3b4a06d7d775c058d4d80`。
升级 Lean 时应同步检查这些版本。可以通过 `COMPARATOR_BIN` 指定其他兼容的
官方 comparator；`landrun` 和 `lean4export` 必须在 PATH 中。

## 判定过程

运行 agent 前复制原始题目、项目 Lean 源码、Lake 配置和依赖缓存。
agent 结束后，只将候选源码交给独立副本中的 comparator；不采用 agent
生成的构建产物或修改后的依赖。临时磁盘空间需容纳 `.lake` 副本，支持时使用 reflink。

Comparator 比较指定定理的陈述及其依赖定义，检查公理，并用 Lean 内核重放证明。
只允许 `propext`、`Quot.sound` 和 `Classical.choice`。遵循官方建议，同时使用
systemd 的 AF_UNIX 限制和 landrun 沙箱；工具缺失或沙箱失败时不会降级判定。

- 退出 0：comparator 接受全部指定定理。
- 非零退出：证明被拒绝，或工具、沙箱、参数等出错。

agent 自报成功或编译输出中没有 `sorry` 都不能替代此判定。
日志保存在 `Logs/<basename>_<unique-id>/`，包括原题、候选源码、配置、
comparator 输出、agent JSON 流、输入源码哈希和 `result.json`。
临时缓存副本在结束时删除。

初始项目、依赖和工具需要可信。本脚本没有把本地编码 agent 与宿主机隔离；
对恶意 agent 的评测需使用独立的求解和判定环境。
陈述中引用的 private 或模块自动生成名称可能在两个模块间不同，导致合法证明
也被拒绝；任务的 kernel/spec 依赖宜使用公开名称。

## 回归检查

```bash
python3 scripts/test_prove.py
```

使用真实 comparator 和固定输出的模拟 agent，不调用模型 API。
覆盖正常证明、修改陈述/定义、额外公理、残留 sorry、删除目标及多定理判定。
更多说明及 artifact 检查入口见 [English documentation](README.md)。

## 统一 comparator 门禁

`scripts/check-artifact.sh`、`bench/check_ports.sh` 和 `bench/audit_trust.sh`
都必须通过官方 comparator。使用前按上面的步骤安装工具并设置 PATH；工具缺失、
导出失败或 comparator 拒绝都会使门禁失败。CI 通过同一个本地 action 安装固定版本。

```bash
scripts/check-artifact.sh
python3 scripts/check_comparator.py --library
python3 scripts/check_comparator.py --file bench/examples/VectorAdd.lean --trust
python3 scripts/test_check_comparator.py
```

日常检查在独立临时目录冻结当前可信源码和构建缓存，再由 comparator 导出、检查
公理并重放证明（`source-replay` 模式）。它不比较历史 Git 版本，规格变更仍需要审阅。
`prove.sh` 继续对照 agent 修改前保存的原始题目进行比较。

库检查覆盖 manifest 中标为 `proven` 的库定理。独立文件检查枚举 Lean 环境中保留的
全部定理对象，包括私有和宏生成名称，通过别名交给 comparator。仅含定义的测试文件
明确报告零条目标定理，并额外重放一个平凡哨兵；这不代表证明了某个 kernel 正确。
Lean 不保留为定理声明的匿名 `example` 仍属于编译测试，不计入证明目标。

原有 headline、公理、陈述和循环规格检查继续执行。完整 bench 审计合并编译和
comparator 重放，避免重复检查整个语料。批量 worker 共用一份独立快照，各自使用
独立输出模块。直接 `lake build` 仍是构建步骤，证明验收请使用上述门禁。

`Logs/comparator-check-*` 保留源码哈希、目标清单、生成源码、comparator 配置、
诊断、二进制哈希和退出状态；批次结束时删除临时构建缓存。

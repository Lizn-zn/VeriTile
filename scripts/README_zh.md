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

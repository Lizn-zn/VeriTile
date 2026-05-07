# scripts/prove.sh

`lean4` Claude Code 插件 `/lean4:autoprove` 命令的轻量 wrapper。
用于 VeriTile 的 LLM benchmark eval(详见 `PLAN.md` §LLM benchmark protocol)。

## 用法

```bash
scripts/prove.sh <lean_file> [--max-cycles N] [--prompt "extra text"]
```

示例:

```bash
# 用最多 5 个 cycle 关掉 held-out 文件中所有 sorry
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean

# 更激进的搜索
scripts/prove.sh path/to/hard.lean --max-cycles=20

# 带策略提示
scripts/prove.sh path/to/file.lean --prompt "Try induction on n first."
```

## Exit code

- 0 —— `/lean4:autoprove` 报告成功(启发式判断:干净退出、result subtype
  不是 "error"、result text 不提到失败)
- 1 —— 失败(cycle 用尽、卡住、claude error 或参数错误)

日志写到 `Logs/<basename>_<timestamp>.json` 供查看 / 调试。

## 固定的插件版本

wrapper 的行为依赖已安装的 `lean4` Claude Code 插件版本。

截至 2026-04-26,我们使用 `~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/`
下的版本。如果结果不再可复现,先检查插件版本。

## 这 *不是* 什么

- 不是自定义 prover —— 所有证明搜索都发生在 `/lean4:autoprove` 内部
- 不直接连接 lake —— 插件内部处理 `lake env lean`
- 不直接用 Anthropic API —— 入口点是 `claude -p`(Claude Code CLI)

为什么是 wrapper 而不是自定义 Python 工具,见 `PLAN.md` 决策日志条目 5。

## Artifact checker

`scripts/check-artifact.sh` 是 Lean artifact 的本地 release/CI gate。
它会跑 `lake build`、拒绝 Lean `sorry` warning、按
`scripts/artifact-axiom-whitelist.txt` 检查声明的 axiom、校验
`scripts/kernel-manifest.tsv` 中的 per-kernel 注册表,并检查 README
example 链接漂移。

`scripts/kernel-manifest.tsv` 是公开 kernel/example 元数据的 source of truth:
file、theorem 符号、theorem kind、验证状态、source、静态 config、label
和 notes。新增公开 example 或 benchmark 端口前,先看
`documents/KernelManifest.md`。

```bash
scripts/check-artifact.sh
```

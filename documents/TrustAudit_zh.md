# 信任审计 — 使用说明

[English](TrustAudit.md) | **中文**

可机械验证的门禁,证明一条定理没有隐藏的 `sorry`、没有偷带公理、没有自指
spec。一条定理的可信度只取决于它的**陈述**和它的**公理足迹**——与它证明里用到
的引理无关——这几个 command 检查的正是这一点。

## 跑门禁

```bash
# 每条 proven 的库定理都公理干净(134 条)
lake build VeriTile.Meta.TrustReport

# 每个 bench port + showcase 都公理干净(159 个文件)
bash bench/audit_trust.sh                    # 整个语料
bash bench/audit_trust.sh swiglu_fwd         # 只指定 kernel

# 上面两道 + port 完成度检查,CI 一步跑完
bash bench/audit_tritonbench_g.sh
```

各自 exit `0` 当且仅当全部通过。一条 `proven` 定理若没通过,那是**真的 soundness
发现**(有 `sorry`/公理漏进去了)——去修证明,绝不削弱门禁。

## 自己审一条定理

`import VeriTile.Meta.StatementAudit`,然后:

```lean
#axiomsClean my_theorem
-- ✓ my_theorem: axiom footprint ⊆ standard base
```

四个 command:

| Command | 检查 |
|---|---|
| `#axiomsClean T` | 足迹 ⊆ `{propext, Classical.choice, Quot.sound}` —— 主门禁 |
| `#stmtSurfaceSubset T ⊆ [a, b, …]` | `T` 的陈述不得出现白名单外的项目常量 |
| `#specNonCircular s avoiding [k, …]` | spec `s` 的定义不得引用 kernel `k` |
| `#auditStmt T` | 检视 —— 列出 `T` 陈述里的项目常量 |

## 给一个文件加自审计

把检查放在文件末尾(完整范式见 SwiGLU pilot
[`bench/examples/Swiglu.lean`](../bench/examples/Swiglu.lean))。
它们在编译期运行——任何一道不过,文件就编译失败:

```lean
#axiomsClean my_main_theorem
#stmtSurfaceSubset my_main_theorem ⊆ [my_kernel, InputLoadedAt, ComputeRefine.Refines]
#specNonCircular my_spec avoiding [my_kernel]
```

## 东西在哪

- Command 定义:[`VeriTile/Meta/StatementAudit.lean`](../VeriTile/Meta/StatementAudit.lean)。
- 库层驱动(生成):[`VeriTile/Meta/TrustReport.lean`](../VeriTile/Meta/TrustReport.lean)
  —— 用 `python3 scripts/gen_trust_report.py` 从 manifest 重新生成。
- bench 驱动:[`bench/audit_trust.sh`](../bench/audit_trust.sh)。

`TrustReport` 放在 `VeriTileFull` 库里(它要审 `ApproxGeLU`,会拖进重型分析链),
所以日常的 lite `lake build` 仍然快。

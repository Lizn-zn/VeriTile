# 信任审计 — 使用说明

[English](TrustAudit.md) | **中文**

可机械验证的门禁,证明一条定理没有隐藏的 `sorry`、没有偷带公理、没有自指
spec。一条定理的可信度只取决于它的**陈述**和它的**公理足迹**——与它证明里用到
的引理无关——这几个 command 检查的正是这一点。

## 跑门禁

```bash
# 一次性安装固定版本的 comparator、导出器和沙箱工具
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"

# 用 comparator 重放 manifest 中全部 proven 库定理
python3 scripts/check_comparator.py --library

# 审计 bench port、showcase 和基础设施测试
bash bench/audit_trust.sh                    # 整个语料
bash bench/audit_trust.sh swiglu_fwd         # 只指定 kernel

# 上面两道 + port 完成度检查,CI 一步跑完
bash bench/audit_tritonbench_g.sh
```

只有每个选定文件都有结果且全部检查通过时才返回 `0`。非法并发参数、启动失败、
结果缺失或重复也会使门禁失败。`#axiomsClean` 拒绝一条 `proven` 定理表示证明
依赖了未经允许的公理；基础设施失败会单独报告，不能据此认定证明错误。

三个入口都必须通过官方 comparator。库层覆盖 proven manifest 目标；独立文件
导出 Lean 环境中保留的全部定理声明，包括私有和宏生成声明。Comparator 检查
公理依赖，并用 Lean 内核重放证明。日常审计使用冻结的当前源码快照，不比较历史
Git 规格；`scripts/prove.sh` 另外对照 agent 修改前的原题。仅含定义的测试文件会
明确报告零条原始目标定理。安装、范围和日志说明见 [脚本文档](../scripts/README_zh.md)。

bench 审计会追加 `#auditModuleAxioms` 和 `#auditModuleSpecs`。
`specification` 会把定理登记到 Lean 环境，公理门禁检查所有登记的 headline，
以及沿用 `*_correct` / `*_output_summary` 后缀的定理，包括换行、私有和 Unicode
名称。manifest 中的明确名称也会补充进入检查。每个文件报告实际 headline 数；
零表示没有发现需要进行公理审计的 headline。

规格审计从 Lean 编译后的返回类型发现 kernel，
支持多行和带参数的声明，并检查所有 `*Spec` 定义。其他名称的数学规格用
`@[kernel_spec]` 注册。执行语义定义（包括 `denotation` 声明）用
`@[kernel_denotation]` 标明角色，单独计数；它们有意依赖 kernel，不能与独立数学
规格混淆，也不能借这个属性豁免数学正确性目标。输出逐文件列出 kernel、规格和
执行语义数量；独立规格数为零表示该文件没有进行独立性检查。发现规格但没有
发现 kernel 时检查失败。

## 自己审一条定理

`import VeriTile.Meta.StatementAudit`,然后:

```lean
#axiomsClean my_theorem
-- ✓ my_theorem: axiom footprint ⊆ standard base
```

可用 command:

| Command | 检查 |
|---|---|
| `#axiomsClean T` | 足迹 ⊆ `{propext, Classical.choice, Quot.sound}` —— 主门禁 |
| `#auditModuleAxioms` | 从 Lean 环境发现已登记的 headline 和沿用后缀的定理，检查公理足迹 |
| `#stmtSurfaceSubset T ⊆ [a, b, …]` | `T` 的陈述不得出现白名单外的项目常量 |
| `#specNonCircular s avoiding [k, …]` | spec `s` 的定义不得引用 kernel `k` |
| `#auditStmt T` | 检视 —— 列出 `T` 陈述里的项目常量 |
| `#auditModuleSpecs` | 在 Lean 环境中发现本模块 kernel/规格，检查传递依赖并报告覆盖情况 |

## 给一个文件加自审计

把检查放在文件末尾(完整范式见 SwiGLU pilot
[`bench/examples/FusedSwigluEquiv.lean`](../bench/examples/FusedSwigluEquiv.lean))。
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
- 统一 comparator 驱动:[`scripts/check_comparator.py`](../scripts/check_comparator.py)。

`TrustReport` 放在 `VeriTileFull` 库里(它要审 `ApproxGeLU`,会拖进重型分析链),
所以日常的 lite `lake build` 仍然快。

## 审计边界

陈述检查按 Lean 环境中记录的声明来源模块，排除可信的 `Init`、`Std`、
`Lean` 和 `Mathlib` 依赖。不会按名称前缀排除：项目中的 `Nat.wrapper`、
`Real.wrapper` 或 `instSomething` 仍会被检查，包括从其他项目模块导入时。
规格依赖遍历会沿项目别名继续展开，在上述可信依赖处停止。

`#axiomsClean` 检查传递公理依赖；另外两项检查约束陈述与定义中出现的常量。
这些检查不能替代对规格是否表达目标行为、假设是否充分的审阅。
`scripts/check-artifact.sh` 的项目公理白名单是另一项检查：它构建 `VeriTile/`
下的所有模块，再从 Lean 环境枚举实际公理声明，包括私有声明和宏生成的声明。
注释、字符串和源码格式不会影响发现结果。白名单中的公理并不因此获准进入
`#axiomsClean` 所检查的定理。

# 从已接受原语到 TritonBench 等价性证明

本例使用 PR #9 的已发布结果，按用户选择直接信任报告，不重新运行或重放 GPU 实验。
`scripts/export_numerical_rules.py` 检查配置、源码哈希、报告完整性和接受状态，将 42 个实例中的 30 个接受项导出到
[ReportedAdmission.lean](../../VeriTile/Triton/Float/ReportedAdmission.lean)。WARN、REJECT、定义域事件及不适用条目均无对应的导出规则。

## 用户看到的 specification

[TritonBenchVectorAdditionFP.lean](../../bench/examples/TritonBenchVectorAdditionFP.lean)
复用 TritonBench `vector_addition` 的结构，具体化为 `4096×4096` 个元素、block=1024、fp32 输入/计算/输出。
两个 kernel 唯一的差异是 `output = x + y` 与 `output = y + x`：

```lean
specification vector_addition_equiv (R : Rules) :
    originalKernel ≡[R] optimizedKernel := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions (body originalKernel) (body optimizedKernel)
  rw [original_decomposition, optimized_decomposition]
  exact .frame beforeAdd afterAdd (admitted_add_commute R)
```

`R` 固定引用导出的 `fp32_add_commute`；用户不再提供任意 contract/evidence，也不手填 PASS。
它只包含一条数值模型假设：信任该报告中的 ADD-COMMUTE 及其与指定 fp32 加法片段的对应关系。
这个前提保留为 `Rules.add_comm`，它正是“实验通过后 assume 该原子关系”的逻辑表达。
Lean 证明只交换加法输入，检查加载、地址、mask、store 和程序签名保持一致。
Lean 不证明 GPU 确实运行过，也不把统计接受转换成 IEEE 位值等式；本例没有引入全局公理。

`real_projection` 检查 fp32 版本的数学投影等于原来的 TritonBench kernel，已有实数正确性证明继续适用。
实验 shape 是完整数据的形状；kernel 以 block=1024 遍历其连续展平存储。每个被重写的加法直接读取原始独立正态输入，未把中间值擅自当成同分布输入。

## 运行和检查

```bash
python3 scripts/export_numerical_rules.py --trust-report
lake build VeriTile.Meta.StatementAudit VeriTile.Triton.Float.Equivalence VeriTile.Triton.Float.ReportedAdmission TritonBenchSpecExamples
lake env lean bench/examples/TritonBenchVectorAdditionFP.lean
```

最后一条命令完成 Lean 检查。查看浮点证明引用的原子假设使用：

```lean
#print_fp_assumptions vector_addition_equiv
```

输出只有原子假设名称，使用小写下划线形式：

```text
FP assumptions used by vector_addition_equiv:
  add_commute
```

该命令沿证明及其辅助引理寻找 `Derivation.atom`，不把规则表中未引用的条目算进去；
重复的同一实例只显示一次，不同配置的实例仍分别列出，但不打印配置或 gate 状态。
它也适用于普通浮点 theorem，
不要求使用 `specification` 关键字。不引用原子的推导显示 `none`；若证明依赖不可展开的
浮点证明前提，则明确显示 `unresolved FP proof`，不会猜测其原子集合。
遍历涵盖可达证明分支，不声称求出了逻辑上的最小依赖集合。
公理审计是独立检查：示例仍执行 `#auditModuleAxioms`，只隐藏检查通过时的提示。
原来的 `#print_spec ... full` 保留兼容，用于内部详细审计。
`empirical_max` 保留在报告数据里，不被改写为尾部置信上界。

冻结输入后的回归和现有官方 comparator 检查：

```bash
python3 scripts/export_numerical_rules.py --trust-report --check
python3 -m unittest scripts.test_export_numerical_rules scripts.test_specification_surface scripts.test_fp_assumptions -v
python3 scripts/check_comparator.py --file bench/examples/TritonBenchVectorAdditionFP.lean --trust
```

comparator 使用独立输入快照；报告生成和原子绑定必须在证明任务开始之前完成。
如需调用已有证明 agent，入口仍是 `scripts/prove.sh`，目标为本例的完整 specification 名称，无需另一套证明器。

## 范围

导出表已有全部 30 个接受实例，本例完成了其中 fp32 ADD-COMMUTE 到实际 TritonBench 语句的绑定与组合证明。
其他原子的参数化语法绑定应按同一方式明确表达舍入/cast/FMA，不能把一个接受行任意套到不同片段。
导出的 `report:...` 标识绑定发布的 JSON 快照，不伪造仓库外原始 bundle 的 instance key、观测或 PTX 哈希。
更改报告会产生新的标识；更改数值实现会使源码哈希检查失败；更改本例的 shape、block 或精度会使 `report_matches` 编译失败。
任意复合算法仍需由已接受的局部关系推导，不能借这个导出入口把整个算法直接假设为等价。

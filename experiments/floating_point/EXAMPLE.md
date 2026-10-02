# 从已接受原语到 TritonBench 等价性证明

本例使用主目录当前发布的结果。GPU 观测在发布前独立 CPU 重放；Lean 导出阶段按用户选择信任该报告，不重复运行实验。
`scripts/export_numerical_rules.py` 检查配置、源码哈希、报告完整性和接受状态，将 42 个实例中的当前接受项导出到
[ReportedAdmission.lean](../../VeriTile/Triton/Float/ReportedAdmission.lean)。WARN、REJECT、INCONCLUSIVE、定义域事件及不适用条目均无对应的导出规则。

## 用户看到的 specification

两个 Lean 文件分别展示正确性和浮点等价性，各自直接写出原始 Triton kernel。

[TritonBenchVectorAdditionCorrect.lean](../../bench/examples/TritonBenchVectorAdditionCorrect.lean)
包含 kernel、IO 接口、`real_projection` 和实数正确性规格，不引入数值原子假设：

```lean
specification vector_addition_correct (nElements blockSize : Nat) :
    Spec.Real (addIO nElements blockSize ⊨ fun xs ys i => xs i + ys i)
```

`⊨` 保留原 TritonBench 的完整内存规格：在有效 lane 上输出数学和，保证终止，
并保持其他内存不变。该证明复用原来的 TritonBench 实数正确性定理。

[TritonBenchVectorAdditionFPEquiv.lean](../../bench/examples/TritonBenchVectorAdditionFPEquiv.lean)
直接定义原始 kernel 和交换加法操作数后的版本，并证明浮点等价性；不导入 correct 文件。
`nElements` 和 `blockSize` 是任意符号参数，不与实验的尺寸或启动配置核对。
两个 kernel 唯一的差异是 `output = x + y` 与 `output = y + x`：

```lean
specification vector_addition_equiv (nElements blockSize : Nat) (R : Rules blockSize) :
    originalKernel nElements blockSize ≡[R] optimizedKernel nElements blockSize := by
  refine ⟨rfl, ?_⟩
  change Spec.Derivation R.assumptions
    (body (originalKernel nElements blockSize)) (body (optimizedKernel nElements blockSize))
  rw [original_decomposition, optimized_decomposition]
  exact .frame (beforeAdd nElements blockSize) (afterAdd nElements blockSize) (admitted_add_commute R)
```

`R` 固定引用导出的 `fp32_add_commute`；用户不再提供任意 contract/evidence，也不手填 PASS。
它只包含一条数值模型假设：使用实验选出的 fp32 ADD-COMMUTE 原子关系。
这个前提保留为 `Rules.add_comm`，它正是“实验通过后 assume 该原子关系”的逻辑表达。
`Rules blockSize` 中的参数只用于实例化带形状的语法，不要求 `blockSize` 等于实验值。
Lean 证明只交换加法输入，检查加载、地址、mask、store 和程序签名保持一致。
Lean 不证明 GPU 确实运行过，也不把统计接受转换成 IEEE 位值等式；本例没有引入全局公理。

`real_projection` 仅位于 correct 文件；FP 推导不使用它或 `vector_addition_correct`。
流程分成两步：实验使用指定 shape、分布、精度等参数选择 assumption；
Lean 随后只在该 assumption 下证明，不重新检查实验 shape 或输入分布。
因此这里没有 `rows = 4096`、`columns = 4096`、`blockSize = 1024` 之类的固定常量。

## 运行和检查

```bash
python3 scripts/export_numerical_rules.py --trust-report
lake build VeriTile.Meta.StatementAudit VeriTile.Triton.Float.Equivalence VeriTile.Triton.Float.ReportedAdmission TritonBenchSpecExamples
lake env lean bench/examples/TritonBenchVectorAdditionCorrect.lean
lake env lean bench/examples/TritonBenchVectorAdditionFPEquiv.lean
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
python3 scripts/check_comparator.py --file bench/examples/TritonBenchVectorAdditionCorrect.lean --trust
python3 scripts/check_comparator.py --file bench/examples/TritonBenchVectorAdditionFPEquiv.lean --trust
```

comparator 使用独立输入快照；报告生成和原子绑定必须在证明任务开始之前完成。
如需调用已有证明 agent，入口仍是 `scripts/prove.sh`，目标为本例的完整 specification 名称，无需另一套证明器。

## 范围

导出表包含当前报告的全部接受实例，本例完成了其中 fp32 ADD-COMMUTE 到实际 TritonBench 语句的绑定与组合证明。
其他原子的参数化语法绑定应按同一方式明确表达舍入/cast/FMA，不能把一个接受行任意套到不同片段。
导出的 `report:...` 标识绑定发布的 JSON 快照，不伪造仓库外原始 bundle 的 instance key、观测或 PTX 哈希。
更改报告会产生新的标识；更改数值实验实现会使源码哈希检查失败。
本例在任意 `nElements`、`blockSize` 下由同一原子关系推导；没有实验尺寸匹配条件。
规则选择仍区分操作和精度，不能将 bf16 或乘法实验行作为 fp32 加法的来源。
任意复合算法仍需由已接受的局部关系推导，不能借这个导出入口把整个算法直接假设为等价。

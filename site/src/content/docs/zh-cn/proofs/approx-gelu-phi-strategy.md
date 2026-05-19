---
title: "ApproxGeLU 中段策略:Phi / 临界点证明"
---

目标:

```text
|approxGeLU(x) - exactGeLU(x)| <= 1e-3
```

实现使用

```text
c0 = 7978845608028654 / 10000000000000000
k  = 44715 / 1000000
A(x) = c0 * (x + k*x^3)
g(x) = tanh(A(x)) - erf(x/sqrt(2))
D(x) = x/2 * g(x)
```

`D` 是偶函数,所以只需证明正半轴。

已有覆盖:

```text
0 <= x <= 0.83      Taylor-at-zero 证明
3.8 <= x           独立的 tail 证明
```

针对 `[0.83, 3.8]` 的有前景的中段证明:

设

```text
alpha = sqrt(2/pi)
Phi(x) =
  log(c0 / alpha)
  + log(1 + 3*k*x^2)
  + x^2/2
  - 2 * log(cosh(A(x))).
```

则

```text
g'(x)
  = c0*(1+3*k*x^2)*sech(A(x))^2 - alpha*exp(-x^2/2)
  = alpha*exp(-x^2/2) * (exp(Phi(x)) - 1).
```

这把 tanh/erf 的导数对消保留在 `Phi` 内部。

用实数算术得到的数值符号结构:

```text
Phi 零点:
  alpha1 ~= 1.2420843929
  alpha2 ~= 2.5921448834

D' 零点:
  x1 ~= 1.3644977174
  x2 ~= 2.6989413864

D(0.83) ~= -9.3620e-5
D(x1)  ~= -2.3367e-4
D(x2)  ~=  4.7324e-4
D(3.8) ~=  9.8804e-5
```

形式化时为了避免数值根附近过窄的符号 margin,我们使用稍宽的 critical box:

```text
box1 = [1.36449, 1.36451], center 1.3645
box2 = [2.69893, 2.69895], center 2.69894
```

如果证明符号表

```text
D'(x) < 0 on [0.83, 1.36449)
D'(x) > 0 on (1.36451, 2.69893)
D'(x) < 0 on (2.69895, 3.8]
```

并证明两个中心值

```text
|D(1.3645)| <= 1/2000
|D(2.69894)| <= 1/2000
```

那么 Lean checker 已经能把中段证完。这两个 box 足够窄,粗略的全局导数
界 `|D'| <= 10` on `[0,4]` 即可控制每个 box 内的所有点。

形式化计划:

1. 定义 `geluPhi`。
2. 证明导数对消恒等式:

   ```text
   geluHDerivExpr x (geluY x) (geluG x)
     = sqrt(2/pi) * exp(-x^2/2) * (exp(geluPhi x) - 1)
   ```

3. 证明或生成 `geluPhi` 与 `D'` 的区间符号 certificate。
4. 用 MVT/单调性把 `[0.83, 3.8]` 化简为端点/critical-box 上的取值检查。

来自非形式推导的重要修正:实现的 GeLU 用的是十进制有理数 `c0`,
不是定义意义上的 `sqrt(2/pi)`,所以 `Phi` 中需要 `log(c0/alpha)` 项。

## ODE 仿射 Tube 更新

后续基于生成 certificate 的路径绕开了临界点符号表:直接在中段证明
`F = x/2 * H` 的 tube。

对每段 `[a,b]`,候选数据用一个仿射预测子

```text
f(x) ~= f0 + (x-a) * d0
```

加初始误差 `E0` 和导数 residual `R`:

```text
|f(a) - f0| <= E0
|f'(y) - d0| <= R
```

Lean checker `affine_step_forall_mem` 由这些事实证明 scalar tube。
数值上,步长 `0.02` 的 interval-affine certificate 可以覆盖
`[0,3.8]`;实际中段 `[0.83,3.8]` 需要 149 行。仅使用有理数界
`alpha in [79/100, 4/5]`,scout 报告

```text
step = 0.02
worst |F| tube ~= 5.23e-4
```

形式上的 caveat 很关键:`H'` 与 `F'` 的 residual 证明本身想用
正在被证的同一个 state tube。生成的某一行不能简单地假设
`Y/G/H/F` 已经在 tube 内来证 residual,那是循环论证。Lean 这一侧目前
使用的抽象是 self-consistent tube invariant / first-exit lemma:

```text
若
  初始 state 在 tube 内,
  只要 state 在 tube 内,RHS 就映射到 derivative box,
  且仿射 step 把该 derivative box 严格映射到 tube 内,
则
  真实 state 在整段上保持在 tube 内。
```

一旦这个 theorem 存在,每一行生成数据都化归为有理多项式 RHS 包含
加上有理 budget 检查。

当前 Lean hook:

```text
Icc_subset_of_closed_local_right_extension
affine_state_step_F_tube_of_local_right_extension
affine_state_step_F_tube_of_strict_prefix
```

生成器应以 `affine_state_step_F_tube_of_strict_prefix` 为目标:
先证明左端点的 state box,然后在 prefix invariant `Icc a t` 之下,对
每个 `t < b` 都证明 `Y/G/H/F` 的八条严格 slack 不等式。连续性把
这些严格不等式转换为所需的 right-neighborhood extension。

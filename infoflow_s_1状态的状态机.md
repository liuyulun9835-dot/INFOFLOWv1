# InfoFlow · 顺序性点事件 — 数据策略回测模式（S1 方案）

> 目标：把 InfoFlow V1.0 §3.2 的四阶段（潜伏/爆发/过载/衰减）落成**可复现的顺序性点事件**标签，并在 1m OHLCV 上做面向交易的统计验证与门槛评估。

---

## 0. 名词与记号
- 序列：
  - `C_t ∈ [0,1]`：Clarity（结构清晰度，已平滑）
  - `E_t ∈ [0,1]`：I_em（信息释放强度，已平滑）
  - `I_t ∈ R`：ICI（一致性，已平滑，可为负）
- 一阶差分：`ΔC_t = C_t - C_{t-1}`，`ΔE_t = E_t - E_{t-1}`，`ΔI_t = I_t - I_{t-1}`
- Z 标准化（滚动窗口 `L`）：`Z_X(t) = (X_t - μ_t) / σ_t`
- 失配（可选）：`M_t = |ΔC_t - ΔE_t|`（亦可做 Z）
- 推荐阈值：
  - `z_hi = +0.25`（≈60 分位）、`z_lo = -0.25`、`ε = 0.1`（ICI 近零带）
  - 标签以**右闭**（bar close）计算，时间戳为当前 bar 的 `t`。

> 注：指标时间框架建议为 15m 或 30m；回测对齐 1m OHLCV。

---

## 1. 四阶段“点事件”判定（非强制顺序）
### A. 潜伏（Latent）
- 条件：`Z_C(t) ≥ z_hi` 且 `Z_E(t) ≤ z_lo` 且 `|Z_I(t)| ≤ ε` 且 `ΔC_t ≥ 0`。
- 含义：结构已集中、信息未释放、一致性贴近零。

### B. 爆发（Release）
- 条件：`Z_I(t-1) ≤ 0` 且 `Z_I(t) > 0` 且 `Z_E(t) ≥ z_hi` 且 `ΔE_t > 0`。
- 保护（可选）：`Z_C(t) ≥ z_lo`（避免结构恶化时的误点火）。

### C. 过载（Overload）
- 条件（满足其一）：
  1) `Z_E(t) ≥ z_hi` 且 `ΔC_t < 0`；
  2) `Z_M(t) ≥ z_hi`（释放-结构失配升高）。
- 保护（可选）：`ΔI_t ≤ 0`。

### D. 衰减（Decay）
- 条件：`Z_I(t-1) ≥ 0` 且 `Z_I(t) < 0`，并且满足其一：`ΔE_t < 0` 或 `Z_E(t) ≤ z_lo` 或 `Z_M(t) ≥ z_hi`。

> 以上四类均为**点事件**；同一 bar 允许多重触发，后续“交易组装层”再做优先级裁剪（见 §4）。

---

## 2. 顺序性机制（有限状态机，FSM，可选）
若需**强制顺序**（潜伏→爆发→过载→衰减→潜伏…）：
- 当前状态 `S_t ∈ {Latent, Release, Overload, Decay}`，初始为 `Latent`。
- 状态迁移：
  1. `Latent → Release`：满足“爆发”；且 `t - t_Latent ≤ T_max`（如 `T_max = 3L`）。
  2. `Release → Overload`：满足“过载”；且 `ΔI_t ≤ 0`。
  3. `Overload → Decay`：满足“衰减”。
  4. `Decay → Latent`：满足“潜伏”，或经历冷却 `K` 个 bar 且 `|Z_I| ≤ ε`、`Z_C ≥ z_hi`。
- 防抖：每次迁移后冻结 `F` 个 bar（如 `F = 2`）。

> FSM 仅改变**打标签许可**，不改变 §1 的数学条件。

---

## 3. 数据与对齐
- 指标侧：从 LitScript 导出 `time, C, E, I` 三线；时间粒度 = 指标时间框架（如 15m）。
- 价格侧：1m OHLCV（`time, open, high, low, close, volume`）。
- 对齐：以指标 bar close 的时间 `t0`，在 1m 上进行未来收益计算。

```text
收益定义： r(Δ) = (P_{t0+Δ} - P_{t0}) / P_{t0}
Δ ∈ {30, 60, 120} 分钟（可扩展）
```

- 成本：设 `fee_bp ∈ {0, 5, 10}`（基准/ +50% / ×2），以基点从收益中扣减。

---

## 4. 交易组装与去重
- 事件优先级：`Release > Latent > Overload > Decay`（可配置）。
- 同 bar 多事件：取优先级最高者；或合并为“阶段切换”多标签样本用于统计（非交易）。
- 冷却：同向触发 `cooldown` bar 内不重复入场（如 10–20 个 1m bar）。

---

## 5. 统计验证指标
**每类事件**单独统计：
1) **未来收益分布**：`Δ=30/60/120m` 的均值、中位数、胜率、P25/P75、极值、偏度、峰度；
2) **稳健性**：自助法置信区间（bootstrap 5k）、Cliff’s delta / Mann-Whitney U vs. 全体；
3) **风控视角**：最大顺/逆行（`MAE/MFE`）、低分位回撤（P5 MDD_forward）。

可选：
- **成本敏感性**：`fee_bp` 三档对均值/胜率的影响曲线；
- **稀疏性惩罚**：事件样本数 `N` 低于阈值（如 < 50）时仅出描述性统计，不给显著性结论。

---

## 6. 验收门槛（MVP）
- **Release**：在 `Δ=60m` 下，`mean(r_Δ - fee) > 0` 且 `winrate ≥ 55%` 且 `Cliff’s d > 0`；
- **Decay**：在 `Δ=60m` 下，`mean(r_Δ) < 0` 或 对冲/反向策略 `mean(-r_Δ - fee) > 0`；
- **Overload**：`MAE_forward` 显著大于基线（波动放大，提示风险）；
- **Latent**：`r_Δ` 方差低于基线（结构稳定）。

至少一类（通常是 Release）通过门槛；否则 S1 方案不作为交易原型，仅保留为市场状态注释器。

---

## 7. 工程流程（可复制）
### 7.1 数据接口
```yaml
inputs:
  indicators_csv:  path/to/indicator_15m.csv   # time,C,E,I
  ohlcv_1m_csv:    path/to/kline_1m.csv       # time,open,high,low,close,volume
params:
  L: 60
  z_hi: 0.25
  z_lo: -0.25
  eps: 0.1
  horizons: [30, 60, 120]
  fee_bp: [0, 5, 10]
  cooldown_1m: 15
  fsm: false  # true 开启顺序性
```

### 7.2 伪代码（标签 + 回测）
```python
# 读入指标(15m)与OHLCV(1m)，做 rolling z
for t in bars_15m:
    zC, zE, zI = z(C[t]), z(E[t]), z(I[t])
    dC, dE     = C[t]-C[t-1], E[t]-E[t-1]
    M          = abs(dC-dE); zM = z(M)

    latent  = (zC>=z_hi) and (zE<=z_lo) and (abs(zI)<=eps) and (dC>=0)
    release = (zI_prev<=0) and (zI>0) and (zE>=z_hi) and (dE>0)
    overload= ((zE>=z_hi) and (dC<0)) or (zM>=z_hi)
    decay   = (zI_prev>=0) and (zI<0) and ((dE<0) or (zE<=z_lo) or (zM>=z_hi))

    # FSM 可选：若启用则校验合法迁移
    label = pick_one(latent, release, overload, decay, priority=[release, latent, overload, decay])

    # 映射到1m并计算未来收益
    t0 = close_time_of_15m_bar(t)
    for Δ in horizons:
        rΔ = (px[t0+Δ] - px[t0]) / px[t0]
        for fee in fee_bp:
            rec = {"t0":t0, "label":label, "Δ":Δ, "ret":rΔ - fee*1e-4}
            write(rec)
```

### 7.3 输出与报告
- `events.parquet`：逐事件打点及未来收益表
- `summary.csv`：分组统计（均值/胜率/Cliff’s d/样本数）
- `qc.md`：样本覆盖、异常/空值、指标稳定性快照

---

## 8. 风险与失效模式
- 指标偏弱期：`|Z_I|` 长期贴近零，四阶段触发稀疏；
- 极端行情：快速跃迁导致“爆发→过载→衰减”在数个 bar 内连发，需 `cooldown`；
- 跨框架：15m→1m 对齐带来滑点与延迟，需在报告中列示；
- 过拟合：阈值 `z_hi/z_lo/ε` 不做事后调参，可通过 **walk-forward** 进行稳健性验证。

---

## 9. 快速验收清单（CheckList）
- [ ] 指标 CSV（三线）与 1m OHLCV 均已对齐；
- [ ] 标签脚本按 §1 或 §2 产出四阶段事件；
- [ ] 未来 30/60/120m 收益表生成，含三档成本；
- [ ] `summary.csv` 达成 §6 的至少一项门槛；
- [ ] 生成 `qc.md` 截图与异常说明；
- [ ] 归档版本号与参数（L、z_hi、z_lo、eps、cooldown、fsm）。

---

## 10. 附：门槛的直观解读
- **Release 通过**：说明“结构与释放的一致化”确有后验价格优势（短期 α）；
- **Decay 有显著负向**：可为减仓/对冲触发器；
- **Overload 的 MAE 放大**：主要用于风控（提示风险），不直接产 α；
- **Latent 的方差收缩**：适合识别“低噪声等待区”。


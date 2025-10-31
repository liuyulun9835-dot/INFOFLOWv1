# InfoFlow · 斜率相位差点事件 — 数据策略回测模式（S2 方案）

> S2 是 S1 “顺序性点事件” 的动态升级，将索性阈值规则替换为 **“方向体 + 相位差 + 斜率动力”** 触发器，以提升事件标注的灵敏性和预览性。

---

## 0. 名词与记号
- 序列：
  - `C_t ∈ [0,1]`：Clarity (结构清晰度)
  - `E_t ∈ [0,1]`：I_em (信息释放强度)
  - `I_t ∈ R`：ICI (一致性)
- 导数：
  - 一阶：`dC_t = C_t - C_{t-1}`，`dE_t = E_t - E_{t-1}`，`dI_t = I_t - I_{t-1}`
  - 二阶：`aI_t = dI_t - dI_{t-1}` (动力加速度)
- 相位差：`Δφ_t = t_E^* - t_C^*` (在滑动窗口中估计：最近一个峰/谷点时间差)
  - `Δφ_t > 0` → 能量先行；`Δφ_t < 0` → 结构先行。
- 标准化：`Z_X = (X - μ_roll) / σ_roll`，窗口 `L = 60` (可调。)

---

## 1. 新事件规则（斜率+相位差动力版）

| 阶段 | 分类 | 新规则 | 说明 |
|------|------|------|------|
| **A. 潜伏 Latent** |  结构稳定、动能低 | 1. `Z_C > z_hi`, `Z_E < z_lo`<br>2. `|dC_t|, |dE_t| < ε_slope`<br>3. `|Δφ_t| < φ_min` (相位差稳定)<br>4. `aI_t ≈ 0` | 静止期：结构高度集中，能量稳定低清晰。 |
| **B. 爆发 Release** | 能量先行一致性上升 | 1. `dE_t > 0`, `dC_t ≥ 0`, `dI_t > 0`<br>2. `Δφ_t > φ_up` (能量先行)<br>3. `aI_t > 0` (动力加速)| 信息释放与结构同步增强，ICI 速度和动力均正，为 **方向初始化事件**。|
| **C. 过载 Overload** | 能量仍强，结构突然下降 | 1. `dE_t > 0`, `dC_t < 0`<br>2. `Δφ_t ≈ 0` (同步正在破坏)<br>3. `aI_t < 0` (动力衰退)<br>4. 或 `Z_E > z_hi` 而 `Z_C` 未创新高 | 溢点期：释放过度，ICI 由加速转衰，**结构背离时间流**。|
| **D. 衰减 Decay** | 一致性破裂、动能转载 | 1. `Z_I < 0`, `dI_t < 0`, `aI_t ≤ 0`<br>2. `Δφ_t < -φ_down` (结构先行)<br>3. `dC_t, dE_t < 0` | 结构进入解细，能量消散，ICI 下幅快于 0 载进，是 **转折/断流事件**。|

> 参数建议：`z_hi=0.25`，`z_lo=-0.25`，`ε_slope=0.001~0.002`，`φ_min=0.1`，`φ_up=+0.3`，`φ_down=-0.3`。

---

## 2. 事件变化规律与取样优化

### 2.1 时间点优化
原 S1 采样 = 阈值突发时刻 (Z 超阈)，S2 使用 **斜率抵积** 和 **相位转换点** 作为时间标签：
- **Latent → Release** 时：`aE_t` 从负转正，`Δφ_t` 由 0 转 正，即 **动能加速 且 能量预先**，作为新 t0 。
- **Release → Overload** 时：`aI_t` 由 正 转 负，同时 `Δφ_t` 回收 到 0，识别 **信息同步破坏点**。
- **Overload → Decay** 时：`aI_t<0` 持续，`Δφ_t<0`，`Z_I` 下穿 0，标注 **能量消散转折**点。

### 2.2 观测策略
在这种动态模型中，可以通过 ICI 的 **斜率及其二阶导数** 定义 “信息组织速度”：
- 若 `dI_t>0`, `aI_t>0` → 组织在加速。
- 若 `dI_t>0`, `aI_t<0` → 组织在衰退。
- 若 `dI_t<0`, `aI_t<0` → 组织在解细。

---

## 3. 数据流程与标签模型
```yaml
inputs:
  indicators_csv: path/to/indicator_15m.csv
  ohlcv_1m_csv:   path/to/kline_1m.csv
params:
  L: 60
  z_hi: 0.25
  z_lo: -0.25
  eps_slope: 0.0015
  phi_up: 0.3
  phi_down: -0.3
  horizons: [30, 60, 120]
  fee_bp: [0, 5, 10]
```

**算法流程（伪代码）**：
```python
for t in bars_15m:
    zC,zE,zI = z(C[t]),z(E[t]),z(I[t])
    dC,dE,dI = diff(C),diff(E),diff(I)
    aI = dI - dI.shift(1)
    dphi = phase_diff(C,E,window=L)

    latent  = (zC>z_hi)&(zE<z_lo)&(abs(dC)<eps)&(abs(dE)<eps)&(abs(dphi)<phi_min)
    release = (dE>0)&(dC>=0)&(dI>0)&(dphi>phi_up)&(aI>0)
    overload= (dE>0)&(dC<0)&(abs(dphi)<phi_min)&(aI<0)
    decay   = (zI<0)&(dI<0)&(aI<=0)&(dphi<phi_down)&(dC<0)&(dE<0)

    label = pick_one(latent,release,overload,decay,priority=[release,latent,overload,decay])

    # 回测：1m 对齐
    for Δ in horizons:
        rΔ = (px[t+Δ]-px[t])/px[t]
        for fee in fee_bp:
            write(t,label,Δ,rΔ-fee*1e-4)
```

---

## 4. 验证优势
| 项 | S1（阈值法） | S2（斜率+相位差法） |
|----|--------------|------------------|
| 灵敏性 | 延迟2-3bar | 提前1-2bar (先导) |
| 稳健性 | 高 | 略低（可平滑斜率） |
| 采样频率 | 低 | 中等 (增20%-40%) |
| 触发事件 | 过零点 | 斜率抵积/相位转换 |
| 专段说明力 | 状态切换 | 动态组织触发 |

**定性提升**：更早探测市场“能量组织”阶段，信号时序更贴近真实市场动力。

---

## 5. 验收门槛
- 至少一类事件（通常 Release）在 `Δ=60m`下有明显正向收益，`winrate>55%`，`Cliff's d>0`。
- S2 对 S1 优势评估：
  - 时间预期提前 > 1 bar；
  - 采样效率不低于 S1 的 0.8 倍；
  - 成本效率更高，退回消耗更少。

---

## 6. 实施流程

1) **指标标准化模块**  
- 输入：`C,E,I` 三条时间序列（同一粒度，如 15m）。  
- 处理：
  - `zscore`：滚动窗口 `L`（默认 60）计算 `Z_C, Z_E, Z_I`；
  - **斜率**：`dC = C - C[-1]`，`dE`，`dI`；
  - **加速度**：`aI = dI - dI[-1]`；
  - **相位差** `Δφ`：窗口内最近极值时间差法（默认）；如需精细可切换 Hilbert 相位法。  
- 输出：`Z_C, Z_E, Z_I, dC, dE, dI, aI, dφ`。

2) **标签器（事件识别）**  
- 按 §1 的 S2 规则，逐 bar 产生四类互斥标签：`latent / release / overload / decay`。  
- 可选 **FSM 去抖**：状态切换后冻结 `F=2` bars；最大存续时长 `Tmax=3L`，防抖阈 `ε_slope`。  
- 输出：`event_label`（枚举）与 `event_conf`（0~1 置信度，可由规则满足项计分归一得到）。

3) **时间对齐与取样**  
- 事件时间戳使用指标周期的 **bar close**（记为 `t0`）。  
- 对齐 1m OHLCV，在 `{30, 60, 120}` 分钟取前瞻收益：`rΔ=(P[t0+Δ]-P[t0])/P[t0]`。  
- 记录：`t0, label, Δ, rΔ, cost_mode`（`base / +5bp / +10bp`）。

4) **分组统计与显著性检验**  
- 对四类事件分别统计 `mean/median/std/winrate/Skew/Kurtosis/Cliff's d`；  
- 与全样本/随机对照比较；做 KS/ MWU / t‐test（视分布而定）。

5) **报告生成**  
- 图表：
  - 事件叠加时序图（标注 `release/overload/decay` 点）。
  - 未来收益箱线图与分布曲线（四类事件 × 多 Δ）。
  - ICI 斜率/加速度与价格的双轴图；`Δφ` 时序与极值对齐图。  
- 表格：各 Δ 的收益汇总与显著性表；敏感性分析表（`L, ε_slope, φ_up/down`）。

---

## 7. 回测协议（Protocol）

- **数据**：指标（15m/30m/1h 任一）、价格（1m），时间区间按样本可用段自动剪裁；跳过前 `L+5` bars（指标稳定期）。  
- **滑点/成本**：`0/5/10 bp` 三挡；买卖对称（或接入真实成交费率）。  
- **样本独立性**：若事件间隔 < `cooldown K`（默认 2L），仅保留优先级高的一个（`release > latent > overload > decay`）。  
- **多周期一致性**：可选“多粒度共识”过滤：若 `15m` 与 `30m` 在 ±1 bar 同类事件，则置信度 +1 档。

---

## 8. 评估指标（KPIs）

- **命中率**：`winrate(Δ)`；**稳健性**：`median rΔ`、`IQR`。  
- **效应量**：`Cliff's d` / `Cohen's d`；**显著性**：`p‐value (KW/ MWU/ t)`。  
- **提前性**：与 S1 的事件时间差（bar 数）；**覆盖率**：S2 事件数量 / S1 事件数量。  
- **成本敏感**：`PnL_base` vs `PnL_costed` 的差值与排序稳定性。  
- **鲁棒性**：参数扰动（±20%）下结论一致性（符号一致率 ≥ 80%）。

---

## 9. 参数与默认值

```yaml
L: 60
z_hi: 0.25
z_lo: -0.25
eps_slope: 0.0015
phi_min: 0.10
phi_up: 0.30
phi_down: -0.30
horizons: [30, 60, 120]
fees_bp: [0, 5, 10]
freeze_bars: 2
max_duration: 3*L
cooldown: 2*L
priority: [release, latent, overload, decay]
```

---

## 10. 伪代码框架（实现参考）

```python
# 1) 指标导入 & 标准化
X = read_indicators()  # C,E,I at 15m
C,E,I = X.C, X.E, X.I
ZC,ZE,ZI = zscore(C,L), zscore(E,L), zscore(I,L)
dC,dE,dI = diff(C), diff(E), diff(I)
aI = diff(dI)
phi = phase_diff_peaks(C,E,window=L)  # 或 hilbert_phase(C,E)

# 2) 事件打标（S2 规则）
latent  = (ZC>z_hi)&(ZE<z_lo)&(abs(dC)<eps)&(abs(dE)<eps)&(abs(phi)<phi_min)&(abs(aI)<eps)
release = (dE>0)&(dC>=0)&(dI>0)&(phi>phi_up)&(aI>0)
overload= (dE>0)&(dC<0)&(abs(phi)<phi_min)&(aI<0)
decay   = (ZI<0)&(dI<0)&(aI<=0)&(phi<phi_down)&(dC<0)&(dE<0)
label = resolve_conflict([release,latent,overload,decay], priority)

# 3) 1m 对齐与取样
returns = sample_forward_returns(px_1m, t0=bar_close_times, horizons, fees_bp)

# 4) 统计与可视化
stats = grouped_stats(returns, by=label)
plots = make_plots(C,E,I,phi,dI,aI,events=label)
export_report(stats, plots)
```

---

## 11. 数据与产出结构

**输入**  
- `indicators_15m.csv`: `time, clarity, iem, ici`  
- `kline_1m.csv`: `time, open, high, low, close, volume`

**输出**  
- `events_15m.csv`: `time, label, conf, ZC, ZE, ZI, dC, dE, dI, aI, phi`
- `forward_stats.csv`: `label, Δ, mean, median, std, winrate, cliffs_d, p_value`
- `report.html/.md`: 图表与结论摘要

---

## 12. 风险与注意事项
- 
- **极端行情**：剧烈跳空会破坏 `Δφ` 的平滑性；建议在极端波动时扩大 `L` 或暂时只用斜率条件。  
- **样本泄漏**：相位估计窗口不能跨越 `t0`，保持右闭无前视。  
- **过拟合**：背离阈值 `φ_up/down` 请做留出集网格搜索，并按效应量而非 p‐value 单独优化。

---

## 13. 下一步
1. 先用你已上传的 15m 指标与 1m 价格跑一版 **S2 vs S1** 的对照检验；  
2. 若 `release@60m` 的效应量更高、且提前性>1 bar，则将 S2 录入 **MVP 白名单**；  
3. 针对 `φ_up/down` 做 3×3 网格扰动，验证鲁棒性后，进入半自动信号层。  


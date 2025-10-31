# **L2 × L3 综合回测方案（v0.1）— 纯 L2 数据采集（1m/5m/15m/1h）下的双线测试**

> 目的：在仅依赖 L2 指标线（Clarity=C、L_em=E、ICI=I）与由其触发的 L3 事件标签的前提下，完成**两条并行回测**：
> 1) **L3：事件 → 后续波动性**（验证“事件是否改变未来不确定性结构”）；
> 2) **L2：指标线 → 特征性**（验证“这些线在频域/相位/形状层面到底长什么样，噪声与信息如何分离”）。
>
> 约束：不做方向收益回测；所有检验只围绕“未来波动结构”。

---

## 0. 数据域与 I/O 规范
- **时间尺度**：`{1m, 5m, 15m, 1h}`；其中 5m/15m/1h 由 1m 复采样，或直接并行抓取（以时间戳对齐为准）。
- **L2 指标线**：C、E、I 三线（TradingLite / 你的 TL 导出）；为每个 tf 生成宽表（timestamp, C, E, I）。
- **L3 事件**：由 L2 线触发的离散事件（release/decay/latent/overload/…），单表形式（timestamp, event_name, strength, tf）。
- **价格基线**：OHLC（同 tf）。仅用于波动度量与标准化，不参与方向性计算。
- **路径继承**：继续使用既有 CONFIG（TL_BASE / PX_BASE / TIMEFRAMES / OUTDIR 等）与工具函数（_read_tradinglite_tf, _read_px_1m, _resample, _winsorize, _zscore_rolling）。
- **对齐原则**：以 utc 时间戳为主键；禁止前向填充；缺失段剔除并写入 QC 日志。

---

## 1. L3 路线：事件 → 后续波动性（升级版）
### 1.1 现状与第一性原理
- 从“信息流守恒 & 结构可分性”出发：事件是参与者信念相干性的重组；后续**波动结构**变化应先于可持续的方向差异。
- 目标：检验不同事件之后的**未来波动**是否存在**统计显著差异**与**稳定效应量**。

### 1.2 事件定义（可迭代模块）
> 事件是“可组合语法”（见 §2.2），当前占位口径可采用：
- **Release**：`I` 上穿 0 轴（`I_{t-1} < 0 ∧ I_t > 0`）。
- **Decay**：`I` 下穿 0 轴（`I_{t-1} > 0 ∧ I_t < 0`）。
- **Overload**：`|I_t| > θ_high ∧ C_t > θ_c_high`（极端摆动 + 高结构）。
- **Latent**：`|I_t| < θ_low ∧ |C_t| < θ_c_low`（能量/结构双低）。
> 阈值 θ、θ_c 初期用分位/滚动 z-score（如 |z|>2）；后续替换为“事件原子语法”组合体。

### 1.3 结果变量（未来波动的三口径）
- **RV_H**：未来 H bar 的 log-return 平方和；
- **RV_rel_H**：RV_H / 事件前基线（如过去 K×tf 的 RV 均值或 ATR 近似）；
- **HL 类近似**：Parkinson 或 Garman–Klass 多期合成（备查）。

### 1.4 实验控制
- **冷却与重叠**：t0 所属 bar 不纳入 RV；同窗多事件冲突采取“最先触发优先”或“强度最大优先”，并设置 embargo（=H）。
- **分层**：会话/时段分层（固定效应）；对照组可用同日同段随机非事件 bar 匹配（可选）。

### 1.5 统计检验与效应量
- **总体差异**：Kruskal–Wallis；
- **两两比较**：Dunn/Conover + FDR（BH）多重校正；
- **方差齐性**：Levene / Brown–Forsythe；
- **效应量**：Cliff’s δ + 中位差 + bootstrap CI；
- **敏感性**：事件 × H × tf 的显著性/效应量热图。

### 1.6 产出与路径
- `${OUTDIR}/{tf}/events.parquet`：事件清单；
- `${OUTDIR}/{tf}/rv_stats.parquet`：逐事件 RV 指标；
- `${OUTDIR}/{tf}/kw_results.csv`、`dunn_posthoc.csv`、`effect_size.csv`；
- `${OUTDIR}/{tf}/sensitivity_matrix.csv`；
- `${OUTDIR}/summary/report_event_volatility.md`（含热图、结论、签名）；
- `${OUTDIR}/qc/qc_log.json`。

---

## 2. L2 路线：指标线的特征性回测（SNR/频域/相位/形状）
### 2.1 目的与原则
- 目的：回答“C/E/I 这三条线**各自**与**彼此**在结构上到底是什么”，明确**信噪边界**与**可解释频带**，为事件语法与 L3 触发提供坚实底座。
- 原则：只以“未来波动”作为监督（互信息或相关），不引入方向收益。

### 2.2 事件原子语法（供 L3/特征共享）
- **单线原子**：Cross（阈值/零轴）、Extrema（极值+曲率阈值）、Burst（短窗能量突发）、RegimeSwitch（主频跃迁）、EntropyDrop（复杂度骤降）。
- **跨线原子**：Coherence↑（指定频带的相干上升）、PhaseSlip（相位差异常跳变）、Lead–Lag（互相关峰偏移稳定）。
- **组合子句**：原子以 AND/OR/SEQUENCE 组合（示例：`Coherence↑@低频 ∧ EntropyDrop ∧ |I|>θ`）。
> 以上原子会被用于两侧：一是 L3 事件重构；二是 L2 特征工程与筛选。

### 2.3 特征簇（按“频域—时间域—相干—形状”四条线）
**A. 频域/能量**  
- Welch PSD 主峰频率、峰宽、累计能量 80/90% 带宽；
- 低频/中频/高频能量比（自适应分频或分位分频）；
- ΔSNR：去噪前后 SNR 提升（小波阈值/SSA/EMD 任一标准管线）。

**B. 时间域/稳健统计**  
- 滑窗均值/中位/IQR/偏度/峰度；
- 局部 RV、短窗 |Δline| 分位；
- 样本复杂度/样本熵/Permutation Entropy。

**C. 相干与相位**（跨 C/E/I）  
- 特定频带的 magnitude-squared coherence；
- 固定频带相位差的均值/方差/异常跳变计数；
- 互相关峰值与滞后量（正负 bar）。

**D. 形状学**  
- 极值点密度、极值曲率分布（离散二阶差分）；
- 突发检测（Bollinger 风格的 z 爆点）与持续时长；
- 频带跃迁次数/停留时长（由 STFT/CWT 主频随时间的轨迹离散化）。

### 2.4 特征评价与筛选
- **有效性**：与未来 RV 的互信息、HSIC 或 distance correlation；
- **稳定性**：跨时间、跨 tf 的排名稳定性（Kendall/IC 稳定度）；
- **冗余控制**：特征间相关性聚类 + 代表元选取（避免多重共线）；
- **解释性**：频带/相位可视化可读；优先留“能画出来”的特征。

### 2.5 去噪与“最小解释器”（可选）
- 候选：小波包 best-basis（Shannon 代价）、PCA/ICA、SSA、轻量 HMM/HSMM；
- 评估：压缩比、MDL（描述长度）、与未来 RV 的 MI、跨期稳定性；
- 目标：选出**最少成分**且**解释力不降**的解码器，用作事件原子/阈值的基底。

### 2.6 产出与路径
- `${OUTDIR}/{tf}/l2_features.parquet`：逐时间戳特征表（C/E/I + 组合特征）；
- `${OUTDIR}/{tf}/feature_scores.csv`：有效性/稳定性/冗余评估；
- `${OUTDIR}/{tf}/snr_report.md`：SNR 与频谱快照；
- `${OUTDIR}/summary/report_l2_characteristics.md`：L2 特征性回测总述；
- `${OUTDIR}/qc/qc_log_l2.json`。

---

## 3. 统一验证矩阵与验收线
| 维度 | 指标 | 基线验收 |
|---|---|---|
| **事件显著性（L3）** | KW p 值（FDR 后）、Cliff’s δ | `p_adj<0.05` 且 `|δ|≥中等`（≥0.33 级别可优先） |
| **跨尺度一致性** | 5m/15m/1h 的结果同向 | ≥2 个尺度同结论 |
| **特征有效性（L2）** | MI(RV)、排名稳定性 | MI 显著>0；IC/τ 稳定 |
| **去噪收益** | ΔSNR、MDL、MI 变化 | ΔSNR>0；MDL↓且 MI 不降 |
| **工程可复现** | schema/signature/manifest | 全部产物签名化并落档 |

---

## 4. 执行顺序（里程碑视图）
1) **数据抓取与对齐**：四尺度 L2 线 + OHLC；QC 缺口报告；
2) **L3 事件占位实现**：零轴/阈值口径快速跑通，生成第一版 `events.parquet`；
3) **L3 波动性回测**：RV/RV_rel + KW/Dunn/FDR + 敏感性热图；
4) **L2 特征工程 v1**：频域/相位/形状四簇特征生成与评分；
5) **去噪/最小解释器试点**（可选）：best-basis or PCA/ICA；
6) **事件语法替换**：用“原子语法”组合重构事件触发，重跑 L3；
7) **统一报告与白名单**：将稳定通过验收线的事件与特征入库；
8) **冻结结果 → 治理**：更新蓝皮书/validator 文档与签名；
9) **（预留）方向性探索入口**：仅在 L3+L2 稳定后开启（独立 PR）。

---

## 5. 报告与治理
- **Schema**：`schema_version: l2l3_backtest_v0.1`，记录 `build_id / data_manifest_hash / event_rule_version / feature_set_hash / utc_timestamp`；
- **文档**：
  - `report_event_volatility.md`（L3）
  - `report_l2_characteristics.md`（L2）
  - `event_atoms.md` + `event_grammar.md`（语法库）
- **看板指标**（dashboard 占位）：事件显著性面板、特征 MI 排名、ΔSNR 曲线、跨尺度一致性热图。

---

### 附：占位事件清单（可替换）
- Release：`I` 上穿 0；Decay：`I` 下穿 0；Latent：`|I|<θ & |C|<θ_c`；Overload：`|I|>θ_high & C>θ_c_high`；Transition：相邻事件间隔 `Δt≤N_bar`。

> 注：本方案为工程化蓝本；实际口径与参数将以 S1/S2/S3 版本事件定义与特征白名单为准，在同一统计与治理框架下逐步替换。


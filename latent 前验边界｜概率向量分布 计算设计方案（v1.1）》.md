# #《latent 前验边界｜概率向量分布 计算设计方案（v1.1）》
**核心目的（一句话）**：**划定 latent 的“前边界”**。当盘面进入“前夜形态”（低波动且继续收敛、三种过程概率向量回到历史前验椭球、强度不过低）时，给出**可执行、可治理**的进入标记，服务 **校准与对齐**，并自然耦合 **ENF 负向过滤** 与“前夜→点火”的序列事件。

# 0. 读者须知<!-- {"fold":true} -->
* •	本文是**产品级计算方案**，供数据工程、特征工程、算法/回测、治理与数据分析等团队协作实施。
* •	全文**只用 L2 数据**（OHLCV + C/E/ICI 单阈），避免升阶与过度压缩；所有阈值来自**历史分位**，保证可解释与可复现。

⠀
# 1. 现状（我们手里已经有什么）
* •	有标注好的 **latent 事件时刻**（或等价的进入/退出口径）。
* •	有稳定的 **OHLCV** 历史（建议 **15m 判、5m 复核；1m** 仅用于执行下放）。
* •	已确定三种“过程概率”构造思路（全部停留在 L2 层，避免过度升阶）：
  * **◦	A | σ-only**：低波动 = 更接近前验边界。
  * **◦	B | σ+衰减（默认）**：低波动 + 波动继续下行。
  * **◦	C | 轻量保险**：B + 一条 **|ICI| < θ_I** 的**单阈值**（不做导数/频域）。

⠀
# 2. 问题（我们要把哪件事说清）
1. 1	在历史数据里，**latent 事件前**的“概率三向量”到底长什么样（**形状与强度**）？
2. 2	给定任意当下时刻 *t*，我们能否用三维概率向量的**“形状偏差”**判断它是否“回到了历史的前夜样子”？
3. 3	把这个判断变成**可治理的基线与阈值**，上线后能稳定地给出“**进入前验边界**”的信号，并与 **ENF/实盘开关**耦合。

⠀
# 3. 方法与过程（怎么从原始数据走到可用信号）
### 3.1 特征与三种概率算子（负责：特征工程 + 算法）
**输入**：ohlcv_{tf}.parquet（timestamp, open, high, low, close, volume），lines_{tf}.parquet（timestamp, clarity, em, ici）
**从 OHLCV + ICI 得到的原子特征**
* **•	近端波动** (\sigma_t)（择一实现；窗口 Wσ=10 bars）：
  * *◦	Parkinson*：( \sigma_t = \sqrt{\frac{1}{4\ln 2} \cdot \operatorname{mean}_{Wσ}\big( \ln(\text{high}/\text{low})\big)^2 })
  * *◦	log-return stdev*：(\sigma_t = \operatorname{stdev}_{Wσ}(\Delta\ln \text{close}))
* **•	滚动分位秩**：(q_t = \operatorname{qrank}_T(\sigma_t) \in [0,1])（T=200）
* **•	概率代理（与波动反比）**：(p^{(\sigma)}*t = \operatorname{clip}\Big(\dfrac{q*{hi} - q_t}{q_{hi} - q_{lo}},,0,1\Big))（默认 q_lo=0.2, q_hi=0.7）
* **•	波动衰减率**：对 (\ln\sigma_t) 取 EMA(span=10) 后差分：(d_t = -\Delta,\operatorname{EMA}(\ln\sigma_t),; d_t>0\Rightarrow \sigma \text{ 在收敛})
* **•	轻量结构保险（可选）**：(1_{{|\mathrm{ICI}_t| < \theta_I}})（仅单阈，不取导数/频域；θ_I 取 ICI 的 0.7 分位）

⠀**统一的 "hazard → 过程概率" 形式（****(\tau\in{3,6,12})****************************************）**
* •	A：(\lambda_t = \kappa, \operatorname{ReLU}\big(p^{(\sigma)}_t - \theta\big))
* •	B（默认）：(\lambda_t = \kappa, \operatorname{ReLU}\big(\alpha d_t + (1-\alpha)p^{(\sigma)}_t - \theta\big),; \alpha=0.5)
* •	C：在 B 的基础上，若 (|\mathrm{ICI}_t| \ge \theta_I) 则 (\lambda_t \times= 0.5)（或置 0）
* **•	过程概率**（在线一步近似）：(\Pi_t(\tau) = 1 - e^{-\tau \lambda_t})
* •	三维概率向量：(\mathbf P_t = [p_A, p_B, p_C]^\top = [\Pi^{(A)}_t,\Pi^{(B)}_t,\Pi^{(C)}_t]^\top)

⠀**产物**：features/features_{tf}.parquet（timestamp, sigma, qrank, d, ici, mask_ici_opt）；probs/probs_{tf}_{method}.parquet（timestamp, tau, pA, pB, pC, lambdaA, lambdaB, lambdaC）

### 3.2 两遍回测（校准，而非预测；负责：算法/回测 + 数据工程）
**PASS-1：****知道未来****（刻“前夜模板/卡尺”）
目标**：在每个 latent 的触发点 (t_*) 前，从窗口 ([t_* - W_{pre}, t_* - 1])（W_pre=10）里挑出**“波动率达标”的最近时刻** (t)，收集当时的 (\mathbf P_t)。
* **•	波动达标口径**：用**分位带**（而非均值±nσ）
  * ◦	(\sigma_t \in [\underline q, \overline q])（如 [0.2, 0.7] 的滚动分位）或 (|z_{MAD}(\sigma_t)| \le n)（稳健 z）
* •	对每个事件：
  1. 1	找到最近的达标 (t)；
  2. 2	记录 (\mathbf P_t)；
  3. 3	记录与**上一个非 latent 事件结束时刻** (t_0) 的时距 (t_j = t - t_0)（若不存在则 NA）
* **•	聚合得到“模板”**：
  * ◦	形状均值/协方差：(\mu = \mathbb E[\mathbf P],; \Sigma = \operatorname{Cov}(\mathbf P))（**协方差建议用 Ledoit–Wolf 收缩**）
  * ◦	马氏距离：(D = \sqrt{(\mathbf P - \mu)^\top \Sigma^{-1} (\mathbf P - \mu)})；记录其分位阈值 (q_{80}, q_{90})
  * ◦	强度标尺：三向量**调和均值** (H = \mathrm{HM}(\mathbf P)) 的中位数 (\tilde H)
  * ◦	时间卡尺：(t_j) 的分位区间 ([q_{25}, q_{50}, q_{75}])

⠀这一步产出“**前夜椭球**”（(\mu,\Sigma)）+“**形状阈**”（(q_{80})）+“**强度阈**”（(\tilde H)）+“**时间卡尺**”（(t_j) 的分布）。
**输出**：templates/latent_pre_templates_{tf}.json（mu:[3], Sigma:[3x3], D_thr:{q80,q90}, H_med, tj_quantiles, asof）与 templates/pass1_events_aligned_{tf}.csv
**工程要点与踩坑**
* **•	先聚后算**：若 5m→15m，先聚合再算 σ/qrank/d/Π；
* **•	无未来函数**：滚动分位与 EMA 仅用历史；时间戳统一**右闭**；
* **•	协方差收缩**：样本少或相关高时必开收缩；
* **•	异常样本**：D 远超 q90 的事件前样本应单独复核（可能口径错误或异常行情）。

⠀
**PASS-2：****避免未来****（用模板量现在）
目标**：每个**非 latent 事件**结束后的时段，在“卡尺区间”里检测是否“回到前夜样子”。
* •	对每个非 latent 结束时刻 (t_0)，在 ([t_0 + q_{25}(t_j),, t_0 + q_{75}(t_j)]) 内逐 bar 计算 (\mathbf P_t)
* •	两道门：
  * **◦	形状门**：(D_t \le q_{80})（落入前夜椭球）
  * **◦	强度门**：(H_t \ge \tilde H)（防“三低但形似”）
* **•	连续**满足 **K 根**（K=3–5） ⇒ 标记“**进入前验边界**”
* •	评价与选择：覆盖率（latent 前命中比例）、平均提前量、误触率、**r–绩效曲线**（把标记段作为 ENF 剔除/降权后策略 KPI 的改善）

⠀**输出**：eval_pass2/pass2_online_eval_{tf}.csv 与 curves/r_perf_curve_{tf}.csv
**工程要点与踩坑**
* **•	K 连 + 最小段长**：避免抖动；
* **•	多 tf 一致性**：15m 判、5m 复核，误差 < 10%；
* **•	参数敏感度**：扫 τ∈{3,6,12}, D_thr∈{80,85,90}，曲线应平滑单调；
* **•	禁做/降权**的实际效果要和交易成本联动评估（次数、滑点、费率）。

⠀
### 3.3 离线基线与在线监测的统一（负责：算法/治理）
* **•	离线基线**：保存 (\mu,\Sigma,q_{80},\tilde H) 与 (t_j) 分布（**分月/分季**滚动重估，抗非平稳）。
* **•	在线监测**：实时计算 (\mathbf P_t)，得 (D_t, H_t)，执行“两道门 + K 连”的逻辑。
* **•	执行策略**（与 ENF 耦合）：进入前验边界 ⇒ **禁做/降权**；离开边界 ⇒ **恢复**。

⠀
# 4. 产物（文件/字段规范一口气齐）
* •	probs/pre_prob_timeseries_{tf}_{method}.parquet
  * ◦	列：timestamp, tau, pA, pB, pC, sigma, qrank, d, ici(opt)
* •	templates/latent_pre_templates_{tf}.json
  * ◦	字段：mu:[3], Sigma:[3x3], D_thr:{q80,q90}, H_med:float, tj_quantiles:{q25,q50,q75}, asof
* •	templates/pass1_events_aligned_{tf}.csv
  * ◦	列：event_id, t_star, t_pick, tau, pA, pB, pC, D, H, t_j
* •	eval_pass2/pass2_online_eval_{tf}.csv
  * ◦	列：nonlatent_id, t0, t, tau, pA, pB, pC, D, H, gate1, gate2, flag, lead
* •	curves/r_perf_curve_{tf}.csv
  * ◦	列：remove_ratio, sharpe, sortino, maxdd, kelly, hitrate
* •	governance/signature.json
  * ◦	字段：tf, sigma_method, W_sigma, T_qrank, q_lo, q_hi, ema_span, alpha, theta, kappa, tau_grid, K_consec, ici_threshold_opt, template_asof, recalib_days

⠀
# 5. 角色分工与详细需求（像产品经理一样拆给各部门）
### 5.1 数据工程（DE）
* **•	任务**：构建标准化输入、时间对齐、先聚后算、产物目录落地；完成 I/O 接口与 QC。
* **•	公式落地**：无；负责窗口函数与分位计算的正确实现与时间口径（右闭）。
* **•	关键点**：
  * ◦	聚合遵循 **先聚后算**；
  * ◦	缺失不前向填充（NaN 即 NaN）；
  * ◦	时间戳统一 UTC 右闭；
  * ◦	产物体积 < 50MB/文件；
  * ◦	signature.json 是**唯一口径**。

⠀5.2 特征工程（FE）
* **•	任务**：实现 (\sigma_t, q_t, p^{(\sigma)}*t, d_t, 1*{|ICI|<\theta_I})；输出 features_*.parquet。
* **•	公式**：详见 3.1；参数默认：Wσ=10, T=200, q_lo=0.2, q_hi=0.7, ema_span=10, θ_I=q70。
* **•	坑**：
  * ◦	“滚动分位”必须是**只用过去样本**；
  * ◦	EMA 不得跨日断档；
  * ◦	5m→15m 必须在 15m 上重算一遍特征。

⠀5.3 概率算子与回测（Algo/BT）
* **•	任务**：A/B/C 三法计算 (\lambda_t) 与 (\Pi_t(\tau))，生成 (\mathbf P_t)；实现 PASS-1/2；完成评估与曲线。
* **•	公式**：详见 3.1、3.2。
* **•	工程要点**：
  * ◦	马氏距离使用 **收缩协方差**（Ledoit–Wolf）；
  * ◦	PASS-1 仅用事件**前窗口**；
  * ◦	PASS-2 两道门 + **K 连**；
  * ◦	记录 **lead** 分布与 ENF r–绩效；
  * ◦	代码参数全部写入 signature.json 并在日志首行回显。

⠀5.4 治理与运维（Gov/Ops）
* **•	任务**：参数与模板的版本化、重估调度、漂移监控与报警、验收签署。
* **•	要求**：
  * ◦	templates/*.json 文件名含 tf+asof；
  * ◦	月/季重估；
  * ◦	监控 cond(Σ)、||μ_t-μ_{t-Δ}||、误触率；
  * ◦	任何改动必须更新 template_asof 与版本号。

⠀5.5 数据分析（DA）
* **•	任务**：读取模板与评估文件，出结论与报告。
* **•	怎么看数据**：
  1. **1	模板质检**：D 分布应集中；Σ 条件数适中；异常点单列；
  2. **2	在线可行性**：覆盖率↑、误触↓、平均提前量在 tj 区间；
  3. **3	r–绩效**：随着剔除比例 r 增，Sharpe/Sortino 上升或稳定，MaxDD 下降；
  4. **4	敏感度**：扫 τ, K, D_thr，找稳健区；
  5. **5	结论写法**：是否达到“可用 ENF 负向过滤”的门槛。
* **•	H0/H1**：
  * **◦	H0**：非 latent 时段的 (\mathbf P_t) 与模板（(\mu,\Sigma)）同分布（或其 D 分布无差）。
  * **◦	H1**：latent 前窗口的 (\mathbf P_t) 更接近模板（D 显著落入阈内），且 H 不低于 H_med。

⠀
# 6. 实际意义与后续耦合
* **•	意义**：把“**latent 的前夜**”变成**可度量的三向形状** + **强度卡尺** + **时间卡尺**，并且**无未来函数**就能在盘中识别；主要目标是**划定前边界**，不是方向预测。
* **•	耦合**：
  1. **1	ENF 接口**：进入边界 → mask=0（禁做/降权），最小段长来自 PASS-1 的 run-length；r–绩效曲线定剔除比例与阈值。
  2. **2	序列事件**：统计“前夜建立（K 连）→ latent 点火”的 lead 分布与可靠度；
  3. **3	执行层**：15m/5m 判定，下放到 5m/1m 执行；
  4. **4	再校准**：按月/季重估 (\mu,\Sigma, D_{thr}, H_{med}, t_j)；
  5. **5	扩展（非必需）**：若 σ 口径失效，再尝试 Copula 形状对齐；保持 L2 主体。

⠀
# 7. 最小实现清单（你现在就能开工）
1. 1	实现 A/B/C 三算子，输出三向概率时间序列（probs_*）。
2. 2	跑 PASS-1：刻模板（μ, Σ, D_{q80}, H_med, t_j 分布），落盘为 templates/*.json。
3. 3	跑 PASS-2：两道门 + K 连；输出覆盖率、提前量、误触率与 r–绩效曲线。
4. 4	在回测器中接 ENF 负向过滤开关；对比“接/不接”的策略 KPI。
5. 5	写入与更新 signature.json 与 *_templates.json（含 asof 与重估频率）。



# 8. 验收 Checklist（详细版）
   验收核心：**可复现、可解释、性能达标、治理合规**。    通过该清单，各团队可在独立责任范围内逐项完成验收。

###  8.1 模板与文件完整性（DE）
        负责人**：数据工程（Data Engineering）          目标**：确保所有模板与数据文件的完整性、命名规范、一致性与可追溯性。
* **•	模板文件存在**：检查 templates/*.json 是否生成；字段包含：mu, Sigma, D_thr, H_med, tj_quantiles, asof，无 NaN 或空值。
* **•	模板再现性**：同一输入与参数重跑一次，μ 与 Σ 差异 < 1e-6（可用 np.allclose 校验）。
* **•	文件体积控制**：所有 .csv 与 .parquet 文件体积 < 50 MB；如超限，必须拆分或压缩。
* **•	目录结构规范**：backtest/tf=15m/ 下应包含子目录：features/, probs/, templates/, eval_pass2/, curves/, governance/。
* **•	命名一致性**：文件命名需含时间分辨率与日期，例如：latent_pre_templates_15m_2025-10-31.json。
* **•	治理文件完整**：signature.json 必须存在且字段齐全；template_asof 日期与模板文件一致。
* **•	时间戳一致性**：所有时间字段（timestamp, t0, t*）均为 UTC，右闭口径，序列连续无重叠。

⠀
### 8.2 矩阵稳定性（FE/Algo）
       负责人**：特征工程（Feature Engineering） + 算法工程（Algorithm）         目标**：保证协方差矩阵 Σ 与均值 μ 的数值稳定性与可解释性。
* **•	条件数限制**：cond(Σ) < 1000；若超过，必须启用 **Ledoit–Wolf 收缩**重新计算。
* **•	收缩有效性**：收缩前后相对差 < 10%：\ $$ $$
* **•	矩阵正定性**：Σ 必须为正定矩阵（所有特征值 > 0），否则强制添加正则项 εI（ε=1e-6）。
* **•	均值漂移监控**：相邻两期模板均值差 ||μ_t - μ_{t-Δ}||₂ < 0.05；若超过，触发“再标定”任务。
* **•	样本充足性**：用于计算 Σ 的样本数量 ≥ 50；若不足，合并相邻时段或扩大窗口。
* **•	单位检查**：Σ 维度 = 3×3；μ 长度 = 3，对应 [pA, pB, pC]。
* **•	异常检测**：若 Σ 的某元素 > 1 或 < -1，视为异常相关性，需重新取样。

⠀
### 8.3 回测性能（Algo/BT）
**	负责人**：回测工程（Backtest Team） **	目标**：验证模型在历史数据上的有效性与稳健性。
* **•	覆盖率**：latent 事件前被成功检测比例 ≥ 70%。
* **•	误触率**：非 latent 时段被误判为前夜区的比例 ≤ 15%。
* **•	平均提前量**：检测提前时间位于 [q25(tj), q75(tj)] 区间内。
* **•	跨时间框架一致性**：15 m 与 5 m 的提前量误差 < 10%。
* **•	参数敏感性测试**：扫描 τ∈{3,6,12}, D_thr∈{80,85,90}, K∈{3,5}；性能曲线应平滑单调。
* **•	r–绩效指标**：输出 r_perf_curve_{tf}.csv，保证随着剔除比例 r 增加： Sharpe ↑、Sortino ↑、MaxDD ↓、Kelly ↑。
* **•	重复实验一致性**：相同配置重跑结果波动 < ±5%。

⠀
### 8.4 策略与 ENF（Strategy/Exec）
**	负责人**：策略与执行团队 **	目标**：确认 ENF 负向过滤逻辑生效且对策略绩效产生正向改善。
* **•	绩效改善**：
  * ◦	Sharpe ↑ ≥ 10%
  * ◦	Sortino ↑ ≥ 10%
  * ◦	MaxDD ↓ ≥ 10%
  * ◦	Kelly ↑ ≥ 5%
* **•	交易频率**：禁做段期间交易次数下降 ≥ 20%；执行层订单活跃度与成本下降。
* **•	禁做段一致性**：禁做段持续时间与模板 tj_quantiles 一致；偏差 ≤ 10%。
* **•	边界切换平滑性**：进入与退出前验边界无高频震荡；连续触发次数占比 ≤ 5%。
* **•	策略报告输出**：生成 strategy_eval_report.md，包含前后对比表格与图表。

⠀
### 8.5 治理与日志（Gov/Ops）
**	负责人**：系统治理与运维（Governance / Ops） **	目标**：确保模型可追溯、可监控、可恢复。
* **•	重估调度**：月/季自动重估模板；日志记录在 governance/logs/recalib_*.log。
* **•	日志内容**：包含 start_time, end_time, cond(Σ), μ_drift, recalib_status, warnings。
* **•	报警机制**：若条件数、均值漂移或误触率超阈，系统自动触发邮件/消息警报。
* **•	版本签名**：每次更新 templates/*.json 时生成 SHA256 hash；写入 signature.json。
* **•	治理合规**：修改模板或参数必须更新 template_asof 与 version_id。
* **•	容错与恢复**：出现异常模板可回滚至上次稳定版本（通过 hash 快照恢复）。

⠀
### 8.6 数据分析结论（DA）
**	负责人**：数据分析师（Data Analyst） **	目标**：验证统计假设、总结结果并撰写报告。
* **•	分布检验**：D_t 分布应单峰集中，峰值位于 0–q80 区间；异常值 < 5%。
* **•	显著性验证**：KS 或 energy distance 检验，p < 0.05 拒绝 H₀；证明前夜期 Pₜ 更接近模板。
* **•	假设声明**：
  * **◦	H₀**：非 latent 时段 Pₜ 与模板 (μ, Σ) 同分布；
  * **◦	H₁**：latent 前窗口 Pₜ 更接近模板（D 下降，H ≥ H_med）。
* **•	可视化要求**：绘制 D 分布、提前量分布、r–绩效曲线、三维 P 散点椭球图。
* **•	报告输出**：《前验边界模板校准报告》，包括 图表 + 结论 + 阈值敏感性表；存入 backtest/reports/ 目录。

⠀
# 9. 附：一页速览（供同学查阅）
* **•	目的**：划定 latent 前边界 → ENF 负向过滤。
* **•	输入**：OHLCV + ICI（单阈）。
* **•	核心**：三向概率向量 (\mathbf P_t)；模板（(\mu,\Sigma)）+ 两道门（D≤q80 & H≥H_med）+ K 连。
* **•	两遍**：PASS-1（刻模板/卡尺）、PASS-2（无未来函数评估）。
* **•	产物**：features_* probs_* templates_* eval_pass2_* curves_* signature.json。
* **•	验收**：可复现、性能达标、治理合规。

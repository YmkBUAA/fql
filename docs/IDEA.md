# FQL-AR 的核心想法：Within-Mode Baseline for Multi-Modal BC

> 这份文档是 fql_ar 的"研究 narrative"，回答三个问题：为什么要做、想解决什么具体的失败模式、怎么做。
> 实现细节（具体超参、诊断指标、phase-1 验证流程）见 [`fql_ar_design.md`](fql_ar_design.md)。

---

## 1. 背景

### 1.1 Advantage-weighted BC 是 offline / offline-to-online RL 的主流框架

从 AWR (Peng 2019) 到 IQL (Kostrikov 2021) 再到扩散策略时代的 IDQL / QGPO / EDP，offline 与 offline-to-online RL 的主流做法都是 **advantage-weighted behavior cloning**：

```
L_BC = E_{(s,a) ~ data} [ w(s, a) · || policy(s) - a ||^2 ]
w(s, a) = f( Q(s, a) - b(s) )
```

其中 `b(s)` 是某种状态级基线，`f` 是单调非减的权函数（典型选择 `exp(·/β)`、indicator 1[·>0]、softmax）。这个框架的设计意图是：**用 Q 当 gate，让 BC 多学好动作、少学差动作**，避免 plain BC 上数据中的次优样本污染策略。

### 1.2 主流基线全部是"跨模平均"

不同方法在 `b(s)` 的具体构造上有所差异，但**结构是一样的**：

| 方法 | b(s) 的构造 |
|---|---|
| AWR / AWAC | V(s) 用 MC 回归得到 |
| IQL | V(s) 用 Q 的 expectile 隐式估计 |
| CRR (mean) | b(s) = E_{a' ~ π}[Q(s, a')] —— K 条独立采样均值 |
| fql_v | b(s) = E_{a' ~ BC_flow}[Q(s, a')] —— K 条 BC 流采样均值 |

这些构造**共同的隐含假设**是：策略 π（或 BC 流）在状态 s 下的动作分布是单模的，或者多模之间的 Q 值大致相等 —— 这样"跨模平均"才是有意义的"典型 Q 值"。

### 1.3 但真实数据是多模的

真实任务的演示数据，模式数 ≥ 2 是常态：

- **PushT** (Diffusion Policy 论文的招牌环境)：从左推 vs 从右推，是两个有效模式。
- **Robomimic multi-human**：不同人的演示构成多模。
- **OGBench cube-double**：抓哪个 cube、从哪个方向接近，构成多模。
- **D4RL antmaze-diverse**：多条到达目标的路径。

正因为多模性常见，**扩散 / 流策略**（Diffusion Policy, FQL, IDQL, ...）才在 BC 端取代了高斯策略 —— 它们能拟合多模 BC 分布。但**优势加权这一端的基线构造，仍然停留在跨模平均的范式**，没有跟上策略族的更新。

---

## 2. 动机

### 2.1 跨模平均基线在多模数据上有一个具体的失败模式：mode collapse

考虑一个状态 s，BC 数据有两个模式 A、B，比例 50:50。设：

- `q_A := E[Q(s, a) | a ∈ mode A]`
- `q_B := E[Q(s, a) | a ∈ mode B]`
- 不失一般性 `q_A < q_B`（mode B 整体上"更好"）

**fql_v 风格的 V 基线**：
```
V(s) = E_{a' ~ BC_flow}[Q(s, a')] ≈ 0.5·q_A + 0.5·q_B = mean(q_A, q_B)
```

考虑数据中一个 **mode A 内部的优秀样本** a：`Q(s, a) = q_A + ε`，其中 ε > 0 是 mode A 内部的真实改进。

```
Δ_V(s, a) = Q(s, a) - V(s)
          = q_A + ε - 0.5(q_A + q_B)
          = ε + 0.5(q_A - q_B)
```

如果 `q_A - q_B` 的差距大于 mode 内部的方差 ε，**则 Δ_V < 0**：即便 a 是它所在模式内的优秀样本，跨模均值基线仍然会给它一个负的优势信号 → BC 重加权会**下权 mode A 的所有动作**，无论它们在 mode A 内部多好。

随着训练推进：
1. mode A 的所有动作被系统性下权 → BC 流逐渐忘记 mode A
2. BC 流的采样越来越偏向 mode B → V(s) 进一步漂向 q_B
3. mode A 内残留的好动作的 Δ_V 进一步变得更负 → 加速塌缩
4. 收敛时 BC 流坍缩为单模（mode B），多模性永久丢失

**这就是跨模均值基线的 mode collapse 失败模式**，是一个由"baseline 与 a 的模归属不匹配"驱动的正反馈循环。

### 2.2 PushT 思想实验

具体化上面的失败模式：在 PushT 上，假设演示数据里"从左推"和"从右推"各占一半，但执行层面"从左推"略略容易成功（q_left > q_right）。

- 用 V baseline：右推数据的 Δ_V = q_right − mean(q_left, q_right) < 0 → 右推被下权 → BC 流逐渐只学左推 → 多模 BC 退化为单模 BC → 失去对环境分布漂移、对手行为变化、初始位姿扰动等的鲁棒性。
- 即便 q_left ≈ q_right，BC 流采样的随机噪声也会让某一模在 buffer 中暂时过表达，触发同样的循环。

**关键观察**：跨模均值基线**告诉你的是"a 比一个随机 BC 采样好不好"**。如果你想要的是"**a 在它自己的 mode 内是不是好动作**"，跨模均值就是错的目标函数。

### 2.3 现有方法的局限

如果想避开 mode collapse，需要一个**within-mode** 的基线 —— 与 a 在同一个 mode 内的"典型 Q 值"。检查现有路线：

| 路线 | 基线对 a 的依赖 | 多模数据上的行为 |
|---|---|---|
| AWR / IQL / fql_v | 跨模平均 | mode collapse |
| CRR (K-mean) | K 条独立采样 → 仍跨模 | mode collapse |
| RLOO / GRPO | batch 内其它独立样本 | 跨模（取决于 batch 模式分布） |
| Q-Prop | b(s,a) = Q(s,π(s)) + ∇Q·(a−π(s)) | 围绕单模 π(s) 一阶展开，不能保 mode |
| **fql_pi**（确定性策略 a' = π(s)） | 单模 π | 主动下权所有非 π-模式的动作 → mode collapse 加速 |

**所有这些方法的 a' 要么与 a 独立（跨模），要么由当前学习策略的 mode 决定（单模 π 主导）**。没有一种构造能保证"a' 落在 a 所在的模式内"。

---

## 3. 做法

### 3.1 关键观察：流模型的部分去噪天然落回 a 所在的模式

考虑流模型的 CFM 训练路径：`x_t = (1-t)·ε + t·a`，`v_θ` 拟合 `a − ε`。一旦 `actor_bc_flow` 训得收敛，从 `x_t` 出发沿 `v_θ` 的 ODE 积分到 `t=1`，会**收敛到离 x_t 最近的 BC 模**（流的几何性质 —— 多模 BC 下 ODE 倾向于走向最近 attractor）。

记这个算子为：
```
denoise(s, x_t, t)  :=  ODE_Euler( actor_bc_flow ; from x_t at time t to t=1 )
```

那么**通过 a 自身的扰动构造 a'**：
```
ε ~ N(0, I), t ~ U(t_lo, t_hi)
x_t = (1-t)·ε + t·a            # 沿 CFM 训练路径的"加噪"
a'  = denoise(s, x_t, t)        # 流自身的"去噪还原"
```

由 ODE 的连续性 + Lipschitz `v_θ`：
- t 越大，`x_t` 离 a 越近，a' 越靠近 a（极限 t→1：`a' = a`，无信号）
- t 越小，`x_t` 越接近纯噪声，a' 越倾向于"任意 BC 模式的代表"（极限 t→0：a' = 全新 BC 采样，等价于 fql_v K=1）
- t ∈ (0, 1) 中段：**a' 大概率落在 a 所在的 mode 内**（因为 ODE 走向最近 attractor），构成 within-mode baseline

**这是流策略族独有的几何性质**：高斯/确定性/混合策略都没有"沿训练路径的可微逆向操作"，做不出连续 t 旋钮的局部还原。

### 3.2 算法

每个训练步：

```
# Step 1: 标准 FQL critic 损失
critic_loss = MSE( Q(s, a), r + γ·mask·Q_target(s', sample_actions(s')) )

# Step 2: 构造 within-mode advantage
ε ~ N(0, I)                                   # (B, action_dim)
t ~ U(adv_t_lo, adv_t_hi)                     # (B, 1), 默认 [0.4, 0.7]
x_t = (1-t)·ε + t·a
a'  = ODE_Euler( actor_bc_flow ; x_t → t=1, n_steps = adv_flow_steps )
a'  = stop_grad( clip(a', -1, 1) )
Δ   = stop_grad( Q_target(s, a) - Q_target(s, a') )

# Step 3: MAD 归一化 + ESS-targeted softmax
d_med   = median(Δ)
β       = 1.4826 · MAD(Δ)
d_norm  = (Δ - d_med) / max(β, 1e-6)
τ*      = bisect(log_τ s.t. ESS(d_norm / τ) = ess_target)
w_exp   = softmax(d_norm / τ*) · B            # 均值 1 的权重

# Step 4: critic R^2 可靠性 gate (offline 阶段关掉重加权)
r2          = 1 - critic_loss_ema / var_target_q_ema
gate_critic = sigmoid( (r2 - r2_target) / κ )
gate_c      = gate_critic · (online OR not weighted_bc_online_only)
bc_weights  = stop_grad( 1 + gate_c · (w_exp - 1) )

# Step 5: 重加权 BC 流损失（CFM 路径独立采样）
ε_k ~ N(0, I), t_k ~ U(0, 1)                  # 与 Step 2 的 (ε, t) 独立
target_k    = a - ε_k
loss_k      = || actor_bc_flow(s, (1-t_k)·ε_k + t_k·a, t_k) - target_k ||^2
bc_flow_loss = mean( bc_weights · mean_k(loss_k) )

# Step 6: FQL 蒸馏 + Q 损失（不变）
distill_loss = MSE( actor_onestep_flow(s, ε), full_BC_flow(s, ε) )
q_loss       = -mean( Q(s, clip(actor_onestep_flow(s, ε))) )

# Step 7: 总损失 + 软目标更新 + EMA 更新 critic gate 统计
loss = critic_loss + bc_flow_loss + α·distill_loss + q_loss
```

### 3.3 关键设计决策

#### 为什么 (ε, t) 在 Step 2 和 Step 5 独立采样？
因为 Step 2 决定**这个样本拿多大的权重**，Step 5 决定**这个样本贡献的梯度方向**。共享 (ε, t) 会让"权重大小"和"梯度方向"在同一个噪声向量上耦合，引入有偏的优化信号。独立采样切断这个耦合。

#### 为什么 t ∈ [0.4, 0.7] 而不是更小或更大？
- t < 0.3：x_t 几乎是 a，ODE 步长很小、a' ≈ a，Δ → 0，无信号。
- t > 0.8：x_t 接近纯噪声，a' 退化为 fql_v K=1（跨模采样），失去 within-mode 性质。
- [0.4, 0.7] 是经验上"扰动够大产生 Δ、但还没大到逃出 a 的 mode"的中段。t 是连续旋钮，可以 t-sweep 调出 sweet spot。

#### 为什么用 `target_critic` 而不是 `critic` 算 Δ？
TD 引导的 Q 在线更新中波动较大，target_critic 提供一个**慢变的、稳定的**基线。Δ 用 stop_grad 切断梯度回传，不影响 Q 训练。

#### 为什么需要 `gate_critic`？
offline 早期 critic 还没拟合好，r² 接近 0 甚至为负，此时 Δ 的信号是 Q 网络初始化噪声，不是真正的 advantage。`gate_critic` 用 R² 作为可靠性指标，在 critic 收敛之前把 `bc_weights` 钉在 1（等价于 plain BC）。

#### 为什么 `weighted_bc_online_only=True`？
offline 阶段的目标是把 BC 流忠实拟合到数据；mode collapse 主要发生在 online 阶段（best-of-n 让 buffer 变得不平衡）。所以默认只在 online 启用重加权，offline 阶段跑 plain FQL。这也是一个安全设置 —— 即便假设错误，offline 性能不会差于 plain FQL。

### 3.4 与替代方案的对比

| 方法 | a' 构造 | a 依赖 | within-mode? | offline 安全 |
|---|---|---|---|---|
| **fql_ar (本方法)** | `denoise(noise(a))` 多步流局部还原 | ✅ | ✅ | ✅ on BC manifold |
| fql_v | E[Q(s, a' ~ BC)] K 条流采样均值 | ❌ | ❌ 跨模均值 | ✅ on BC manifold |
| **fql_pi (ablation)** | `onestep_flow(s, ε)` 单步蒸馏头 | ❌ | ❌ 单模 π | ❌ π 可能 OOD |
| AWR / IQL | V(s) 状态级估计 | ❌ | ❌ 跨模 | ✅ |
| CRR mean baseline | E[Q(s, a' ~ π)] | ❌ | ❌ 跨模 | 取决于 π |
| Q-Prop | Q(s, μ) + ∇Q·(a-μ) Taylor 展开 | 弱 | ❌ 单模 | ❌ 看 b 网络是否 OOD |
| RLOO / GRPO | batch 内其它样本均值 | 弱 | ❌ 跨模 | – |

**fql_ar 在这张表里唯一全打勾的位置**：sample-anchored、within-mode、offline-safe。这三条同时成立来自一个事实 —— 流策略的 ODE 反向是策略**自身**提供的、确定性的、可微的、与 a 强相关的算子。换任何其它策略族都拿不到。

### 3.5 退化情形：t 边界两端的语义

| t 设置 | 退化为 | 失败模式 |
|---|---|---|
| t = 0 | fql_v K=1 | a' 是新 BC 采样，跨模平均，失去 within-mode 性质 |
| t = 1 | a' = a | Δ ≡ 0，无信号，等价于 plain BC |
| t ∈ (0, 1) | fql_ar | 真正的 within-mode advantage |

**这是 fql_ar 论文的"signature t-sweep curve"**：success(t) 应当在两端低、中段凸起。这条曲线是 fql_ar 主张"中段是 sweet spot"的最直接经验验证。

### 3.6 经验签名（offline-safe 与 mode preservation 的可观测证据）

如果 fql_ar 的机制按设计运行，应当观察到下面 5 条签名。每条标了 cube-double-noisy（sd000、K=1、ess=0.7）上的实测状态。完整图见 [`/visualization/figures/`](../visualization/figures/)：

| # | 签名 | 预测 | cube-double-noisy 实测 | 状态 |
|---|---|---|---|---|
| 1 | `flowar/dist_a_aprime_p50` | ∈ [0.10, 0.40] | ~0.40 | ✅ 命中目标带 |
| 2 | `flowar/delta_std`（online） | > 0.5 | 1.0+ | ✅ 信号有 spread |
| 3 | `flowar/frac_a_better`（online） | ≈ 0.5（within-mode 预测） | 0.47 | ✅ 与 v2 一致、与 v1 (>0.6) 反向 |
| 4 | `flowar/delta_mean`（online） | ≈ 0（within-mode 预测） | -0.05 ~ -0.12 | ✅ 与 v2 一致、与 v1 (>0) 反向 |
| 5 | 训练后 BC 流仍多模 + fql_v 在 q_A≠q_B 时塌缩 | fql_ar 保模、fql_v 塌 | **fql_ar 保模**（√），**fql_v 也保模**（在此环境）| ⚠️ 见 §4.5 |

签名 (1)–(4) 来自训练时实时统计（[`plot_train_diagnostics.py`](../visualization/plot_train_diagnostics.py)）。**签名 (3) 和 (4) 与 v1 "best-of-n 在线吸收" 的预测（Δ 系统性 > 0、frac_a_better > 0.6）正好反号** —— 这是 v2 within-mode framework 击败 v1 framework 的核心实证。

签名 (5) 通过 PCA-2D 散点检查（[`plot_action_distribution.py`](../visualization/plot_action_distribution.py)）。在 cube-double-noisy 上 fql_v K=4 与 fql_ar K=1 的最终 BC 分布**几乎一致**，**两者都保住了多模性**。这不反驳 v2 framing 但说明 cube-double-noisy 不是 mode collapse 的触发场（详见下一节）。

---

## 4. 与 v1 设计的 framing 差异（修订记录）

| 维度 | v1 ([fql_ar_design.md](fql_ar_design.md)) | v2 (本文档) |
|---|---|---|
| 核心机制 | best-of-n online absorption：好动作进 buffer → Δ 系统性 > 0 → 上权 | within-mode baseline：a' 落回 a 的 mode → 跨模平均的 mode collapse 被避免 |
| 主要预测 | online 阶段 `delta_mean > 0`、`frac_a_better > 0.6` | online 阶段 `delta_mean ≈ 0`、`frac_a_better ≈ 0.5` |
| 与现有数据 | 不一致（实测 `delta_mean < 0`、`frac_a_better ≈ 0.47`）| **一致** |
| 主要对手 | 模糊（vs 通用 advantage-weighted BC） | 精确（vs 跨模平均基线在多模数据上的 mode collapse） |
| 主要 motivating example | cube-double-noisy | 2D bimodal toy + PushT-asymmetric（cube-double 降为补充证据，见 §4.5） |
| 核心实验 | 在 cube-double-noisy 上 lift-off 比 fql_v 早 200k+ | 在 mode-asymmetric 多模 benchmark 上 fql_v 出现可观测 mode collapse、fql_ar 保住多模（在 cube-double-noisy 上两者持平、不塌缩 —— 见 §4.5） |
| 控制变量论 (variance reduction) | 主线 | 副线（写入附录 / "additional theoretical hook"） |

v1 的"在线吸收"框架在工程上仍然描述了一个真实存在的次级效应（在线阶段 buffer 渐进改善），但**它不是 fql_ar 的主因**。v2 的 within-mode 框架是**与数据契合**的主因。

---

## 4.5 一个实测发现 + 故事进一步收紧

跑完 [`/visualization`](../visualization/) 三个图后浮出两个非平凡的实测事实，需要写进文档避免后续返工。

### 发现 A：cube-double-noisy task1/2/3 共享同一份 demonstration 数据

PCA-2D 散点图（[`figures/1_bc_modes_by_task.png`](../visualization/figures/1_bc_modes_by_task.png)）显示三个 task 在 offline-end 的 BC 流分布**完全一致**（per-state PCA std 精确相等到 3 位小数）。这是 OGBench singletask 变体的设计：**只换 reward / goal 定义，不换数据**。

→ **故事修正**：原本的"task2/3 比 task1 多模性更强 → 难学"假设是错的。三个 task 看到的是**同一份多模 BC 数据**。fql 在 task1 能解、task2/3 不能解的差异**完全来自 reward 把模式区分开**：
- task1：两个 cube 的抓法在该任务下都成立 → q_A ≈ q_B
- task2/3：reward 偏向特定操作流程 → q_A ≠ q_B

**这反而把 v2 framing 收紧了**：现在的命题不是"多模数据本身有问题"，而是"**同一份多模数据上，reward 让 mode 间 Q 不对称时，跨模均值基线无法区分'mode 内的好动作'和'mode 间的好 mode'**"。这是更精确、更可证伪的命题。

### 发现 B：cube-double-noisy 不触发 mode collapse

[`figures/2_mode_preservation_task{2,3}.png`](../visualization/figures/) 显示 fql_v K=4 与 fql_ar K=1 在 task2/3 终态 BC 分布**视觉上无差**，per-state PCA std 平均差距 < 10%。两者都保住多模性。两者 success 都 ≈ 0.99。

为什么没塌：
- **K=4 的 V 估计稳定**：4 条 BC 采样均值，cross-mode 平均的方差被 4× 压住，V 不会被一次性偏向某 mode
- **q_A − q_B 在 cube-double-noisy 上不够大**：reward 区分了模式（plain fql 失败说明 q_A ≠ q_B），但**差距没大到压垮 V baseline 的稳态平衡**

这条**不反驳 v2 framing**，但**反驳 cube-double-noisy 是 v2 的合适验证场**。要观察 mode collapse，需要 stress test：
- **PushT-asymmetric**：人为扩大 q_A − q_B（封一侧、加摩擦）
- **fql_v K=1**（去掉 K=4 的方差缓冲）
- **2D bimodal toy**：可调 asymmetry 旋钮 + 分布演化可视化

### 当前数据能撑起的命题（诚实版）

| 命题 | 证据强度 | 来源 |
|---|---|---|
| BC 流在 cube-double-noisy 上确实是多模的 | ✅ 强 | viz 1 panel scatter |
| within-mode framework 与训练动力学吻合（v1 反号） | ✅ 强 | viz 3 (`frac_a_better`、`delta_mean`） |
| fql_ar 在 cube-double-noisy task2/3 上 ≈ fql_v K=4 ≫ fql | ✅ 强 | [`exp/fql/Debug`](../exp/fql/Debug) eval.csv |
| fql_ar K=1 用更少 CFM 采样达到 fql_v K=4 性能 | ✅ 强 | 同上 |
| **fql_v 在多模数据上会 mode-collapse、fql_ar 不会** | ❌ 未观测 | 需新环境 |
| fql_ar > fql_v 在某类 multi-modal benchmark 上 | ❌ 未观测 | 需新环境 |

**结论**：v2 framing 在 mechanism 层（训练诊断）已被 cube-double-noisy 现有数据**部分支持**；在 capability 层（mode collapse 是真失败）**仍然需要新环境验证**。论文 framing 现在不能写"我们防止 mode collapse"，应当写"**我们的训练诊断与 within-mode 比较自洽（mechanism）；在 mode 不对称环境上 fql_v 出现塌缩、fql_ar 保住模（待补 capability 实验）**"。

---

## 5. 直接的实验路线图

按 v2 framing 和 §4.5 实测发现重新优先化。**先做能 stress-test mode collapse 假设的实验，再做 benchmark 数字**。

### Tier 1：mode collapse 直接证据（论文 Figure 1 + 主线立论）

1. **2D bimodal toy benchmark**（自制）—— 1D 状态 + 1D/2D 动作 + 双模 BC 数据，asymmetry `q_A − q_B` 是连续旋钮。
   - 跑 fql、fql_v K∈{1,4}、fql_ar、fql_pi
   - 出图：BC 流分布随训练的演化热图；asymmetry-vs-collapse-time 曲线
   - **预期**：fql_v K=1 在 asymmetry 大时显著塌缩；K=4 半塌；fql_ar 全程保模
   - 成本：训练 < 5 分钟/run，1 周内全套出图

2. **PushT-asymmetric**（自制 PushT 变体）—— 标准 PushT 上单侧加阻挡 / 摩擦 / 缩小到达区，使 q_left ≠ q_right。
   - 训练：原始 PushT（双模均衡）；测试：standard PushT + asymmetric variant
   - **mode preservation 测试**：训完 fql_v / fql_ar，看 BC 流采样是否仍双模
   - **robustness 测试**：训练时双侧通；测试时**封死 fql_v 学到的优势侧**，看是否能 fallback。这是 mode preservation **功能价值**的最强证据
   - 成本：PushT 复现 + 改 1 处环境参数；2-3 周

### Tier 2：消融（mechanism 层证据）

3. **negative ablation: fql_pi**（已有 [`agents/fql_pi.py`](../agents/fql_pi.py)）—— 证明 policy-anchored 基线塌掉。
4. **degenerate ablation: fql_ar @ t=0**（已支持，[main.py](../main.py) 命令行调 `adv_t_lo=adv_t_hi=0` + `adv_flow_steps=10`）—— 证明退化到 fql_v K=1 后失去 within-mode 性质。
5. **t-sweep "signature curve"**：t ∈ {0.0, 0.2, 0.4, 0.55, 0.7, 0.9, 1.0}，每点 3 seeds。验证两端低、中段凸起。

### Tier 3：标准 benchmark（capability 数字）

6. **OGBench cube-double / cube-triple / scene / puzzle**：3 seeds 补齐当前 sd000 数据。fql_ar ≈ fql_v K=4 ≫ fql 的故事用 compute 优势包装（K=1 vs K=4）。
7. **Robomimic Multi-Human**（square / transport）：标准多模基准；6 个操作员构成自然 q-asymmetry。
8. **D4RL antmaze-diverse**：选填，作为 online absorption 副线证据。
9. **不做 D4RL locomotion**：单模、不对 v2 framing 有帮助。

### Tier 4：理论与副线

10. **mode preservation 形式化定理**：在某个 BC 拟合假设下证明 fql_ar 的 BC 重加权梯度不诱导 mode collapse；fql_v 在 q 模式不等时诱导 mode collapse。
11. **控制变量方差缩减界**：v1 的 "Theoretical hook"，作为 additional contribution 写附录。
12. **compute 效率对比**：fql_ar K=1 vs fql_v K=4 wall-clock + 显存。

### 关键改动 vs v2.0 路线图

- Tier 1 把 **toy + PushT-asymmetric** 提到首位（v2.0 这两个在"加分"项）
- **OGBench cube-double 从主验证降到补充证据**（v2.0 把它当主基准，§4.5 揭示它不触发 collapse）
- 增加 robustness test（PushT 封侧 fallback）作为"mode preservation 功能价值"的证据
- 删除"在 cube-double-noisy 上 fql_ar 比 fql_v 早 200k lift-off"这条命题（实测两者持平）

---

## 6. 一句话总括

> **多模数据上的 advantage-weighted BC 需要的是 within-mode 比较，不是跨模平均。流策略的 ODE 反向是目前已知唯一一类能做出 within-mode、sample-anchored、offline-safe 三者兼得的基线的策略族。fql_ar 把这件事写出来。**

---

## 修订日志

- **v1**（[fql_ar_design.md](fql_ar_design.md)）：best-of-n online absorption framing，预测 `delta_mean > 0`、`frac_a_better > 0.6`。
- **v2.0**（IDEA.md 初版）：换为 within-mode baseline framing，预测 `delta_mean ≈ 0`、`frac_a_better ≈ 0.5`。cube-double-noisy 列为主验证场，假设 task1/2/3 多模性不同。
- **v2.5**（本版）：实测发现 cube-double-noisy task1/2/3 共享数据 → 故事收紧为"同数据 + reward 不对称"；实测 fql_v 在 cube-double-noisy 上**未塌缩** → cube-double-noisy 降级为补充证据，主验证场转向 toy bimodal + PushT-asymmetric。viz 3 训练诊断仍干净支持 within-mode framework；viz 1/2 BC 流分布只支持"多模存在"，不支持"fql_v 塌缩"。

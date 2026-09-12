# 老师也可能一本正经地教错：TGOPD 给 on-policy 蒸馏加了一道"先验证再教"的门

你有没有想过一个问题：蒸馏的时候，如果老师（teacher）在某道题上其实根本不会做，但偏偏答得自信满满，会发生什么？

在 on-policy distillation（OPD）的训练里，这不是假设，是常态。学生模型（student）自己采样 rollouts，冻结的 teacher 在每个 token 上给出 reverse KL 的稠密监督信号——这套方法让 student 达到 teacher 水平的速度比纯 RLVR 快差不多一个数量级。问题在于，reverse KL 是 mode-seeking 的：teacher 越是自信地把概率集中在错误答案上，student 收到的梯度就越强烈地指向这个错误。

上周看到的这篇 arXiv 2609.02998 论文给了一个我觉得"对，就该这么做"的解法：**先验证，再蒸馏**。

**核心摘要**：这篇论文发现 Vanilla OPD 对所有 prompt 无条件接纳 teacher 监督，但 teacher 的 confidence 根本无法区分它到底是"自信地对"还是"自信地错"——在 code 领域用 confidence 做可靠性判断的 AUROC 只有 0.51，接近瞎猜。作者提出 TGOPD（Teacher-Gated On-Policy Distillation）：用 teacher 自己跑 K=3 条 probe rollouts，交给 verifier 打分，三取二通过才放行稠密蒸馏，否则退回 verifier-grounded 的 GRPO。效果上，TGOPD 在 4B 和 35B 两个规模、数学/代码/指令遵循六个单域设置中全部超过 Vanilla OPD，多域训练下七个 benchmark 平均分分别提升 1.14 和 0.95 个点。顺手还解决了一个工程痛点：异步 OPD 中 teacher 节点 GPU 利用率常年趴在 9.8% 的位置，probe 计算恰好填进闲置窗口，利用率直接拉到 78.9%。我的判断：这不是底层理论突破，但抓的痛点非常真实，且"可靠性估计吃闲置算力"这个设计漂亮得让人拍大腿。

---

**论文信息**

- 标题：Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation
- 作者：Zhiwei Zhang, Zechen Sun, Fei Zhao, Kang Peng, Bin Liang, Huayu Deng, Yao Hu, Kam-Fai Wong, Mu Chuan（AllSpark Team）
- 链接：https://arxiv.org/abs/2609.02998 （提交于 2026 年 9 月 2 日，17 页，6 图 7 表）

---

## 🎯 问题：蒸馏流水线上的两个"房间里的大象"

先说一个工程上很扎心的事实。典型的异步 OPD 部署里，student 在专门的 inference 节点上自回归生成 rollouts，teacher 在独立节点上对已生成的 tokens 做一次 forward 打分。打分这个活儿比生成便宜太多了，而且必须等 student 的 batch 准备好才能开工——于是 teacher 节点的大部分时间都在空转。

空转到什么程度？论文用 4B 单域 run 的实测数据说话：

![compute_waste_square_v2.png](https://www.mulanai.com/fs/files/0912_55fe0e9b_compute_.png)

*图1：teacher 节点的 GPU 利用率时间序列（上）与平均值对比（下）。橙色是标准 OPD——teacher 节点平均利用率只有 10%，59% 的时间低于 5%，曲线呈现典型的"短暂尖峰+长时间归零"的 bursty 形态。蓝色是 TGOPD，probe 计算填进尖峰之间的空隙后，利用率拉到 79%，集群平均利用率从 51% 提到 69%。*

说实话看到这个图我第一反应是心疼——一整台 8-GPU 节点，90% 的时间在烧钱干等。

第二个问题更要命：teacher 的可靠性是因 prompt 而异的，但 Vanilla OPD 根本不看这个。作者做了个诊断实验——每个领域抽 2400 个 prompts，每题让 teacher 离线采 10 条回答，用 verifier 算出细粒度 pass rate（记为 $q_T^{(10)}$），然后看 teacher 的 token 置信度能不能区分高/低可靠性样本：

![motivation_row_v2.png](https://www.mulanai.com/fs/files/0912_0cad56c7_motivati.png)

*图2：诊断实验结果。左图（code 域）和中图（math 域）画的是高/低可靠性两组样本的 teacher mean token log-prob 分布——code 域两组分布几乎完全重叠，AUROC 只有 0.51；math 稍好但也只有 0.73。右图更扎心：在低可靠性区间（$q_T^{(10)} \lt 0.5$，蓝色阴影），teacher 置信度最高的那条 sampled response 在 code 上 84% 是错的、math 上 61% 是错的。*

"自信地错"这件事用 entropy 或者 teacher-student likelihood agreement 这类分布代理指标是看不出来的——它们度量的是不确定性或一致性，不是答案对不对。而 reverse KL 偏偏就喜欢往 teacher 的高概率行为上靠。Vanilla OPD 实际上在系统性地放大这类错误。

这就是关键。两个问题的答案居然是同一个：**让 teacher 在闲下来的时候，自己先做几道题自测一下**。

## 🧠 方法：TGOPD 的三件套

一句话讲清核心 idea：每个 prompt 进训练之前，先让 teacher 自己解 K 次，用 verifier 统计通过率，通过率高就上稠密蒸馏，通不过就退回 GRPO——两种监督从不在同一个 prompt 上混用。

![tgopd_framework_v2.png](https://www.mulanai.com/fs/files/0912_ea9759f8_tgopd_fr.png)

*图3：TGOPD 框架对比。上半部分是 Vanilla OPD：prompt 进来，student 采样 G 条 rollouts，teacher 直接对所有 prompts 施加 OPD 更新。下半部分是 TGOPD：prompt 同时走两条路——student 照常采样，teacher 额外生成 K 条 probe rollouts 交给 verifier 算出可靠性 $q_T(x)$，然后进入自适应监督分支：$q_T(x) \geq \tau$（可靠）走 OPD 的逐 token 优势，否则（不可靠）走 GRPO 的整轨迹相对优势。*

拆开看技术细节。

**可靠性估计**。teacher 在 prompt $x$ 上的可靠性被定义为期望 verifier 奖励，说白了就是"teacher 随手做一次能做对的概率"：

$$R_T(x) = \mathbb{E}_{y \sim \pi_T(\cdot \mid x)}\left[r(x, y)\right] \in [0, 1]$$

估计方式简单粗暴：teacher 独立生成 $K_T$ 条 probe rollouts，用**和 student rollouts 完全相同的 verifier** 打分（code 用单测，math 和 IF 用规则判分器），经验通过率：

$$q_T(x) = \frac{1}{K_T}\sum_{k=1}^{K_T} r(x, \hat{y}^k)$$

因为 probes 是 i.i.d. 采样且 $r$ 是二值的，$K_T q_T(x)$ 服从二项分布——这个估计量无偏，方差 $R_T(1-R_T)/K_T$ 随 $K_T$ 线性下降。主实验 $K_T=3$ 就够用，不需要精确排序 prompts，只要粗粒度的"靠谱/不靠谱"判断。

**硬门控**。可靠性过了阈值才放行：

$$g(x) = \mathbb{1}\left[q_T(x) \geq \tau\right]$$

默认 $K_T=3$、$\tau=2/3$，也就是三取二多数表决。

**路由，而不是混合**。这是我觉得设计里最克制也最对的一个决定。门开着，OPD 的逐 token 优势完全接管，verifier 的轨迹级奖励不进梯度；门关着，teacher 监督整体撤回，改用 GRPO——rollout 组内奖励中心化（遵循 Dr. GRPO 惯例，不做标准差归一化）。如果组内全对或全错，中心化后优势恰好为零，这个 prompt 干脆不产生更新。合成到一条公式里：

$$\hat{A}^{\mathrm{TGOPD}}_{i,t} = g(x)\,\hat{A}^{\mathrm{OPD}}_{i,t} + \left(1 - g(x)\right)\hat{A}^{\mathrm{GRPO}}_{i,t}$$

注意 $g(x) \in \{0, 1\}$，这是选择器不是插值器。Vanilla OPD（门恒开）和纯 GRPO（门恒关）只是它的两个退化端点。

为什么不在一条 trajectory 内部混合两种信号？我的理解是，OPD 的优势来自 teacher-student 的 log-likelihood 差，GRPO 的优势来自组内相对奖励——两者量纲和来源完全不同，混在一起校准会引入一堆说不清的超参。Prompt 级别一刀切，干净。

**为什么 probe 几乎不花额外的钱**。这是全文我最喜欢的设计。teacher 在 cycle 开始时就并发发出 $K_T$ 条 probe decodes，跟 student 的生成同时进行——正好占用 teacher 节点原本闲置的那个窗口。如果 probes 在 student batch 完成前跑完，墙上时间一分不加。而 teacher 的 scoring pass 保持无条件执行（单次 forward 很便宜，条件化反而增加流水线分支）。所以 TGOPD 的全部额外工作就是 K 条 probe decodes，且大部分被闲置容量吸收。附录的匹配实验显示，35B CodeIO run 的 500 个对齐 cycle 里平均 step time 只增加 5.9%，decode throughput 变化小于 0.1%——审计不是严格免费，但接近。

## 🔧 系统实测：利用率真的被填满了

论文在四种部署配置（4B/35B × 单域 SOPD/多域 MOPD）下各采了一小时窗口、15 秒分辨率的利用率数据：

![gpu_util_timeseries_grid.png](https://www.mulanai.com/fs/files/0912_a6d8ed02_gpu_util.png)

*图4：四种配置下 teacher 节点的 GPU 利用率时间序列。可以明显看到标准 OPD（橙色）全是低矮尖峰，TGOPD（蓝色）把曲线整体抬到了高水位。MOPD 下提升稍小（67%/58%），因为 probing 也是按领域路由的——一个节点被切成 code 4 卡、math 2 卡、IF 2 卡三个引擎，只有当前 prompt 所属领域的引擎在工作，呈双峰分布。*

汇总成分布和均值更直观：

![gpu_util_analysis.png](https://www.mulanai.com/fs/files/0912_ff115786_gpu_util.png)

*图5：左图是四种配置 teacher 节点利用率的核密度分布对比——橙色 Standard OPD 全部堆在 10% 以下的底部，蓝色 TGOPD 整体移到 58%-83% 的高位。右图是集群平均利用率：4B SOPD 提升 18.0 个点（51%→69%），35B SOPD 提升 11.8 个点，两种 MOPD 配置分别提升 12.7 和 8.6 个点；rollout 节点利用率在四种配置下基本不变（波动在正常 run-to-run 方差内），说明没有持续资源争用。*

几个值得记住的数：idle（低于 5% 利用率）样本占比从 59%-78% 降到 0%-2%；35B SOPD 的 teacher 节点利用率拉到 82.8%，是四种配置里最高的。这个闲置现象是结构性的，跟具体配置无关——只要 scoring 比 decoding 便宜且必须等 batch，teacher 就一定闲。

## 📊 实验：六个单域全胜，唯一正的 code 迁移

实验设置交代一下：student 是 Qwen3.5-4B（dense）和 Qwen3.6-35B-A3B（MoE，3B 激活参数）；三个领域——数学（DAPO-Math-17K 训练）、代码（CodeI/O 的 input-output prediction）、指令遵循（Nemotron-Cascade 2 过滤 prompts）。每个领域先用 GRPO 训一个专用 teacher 再冻结。基线包括 Vanilla OPD、TrOPD（token 级 trust region）、RG-OPD（verifier 一致性做轨迹级门控）、RLSD-style（方向-幅度分解）。框架是 slime 的异步 rollout-update 架构，所有方法共享 IcePop 做 train-inference mismatch 校正。

**单域结果（Table 1）**，4B 的关键数字：

| 方法 | AIME 2025 | AIME 2026 | HMMT-Feb | LiveCodeBench | OJBench | IFBench | IFEval |
|---|---|---|---|---|---|---|---|
| Base Model | 47.8 | 58.1 | 40.5 | 39.4 | 14.4 | 35.9 | 85.3 |
| Teacher | 63.4 | 73.4 | 54.8 | 53.3 | 17.0 | 55.9 | 85.3 |
| Vanilla OPD | 61.1 | 71.2 | 54.3 | 42.3 | 18.8 | 48.5 | 83.9 |
| TrOPD | 62.1 | 70.5 | 56.8 | 49.6 | 18.1 | 53.2 | 84.6 |
| RG-OPD | 63.5 | 73.6 | 54.1 | 45.6 | 19.4 | 54.3 | 84.1 |
| RLSD-style | 48.8 | 56.7 | 40.3 | 38.4 | 15.1 | 31.5 | 79.8 |
| TGOPD | 64.8 | 73.5 | 52.7 | 47.1 | 20.0 | 50.4 | 85.2 |

35B 那边我挑最戏剧性的说：**35B 代码域是最干净的胜负手**。LiveCodeBench 上，所有其他蒸馏方法全部产生负迁移——Vanilla OPD 比 base 还低 0.8 个点，TrOPD 低 2.5，RG-OPD 低 3.5，RLSD-style 低 4.1。而 TGOPD 是唯一正向迁移的方法：比 base 高 3.0 个点（64.0 vs 61.0），甚至比 teacher 本身还高 1.3 个点。OJBench 上同样超过 teacher（28.7 vs 27.6）。

为什么偏偏是 code？回到图2，code teacher 的"自信地错"用 confidence 检测的 AUROC 只有 0.51——基本等于掷硬币。别的领域 teacher 错了多少还留点痕迹，code 领域的错误是完全隐形的，所以对 code 来说，prompt 级可靠性门控的边际收益最大。整体增益排序也印证了这一点：code 最大（两个规模平均提升约 3 个点），IF 次之，math 最小。

RLSD-style 的对照也很有意思：它对所有 prompts 统一削弱 teacher 的作用，结果 14 个 benchmark 列里有 8 列低于没训练过的 base——均匀减少 teacher 角色，会把有用的监督也一起砍掉。TGOPD 的思路恰恰是反的：大多数 prompts 上稠密信号原样保留，只在审计失败处撤回。

**多域 MOPD（Table 2）**，无需任何修改直接复用（门控是 per-prompt、per-teacher 的）：4B 平均分从 53.40 提到 54.54（提升 1.14 个点，赢 6/7 列），35B 从 60.99 提到 61.94（提升 0.95 个点，赢 5/7 列）。少数几列回退都在半分以内，处于早期 checkpoint 的正常波动范围。

## 🔬 消融：阈值是倒 U 形，"堵住错误信号"才是收益主体

**阈值消融**（$K_T=5$ 扫五个阈值，4B 数学域，99 steps）：

![threshold_ablation.png](https://www.mulanai.com/fs/files/0912_ec52ea9a_threshol.png)

*图6：门控阈值 $\tau$ 的消融曲线，三个数学 benchmark 全部呈倒 U 形。阈值太松（$\leq 2/5$）放进太多低可靠性信号，增益变小；太紧（$\geq 4/5$）又把有用监督拒之门外——$\tau=5/5$ 时 HMMT-Feb 甚至跌破 Vanilla OPD 基线。峰值在 $\tau=3/5$：AIME 2026 达 73.1、AIME 2025 达 64.4、HMMT-Feb 达 58.0，分别超 Vanilla OPD 1.9、3.3、3.7 个点。*

峰值 60% 跟主实验默认的 2/3（约 67%）挨得很近——说明在"多数表决"附近存在一个相当宽的最优区间，工程上不用精细调这个阈值。

**门关之后怎么办**（GRPO fallback vs Mask only）的消融更有信息量。跨四个设置统计，masking 平均提升 1.08 个点，full fallback 平均提升 1.20 个点。也就是说，大约 90% 的收益在"把不可靠的 teacher 信号直接掐掉、什么都不补"时就已经拿到了。

这个数字值得停下来想想。TGOPD 涨点的主体不是 GRPO fallback 提供了什么好信号，而是**阻止了坏信号进入梯度**。这和图2的诊断闭环了——真正伤训练的是那些 confidently wrong 的稠密监督。fallback 只在 student 自己也能解决一部分 gated-off prompts 时锦上添花（组内有对有错才有相对信号），code 域因为执行式 verifier 信号直接，fallback 在 OJBench 上两个规模都赢了 masking。

## 🤔 我的判断

这篇论文最值钱的地方，是把一个系统观察和一个算法洞察缝在了一起：teacher 节点反正闲着，不如让它先自测——自测结果同时解决了可靠性问题和利用率问题。这种"一个改动治两个病"的设计，通常比单纯的算法创新更值得工程团队抄作业。

批判几句也说在前面。

关于 baseline：TrOPD 和 RG-OPD 在不少列上跟 TGOPD 互有胜负（比如 4B HMMT-Feb 上 TrOPD 56.8 对 TGOPD 52.7），TGOPD 赢在 6/6 设置的稳定性而不是每列都碾压。增益绝对值多在 1-3 个点，属于扎实但不惊人的幅度。

关于适用范围：整套方法依赖 automatic verifier。open-ended 任务（写作、对话）没有现成的二元判分器，可靠性估计这条路走不通——作者自己也承认这是最重要的扩展方向。

关于"先验证再蒸馏"这个 idea 的新颖性：坦白讲，用 verifier 给 teacher 信号做质量过滤并不是没人想过，RG-OPD 已经在 trajectory 级做了类似的事。TGOPD 的差异在于决策的粒度和时机——在稠密监督被接纳之前、在 prompt 级、用完整 teacher rollouts 做排他性路由，而不是事后剔除或校准。粒度选择带来的工程简洁性（probe 填闲置窗口）是它真正区别于前人的地方。

还有一个我自己也没完全想透的点：fallback 和 masking 的差距其实很小（0.15-0.60 个点），4B 多域下 masking 反而更好。作者选 fallback 的理由是"一致性"，但如果你的场景 verifier 信号弱或组内方差经常为零，直接 mask 掉可能更省事。这块结论别当成铁律。

**工程启发**：如果你在做异步蒸馏架构，不管用不用这套门控，"teacher 节点利用率常年个位数"这个现象都值得去自己集群上量一下——59%-78% 的时间低于 5% 利用率，这个浪费是结构性的。哪怕只做利用率回收，probe 类任务填充闲置窗口的思路也能直接搬。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*

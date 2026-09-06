# 把聊天历史压成向量直接喂给模型？LatentPress 绕开了"压缩必须还原成文字"的老规矩

## 核心摘要

做 Agent 或者长上下文应用的都有体感：历史越攒越多，每次重读都贵得肉疼，于是大家要么让 LLM 写摘要、要么把文本渲染成图片走 OCR——但这两条路的终点都是"还原成文字"，只因为消费方是语言模型就非得经过人类可读的文本，这个绕路其实很别扭。这篇论文（arXiv:2609.01507）提出的 LatentPress 干了件直接的事：用一个只有千万级参数的小 writer（约占 decoder 的 0.1%），把对话历史和长文档写成一串连续的 soft token，冻结的 LLM 通过 input-embedding 接口直接读，推理时完全不做文本重建。结果挺能打：LongMemEval 上 7.7 倍压缩下准确率 0.504，反而比不压缩的原始证据（0.490）还高，文本摘要只有 0.184，DeepSeek-OCR 压缩到 9.3 倍时掉到 0.312；写入一段对话只要 43ms，比摘要和 OCR 快了一个数量级。我的判断：这不是什么颠覆性架构，而是把"压缩产物应该长什么样"这个问题问对了——soft token 压缩这个老方向，被它做成了一个轻量、可落地的工程接口。

---

## 📖 论文信息

- **标题**：LatentPress: Context Compression Beyond Text and Vision
- **作者**：Zhengze Zhou（Cornell University）、Hejian Sang（Iowa State University），共同一作
- **发表**：arXiv:2609.01507v2 [cs.LG]，2026 年 9 月
- **代码**：https://github.com/HJSang/LatentPress

---

## 🎯 问题动机：给模型看的压缩产物，为什么非得是人能读的？

先说个反直觉的观察。现在主流的上下文压缩方案，不管是 LLM 生成摘要，还是最近很火的 DeepSeek-OCR 这类视觉压缩（把文本渲染成图片，再用 OCR 解码回来），它们输出的最终形态都是**文字**。可人又不看这些压缩产物，消费方自始至终只有一个——语言模型。

那问题就来了：如果最终读者是模型，为什么中间要绕一圈人类可读的文字？

作者的答案很直白：不需要。模型真正消费的是 embedding，不是文字本身。所以压缩上下文完全可以写成第三种形态——不是文本、不是图像，而是**连续的 memory token**，直接插进冻结 decoder 的 input-embedding 层。这个思路说实话不算全新，Gist、AutoCompressor、ICAE、xRAG 都在 soft token 压缩这条路上走过，但 LatentPress 把组合方式调了个个儿：reader 完全冻结、只训练一个 reader 匹配的小 adapter、向量直接进 embedding 层不做重建、还能对不同段落用不同压缩率。

光说不练假把式，看看它和前辈们的定位差异（论文 Table 1 的整理）：

| 方法 | 训练什么 | 可训练规模 | 推理时要重建文字吗 | 压缩率 |
|------|---------|-----------|------------------|--------|
| Gist | 整个 decoder（全量微调） | decoder 级 | 否 | 固定 |
| AutoCompressor | 整个 LLM（递归摘要） | LLM 级 | 否 | 统一 |
| ICAE | LLM encoder（LoRA） | LLM 级 | 是（自编码回文本） | 统一 |
| xRAG | 只训投影层（LLM 冻结） | 小投影层 | 否 | 单 token |
| DeepSeek-OCR | 视觉模型 | 视觉模型级 | 是（OCR 解码） | 按分辨率 |
| **LatentPress** | **只训 adapter，decoder 冻结** | **约 0.1%** | **否** | **可变、按角色分配** |

这张表里我觉得最值钱的两列是"训练什么"和"推理时重建吗"。Gist 和 AutoCompressor 要动 decoder，换个底座就得重训；ICAE 虽然只训 LoRA，但写入时要跑完整 LLM encoder；xRAG 冻结了 reader 却只压单个检索段落成一个 token。LatentPress 是这张表里唯一一个"decoder 一点不动 + 直读不重建 + 整段多轮历史可变压缩率"三条全占的。这个组合，才是它真正的卖点。

---

## 🏗️ 方法核心：Write / Read 分离，贵的部分永远不动

先上一句话版直觉：**压缩和阅读拆成两个角色，Write 用一个借来的小 encoder 把文本写成向量，Read 让冻结的 LLM 把这些向量当 embedding 前缀直接读。**

![图1：LatentPress 整体流程](https://arxiv.org/html/2609.01507v2/figures/softmem_overview.png)

*图1：LatentPress 流程概览（以对话场景为例）。A：一段冗长且异质的历史，包含 user、assistant、tool、environment 等不同信息价值的段落；B：Compressor 在近实时的一次前向传播里把长上下文压成一串 soft token（z1、z2、z3……）；C：冻结 LLM 把"soft memory + 问题"拼成一条输入序列直接解码出答案，全程没有文本重建步骤。*

形式化一点说，给定由 $T$ 个段落组成的上下文 $x=(x_1,\ldots,x_T)$（对话轮次或文档分块），小 writer 把它映射成一串连续向量 $m$，冻结 decoder $f_\theta$ 直接把 $m$ 和问题的 embedding 拼起来读：

$$m = \text{Write}_\phi(x; \pi), \qquad y = f_\theta\big([m; \mathrm{emb}(q)]\big)$$

其中 $\phi$ 是 writer 参数，$\pi$ 是每个段落的压缩率策略。

### writer 怎么搭：借两层，加一个 adapter

具体实现挺克制的。writer 直接**借用冻结 decoder 的底部 $L{=}2$ 层 transformer**（深拷贝一份，梯度不回流到 reader），上面接一个线性 adapter $A \in \mathbb{R}^{d\times d}$，初始化为恒等矩阵——也就是说 writer 起步时输出约等于原始 token embedding，训练让它只在必要的地方偏离。这个初始化设计我觉得挺聪明的，等于给压缩器一个"无损起点"，让训练去决定哪里可以压、压多少。

可训练参数量小得离谱：Qwen2.5-7B 对应 12.849M、Qwen3-8B 对应 16.781M、Qwen3-1.7B 只有 4.196M、Qwen2.5-14B 是 26.220M。全部约为对应 decoder 的 0.1%。因为 soft token 绑定了具体 reader 的 embedding 空间，跨 reader 实验里每个 reader 训一个自己的 writer head。

### 压缩率怎么定：故意做得"笨"

这是论文里一个挺诚实的选择。压缩率策略 $\pi=(k_1,\ldots,k_T)$ 完全是手工指定的，不学习。两种规则：

- **Uniform pooling**：所有段落统一 $k_i=k$，用于无结构的长文档；
- **Role-based**：利用对话天然的角色结构，user 轮 $k_{\text{user}}=1$ 完全无损（user 输入往往短但信息密度高），assistant 轮 $k_{\text{assistant}} \in \{8,16,32\}$ 大力压缩。

你想想看，对话里真正承载"用户事实"的往往是 user 那几短句，assistant 的长篇回复才是体积大头。保住 user 轮、狠压 assistant 轮，这个启发式简单到有点土，但后面的实验证明它确实 work。作者也坦白：学习式的动态压缩率留给未来工作，这篇的重点是验证接口本身。

### 训练目标：重建 + 前向 KL 蒸馏

writer 的训练信号有两项：

$$\mathcal{L}(\phi) = \mathcal{L}_{\mathrm{rec}} + \lambda\, \mathcal{L}_{\mathrm{fkl}}$$

$$\mathcal{L}_{\mathrm{rec}} = -\frac{1}{N}\sum_{t=1}^{N}\log p_{\mathrm{comp},t}(y_t), \qquad \mathcal{L}_{\mathrm{fkl}} = \frac{1}{N}\sum_{t=1}^{N}\operatorname{KL}\!\left(p_{\mathrm{full},t}\,\|\,p_{\mathrm{comp},t}\right)$$

第一项让压缩上下文能恢复目标 token；第二项是前向 KL，把"完整上下文下冻结 decoder 的 next-token 分布"蒸馏进"压缩上下文下的分布"，$\lambda=1.0$。这个蒸馏项的直觉很工程：我们不在乎压缩向量"像不像原文"，只在乎**冻结 reader 读到它之后的行为跟读到原文一样**。注意一个容易混淆的点——训练里有重建损失，但推理时没有任何文本重建，论文管这叫"reconstruction-free inference"，别把训练和推理搞混了。

---

## 🧪 实验一：对话记忆，压缩 7.7 倍反而超过不压缩

第一个战场是 LongMemEval，500 道记忆问答，用的是 oracle-evidence 设置（每题只配 ground-truth 证据会话，隔离掉检索环节，纯考察"压缩后还能不能读对"）。writer 在 2000 条 UltraChat 对话上训练（无 QA 标签），零样本迁移到 LongMemEval，judge 是 Llama-3.1-70B-Instruct。

![图2：LongMemEval 准确率-压缩率前沿](https://www.mulanai.com/fs/files/0906_58143dc5_longmeme.png)

*图2：三个 reader（Qwen2.5-7B / Qwen3-8B / Qwen3-1.7B）上的准确率-压缩率前沿。橙色是 role-aware LatentPress，在三个 reader 上都几乎是一条平稳直线；蓝色 DeepSeek-OCR 随压缩率上升明显下滑；红色叉号的文本摘要在每个 reader 上都是最弱的一个点；灰色菱形是不压缩的 oracle 证据基线。*

Qwen2.5-7B 上的完整对比（论文 Table 2）：

| 方法 | 压缩率 | Overall 准确率 |
|------|--------|---------------|
| 不压缩 oracle 证据 | 1.0× | 0.490 |
| LatentPress（$k_a{=}8$） | 4.62× | 0.476 ± 0.014 |
| LatentPress（$k_a{=}16$） | 6.27× | 0.478 ± 0.020 |
| LatentPress（$k_a{=}32$） | 7.70× | **0.504 ± 0.024** |
| ICAE | 4.12× / 8.96× / 17.28× | 0.452 / 0.318 / 0.174 |
| DeepSeek-OCR | 2.33× / 5.97× / 9.34× | 0.426 / 0.390 / 0.312 |
| 文本摘要 | 12.06× | 0.184 |

有几个数字值得停下来看。

**0.504 对 0.490——压缩后反超不压缩。** 这个结果我第一次看到时愣了一下。作者的解释是 oracle 设置下 reader 拿到的虽然是对的证据，但题目需要跨会话聚合、时序推理、知识更新追踪，冗长的原文反而稀释了关键信息；LatentPress 把 user 短轮无损保留、长轮压掉，某种程度上起了"去噪"作用。这个现象在更弱的 Qwen3-1.7B 上更明显——压缩版 0.434 直接把 OCR 的 0.264 甩开一大截，说明小模型更吃"替它把信息浓缩好"这一套。

**ICAE 掉得很难看。** 4 倍时还有 0.452，17 倍直接崩到 0.174。同样的冻结 reader、同样的评测，LatentPress 在相近压缩率下领先一大截——差距的来源大概率就是 ICAE 那套"编码-重建"的目标和 role-aware 的分配策略。

**文本摘要全线垫底。** 0.184 对 0.504，这个差距大到有点残忍。摘要这种"由 LLM 主观取舍信息"的路子，在需要精确事实的记忆任务上确实天然吃亏。

跨 backbone 的泛化（Table 3）也做了：Qwen3-8B 上 LatentPress 拿到 0.506/0.514/0.494，OCR 在低压缩率 2.33× 时以 0.542 领先，但压缩一加大就被反超；role-aware 相对 uniform pooling 的优势在三个 reader 上达到 **0.34 到 0.45 个点**——等等，这个幅度意味着"按角色分配压缩率"这个土办法贡献了论文里最大的一块收益，比换模型、换 backbone 都管用。

---

## 🧪 实验二：长文档 QA，4-8 倍压缩能干过原文阅读

对话有角色结构可以白嫖，那没有这个结构的长文档呢？作者把场景切到 LongBench-QA 英文六个子集（NarrativeQA、Qasper、MultiFieldQA-en、HotpotQA、2WikiMultihopQA、MuSiQue），改用 uniform 压缩，考察两种监督来源：跨域迁移（用 LongMemEval 派生的 QA 训 writer，直接考文档）和域内适配（直接在 LongBench-QA 训练集上训）。

不压缩的参考线：Qwen2.5-14B 是 47.93 分，Qwen2.5-7B 是 43.80，Qwen3-8B 只有 30.80（non-thinking 模式）。

![图3：LongBench-QA 准确率-压缩率前沿](https://www.mulanai.com/fs/files/0906_ac9b4650_longbenc.png)

*图3：三个 reader 上的 LongBench-QA 总分-压缩率曲线。橙色 in-domain 曲线在 4× 和 8× 处压过灰色菱形的原文基线，16× 处跌破；绿色 cross-domain 只在 4× 勉强追平；蓝色 OCR 和红色文本摘要全程被压着打。*

域内适配的完整数字（Table 4，五次种子均值）：

| Reader | 设置 | Overall 分数 |
|--------|------|-------------|
| Qwen2.5-7B | 原文 1× | 43.80 |
| | in-domain 4× | **49.06 ± 2.30** |
| | in-domain 8× | 43.77 ± 2.83 |
| | in-domain 16× | 37.78 ± 3.46 |
| Qwen3-8B | 原文 1× | 30.80 |
| | in-domain 4× | **39.62 ± 2.31** |
| | in-domain 8× | 36.93 ± 2.82 |
| | in-domain 16× | 26.12 ± 3.33 |
| Qwen2.5-14B | 原文 1× | 47.93 |
| | in-domain 4× | **57.99 ± 2.35** |
| | in-domain 8× | 52.18 ± 2.82 |
| | in-domain 16× | 40.30 ± 3.51 |

Qwen2.5-14B 上 4 倍压缩拿到 57.99，比原文阅读的 47.93 高了 10 个点——说实话这个提升幅度大到让我想多看两眼。压缩竟然起到了正则化的作用，把无关细节挤掉之后 reader 反而答得更准。Qwen3-8B 上也是类似剧本：30.80 涨到 39.62。

不过别高兴太早。**16 倍压缩全线崩盘**，三个 reader 都跌破原文基线。到那个程度，逐字细节的损失开始暴露真实代价，压缩不是免费的午餐。另外跨域迁移只在 4× 勉强打平原文（Qwen2.5-7B 上 45.13 对 43.80），更高压缩率就掉队——想拿到"超过原文"的红利，基本得付出域内训练的代价。

---

## ⚡ 效率：写入 43ms，读取快 5-9 倍

准确率之外，这篇论文把效率拆成写入和读取两笔账，都给了实测。

**写入成本**（Qwen3-8B backbone、bf16、单张 H100 80GB、batch 8）：LatentPress 每段对话 43ms，就是一次前向传播。对比下来：DeepSeek-OCR 要渲染页面再自回归做光学解码，844–1056ms（约 22 倍）；文本摘要 407–645ms（9–15 倍）；最接近的 soft-token 对手 ICAE 也要 350–700ms（8–15 倍），因为 ICAE 编码要跑完整 LLM，而 LatentPress 只借了两层。

**读取成本**（30 条 LongBench-QA 样本、模型热加载、纯推理延迟，$f8$ 配置）：

| Reader | 原文上下文 | LatentPress f8 | 缓存 OCR |
|--------|-----------|---------------|---------|
| Qwen2.5-7B | 2.44s | 0.49s | 2.71s |
| Qwen2.5-14B | 4.14s | 0.49s | 4.34s |
| Qwen3-8B | 3.97s | 0.43s | 4.03s |

比原文推理快 5.0–9.2 倍，比缓存好的 OCR 路线快 5.5–9.4 倍。端到端整任务时间（含 adapter 训练）比冷启动 OCR 管线短 6.0–13.7 倍。

公平起见说一句：这些加速比的比较对象都是"重建式"路线，而不是同样一次前向就能写完的其他 soft-token 方法。和 ICAE 比写入速度赢面很大，但读取延迟两者其实接近——都是往冻结 decoder 里塞一段短前缀。

---

## 🤔 我的判断

这篇论文最值钱的地方，**是它把一个表述问题变成了接口问题**。"压缩上下文该长什么样"以前默认答案是"更短的文字"，LatentPress 证明了答案可以是"reader 自己的 embedding 空间里的一串向量"，而且只需要 0.1% 的可训练参数就能把这个接口训出来。对工程实践来说，这个属性很香：底座模型升级不用重训压缩管线主体，换个 reader 重训一个小 adapter 就行。

但泼几盆冷水。

**评测设置偏理想化。** LongMemEval 用的是 oracle evidence——检索难题被绕开了。真实部署里，压缩器面对的从来不是"刚好对的几段证据"，而是掺杂大量噪声的完整 haystack。作者自己也承认这一点，把接检索器留给了未来工作。所以 0.504 这个数字要打折看，它回答的是"读"的问题，不是"找+读"的问题。

**role-based 策略的边界很窄。** $k_{\text{user}}=1$ 这个设计在对话记忆里是大杀器，但它其实是利用了"用户轮短而密、助手轮长而稀"这个特定先验。换到 tool call 日志、多 agent 协作轨迹这类结构不同的历史，这套启发式未必成立——论文在长文档上退回 uniform 压缩，某种程度上已经说明了问题。真正的 dynamic compression policy（按段落重要性学习压缩率）才是把这个方向推向下一个台阶的关键，作者把它写在 future work 里，我觉得那才是下半场的主线。

**每个 reader 一个 writer 也是隐性成本。** soft token 绑死在特定 embedding 空间，意味着多模型混部的系统里每接一个底座就要训一个 head。好消息是 head 只有千万级参数，坏消息是"换一个模型重训一次"这条链路依然躲不掉。

和同期工作摆在一起看：ICAE 证明了 soft token 压缩可行但要动 LLM 级编码器，DeepSeek-OCR 带火了视觉压缩但绕不开自回归重建，xRAG 冻结 reader 但只服务检索单跳。LatentPress 更像是把这些路线里正确的零件拆下来重新组装——它不是什么底层突破，是一次相当务实的工程整合，而且整合得确实漂亮。对在做长期记忆、Agent 历史管理的人来说，"writer 借两层 + adapter 恒等初始化 + 重建+KL 蒸馏"这套配方直接可以抄。

---

## 📝 收尾

压缩上下文这个需求只会越来越刚性——Agent 跑得越久，历史越重。LatentPress 给出的启发是：别再把"人类可读"当成压缩产物的默认约束，机器消费的上下文就该用机器原生的形态来存。下一步的看点很明确：谁能把压缩率策略学成动态的，谁就能把 16× 那条崩掉的曲线救回来。

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我

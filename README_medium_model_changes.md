# 中号残差模型改动记录

本文档记录 `v0.3` 状态扩充小模型之后的新增改动，供后续训练复盘、Botzone 提交说明和实验报告撰写使用。

## 作业要求摘录

根据课程作业说明，后续报告和提交材料需要覆盖以下内容：

- 智能体需要能在 Botzone 国标麻将环境中正常运行并参加评测。
- 研究性报告建议 3-8 页，内容包括算法细节、优化和创新点、实验现象和启发等。
- 完整代码和模型需要一并提交，说明安装流程和脚本执行方法。
- 训练好的策略网络模型需要保存为 `.pt` 文件。
- 鼓励探索特征处理、模型结构、强化学习/监督学习结合、参数调优和策略创新。
- 评分重点包括 Bot 性能、算法创新点、代码性能和报告写作。

## 基线版本

`v0.3` 标签对应“状态扩充之后的小模型最终版”：

- 输入特征：`148 x 4 x 9` 的局面张量 + `10` 维 global 特征。
- 动作空间：`235` 个离散动作，仍通过 legal action mask 屏蔽非法动作。
- SL 小模型参数量：约 `1.05M`。
- RL 小模型参数量：约 `1.65M`。

该版本适合作为后续对比实验中的 small baseline。

## 新增模型结构

本次改动将原来的浅层 CNN 扩展为中号残差 CNN，主要参考了比赛中常见的 “stem + ResBlock + mask output” 结构。

### SL 策略网络

文件：`SL/model.py`

新结构：

```text
observation(148, 4, 9)
  -> Conv2d 148 -> 192, 3x3
  -> BatchNorm2d + ReLU
  -> Conv2d 192 -> 128, 3x3
  -> BatchNorm2d + ReLU
  -> ResidualBlock(128) x 6
  -> Flatten

global(10)
  -> Linear 10 -> 64
  -> ReLU

concat
  -> Linear (128*4*9 + 64) -> 512
  -> ReLU
  -> Linear 512 -> 235
  -> legal action mask
```

参数量从约 `1.05M` 提升到约 `4.76M`。

### RL Actor-Critic 网络

文件：`RL/model.py`

RL 使用与 SL 相同的 residual encoder：

```text
shared residual encoder
  -> policy head: Linear -> 512 -> 235
  -> value head:  Linear -> 512 -> 1
```

参数量从约 `1.65M` 提升到约 `7.16M`。

SL checkpoint 仍可以初始化 RL 的 policy encoder 和 policy head，value head 保持随机初始化，用于后续 PPO 学习状态价值。

## 算法改动说明

### 1. ResidualBlock

每个残差块结构为：

```text
x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> +x -> ReLU
```

使用残差连接的原因：

- 可以加深 CNN 而不容易出现梯度传播困难。
- 更深的网络能学习多层组合牌理，例如搭子形状、进张、弃牌河、防守风险、副露信息之间的组合关系。
- 对麻将这种局面复杂但输入空间较小的任务，残差块比单纯堆全连接层更适合提取局部牌型模式。

### 2. BatchNorm

卷积层后加入 BatchNorm，作用是稳定中间特征分布：

- 提升深层 CNN 的训练稳定性。
- 减少学习率稍大时的震荡。
- 配合残差块能让 SL 预训练更容易收敛。

### 3. 512 hidden head

`512` 不是动作数，最终输出仍然是 `235` 个动作 logits。

这里的 `512` 是动作打分前的隐藏表达维度：

```text
hidden features -> Linear -> 512 -> ReLU -> Linear -> 235
```

使用 512 hidden head 的原因：

- 235 个动作只是输出类别数，不等于模型评估动作所需的思考维度。
- 麻将动作质量依赖手牌、副露、弃牌、剩余牌、风位和是否杠后等组合信息。
- 更宽的 head 能在输出动作分数前保留更多策略特征，减少从局面特征到动作 logits 的瓶颈。

如果后续发现过拟合或训练太慢，可以把 `HEAD_SIZE` 调成 `384` 或回退到 `256` 做消融实验。

### 4. AdamW + Weight Decay

SL 和 RL 训练器从 Adam 改为 AdamW，并默认使用：

```text
weight_decay = 1e-4
```

好处：

- 在模型容量变大后降低过拟合风险。
- AdamW 的权重衰减实现比普通 Adam + L2 更稳定，适合深层神经网络训练。

### 5. Gradient Clipping

SL 和 RL 默认加入：

```text
grad_clip = 1.0
```

好处：

- 防止深层网络训练初期出现梯度过大。
- PPO 中多轮更新同一批样本时，梯度裁剪可以减少策略更新过猛导致的不稳定。

## 预期效果

相比 `v0.3` 小模型，新模型预期有以下提升：

- 更强的牌型组合表达能力。
- 更强的中后期攻守判断容量。
- SL 阶段更容易利用 586 万级监督样本。
- RL 阶段 policy/value 共享更强 encoder，理论上能得到更稳定的策略改进。

需要注意：结构增强不等于立即变强，最终效果仍需要通过 SL 验证集指标和 Botzone/自对弈胜率确认。

## 推荐训练流程

### 1. 重新训练 SL 中号模型

旧 small checkpoint 不能直接 resume 到新结构，建议重新开始 SL 预训练：

```bash
python3 SL/supervised.py --epochs 20 --batch-size 1024 --lr 5e-4 --weight-decay 1e-4 --grad-clip 1.0
```

如果显存或 NPU 内存不足，可以降低 batch size：

```bash
python3 SL/supervised.py --batch-size 512
```

### 2. 用 SL checkpoint 初始化 RL

选择 SL 训练中表现最好的 checkpoint：

```bash
python3 RL/train.py --sl-init-checkpoint SL/model/checkpoint/<epoch>.pkl --lr 1e-4 --weight-decay 1e-4 --grad-clip 1.0
```

如果 RL 波动较大，可尝试：

```bash
python3 RL/train.py --lr 5e-5 --grad-clip 0.5
```

## 建议记录的实验指标

为了后续报告更完整，建议记录以下实验：

| 实验项 | 目的 | 建议记录 |
| --- | --- | --- |
| small(v0.3) vs medium | 比较模型容量提升是否有效 | SL loss、验证 acc、Botzone/自对弈表现 |
| HEAD_SIZE 256/384/512 | 判断 head 宽度是否必要 | 收敛速度、验证 acc、训练耗时 |
| RES_BLOCKS 3/6/9 | 判断深度收益 | 参数量、训练稳定性、最终表现 |
| Adam vs AdamW | 判断权重衰减收益 | 验证 acc、过拟合迹象 |
| grad_clip 开/关 | 判断训练稳定性 | loss spike、RL reward 波动 |

## 多模型对局评测

新增脚本：

```text
RL/evaluate_models.py
```

用途：

- 加载多个 checkpoint，在本地 `RL/env.py` 环境中跑多局对战。
- 自动识别 small/medium 架构。
- 兼容 SL checkpoint 的 `_tower_head` 和 RL checkpoint 的 `_logits`。
- 支持 `1v1v1v1` 自由对战统计。
- 支持 `2v2` 统计，默认座位 `A,B,A,B`，也就是 0/2 为一队，1/3 为一队。
- 默认输出汇总 JSON；需要排查单局结果时，可额外传 `--output` 保存每局 CSV。

### 1v1v1v1 示例

```bash
python3 RL/evaluate_models.py \
  --model baseline=model/checkpoint_2/cpu/19.pkl \
  --model slmedium=SL/model_medium/checkpoint/20.pkl \
  --model rl=RL/checkpoint/model_1000.pt \
  --model rlpro=RL/checkpoint/model_2000.pt \
  --seats baseline,slmedium,rl,rlpro \
  --mode ffa \
  --games 200 \
  --device auto \
  --summary-output eval_ffa_summary.json
```

脚本默认会轮换座位，降低东南西北座位差异对结果的影响。

### 2v2 示例

两个模型做 2v2 时，如果只传两个模型，默认座位为 `A,B,A,B`：

```bash
python3 RL/evaluate_models.py \
  --model baseline=model/checkpoint_2/cpu/19.pkl \
  --model slmedium=SL/model_medium/checkpoint/20.pkl \
  --mode team \
  --games 200 \
  --device auto \
  --summary-output eval_2v2_summary.json
```

也可以手动指定座位：

```bash
python3 RL/evaluate_models.py \
  --model baseline=model/checkpoint_2/cpu/19.pkl \
  --model slmedium=SL/model_medium/checkpoint/20.pkl \
  --seats baseline,slmedium,baseline,slmedium \
  --mode team \
  --games 200 \
  --device auto
```

### 建议评测指标

汇总 JSON 中重点看：

- `avg_reward`：平均每个座位的分数，最重要。
- `win_rate`：第一名率或队伍胜率。
- `invalids`：无效动作次数，应该尽量为 0。
- 2v2 时看 `teams` 里的队伍平均分和队伍胜率。

建议每次至少跑 `200` 局；如果模型差距很小，增加到 `1000` 局更稳。

如果需要保存逐局明细，额外加：

```bash
--output eval_games.csv
```

## 后续报告可写的创新点

- 在状态扩充特征基础上，将浅层 CNN 升级为残差 CNN，提高对复杂牌理组合的建模能力。
- 保留 legal action mask，保证网络只在合法动作集合内决策。
- 采用 SL 预训练 + PPO 强化学习微调的两阶段训练方式。
- 用 AdamW、weight decay 和 gradient clipping 提升大模型训练稳定性。
- 使用 `v0.3` 小模型作为对照组，比较模型容量增加前后的性能差异。
- 增加离线多模型对战评测脚本，通过平均得分和胜率比较不同 checkpoint 的实战表现。

## 当前验证结果

已完成的代码级验证：

- `python3 -m py_compile` 通过。
- SL forward 通过，输出形状为 `(batch, 235)`。
- RL forward 通过，输出形状为 `logits(batch, 235)` 和 `value(batch, 1)`。
- 新 SL checkpoint 到 RL policy 分支的参数映射通过。

尚未完成的实验验证：

- 新模型尚未重新完成 SL 训练。
- 尚未进行 RL 微调。
- 尚未进行 Botzone 积分赛或稳定自对弈评测。

## 后续备选结构：共享花色 1D 编码器

该方案是后续可选的模型结构方向，当前先作为实验记录保存，方便后续继续迭代时查阅。

### 设计动机

麻将中 `1-9` 的数字顺序有明确意义，例如顺子、两面、边张、老少副、组合龙等都依赖数字结构；但万、条、筒三门花色的具体身份大多数情况下没有天然大小或相邻关系。

因此，直接在 `3 x 9` 的花色平面上做普通 2D 卷积，可能会引入一个不完全合理的先验：

```text
W 和 T 相邻，T 和 B 相邻
```

但实际上花色之间更像是可交换的离散类别，而不是连续空间轴。更合理的 inductive bias 是：

```text
数字轴有顺序，需要卷积；
花色轴可交换，底层数字模式应共享参数。
```

### 结构草案

将 suit 输入从：

```text
batch x channels x 3 x 9
```

reshape 为：

```text
(batch * 3) x channels x 9
```

然后对三门花色使用同一个 1D encoder：

```text
W 1-9 -> shared 1D suit encoder -> e_W
T 1-9 -> shared 1D suit encoder -> e_T
B 1-9 -> shared 1D suit encoder -> e_B
```

再将三个花色 embedding 拼接：

```text
concat(e_W, e_T, e_B)
```

与字牌塔和 global 特征融合：

```text
concat(suit_embeddings, honor_embedding, global_embedding)
  -> MLP
  -> 235 action logits
```

### 不建议直接 pooling

不建议将三个花色 embedding 直接做 mean/max pooling：

```text
mean(e_W, e_T, e_B)
```

原因是 pooling 会弱化甚至抹掉三门花色之间的结构差异，使模型更难学习跨花色番型和动作定位，例如：

- 三色三同顺
- 三色三步高
- 花龙
- 组合龙
- 清一色
- 混一色
- 缺一门
- 五门齐

保留 `concat(e_W, e_T, e_B)` 的好处是：底层共享数字牌模式，高层仍能判断三门花色分别处于什么状态。

### 对碰、杠、番型判断的影响

该结构不应导致模型学不到碰或杠。碰、杠依赖的是同一张牌的张数、当前被打出的牌、legal action mask 和副露后的价值判断。

共享 1D encoder 反而可能提高样本效率，因为：

```text
W5 对子、T5 对子、B5 对子本质上是同一种对子结构。
```

只要不对三个花色做纯 pooling，而是保留三个 embedding 的 concat，融合层仍能区分：

```text
Peng W5 / Peng T5 / Peng B5
```

以及不同花色组合对应的番型价值。

### Honor 塔建议

字牌没有数字顺序，东南西北中发白不构成连续数轴，因此 Honor 塔更适合使用：

```text
Flatten -> Linear -> Residual MLP
```

而不是 1D 卷积。这样可以避免模型错误学习“东和南比东和白更相邻”之类的伪局部关系。

### 预期优点

- 更符合麻将中“数字有序、花色近似可交换”的结构先验。
- 用共享参数学习三门花色的共同数字模式，提升样本效率。
- 通过 concat 保留跨花色番型判断能力。
- 与 72 种花色/数字/中发白数据增强方向一致。

### 潜在风险

- 模型结构变化后需要重新训练，旧 checkpoint 基本不能继承。
- 如果融合层容量不足，可能影响跨花色番型判断。
- 如果错误使用 pooling，可能削弱动作定位和三色类番型学习。
- 需要重新比较 SL 验证集 accuracy、Botzone 表现和每 epoch 训练耗时。

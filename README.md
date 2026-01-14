# CS336
```text
.
├── alignment/
│   ├── grpo/
│   │   ├── README.md       # 实验文档
│   │   ├── config.json     # 训练配置
│   │   └── ...
│   ├── sft/
│   │   ├── README.md       # 实验文档
│   │   ├── config.json     # 训练配置
│   │   └── ...
│   ├── README.md           # 环境配置
│   └── ...
├── pretraining/
│   ├── README.md           # 环境配置+实验文档
│   ├── config.json         # 训练配置
│   └── ...
├── .gitattributes
├── .gitignore
├── README.md
└── ...
```
# Pretraining
## Architecture

<img width="747" height="503" alt="image" src="https://github.com/user-attachments/assets/2d8f25a6-40d9-487e-8f30-a5b6854b5eab" />

## Experiments
最优学习率为8e-4，使用数据集TinyStories
|                    | 学习率配置 (max) | 最小验证损失 (Val Loss)   | 最小训练损失 (Train Loss) |
| ------------------ | ----------- | ------------------- | ------------------- |
| Baseline (PreNorm) | 8e-4        | 1.3233 (Step 11900) | 1.1614 (Step 19900) |
| PostNorm           | 8e-4        | 1.3347 (Step 18100) | 1.1831 (Step 19900) |
| Remove RoPE        | 8e-4        | 1.3944 (Step 18100) | 1.2588 (Step 20000) |
| SiLU               | 8e-4        | 1.3640 (Step 19400) | 1.2173 (Step 19900) |
| Remove RMSNorm     | 8e-4        | -                   | -                   |
| Remove RMSNorm     | 3e-4        | 1.5351 (Step 18100) | 1.5458 (Step 19900) |
# Alignment
## Experiments
使用Qwen-2.5-Math-1.5B,使用数据集GSM8K，使用的奖励函数为r1_reward_fn：

```
其判定逻辑包含两个关键维度：
- 格式奖励 (Format Reward) ：通过正则匹配检查模型输出是否包含 </think> <answer> 和 </answer> 这一对关键标签。这是模型进行结构化思考的“物理标志”。
- 答案奖励 (Answer Reward) ：在通过格式校验后，函数会从 <answer> 标签中提取内容，并利用 SymPy 和 MathD 归一化逻辑与标准答案进行符号级或数值级的比对。
```

|                           | 学习率配置 (max) | 最高验证准确率 (Val Acc) |
| ------------------------- | ----------- | ----------------- |
| SFT                       | 1e-4        | 0.6641 (step 384) |
| Reinforce (no Baseline)   | 5e-5        | 0.5072 (Step 40)  |
| Reinforce (with Baseline) | 5e-5        | 0.5800 (Step 100) |
| GRPO (Standard)           | 5e-5        | 0.7672 (Step 70)  |
| GRPO (max_tokens norm)    | 5e-5        | 0.7703 (Step 140) |
| GRPO (Remove Std Norm)    | 5e-5        | 0.7309 (Step 100) |
| Dr.GRPO                   | 5e-5        | 0.7650 (Step 100) |




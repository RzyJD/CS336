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
└── README.md
```
# Pretraining
## Architecture

<img width="747" height="503" alt="image" src="https://github.com/user-attachments/assets/2d8f25a6-40d9-487e-8f30-a5b6854b5eab" />

## Experiments
最优学习率为8e-4
|                    | 学习率配置 (max) | 最小验证损失 (Val Loss)   | 最小训练损失 (Train Loss) |
| ------------------ | ----------- | ------------------- | ------------------- |
| Baseline (PreNorm) | 8e-4        | 1.3233 (Step 11900) | 1.1614 (Step 19900) |
| PostNorm           | 8e-4        | 1.3347 (Step 18100) | 1.1831 (Step 19900) |
| Remove RoPE        | 8e-4        | 1.3944 (Step 18100) | 1.2588 (Step 20000) |
| SiLU               | 8e-4        | 1.3640 (Step 19400) | 1.2173 (Step 19900) |
| Remove RMSNorm     | 8e-4        | -                   | -                   |
| Remove RMSNorm     | 3e-4        | 1.5351 (Step 18100) | 1.5458 (Step 19900) |

# Setup
## Environment
 run `pip install uv`/`brew install uv` to install uv.
 
## Download data
Download the TinyStories data and a subsample of OpenWebText

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
# Architecture
<img width="747" height="503" alt="image" src="https://github.com/user-attachments/assets/2d8f25a6-40d9-487e-8f30-a5b6854b5eab" />
## Training
参数初始化：
$$
\mathcal{N}\left(\mu=0, \sigma^2=\frac{2}{d_{\text {in }}+d_{\text {out }}}\right) \text { truncated at }[-3 \sigma, 3 \sigma] .
$$
使用AdamW
使用余弦退火
使用梯度裁剪
超参数配置：
python pretraining/TrainingScript.py \
    --lr 8e-4 \
    --batch_size 64 \
    --context_length 256 \
    --vocab_size 10000 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000.0 \
    --d_model 512 \
    --beta1 0.9 \
    --beta2 0.999 \
    --weight_decay 0.0001 \
    --warmup_iters 2000 \
    --cosine_cycle_iters 20000 \
    --max_l2_norm 1 \
## Experiments
### 学习率搜索
最优学习率8e-4，最小val loss 1.3233
<img width="1143" height="898" alt="image" src="https://github.com/user-attachments/assets/1e1f1392-c778-4136-8d80-e9fb2a7f2c67" />
### 对比实验
|                    | 学习率配置 (max) | 最小验证损失 (Val Loss)   | 最小训练损失 (Train Loss) |
| ------------------ | ----------- | ------------------- | ------------------- |
| Baseline (PreNorm) | 8e-4        | 1.3233 (Step 11900) | 1.1614 (Step 19900) |
| PostNorm           | 8e-4        | 1.3347 (Step 18100) | 1.1831 (Step 19900) |
| Remove RoPE        | 8e-4        | 1.3944 (Step 18100) | 1.2588 (Step 20000) |
| SiLU               | 8e-4        | 1.3640 (Step 19400) | 1.2173 (Step 19900) |
| Remove RMSNorm     | 8e-4        | -                   | -                   |
| Remove RMSNorm     | 3e-4        | 1.5351 (Step 18100) | 1.5458 (Step 19900) |
#### 去除RMSNorm
学习率为8e-4的时候梯度爆炸
<img width="1272" height="360" alt="image" src="https://github.com/user-attachments/assets/02d82e5b-d7bd-4e69-8119-4f4bf496386f" />
3e-4的时候不错
最小val loss: 1.5351
<img width="1426" height="366" alt="image" src="https://github.com/user-attachments/assets/e6249b32-db54-44f2-bcd4-ba684739c3ef" />
#### 使用postnorm
区别不大，梯度稍大 
最小val loss: 1.3233
<img width="1283" height="358" alt="image" src="https://github.com/user-attachments/assets/c946839b-967f-4a66-97aa-74060d1820be" />
#### 去除RoPE
最小val loss: 1.3944
<img width="1260" height="357" alt="image" src="https://github.com/user-attachments/assets/6b1b2d32-169e-4ab0-a174-04d9406e390e" />
#### 使用silu
最小val loss: 1.3944
<img width="1198" height="338" alt="image" src="https://github.com/user-attachments/assets/d2e600dd-1dc6-4556-8694-ca6a4697dc2a" />

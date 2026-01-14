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
![Pasted image 20251220154500.png](app://cc5041c08089917614e1609928d33da1421c/Users/renzeyu/Documents/Obsidian%20Vault/Pasted%20image%2020251220154500.png?1766216700089)
## Training and Experiments
参数初始化：
$$
\mathcal{N}\left(\mu=0, \sigma^2=\frac{2}{d_{\text {in }}+d_{\text {out }}}\right) \text { truncated at }[-3 \sigma, 3 \sigma] .
$$
使用AdamW
使用余弦退火
使用梯度裁剪
只搜索了学习率
最优学习率8e-4
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
![[Pasted image 20260114211206.png]]
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
# Introduction
- Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model. 
- Train a minimal language model on TinyStories
# Architecture

<img width="747" height="503" alt="image" src="https://github.com/user-attachments/assets/2d8f25a6-40d9-487e-8f30-a5b6854b5eab" />

 其中FFN使用SwiGLU，自注意力机制在推理阶段使用KV-Cache
# Training
## Data
- 数据集：TinyStories，训练集由2.12M个小故事组成
- 优点：词汇精简（1500 个基础词）：降低手动实现分词器的复杂度，无需处理生僻词与复杂语义映射，文本结构简单（2-3 段短故事、情节单一）
- 总共2717700个小故事，每个故事用<|endoftext|>隔开，示例：
```
u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>
```
故事长度分布：

<img width="666" height="415" alt="Pasted image 20260114224204" src="https://github.com/user-attachments/assets/496c9beb-7849-47ce-9f73-0f7fb6d3dd09" />

## 训练配置
参数初始化：

$$
\mathcal{N}\left(\mu=0, \sigma^2=\frac{2}{d_{\text {in }}+d_{\text {out }}}\right) \text { truncated at }[-3 \sigma, 3 \sigma] .
$$

使用训练的BPE-Tokenizer
使用AdamW
使用余弦退火
使用梯度裁剪
超参数配置：
```
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
```
## Experiments
### 学习率搜索
最优学习率8e-4，最小val loss 1.3233

<img width="1143" height="898" alt="image" src="https://github.com/user-attachments/assets/1e1f1392-c778-4136-8d80-e9fb2a7f2c67" />

### 消融实验
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
最小val loss: 1.3640

<img width="1198" height="338" alt="image" src="https://github.com/user-attachments/assets/d2e600dd-1dc6-4556-8694-ca6a4697dc2a" />

# Generating

设置max_tokens=256
```
Prompt:Once upon a time, there was a little boy named Sam
```
默认设置temperature=1,top-p=0.9
```
Output:
Once upon a time, there was a little boy named Sam. Sam had a glove that he loved very much. One day, Sam went to the park to play with his friends.
While they were playing, Sam's friend, Sam, said, "Hey Sam, do you want to try my big glove? I like to wear it!" Sam thought for a moment and said, "Yes, I do. Let's play a game to see who can stay up longer."
They played a game of tag, and Sam tried to remain obedient and get stuck first. Sam stayed still on the swings, but this time, Tim stayed on the grass. He showed Sam how to hold the glove tightly. Sam knew that they both had to be patient and help friends.
Sam stayed with Tim and played more games, and soon they were all very good at playing tag. They learned that sometimes, you need to try to settle down and try again, like Sam. The moral of the story is to be patient and not get stubborn.
```
调整温度和top-p，观察生成文本的差别
temperature=1.2, top-p=0.9
```
Output:
Once upon a time, there was a little boy named Sam. Sam was a cheerful boy who loved to smile and play with his toy blocks. One day, Sam found a big, red ball. He played with it for a long time.
Then, Sam wanted to make a spicy sauce for lunch. He needed to add water to it. He went to the kitchen and got a big cup. He added the hot sauce to the water and waited for it to be done.
The sun was hot, and Sam felt the heat on his face. He put the big, red ball outside and waited. When it was finally popped, Sam pulled the big, red ball inside and started to play with it. He was very happy that he could make his spicy sauce feel good inside.
```
出现了语义漂移。从玩球突然跳跃到做辣酱，说明高温度让模型选择了概率极低的词汇，导致上下文关联被噪声干扰，逻辑断。
temperature=0.5, top-p=0.9
```
Output:
Once upon a time, there was a little boy named Sam. Sam loved to play outside in the sun. One day, he saw a big, red ball in the sky. He wanted to play with it.
Sam asked his mom, "Can I play with the ball in the sky?" His mom said, "No, Sam. It is too far away. You can play with the ball in the sky."
Sam was sad, but he did not give up. He went outside and started to play. He threw the ball high in the sky. But then, the ball went too far and hit a tree. The ball fell down and hit the ground.
Sam was sad and scared. He ran to his mom and said, "Mom, I am sorry. I broke the ball." His mom hugged him and said, "It's okay, Sam. We can get a new ball to play with."
Sam learned that it is important to be careful when playing outside. He also learned that sometimes, things that seem bad can be good.
```
格式固定，具有逻辑，但是ball in the sky不符合现实，过于稳妥
temperature=1,top-p=1
```
Once upon a time, there was a little boy named Sam. Sam loved to play outside, but he was always careful when he played. Today, he was playing with his plant. He liked to pretend that his plant was a great bean plant.
Sam had a friend named Tim. Tim was a smart little boy. They liked to play together. Today, they wanted to build a big bean. They counted the beans as they went from little seeds. Sam was very happy.
They started a magic storm. The rain was big and scary. It made new plants and trees grow even more amazing. Tim and Sam were amazed. They knew that even a small bean can be challenges. So they played every day, and they were the best of friends.
```
突然出现大暴雨，转折较为突兀
temperature=1,top-p=0.3
 ```
Once upon a time, there was a little boy named Sam. Sam was very eager to go to the park. He wanted to play with his friends and have fun.
Sam's mom said, "Sam, you need to pack your toys and go to the park." Sam was excited to go to the park. He put his toys in a bag and got ready to go.
At the park, Sam saw his friend, Lily. They played on the swings and had a lot of fun. Sam was happy that he packed his toys and got to go to the park. He couldn't wait to go back and play with Lily again.
 ```
平铺直叙，无错误
 
结论：温度和top-p越高，生成内容越丰富多样，但极易出现逻辑漂移和胡言乱语；越低，文本越稳定严谨，但平铺直叙、单调乏味，且可能为了保守选择常见词语导致错误。

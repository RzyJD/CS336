# Experiment

| 参数名                         | 取值        |
| --------------------------- | --------- |
| n_grpo_steps                | 200       |
| log_interval                | 2         |
| val_interval                | 10        |
| epochs_per_rollout_batch    | 1         |
| loss_type                   | grpo_clip |
| advantage_eps               | 1e-6      |
| group_size                  | 8         |
| use_std_normalization       | true      |
| clip_range                  | 0.2       |
| sampling_temperature        | 1.0       |
| sampling_min_tokens         | 4         |
| sampling_max_tokens         | 1024      |
| top_p                       | 1.0       |
| rollout_batch_size          | 256       |
| train_batch_size            | 256       |
| gradient_accumulation_steps | 128       |
| learning_rate               | 5e-5      |
| max_l2_norm                 | 1.0       |
| beta1                       | 0.9       |
| beta2                       | 0.95      |
| weight_decay                | 0.0       |

只搜索学习率
1e-4,1e-5

<img width="352" height="331" alt="image" src="https://github.com/user-attachments/assets/33c53138-9d33-4ed5-91c8-83856249780f" />

1e-5欠拟合，1e-5崩了

尝试3e-5,5e-5,8e-5

<img width="361" height="337" alt="image" src="https://github.com/user-attachments/assets/76e36705-c483-416f-aa26-26c7fbf264ed" />

5e-5上限更高，回答更简洁

<img width="1026" height="332" alt="image" src="https://github.com/user-attachments/assets/039e74cb-e650-470f-89c1-492a73167522" />

## 对比实验
###与策略梯度模型的对比
不带减去baseline的策略梯度，减去baseline的策略梯度，GRPO进行对比
<img width="1214" height="969" alt="image" src="https://github.com/user-attachments/assets/2187a98d-5c49-4a53-a8cb-851c09def79d" />
<img width="756" height="453" alt="image" src="https://github.com/user-attachments/assets/2172b3ca-a7b0-4f0c-b6a4-b9d2cc30d4ec" />

不减baseline的策略梯度模型出现模型坍塌，陷入局部最优（回答长度骤降，gradnorm趋于0）
```
In Tate’s garden pond, there are 4 male guppies, 7 female guppies, 3 male goldfishes, and 5 female goldfishes.  He buys 2 male guppies, 1 female guppy, 2 male goldfishes, and 3 female goldfishes.  How many more female fishes than male fishes does Tate have?
Assistant: <think>
2025-12-20 12:56:37,673 - INFO - [sample 1] output= 7 + 5 - 2 - 3 = 7 </think> <answer> 7 </answer>
2025-12-20 12:56:37,673 - INFO - [sample 1] reward={'answer_reward': 0.0, 'format_reward': 1.0, 'reward': 0.0}
```

```
User: Arianna plants a garden that has 10 rows of flowers with 20 flowers in each row. Currently, only 4/5 of the planted flowers have bloomed. How many flowers in Arianna's garden have bloomed?
Assistant: <think>
2025-12-20 13:03:37,676 - INFO - [sample 0] output=4/5 of 200 = 160</think> <answer>160</answer>
2025-12-20 13:03:37,676 - INFO - [sample 0] reward={'answer_reward': 1.0, 'format_reward': 1.0, 'reward': 1.0}
```
只计算不推理
加上baseline不clip导致在100步左右出现格式遗忘，gradnorm此时也较大不够稳定，参数更新方向不稳定
也存在直接计算不写过程的情况
```
User: Matt wants cookies for a snack, but his dad wants him to eat healthily. He tells Matt he can have half as many cookies as the number of carrot sticks he eats, plus two extra for cleaning his room. If Matt wants to eat five cookies in total, how many carrot sticks does he have to eat?
Assistant: <think>
2025-12-20 16:31:39,404 - INFO - [sample 2] output= Let the number of carrot sticks Matt has to eat be $x$, then $\frac{1}{2}x+2=5$, so $x=6$. </think> <answer> 6 </answer>
2025-12-20 16:31:39,404 - INFO - [sample 2] reward={'answer_reward': 1.0, 'format_reward': 1.0, 'reward': 1.0}
```
也有比较完美的回答
```
User: Matt wants cookies for a snack, but his dad wants him to eat healthily. He tells Matt he can have half as many cookies as the number of carrot sticks he eats, plus two extra for cleaning his room. If Matt wants to eat five cookies in total, how many carrot sticks does he have to eat?
Assistant: <think>
2025-12-20 15:49:41,776 - INFO - [sample 2] output= Let the number of carrot sticks Matt has to eat be $x$. The number of cookies Matt can have is $\dfrac {x} {2}+2$. Since Matt wants 5 cookies, we have $\dfrac {x} {2}+2=5$. So, $x=6$. Therefore, Matt has to eat $\fbox{6}$ carrot sticks. </think> <answer> 6 </answer>
2025-12-20 15:49:41,776 - INFO - [sample 2] reward={'answer_reward': 1.0, 'format_reward': 1.0, 'reward': 1.0}
```
GRPO结果最好
```
User: Matt wants cookies for a snack, but his dad wants him to eat healthily. He tells Matt he can have half as many cookies as the number of carrot sticks he eats, plus two extra for cleaning his room. If Matt wants to eat five cookies in total, how many carrot sticks does he have to eat?
Assistant: <think>
2025-12-20 05:35:52,514 - INFO - [sample 2] output= First, let’s use C to express the number of carrot sticks Matt eats. According to his father's instruction, Matt can have C/2 + 2 cookies. Since Matt wants to eat five cookies, we can set up the equation C/2 + 2 = 5. Solving this equation, we get C/2 = 3, so C = 6. Therefore, Matt has to eat six carrot sticks.</think> <answer> 6 </answer>
2025-12-20 05:35:52,515 - INFO - [sample 2] reward={'answer_reward': 1.0, 'format_reward': 1.0, 'reward': 1.0}
```

### Dr.GRPO
GRPO的损失函数：

<img width="782" height="191" alt="image" src="https://github.com/user-attachments/assets/196147ad-b952-4590-be6c-ad9604b96aee" />

其中

<img width="397" height="117" alt="image" src="https://github.com/user-attachments/assets/222fea63-7ac8-4129-881c-284005438b59" />

Dr.GRPO在GRPO的基础上使用固定的常数（模型生成的最大tokens）去归一化每个token的loss而不是使用该句话的长度归一化loss

<img width="724" height="136" alt="image" src="https://github.com/user-attachments/assets/2cafc94a-ef6e-4bc2-b32d-54dd9cdade98" />

C为采样的时候生成的一句话的最大长度，在本次实验中设置为1024

对比GRPO和Dr.GRPO的模型性能

<img width="1215" height="974" alt="image" src="https://github.com/user-attachments/assets/891149d2-4006-4e00-8e77-69673b797e53" />

<img width="1219" height="341" alt="image" src="https://github.com/user-attachments/assets/fad8c04a-b2f6-46c7-9d9f-b36d31c85323" />

<img width="752" height="452" alt="image" src="https://github.com/user-attachments/assets/2b01e1c4-caef-433b-9d48-1ebcfbccda5a" />

稳定性：Dr.GRPO 明显好于GRPO，梯度时刻保持在极低水平 

准确率：上限相似，所有变体最终都达到了 0.7 - 0.8 左右的准确率。

所有变体均出现了塌陷的现象

Dr.GRPO的回答长度明显变长

显然Dr.GRPO训练稳定性增加是由max-tokens长度归一化引起的



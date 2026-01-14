# Experiment
## 超参数设置

'''
python train.py \
    --batch_size 128 \
    --gradient_accumulation_steps 2 \
    --max_iters 512 \
    --train_subset_size 0 \ #使用全量数据进行训练
    --seed 56 \
    --max_l2_norm 1.0 \
    --learning_rate 1e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 1e-4 \
    --warmup_iters 51.2 \
    --min_lr 1e-5 \
    --cosine_cycle_iters 512 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_tokens 1024 \
    --sample_size 10 \
    --normalize_constant 1.0
    ''' 
    
## 学习率搜索

首先进行大范围搜索：

<img width="840" height="340" alt="image" src="https://github.com/user-attachments/assets/91ac430e-dcd2-4724-a4ab-4bb873cfce38" />

在5e-5到5e-4之间搜索：

<img width="839" height="339" alt="image" src="https://github.com/user-attachments/assets/b939a55f-547a-4412-a70f-d99e6c1df0c2" />

1e-4的验证准确率最高，0.6641

## 分析
**格式**：

<img width="430" height="336" alt="image" src="https://github.com/user-attachments/assets/e79740c3-5057-4feb-ab91-a4a6219943bb" />

学习率过小会学不到格式，正常或者较大学习率均能使得模型迅速学到回答格式

**回答长度**

<img width="1244" height="335" alt="image" src="https://github.com/user-attachments/assets/494ab47f-7bd0-4080-b2f3-ad3dc97fcdc3" />

小学习率会导致回答长度整体偏长，而其余学习率则接近数据集的平均回答长度

**回答稳定性**

<img width="434" height="337" alt="image" src="https://github.com/user-attachments/assets/9fb085e2-52f0-41d9-abe1-00d5d0192a48" />

小学习率会导致模型预测稳定性下降，倾向于瞎猜

**训练稳定性**

<img width="433" height="336" alt="image" src="https://github.com/user-attachments/assets/4861f942-d389-4f2d-a06d-64cbb2cd5780" />

小学习率学不到东西，梯度一直保持高位，大学习率前期梯度不稳定，1e-4非常健康

**回答文本**

1e-4最好，回答比较清晰，长度合适

```
'prompt': 'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Sandy walked from Holloway Shores to Sun Valley Shores for 8 hours on a particular day. She also walked half as many hours on the second day as she walked on the first day, to Hollock valley shores. Calculate the total time Sandy took to walk in the two days in minutes.\nAssistant: <think>'

'output': 'On the second day, she walked half as many hours as she did on the first day, which is 8 * 0.5 = <<8*0.5=4>>4 hours.\nIn total, she walked for 8 + 4 = <<8+4=12>>12 hours.\nIn minutes, this is a total of 12 * 60 = <<12*60=720>>720 </think> <answer> 720 </answer>', 'correct_answer': '720', 'reward': {'answer_reward': 1.0, 'format_reward': 1.0, 'reward': 1.0}}]
```

如果学习率过小或者过大，模型都会重复输出

1e-3：

<img width="680" height="452" alt="image" src="https://github.com/user-attachments/assets/d2ae2144-05a9-4427-a1eb-7b45e32cade3" />

5e-5：

<img width="693" height="418" alt="image" src="https://github.com/user-attachments/assets/7cd5a815-884e-4abb-b1f1-e5300be16072" />

1e-5：

<img width="693" height="418" alt="image" src="https://github.com/user-attachments/assets/db9eb84d-6279-415f-8b2c-6fe9ac4619c9" />

在2e-4的输出中发现了识别错误
```
Prompt:In Tate’s garden pond, there are 4 male guppies, 7 female guppies, 3 male goldfishes, and 5 female goldfishes... How many more female fishes than male fishes does Tate have?

Output:
Tate has 7 female guppies + 5 female goldfishes = <<7+5=12>>12 female fishes.
He also has 3 male guppies + 2 male goldfishes = <<3+2=5>>5 male fishes.
Therefore, Tate has 12 - 5 = <<12-5=7>>7 more female fishes than male fishes. </think> <answer> 7 </answer>
```
3 male gold fished识别成2,4 male guppies 识别成3

## 子集训练对比实验

<img width="437" height="336" alt="image" src="https://github.com/user-attachments/assets/bf1b116d-170b-4365-a3e3-5b808c99189b" />

全量数据训练效果较好

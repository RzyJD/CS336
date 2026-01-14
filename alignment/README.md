# Setup
1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```
# Introduction
1.Conduct a Zero-shot prompting baseline for the GSM8K dataset of competition math problems

2.Conduct SFT on Qwen-Math-2.5-1.5B using CoT-format GSKM8K dataset

3.Conduct GRPO for improving reasoning performance with verified rewards.

# Data
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

- These problems take between 2 and 8 steps to solve.

- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.

- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
  
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues"

示例：
```python
{
    'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
}
```
**数据分布**:

<img width="263" height="95" alt="image" src="https://github.com/user-attachments/assets/44e89fe5-7605-4052-910d-fd3c0cb48816" />

<img width="469" height="534" alt="image" src="https://github.com/user-attachments/assets/addd97fb-74d3-4eac-acd1-a120d8b5fdd9" />

**数据处理**：

采用r1_zero prompt将问题打造成与模型对话的方式

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.  ↪→  ↪→  ↪→  ↪→  User: {question} Assistant: <think>
```

将{question}中的内容替换为数据集中的问题

prompt示例：

```
"prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAssistant: <think>",
```
同时对答案进行构造使得其符合奖励函数的解析格式：

```
"Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. </think> <answer> 18 </answer>"
```

## 奖励函数以及评估指标

采用稀疏奖励机制，通过三个维度对模型输出进行判定（所有奖励均为0/1赋值）：

'''
格式奖励（Format Reward）：利用正则表达式检测输出中是否完整包含</think>、<answer>和</answer>标签。满足格式要求计1分，否则计0分。
答案奖励（Answer Reward）：在<answer>标签内提取内容，利用SymPy库进行数学等价性校验，或利用MathD逻辑进行标准答案归一化比对。计算结果正确计1分，否则计0分。
综合奖励（Reward）：同时满足格式奖励和答案奖励。两者均满足计1分，任一不满足则计0分。
评估指标为验证集综合奖励准确率。
'''

评估指标为**验证集综合奖励准确率**

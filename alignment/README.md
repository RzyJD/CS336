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

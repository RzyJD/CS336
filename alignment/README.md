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

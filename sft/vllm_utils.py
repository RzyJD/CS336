import torch
from vllm import LLM
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# 1. 初始化 vLLM 的函数
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    初始化 vLLM 实例。
    记得要 import set_random_seed。
    """
    # ... 这里写你的代码 ...
    # 关键点：
    # 1. 设置随机种子
    vllm_set_random_seed(seed)
    # 2. 创建 world_size_patch (torch.distributed.get_world_size -> 1)
    world_size_patch=patch('torch.distributed.get_world_size',return_value=1)
    profiling_patch=patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",return_value=None)
    # 3. 创建 profiling_patch (vLLM的内存检查 -> None)
    # 4. 用 with 语句包裹 LLM 的初始化
    #enable_prefix_caching:前缀缓存，存储公共前缀（例如prompt中system部分的kv cache，提高推理速度
    #显存占用率
    with world_size_patch,profiling_patch:
        return LLM(model=model_id, device=device, dtype=torch.bfloat16, enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization)


# 2. 权重加载函数
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    把 policy 的权重加载到 llm 里。
    """
    # ... 这里写你的代码 ...
    # 关键点：
    # 1. 获取 policy.state_dict()
    # 2. 找到 llm 内部深处的 model 对象 (llm.llm_eng
    #注意要把state_dict先搬回cpu再载入！！！！
    state_dict={k:v.detach().to('cpu') for k, v in policy.state_dict().items()}
    llm_model= llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
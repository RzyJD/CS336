import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from vllm import SamplingParams
from vllm_utils import init_vllm, load_policy_into_vllm_instance
from reward import evaluate_vllm, accuracy

base_model_path = "/root/autodl-tmp/Qwen/Qwen2___5-Math-1___5B"
sft_checkpoint_path = "/root/autodl-tmp/a5/sft/sft_checkpoint/checkpoint_bs64_lr0.0001_nfull.pt"
val_path = "/root/autodl-tmp/a5/sft/dealed_val.jsonl"
reward_output_path = "/root/autodl-tmp/a5/shiyan/output.jsonl"
device = "cuda:0"
seed = 56

def load_sft_policy():
    policy = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    ckpt = torch.load(sft_checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "module" in ckpt:
        state = ckpt["module"]
    else:
        state = ckpt
    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    try:
        incompatible = policy.load_state_dict(state, strict=False)
    except RuntimeError:
        stripped = {}
        for k, v in state.items():
            if k.startswith("model."):
                stripped[k[len("model.") :]] = v
            elif k.startswith("module."):
                stripped[k[len("module.") :]] = v
            else:
                stripped[k] = v
        incompatible = policy.load_state_dict(stripped, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print("policy.load_state_dict missing_keys:", len(missing))
        print("policy.load_state_dict unexpected_keys:", len(unexpected))
    return policy

def main():
    vllm_model = init_vllm(base_model_path, device, seed, gpu_memory_utilization=0.9)
    vllm_inner_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
    base_state_dict = {
        k: v.detach().to("cpu").float().clone()
        for k, v in vllm_inner_model.state_dict().items()
    }
    policy = load_sft_policy()
    load_policy_into_vllm_instance(policy, vllm_model)
    sft_state_dict = {
        k: v.detach().to("cpu").float().clone()
        for k, v in vllm_inner_model.state_dict().items()
    }
    total_diff_sq = 0.0
    total_base_sq = 0.0
    for name, base_param in base_state_dict.items():
        if name not in sft_state_dict:
            continue
        sft_param = sft_state_dict[name]
        if base_param.shape != sft_param.shape:
            continue
        diff = sft_param - base_param
        diff_norm = torch.norm(diff).item()
        base_norm = torch.norm(base_param).item()
        total_diff_sq += diff_norm**2
        total_base_sq += base_norm**2
    total_diff_norm = total_diff_sq**0.5
    total_base_norm = total_base_sq**0.5
    global_rel_diff = total_diff_norm / (total_base_norm + 1e-12)
    print("=== vLLM 权重变化检查 ===")
    print(f"L2(ΔW) = {total_diff_norm:.6e}")
    print(f"L2(W_base) = {total_base_norm:.6e}")
    print(f"相对差异 L2(ΔW)/L2(W_base) = {global_rel_diff:.6e}")

    data = load_dataset("json", data_files={"test": val_path})["test"]
    prompts = data["prompt"]
    correct_answer = data["correct_answer"]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
    )

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=None,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        correct_answer=correct_answer,
        reward_path=reward_output_path,
    )

    output_with_reward = load_dataset("json", data_files={"test": reward_output_path})["test"]
    acc = accuracy(output_with_reward)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

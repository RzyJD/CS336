import vllm,json
from vllm import SamplingParams
import torch
from vllm_utils import init_vllm
from datasets import load_dataset
from drgrpo_grader import r1_zero_reward_fn
val_device=torch.device('cuda:1')
state_dict=torch.load('/root/autodl-tmp/a5/sft/sft_checkpoint/checkpoint_bs128_lr0.0001_nfull.pt')['model_state_dict']
model=init_vllm('/root/autodl-tmp/Qwen/Qwen2___5-Math-1___5B',val_device,seed=56,gpu_memory_utilization=0.85)
llm_model= model.llm_engine.model_executor.driver_worker.model_runner.model
state_dict_cpu={k:v.detach().to('cpu') for k,v in state_dict.items()}
llm_model.load_weights(state_dict_cpu.items())
data=load_dataset('json',data_files={'train':'/root/autodl-tmp/a5/sft/dealed_train.jsonl'})['train']
eval_sampling_params = SamplingParams(
temperature=1.0, top_p=1.0, max_tokens=1024
)
eval_sampling_params.stop = ["</answer>"] 
eval_sampling_params.include_stop_str_in_output = True
prompts=data['prompt']
outputs = model.generate(prompts, eval_sampling_params) 
filtered_data={}
with open('/root/autodl-tmp/a5/sft/sft_checkpoint/filtered_train.jsonl','w',encoding='utf-8') as f:
    pass
for i,output in enumerate(outputs): 
        prompt = output.prompt 
        generated_text = output.outputs[0].text
        reward=r1_zero_reward_fn(generated_text,data[i]['correct_answer'])
        if reward['reward']==1.0:
            with open('/root/autodl-tmp/a5/sft/sft_checkpoint/filtered_train.jsonl','a',encoding='utf-8') as f:
                f.write(json.dumps({"prompt":prompt,'ground_truth':generated_text,'correct_answer':data[i]['correct_answer']},ensure_ascii=False)+'\n')
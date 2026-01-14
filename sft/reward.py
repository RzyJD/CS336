from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from datasets import load_dataset
import json
from datasets import Dataset
def evaluate_vllm(vllm_model,reward_fn,prompts,eval_sampling_params,correct_answer):
    eval_sampling_params.stop = ["</answer>"] 
    eval_sampling_params.include_stop_str_in_output = True
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results=[]
    for i,output in enumerate(outputs): 
        prompt = output.prompt 
        generated_text = output.outputs[0].text
        reward=r1_zero_reward_fn(generated_text,correct_answer[i])
        result={"prompt":prompt,'output':generated_text,'correct_answer':correct_answer[i],'reward':reward}
        results.append(result)
    return Dataset.from_list(results)

def accuracy(output_with_reward):
    answer_reward=output_with_reward['reward']['answer_reward']
    format_reward=output_with_reward['reward']['format_reward']
    reward=output_with_reward['reward']['reward']
    answer_reward_acc=sum(answer_reward)/len(answer_reward)
    format_reward_acc=sum(format_reward)/len(format_reward)
    reward_acc=sum(reward)/len(reward)
    return answer_reward_acc,format_reward_acc,reward_acc

if __name__=='__main__':
    output_with_reward=load_dataset('json',data_files={'test':'/root/autodl-tmp/a5/sft/reward_output.jsonl'})['test']
    accuracy(output_with_reward)

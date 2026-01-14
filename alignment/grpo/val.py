import numpy as np 
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from datasets import Dataset
import json

def evaluate_vllm(vllm_model,reward_fn,prompts,eval_sampling_params,correct_answer,reward_path):
    outputs = vllm_model.generate(prompts, eval_sampling_params) 
    output_with_reward=[]
    for i,output in enumerate(outputs): 
        prompt = output.prompt 
        generated_text = output.outputs[0].text
        reward=reward_fn(generated_text,correct_answer[i])
        result={"prompt":prompt,'output':generated_text,'correct_answer':correct_answer[i],'reward':reward}
        output_with_reward.append(result)
    return Dataset.from_list(output_with_reward)
def accuracy(output_with_reward):
    answer_reward=output_with_reward['reward']['answer_reward']
    format_reward=output_with_reward['reward']['format_reward']
    total_reward=output_with_reward['reward']['reward']
    return sum(total_reward)/len(total_reward),sum(answer_reward)/len(answer_reward),sum(format_reward)/len(format_reward)
def validate(output,seed):
    sample = output.shuffle(seed=seed).select(range(3)).to_list()
    reward_acc,answer_reward_acc,format_reward_acc=accuracy(output)
    average_response_length=np.mean([len(answer) for answer in output['output']])
    correct_response_length=[len(response['output']) for response in output if int(response['reward']['reward'])==1 ]
    average_correct_response_length=np.mean(correct_response_length) if correct_response_length else 0
    incorrect_answer_correct_format_response_length=[len(response['output']) for response in output if int(response['reward']['reward'])==0 and int(response['reward']['format_reward'])==1 ]
    average_incorrect_answer_correct_format_response_length=np.mean(incorrect_answer_correct_format_response_length) if incorrect_answer_correct_format_response_length else 0
    average_incorrect_response_length=[len(response['output']) for response in output if int(response['reward']['format_reward'])==0 ]
    average_incorrect_response_length=np.mean(average_incorrect_response_length) if average_incorrect_response_length else 0

    return sample,reward_acc,answer_reward_acc,format_reward_acc,average_response_length,average_correct_response_length,average_incorrect_answer_correct_format_response_length,average_incorrect_response_length

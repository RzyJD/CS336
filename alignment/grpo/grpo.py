from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import random
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from vllm_utils import init_vllm,load_policy_into_vllm_instance
from grpopieces import get_group_responses,compute_group_normalized_rewards,tokenize_prompt_and_output,get_response_log_probs,grpo_microbatch_train_step
from val import accuracy,validate,evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn,question_only_reward_fn
from torch.nn.utils import clip_grad_norm_
from nn_utils import Cosineannealing,Savecheckpoint,Loadcheckpoint
import os 
import json
from vllm import SamplingParams
from operator import itemgetter
from datetime import datetime
import logging
import wandb
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def parse_args():
    
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--tokenizer", type=str, required=False)
    parser.add_argument("--dealed_train_data", type=str, required=False)
    parser.add_argument("--dealed_val_data", type=str, required=False)
    parser.add_argument("--n_grpo_steps", type=int, required=False)
    parser.add_argument("--learning_rate", type=float, required=False)
    parser.add_argument("--advantage_eps", type=float, required=False)
    parser.add_argument("--rollout_batch_size", type=int, required=False)
    parser.add_argument("--group_size", type=int, required=False)
    parser.add_argument("--sampling_temperature", type=float, required=False)
    parser.add_argument("--sampling_min_tokens", type=int, required=False)
    parser.add_argument("--sampling_max_tokens", type=int, required=False)
    parser.add_argument("--top_p", type=float, required=False)
    parser.add_argument("--epochs_per_rollout_batch", type=int, required=False)
    parser.add_argument("--train_batch_size", type=int, required=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=False)
    parser.add_argument("--gpu_memory_utilization", type=float, required=False)
    parser.add_argument("--clip_range", type=float, required=False)
    parser.add_argument("--max_l2_norm", type=float, required=False)
    parser.add_argument("--beta1", type=float, required=False)
    parser.add_argument("--beta2", type=float, required=False)
    parser.add_argument("--weight_decay", type=float, required=False)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        required=False,
    )
    parser.add_argument(
        "--use_std_normalization",
        type=bool,
        required=False,
    )
    parser.add_argument("--log_interval", type=int, required=False)
    parser.add_argument("--val_interval", type=int, required=False)
    parser.add_argument("--reward_path", type=str, required=False)
    parser.add_argument("--train_data", type=str, required=False)
    parser.add_argument("--val_data", type=str, required=False)
    parser.add_argument("--prompt_path", type=str, required=False)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--checkpoint_dir", type=str, required=False)
    parser.add_argument("--load_checkpoint", type=str, required=False)
    parser.add_argument("--length_normalization", type=str, required=False)
    parser.add_argument("--reward_fn", type=str, required=False, choices=["r1_reward_fn", "question_only_reward_fn"])
    return parser.parse_args()
def setup_logging(checkpoint_dir,args,loss_type):
    os.makedirs(checkpoint_dir, exist_ok=True)
    lr_str = f"{getattr(args, 'learning_rate', 0):g}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(checkpoint_dir, f"train_lr{lr_str}_{loss_type}_{timestamp}.log")
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
def main():
    args=parse_args()
    
    #载入configs
    config_path='/root/assignment5-alignment/cs336_alignment/grpo/config.json'
    #设定推理和训练设备
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_device=torch.device('cuda:0')
    #设定验证设备
    val_device=torch.device('cuda:1')
    #读取config文件，传入args
    if os.path.exists(config_path):
        with open(config_path,'r') as f:
            config=json.load(f)
        for key,value in config.items():
            if hasattr(args,key) and getattr(args,key) is None:
                setattr(args,key,value)
    seed=args.seed
    set_seed(seed)
    if args.use_std_normalization == False:
        run_name = f"rlhf_{args.loss_type}_lr_{args.learning_rate}_nostd"
    else:
        run_name = f"rlhf_{args.loss_type}_lr_{args.learning_rate}"
    
    if args.reward_fn == "r1_reward_fn":
        reward_fn = r1_zero_reward_fn
        run_name += "_r1"
    else:
        reward_fn = question_only_reward_fn
        run_name += "_qonly"

    if args.length_normalization=="max_tokens":
        run_name+=f"_max_tokens"
    run_name+=f'_{args.epochs_per_rollout_batch}'
    wandb.init(project="alignment-grpo", config=config, name=run_name)
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")
    train_device=torch.device('cuda:0')
    infer_device=torch.device('cuda:1')
    logger = setup_logging(args.checkpoint_dir, args,args.loss_type)
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, ( "train_batch_size must be divisible by gradient_accumulation_steps" )  
    assert args.rollout_batch_size % args.group_size == 0, ( "rollout_batch_size must be divisible by group_size" )  
    assert args.train_batch_size >= args.group_size, ( "train_batch_size must be greater than or equal to group_size" )
    logger.info("="*60)
    logger.info("训练配置汇总：")
    logger.info(f"1. 数据配置：每次从数据集中抽取{int(args.rollout_batch_size/args.group_size)}个prompt，生成{args.group_size}组回答，得到一个batch_size为{args.rollout_batch_size}的问答对，训练{args.epochs_per_rollout_batch}轮以后更新old_policy,梯度积累步数为{args.gradient_accumulation_steps}步")
    logger.info(f"2. 训练配置:迭代次数={args.n_grpo_steps}, 余弦退火配置：warmup_iters={args.n_grpo_steps/10}, cosine_cycle_iters={args.n_grpo_steps}, max_lr={args.learning_rate}, min_lr={args.learning_rate/10}.'AdamW配置'beta1={args.beta1},beta2={args.beta2}, weight_decay={args.weight_decay}, max_l2_norm={args.max_l2_norm}")
    logger.info(f"3. 奖励函数配置: 使用 {args.reward_fn}")
    logger.info(f"4. 日志/Checkpoint：checkpoint_dir={args.checkpoint_dir}")
    data=load_dataset('json',data_files={'train':args.dealed_train_data,'test':args.dealed_val_data})
    #一定要丢掉最后小于每轮抽的问题数的数据，不然rollout_response会小于rolloutbatchsize，在训练阶段通过accumulatesteps切片会出现空数据无法计
    train_data=data['train']
    logger.info(f'训练集加载完成！训练集大小为{len(train_data)}')
    val_data=data['test']
    logger.info(f'验证集加载完成! 验证集大小为{len(val_data)}')   
    logger.info(f"length_normalization={args.length_normalization}") 
    logger.info(f"use_std_normalization={args.use_std_normalization}")
    logger.info(f"epochs_per_rollout_batch={args.epochs_per_rollout_batch}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": train_device.index},
    )
    infermodel=init_vllm(args.model,infer_device,seed,args.gpu_memory_utilization)
    vllm_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.group_size,
)
    vllm_sampling_params.stop = ["</answer>"] 
    vllm_sampling_params.include_stop_str_in_output = True
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay)
    #iteration=Loadcheckpoint(args.load_checkpoint,model,optimizer)
    #logger.info(f'成功从{args.load_checkpoint}加载模型参数！,从第{iteration}步开始训练！')
    max_acc=0
    n_prompts_per_rollout_batch=int(args.rollout_batch_size/args.group_size)
    for step in range(0,args.n_grpo_steps+1):
        batch_index=random.sample(range(len(train_data)),n_prompts_per_rollout_batch)
        batch=train_data.select(batch_index)
        load_policy_into_vllm_instance(model,infermodel)
        prompts=batch['prompt']
        correct_answers=batch['correct_answer']
        group_outputs = infermodel.generate(prompts, vllm_sampling_params) 
        #先用旧模型计算整体数据的logprobs储存，在训练循环进行切片，如果在训练循环内反复加载statedict会破坏张量的计算图
        rollout_responses,repeated_ground_truths,repeated_prompts=get_group_responses(group_outputs,correct_answers)
        mini_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
        with torch.inference_mode():
            if args.loss_type=='grpo_clip': 
                #分batch计算每个batch 的log probs，因为id化要padding所有序列到最大长度，如果不分batch就会把所有的序列padding成整个数据集的最大长度、
                old_log_probs=[]
                for t in range(args.gradient_accumulation_steps):
                    B=tokenize_prompt_and_output(repeated_prompts[t*mini_batch_size:(t+1)*mini_batch_size],rollout_responses[t*mini_batch_size:(t+1)*mini_batch_size],tokenizer)
                    batch_old_input_ids=B['input_ids'].to(train_device)
                    batch_old_labels=B['labels'].to(train_device)
                    batch_old_log_probs=get_response_log_probs(model,batch_old_input_ids,batch_old_labels,False)['log_probs']
                    old_log_probs.append(batch_old_log_probs)
                cliprange=args.clip_range
            else:
                old_log_probs=None
                cliprange=None
        if args.loss_type=='no_baseline':
            _,raw_rewards,rewards_mean=compute_group_normalized_rewards(reward_fn,rollout_responses,repeated_ground_truths,args.group_size,args.advantage_eps,args.use_std_normalization)
            advantages=None
        else:
            advantages,raw_rewards,rewards_mean=compute_group_normalized_rewards(reward_fn,rollout_responses,repeated_ground_truths,args.group_size,args.advantage_eps,args.use_std_normalization)
        


        lr=Cosineannealing(step,args.learning_rate,args.learning_rate/10,args.n_grpo_steps/10,args.n_grpo_steps)
        if step % args.log_interval == 0:
            token_entropy=[]
            response_mask=[]
            reward_acc=rewards_mean['reward_acc']
            answer_reward_acc=rewards_mean['answer_reward_acc']
            format_reward_acc=rewards_mean['format_reward_acc']
            logger.info(f'第{step}步rollout统计（本batch均值）：mean_reward={reward_acc:.4f}，mean_answer_reward={answer_reward_acc:.4f}，mean_format_reward={format_reward_acc:.4f}')
            wandb.log({"train_step": step,"train/reward_acc": reward_acc,"train/answer_reward_acc": answer_reward_acc,"train/format_reward_acc": format_reward_acc})
        for params in optimizer.param_groups:
            params['lr']=lr
        
        for train_step in range(args.epochs_per_rollout_batch):
            loss=0
            for t in range(args.gradient_accumulation_steps):
                batch_rollout_responses=rollout_responses[t*mini_batch_size:(t+1)*mini_batch_size]
                batch_repeated_prompts=repeated_prompts[t*mini_batch_size:(t+1)*mini_batch_size]
                
                if args.loss_type == 'no_baseline':
                    batch_raw_rewards=raw_rewards[t*mini_batch_size:(t+1)*mini_batch_size].to(train_device)
                    batch_advantages=None
                else:
                    batch_raw_rewards=None
                    batch_advantages=advantages[t*mini_batch_size:(t+1)*mini_batch_size].to(train_device)
                A=tokenize_prompt_and_output(batch_repeated_prompts,batch_rollout_responses,tokenizer)
                batch_input_ids=A['input_ids'].to(train_device)
                batch_labels=A['labels'].to(train_device)
                batch_response_mask=A['response_mask'].to(train_device)
                #token entropy:[b,s]
                if step % args.log_interval == 0 and train_step ==0:
                    batch_policy_log_probs,batch_token_entropy=itemgetter('log_probs','token_entropy')(get_response_log_probs(model,batch_input_ids,batch_labels,True))
                    token_entropy.append(batch_token_entropy.flatten())
                    response_mask.append(batch_response_mask.flatten())
                else:
                    batch_policy_log_probs=get_response_log_probs(model,batch_input_ids,batch_labels,False)['log_probs']
                if args.loss_type == 'grpo_clip':
                    batch_old_log_probs=old_log_probs[t]
                else:
                    batch_old_log_probs=None
                grpo_microbatch_train_step(batch_policy_log_probs,batch_response_mask,args.gradient_accumulation_steps,args.loss_type,batch_raw_rewards,batch_advantages,
    batch_old_log_probs,cliprange,args.length_normalization,args.sampling_max_tokens
)[0]           

            grad_norm = clip_grad_norm_(model.parameters(),args.max_l2_norm)
            optimizer.step()
            optimizer.zero_grad()
            if step % args.log_interval == 0:
                if train_step==0: 
                    #用torch cat将列表中的tensor转化为大tensor转化为tensor
                    token_entropy=torch.cat(token_entropy)
                    response_mask=torch.cat(response_mask)
                    token_entropy_mask=token_entropy*response_mask
                    token_average_entropy=token_entropy_mask.sum()/response_mask.sum()
                    logger.info(f'第{step}步训练统计：response token平均熵（第0轮、更新前):{token_average_entropy.item():.4f}')
                    wandb.log({"train_step":step,'train/token_average_entropy':token_average_entropy.item()})
                grad_norm_value = grad_norm.item()
                logger.info(f"【训练迭代】 步数: {step:6d} | 第{train_step}轮训练 | grad_norm_value: {grad_norm_value}")
                wandb.log({"train_step":step,'train/grad_norm_value':grad_norm_value})
        if step % args.val_interval == 0:
            logger.info(f'【验证迭代】 步数: {step:6d}')
            load_policy_into_vllm_instance(model,infermodel)
            val_sampling_params = SamplingParams(temperature=args.sampling_temperature,top_p=args.top_p,max_tokens=args.sampling_max_tokens,min_tokens=args.sampling_min_tokens,n=1)
            val_sampling_params.stop = ["</answer>"] 
            val_sampling_params.include_stop_str_in_output = True
            output=evaluate_vllm(infermodel,reward_fn,val_data['prompt'],val_sampling_params,val_data['correct_answer'],args.reward_path)

            sample,reward_acc,answer_reward_acc,format_reward_acc,average_response_length,average_correct_response_length,average_incorrect_answer_correct_format_response_length,average_incorrect_response_length=validate(output,args.seed)

            logger.info(f"【验证迭代】 步数: {step:6d} | answer_reward_acc: {answer_reward_acc:.4f},format_reward_acc: {format_reward_acc:.4f},reward_acc: {reward_acc:.4f}")
            wandb.log({"train_step":step,'eval/answer_reward_acc':answer_reward_acc,'eval/format_reward_acc':format_reward_acc,'eval/reward_acc':reward_acc})
            logger.info(f"【生成样本】 平均回答长度: {average_response_length:.4f}")
            wandb.log({"train_step":step,'eval/average_response_length':average_response_length})
            logger.info(f"【生成样本】 正确回答平均长度: {average_correct_response_length:.4f}")
            wandb.log({"train_step":step,'eval/average_correct_response_length':average_correct_response_length})
            logger.info(f"【生成样本】 错误回答平均长度: {average_incorrect_response_length:.4f}")
            wandb.log({"train_step":step,'eval/average_incorrect_response_length':average_incorrect_response_length})
            logger.info(f"【生成样本】 错误回答正确格式平均长度: {average_incorrect_answer_correct_format_response_length:.4f}")
            wandb.log({"train_step":step,'eval/average_incorrect_answer_correct_format_response_length':average_incorrect_answer_correct_format_response_length})
            for i, j in enumerate(sample):
                logger.info(f"[sample {i}] prompt={j['prompt']}")
                logger.info(f"[sample {i}] output={j['output']}")
                logger.info(f"[sample {i}] reward={j['reward']}")
            if answer_reward_acc>=max_acc:
                max_acc=answer_reward_acc
                max_acc_steps=step
                # model_save_path=Savecheckpoint(args.learning_rate,model,optimizer,step,args.checkpoint_dir,args.loss_type,args.length_normalization,args.use_std_normalization,args.epochs_per_rollout_batch)
                # logger.info(f'验证准确率提升，保存模型至{model_save_path},当前最大准确率为 {max_acc:.4f}，在步数 {max_acc_steps:6d}')
            logger.info(f'当前最大准确率为 {max_acc:.4f}，在步数 {max_acc_steps:6d}')
    wandb.finish()        
if __name__=='__main__':
    main()

                


            






        

        

            
            
        





    
    

                
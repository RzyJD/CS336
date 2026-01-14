from datasets import load_dataset,load_from_disk
import json
import argparse
import os
import time
import random
from datetime import datetime
import logging
import wandb
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn 
from nn_utils import Cosineannealing,Savecheckpoint
from torch.nn.utils import clip_grad_norm_
from sftpieces import sft_microbatch_train_step,tokenize_prompt_and_output,get_response_log_probs,log_generations
from torch.utils.data import DataLoader
from datasets import load_from_disk
from reward import evaluate_vllm,accuracy
from vllm_utils import init_vllm,load_policy_into_vllm_instance
from drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")

    # --- Data & Model ---
    parser.add_argument("--train_data", type=str, required=False, default=None)
    parser.add_argument("--dealed_train_data", type=str, required=False, default=None)
    parser.add_argument("--model", type=str, required=False, default=None)
    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=False, default=None)
    parser.add_argument("--train_subset_size", type=int, required=False, default=None)

    # --- Training Hyperparams ---
    parser.add_argument("--batch_size", type=int, required=False, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=None)
    parser.add_argument("--max_iter", type=int, required=False, default=None)
    parser.add_argument("--normalize_constant", type=float, required=False, default=None)
    parser.add_argument("--return_token_entropy", action="store_true", required=False, default=None)

    # --- Optimizer ---

    parser.add_argument("--beta1", type=float, required=False, default=None)
    parser.add_argument("--beta2", type=float, required=False, default=None)
    parser.add_argument("--weight_decay", type=float, required=False, default=None)
    parser.add_argument("--max_l2_norm", type=float, required=False, default=None)

    # --- Scheduler ---
    parser.add_argument("--learning_rate", type=float, required=False, default=None)
    parser.add_argument("--warmup_iters", type=int, required=False, default=None)
    parser.add_argument("--cosine_cycle_iters", type=int, required=False, default=None)

    # --- System & Logging ---
    parser.add_argument("--device", type=str, required=False, default=None)
    parser.add_argument("--log_interval", type=int, required=False, default=None)
    parser.add_argument("--save_interval", type=int, required=False, default=None)
    parser.add_argument("--val_interval", type=int, required=False, default=None)
    parser.add_argument("--resume", type=str, required=False, default=None)
    parser.add_argument("--disable_log", action="store_true", required=False, default=None)
    parser.add_argument("--save_model", type=bool, required=False, default=None)
    #验证集相关参数
    parser.add_argument("--prompt_path", type=str, required=False, default=None)
    parser.add_argument("--val_data", type=str, required=False, default=None)
    parser.add_argument("--dealed_val_data", type=str, required=False, default=None)
    #推理模型相关参数
    parser.add_argument("--temperature", type=float, required=False, default=None)
    parser.add_argument("--top_p", type=float, required=False, default=None)
    parser.add_argument("--max_tokens", type=int, required=False, default=None)
    parser.add_argument("--max_iters",type=int,required=False,default=None)
    parser.add_argument("--sample_size",type=int,required=False,default=None)
    parser.add_argument("--log_generations",type=int,required=False,default=None)
    return parser.parse_args()

def setup_logging(checkpoint_dir,args):
    os.makedirs(checkpoint_dir, exist_ok=True)
    subset = getattr(args, 'train_subset_size', None)
    subset_str = 'full' if (subset is None or subset == 0) else str(subset)
    lr_str = f"{getattr(args, 'learning_rate', 0):g}"
    bs_str = str(getattr(args, 'batch_size', 'NA'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(checkpoint_dir, f"train_bs{bs_str}_lr{lr_str}_n{subset_str}_{timestamp}.log")
    log_level = logging.CRITICAL if args.disable_log else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    #实例化a
    seed=56
    set_seed(seed)
    args=parse_args()
    #载入configs
    config_path='/root/assignment5-alignment/cs336_alignment/config.json'
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
    logger = setup_logging(args.checkpoint_dir, args)
    # 打印所有参数
    logger.info("="*60)
    logger.info("训练配置汇总：")
    logger.info(f"1. 数据配置：train_data={args.train_data},"
            f"batch_size={args.batch_size}, train_subset_size={args.train_subset_size}")
    logger.info(f"2. 训练配置：max_iter={args.max_iters}, beta1={args.beta1}, "
            f"beta2={args.beta2}, weight_decay={args.weight_decay}, max_l2_norm={args.max_l2_norm}")
    logger.info(f"3. 日志/Checkpoint：train_device={train_device}, val_device={val_device},checkpoint_dir={args.checkpoint_dir}, "
            f"log_interval={args.log_interval}")
    run_name = f"sft_ds{config['train_subset_size']}_lr_{args.learning_rate}_bs_{args.batch_size}"
    wandb.init(project="alignment-sft", config=config, name=run_name)
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")
    data=load_dataset('json',data_files={'train':args.dealed_train_data,'test':args.dealed_val_data})
    train_data=data['train']
    val_data=data['test']
    #加载训练集
    if os.path.exists(args.dealed_train_data):
            logger.info(f'从{args.dealed_train_data}加载训练集')
            original_train_size = len(train_data)
            logger.info(f'训练集加载完成！训练集大小为{original_train_size}')
            if args.train_subset_size is not None and args.train_subset_size > 0:
                n = min(args.train_subset_size, original_train_size)
                train_data = train_data.shuffle(seed=seed).select(range(n))
                logger.info(f'训练集子集大小设为 {n}，原始大小为 {original_train_size}')
            else:
                logger.info(f'使用完整训练集，大小为 {original_train_size}')
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)
    train_loader=DataLoader(
        train_data,
        batch_size=int(args.batch_size/args.gradient_accumulation_steps),
        shuffle=True,
        generator=dl_generator,
    )
    warmup_iters=args.max_iters/10
    logger.info(f"学习率调度自适应：训练步数:{args.max_iters}, warmup_iters={warmup_iters}, cosine_cycle_iters={args.max_iters}, max_lr={args.learning_rate}, min_lr={args.learning_rate/10}")

    #由于是梯度积累，相当于是取一个大小为batch_size/accumulation_steps的minibatch计算loss并累加，等累加的数量达到batchsize的时候更新参数
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

#初始化推理模型
    infermodel=init_vllm(args.model,val_device,seed=seed)
#初始化训练模型     
    model=AutoModelForCausalLM.from_pretrained(args.model,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16,device_map={"": train_device.index})
    #开启梯度检查点
    model.gradient_checkpointing_enable() 
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,betas=(args.beta1,args.beta2),weight_decay=args.weight_decay)
    model.train()
    loss=0
    actual_steps=1
    eval_idx=0
    prev_vllm_state = None
    prev_train_state = None
    max_acc=0
    max_acc_steps=0
    while actual_steps<args.max_iters:
        for t,batch in enumerate(train_loader):
        #注意学习率的变化是以实际的steps来计算   
            lr=Cosineannealing(actual_steps,args.learning_rate,args.learning_rate/10,warmup_iters,args.max_iters)
            for params in optimizer.param_groups:
                params['lr']=lr
            A=tokenize_prompt_and_output(batch['prompt'],batch['ground_truth'],tokenizer)
            input_ids=A['input_ids'].to(train_device)
            labels=A['labels'].to(train_device)
            response_mask=A['response_mask'].to(train_device)
            log_probs=get_response_log_probs(model,input_ids,labels,args.return_token_entropy)['log_probs']
            # .item() 将 GPU Tensor 数据搬运到 CPU 并转为 Python float；
            # 这一步是为了打印日志，同时彻底切断与 GPU 显存的联系，防止变量累积导致 OOM
            loss+=sft_microbatch_train_step(log_probs,response_mask,args.gradient_accumulation_steps,args.normalize_constant)[0].item()
            if (t+1)% args.gradient_accumulation_steps==0:
                grad_norm = clip_grad_norm_(model.parameters(),args.max_l2_norm)
                grad_norm_value = grad_norm.item()
                optimizer.step()
                optimizer.zero_grad()
                if actual_steps % args.log_interval==0:
                    logger.info(f"【训练迭代】 步数: {actual_steps:6d} | 训练损失: {loss:.4f} | 当前学习率: {lr:.6f} | 梯度范数: {grad_norm_value:.4f} ｜ 是否截断: {int(grad_norm_value > args.max_l2_norm)}")
                    wandb.log({"train_step": actual_steps, "train/loss": loss, "train/lr": lr, "train/grad_norm": grad_norm_value, "train/clipped": int(grad_norm_value > args.max_l2_norm)})
                if actual_steps% args.val_interval==0:
                    load_policy_into_vllm_instance(model,infermodel)
                    eval_sampling_params = SamplingParams(
temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
)
                    output_with_reward=evaluate_vllm(infermodel,r1_zero_reward_fn,val_data['prompt'],eval_sampling_params,val_data['correct_answer'])
                    answer_reward_acc,format_reward_acc,reward_acc=accuracy(output_with_reward)
                    sample=output_with_reward.shuffle(seed=seed).select(range(args.sample_size))
                    logger.info(f"【验证迭代】 步数: {actual_steps:6d} | 答案准确率: {answer_reward_acc:.4f} | 格式准确率: {format_reward_acc:.4f} | 总准确率: {reward_acc:.4f}")
                    wandb.log({"train_step": actual_steps, "eval/answer_acc": answer_reward_acc, "eval/format_acc": format_reward_acc, "eval/reward_acc": reward_acc})
                    if answer_reward_acc>max_acc:
                        max_acc=answer_reward_acc
                        max_acc_steps=actual_steps
                        logger.info(f'验证集答案准确率提升至 {answer_reward_acc:.4f},保存模型')
                        Savecheckpoint(args.train_subset_size,args.batch_size,args.learning_rate,model,optimizer,actual_steps,args.checkpoint_dir,args,logger)
                    logger.info(f'当前最大答案准确率为 {max_acc:.4f}，在步数 {max_acc_steps:6d}')
                    eval_idx+=1

                    if eval_idx % args.log_generations==0:
                        logger.info(f"【验证迭代】 步数: {actual_steps:6d} | 抽取{args.sample_size}个样本")
                        model.eval()
                        with torch.no_grad():
                            log_generations(sample,train_device,model,tokenizer,logger,actual_steps)
                        model.train()
                actual_steps+=1
                if actual_steps > args.max_iters:
                    break
                loss=0
    wandb.finish()
if __name__=='__main__':
    main()

    

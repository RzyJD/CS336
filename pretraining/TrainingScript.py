# 1. 导入必要的库
import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
import logging
import random
from einops import rearrange, einsum
# 2. 导入自己的模型和工具（根据你的实际文件命名调整）
from Transformer_with_KV_Cache import Transformer_lm,Transformer_lm_without_RMSnorm,Transformer_lm_without_RoPE  # 你的Transformer模型
from Transformer_training_module import (  # 你的训练工具函数
    AdamW, Cross_entropy, Gradient_clipping,
    Dataloader, Savecheckpoint, Loadcheckpoint,
    Cosineannealing
)
import json
import wandb
import math
torch.set_float32_matmul_precision('high')
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Transformer Language Model Training")
    parser.add_argument("--train_data", type=str, required=False,default=None,help="训练数据路径（np.memmap格式）")
    parser.add_argument("--val_data", type=str, required=False,default=None, help="验证数据路径（np.memmap格式）")
    parser.add_argument("--batch_size",type=int,required=False,default=None)
    parser.add_argument("--context_length",type=int,required=False,default=None)
    
    # 模型参数（定义你的Transformer结构）transformer_lm中的参数
    parser.add_argument("--vocab_size",type=int,required=False,default=None)
    parser.add_argument("--num_layers",type=int,required=False,default=None)
    parser.add_argument("--num_heads",type=int,required=False,default=None)
    parser.add_argument("--d_ff",type=int,required=False,default=None)
    parser.add_argument("--rope_theta",type=float,required=False,default=None)
    parser.add_argument("--d_model",type=int,required=False,default=None)
    # 训练参数（控制训练过程）优化器，学习率调度，梯度裁剪，损失函数
    #优化器
    parser.add_argument("--beta1",type=float,required=False,default=None)
    parser.add_argument("--beta2",type=float,required=False,default=None)
    parser.add_argument("--weight_decay",type=float,required=False,default=None)
    
    #学习率
    parser.add_argument("--lr",type=float,required=False,default=None)
    parser.add_argument("--warmup_iters",type=float,required=False,default=None)
    parser.add_argument("--cosine_cycle_iters",type=float,required=False,default=None)
    #梯度裁剪
    parser.add_argument("--max_l2_norm",type=float,required=False,default=None)
     # 设备参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")   
    # 日志和Checkpoint参数（保存和监控训练）
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N iterations")
    parser.add_argument("--val_interval", type=int, default=1000, help="Validate every N iterations")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--max_iter", type=int, default=None,help="Max itering time")   
    parser.add_argument("--memmap",type=bool,default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_log", type=bool, default=False, 
                    help="是否一键关闭普通日志（仅保留严重错误日志）：True=关闭，False=开启")
    #对比试验
    parser.add_argument("--silu",type=bool,default=False)
    parser.add_argument("--postnorm",type=bool,default=False)
    parser.add_argument("--remove_rmsnorm",type=bool,default=False)
    parser.add_argument("--remove_rope",type=bool,default=False)
    
    return parser.parse_args()

def setup_logging(checkpoint_dir,args):
    """配置日志输出到控制台和文件"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_level=logging.CRITICAL if args.disable_log else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)

#读取大文件
def load_large_dataset(path,method):
    if method==True:
        return np.load(path,mmap_mode='r')
    else:
        return np.load(path)        
def validate(model,optimizer,val_dataset,args):
        model.eval()
        num_batches=5
        total_loss=0
        for _ in range(num_batches):
            x,y=Dataloader(val_dataset,args.batch_size,args.context_length,args.device)
            logits,_=model(x)
            logits=rearrange(logits,'b s v -> (b s) v')
            y=rearrange(y,'b s->(b s)')
            total_loss+=Cross_entropy(logits,y)
        total_loss=total_loss/num_batches
        model.train()
        return total_loss
def main():
    args=parse_args()
    config_path='config.json'
    # 解析命令行,载入日志
    if os.path.exists(config_path):
        with open(config_path,'r') as f:
            config=json.load(f)
        #将日志传入args
        for key,value in config.items():
            if hasattr(args,key) and getattr(args,key) is None:
                setattr(args,key,value)
    set_seed(56)
    run_name = f"pt_lr_{args.lr}"
    logger=setup_logging(args.checkpoint_dir,args)
    # 打印所有参数
    logger.info("="*60)
    logger.info("训练配置汇总：")
    logger.info(f"1. 数据配置：train_data={args.train_data}, val_data={args.val_data}, "
            f"batch_size={args.batch_size}, context_length={args.context_length}, memmap={args.memmap}")
    logger.info(f"2. 模型配置：vocab_size={args.vocab_size}, d_model={args.d_model}, "
            f"num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}, rope_theta={args.rope_theta}")
    logger.info(f"3. 训练配置：max_iter={args.max_iter}, lr={args.lr}, beta1={args.beta1}, "
            f"beta2={args.beta2}, weight_decay={args.weight_decay}, max_l2_norm={args.max_l2_norm}")
    logger.info(f"4. 调度配置：max_lr={args.lr}, min_lr={args.lr/10}, "
            f"warmup_iters={args.warmup_iters}, cosine_cycle_iters={args.cosine_cycle_iters}")
    logger.info(f"5. 日志/Checkpoint：device={args.device}, checkpoint_dir={args.checkpoint_dir}, "
            f"log_interval={args.log_interval}, val_interval={args.val_interval}, save_interval={args.save_interval}")
    logger.info(f"6. seed={56}")
    logger.info("="*60)
    # 加载数据集
    if os.path.exists(args.train_data):
        logger.info('加载训练集')
        train_dataset=load_large_dataset(args.train_data,args.memmap)
        logger.info(f'训练集加载完成！训练集大小为{len(train_dataset)}')
    if os.path.exists(args.val_data):
        logger.info('加载验证集')
        val_dataset=load_large_dataset(args.val_data,args.memmap)
        logger.info(f'验证集加载完成！验证集大小为{len(val_dataset)}')
    # 初始化模型

    if args.remove_rmsnorm == True:
        run_name+='_remove_rmsnorm'
        logger.info('RMSNorm移除对比实验')
        model=Transformer_lm_without_RMSnorm(vocab_size=args.vocab_size,context_length=args.context_length,d_model=args.d_model,num_layers=args.num_layers,num_heads=args.num_heads,d_ff=args.d_ff,rope_theta=args.rope_theta).to(args.device)
    elif args.remove_rope==True:
        run_name+='_remove_rope'
        logger.info('ROPE移除对比实验')
        model=Transformer_lm_without_RoPE(vocab_size=args.vocab_size,context_length=args.context_length,d_model=args.d_model,num_layers=args.num_layers,num_heads=args.num_heads,d_ff=args.d_ff).to(args.device)
    else:
        if args.postnorm==True:
            run_name+='_postnorm'
            logger.info('PostNorm对比实验')
        elif args.silu==True:
            run_name+='_swilu'
            logger.info('SiLU对比实验')
        model=Transformer_lm(vocab_size=args.vocab_size,context_length=args.context_length,d_model=args.d_model,num_layers=args.num_layers,num_heads=args.num_heads,d_ff=args.d_ff,rope_theta=args.rope_theta,swilu=args.silu,postnorm=args.postnorm).to(args.device)
    wandb.init(project="pretraining", config=config, name=run_name)
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")
    # 初始化优化器
    optimizer=AdamW(model.parameters(),args.lr,(args.beta1,args.beta2),args.weight_decay)
    # 训练循环
    t_start=0
    model.train()
    if args.resume is not None:
        logger.info('加载检查点')
        t_start=Loadcheckpoint(args.resume,model,optimizer)
        logger.info(f'检查点加载完成！从{t_start}步开始训练')
    min_loss=10000
    for t in range(t_start,args.max_iter+1): 
    #如果需要加载检查点，则从检查点开始训练
    # 1. 学习率调度
        lr=Cosineannealing(t,args.lr,args.lr/10,args.warmup_iters,args.cosine_cycle_iters)
        for params in optimizer.param_groups:
            params['lr']=lr   
    # 2. 加载训练数据
        x,y=Dataloader(train_dataset,args.batch_size,args.context_length,args.device)
    # 3. 前向传播计算损失
        optimizer.zero_grad()
        logits,_=model(x)
        logits=rearrange(logits,'b s v -> (b s) v')
        y=rearrange(y,'b v->(b v)')
        loss=Cross_entropy(logits,y)
    # 4.l 反向传播与参数更新
        loss.backward()
        l2_norm=Gradient_clipping(model.parameters(),args.max_l2_norm)
        if math.isnan(l2_norm) or math.isinf(l2_norm) or l2_norm==None:
            logger.info(f'梯度爆炸或消失，L2范数为{l2_norm}')
            l2_norm=100000
        optimizer.step()
    #日志 
        if t % args.log_interval==0:
            logger.info(f"【训练迭代】 步数: {t:6d} | 训练损失: {loss.item():.4f} | 当前学习率: {lr:.6f} | L2范数: {l2_norm:.4f}")
            wandb.log({"train_step": t,"train/loss":loss.item(),"train/lr":lr,"train/l2_norm":l2_norm})
        if t % args.val_interval==0:
            val_loss=validate(model,optimizer,val_dataset,args)
            logger.info(f"【验证结果】 步数: {t:6d} | 验证集平均损失: {val_loss.item():.4f}")     
            wandb.log({"train_step": t,"eval/loss":val_loss.item()})
            if val_loss<min_loss:
                min_loss=val_loss.item()
                min_loss_step=t
                checkpoint_path=os.path.join(args.checkpoint_dir,f'checkpoint_{run_name}')
                Savecheckpoint(model,optimizer,t,checkpoint_path)
                logger.info(f'验证集损失降低，保存模型至{checkpoint_path},当前最小损失为 {min_loss:.4f}，在步数 {min_loss_step:6d}')
if __name__ =="__main__":
    main()

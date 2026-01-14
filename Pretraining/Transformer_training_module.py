import torch
import torch.nn as nn 
import math
from einops import rearrange, einsum
from collections.abc import Callable, Iterable 
from typing import Optional
import numpy as np
def Cross_entropy(inputs,targets):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
 
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    #先计算所有inputs的softmax，再取对数会导致数值下溢
    inputs_max,_=torch.max(inputs,dim=-1,keepdim=True)
    log_sum=inputs_max+torch.log(torch.sum(torch.exp(inputs-inputs_max),dim=-1,keepdim=True))
    targets=rearrange(targets,'(batchsize one) -> batchsize one',one=1)#将targets中index的维度与input进行匹配
    #torch.gather(input, dim, index)
    #gather 函数会创建一个和 index 张量形状完全相同的新张量，新张量中的每个值都是根据 index 张量提供的坐标，沿着指定的维度 dim，从 input 张量中“捡”出来的。
    select_inputs=torch.gather(inputs,1,targets)
    select_probs=select_inputs-log_sum
    cross_entropy=-select_probs.mean()
    return cross_entropy   
class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr):
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}") 
        #定义优化器的默认超参数
        defaults = {"lr": lr} 
        #调用父类Optimizer的构造函数，传入参数和默认超参数
        super().__init__(params, defaults)     
     # 核心方法：执行一次参数更新（即优化步骤）    
    def step(self,closure=None):
        # 处理闭包函数：如果提供了closure，则调用它并获取损失值；否则损失为None
        loss = None if closure is None else closure()
        # 遍历所有参数组（param_groups是Optimizer基类中管理参数的结构，支持对不同参数设置不同超参数
        
        for group in self.param_groups:
            lr=group['lr']
            # 遍历当前参数组中的所有参数p（p是需要更新的模型参数，如权重、偏置等）
            for p in group['params']:
                if p.grad is None:
                    continue
                # 获取与参数p关联的状态字典（用于存储迭代次数等需要跨步骤保存的信息）
                state = self.state[p]
                # 从状态中获取迭代次数t：如果是第一次更新，默认t=0
                t = state.get("t", 0)
                # 获取参数p的梯度数据（.grad.data是梯度张量，存储了损失对p的偏导数）
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        # 返回损失值（用于监控训练过程）
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,betas,weight_decay,eps=1e-8):
        #存储超参数
        defaults={'lr':lr,'betas':betas,'weight_decay':weight_decay,'epsilon':eps}
        super().__init__(params,defaults)
    def step(self,closure=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            #获取超参数
            lr=group['lr']
            beta1,beta2=group['betas']
            weight_decay=group['weight_decay']
            epsilon=group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                state=self.state[p] #读取参数p的各种存储的状态，Adamw中包含两个beta         
                #第一次迭代之前，往状态字典存储需要存储的状态
                if len(state)==0:
                    #动量
                    state['step'] = 0
                    state['exp_avg']=torch.zeros_like(p)
                    #自适应学习率
                    state['exp_avg_sq']=torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step']+=1
                t=state['step']
                grad=p.grad.data
                exp_avg= beta1*exp_avg+(1-beta1)*grad
                exp_avg_sq=beta2*exp_avg_sq+(1-beta2)*(grad**2)
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                alpha=lr*(math.sqrt(1-beta2**t)/(1-beta1**t))
                #p.data=p.data-alpha*exp_avg/(torch.sqrt(exp_avg_sq)+epsilon)
                p.data.addcdiv_(exp_avg,torch.sqrt(exp_avg_sq)+epsilon,value=-alpha)

                p.data=p.data-lr*weight_decay*p.data             
def Cosineannealing(t,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if t<warmup_iters:
        lr=t/warmup_iters*max_learning_rate
    elif t>=warmup_iters and t<=cosine_cycle_iters:
        lr=min_learning_rate+0.5*(1+math.cos((t-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    elif t>cosine_cycle_iters:
        lr=min_learning_rate
    return lr
def Gradient_clipping(parameters,max_l2_norm,epsilon=1e-6):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # model.parameters() 返回一个可迭代对象，不是单个张量。
    # 必须通过循环来访问每个参数 p 和它的梯度 p.grad。
    # 每个参数
    l2=0
    
    for p in parameters:#参数可能是矩阵（比如Wq，Wk，Wv，不是单个单个存储的）
        with torch.no_grad():# 原地计算梯度，此操作不应被记录在计算图中
            if p.grad is not None:
                norm=p.grad.norm(2)
                l2+=norm**2
    l2_norm=torch.sqrt(l2)
    if l2_norm<=max_l2_norm:
        return l2_norm
    if l2_norm>max_l2_norm:
        clip=max_l2_norm/(l2_norm+epsilon)
        for p in parameters:
            if p.grad is not None:
                with torch.no_grad():
                    p.grad.mul_(clip)
        return l2_norm
    
        
def Dataloader(dataset,batch_size,context_length,device):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    #随机采样
    #将Data变成（问题，答案），对于长度为context_length+1的IDs，[0,context_length-1]为问题数据，[1,context_length]为答案数据
        #对于每一个批次（batch），你应该数据集中随机选择 batch_size 个起始位置，然后从这些位置开始提取长度为 context_length 的序列来构建你的 Question 和 Answer 张量。
        #随机获取batch_size个起始位置
    max_start=len(dataset)-context_length
    if max_start <= 0:
        raise ValueError(f"数据集长度({len(dataset)}) < context_length({context_length})")
    ix=torch.randint(0,len(dataset)-context_length,(batch_size,))
    #使用torch.stack进行叠加生成Q和V
    question=torch.stack([torch.tensor(dataset[i:i+context_length],dtype=torch.long) for i in ix] )
    
#torch.stack(Tensor,dim) Tensork可以是张量序列或者元组
    answer=torch.stack([torch.tensor(dataset[i+1:i+context_length+1],dtype=torch.long) for i in ix] )
    question,answer=question.to(device),answer.to(device)
    return question,answer
def Savecheckpoint(model,optimizer,iteration,out):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoints={
        'iteration':iteration,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }
    torch.save(checkpoints,out)
def Loadcheckpoint(
    src,
    model,
    optimizer
):
    # 1. 使用 torch.load() 加载整个检查点字典。
    #    我们加上 map_location='cpu' 是一个好习惯，
    #    这样即使检查点是在 GPU 上保存的，也能在 CPU 环境下成功加载。
    #    之后模型可以再手动移到 GPU。
    checkpoint = torch.load(src, map_location='cpu')
    # 2. 从字典中取出状态，并使用 .load_state_dict() 方法恢复它们。
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 3. 返回保存的迭代次数，以便调用者可以从正确的步数继续。
    return checkpoint['iteration']




    
    
          

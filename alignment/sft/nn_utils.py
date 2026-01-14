import torch
import numpy as np
import os

def Savecheckpoint(train_subset_size,batch_size,max_lr_dyn,model,optimizer,iteration,out,args,logger):
    checkpoint={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),    'iteration': iteration}
    subset_str = 'full' if (train_subset_size is None or train_subset_size == 0) else str(train_subset_size)
    fname = f"checkpoint_bs{batch_size}_lr{max_lr_dyn:g}_n{subset_str}.pt"
    model_save_path=os.path.join(out, fname)
    if getattr(args, 'save_model', False):
        torch.save(checkpoint,model_save_path)
        logger.info(f"模型权重保存至：{model_save_path}")
    else:
        logger.info(f"未保存模型!")
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
def Cosineannealing(t,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if t<warmup_iters:
        return (t/warmup_iters)*max_learning_rate
    if t>=warmup_iters and t<=cosine_cycle_iters:
        lr=min_learning_rate+0.5*(1+np.cos((t-warmup_iters)/(cosine_cycle_iters-warmup_iters)*np.pi))*(max_learning_rate-min_learning_rate)
        return lr
    if t>=cosine_cycle_iters:
        return min_learning_rate


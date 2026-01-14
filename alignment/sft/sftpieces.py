from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from einops import rearrange, einsum
import numpy as np 
import torch.nn.functional as F
def tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer):
    '''
    Tokenize the  prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).
    returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
    input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
    labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): shifted input ids, i.e., the input ids without the first token.
    response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) 1): a mask on the response tokens in the labels.
    '''
    #先tokenizer，再拼接，获取mask，再padding，再转tensor
    #不取最后一个token因为最后一个token没有label
    #记得转为tensor，转为tensor之前要将长度不统一的ids padding到最大长度
    #tokenizer
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False)
                  for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False)
                  for o in output_strs]
    #拼接input和output，获取最大的输入长度
    concat=[]
    mask=[]
    for i,o in zip(prompt_ids,output_ids):
        concat.append(i+o)
        mask.append([0 for _ in i]+[1 for _ in o])
    prompt_and_output_len=max(len(i) for i in concat)
    #获取padding的id
    pad_ids=tokenizer.pad_token_id
    #将短序列填补至最大长度，mask对应填补0至最大长度
    for i in range(len(concat)):
        concat[i]=concat[i]+[pad_ids]*(prompt_and_output_len-len(concat[i]))
        mask[i]=mask[i]+[0]*(prompt_and_output_len-len(mask[i]))
    concat=torch.tensor(concat)
#转化为tensor和切片，注意mask也要切，和response切法相同
    mask=torch.tensor(mask)
    mask=mask[...,1:]
    mask=mask.bool()
    input_ids=concat[...,:-1]
    output_ids=concat[...,1:]
    return {'input_ids':input_ids,'labels':output_ids,'response_mask':mask,'max_len':prompt_and_output_len}
def compute_entropy(logits):
    '''Args:  logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.  
    Returns:  torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.'''
    logits_max,_=torch.max(logits,dim=-1,keepdim=True)
    A=torch.exp(logits-logits_max)
    B=torch.sum(torch.exp(logits-logits_max),dim=-1,keepdim=True)
    C=logits-logits_max-torch.log(torch.sum(torch.exp(logits-   logits_max),dim=-1,keepdim=True))
    D=A*C/B
    #entropy:[b,s,v]
    entropy=torch.sum(D,dim=-1)
    return -entropy
def get_response_log_probs(model,input_ids,labels,return_token_entropy):
       """
    函数功能描述（可根据实际函数用途补充，例如：计算模型预测的对数概率和 token 熵）

    Args:
        model (PreTrainedModel): HuggingFace 模型，用于评分（需放置在正确的设备上；
            若不需要计算梯度，需设置为推理模式）。
        input_ids (torch.Tensor): 张量形状为 (batch_size, sequence_length)，
            由分词方法生成的「prompt + response」拼接后的 tokens。
        labels (torch.Tensor): 张量形状为 (batch_size, sequence_length)，
            由分词方法生成的标签。
        return_token_entropy (bool): 若为 True，会通过调用 compute_entropy 函数额外返回
            每个 token 的熵值。

    Returns:
        dict[str, torch.Tensor]: 包含以下键的字典：
            - "log_probs": 张量形状为 (batch_size, sequence_length)，表示条件对数概率
              $\log p_\theta\left(x_t \mid x_{<t}\right)$；
            - "token_entropy": 可选键，张量形状为 (batch_size, sequence_length)，
              每个位置的 token 熵值，仅当 return_token_entropy=True 时存在。
    """
       #output:[b,s,v]
       output0=model(input_ids).logits
       b,s=input_ids.shape
       #合并output和input符合crossentropy的输入格式[batch,v],[batch]
       output=rearrange(output0,'b s v -> (b s) v')
       labels=rearrange(labels,'b s -> (b s)')
       log_probs=F.cross_entropy(output,labels,reduction='none')#reduction设置为None返回的就是单独未求和的概率[b*s]
       #把logprobs格式展开为[b,s]
       log_probs=-rearrange(log_probs,'(b s)-> b s',b=b,s=s)
       if return_token_entropy==True:
            with torch.no_grad():
                token_entropy=compute_entropy(output0)
            return {'log_probs':log_probs,'token_entropy':token_entropy.detach()}
       else:
            return {'log_probs':log_probs}

def masked_normalize(tensor,mask,normalize_constant,dim=None):
        """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor (torch.Tensor): The tensor to sum and normalize.
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant (float): The constant to divide by for normalization.
        dim (int | None): The dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor
    """
        mask=mask.int()
        masked_tensor=mask*tensor
        if dim == None:
            return torch.sum(masked_tensor)/normalize_constant
        else:
            return torch.sum(masked_tensor,dim=dim)/normalize_constant
        

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log-probabilities
            from the SFT policy being trained.
        response_mask (torch.Tensor): Shape (batch_size, sequence_length), 1 for response tokens,
            0 for prompt/padding.
        gradient_accumulation_steps (int): Number of microbatches per optimizer step.
        normalize_constant (float, optional): The constant by which to divide the sum.
            It is fine to leave this as 1.0. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - Scalar tensor: The microbatch loss, adjusted for gradient accumulation.
              We return this so we can log it.
            - Metadata dict: Dict with metadata from the underlying loss call, and any other
              statistics you might want to log.

    Implementation tips:
        • You should call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    micro_loss=-masked_normalize(policy_log_probs,response_mask,normalize_constant)
    micro_loss/=gradient_accumulation_steps
    micro_loss=micro_loss/len(policy_log_probs)
    micro_loss.backward()
    return (micro_loss,{'metadata':None})
import wandb

def log_generations(sample,train_device,model,tokenizer,logger,actual_steps):
    logger.info(f'样本：{list(sample)}')
    answer_reward=sample['reward']['answer_reward']
    format_reward=sample['reward']['format_reward']
    reward=sample['reward']['reward']
    answer_reward_accuracy=sum(answer_reward)/len(answer_reward)
    format_reward_accuracy=sum(format_reward)/len(format_reward)
    reward_accuracy=sum(reward)/len(reward)
    logger.info(f"【生成样本】 答案奖励准确率: {answer_reward_accuracy:.4f} | 格式奖励准确率: {format_reward_accuracy:.4f} | 总奖励准确率: {reward_accuracy:.4f}")
    wandb.log({"train_step":actual_steps,'eval/answer_reward_accuracy':answer_reward_accuracy,'eval/format_reward_accuracy':format_reward_accuracy,'eval/reward_accuracy':reward_accuracy})
    average_response_length=np.mean([len(answer) for answer in sample['output']])
    logger.info(f"【生成样本】 平均回答长度: {average_response_length:.4f}")
    wandb.log({"train_step":actual_steps,'eval/average_response_length':average_response_length})
    correct_response_length=[len(response['output']) for response in sample if int(response['reward']['answer_reward'])==1 ]
    average_correct_response_length=np.mean(correct_response_length) if correct_response_length else 0
    logger.info(f"【生成样本】 正确回答平均长度: {average_correct_response_length:.4f}")
    wandb.log({"train_step":actual_steps,'eval/average_correct_response_length':average_correct_response_length})
    incorrect_response_length=[len(response['output']) for response in sample if int(response['reward']['answer_reward'])==0 ]
    average_incorrect_response_length=np.mean(incorrect_response_length) if incorrect_response_length else 0
    logger.info(f"【生成样本】 错误回答平均长度: {average_incorrect_response_length:.4f}")  
    wandb.log({"train_step":actual_steps,'eval/average_incorrect_response_length':average_incorrect_response_length})
    X=tokenize_prompt_and_output(list(sample['prompt']),list(sample['output']),tokenizer)
    input_ids=X['input_ids'].to(train_device)
    labels=X['labels'].to(train_device)
    response_mask=X['response_mask'].to(train_device)
    logits=model(input_ids).logits
    token_entropy=compute_entropy(logits)
    token_entropy_mask=token_entropy*response_mask
    #tensor求和转标量先占平再求和最后用item转标量
    token_average_entropy=token_entropy_mask.flatten().sum()/(response_mask.flatten().sum())
    logger.info(f"【生成样本】 平均token熵: {token_average_entropy.item():.4f}")
    wandb.log({"train_step":actual_steps,'eval/token_average_entropy':token_average_entropy.item()})
    
if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    prompt_strs=['胡稷藩','张恒国']
    output_strs=['儿子','曾孙']
    input_ids,labels,response_mask=tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer).values()
    log_probs=get_response_log_probs(model,input_ids,labels,return_token_entropy=False)['log_probs']
    print(log_probs)
    print(sft_microbatch_train_step(log_probs,response_mask,2,1.0))

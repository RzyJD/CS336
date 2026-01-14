import torch
import numpy as np 
from drgrpo_grader import r1_zero_reward_fn
import torch
from einops import rearrange
import torch.nn.functional as F
def get_group_responses(group_outputs,correct_answers):
    rollout_response=[]
    repeated_ground_truth=[]
    repeated_prompt=[]
    for i,outputs in enumerate(group_outputs):
        prompt=outputs.prompt 
        ground_truth=correct_answers[i]
        for output in outputs.outputs:
            generated_text = output.text
            rollout_response.append(generated_text)
            repeated_ground_truth.append(ground_truth)
            repeated_prompt.append(prompt)
    return rollout_response,repeated_ground_truth,repeated_prompt
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing a dict with keys
            "reward", "format_reward", and "answer_reward". 
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is rollout_batch_size =
            n_prompts_per_rollout_batch * group_size.
        排成一维列表方便以后处理
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is rollout_batch_size,
            because the ground truth for each example is repeated group_size times.
        group_size: int
            Number of responses per question (group).
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        normalize_by_std: bool
            If True, divide by the per-group standard deviation; otherwise subtract only the
            group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            - advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            - raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            - metadata: your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    
    一个rollout=一个trajectory=一句话
    采样了n_prompt_rollout_batch=个问题
    每个问题生成了group_size回答
    这批数据的batchsize=问题数✖️生成的回答个数
    思路：
    1.把responses和groundtruth用for循环输入到reward_fn中得到每个问题的回答
    2.计算每组的奖励平均值和方差
    3.计算每个回答的优势
    """
    #1.由于response和groundtruth同结构，直接用for循环输入到reward_fn得到每个问题的reward
    rewards=[]
    for response,truth in zip(rollout_responses,repeated_ground_truths):
        rewards.append(reward_fn(response,truth))
    answer_reward=[r['answer_reward'] for r in rewards]
    format_reward=[r['format_reward'] for r in rewards]
    reward=[r['reward'] for r in rewards]
    answer_reward_acc=torch.tensor(answer_reward).mean().item()
    format_reward_acc=torch.tensor(format_reward).mean().item()
    reward_acc=torch.tensor(reward).mean().item()                                        
    reward=torch.tensor(reward)
    #2.计算每组的奖励平均值和方差
    num_prompt=int(len(rollout_responses)/group_size) 
    reward=rearrange(reward,'(n g) -> n g',n=num_prompt,g=group_size)
    if normalize_by_std == True:
        reward_std,reward_mean=torch.std_mean(reward,dim=-1,keepdim=True)
        advantage=(reward-reward_mean)/(reward_std+advantage_eps)
        advantage=rearrange(advantage,'n g -> (n g)',n=num_prompt,g=group_size)
    else:
        reward_mean=torch.mean(reward,dim=-1,keepdim=True)
        advantage=reward-reward_mean
        advantage=rearrange(advantage,'n g -> (n g)',n=num_prompt,g=group_size)
    reward=rearrange(reward,'n g -> (n g)',n=num_prompt,g=group_size)
    return advantage,reward,{'reward_acc':reward_acc,'answer_reward_acc':answer_reward_acc,'format_reward_acc':format_reward_acc}

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages (torch.Tensor):
            Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs (torch.Tensor):
            Shape (batch_size, sequence_length), log-probs for each token.

    Returns:
        torch.Tensor:
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions
            in the training loop).
    """
    #改变raw_rewardsd的形状与log probs匹
    raw_rewards_or_advantages=rearrange(raw_rewards_or_advantages,'(b o)->b o',o=1)
    return -raw_rewards_or_advantages * policy_log_probs
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO clipped loss.

    Args:
        advantages (torch.Tensor):
            Shape (batch_size, 1), per-example advantages A.
        policy_log_probs (torch.Tensor):
            Shape (batch_size, sequence_length), per-token log probs from the
            policy being trained.
        old_log_probs (torch.Tensor):
            Shape (batch_size, sequence_length), per-token log probs from the
            old policy.
        cliprange (float):
            Clip parameter ε (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: torch.Tensor of shape (batch_size, sequence_length), the
              per-token clipped loss.
            - metadata: dict containing whatever you want to log. We suggest
              logging whether each token was clipped or not.
    """
    #注意把log转化为普通的概率
    #注意增加维度与logprobs维度的数量对齐
    advantages=rearrange(advantages,'(b o)->b o',o=1)
    ratio=torch.exp(policy_log_probs-old_log_probs)
    f=advantages*(ratio)
    g=torch.clamp(ratio,1-cliprange,1+cliprange)*advantages
    grpo_clip_loss=-torch.minimum(f,g)
    return (grpo_clip_loss,{'ratio':ratio})
import torch
from typing import Literal


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (torch.Tensor):
            Shape (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained.
        loss_type:
            One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards (torch.Tensor | None):
            Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages (torch.Tensor | None):
            Required for "reinforce_with_baseline" and "grpo_clip";
            shape (batch_size, 1).
        old_log_probs (torch.Tensor | None):
            Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange (float | None):
            Required for "grpo_clip"; scalar ε used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: (batch_size, sequence_length), per-token loss.
            - metadata: dict, statistics from the underlying routine
              (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type=='no_baseline':
        loss=compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs)
    elif loss_type=='reinforce_with_baseline':
        loss=compute_naive_policy_gradient_loss(advantages,policy_log_probs)
    else:
        loss=compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)[0]
    return (loss,{})
import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those
    elements where mask == 1.

    Args:
        tensor (torch.Tensor):
            The data to be averaged.
        mask (torch.Tensor):
            Same shape as tensor; positions with 1 are included in the mean.
        dim (int | None):
            Dimension over which to average. If None, compute the mean over all
            masked elements.

    Returns:
        torch.Tensor:
            The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor=mask*tensor
    if dim is not None:
        masked_mean=torch.sum(masked_tensor,dim=dim)/torch.sum(mask,dim=dim)
    else:
        masked_mean=torch.sum(masked_tensor)/torch.sum(mask)
    return masked_mean
import torch
from typing import Literal


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: Literal["masked_mean", "max_tokens"] = "masked_mean",
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (torch.Tensor):
            Shape (batch_size, sequence_length), per-token log-probabilities
            from the policy being trained.
        response_mask (torch.Tensor):
            Shape (batch_size, sequence_length), 1 for response tokens,
            0 for prompt/padding.
        gradient_accumulation_steps (int):
            Number of microbatches per optimizer step.
        loss_type (Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]):
            One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards (torch.Tensor | None):
            Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages (torch.Tensor | None):
            Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs (torch.Tensor | None):
            Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange (float | None):
            Clip parameter ε for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: scalar tensor. The microbatch loss, adjusted for gradient
              accumulation. We return this so we can log it.
            - metadata: dict with metadata from the underlying loss call, and
              any other statistics you might want to log.
    """
    loss_raw=compute_policy_gradient_loss(policy_log_probs,loss_type,raw_rewards,advantages,old_log_probs,cliprange)[0]
    if length_normalization=="masked_mean":
        loss=masked_mean(loss_raw,response_mask,dim=1)/gradient_accumulation_steps 
    else:
        loss=masked_normalize(loss_raw,response_mask,max_tokens,dim=1)/gradient_accumulation_steps 
    loss=torch.mean(loss)
    loss.backward()
    return (loss.detach().item(),{})    
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
            return {'log_probs':log_probs,'token_entropy':compute_entropy(output0).detach()}
       else:
            return {'log_probs':log_probs}
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
def masked_normalize(tensor,mask,normalize_constant,dim):
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
if __name__ == "__main__":

    # ========== 1. 先跑 compute_group_normalized_rewards ==========

    reward_fn = r1_zero_reward_fn
    group_size = 2
    advantage_eps = 1e-8
    normalize_by_std = True  # 或 False 都可以

    # 假设有 2 道题，每题 2 个回答 → rollout_batch_size = 4
    rollout_responses = [
        "<think>理由</think> <answer>\\boxed{42}</answer>",  # 题目1-回答1（正确）
        "<think>理由</think> <answer>\\boxed{0}</answer>",   # 题目1-回答2（错误）
        "<think>理由</think> <answer>\\boxed{7}</answer>",   # 题目2-回答1（正确）
        "<think>理由</think> <answer>\\boxed{-1}</answer>",  # 题目2-回答2（错误）
    ]
    repeated_ground_truths = [
        "\\boxed{42}", "\\boxed{42}",  # 题目1的 ground truth 重复 group_size 次
        "\\boxed{7}",  "\\boxed{7}",   # 题目2的 ground truth 重复 group_size 次
    ]

    advantages_flat, raw_rewards, meta_rewards = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )

    print("advantages_flat:", advantages_flat.shape, advantages_flat)
    print("raw_rewards:", raw_rewards.shape, raw_rewards)

    # ========== 2. 准备输入给 compute_grpo_clip_loss ==========

# ========== 2. 准备输入给 compute_grpo_clip_loss ==========

    batch_size = advantages_flat.shape[0]  # 4
    seq_len = 3



    policy_log_probs = torch.tensor(
        [
            [-0.05, -0.10, -0.51],
            [-0.04, -0.09, -0.92],
            [-0.03, -0.08, -0.69],
            [-0.06, -0.11, -0.60],
        ],
        dtype=torch.float32,
    )

    old_log_probs = torch.tensor(
        [
            [-0.05, -0.10, -0.41],
            [-0.04, -0.09, -0.81],
            [-0.03, -0.08, -0.80],
            [-0.06, -0.11, -0.51],
        ],
        dtype=torch.float32,
    )

    response_mask = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    cliprange = 0.2

    # ========== 3. 跑 compute_grpo_clip_loss ==========
    
    gradient_accumulation_steps=2  
    print(grpo_microbatch_train_step(policy_log_probs,response_mask,gradient_accumulation_steps,'grpo_clip',raw_rewards,advantages_flat,old_log_probs,cliprange))
        

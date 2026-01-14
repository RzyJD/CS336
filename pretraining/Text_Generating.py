
import torch
import torch.nn as nn 
import math
from einops import rearrange, einsum
from Get_tokenizer import Tokenizer
from Transformer_with_KV_Cache import Transformer_lm
from Transformer_training_module import Loadcheckpoint
import argparse
import os
import json
'''以用户提示为初始输入（x1...t）；
进入循环：
a. 模型预测下一个 token 的 logits，用温度缩放的 softmax 转为概率分布；
b. 用 top-p 采样从分布中选一个新 token（xt+1）；
c. 将新 token 拼接到输入序列中；
循环终止条件：生成结束符<|endoftext|>，或达到最大生成 token 数。'''

class Decoder:
    def __init__(self,tokenizer,model,end_token_id):
        self.model=model
        self.tokenizer=tokenizer
        self.end_token_id=end_token_id
        self.device='cuda'
    @staticmethod
    def Softmax(in_features,dim,tem=1):
        """
        Given a tensor of inputs, return the output of softmaxing the given `dim`
        of the input.

        Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is        arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

        Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
        """
    #先进行数值稳定性处理,每个数减去第一个维度的最大值防止大数导致e溢出
    #torch.max()会返回最大值和最大值所在的索引
        x_max,_=torch.max(in_features,dim=dim,keepdim=True)
        x_stable=in_features-x_max
        x_exp=torch.exp(x_stable/tem)
        x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
        x_softmax=x_exp/(x_sum+1e-9)
        return x_softmax
    @staticmethod
    def top_p_sample(probs,top_p):
        #probs:[batch,1,vocab_size]
        probs_val,probs_index=torch.sort(probs,dim=-1,descending=True) #torch.sort可以返回排序后的值和索引
        cum_probs=torch.cumsum(probs_val,dim=-1)
        sample=cum_probs>=top_p #返回的是布尔类型
        #
        ind=torch.argmax(sample.float())#返回最大值第一次出现的索引
        candidate_index=probs_index[:ind+1]
        candidate_val=probs_val[:ind+1]
        candidate_val_norm=candidate_val/candidate_val.sum()
        return candidate_index,candidate_val_norm
    def generate(self,prompt,top_p,temperature,max_ge_token):
        ids=torch.tensor(self.tokenizer.encode(prompt),dtype=torch.long)
        #第一次推理采用训练模式
        input_ids=ids
        output_ids=ids
        ids=rearrange(input_ids,'(b s) -> b s',b=1)
        #获取下一个tokens的logits
        self.model.eval()
    
        logits,kv_cache=self.model(ids)
        last=logits[0,-1,:]

        for _ in range(max_ge_token):
            probs_ini=self.Softmax(last,-1,temperature)
            candidate_index,candidate_val=self.top_p_sample(probs_ini,top_p)
            num=torch.rand(1)
            accu=torch.cumsum(candidate_val,dim=-1)
            accu_bool=accu>=num
            mask=accu_bool.float()
            idx=torch.argmax(mask)
            next_token=candidate_index[idx]
            if next_token==self.end_token_id:
                break
            output_ids=torch.cat([output_ids,next_token.unsqueeze(0)])
            input_ids=output_ids[-1]
            logits,kv_cache=self.model(input_ids.unsqueeze(0).unsqueeze(0),kv_cache)
            last=logits[0,-1]
        return self.tokenizer.decode(output_ids.tolist())
class Generator():
    @staticmethod  # 修正为静态方法
    def parse_args():
        parser = argparse.ArgumentParser(description="Generating")
        parser.add_argument("--context_length", type=int, required=False, default=None)   
        parser.add_argument("--num_layers", type=int, required=False, default=None)
        parser.add_argument("--num_heads", type=int, required=False, default=None)
        parser.add_argument("--d_ff", type=int, required=False, default=None)
        parser.add_argument("--rope_theta", type=float, required=False, default=None)
        parser.add_argument("--d_model", type=int, required=False, default=None)
        return parser.parse_args()
        

        
    def __init__(self, config_path, vocab_path, merge_path, checkpoints_path,special_tokens,end_token=None):
        self.args = self.parse_args()  # 解析参数
        self.config_path = config_path
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            for key, value in self.config.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)
        self.tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens)
        # 使用self.args引用参数
        
        self.model = Transformer_lm(
            len(self.tokenizer.vocab),
            self.args.context_length,
            self.args.d_model,
            self.args.num_layers,
            self.args.num_heads,
            self.args.d_ff,
            self.args.rope_theta
        )
        
        self.checkpoints=torch.load(checkpoints_path)
        self.model.load_state_dict(self.checkpoints['model_state_dict'])
        if end_token is not None:
            self.end_token_bytes=bytes(end_token,'utf-8')
            
            if self.end_token_bytes not in self.tokenizer.vocab_reverse:
                raise ValueError(f"({end_token}) 未在词汇表中找到。")
            else:
                end_token_id=self.tokenizer.vocab_reverse[self.end_token_bytes]
                print(end_token_id)
        else:
            end_token_id=None
        self.decoder = Decoder(self.tokenizer, self.model, end_token_id)  # 确保46对应正确的结束符id


    def Generating(self, prompt, top_p=0.9, temperature=1.2, max_ge_token=256):
        # 传递方法参数
        return self.decoder.generate(prompt, top_p=top_p, temperature=temperature, max_ge_token=max_ge_token)

# 实例化时确保特殊token与end_token_id匹配
generator = Generator('config.json', 'vocab_TinyStories.pkl', 'merges_TinyStories.pkl','/root/autodl-tmp/.autodl/pretraining/checkpoint/checkpoint_pt_lr_0.0008',['<|endoftext|>'],'<|endoftext|>')
temperature=[1.2,0.5]
top_p=[0.9,1,0.3]
for t in temperature:
    print(t,generator.Generating('Once upon a time, there was a little boy named Sam.',temperature=t))
for p in top_p:
    print(p,generator.Generating('Once upon a time, there was a little boy named Sam.',top_p=p))

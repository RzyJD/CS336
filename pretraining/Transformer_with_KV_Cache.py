import torch
import torch.nn as nn 
import math
from einops import rearrange, einsum
from collections.abc import Callable, Iterable 
from typing import Optional
class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        
        #调用父类的构造方法
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        #构造参数矩阵 
        W0=torch.empty(self.out_features,self.in_features,device=device,dtype=dtype)
        #参数初始化(初始化规则需要记住)
        self.sigma=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(W0,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        #赋值给Paramater
        self.W=nn.Parameter(W0)
        #定义前向传播过程，后面会自动调用
    def forward(self,x):
        return x @ self.W.T
class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        E0=torch.empty(self.num_embeddings,self.embedding_dim,device=device,dtype=dtype)
        self.sigma=math.sqrt(2/(num_embeddings+embedding_dim))
        nn.init.trunc_normal_(E0,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        self.E=nn.Parameter(E0)
    def forward(self,token_ids:torch.Tensor):
        return self.E[token_ids.long()]
class RMSnorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
            '''
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
            要求计算的时候将x变为float32，计算完成后再变回去'''

            super().__init__()
            self.d_model=d_model
            self.eps=eps
            #初始化g，初始化为1
            self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        def forward(self,x):
             #x:(batch_size, sequence_length, d_model)
             input_type=x.dtype
             x=x.to(torch.float32)
             #计算RMS，先平方，再沿着d_model求和，再算均值，再求倒数平方根
             RMS=torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)#保留d_model维度
             '''
             RMS：(batch_size,sequence_length,1)
             x:(batch_size,sequence_length,d_model)
             g:(d_model)
             *代表逐元素相乘
             RMS与x相乘时，会把RMS在最后一个维度复制d_model次，然后再逐元素相乘
             RMS*x:(batch_size,sequence_lenght,d_model)
             再与g相乘：由于前两个维度不存在，会把g复制成(batch_size,sequence_length,d_model)再逐元素相乘，相当于是把RMS*x的
             '''
             RMSnorm=x*RMS*self.g
             return RMSnorm.to(input_type)
class SwiGLU(nn.Module):
     def __init__(self,d_model:int,d_ff:int):
        super().__init__() 
        self.d_model=d_model
        self.d_ff=d_ff
        W_1=torch.empty(self.d_ff,self.d_model)
        W_3=torch.empty(self.d_ff,self.d_model)
        W_2=torch.empty(self.d_model,self.d_ff)
        self.sigma=math.sqrt(2/(d_ff+d_model))
        nn.init.trunc_normal_(W_1,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        nn.init.trunc_normal_(W_3,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        nn.init.trunc_normal_(W_2,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        self.W1=nn.Parameter(W_1)
        self.W3=nn.Parameter(W_3)
        self.W2=nn.Parameter(W_2)
     def forward(self,x):
        SiLU=(x@self.W1.T)*torch.sigmoid(x@self.W1.T)
        SwiGLU=(SiLU*(x@self.W3.T))@(self.W2.T)
        return SwiGLU
class SiLU(nn.Module):   
     def __init__(self,d_model:int,d_ff:int):
        super().__init__() 
        self.d_model=d_model
        self.d_ff=d_ff
        W_1=torch.empty(self.d_ff,self.d_model)
        W_2=torch.empty(self.d_model,self.d_ff)
        self.sigma=math.sqrt(2/(d_ff+d_model))
        nn.init.trunc_normal_(W_1,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)   
        nn.init.trunc_normal_(W_2,mean=0,std=self.sigma,a=-3*self.sigma,b=3*self.sigma)
        self.W1=nn.Parameter(W_1)
        self.W2=nn.Parameter(W_2)
     def forward(self,x):
        gate=x@self.W1.T
        activation=(gate)*torch.sigmoid(gate)
        SiLU=activation@(self.W2.T)
        return SiLU

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size,sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        super().__init__()
        self.d_k=d_k
        self.theta=theta
        self.max_seq_len=max_seq_len   
        #1.根据max_seq_len和d_k计算出每个位置的每个维度的旋转角  
        #预存储每个维度的初始旋转角度 
        #[2/d_k]
        d=torch.arange(0,self.d_k-1,2)
        theta_d=1/(self.theta ** (d/self.d_k))
        #创建位置索引
        #[max_seq_len]
        theta_p=torch.arange(max_seq_len)     
        #计算每个位置，每个维度的旋转角度，使用einsum
        #[max_seq_len,d_k/2]
        #字符串是输入和输出的维度
        
        angle=einsum(theta_p,theta_d,'i,j->i j')
        #2.将旋转角度转化为复数
        #https://docs.pytorch.ac.cn/docs/stable/generated/torch.ones_like.html
        #torch.polar(模长，旋转角度)使用极坐标创建复数
        #[max_seq_len,d_k/2]
        angle_c=torch.polar(torch.ones_like(angle),angle).to(device)
        #对于不需要训练的超参数，使用缓冲区保存
        self.register_buffer("angle_c_cached", angle_c, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        处理输入张量 x，应用 RoPE。此版本使用 einops.rearrange 提升代码可读性。

        Args:
        x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)。
        token_positions (torch.Tensor): x 中 token 的位置，形状为 (..., seq_len)。

        Returns:
        torch.Tensor: 应用 RoPE 后的张量，形状与输入 x 相同。
        """
        # 1. 根据 token_positions 从缓存中提取对应的旋转因子
        #token_positions:[batch_size,seq_len]
        #freqs_cis:[batch_size,seq_len,d_k/2]
        theta_x=self.angle_c_cached[token_positions]
        theta_x = theta_x.unsqueeze(1)

        # 2. 将输入张量 x 转换为复数形式 (使用 rearrange)
        #先将x的d_k维度拆分成[d_k/2,2],以符合复数转换的固定格式
        #使用
        #x_reshaped:[batch_size,seq_len,d_k/2,2]
        #x_complex:[batch_size,seq_len,d_k/2]
        #https://docs.pytorch.ac.cn/docs/stable/generated/torch.view_as_complex.html
        #https://blog.csdn.net/qq_50001789/article/details/136158442
        x_reshaped=rearrange(x,'... (d two)->... d two',two=2)
        x_complex=torch.view_as_complex(x_reshaped.contiguous())

        # 3. 执行核心操作：复数乘法
        x_rotation=x_complex*theta_x


        # 4. 将结果转换回实数并 reshape 成原始形状 (使用 rearrange)
        x_real=torch.view_as_real(x_rotation)
        x_out=rearrange(x_real,'... d two->... (d two)',two=2)
        return x_out
def Softmax(in_features,dim):
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
    x_exp=torch.exp(x_stable)
    x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    x_softmax=x_exp/x_sum
    return x_softmax
def Scaled_dot_product_attention(Q,K,V,mask):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    #X.shape[b]获取X的第b个维度的长度
    #1计算点积,除以根号dk注意要把int转化成tensor张量
    scaled_dot_product=einsum(Q,K,'... i k, ... j k -> ... i j')/torch.sqrt(torch.tensor(Q.shape[-1]))
    #2 mask
    if mask is None:
        softmax_scaled_dot_product=Softmax(scaled_dot_product,-1)
    else:
        mask_dot_product=torch.where(mask,scaled_dot_product,-torch.inf)
        #3对keys进行softmax
        softmax_scaled_dot_product=Softmax(mask_dot_product,-1)
    #4与v相乘
    attention=einsum(softmax_scaled_dot_product,V,'... i k, ... k j -> ... i j')
    #torch.where(condition,value_if_true,value_if_false)
    #https://docs.pytorch.ac.cn/docs/stable/generated/torch.where.html
    #注意是对keys进行softmax
    return attention
class Multihead_self_attention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=int(d_model/num_heads)
        self.d_v=int(d_model/num_heads)
        W_q=torch.empty(self.d_k*self.num_heads,self.d_model)
        W_k=torch.empty(self.d_k*self.num_heads,self.d_model)
        W_v=torch.empty(self.d_v*self.num_heads,self.d_model)
        W_o=torch.empty(self.d_model,self.d_v*self.num_heads)
        self.sigmaqk=math.sqrt(2/(self.d_k*self.num_heads+d_model))
        self.sigmav=math.sqrt(2/(self.d_v*self.num_heads+self.d_model))
        self.sigmao=math.sqrt(2/(self.d_model+self.d_v*self.num_heads))   
        nn.init.trunc_normal_(W_q,mean=0,std=self.sigmaqk,a=-3*self.sigmaqk,b=3*self.sigmaqk)
        nn.init.trunc_normal_(W_k,mean=0,std=self.sigmaqk,a=-3*self.sigmaqk,b=3*self.sigmaqk)
        nn.init.trunc_normal_(W_v,mean=0,std=self.sigmav,a=-3*self.sigmav,b=3*self.sigmav)
        nn.init.trunc_normal_(W_o,mean=0,std=self.sigmao,a=-3*self.sigmao,b=3*self.sigmao)
        self.W_q=nn.Parameter(W_q)
        self.W_k=nn.Parameter(W_k)
        self.W_v=nn.Parameter(W_v)
        self.W_o=nn.Parameter(W_o)
        #目标：使用一次矩阵乘法直接得到QKV
        #使用函数：torch.stack([A,B,C],dim=),torch.tile(A,())torch.chunk()
        #先拼接Wq，Wk，Wv
        #Wqkv:[3,d_k(d_v),d_,model]

    def forward(self,in_features,kv_cache):
        
        Wqkv=torch.stack([self.W_q,self.W_k,self.W_v],dim=0)
        qkv=einsum(Wqkv,in_features,'three hdk dmodel , ... numseq dmodel -> ... three numseq hdk')
        #多头拆单头
        qkv=rearrange(qkv,' ... three numseq (h dk) -> ... three h numseq dk',h=self.num_heads)
        #使用python的解包特性来获取q，k，v
        Q,K,V=rearrange(qkv,'... three h numseq dk -> three ... h numseq dk')

        if kv_cache is not None:
            K_past,V_past=kv_cache
            K_now=torch.cat([K_past,K],dim=-2)
            V_now=torch.cat([V_past,V],dim=-2)
            mask=None
        else:
                    #4.创建mask矩阵(下三角为True，上三角为False)
            #torch.tril(input, diagonal=0)将input的上三角部分设置为0
            seq_len=in_features.shape[-2]
            mask0=torch.ones((seq_len,seq_len),device=in_features.device) 
            mask0=torch.tril(mask0)
            mask=torch.where(mask0>0,True,False)
            #torch.triu(input, diagonal=0)
            #5.对每个头应用自注意力机制
            K_now=K
            V_now=V
        #6.将结果拼回[...,sequence_length,d_k]
        attention=Scaled_dot_product_attention(Q,K_now,V_now,mask)
        multihead=rearrange(attention,'... head numseq dk -> ... numseq (head dk)')
        #乘以缩放矩阵,注意缩放矩阵也要多头扩充维度
        multihead_attention=einsum(multihead,self.W_o,'... numseq hdv, dmodel hdv -> ... numseq dmodel')
        #存储新的KV
        kv_cache=(K_now,V_now)
        return multihead_attention,kv_cache
class Multihead_self_attention_with_rope(nn.Module):
    def __init__(self,d_model,num_heads,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=int(d_model/num_heads)
        self.d_v=int(d_model/num_heads)
        W_q=torch.empty(self.d_k*self.num_heads,self.d_model)
        W_k=torch.empty(self.d_k*self.num_heads,self.d_model)
        W_v=torch.empty(self.d_v*self.num_heads,self.d_model)
        W_o=torch.empty(self.d_model,self.d_v*self.num_heads)
        self.sigmaqk=math.sqrt(2/(self.d_k*self.num_heads+d_model))
        self.sigmav=math.sqrt(2/(self.d_v*self.num_heads+self.d_model))
        self.sigmao=math.sqrt(2/(self.d_model+self.d_v*self.num_heads))   
        nn.init.trunc_normal_(W_q,mean=0,std=self.sigmaqk,a=-3*self.sigmaqk,b=3*self.sigmaqk)
        nn.init.trunc_normal_(W_k,mean=0,std=self.sigmaqk,a=-3*self.sigmaqk,b=3*self.sigmaqk)
        nn.init.trunc_normal_(W_v,mean=0,std=self.sigmav,a=-3*self.sigmav,b=3*self.sigmav)
        nn.init.trunc_normal_(W_o,mean=0,std=self.sigmao,a=-3*self.sigmao,b=3*self.sigmao)
        self.W_q=nn.Parameter(W_q)
        self.W_k=nn.Parameter(W_k)
        self.W_v=nn.Parameter(W_v)
        self.W_o=nn.Parameter(W_o)
        #目标：使用一次矩阵乘法直接得到QKV
        #使用函数：torch.stack([A,B,C],dim=),torch.tile(A,())torch.chunk()
        #先拼接Wq，Wk，Wv
        #Wqkv:[3,d_k(d_v),d_,model]
        self.rope=RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len)

    def forward(self,in_features,token_positions,kv_cache):
        
        Wqkv=torch.stack([self.W_q,self.W_k,self.W_v],dim=0)
        qkv=einsum(Wqkv,in_features,'three hdk dmodel , ... numseq dmodel -> ... three numseq hdk')
        #多头拆单头
        qkv=rearrange(qkv,' ... three numseq (h dk) -> ... three h numseq dk',h=self.num_heads)
        #使用python的解包特性来获取q，k，v
        Q,K,V=rearrange(qkv,'... three h numseq dk -> three ... h numseq dk')
        #对QK使用rope

        Q_rope=self.rope(Q,token_positions)
        K_rope=self.rope(K,token_positions)

        if kv_cache is not None:
            K_past,V_past=kv_cache
            K_now=torch.cat([K_past,K_rope],dim=-2)
            V_now=torch.cat([V_past,V],dim=-2)
            mask=None
        else:
                    #4.创建mask矩阵(下三角为True，上三角为False)
            #torch.tril(input, diagonal=0)将input的上三角部分设置为0
            seq_len=in_features.shape[-2]
            mask0=torch.ones((seq_len,seq_len),device=in_features.device) 
            mask0=torch.tril(mask0)
            mask=torch.where(mask0>0,True,False)
            #torch.triu(input, diagonal=0)
            #5.对每个头应用自注意力机制
            K_now=K_rope
            V_now=V
        #6.将结果拼回[...,sequence_length,d_k]
        attention=Scaled_dot_product_attention(Q_rope,K_now,V_now,mask)
        multihead=rearrange(attention,'... head numseq dk -> ... numseq (head dk)')
        #乘以缩放矩阵,注意缩放矩阵也要多头扩充维度
        multihead_attention=einsum(multihead,self.W_o,'... numseq hdv, dmodel hdv -> ... numseq dmodel')
        #存储新的KV
        kv_cache=(K_now,V_now)
        return multihead_attention,kv_cache
class Transformer_block(nn.Module):
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """    
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.rmsnorm1=RMSnorm(self.d_model,eps=1e-5)
        self.rmsnorm2=RMSnorm(self.d_model,eps=1e-5)
        self.multiattention= Multihead_self_attention_with_rope(self.d_model,self.num_heads,self.theta,self.max_seq_len)

        self.ffn=SwiGLU(self.d_model,self.d_ff)
        

    def forward(self,in_features,kv_cache):

        batch_size=in_features.shape[-3]
        shape=in_features.shape[:-1]
        current_device=in_features.device
        if kv_cache is None:
        #使用广播法生成token位置序列
        #seq:[seq_len],in_features:[...,seqlen,d_model]
            seq_len=in_features.shape[-2]
            seq=torch.arange(seq_len,device=current_device)
            token_positions=torch.zeros(shape,dtype=torch.long,device=current_device)+seq #注意索引必须是整数，指定dtype
        else:
            #获取之前序列长度
            #K-cache:[...,batch,seq,dv]
            #in_features[b,1,d_model]
            #shape:[b,1]
            K_cache,_=kv_cache
            past_seq_len=K_cache.shape[-2]
            now_seq_len=in_features.shape[-2]
            token_positions=torch.arange(past_seq_len,past_seq_len+now_seq_len,dtype=torch.long,device=current_device)
            #解包参数传入
            token_positions=token_positions.view(*shape)
        in_features_norm1=self.rmsnorm1(in_features)
        output1_sublayer1,kv_cache=self.multiattention(in_features_norm1,token_positions,kv_cache)
        input_sublayer2=output1_sublayer1+in_features
        input_sublayer2_norm=self.rmsnorm2(input_sublayer2)
        output1_sublayer2=self.ffn(input_sublayer2_norm)
        output=output1_sublayer2+input_sublayer2
        return output,kv_cache

class Transformer_lm(nn.Module):

    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
        """    
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta,swilu=False,postnorm=False):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.max_sequence_length=4096
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.rope_theta=rope_theta
        self.postnorm=postnorm
        self.token_embedding=Embedding(self.vocab_size,self.d_model)
        if swilu == True:
            self.transform_blocks=nn.ModuleList([Transformer_block_SiLU(self.d_model,self.num_heads,self.d_ff,self.rope_theta,self.max_sequence_length) for layer in range(self.num_layers)])
        elif postnorm == True:
            self.transform_blocks=nn.ModuleList([Transformer_block_postnorm(self.d_model,self.num_heads,self.d_ff,self.rope_theta,self.max_sequence_length) for layer in range(self.num_layers)])
        else:
         self.transform_blocks=nn.ModuleList([Transformer_block(self.d_model,self.num_heads,self.d_ff,self.rope_theta,self.max_sequence_length) for layer in range(self.num_layers)])
        self.rmsnorm_final=RMSnorm(d_model,eps=1e-5)
        self.linear_final=Linear(d_model,vocab_size)

    def forward(self,in_indices,KV_cache=None):
        in_indices_trunk=in_indices[...,:self.context_length]
        x=self.token_embedding(in_indices_trunk)
        NEW_KV_cache=[]
        for i,layer in enumerate(self.transform_blocks):
            kv_cache_layer=KV_cache[i] if KV_cache is not None else None
            x,new_kv_cache_layer=layer(x,kv_cache_layer)
            NEW_KV_cache.append(new_kv_cache_layer)
        if not self.postnorm:
            x=self.rmsnorm_final(x)
        output=self.linear_final(x)
        return output,NEW_KV_cache
class Transformer_block_without_RMSnorm(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.multiattention= Multihead_self_attention_with_rope(self.d_model,self.num_heads,self.theta,self.max_seq_len)
        self.ffn=SwiGLU(self.d_model,self.d_ff)
        

    def forward(self,in_features,kv_cache):

        batch_size=in_features.shape[-3]
        shape=in_features.shape[:-1]
        current_device=in_features.device
        if kv_cache is None:
        #使用广播法生成token位置序列
        #seq:[seq_len],in_features:[...,seqlen,d_model]
            seq_len=in_features.shape[-2]
            seq=torch.arange(seq_len,device=current_device)
            token_positions=torch.zeros(shape,dtype=torch.long,device=current_device)+seq #注意索引必须是整数，指定dtype
        else:
            #获取之前序列长度
            #K-cache:[...,batch,seq,dv]
            #in_features[b,1,d_model]
            #shape:[b,1]
            K_cache,_=kv_cache
            past_seq_len=K_cache.shape[-2]
            now_seq_len=in_features.shape[-2]
            token_positions=torch.arange(past_seq_len,past_seq_len+now_seq_len,dtype=torch.long,device=current_device)
            #解包参数传入
            token_positions=token_positions.view(*shape)
        output1_sublayer1,kv_cache=self.multiattention(in_features,token_positions,kv_cache)
        input_sublayer2=output1_sublayer1+in_features
        output1_sublayer2=self.ffn(input_sublayer2)
        output=output1_sublayer2+input_sublayer2
        return output,kv_cache

class Transformer_lm_without_RMSnorm(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.max_sequence_length=4096
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.rope_theta=rope_theta
        self.token_embedding=Embedding(self.vocab_size,self.d_model)
        self.transform_blocks=nn.ModuleList([Transformer_block_without_RMSnorm(self.d_model,self.num_heads,self.d_ff,self.rope_theta,self.max_sequence_length) for layer in range(self.num_layers)])
        self.linear_final=Linear(d_model,vocab_size)

    def forward(self,in_indices,KV_cache=None):
        in_indices_trunk=in_indices[...,:self.context_length]
        x=self.token_embedding(in_indices_trunk)
        NEW_KV_cache=[]
        for i,layer in enumerate(self.transform_blocks):
            kv_cache_layer=KV_cache[i] if KV_cache is not None else None
            x,new_kv_cache_layer=layer(x,kv_cache_layer)
            NEW_KV_cache.append(new_kv_cache_layer)
        output=self.linear_final(x)
        return output,NEW_KV_cache

class Transformer_block_without_RoPE(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.rmsnorm1=RMSnorm(self.d_model,eps=1e-5)
        self.rmsnorm2=RMSnorm(self.d_model,eps=1e-5)
        self.multiattention= Multihead_self_attention(self.d_model,self.num_heads)
        self.ffn=SwiGLU(self.d_model,self.d_ff)
        

    def forward(self,in_features,kv_cache):
        batch_size=in_features.shape[-3]
        shape=in_features.shape[:-1]
        current_device=in_features.device
        in_features_norm1=self.rmsnorm1(in_features)
        output1_sublayer1,kv_cache=self.multiattention(in_features_norm1,kv_cache)
        input_sublayer2=output1_sublayer1+in_features
        input_sublayer2_norm=self.rmsnorm2(input_sublayer2)
        output1_sublayer2=self.ffn(input_sublayer2_norm)
        output=output1_sublayer2+input_sublayer2
        return output,kv_cache

class Transformer_lm_without_RoPE(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.max_sequence_length=4096
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.token_embedding=Embedding(self.vocab_size,self.d_model)
        self.transform_blocks=nn.ModuleList([Transformer_block_without_RoPE(self.d_model,self.num_heads,self.d_ff) for layer in range(self.num_layers)])

        self.rmsnorm_final=RMSnorm(d_model,eps=1e-5)
        self.linear_final=Linear(d_model,vocab_size)

    def forward(self,in_indices,KV_cache=None):
        in_indices_trunk=in_indices[...,:self.context_length]
        x=self.token_embedding(in_indices_trunk)
        NEW_KV_cache=[]
        for i,layer in enumerate(self.transform_blocks):
            kv_cache_layer=KV_cache[i] if KV_cache is not None else None
            x,new_kv_cache_layer=layer(x,kv_cache_layer)
            NEW_KV_cache.append(new_kv_cache_layer)
        x=self.rmsnorm_final(x)
        output=self.linear_final(x)
        return output,NEW_KV_cache
class Transformer_block_SiLU(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.rmsnorm1=RMSnorm(self.d_model,eps=1e-5)
        self.rmsnorm2=RMSnorm(self.d_model,eps=1e-5)
        self.multiattention= Multihead_self_attention_with_rope(self.d_model,self.num_heads,self.theta,self.max_seq_len)
        self.ffn=SiLU(self.d_model,self.d_ff)
    def forward(self,in_features,kv_cache):
        batch_size=in_features.shape[-3]
        shape=in_features.shape[:-1]
        current_device=in_features.device
        if kv_cache is None:
        #使用广播法生成token位置序列
        #seq:[seq_len],in_features:[...,seqlen,d_model]
            seq_len=in_features.shape[-2]
            seq=torch.arange(seq_len,device=current_device)
            token_positions=torch.zeros(shape,dtype=torch.long,device=current_device)+seq #注意索引必须是整数，指定dtype
        else:
            #获取之前序列长度
            #K-cache:[...,batch,seq,dv]
            #in_features[b,1,d_model]
            #shape:[b,1]
            K_cache,_=kv_cache
            past_seq_len=K_cache.shape[-2]
            now_seq_len=in_features.shape[-2]
            token_positions=torch.arange(past_seq_len,past_seq_len+now_seq_len,dtype=torch.long,device=current_device)
            #解包参数传入
            token_positions=token_positions.view(*shape)
        in_features_norm1=self.rmsnorm1(in_features)
        output1_sublayer1,kv_cache=self.multiattention(in_features_norm1,token_positions,kv_cache)
        input_sublayer2=output1_sublayer1+in_features
        input_sublayer2_norm=self.rmsnorm2(input_sublayer2)
        output1_sublayer2=self.ffn(input_sublayer2_norm)
        output=output1_sublayer2+input_sublayer2
        return output,kv_cache
class Transformer_block_postnorm(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.d_k=d_model/num_heads
        self.d_v=d_model/num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.rmsnorm1=RMSnorm(self.d_model,eps=1e-5)
        self.rmsnorm2=RMSnorm(self.d_model,eps=1e-5)
        self.multiattention= Multihead_self_attention_with_rope(self.d_model,self.num_heads,self.theta,self.max_seq_len)
        self.ffn=SwiGLU(self.d_model,self.d_ff)
        

    def forward(self,in_features,kv_cache):

        batch_size=in_features.shape[-3]
        shape=in_features.shape[:-1]
        current_device=in_features.device
        if kv_cache is None:
        #使用广播法生成token位置序列
        #seq:[seq_len],in_features:[...,seqlen,d_model]
            seq_len=in_features.shape[-2]
            seq=torch.arange(seq_len,device=current_device)
            token_positions=torch.zeros(shape,dtype=torch.long,device=current_device)+seq #注意索引必须是整数，指定dtype
        else:
            #获取之前序列长度
            #K-cache:[...,batch,seq,dv]
            #in_features[b,1,d_model]
            #shape:[b,1]
            K_cache,_=kv_cache
            past_seq_len=K_cache.shape[-2]
            now_seq_len=in_features.shape[-2]
            token_positions=torch.arange(past_seq_len,past_seq_len+now_seq_len,dtype=torch.long,device=current_device)
            #解包参数传入
            token_positions=token_positions.view(*shape)
        output1_sublayer1,kv_cache=self.multiattention(in_features,token_positions,kv_cache)
        input_sublayer2=self.rmsnorm1(output1_sublayer1+in_features)
        output1_sublayer2=self.ffn(input_sublayer2)
        output=self.rmsnorm2(output1_sublayer2+input_sublayer2)
        return output,kv_cache




    







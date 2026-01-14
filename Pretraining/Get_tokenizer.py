import os
import collections
from typing import List, Tuple, Dict, Set
import json
import regex    
from collections import defaultdict
import pickle
from BPE_Training import run_train_bpe1
# --- 新增的 imports ---
import numpy as np  # 您之前只在 main 块中导入了它
from multiprocessing import Pool # 导入工作池
import os # 导入以获取CPU核心数
import math # 导入以辅助分块计算
from tqdm import tqdm # 这是一个强烈推荐的进度条库 (如果未安装，请 pip install tqdm)
# --- 您的 BPE_Training import ---
from BPE_Training import run_train_bpe1
class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        #注意special_token为None的时候一定要设置空列表，不然后面map无法处理
        #为确保最长匹配，在构建正则表达式前对特殊符号按长度降序排序，这能强制引擎优先尝试匹配更长的token，防止其被短的token错误地提前截断
        ## 例如，为防止特殊符号 "<|username|>" 被 "<|user|>" 错误地提前匹配，排序后能确保前者优先，不排序的话匹配后username会被直接切断。
        self.special_tokens=sorted(special_tokens,reverse=True) if special_tokens is not None else []
        self.merge_dict = {pair: rank for rank, pair in enumerate(self.merges)}
        # 创建一个反向词汇表，用于最后从 `bytes` 快速查找到 `int` ID。
        self.vocab_reverse = {b: i for i, b in self.vocab.items()}
    
    @classmethod 
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab=pickle.load(f)
    
        # 保存合并操作记录到文件 (使用 pickle)
        with open(merges_filepath, "rb") as f:
            merges=pickle.load(f)
        return cls(vocab,merges,special_tokens)#会将括号内的传回给init参数
    @staticmethod #工具函数使用静态方法，调用的时候要加self
    def merge_encode(word_bytes: list[bytes], merge_pair: tuple[bytes, bytes]) -> list[bytes]:
        """
    辅助函数：在一个由字节组成的列表（代表一个单词）中，执行一次合并操作。

    它会遍历整个列表，将所有出现的 `merge_pair` 替换为合并后的新字节。
    这个函数每次只合并一种指定的 `pair`。

    Args:
        word_bytes (list[bytes]): 代表一个单词的、由单字节或多字节组成的列表。
                                  例如 [b'l', b'o', b'w']。
        merge_pair (tuple[bytes, bytes]): 需要被合并的字节对。例如 (b'l', b'o')。

    Returns:
        list[bytes]: 返回一个新的列表，其中指定的 `merge_pair` 已被合并。
                     例如，输入 ([b't', b'i', b't', b'i'], (b't', b'i')) -> [b'ti', b'ti']。
        """
        # 将要合并的字节对拼接成一个新的、更长的字节
        token_merge = merge_pair[0] + merge_pair[1]
    
        # 初始化一个新列表，用于存放合并后的结果，遵循“不可变”思想（不修改原始列表）
        new_word_bytes = []
        i = 0
        while i < len(word_bytes):
            # 边界检查：确保 i 不是最后一个元素的索引，这样 word_bytes[i+1] 才不会越界
            # 逻辑上等同于 `i < len(word_bytes) - 1`
            if i != len(word_bytes) - 1 and (word_bytes[i], word_bytes[i+1]) == merge_pair:
                # 如果当前对匹配，则将合并后的新字节添加到结果中
                new_word_bytes.append(token_merge)
                # 指针前进2位，因为我们消耗了 word_bytes 中的两个元素
                i += 2
            else:
                # 如果当前对不匹配，或者已经到了列表末尾，则只将当前元素加入结果
                new_word_bytes.append(word_bytes[i])
                # 指针前进一步
                i += 1
            
        return new_word_bytes

    def encode(self, text: str):
        #总体思路：根据special token将整个text分成小文本块串行处理，在分割后的文本块中根据一定的规则（正则表达式）将长文本分割成单词，在单词内部进行合并转化成id

        # --- 构建用于分割文本并保留特殊字符的正则表达式 ---
        # 1. `map(regex.escape, ...)`: 对所有特殊token进行转义，确保它们只被当作普通文本进行匹配（例如，'|'变成'\|'）。
        # 2. `'|'.join(...)`: 用'|'(逻辑或)将转义后的token连接起来，形成一个“匹配任意一个token”的组合模式。
        # 3. `f"({pattern})"`: 在最外层加上括号，创建为一个捕获组。
        #    这是最关键的一步，它告诉 `regex.split` 在分割时，要将匹配到的分隔符（即特殊token）本身也保留在结果中。

        #注意如果special_tokens如果为空则pattern为['']，而‘’存在于text的任何位置，每个字符都匹配，所以chunks会把text拆成单个字符，如果text没有speicaltoekn则把整个text看成文本快
        if self.special_tokens:
            pattern='|'.join(map(regex.escape,self.special_tokens))
            pattern_capture=f'({pattern})'
            chunks=regex.split(pattern_capture,text)
        else:
            chunks=[text]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_tokenize=[]

        for chunk in chunks:
            repr(chunk)
            #如果chunk本身就是speicaltoken 则直接在词表查找后转化成id
            if chunk in self.special_tokens:
                special=chunk.encode('utf-8')
                text_tokenize.append(self.vocab_reverse[special])                
            else:
                indices=regex.findall(PAT, chunk)
                #indices：list[words]
                # 对每一个“单词”（现在是 bytes 对象）进行独立的BPE合并处理
                for word in indices:
                    if not word:  # 如果预分词结果是空字符串，则跳过
                        continue
                    else:    
                # 将一个单词的字节流（例如 b'low'）分解成由单个字节组成的列表（例如 [b'l', b'o', b'w']）
                #遍历 bytes 对象得到的是整数，需用 bytes([整数]) 将其转回单字节 bytes 对象。
                        word_bytes = [bytes([token]) for token in word.encode('utf-8')]          #[b'l', b'o', b'w']


                # --- 核心：循环贪心合并 ---
                # 这是一个持续进行的循环，它会不断在 `word_bytes` 内部寻找最优的合并机会，
                # 直到这个单词内部再也没有任何可以根据 `merge_dict` 规则合并的字节对为止。
                        while True:
                    # 在当前 `word_bytes` 列表的所有相邻字节对中，找出那些存在于 `merge_dict` 中的“有效对”，
                    # 并构建一个 `有效对 -> 优先级排名` 的临时字典 `merge_rank`。
                            merge_rank = {
                            pair: self.merge_dict[pair] 
                            for pair in zip(word_bytes, word_bytes[1:]) 
                            if pair in self.merge_dict
                            }
            
                    # 如果 `merge_rank` 为空，说明当前单词内部已无任何可合并的字节对，
                    # 于是跳出 `while True` 循环，完成对这个单词的处理。
                            if not merge_rank:
                                break
            
                    # 贪心策略：在所有“当前可选”的合并中，根据 `.get` 方法找到 `merge_rank` 中
                    # rank 值最小（也就是优先级最高）的那一个字节对。
                            merge_pair = min(merge_rank, key=merge_rank.get)
                    # 调用辅助函数，执行合并，并用合并后的新列表覆盖旧的 `word_bytes`。
                    # 下一轮 `while` 循环将在这个更短的新列表上进行。
                            word_bytes = self.merge_encode(word_bytes, merge_pair)
        
                    # --- 步骤 3: 汇总结果 ---
                    # 当一个单词的合并循环结束后，`word_bytes` 中就是最终的 token 字节块。
                    # 使用列表推导式和 `vocab_reverse` 字典，将这些字节块高效地转换为 token ID，
                    # 并使用 `.extend()` 方法将它们全部追加到最终结果列表中。
                        text_tokenize.extend([self.vocab_reverse[b] for b in word_bytes]) 
        return text_tokenize
    # 返回拼接好的完整的 token ID 列表regex.findall(PAT, chunk)
    def decode(self,ids):
        text_list=[self.vocab[id] for id in ids]
        # 添加 errors='replace' 参数，以处理单个BPE token可能是无效或不完整UTF-8字节的情况，从而防止解码时崩溃。
        text_decode=b''.join(text_list).decode('utf-8',errors='replace')#先拼接后解码，因为汉字可能由三个字节组成，前两个字节可能不在列表的同一个元素中，解码会报错
        return text_decode
    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            ids_for_chunk = self.encode(text_chunk)  # 得到一个列表，例如 [50, 243, 11]
        
        # 下面这行代码，和上面那个 for 循环的功能完全一样
            yield from ids_for_chunk
tokenizer=None
def initial_worker(vocab_path,merges_path,special_tokens):
    global tokenizer
    tokenizer=Tokenizer.from_files(vocab_path,merges_path,special_tokens)
def worker_function(text):
    return tokenizer.encode(text)
if __name__=='__main__':
    VOCAB_PATH = 'vocab_TinyStories.pkl'
    MERGES_PATH = 'merges_TinyStories.pkl'
    SPECIAL_TOKENS = ['<|endoftext|>']
    INPUT_TXT = 'TinyStoriesV2-GPT4-valid.txt'
    OUTPUT_NPY = 'TinyStoriesV2_GPT4_valid_data.npy'  
    with open(INPUT_TXT,'r') as f:
        TEXT=f.read()  
    if SPECIAL_TOKENS:
        pattern='|'.join(map(regex.escape,SPECIAL_TOKENS))
        pattern_capture=f'({pattern})'
        chunks=regex.split(pattern_capture,TEXT)
    else:
        chunks=[TEXT]
    #创建全局变量，方便在每个线程之内直接调用
    num_cpus=os.cpu_count() or 1
    target_task_num = num_cpus * 50  # 目标任务数：CPU核心数×50（8核→400个任务）
    chunk_total = len(chunks)
    # 计算每个合并块包含的原始chunk数量（向上取整，避免遗漏）
    chunks_per_block = math.ceil(chunk_total / target_task_num)
    
    # 合并逻辑：将chunks按chunks_per_block个为一组，拼接成大文本块
    merged_chunks = []
    for i in range(0, chunk_total, chunks_per_block):
        # 取当前组的原始chunk，拼接成一个大文本
        block_text = ''.join(chunks[i:i+chunks_per_block])
        merged_chunks.append(block_text)
    print(f"合并前chunk数量：{chunk_total}")
    print(f"合并后任务数量：{len(merged_chunks)}（每块含{chunks_per_block}个原始chunk）")
    print(f'cpu的数量是{num_cpus},一共要处理{len(chunks)}个chunk')
    #获取cpu的数量
    result=[]
    with Pool(processes=num_cpus,initializer=initial_worker,initargs=(VOCAB_PATH,MERGES_PATH,SPECIAL_TOKENS)) as pool:
        for ids in tqdm(pool.imap_unordered(worker_function, merged_chunks), 
                       total=len(merged_chunks), desc='Encoding'):
            result.extend(ids)  # 用extend高效拼接列表
    result=np.array(result,dtype=np.int64)
    np.save(OUTPUT_NPY,result)
    
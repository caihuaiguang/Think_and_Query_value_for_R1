import os
import json
import argparse
import torch
import jsonlines
from tqdm import tqdm
from torch import distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 配置参数
CHAT_TEMPLATE = "{}"
RESPONSE_TEMPLATE = "{}"
MAX_QUERY_TOKENS = 10000  # 10k tokens
MAX_TOTAL_LENGTH = 128000 # 8k tokens
PAD_TO_MULTIPLE_OF = 8   # 内存对齐优化
FLASH_ATTENTION = False   # 启用Flash Attention
MODEL_NAME = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 修改为实际模型路径
DATA_PATH = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id.jsonl"  # 修改为实际数据路径
output_path = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_7B_{}.jsonl"
class OptimizedDataset(Dataset):
    """内存优化的数据集处理类"""
    
    def __init__(self, tokenizer, input_type):
        self.tokenizer = tokenizer
        self.input_type = input_type
        
        # 加载数据
        with open(DATA_PATH, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        # 预计算样本长度
        self.lengths = self._precompute_lengths()

    def _precompute_lengths(self):
        """预计算样本长度用于动态batching"""
        return [self._estimate_length(item) for item in self.data]

    def _estimate_length(self, item):
        """估计样本长度"""
        query_len = len(self.tokenizer.encode(item["query"], add_special_tokens=False))
        response_len = len(self.tokenizer.encode(item["response"], add_special_tokens=False))
        return query_len + response_len + 10  # 添加模板字符余量

    def __len__(self):
        return len(self.data)

    def _truncate_tokens(self, tokens, max_length):
        """智能截断策略：保留头部和尾部关键信息"""
        if len(tokens) <= max_length:
            return tokens
        head = tokens[:max_length//2]
        tail = tokens[-(max_length - max_length//2):]
        return head + tail

    def __getitem__(self, idx):
        ret = {} 
        text = self.data[idx]
        
        # 处理query部分，限制为120k tokens
        raw_query = text["query"]
        # 编码为tokens，不添加特殊标记（如[CLS]、[SEP]）
        query_tokens = self.tokenizer.encode(raw_query, add_special_tokens=False)
        max_query_tokens = 5 * 1000  # 5k tokens
        if len(query_tokens) > max_query_tokens:
            # 策略：保留前60k和后60k tokens，确保总长度120k
            head = query_tokens[:int(max_query_tokens/2)]
            tail = query_tokens[-int(max_query_tokens/2):]
            truncated_query_tokens = head + tail
            # 确保合并后的长度不超过max_query_tokens（可能存在合并后超出）
            truncated_query_tokens = truncated_query_tokens[:max_query_tokens]
            # 解码为文本（可能引入部分不连贯，但有效控制长度）
            processed_query = self.tokenizer.decode(truncated_query_tokens, clean_up_tokenization_spaces=True)
            print("query_tokens: ", len(query_tokens))
        else:
            processed_query = raw_query

        # # 使用处理后的query构建输入
        # inputs = self.tokenizer(
        #     CHAT_TEMPLATE.format(
        #         processed_query,  # 使用处理后的query
        #         "<think>" + text["think"] + "</think>" if self.is_thinking else ""
        #     ),
        #     add_special_tokens=False,
        #     truncation=True,
        #     max_length=40*1000
        # )
        
        # response = self.tokenizer(
        #     RESPONSE_TEMPLATE.format(text["response"]),
        #     truncation=True,
        #     max_length=40*1000,
        #     add_special_tokens=False
        # )

        # 使用处理后的query构建输入
        if self.input_type == "qta":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    processed_query,  # 使用处理后的query
                    "<think>" + text["think"] + "</think>"
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.input_type == "qa" or self.input_type == "q":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    processed_query,  # 使用处理后的query
                    ""
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.input_type == "ta":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    "",  # 使用处理后的query
                    "<think>" + text["think"] + "</think>"
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.input_type == "a":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    "",
                    ""
                ),
                add_special_tokens=False,
                truncation=False
            )
        if self.input_type == "q":
            response = self.tokenizer(
                RESPONSE_TEMPLATE.format(""),
                add_special_tokens=False,
                truncation=False,
            )
        else:   
            response = self.tokenizer(
                RESPONSE_TEMPLATE.format(text["response"]),
                add_special_tokens=False,
                truncation=False,
            )
        
        input_ids = inputs.input_ids + response.input_ids
        if self.input_type == "q":
            labels = inputs.input_ids + response.input_ids
        else:
            labels = [-100] * len(inputs.input_ids) + response.input_ids
        attention_mask = inputs.attention_mask + response.attention_mask
        

        ret["input_ids"] = input_ids
        ret["labels"] = labels
        ret["attention_mask"] = attention_mask
        ret["trace_id"] = text["trace_id"]
        return ret

class MemoryOptimizedCollator:
    """内存优化的数据整理器"""
    
    def __init__(self, tokenizer):
        self.collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            padding=True,
            return_tensors="pt"
        )
    
    def __call__(self, batch):
        # 分离非张量数据
        trace_ids = [item.pop("trace_id") for item in batch]
        
        # 调用HF的标准整理器
        batch = self.collator(batch)
        
        # 重新添加trace_id
        batch["trace_ids"] = trace_ids
        return batch

def initialize_distributed():
    """初始化分布式环境"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def load_model(local_rank):
    """优化后的模型加载方法"""
    # 明确指定设备
    device = torch.device(f"cuda:{local_rank}")
    
    # 精确控制设备映射
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": device}  # 强制分配到指定GPU
    ).to(device)
    
    # 保持冻结参数（推理模式）
    model.eval()
    return model

def compute_log_probs(model, batch):
    """内存高效的概率计算"""

    device = next(model.parameters()).device
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            outputs = model(**inputs)
    
    # 优化计算：仅计算必要部分
    logits = outputs.logits
    labels = batch["labels"].to(device)
    
    # 计算每个token的log概率
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    valid_mask = (shift_labels != -100)
    
    # 稀疏计算
    nll_loss = -log_probs.gather(
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # 聚合结果
    results = []
    for i in range(labels.size(0)):
        mask = valid_mask[i]
        if not mask.any():
            score = 0.0
        else:
            score = nll_loss[i][mask].mean().item()
        results.append({
            "trace_id": batch["trace_ids"][i],
            "log_prob": score
        })
    
    return results

def main():
    # 初始化分布式环境
    rank, local_rank, world_size = initialize_distributed()
    
    # 加载模型
    model = load_model(local_rank)

    print(f"Rank {rank} model device:", next(model.parameters()).device)

    model.eval()
    # 修改DDP包装
    # model = DDP(model,  # 现在model已经是half精度
    #         broadcast_buffers=False,
    #         device_ids=[local_rank],
    #         bucket_cap_mb=16,
    #         find_unused_parameters=True)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        model_max_length=MAX_TOTAL_LENGTH
    )
    
    # 准备数据
    dataset = OptimizedDataset(tokenizer, input_type=args.type)
    collator = MemoryOptimizedCollator(tokenizer)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 根据显存调整
        sampler=sampler,
        collate_fn=collator,
        pin_memory=True
    )
    
    # 推理循环
    results = []
    for batch in tqdm(dataloader, desc=f"Rank {rank}"):
        batch_results = compute_log_probs(model, batch)
        results.extend(batch_results)
    
    # 收集所有结果
    gathered_results = [None] * world_size
    distributed.all_gather_object(gathered_results, results)
    
    # 主进程处理结果
    if rank == 0:
        final_results = []
        for res in gathered_results:
            final_results.extend(res)
        
        # 去重处理
        seen = set()
        dedup_results = []
        for item in final_results:
            if item["trace_id"] not in seen:
                seen.add(item["trace_id"])
                dedup_results.append(item)
        
        # 保存结果
        with jsonlines.open(output_path.format(args.type), "w") as f:
            f.write_all(dedup_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="qta",
                        choices=["qta", "qa", "ta", "a", "q"])
    args = parser.parse_args()
    
    # 配置优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main()
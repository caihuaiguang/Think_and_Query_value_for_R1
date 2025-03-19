import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, dataset
import json
import torch
import os
import torch.nn.functional as F
import jsonlines
from tqdm import tqdm
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

CHAT_TEMPLATE = "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>{}"
RESPONSE_TEMPLATE = "{}<｜end▁of▁sentence｜>"


def list_of_lists_to_tensor(data, padding_value=0):
    """
    将嵌套列表转换为二维张量。
    :param data: 嵌套列表，例如 [[1, 2], [3, 4, 5], [6]]
    :param padding_value: 用于填充的值，默认为 0
    :return: 二维张量
    """
    # 将每个子列表转换为张量
    tensors = [torch.tensor(sublist) for sublist in data]

    # 使用 pad_sequence 对齐张量
    padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    return padded_tensor

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, input_files, is_thinking, max_length=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(input_files,'r') as frec:
            self.alldata = [json.loads(line) for line in frec.readlines()]
        self.is_thinking = is_thinking


    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, idx):
        ret = {'input_ids': [], "labels": [], "attention_mask": []}
        text = self.alldata[idx]
        
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
        if self.type == "qta":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    processed_query,  # 使用处理后的query
                    "<think>" + text["think"] + "</think>"
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.type == "qa":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    processed_query,  # 使用处理后的query
                    ""
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.type == "ta":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    "",  # 使用处理后的query
                    "<think>" + text["think"] + "</think>"
                ),
                add_special_tokens=False,
                truncation=False
            )
        elif self.type == "a":
            inputs = self.tokenizer(
                CHAT_TEMPLATE.format(
                    "",
                    ""
                ),
                add_special_tokens=False,
                truncation=False
            )
        
        response = self.tokenizer(
            RESPONSE_TEMPLATE.format(text["response"]),
            add_special_tokens=False,
            truncation=False,
        )
        
        input_ids = inputs.input_ids + response.input_ids
        labels = [-100] * len(inputs.input_ids) + response.input_ids
        attention_mask = inputs.attention_mask + response.attention_mask
        
        ret["input_ids"] = input_ids
        ret["labels"] = labels
        ret["attention_mask"] = attention_mask
        ret["trace_id"] = text["trace_id"]
        return ret

def custom_collate_fn(batch):

    
    input_ids = [item["input_ids"][0] for item in batch]
    labels = [item["labels"][0] for item in batch]
    attention_mask = [item["attention_mask"][0] for item in batch]
    trace_id = [item["trace_id"] for item in batch]

    all_ret = {'input_ids': [], "labels": [], "attention_mask": []}
    # 填充输入序列到相同长度
    all_ret["input_ids"] = list_of_lists_to_tensor(input_ids, 0)
    all_ret["attention_mask"] = list_of_lists_to_tensor(attention_mask, 0)
    all_ret["labels"] = list_of_lists_to_tensor(labels, -100)
    all_ret["trace_id"] = trace_id
    return all_ret

def gather_objects(objects):
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    if rank == 0:
        results = [None for _ in range(world_size)]
        torch.distributed.gather_object(objects, results)
        if isinstance(objects, list):
            results = [r for result in results for r in result]
    else:
        torch.distributed.gather_object(objects, dst=0)
        results = None
    return results



def Forward(model, batch):
    batched_data = batch
    labels = batched_data.pop("labels")
    trace_ids = batch["trace_id"]
    losses = []
    with torch.no_grad(): 
        logits = model(**batched_data).logits
        for i in range(len(logits)):
            label = labels[i]
            len_query = torch.nonzero(label != -100, as_tuple=False)[0].item()
            logit = logits[i, len_query: -1, :]
            label = label[len_query + 1: ]

            log_probs = F.log_softmax(logit, dim=-1)
    
            valid_mask = (labels != -100)
        
            sample_labels = label
            sample_log_probs = log_probs

            # print(sample_labels)
            valid_indices = sample_labels != -100
            valid_sample_labels = sample_labels[valid_indices]
            valid_sample_log_probs = sample_log_probs[valid_indices]
            
            # 计算每个样本的有效位置的 log 概率
            if len(valid_sample_labels) == 0:
                # 如果没有有效位置，损失设为 0 或其他默认值
                sample_loss = torch.tensor(0.0).to(device)
            else:
                # 提取对应的 log 概率
                valid_log_probs = valid_sample_log_probs.gather(-1, valid_sample_labels.unsqueeze(-1)).squeeze(-1)
                sample_loss = -valid_log_probs.mean()
            
            losses.append(sample_loss)
    
    # 将损失列表转化为张量，保留 batch_size 维度
    loss = torch.stack(losses).cpu().tolist()
    
    # 打印损失值
    # print(f"Loss: {loss}")
    
    return [{"trace_id": trace_ids[i], "mean(log_probability)": loss[i]} for i in range(len(trace_ids))]

def deduplicate_dict_list(dict_list, keys=None):
    """
    对包含字典的列表进行去重。
    :param dict_list: 包含字典的列表
    :param keys: 用于去重的键列表，如果为 None，则整个字典作为去重依据
    :return: 去重后的列表
    """
    seen = set()
    deduplicated_list = []
    
    for item in dict_list:
        # 如果没有指定键，则使用整个字典进行去重
        if keys is None:
            # 使用 frozenset 或 json.dumps 来处理不可哈希的字典
            frozen_item = frozenset(item.items())
            # 或者使用 json.dumps，并排除键顺序的影响
            # frozen_item = json.dumps(item, sort_keys=True)
        else:
            # 提取指定键的值并生成元组
            unique_key = tuple(item.get(key, None) for key in keys)
        
        if unique_key not in seen:
            seen.add(unique_key)
            deduplicated_list.append(item)
    
    return deduplicated_list

# 主程序
def main():
    # model_name = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # model_name = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = 131072
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
    data_path = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id.jsonl"
    # output_path = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_1_5B_{}.jsonl"
    output_path = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_7B_{}.jsonl"
    # output_path = "/data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_32B_{}.jsonl"
    is_cot = True
    dataset1 = SimpleDataset(tokenizer, data_path, is_cot)

    world_size = int(os.environ["WORLD_SIZE"])  # 全局进程个数
    rank = int(os.environ["RANK"])  # 当前进程编号(全局)
    local_rank = int(os.environ["LOCAL_RANK"])  # 每台机器上的进程编号(局部)
    distributed.init_process_group("nccl")
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    train_sampler = DistributedSampler(
            dataset1, num_replicas=world_size, rank=rank, shuffle=False)
    trainloader = DataLoader(
                            dataset=dataset1,
                            batch_size=1,
                            sampler=train_sampler,
                            collate_fn=custom_collate_fn)

    print(f"Using {torch.cuda.device_count()} GPUs")
    model.cuda()
    model.eval()
    model = DDP(model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    # 设置模型为评估模式
    

    ret1 = []
    with torch.no_grad():
        for batch in tqdm(trainloader):
            torch.cuda.empty_cache()
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # 将数据移动到 GPU
            # for k, v in batch.items():
            #     print(v.size())
            outputs = Forward(model.module, batch)
            ret1 += outputs

    ret1 = gather_objects(ret1)
    distributed.barrier()
    if rank == 0:
        ret1 = deduplicate_dict_list(ret1, keys=["trace_id"])
        print(len(ret1))
        with jsonlines.open(output_path.format(is_cot), 'w') as writer:
            for data in ret1:
                writer.write(data)
    

if __name__ == "__main__":
    main()
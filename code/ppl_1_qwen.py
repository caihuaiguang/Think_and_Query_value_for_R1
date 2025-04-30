import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import os
import torch.nn.functional as F
import jsonlines
import argparse
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

QUERY_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
THINK_TEMPLATE = "<think>\n{}\n</think>\n\n"
ANSWER_TEMPLATE = "{}<|im_end|>\n"

def list_of_lists_to_tensor(data, padding_value=0):
    tensors = [torch.tensor(sublist) for sublist in data]
    padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    return padded_tensor

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, input_files, input_type, max_length=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(input_files,'r') as frec:
            self.alldata = [json.loads(line) for line in frec.readlines()]
        self.input_type = input_type

    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, idx):
        ret = {'input_ids': [], "labels": [], "attention_mask": []}
        text = self.alldata[idx]
        
        # 处理query部分，限制为120k tokens
        raw_query = text["query"]
        query_tokens = self.tokenizer.encode(raw_query, add_special_tokens=False)
        max_query_tokens = 10 * 1000
        if len(query_tokens) > max_query_tokens:
            head = query_tokens[:int(max_query_tokens/2)]
            tail = query_tokens[-int(max_query_tokens/2):]
            truncated_query_tokens = head + tail
            truncated_query_tokens = truncated_query_tokens[:max_query_tokens]
            processed_query = self.tokenizer.decode(truncated_query_tokens, clean_up_tokenization_spaces=True)
        else:
            processed_query = raw_query
        
        query = self.tokenizer(
            QUERY_TEMPLATE.format(processed_query),
            add_special_tokens=False,
            truncation=False
        )
        think = self.tokenizer(
            THINK_TEMPLATE.format(text["think"]),
            add_special_tokens=False,
            truncation=False
        )
        answer = self.tokenizer(
            ANSWER_TEMPLATE.format(text["response"]),
            add_special_tokens=False,
            truncation=False
        )
        
        # 根据input_type处理不同模式
        if self.input_type == "a_qt":
            labels = [-100] * len(query.input_ids) + [-100] * len(think.input_ids) + answer.input_ids
        elif self.input_type == "a_q":
            think = self.tokenizer(THINK_TEMPLATE.format(""), add_special_tokens=False, truncation=False)
            labels = [-100] * len(query.input_ids) + [-100] * len(think.input_ids) + answer.input_ids
        elif self.input_type == "ta_q":
            labels = [-100] * len(query.input_ids) + think.input_ids + answer.input_ids
        elif self.input_type == "ta_empty":
            query = self.tokenizer(QUERY_TEMPLATE.format(""), add_special_tokens=False, truncation=False)
            labels = [-100] * len(query.input_ids) + think.input_ids + answer.input_ids
        elif self.input_type == "a_t":
            query = self.tokenizer(QUERY_TEMPLATE.format(""), add_special_tokens=False, truncation=False)
            labels = [-100] * len(query.input_ids) + [-100] * len(think.input_ids) + answer.input_ids
        elif self.input_type == "a_empty":
            query = self.tokenizer(QUERY_TEMPLATE.format(""), add_special_tokens=False, truncation=False)
            think = self.tokenizer(THINK_TEMPLATE.format(""), add_special_tokens=False, truncation=False)
            labels = [-100] * len(query.input_ids) + [-100] * len(think.input_ids) + answer.input_ids

        input_ids = query.input_ids + think.input_ids + answer.input_ids
        attention_mask = query.attention_mask + think.attention_mask + answer.attention_mask
        
        ret["input_ids"].append(input_ids)
        ret["labels"].append(labels)
        ret["attention_mask"].append(attention_mask)
        ret["trace_id"] = text["trace_id"]
        return ret

def custom_collate_fn(batch):
    input_ids = [item["input_ids"][0] for item in batch]
    labels = [item["labels"][0] for item in batch]
    attention_mask = [item["attention_mask"][0] for item in batch]
    trace_id = [item["trace_id"] for item in batch]

    all_ret = {'input_ids': [], "labels": [], "attention_mask": []}
    all_ret["input_ids"] = list_of_lists_to_tensor(input_ids, 0)
    all_ret["attention_mask"] = list_of_lists_to_tensor(attention_mask, 0)
    all_ret["labels"] = list_of_lists_to_tensor(labels, -100)
    all_ret["trace_id"] = trace_id
    return all_ret

def Forward(model, batch, device):
    batched_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels = batched_data.pop("labels")
    trace_ids = batch["trace_id"]
    losses = []
    
    with torch.no_grad(): 
        logits = model(**batched_data).logits
        for i in range(len(logits)):
            label = labels[i]
            len_query = torch.nonzero(label != -100, as_tuple=False)[0].item()
            logit = logits[i, len_query:-1, :]
            label = label[len_query + 1:]
            
            log_probs = F.log_softmax(logit, dim=-1)
            valid_mask = (label != -100)
            
            if valid_mask.sum() == 0:
                sample_loss = torch.tensor(0.0).to(device)
            else:
                valid_log_probs = log_probs.gather(-1, label.unsqueeze(-1)).squeeze(-1)
                valid_log_probs = valid_log_probs[valid_mask]
                sample_loss = -valid_log_probs.mean()
                
            losses.append(sample_loss)
    
    loss = torch.stack(losses).cpu().tolist()
    return [{"trace_id": trace_ids[i], "mean(log_probability)": loss[i]} for i in range(len(trace_ids))]

def deduplicate_dict_list(dict_list, keys=None):
    seen = set()
    deduplicated_list = []
    for item in dict_list:
        unique_key = tuple(item.get(key, None) for key in keys) if keys else frozenset(item.items())
        if unique_key not in seen:
            seen.add(unique_key)
            deduplicated_list.append(item)
    return deduplicated_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="qta")
    args = parser.parse_args()
    input_type = args.type
    
    # 单卡设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型加载
    model_name = "/root/autodl-tmp/models/Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = 131072
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    
    # 数据准备
    data_path = "/root/autodl-tmp/Think_and_Query_value_for_R1/data/distill_r1_110k_sft_with_id.jsonl"
    output_path = f"/root/autodl-tmp/Think_and_Query_value_for_R1/data/qwen3_output_0_6B_{input_type}.jsonl"
    
    dataset = SimpleDataset(tokenizer, data_path, input_type)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    
    # 推理过程
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            outputs = Forward(model, batch, device)
            results.extend(outputs)
    
    # 结果去重和保存
    final_results = deduplicate_dict_list(results, keys=["trace_id"])
    print(f"Processed {len(final_results)} unique samples")
    
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(final_results)

if __name__ == "__main__":
    main()

import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import tiktoken


# 900 为均值，300 为标准差
def add_trace_id(input_file_path, output_file_path, with_think_map, without_think_map):
    source_1 = open(input_file_path, "r")
    target = open(output_file_path, "w")
    for line in source_1:
        d = json.loads(line)
        index += 1
        trace_id = d["trace_id"]
        with_think_loss = with_think_map.get(trace_id)
        without_think_loss = without_think_map.get(trace_id)

        all_data = {
            "trace_id": trace_id,
            "query": d["trace_id"],
            "think": d["think"],
            "response": d["response"],
            "repo_name": d["repo_name"],
            "prompt_tokens_len": d["prompt_tokens_len"],
            "reasoning_content_tokens_len": d["reasoning_content_tokens_len"],
            "content_tokens_len": d["content_tokens_len"],
            "score": d["score"],
            "with_think_loss": with_think_loss,
            "without_think_loss": without_think_loss,
            "think_value": without_think_loss - with_think_loss,
        }
        target.write(json.dumps(all_data, ensure_ascii=False) + "\n")
    print(skip_num)



if __name__ == "__main__":
    input_file_path = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id.jsonl"
    output_file_path = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_think_value.jsonl"
    with_think_map = {}
    without_think_map = {}
    with open("/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_1_5B_True.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            with_think_map[d["trace_id"]] = d["mean(log_probability)"]
    with open("/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/output_1_5B_False.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            without_think_map[d["trace_id"]] = d["mean(log_probability)"]
    add_trace_id(input_file_path, output_file_path, with_think_map, without_think_map)

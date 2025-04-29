import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import tiktoken


# 900 为均值，300 为标准差
def add_trace_id(input_file_path, output_file_path):
    source_1 = open(input_file_path, "r")
    target = open(output_file_path, "w")
    index = 0
    skip_num = 0
    for line in source_1:
        d = json.loads(line)
        index += 1
        trace_id = index
        query = d["instruction"]

        think_response = d["output"]
        think_parts = think_response.split("</think>")
        if len(think_parts) > 1:
            think = think_parts[0].replace("<think>", "")
            response = think_parts[1]
        else:
            skip_num+=1
            continue

        all_data = {
            "trace_id": trace_id,
            "query": query,
            "think": think,
            "response": response,
            "repo_name": d["repo_name"],
            "prompt_tokens_len": d["prompt_tokens_len"],
            "reasoning_content_tokens_len": d["reasoning_content_tokens_len"],
            "content_tokens_len": d["content_tokens_len"],
            "score": d["score"],
        }
        target.write(json.dumps(all_data, ensure_ascii=False) + "\n")
    print(skip_num)



if __name__ == "__main__":
    input_file_path = "/root/Think_and_Query_value_for_R1/data/distill_r1_110k_sft.jsonl"
    output_file_path = "/root/Think_and_Query_value_for_R1/data/distill_r1_110k_sft_with_id.jsonl"
    add_trace_id(input_file_path, output_file_path)

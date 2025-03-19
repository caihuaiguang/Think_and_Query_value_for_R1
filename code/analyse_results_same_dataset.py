import json
import numpy as np
import seaborn as sns
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict

from transformers import AutoTokenizer
"""
    # general 58573
    'coig/neo': 52938, 
    'zhihu/zhihu_score9.0-10_clean_v10': 2561,
    'xhs/xhs': 1507,
    'ruozhiba/ruozhiba_ruozhiba': 240, 
    'human_value/100poison': 906,  
    'logi_qa/logi-qa': 421, 

    # math 36987
    'Haijian/Advanced-Math': 587, 
    'meta-math/GSM8K_zh': 8789, 
    'EduChat-Math': 20113,  
    'gavinluo/applied_math': 7498,  

    # stem 12000
    'stem_zh/chem': 3000, 
    'stem_zh/bio': 3000, 
    'stem_zh/med': 3000, 
    'stem_zh/phy': 3000,

    # exam 2440
    'exam/coig_exam': 1962, 
    'exam/kaoyan': 377, 
    'human_value/coig_human': 101
"""

"""
    "trace_id": trace_id,
    "query": d["trace_id"],
    "think": d["think"],
    "response": d["response"],
    "repo_name": d["repo_name"],
    "prompt_tokens_len": d["prompt_tokens_len"],
    "reasoning_content_tokens_len": d["reasoning_content_tokens_len"],
    "content_tokens_len": d["content_tokens_len"],
    "score": d["score"],
    "think_value": d["think_value"],
    "query_value": d["query_value"],
    "answer_ppl": d["answer_ppl"]
"""


def plot_dis(data, save_fig_path, title=None):
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=data, color="skyblue", inner="quartile")
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Position')
    plt.legend()
    plt.title(title or "Violin Plot with Zero Position", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def read_json_data(file_path):
    """Read JSON file and return a dictionary mapping trace_id to value"""
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_dict[data['trace_id']] = data['mean(log_probability)']
            # data_dict[data['trace_id']] = np.exp(data['mean(log_probability)'])
    return data_dict


def print_stats(data, label):
    """Print statistical information for a dataset"""
    print(f"{label}: avg={np.mean(data):.2f}, median={np.median(data):.2f}, min={min(data)}, max={max(data)}")


def plot_overlapping_distributions(data_dict, save_path, title=None, value_type="Think"):
    """Plot distributions with improved visualization"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_dict)))
    
    for i, (name, values) in enumerate(sorted(data_dict.items(), key=lambda x: np.median(x[1]), reverse=True)):
        if len(values) > 1:
            sns.kdeplot(values, fill=True, alpha=0.4, color=colors[i], 
                       linewidth=2, label=f"{name} (median= {np.median(values):.4f})")
            # plt.axvline(x=np.median(values), color=colors[i], linestyle='--', alpha=0.7)
    
    # plt.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Zero Position')
    plt.title(title or f"{value_type} Distribution by Repository", fontsize=14)
    plt.xlabel("Value"), plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_data(input_path, data_maps):
    """Process input data and calculate values"""
    repo_data = defaultdict(lambda: defaultdict(list))
    
    with open(input_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            
            # Calculate values
            # think_value = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            # query_value = data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id]
            # think_ratio = think_value/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            # query_ratio = query_value/data_maps['ta_empty'][trace_id]


            # choice 1
            think_value = (data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]) / 2
            query_value = (data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id] + 
                         data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]) / 2
            think_ratio = think_value / data_maps['a_empty'][trace_id] 
            query_ratio = query_value / data_maps['a_empty'][trace_id] 
            # think_ratio = think_value/(think_value+query_value) 
            # query_ratio = query_value/(think_value+query_value) 
            # if (data_maps['a_empty'][trace_id]-data_maps['a_qt'][trace_id] < 0):
            #     print("here!", d)
            
            # choice 2
            # think_value = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            # think_ratio = think_value/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            # query_value = data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id]
            # query_ratio = query_value/data_maps['ta_empty'][trace_id]

            # choice 3
            # think_value = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            # think_ratio = think_value/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            # query_value = data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id]
            # query_ratio = query_value/data_maps['a_empty'][trace_id]

            repo = d['repo_name']
            repo_data[repo]['think_value'].append(think_value)
            repo_data[repo]['query_value'].append(query_value)
            repo_data[repo]['think_ratio'].append(think_ratio)
            repo_data[repo]['query_ratio'].append(query_ratio)

            # repo_data[repo]['a_qt_ppl'].append(data_maps['a_qt'][trace_id])
            # repo_data[repo]['a_q_ppl'].append(data_maps['a_q'][trace_id])
            # repo_data[repo]['qt_value'].append(data_maps['a_qt'][trace_id] - data_maps['a_q'][trace_id])
            # Calculate think_ratio safely
    
    return repo_data


def main():
    # Configuration
    base_dir = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data"
    model_size = "7B"
    save_fig_dir = f"{base_dir}/{model_size}_results/"
    os.makedirs(save_fig_dir, exist_ok=True)
    
    # Define output files
    output_files = {
        'query_value_less_0': f"{base_dir}/distill_r1_110k_sft_with_id_and_query_value_less_0.jsonl",
        'think_value_less_than_query_value': f"{base_dir}/distill_r1_110k_sft_with_id_and_think_value_less_than_query_value.jsonl",
        'refuse': f"{base_dir}/distill_r1_110k_sft_with_id_refuse.jsonl",
        'query_value_sorted': f"{base_dir}/distill_r1_110k_sft_with_id_and_query_value_sorted.jsonl",
        'think_value_sorted': f"{base_dir}/distill_r1_110k_sft_with_id_and_think_value_sorted.jsonl",
        'query_ratio_sorted': f"{base_dir}/distill_r1_110k_sft_with_id_and_query_ratio_sorted.jsonl",
        'think_ratio_sorted': f"{base_dir}/distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl"
    }
    
    # Load and process data
    input_path = f"{base_dir}/distill_r1_110k_sft_with_id.jsonl"
    data_maps = {
        'a_qt': read_json_data(f'{base_dir}/output_{model_size}_a_qt.jsonl'),
        'a_q': read_json_data(f'{base_dir}/output_{model_size}_a_q.jsonl'),
        'ta_q': read_json_data(f'{base_dir}/output_{model_size}_ta_q.jsonl'),
        'ta_empty': read_json_data(f'{base_dir}/output_{model_size}_ta_empty.jsonl'),
        'a_t': read_json_data(f'{base_dir}/output_{model_size}_a_t.jsonl'),
        'a_empty': read_json_data(f'{base_dir}/output_{model_size}_a_empty.jsonl')
    }
    
    # Read all data and store with calculated values
    all_data = []
    with open(input_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            # choice 1
            think_value = (data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]) / 2
            query_value = (data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id] + 
                         data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]) / 2
            think_ratio = think_value / data_maps['a_empty'][trace_id] 
            query_ratio = query_value / data_maps['a_empty'][trace_id] 
            # think_ratio = think_value/(think_value+query_value) 
            # query_ratio = query_value/(think_value+query_value) 
            # if (data_maps['a_empty'][trace_id]-data_maps['a_qt'][trace_id] < 0):
            #     print("here!", d)
            
            # choice 2
            # think_value = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            # query_value = data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id]
            # think_ratio = think_value/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            # query_ratio = query_value/data_maps['ta_empty'][trace_id]
            

            # choice 3
            # think_value = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            # think_ratio = think_value/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            # query_value = data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id]
            # query_ratio = query_value/data_maps['a_empty'][trace_id]

            d.update({
                'think_value': think_value,
                'query_value': query_value,
                'think_ratio': think_ratio,
                'query_ratio': query_ratio
            })
            all_data.append(d)

    # Write sorted files
    for key, data in [
        ('query_value_sorted', sorted(all_data, key=lambda x: x['query_value'])),
        ('think_value_sorted', sorted(all_data, key=lambda x: x['think_value'])),
        ('query_ratio_sorted', sorted(all_data, key=lambda x: x['query_ratio'])),
        ('think_ratio_sorted', sorted(all_data, key=lambda x: x['think_ratio'])),
    ]:
        with open(output_files[key], 'w') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(output_files["query_value_less_0"], 'w') as f:
        for d in all_data:
            if d["query_value"] < 0:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    with open(output_files["refuse"], 'w') as f:
        for d in all_data:
            if "我是由中国的深度求索（DeepSeek）公司" in d["think"] or "我是由中国的深度求索（DeepSeek）公司" in d["response"]:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    

    think_large_query_ratio = 0
    with open(output_files["think_value_less_than_query_value"], 'w') as f:
        for d in all_data:
            if d["query_value"] < d["think_value"]:
                think_large_query_ratio += 1
            else:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    think_large_query_ratio /= len(data)

    # Continue with existing visualization code
    repo_data = process_data(input_path, data_maps)
    
    # Define categories
    categories = {
        "general": ["coig/neo", "zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", 
                   "ruozhiba/ruozhiba_ruozhiba", "human_value/100poison", "logi_qa/logi-qa"],
        "community": ["zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", "ruozhiba/ruozhiba_ruozhiba"],
        "math": ["Haijian/Advanced-Math", "meta-math/GSM8K_zh", "EduChat-Math", "gavinluo/applied_math"],
        "stem": ["stem_zh/chem", "stem_zh/bio", "stem_zh/med", "stem_zh/phy"],
        "exam": ["exam/coig_exam", "exam/kaoyan", "human_value/coig_human"],
        "all": ["coig/neo", "zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", 
                   "ruozhiba/ruozhiba_ruozhiba", "human_value/100poison", "logi_qa/logi-qa",
                   "zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", "ruozhiba/ruozhiba_ruozhiba",
                   "Haijian/Advanced-Math", "meta-math/GSM8K_zh", "EduChat-Math", "gavinluo/applied_math",
                   "stem_zh/chem", "stem_zh/bio", "stem_zh/med", "stem_zh/phy",
                   "exam/coig_exam", "exam/kaoyan", "human_value/coig_human"]
    }
    
    # Plot distributions for each category
    metrics = ['think_value', 'query_value', 'think_ratio', 'query_ratio']
    for category, repos in categories.items():
        for metric in metrics:
            category_data = {repo: repo_data[repo][metric] 
                           for repo in repos if repo in repo_data}
            if category_data:
                save_path = os.path.join(save_fig_dir, f'{metric}_{category}_overlapping.png')
                plot_overlapping_distributions(category_data, save_path,
                    title=f"{metric.replace('_', ' ').title()} Distribution - {category.upper()}")

    # Print repository sample counts
    sample_sum = 0
    print("Repository sample counts:")
    for repo_name, values in sorted(repo_data.items(), key=lambda x: len(x[1]['think_value']), reverse=True):
        print(f"{repo_name}: {len(values['think_value'])}")
        sample_sum += len(values['think_value'])
    print("total_samples:", sample_sum)
    
    # Analyze token length statistics
    token_stats = {
        'reasoning_lengths': [],
        'content_lengths': [],
        'ratios': []
    }
    
    repo_token_stats = {}
    
    with open(input_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            repo_name = d['repo_name']
            reasoning_len = d.get('reasoning_content_tokens_len', 0)
            content_len = d.get('content_tokens_len', 0)
            ratio = reasoning_len / content_len if content_len > 0 else 0
            
            # Add to overall stats
            token_stats['reasoning_lengths'].append(reasoning_len)
            token_stats['content_lengths'].append(content_len)
            token_stats['ratios'].append(ratio)
            
            # Add to repo-specific stats
            if repo_name not in repo_token_stats:
                repo_token_stats[repo_name] = {
                    'reasoning_lengths': [],
                    'content_lengths': [],
                    'ratios': []
                }
            
            repo_token_stats[repo_name]['reasoning_lengths'].append(reasoning_len)
            repo_token_stats[repo_name]['content_lengths'].append(content_len)
            repo_token_stats[repo_name]['ratios'].append(ratio)

    # # Print overall token statistics
    # print("\nToken Length Statistics:")
    # print_stats(token_stats['reasoning_lengths'], "Reasoning token length")
    # print_stats(token_stats['content_lengths'], "Content token length")
    # print_stats(token_stats['ratios'], "Reasoning/Content ratio")
    
    # Print repo-specific statistics
    print("\nMedian Statistics by Repository:")
    for repo_name, stats in sorted(repo_token_stats.items(), key=lambda x: len(x[1]['reasoning_lengths']), reverse=True):
        count = len(stats['reasoning_lengths'])
        median_reasoning = np.median(stats['reasoning_lengths'])
        median_content = np.median(stats['content_lengths'])
        median_ratio = np.median(stats['ratios'])
        print(f"{repo_name} (n={count}): reasoning={median_reasoning}, content={median_content}, ratio={median_ratio:.2f}")
    print("Percentage of cases where 'think' plays a dominant role over 'query' in determining 'answer':", think_large_query_ratio)

if __name__ == "__main__":
    main()




"""
grep "我是由中国的深度求索（DeepSeek）公司" doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id.jsonl | wc -l
124


grep "我是由中国的深度求索（DeepSeek）公司" doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id_and_query_value_less_0.jsonl | wc -l
43

# query 的价值-(Loss(ta|q) - Loss(ta|empty))

# think的价值，应该是给定q的情况下，t出不出现对a的影响 Loss(a|q) - Loss(a|qt)


1. think和query价值之间的关系：Percentage of cases where 'think' plays a dominant role over 'query' in determining 'answer': 0.9926545454545455

2. think_ratio看不同社区、学科、难度之间的关系

3. query value小于0时候和拒答之间的关系
"""

import json
import numpy as np
import seaborn as sns
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm

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


from matplotlib import font_manager
noto_fonts = [f.name for f in font_manager.fontManager.ttflist if 'Noto' in f.name]
print(noto_fonts)


# font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"  # 常见路径
# try:
#     font_manager.fontManager.addfont(font_path)
#     plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 使用完整字体名
#     plt.rcParams['axes.unicode_minus'] = False
# except FileNotFoundError:
#     print(f"错误：字体文件 {font_path} 不存在，请检查路径！")

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


def plot_overlapping_distributions(repo_data_dict, save_fig_path, title=None, value_type="Think"):
    """Plot overlapping distributions for different repositories"""
    plt.figure(figsize=(12, 8))
    
    # Define a colormap with good contrast
    colors = plt.cm.tab20(np.linspace(0, 1, len(repo_data_dict)))
    
    # Sort repos by sample size (largest first)
    sorted_repos = sorted(repo_data_dict.items(), key=lambda x: np.median(x[1]), reverse=True)
    
    for i, (repo_name, values) in enumerate(sorted_repos):
        # Calculate kernel density estimate
        if len(values) > 1:  # KDE requires at least 2 points
            color = colors[i]
            # Use seaborn's kdeplot with matching colors
            # sns.kdeplot(values, fill=True, alpha=0.4, color=color, 
            #             linewidth=2, label=f"{repo_name} (样本数={len(values)}，中位数={np.median(values):.3f})")
            sns.kdeplot(values, fill=True, alpha=0.4, color=color, 
                        linewidth=2, label=f"{repo_name} (median ={np.median(values):.3f})")
            
            # Add a small vertical line at the median
            # median_val = np.median(values)
            # plt.axvline(x=median_val, color=color, linestyle='--', alpha=0.7)
    
    # Add a vertical line at zero
    # plt.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Zero Position')
    
    plt.title(title or f"{value_type} Value Distribution by Repository", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a legend with smaller font size and place it outside the plot
    plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Configuration
    model_size = "7B"  # Options: "1_5B" or "7B"
    base_dir = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data"
    input_file_path = f"{base_dir}/distill_r1_110k_sft_with_id.jsonl"
    output_file_think_less_than_0 = f"{base_dir}/distill_r1_110k_sft_with_id_and_think_less_than_0.jsonl"
    output_file_think_greater_than_0 = f"{base_dir}/distill_r1_110k_sft_with_id_and_think_greater_than_0.jsonl"
    output_file_query_less_than_0 = f"{base_dir}/distill_r1_110k_sft_with_id_and_query_less_than_0.jsonl"
    output_file_qt_less_than_0 = f"{base_dir}/distill_r1_110k_sft_with_id_and_qt_less_than_0.jsonl"
    output_file_query_greater_than_0 = f"{base_dir}/distill_r1_110k_sft_with_id_and_query_greater_than_0.jsonl"
    output_file_query_sorted = f"{base_dir}/distill_r1_110k_sft_with_id_and_query_sorted.jsonl"
    output_file_think_sorted = f"{base_dir}/distill_r1_110k_sft_with_id_and_think_sorted.jsonl"
    output_file_think_ratio_sorted = f"{base_dir}/distill_r1_110k_sft_with_id_and_think_ratio_ssorted.jsonl"
    
    save_fig_dir = f"{base_dir}/{model_size}_2_results/"
    os.makedirs(save_fig_dir, exist_ok=True)
    
    # Load data
    qa_data_map = read_json_data(f'{base_dir}/output_{model_size}_qa.jsonl')
    qta_data_map = read_json_data(f'{base_dir}/output_{model_size}_qta.jsonl')
    ta_data_map = read_json_data(f'{base_dir}/output_{model_size}_ta.jsonl')
    a_data_map = read_json_data(f'{base_dir}/output_{model_size}_a.jsonl')
    
    # Calculate differences (using think_value_ratio)
    think_value_list = []
    query_value_list = []
    think_value_map = {}
    query_value_map = {}
    
    for trace_id in qa_data_map:
        if trace_id in qta_data_map:
            # think_value = (qa_data_map[trace_id] - qta_data_map[trace_id] + a_data_map[trace_id] - ta_data_map[trace_id])/2
            # query_value = (a_data_map[trace_id] - qa_data_map[trace_id] + ta_data_map[trace_id] - qta_data_map[trace_id])/2
            think_value = qa_data_map[trace_id] - qta_data_map[trace_id]
            if (think_value == 0):
                print(1)
            query_value = 0
            think_value_list.append(think_value)
            query_value_list.append(query_value)
            think_value_map[trace_id] = think_value
            query_value_map[trace_id] = query_value
    
    # Map trace_ids to repo_names and values
    id_value_datasets_map = {}
    data_with_values = []  # Store all data entries with calculated values
    
    with open(input_file_path, 'r') as f, \
         open(output_file_think_less_than_0, 'w') as f_think_less_than_0, \
         open(output_file_think_greater_than_0, 'w') as f_think_greater_than_0, \
         open(output_file_query_less_than_0, 'w') as f_query_less_than_0, \
         open(output_file_qt_less_than_0, 'w') as f_qt_less_than_0, \
         open(output_file_query_greater_than_0, 'w') as f_query_greater_than_0:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            if trace_id in think_value_map:
                repo_name = d['repo_name']
                d['think_value'] = think_value_map[trace_id]
                d['query_value'] = query_value_map[trace_id]
                # d['think_ratio'] = think_value_map[trace_id]/(query_value_map[trace_id]+ think_value_map[trace_id])
                d['think_ratio'] = (qa_data_map[trace_id] - qta_data_map[trace_id])/(qa_data_map[trace_id])
                id_value_datasets_map[trace_id] = [repo_name, think_value_map[trace_id], query_value_map[trace_id], qta_data_map[trace_id], a_data_map[trace_id], qa_data_map[trace_id]]
                data_with_values.append(d)  # Add to our list for sorting later
                
                if think_value_map[trace_id] < 0:
                    f_think_less_than_0.write(json.dumps(d) + '\n')
                else:
                    f_think_greater_than_0.write(json.dumps(d) + '\n')
                if query_value_map[trace_id] < 0:
                    f_query_less_than_0.write(json.dumps(d) + '\n')
                else:
                    f_query_greater_than_0.write(json.dumps(d) + '\n')
                if (d['think_value'] +  d['query_value'] <= 0 ):
                    f_qt_less_than_0.write(json.dumps(d) + '\n')
    
    # Sort data by think_value and query_value
    think_sorted_data = sorted(data_with_values, key=lambda x: float(x['think_value']))
    query_sorted_data = sorted(data_with_values, key=lambda x: float(x['query_value']))
    think_ratio_sorted_data = sorted(data_with_values, key=lambda x: float(x['think_ratio']))
    
    # Save sorted data to files
    with open(output_file_think_sorted, 'w') as f_think_sorted:
        for d in think_sorted_data:
            f_think_sorted.write(json.dumps(d,ensure_ascii=False) + '\n')
            
    with open(output_file_query_sorted, 'w') as f_query_sorted:
        for d in query_sorted_data:
            f_query_sorted.write(json.dumps(d,ensure_ascii=False) + '\n')
    with open(output_file_think_ratio_sorted, 'w') as f_sorted:
        for d in think_ratio_sorted_data:
            f_sorted.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    # Group data by repo_name
    repo_think_data = {}
    repo_query_data = {}
    repo_qta_ppl_data = {}
    repo_qa_ppl_data = {}
    repo_a_ppl_data = {}
    repo_qt_value = {}
    repo_think_ratio = {}

    for trace_id, [repo_name, think_value, query_value, answer_ppl, answer_single_ppl, qa_ppl] in id_value_datasets_map.items():
        if repo_name not in repo_think_data:
            repo_think_data[repo_name] = []
        if repo_name not in repo_query_data:
            repo_query_data[repo_name] = []
        if repo_name not in repo_qta_ppl_data:
            repo_qta_ppl_data[repo_name] = []
        if repo_name not in repo_qa_ppl_data:
            repo_qa_ppl_data[repo_name] = []
        if repo_name not in repo_a_ppl_data:
            repo_a_ppl_data[repo_name] = []
        if repo_name not in repo_qt_value:
            repo_qt_value[repo_name] = []
        if repo_name not in repo_think_ratio:
            repo_think_ratio[repo_name] = []
        repo_think_data[repo_name].append(think_value)
        repo_query_data[repo_name].append(query_value)
        repo_qta_ppl_data[repo_name].append(answer_ppl)
        repo_qa_ppl_data[repo_name].append(qa_ppl)
        repo_a_ppl_data[repo_name].append(answer_single_ppl)
        repo_qt_value[repo_name].append(answer_single_ppl - answer_ppl)
        repo_think_ratio[repo_name].append(think_value/qa_data_map[trace_id])
    
    # # Plot individual distributions for each repo
    # for repo_name, values in repo_think_data.items():
    #     save_fig_path = os.path.join(save_fig_dir, f'Think_{repo_name.replace("/", "_")}_distribution.png')
    #     plot_dis(values, save_fig_path, title=f"Distribution for {repo_name} (n={len(values)})")

    # for repo_name, values in repo_query_data.items():
    #     save_fig_path = os.path.join(save_fig_dir, f'Query_{repo_name.replace("/", "_")}_distribution.png')
    #     plot_dis(values, save_fig_path, title=f"Distribution for {repo_name} (n={len(values)})")
    
    # Define categories for repositories
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
    
    # Plot overlapping distributions by category
    for category, repo_list in categories.items():
        # Filter repositories for this category
        category_think_data = {repo: values for repo, values in repo_think_data.items() if repo in repo_list}
        category_query_data = {repo: values for repo, values in repo_query_data.items() if repo in repo_list}
        category_qta_ppl_data = {repo: values for repo, values in repo_qta_ppl_data.items() if repo in repo_list}
        category_qa_ppl_data = {repo: values for repo, values in repo_qa_ppl_data.items() if repo in repo_list}
        category_a_ppl_data = {repo: values for repo, values in repo_a_ppl_data.items() if repo in repo_list}
        category_qt_data = {repo: values for repo, values in repo_qt_value.items() if repo in repo_list}
        category_think_ratio = {repo: values for repo, values in repo_think_ratio.items() if repo in repo_list}

        
        if category_think_data:
            save_fig_path = os.path.join(save_fig_dir, f'Think_value_{category}_overlapping.png')
            plot_overlapping_distributions(category_think_data, save_fig_path, 
                                          title=f"Think Value Distribution - {category.upper()} Category")
        
        if category_query_data:
            save_fig_path = os.path.join(save_fig_dir, f'Query_value_{category}_overlapping.png')
            plot_overlapping_distributions(category_query_data, save_fig_path, 
                                          title=f"Query Value Distribution - {category.upper()} Category",
                                          value_type="Query")
        if category_qta_ppl_data:
            save_fig_path = os.path.join(save_fig_dir, f'qta_ppl_{category}_overlapping.png')
            plot_overlapping_distributions(category_qta_ppl_data, save_fig_path, 
                                          title=f"qta ppl Distribution - {category.upper()} Category",
                                          value_type="qta")
        if category_qa_ppl_data:
            save_fig_path = os.path.join(save_fig_dir, f'qa_ppl_{category}_overlapping.png')
            plot_overlapping_distributions(category_qa_ppl_data, save_fig_path, 
                                          title=f"qa ppl Distribution - {category.upper()} Category",
                                          value_type="qa")
        if category_a_ppl_data:
            save_fig_path = os.path.join(save_fig_dir, f'a_ppl_{category}_overlapping.png')
            plot_overlapping_distributions(category_a_ppl_data, save_fig_path, 
                                          title=f"a ppl Distribution - {category.upper()} Category",
                                          value_type="a")
        if category_qt_data:
            save_fig_path = os.path.join(save_fig_dir, f'qt_value_{category}_overlapping.png')
            plot_overlapping_distributions(category_qt_data, save_fig_path, 
                                          title=f"qt value Distribution - {category.upper()} Category",
                                          value_type="qt_value")
        if category_think_ratio:
            save_fig_path = os.path.join(save_fig_dir, f'Think_ratio_{category}_overlapping.png')
            plot_overlapping_distributions(category_think_ratio, save_fig_path, 
                                          title=f"Think ratio Distribution - {category.upper()} Category",
                                          value_type="Think_ratio")
    
    # Print repository sample counts
    print("Repository sample counts:")
    for repo_name, values in sorted(repo_think_data.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{repo_name}: {len(values)}")
    
    # Analyze token length statistics
    token_stats = {
        'reasoning_lengths': [],
        'content_lengths': [],
        'ratios': []
    }
    
    repo_token_stats = {}
    
    with open(input_file_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            if trace_id in think_value_map:
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
    
    # Print overall token statistics
    print("\nToken Length Statistics:")
    print_stats(token_stats['reasoning_lengths'], "Reasoning token length")
    print_stats(token_stats['content_lengths'], "Content token length")
    print_stats(token_stats['ratios'], "Reasoning/Content ratio")
    
    # Print repo-specific statistics
    print("\nMedian Statistics by Repository:")
    for repo_name, stats in sorted(repo_token_stats.items(), key=lambda x: len(x[1]['reasoning_lengths']), reverse=True):
        count = len(stats['reasoning_lengths'])
        median_reasoning = np.median(stats['reasoning_lengths'])
        median_content = np.median(stats['content_lengths'])
        median_ratio = np.median(stats['ratios'])
        print(f"{repo_name} (n={count}): reasoning={median_reasoning:.2f}, content={median_content:.2f}, ratio={median_ratio:.2f}")

if __name__ == "__main__":
    main()




"""
grep "我是由中国的深度求索（DeepSeek）公司" doc_qa/super_filter_think/thinking_value/data/distill_r1_110k_sft_with_id.jsonl | wc -l
124

# query 的价值-(Loss(ta|q) - Loss(ta|empty))

# think的价值，应该是给定q的情况下，t出不出现对a的影响 Loss(a|q) - Loss(a|qt)


"""

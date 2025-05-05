import json
import numpy as np
import seaborn as sns
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict
import argparse

from transformers import AutoTokenizer
# import matplotlib
# print(matplotlib.get_cachedir())

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

"""
    # general
    'coig/neo': 52893, 
    'zhihu/zhihu_score9.0-10_clean_v10': 2534,
    'xhs/xhs': 1507,
    'ruozhiba/ruozhiba_ruozhiba': 240, 
    'human_value/100poison': 764,  
    'logi_qa/logi-qa': 414, 

    # math
    'Haijian/Advanced-Math': 570, 
    'meta-math/GSM8K_zh': 8776, 
    'EduChat-Math': 19729,  
    'gavinluo/applied_math': 7493,  

    # stem
    'stem_zh/chem': 3157, 
    'stem_zh/bio': 3147, 
    'stem_zh/med': 3163, 
    'stem_zh/phy': 3181,

    # exam
    'exam/coig_exam': 1954, 
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
            # data_dict[data['trace_id']] = data['mean(log_probability)']
            data_dict[data['trace_id']] = data['mean(log_probability)']
            # data_dict[data['trace_id']] = np.exp(data['mean(log_probability)'])
    return data_dict


def print_stats(data, label):
    """Print statistical information for a dataset"""
    print(f"{label}: avg={np.mean(data):.2f}, median={np.median(data):.2f}, min={min(data)}, max={max(data)}")


def plot_overlapping_distributions(data_dict, save_path, title=None, value_type="None"):
    # categories_rename = {
    #     "coig/neo": "neo",
    #     "zhihu/zhihu_score9.0-10_clean_v10": "知乎",
    #     "xhs/xhs": "小红书", 
    #     "ruozhiba/ruozhiba_ruozhiba": "弱智吧",
    #     "human_value/100poison": "百毒",
    #     "logi_qa/logi-qa": "逻辑问答",
    #     "Haijian/Advanced-Math": "高等数学",
    #     "meta-math/GSM8K_zh": "小学数学", 
    #     "EduChat-Math": "应试数学",
    #     "gavinluo/applied_math": "应用数学",
    #     "stem_zh/chem": "化学", 
    #     "stem_zh/bio": "生物", 
    #     "stem_zh/med": "医学", 
    #     "stem_zh/phy": "物理",
    #     "exam/coig_exam": "常识考试", 
    #     "exam/kaoyan": "考研", 
    #     "human_value/coig_human": "价值观问答",
    # }
    """Plot distributions with improved visualization"""
    plt.figure(figsize=(15, 8))  # Increased figure width to accommodate larger legend
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_dict)))
    
    # for i, (name, values) in enumerate(sorted(data_dict.items(), key=lambda x: np.mean(x[1]), reverse=True)):
    #     if len(values) > 1:
    #         sns.kdeplot(values, fill=True, alpha=0.4, color=colors[i], 
    #                    linewidth=2, label=f"{name} (mean={np.mean(values):.4f})")
            
    for i, (name, values) in enumerate(sorted(data_dict.items(), key=lambda x: np.median(x[1]), reverse=True)):
        if value_type == "think_ratio":
            print("think_ratio", name, f"{np.median(values):.4f}")
        elif value_type == "query_ratio":
            print("query_ratio", name, f"{np.median(values):.4f}")
        elif value_type == "qt_ratio":
            print("qtanda_diff", name, f"{1-np.median(values):.4f}")
            
        if len(values) > 1:
            # sns.kdeplot(values, fill=True, alpha=0.4, color=colors[i], 
            #            linewidth=2, label=f"{name} (median={np.median(values):.4f})")
            sns.kdeplot(values, fill=True, alpha=0.4, color=colors[i], 
                       linewidth=2, label=f"{name} (中位数：{np.median(values):.4f})")
            # sns.kdeplot(values, fill=True, alpha=0.4, color=colors[i], 
            #            linewidth=2, label=f"{name} (均值：{np.mean(values):.4f})")
            # plt.axvline(x=np.median(values), color=colors[i], linestyle='--', alpha=0.7)
    
    # plt.axvline(x=0, color='black', linestyle='-', linewidth=1, label='Zero Position')
    plt.title(title or f"{value_type} Distribution by Repository", fontsize=14)
    plt.xlabel("数值"), plt.ylabel("密度")
    plt.grid(True, alpha=0.3)
    
    # Updated legend formatting
    # plt.legend(fontsize=10,  # Increased font size
    #           loc='upper right',  
    #           bbox_to_anchor=(1.02, 0.5),  # Moved legend outside plot
    #           borderaxespad=0)

    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust layout with more space for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.5)  # Make room for legend on right
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
            think_value1 = (data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]) / 2
            query_value1 = (data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id] + 
                         data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]) / 2
            think_ratio1 = think_value1 / data_maps['a_empty'][trace_id] 
            query_ratio1 = query_value1 / data_maps['a_empty'][trace_id] 
            think_ratio7 = think_value1 / (data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]) if data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id] != 0 else 0
            query_ratio7 = query_value1 / (data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]) if data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id] != 0 else 0
            qt_value = data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]
            qt_ratio = (qt_value) / data_maps['a_empty'][trace_id]
            
            # choice 2
            think_value2 = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            query_value2 = data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id]
            think_ratio2 = think_value2/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            think_ratio2_2 = think_value2/data_maps['a_empty'][trace_id]
            query_ratio2 = query_value2/data_maps['ta_empty'][trace_id]
            query_ratio6 = query_value2/data_maps['a_empty'][trace_id]
            
            # choice 3
            think_value3 = data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]
            query_value3 = data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]
            think_ratio3 = think_value3/data_maps['a_empty'][trace_id]
            query_ratio3 = query_value3/data_maps['a_t'][trace_id]
            query_ratio3_2 = query_value3/data_maps['a_empty'][trace_id]
            query_value4 = data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id]
            query_ratio4 = query_value4/data_maps['a_empty'][trace_id]

            query_value5 = (data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id] + data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id])/2
            query_ratio5 = query_value5/data_maps['a_empty'][trace_id]


            think_value4 = data_maps['a_empty'][trace_id] - data_maps['ta_empty'][trace_id]
            think_ratio4 = think_value4/data_maps['a_empty'][trace_id]

            think_value5 = (data_maps['a_q'][trace_id] - data_maps['ta_q'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['ta_empty'][trace_id]) / 2
            think_ratio5 = think_value5/data_maps['a_empty'][trace_id]


            qt_value2 = (qt_value) * d['content_tokens_len']

            repo = d['repo_name']
            repo_data[repo]['think_value'].append(think_value1)
            repo_data[repo]['query_value'].append(query_value1)
            repo_data[repo]['think_ratio'].append(think_ratio1)
            repo_data[repo]['query_ratio'].append(query_ratio1)
            repo_data[repo]['think_value2'].append(think_value2)
            repo_data[repo]['query_value2'].append(query_value2)
            repo_data[repo]['think_ratio2'].append(think_ratio2)
            repo_data[repo]['think_ratio2_2'].append(think_ratio2_2)
            repo_data[repo]['query_ratio2'].append(query_ratio2)
            repo_data[repo]['think_value3'].append(think_value3)
            repo_data[repo]['query_value3'].append(query_value3)
            repo_data[repo]['think_ratio3'].append(think_ratio3)
            repo_data[repo]['query_ratio3'].append(query_ratio3)
            repo_data[repo]['query_ratio3_2'].append(query_ratio3_2)
            repo_data[repo]['query_value4'].append(query_value4)
            repo_data[repo]['query_ratio4'].append(query_ratio4)
            repo_data[repo]['query_value5'].append(query_value5)
            repo_data[repo]['query_ratio5'].append(query_ratio5)
            repo_data[repo]['query_ratio6'].append(query_ratio6)
            repo_data[repo]['think_value4'].append(think_value4)
            repo_data[repo]['think_ratio4'].append(think_ratio4)
            repo_data[repo]['think_value5'].append(think_value5)
            repo_data[repo]['think_ratio5'].append(think_ratio5)
            repo_data[repo]['think_ratio7'].append(think_ratio7)
            repo_data[repo]['query_ratio7'].append(query_ratio7)
            repo_data[repo]['qt_ratio'].append(qt_ratio)
            repo_data[repo]['qt_value'].append(qt_value)
            repo_data[repo]['qt_value2'].append(qt_value2)
            repo_data[repo]['reason_tokens'].append(d['reasoning_content_tokens_len'])
            repo_data[repo]['score'].append(d['score'])

            # repo_data[repo]['a_qt_ppl'].append(data_maps['a_qt'][trace_id])
            # repo_data[repo]['a_q_ppl'].append(data_maps['a_q'][trace_id])
            # repo_data[repo]['qt_value'].append(data_maps['a_qt'][trace_id] - data_maps['a_q'][trace_id])
            # Calculate think_ratio safely
    
    return repo_data


def main():
    # Configuration
    base_dir = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data"
    # model_size = "1_5B"
    # model_size = "7B"
    model_size = "14B"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="1_5B")
    args = parser.parse_args()
    model_size = args.model_size
    print(model_size)
    save_fig_dir = f"{base_dir}/{model_size}_results/"
    os.makedirs(save_fig_dir, exist_ok=True)
    
    # Define output files
    output_files = {
        'query_value_less_0': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_query_value_less_0.jsonl",
        'think_value_less_than_query_value': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_think_value_less_than_query_value.jsonl",
        # 'refuse': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_refuse.jsonl",
        # 'query_value_sorted': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_query_value_sorted.jsonl",
        # 'think_value_sorted': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_think_value_sorted.jsonl",
        # 'query_ratio_sorted': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_query_ratio_sorted.jsonl",
        'think_ratio_sorted': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl",
        # 'qt_ratio_sorted': f"{base_dir}/{model_size}_distill_r1_110k_sft_with_id_and_qt_ratio_sorted.jsonl"
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
            think_value1 = (data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]) / 2
            query_value1 = (data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id] + 
                         data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]) / 2
            think_ratio1 = think_value1 / data_maps['a_empty'][trace_id] 
            query_ratio1 = query_value1 / data_maps['a_empty'][trace_id]
            think_ratio7 = think_value1 / (data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]) if data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id] != 0 else 0
            query_ratio7 = query_value1 / (data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]) if data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id] != 0 else 0
            qt_value = data_maps['a_empty'][trace_id] - data_maps['a_qt'][trace_id]
            qt_ratio = (qt_value) / data_maps['a_empty'][trace_id]
            
            # choice 2
            think_value2 = data_maps['a_q'][trace_id] - data_maps['a_qt'][trace_id]
            query_value2 = data_maps['ta_empty'][trace_id] - data_maps['ta_q'][trace_id]
            think_ratio2 = think_value2/data_maps['a_q'][trace_id] if data_maps['a_q'][trace_id] != 0 else 0
            think_ratio2_2 = think_value2/data_maps['a_empty'][trace_id]
            query_ratio2 = query_value2/data_maps['ta_empty'][trace_id]
            query_ratio6 = query_value2/data_maps['a_empty'][trace_id]
            
            # choice 3
            think_value3 = data_maps['a_empty'][trace_id] - data_maps['a_t'][trace_id]
            query_value3 = data_maps['a_t'][trace_id] - data_maps['a_qt'][trace_id]
            think_ratio3 = think_value3/data_maps['a_empty'][trace_id]
            query_ratio3 = query_value3/data_maps['a_t'][trace_id]
            query_ratio3_2 = query_value3/data_maps['a_empty'][trace_id]

            query_value4 = data_maps['a_empty'][trace_id] - data_maps['a_q'][trace_id]
            query_ratio4 = query_value4/data_maps['a_empty'][trace_id]

            query_value5 = (query_value4 + query_value2)/2
            query_ratio5 = query_value5/data_maps['a_empty'][trace_id]



            think_value4 = data_maps['a_empty'][trace_id] - data_maps['ta_empty'][trace_id]
            think_ratio4 = think_value4/data_maps['a_empty'][trace_id]

            think_value5 = (data_maps['a_q'][trace_id] - data_maps['ta_q'][trace_id] + 
                         data_maps['a_empty'][trace_id] - data_maps['ta_empty'][trace_id]) / 2
            think_ratio5 = think_value5/data_maps['a_empty'][trace_id]



            qt_value2 = (qt_value) * d['content_tokens_len']

            d.update({
                'think_value': think_value1,
                'query_value': query_value1,
                'think_ratio': think_ratio1,
                'query_ratio': query_ratio1,
                'think_value2': think_value2,
                'query_value2': query_value2,
                'think_ratio2': think_ratio2,
                'think_ratio2_2': think_ratio2_2,
                'query_ratio2': query_ratio2,
                'think_value3': think_value3,
                'query_value3': query_value3,
                'think_ratio3': think_ratio3,
                'query_ratio3': query_ratio3,
                'query_ratio3_2': query_ratio3_2,
                'query_value4': query_value4,
                'query_ratio4': query_ratio4,
                'query_value5': query_value5,
                'query_ratio5': query_ratio5,
                'query_ratio6': query_ratio6,
                'think_ratio4': think_ratio4,
                'think_value4': think_value4,
                'think_ratio5': think_ratio5,
                'think_value5': think_value5,
                'think_ratio7': think_ratio7,
                'query_ratio7': query_ratio7,
                'qt_value': qt_value,
                'qt_ratio': qt_ratio,
                'qt_value2': qt_value2,
                'a_qt': data_maps['a_qt'][trace_id],
                'a_t': data_maps['a_t'][trace_id],
                'a_q': data_maps['a_q'][trace_id],
                'a_empty': data_maps['a_empty'][trace_id],
                'ta_q': data_maps['ta_q'][trace_id],
                'ta_empty': data_maps['ta_empty'][trace_id],
            })
            all_data.append(d)

    # Write sorted files
    for key, data in [
        # ('query_value_sorted', sorted(all_data, key=lambda x: x['query_value'])),
        # ('think_value_sorted', sorted(all_data, key=lambda x: x['think_value'])),
        # ('query_ratio_sorted', sorted(all_data, key=lambda x: x['query_ratio'])),
        ('think_ratio_sorted', sorted(all_data, key=lambda x: x['think_ratio'])),
    ]:
        with open(output_files[key], 'w') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(output_files["query_value_less_0"], 'w') as f:
        for d in all_data:
            if d["query_value"] < 0:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # with open(output_files["refuse"], 'w') as f:
    #     for d in all_data:
    #         if "我是由中国的深度求索（DeepSeek）公司" in d["think"] or "我是由中国的深度求索（DeepSeek）公司" in d["response"]:
    #             f.write(json.dumps(d, ensure_ascii=False) + '\n')
    

    think_large_query_ratio = 0
    think_large_qt_ratio = 0
    with open(output_files["think_value_less_than_query_value"], 'w') as f:
        for d in all_data:
            if d["query_value"] < d["think_value"]:
                think_large_query_ratio += 1
            else:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
            if d["a_t"] > d["a_qt"]:
                think_large_qt_ratio += 1
    think_large_query_ratio /= len(data)
    think_large_qt_ratio /= len(data)

    # Continue with existing visualization code
    repo_data = process_data(input_path, data_maps)
    
    # Define categories
    categories = {
        "General": ["coig/neo", "zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", 
                   "ruozhiba/ruozhiba_ruozhiba", "human_value/100poison", "logi_qa/logi-qa"],
        "Community": ["zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", "ruozhiba/ruozhiba_ruozhiba"],
        "Math": ["Haijian/Advanced-Math", "meta-math/GSM8K_zh", "EduChat-Math", "gavinluo/applied_math"],
        "STEM": ["stem_zh/chem", "stem_zh/bio", "stem_zh/med", "stem_zh/phy"],
        "Exam": ["exam/coig_exam", "exam/kaoyan", "human_value/coig_human"],
        "All": ["coig/neo", "zhihu/zhihu_score9.0-10_clean_v10", "xhs/xhs", 
                   "ruozhiba/ruozhiba_ruozhiba", "human_value/100poison", "logi_qa/logi-qa",
                   "Haijian/Advanced-Math", "EduChat-Math", "meta-math/GSM8K_zh", "gavinluo/applied_math",
                   "stem_zh/phy", "stem_zh/chem", "stem_zh/med", "stem_zh/bio", 
                   "exam/coig_exam", "exam/kaoyan", "human_value/coig_human"]
    }
    
    # Plot distributions for each category
    metrics = [ 'think_value', 'query_value', 'think_ratio', 'query_ratio', \
                'think_value2', 'query_value2', 'think_ratio2', 'query_ratio2',\
                'think_value3', 'query_value3', 'think_ratio3', 'query_ratio3',\
                'reason_tokens', 'qt_ratio', 'qt_value']
    for category, repos in categories.items():
        for metric in metrics:
            category_data = {repo: repo_data[repo][metric] 
                           for repo in repos if repo in repo_data}
            if category_data:
                save_path = os.path.join(save_fig_dir, f'{metric}_{category}_overlapping.png')
                if category == "all":
                  plot_overlapping_distributions(category_data, save_path,
                      title=f"{category}大类下不同数据集的{metric}分布",
                      value_type=metric)
                else:
                    plot_overlapping_distributions(category_data, save_path,
                      title=f"{category}大类下不同数据集的{metric}分布",
                      value_type="None")
    
    # # Calculate and print average scores for each category
    print("\nAverage Scores by Category:")
    for category, repos in categories.items():
        category_scores = []
        repo_avg_scores = {}
        
        for repo in repos:
            scores = repo_data[repo]['score']
            avg_score = np.mean(scores)
            repo_avg_scores[repo] = avg_score
            category_scores.extend(scores)
        
        category_avg = np.mean(category_scores)
        print(f"\n{category.upper()} - Average Score: {category_avg:.4f}")
        for repo, avg_score in sorted(repo_avg_scores.items(), key=lambda x: x[1], reverse=True):
            sample_count = len(repo_data[repo]['score']) if repo in repo_data else 0
            print(f"  {repo} (n={sample_count}): {avg_score:.4f}")

    # Print repository sample counts
    sample_sum = 0
    print("Repository sample counts:")
    for repo_name, values in sorted(repo_data.items(), key=lambda x: len(x[1]['think_value']), reverse=True):
        print(f"{repo_name}: {len(values['think_value'])}")
        sample_sum += len(values['think_value'])
    print("total_samples:", sample_sum)
    
    # Analyze token length statistics
    token_stats = {
        'think_tokens': [],
        'answer_tokens': [],
        'query_tokens': [],
        'think_divide_answer_ratios': []
    }
    
    repo_token_stats = {}
    
    with open(input_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            trace_id = d['trace_id']
            repo_name = d['repo_name']
            think_tokens = d.get('reasoning_content_tokens_len', 0)
            answer_tokens = d.get('content_tokens_len', 0)
            query_tokens = d.get('prompt_tokens_len', 0)
            ratio = think_tokens / answer_tokens if answer_tokens > 0 else 0
            
            # Add to overall stats
            token_stats['think_tokens'].append(think_tokens)
            token_stats['answer_tokens'].append(answer_tokens)
            token_stats['think_divide_answer_ratios'].append(ratio)
            token_stats['query_tokens'].append(query_tokens)
            # Add to repo-specific stats
            if repo_name not in repo_token_stats:
                repo_token_stats[repo_name] = {
                    'think_tokens': [],
                    'answer_tokens': [],
                    'think_divide_answer_ratios': [],
                    'query_tokens': []
                }
            
            repo_token_stats[repo_name]['think_tokens'].append(think_tokens)
            repo_token_stats[repo_name]['answer_tokens'].append(answer_tokens)
            repo_token_stats[repo_name]['think_divide_answer_ratios'].append(ratio)
            repo_token_stats[repo_name]['query_tokens'].append(query_tokens)
    # # Print overall token statistics
    # print("\nToken Length Statistics:")
    # print_stats(token_stats['reasoning_lengths'], "Reasoning token length")
    # print_stats(token_stats['content_lengths'], "Content token length")
    # print_stats(token_stats['ratios'], "Reasoning/Content ratio")
    
    # Print repo-specific statistics
    print("\nMedian Statistics by Repository:")
    for repo_name, stats in sorted(repo_token_stats.items(), key=lambda x: np.median(x[1]['think_divide_answer_ratios']), reverse=True):
        count = len(stats['think_tokens'])
        median_think = np.median(stats['think_tokens'])
        median_answer = np.median(stats['answer_tokens'])
        median_ratio = np.median(stats['think_divide_answer_ratios'])
        median_query = np.median(stats['query_tokens'])
        print(f"{repo_name} (n={count}): query= {median_query} think={median_think}, answer={median_answer}, ratio={median_ratio:.2f}")
    print("Percentage of cases where 'think' plays a dominant role over 'query' in determining 'answer':", think_large_query_ratio)
    print("Percentage of cases where 'think' plays a dominant role over 'query and think' in determining 'answer':", think_large_qt_ratio)
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
原因：think把query的内容包含进去了，计算一下两者最长公共子序列，然后长度大于原本的0.8的认为think把query内容覆盖了，算算比例。


2. think_ratio看不同社区、学科、难度之间的关系；和 reason token数量的关系，说明我们的方案更能够反应推理的价值，在标答上对比。
启示：一方面think部分在对推理能力要求高的地方发挥了重要作用，另一方面在对推理能力低的地方作用不大。这说明DeepSeek R1文风出彩的原因大概率不来自推理而是模型的本身。

解释为啥要在数学上做，因为对think发挥的作用大，推理能力要求高，

3. query ratio可以看做是输入信息量，低难度数学提供的信息多

4. query value小于0时候和拒答之间的关系
两个例子，都是拒答，然后一个和query密切相关的高，另一个低。

5.  model value
"""

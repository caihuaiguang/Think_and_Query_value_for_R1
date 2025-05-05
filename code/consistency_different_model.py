import json
from scipy.stats import spearmanr
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
def read_json_data(file_path, type_name):
    """读取JSON文件并返回trace_id到值的字典"""
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_dict[data['trace_id']] = data[type_name]
            # data_dict[data['trace_id']] = data['think_ratio']
            # data_dict[data['trace_id']] = data['query_ratio']
            # data_dict[data['trace_id']] = data['qt_ratio']
            # data_dict[data['trace_id']] = data['think_value']
            # data_dict[data['trace_id']] = data['query_value']
            # data_dict[data['trace_id']] = data['qt_value']
    return data_dict

# ================== 秩次计算优化 ==================
# def get_rank_dict(values_dict):
#     """更高效的秩次计算"""
#     sorted_items = sorted(values_dict.items(), key=lambda x: -x[1])
#     return {item[0]: rank+1 for rank, item in enumerate(sorted_items)}

def get_rank_dict(values_dict):
    """正确处理并列值的秩次计算"""
    values = list(values_dict.values())
    # 使用负数实现降序排列，method='average'计算平均秩次
    ranks = rankdata([-v for v in values], method='average')
    return {tid: rank for tid, rank in zip(values_dict.keys(), ranks)}
# ================== 关键修改部分 ==================
# 验证所有模型的trace_id完全一致
base_model = '14B'
models = ['1_5B', '7B', base_model]
base_dir = "/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data"
type_list = ['qt_ratio', 'qt_value', 'qt_value2',
             'think_ratio', 'think_ratio2', 'think_ratio2_2', 'think_ratio3', 'think_ratio4', 'think_ratio5', 'think_ratio7',
             'think_value', 'think_value2', 'think_value3', 'think_value4', 'think_value5', 
             'query_ratio', 'query_ratio2', 'query_ratio3', 'query_ratio3_2', 'query_ratio4', 'query_ratio5', 'query_ratio6', 'query_ratio7', 
             'query_value', 'query_value2', 'query_value3', 'query_value4', 'query_value5', 
            ]
for type_name in type_list:
    base_data = read_json_data(f'{base_dir}/{base_model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', type_name)

    base_trace_ids = sorted(base_data.keys())
    model_diffs = {}
    for model in models:
        model_data = read_json_data(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', type_name)
        model_diffs[model] = {tid: model_data[tid] for tid in base_trace_ids}


    # 生成各模型秩次字典
    rank_dicts = {model: get_rank_dict(model_diffs[model]) for model in models}

    # ================== 相关性分析 ==================
    # 准备对比数据
    base_ranks = [rank_dicts[base_model][tid] for tid in base_trace_ids]

    print(f"\n{type_name}的Spearman秩相关系数分析:")
    for model in models:
        # if model == base_model:
        #     continue
        model_ranks = [rank_dicts[model][tid] for tid in base_trace_ids]
        rho, p_value = spearmanr(base_ranks, model_ranks)
        print(f"{base_model} vs {model}:")
        print(f"Spearman={rho:.4f}")
        # print(f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})")
    print("------------")

# # ================== 添加类型间相关性分析 ==================
# print("\n计算各指标与score的相关性:")
# for model in models:
#     print(f"\n{model}模型的指标相关性分析:")
#     # 获取score数据
#     score_data = read_json_data(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', 'score')
#     score_values = [score_data[tid] for tid in base_trace_ids]
    
#     # 计算每个type与score的相关性
#     for type_name in type_list:
#         type_data = read_json_data(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', type_name)
#         type_values = [type_data[tid] for tid in base_trace_ids]
        
#         rho, p_value = spearmanr(score_values, type_values)
#         print(f"{type_name} vs score:")
#         print(f"  ρ = {rho:.4f}")
#         print(f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})")
#     print("------------")

# # ================== 按repo_name分组分析相关性 ==================
# print("\n按repo_name分组计算各指标与score的相关性:")
# # 添加日志文件
# log_file = open(f'{base_dir}/correlation_analysis.log', 'w', encoding='utf-8')

# # 创建散点图保存目录
# plot_dir = f'{base_dir}/correlation_plots'
# os.makedirs(plot_dir, exist_ok=True)

# def create_scatter_plot(x_values, y_values, x_label, y_label, title, save_path):
#     """创建并保存散点图"""
#     plt.figure(figsize=(10, 6))
    
#     # 使用seaborn改善图的外观
#     sns.scatterplot(x=x_values, y=y_values, alpha=0.5)
    
#     # 添加趋势线
#     # z = np.polyfit(x_values, y_values, 1)
#     # p = np.poly1d(z)
#     # plt.plot(x_values, p(x_values), "r--", alpha=0.8)
    
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
    
#     # 添加相关系数信息
#     rho, p_value = spearmanr(x_values, y_values)
#     plt.text(0.05, 0.5, f'spearman秩相关系数 = {rho:.4f}', 
#              transform=plt.gca().transAxes, 
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

# for model in models:
#     print(f"\n{model}模型的指标相关性分析:")
#     log_file.write(f"\n{model}模型的指标相关性分析:\n")
    
#     # 创建模型特定的图表目录
#     model_plot_dir = os.path.join(plot_dir, model)
#     os.makedirs(model_plot_dir, exist_ok=True)
    
#     # 读取所有数据，包括repo_name和score
#     data_by_repo = {}
#     with open(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             repo_name = data['repo_name']
#             if repo_name not in data_by_repo:
#                 data_by_repo[repo_name] = []
#             data_by_repo[repo_name].append(data)
    
#     # 对每个repo_name分别计算相关性
#     for repo_name in sorted(data_by_repo.keys()):
#         print(f"\n{repo_name}数据集:")
#         log_file.write(f"\n{repo_name}数据集:\n")
#         repo_data = data_by_repo[repo_name]
        
#         # 创建repo特定的图表目录
#         repo_plot_dir = os.path.join(model_plot_dir, repo_name)
#         os.makedirs(repo_plot_dir, exist_ok=True)
        
#         # 添加数据验证信息
#         print(f"数据条数: {len(repo_data)}")
        
#         # 提取该repo下的score值
#         score_values = [item['score'] for item in repo_data]
#         print(f"Score值范围: {min(score_values):.4f} - {max(score_values):.4f}")
        
#         # 计算每个type与score的相关性
#         for type_name in type_list:
#             type_values = [item[type_name] for item in repo_data]
#             print(f"{type_name}值范围: {min(type_values):.4f} - {max(type_values):.4f}")
            
#             if len(set(type_values)) <= 1 or len(set(score_values)) <= 1:
#                 message = f"{type_name} vs score: 数据无变化，无法计算相关系数"
#                 print(message)
#                 log_file.write(message + '\n')
#                 continue
            
#             # 创建散点图
#             plot_title = f'{repo_name} - {type_name} vs Score'
#             plot_path = os.path.join(repo_plot_dir, f'{type_name}_vs_score.png')
#             create_scatter_plot(
#                 type_values, score_values,
#                 type_name, 'Score',
#                 plot_title, plot_path
#             )
            
#             rho, p_value = spearmanr(score_values, type_values)
#             messages = [
#                 f"{type_name} vs score:",
#                 f"  ρ = {rho:.4f}",
#                 f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})"
#             ]
#             for msg in messages:
#                 print(msg)
#                 log_file.write(msg + '\n')
#         print("------------")
#         log_file.write("------------\n")

# log_file.close()



# # ================== 按repo_name分组分析相关性 ==================
# print("\n按repo_name分组计算各指标与reasoning_token的相关性:")
# # 添加日志文件
# log_file = open(f'{base_dir}/correlation_analysis_token.log', 'w', encoding='utf-8')

# # 创建散点图保存目录
# plot_dir = f'{base_dir}/correlation_plots_token'
# os.makedirs(plot_dir, exist_ok=True)

# def create_scatter_plot(x_values, y_values, x_label, y_label, title, save_path):
#     """创建并保存散点图"""
#     plt.figure(figsize=(10, 6))
    
#     # 使用seaborn改善图的外观
#     sns.scatterplot(x=x_values, y=y_values, alpha=0.5)
    
#     # 添加趋势线
#     # z = np.polyfit(x_values, y_values, 1)
#     # p = np.poly1d(z)
#     # plt.plot(x_values, p(x_values), "r--", alpha=0.8)
    
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
    
#     # 添加相关系数信息
#     rho, p_value = spearmanr(x_values, y_values)
#     plt.text(0.05, 0.5, f'spearman秩相关系数 = {rho:.4f}', 
#              transform=plt.gca().transAxes, 
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

# for model in models:
#     print(f"\n{model}模型的指标相关性分析:")
#     log_file.write(f"\n{model}模型的指标相关性分析:\n")
    
#     # 创建模型特定的图表目录
#     model_plot_dir = os.path.join(plot_dir, model)
#     os.makedirs(model_plot_dir, exist_ok=True)
    
#     # 读取所有数据，包括repo_name和reasoning_content_tokens_len
#     data_by_repo = {}
#     with open(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             repo_name = data['repo_name']
#             if repo_name not in data_by_repo:
#                 data_by_repo[repo_name] = []
#             data_by_repo[repo_name].append(data)
    
#     # 对每个repo_name分别计算相关性
#     for repo_name in sorted(data_by_repo.keys()):
#         print(f"\n{repo_name}数据集:")
#         log_file.write(f"\n{repo_name}数据集:\n")
#         repo_data = data_by_repo[repo_name]
        
#         # 创建repo特定的图表目录
#         repo_plot_dir = os.path.join(model_plot_dir, repo_name)
#         os.makedirs(repo_plot_dir, exist_ok=True)
        
#         # 添加数据验证信息
#         print(f"数据条数: {len(repo_data)}")
        
#         # 提取该repo下的score值
#         reasoning_content_tokens_len_values = [item['reasoning_content_tokens_len'] for item in repo_data]
#         print(f"reasoning_content_tokens_len值范围: {min(reasoning_content_tokens_len_values):.4f} - {max(reasoning_content_tokens_len_values):.4f}")
        
#         # 计算每个type与score的相关性
#         for type_name in type_list:
#             type_values = [item[type_name] for item in repo_data]
#             print(f"{type_name}值范围: {min(type_values):.4f} - {max(type_values):.4f}")
            
#             if len(set(type_values)) <= 1 or len(set(reasoning_content_tokens_len_values)) <= 1:
#                 message = f"{type_name} vs reasoning_content_tokens_len: 数据无变化，无法计算相关系数"
#                 print(message)
#                 log_file.write(message + '\n')
#                 continue
            
#             # 创建散点图
#             plot_title = f'{repo_name} - {type_name} vs reasoning_content_tokens_len'
#             plot_path = os.path.join(repo_plot_dir, f'{type_name}_vs_reasoning_content_tokens_len.png')
#             create_scatter_plot(
#                 type_values, reasoning_content_tokens_len_values,
#                 type_name, 'reasoning_content_tokens_len',
#                 plot_title, plot_path
#             )
            
#             rho, p_value = spearmanr(reasoning_content_tokens_len_values, type_values)
#             messages = [
#                 f"{type_name} vs reasoning_content_tokens_len:",
#                 f"  ρ = {rho:.4f}",
#                 f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})"
#             ]
#             for msg in messages:
#                 print(msg)
#                 log_file.write(msg + '\n')
#         print("------------")
#         log_file.write("------------\n")

# log_file.close()


# # ================== 按repo_name分组分析相关性 ==================
# print("\n按repo_name分组计算各指标与reasoning_token_ratio的相关性:")
# # 添加日志文件
# log_file = open(f'{base_dir}/correlation_analysis_token_ratio.log', 'w', encoding='utf-8')

# # 创建散点图保存目录
# plot_dir = f'{base_dir}/correlation_plots_token_ratio'
# os.makedirs(plot_dir, exist_ok=True)

# def create_scatter_plot(x_values, y_values, x_label, y_label, title, save_path):
#     """创建并保存散点图"""
#     plt.figure(figsize=(10, 6))
    
#     # 使用seaborn改善图的外观
#     sns.scatterplot(x=x_values, y=y_values, alpha=0.5)
    
#     # 添加趋势线
#     # z = np.polyfit(x_values, y_values, 1)
#     # p = np.poly1d(z)
#     # plt.plot(x_values, p(x_values), "r--", alpha=0.8)
    
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
    
#     # 添加相关系数信息
#     rho, p_value = spearmanr(x_values, y_values)
#     plt.text(0.05, 0.5, f'spearman秩相关系数 = {rho:.4f}', 
#              transform=plt.gca().transAxes, 
#              bbox=dict(facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

# for model in models:
#     print(f"\n{model}模型的指标相关性分析:")
#     log_file.write(f"\n{model}模型的指标相关性分析:\n")
    
#     # 创建模型特定的图表目录
#     model_plot_dir = os.path.join(plot_dir, model)
#     os.makedirs(model_plot_dir, exist_ok=True)
    
#     # 读取所有数据，包括repo_name和reasoning_content_tokens_len
#     data_by_repo = {}
#     with open(f'{base_dir}/{model}_distill_r1_110k_sft_with_id_and_think_ratio_sorted.jsonl', 'r') as file:
#         for line in file:
#             data = json.loads(line)
#             repo_name = data['repo_name']
#             if repo_name not in data_by_repo:
#                 data_by_repo[repo_name] = []
#             data_by_repo[repo_name].append(data)
    
#     # 对每个repo_name分别计算相关性
#     for repo_name in sorted(data_by_repo.keys()):
#         print(f"\n{repo_name}数据集:")
#         log_file.write(f"\n{repo_name}数据集:\n")
#         repo_data = data_by_repo[repo_name]
        
#         # 创建repo特定的图表目录
#         repo_plot_dir = os.path.join(model_plot_dir, repo_name)
#         os.makedirs(repo_plot_dir, exist_ok=True)
        
#         # 添加数据验证信息
#         print(f"数据条数: {len(repo_data)}")
        
#         # 提取该repo下的score值
#         reasoning_content_tokens_len_ratios = [np.log(item['reasoning_content_tokens_len']+1) for item in repo_data]
#         print(f"reasoning_content_tokens_len_ratios值范围: {min(reasoning_content_tokens_len_ratios):.4f} - {max(reasoning_content_tokens_len_ratios):.4f}")
        
#         # 计算每个type与score的相关性
#         for type_name in type_list:
#             type_values = [item[type_name] for item in repo_data]
#             print(f"{type_name}值范围: {min(type_values):.4f} - {max(type_values):.4f}")
            
#             if len(set(type_values)) <= 1 or len(set(reasoning_content_tokens_len_ratios)) <= 1:
#                 message = f"{type_name} vs reasoning_content_tokens_len_ratios: 数据无变化，无法计算相关系数"
#                 print(message)
#                 log_file.write(message + '\n')
#                 continue
            
#             # 创建散点图
#             plot_title = f'{repo_name} - {type_name} vs reasoning_content_tokens_len_ratios'
#             plot_path = os.path.join(repo_plot_dir, f'{type_name}_vs_reasoning_content_tokens_len_ratios.png')
#             create_scatter_plot(
#                 type_values, reasoning_content_tokens_len_ratios,
#                 type_name, 'reasoning_content_tokens_len_ratios',
#                 plot_title, plot_path
#             )
            
#             rho, p_value = spearmanr(reasoning_content_tokens_len_ratios, type_values)
#             messages = [
#                 f"{type_name} vs reasoning_content_tokens_len_ratios:",
#                 f"  ρ = {rho:.4f}",
#                 f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})"
#             ]
#             for msg in messages:
#                 print(msg)
#                 log_file.write(msg + '\n')
#         print("------------")
#         log_file.write("------------\n")

# log_file.close()
"""

qt_ratio的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9664
14B vs 7B:
Spearman=0.9742
14B vs 14B:
Spearman=1.0000
------------

qt_value的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8918
14B vs 7B:
Spearman=0.8945
14B vs 14B:
Spearman=1.0000
------------

qt_value2的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.7244
14B vs 7B:
Spearman=0.7513
14B vs 14B:
Spearman=1.0000
------------

think_ratio的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9369
14B vs 7B:
Spearman=0.9518
14B vs 14B:
Spearman=1.0000
------------

think_ratio2的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8423
14B vs 7B:
Spearman=0.8666
14B vs 14B:
Spearman=1.0000
------------

think_ratio3的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9658
14B vs 7B:
Spearman=0.9738
14B vs 14B:
Spearman=1.0000
------------

think_ratio4的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9036
14B vs 7B:
Spearman=0.9091
14B vs 14B:
Spearman=1.0000
------------

think_ratio5的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9248
14B vs 7B:
Spearman=0.9376
14B vs 14B:
Spearman=1.0000
------------

think_value的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8381
14B vs 7B:
Spearman=0.8299
14B vs 14B:
Spearman=1.0000
------------

think_value2的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.6785
14B vs 7B:
Spearman=0.7231
14B vs 14B:
Spearman=1.0000
------------

think_value3的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8949
14B vs 7B:
Spearman=0.8864
14B vs 14B:
Spearman=1.0000
------------

think_value4的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8934
14B vs 7B:
Spearman=0.8965
14B vs 14B:
Spearman=1.0000
------------

think_value5的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9240
14B vs 7B:
Spearman=0.9373
14B vs 14B:
Spearman=1.0000
------------

query_ratio的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9188
14B vs 7B:
Spearman=0.9351
14B vs 14B:
Spearman=1.0000
------------

query_ratio2的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9133
14B vs 7B:
Spearman=0.9426
14B vs 14B:
Spearman=1.0000
------------

query_ratio3的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.3829
14B vs 7B:
Spearman=0.5713
14B vs 14B:
Spearman=1.0000
------------

query_ratio4的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9261
14B vs 7B:
Spearman=0.9375
14B vs 14B:
Spearman=1.0000
------------

query_ratio5的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9454
14B vs 7B:
Spearman=0.9572
14B vs 14B:
Spearman=1.0000
------------

query_ratio6的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8983
14B vs 7B:
Spearman=0.9289
14B vs 14B:
Spearman=1.0000
------------

query_value的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.8946
14B vs 7B:
Spearman=0.9138
14B vs 14B:
Spearman=1.0000
------------

query_value2的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9000
14B vs 7B:
Spearman=0.9344
14B vs 14B:
Spearman=1.0000
------------

query_value3的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.2740
14B vs 7B:
Spearman=0.5476
14B vs 14B:
Spearman=1.0000
------------

query_value4的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9063
14B vs 7B:
Spearman=0.9150
14B vs 14B:
Spearman=1.0000
------------

query_value5的Spearman秩相关系数分析:
14B vs 1_5B:
Spearman=0.9303
14B vs 7B:
Spearman=0.9415
14B vs 14B:
Spearman=1.0000
------------


"""
import json
from scipy.stats import spearmanr

def read_json_data(file_path):
    """读取JSON文件并返回trace_id到值的字典"""
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_dict[data['trace_id']] = data['mean(log_probability)']
    return data_dict

# ================== 关键修改部分 ==================
# 验证所有模型的trace_id完全一致
base_model = '32B'
models = ['1_5B', '7B', base_model]

# 先读取32B模型的trace_id作为基准
base_data = {
    'False': read_json_data(f'super_filter_think/data_math_{base_model}/output_False.json'),
    'True': read_json_data(f'super_filter_think/data_math_{base_model}/output_True.json')
}
base_trace_ids = sorted(base_data['False'].keys())

# 验证其他模型是否具有完全相同的trace_id
for model in models:
    if model == base_model:
        continue
    
    # 读取当前模型数据
    model_false = read_json_data(f'super_filter_think/data_math_{model}/output_False.json')
    model_true = read_json_data(f'super_filter_think/data_math_{model}/output_True.json')
    
    # 严格验证三个条件
    assert set(model_false.keys()) == set(base_trace_ids), f"{model}模型False数据trace_id不一致"
    assert set(model_true.keys()) == set(base_trace_ids), f"{model}模型True数据trace_id不一致"
    assert len(model_false) == len(model_true) == len(base_trace_ids), f"{model}数据量不一致"

print("所有模型trace_id验证通过，完全一致")

# ================== 数据处理部分 ==================
# 预计算所有模型的差值
model_diffs = {}
for model in models:
    false_data = read_json_data(f'super_filter_think/data_math_{model}/output_False.json')
    true_data = read_json_data(f'super_filter_think/data_math_{model}/output_True.json')
    model_diffs[model] = {tid: false_data[tid]-true_data[tid] for tid in base_trace_ids}

# ================== 秩次计算优化 ==================
def get_rank_dict(values_dict):
    """更高效的秩次计算"""
    sorted_items = sorted(values_dict.items(), key=lambda x: -x[1])
    return {item[0]: rank+1 for rank, item in enumerate(sorted_items)}

# 生成各模型秩次字典
rank_dicts = {model: get_rank_dict(model_diffs[model]) for model in models}

# ================== 相关性分析 ==================
# 准备对比数据
base_ranks = [rank_dicts[base_model][tid] for tid in base_trace_ids]

print("\nSpearman秩相关系数分析:")
for model in models:
    # if model == base_model:
    #     continue
    model_ranks = [rank_dicts[model][tid] for tid in base_trace_ids]
    rho, p_value = spearmanr(base_ranks, model_ranks)
    print(f"{base_model} vs {model}:")
    print(f"  ρ = {rho:.4f}")
    print(f"  p = {p_value:.2e} ({'显著' if p_value<0.05 else '不显著'})")

# ================== 结果验证 ==================
print("\n数据完整性验证:")
print(f"总样本量: {len(base_trace_ids)}")
print(f"32B模型正差值比例: {sum(1 for v in model_diffs[base_model].values() if v>0)/len(base_trace_ids):.2%}")


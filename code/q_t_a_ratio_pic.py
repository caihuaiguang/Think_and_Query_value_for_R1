import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 原始数据解析
raw_data1_5B = """
think_ratio meta-math/GSM8K_zh 0.3937
think_ratio gavinluo/applied_math 0.3872
think_ratio exam/kaoyan 0.3814
think_ratio EduChat-Math 0.3745
think_ratio Haijian/Advanced-Math 0.3514
think_ratio exam/coig_exam 0.3428
think_ratio logi_qa/logi-qa 0.3083
think_ratio human_value/coig_human 0.2924
think_ratio stem_zh/chem 0.1693
think_ratio stem_zh/phy 0.1626
think_ratio stem_zh/med 0.1327
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1322
think_ratio stem_zh/bio 0.1217
think_ratio coig/neo 0.1082
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0985
think_ratio human_value/100poison 0.0863
think_ratio xhs/xhs 0.0679
query_ratio meta-math/GSM8K_zh 0.2957
query_ratio gavinluo/applied_math 0.2481
query_ratio EduChat-Math 0.1972
query_ratio exam/coig_exam 0.1616
query_ratio logi_qa/logi-qa 0.1468
query_ratio exam/kaoyan 0.1464
query_ratio human_value/coig_human 0.1315
query_ratio Haijian/Advanced-Math 0.1135
query_ratio xhs/xhs 0.0272
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0243
query_ratio coig/neo 0.0233
query_ratio stem_zh/phy 0.0218
query_ratio stem_zh/chem 0.0214
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0203
query_ratio stem_zh/med 0.0195
query_ratio stem_zh/bio 0.0187
query_ratio human_value/100poison 0.0156
qtanda_diff meta-math/GSM8K_zh 0.3103
qtanda_diff gavinluo/applied_math 0.3648
qtanda_diff EduChat-Math 0.4223
qtanda_diff exam/kaoyan 0.4655
qtanda_diff exam/coig_exam 0.4882
qtanda_diff Haijian/Advanced-Math 0.5289
qtanda_diff logi_qa/logi-qa 0.5455
qtanda_diff human_value/coig_human 0.5849
qtanda_diff stem_zh/chem 0.8080
qtanda_diff stem_zh/phy 0.8145
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8461
qtanda_diff stem_zh/med 0.8470
qtanda_diff stem_zh/bio 0.8582
qtanda_diff coig/neo 0.8667
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8801
qtanda_diff human_value/100poison 0.8984
qtanda_diff xhs/xhs 0.9039
"""



raw_data7B = """
think_ratio exam/kaoyan 0.3770
think_ratio meta-math/GSM8K_zh 0.3689
think_ratio exam/coig_exam 0.3480
think_ratio EduChat-Math 0.3477
think_ratio gavinluo/applied_math 0.3477
think_ratio Haijian/Advanced-Math 0.3321
think_ratio logi_qa/logi-qa 0.3280
think_ratio human_value/coig_human 0.2950
think_ratio stem_zh/chem 0.1735
think_ratio stem_zh/phy 0.1666
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1495
think_ratio stem_zh/med 0.1344
think_ratio stem_zh/bio 0.1243
think_ratio coig/neo 0.1166
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.1100
think_ratio human_value/100poison 0.0967
think_ratio xhs/xhs 0.0702
query_ratio meta-math/GSM8K_zh 0.2908
query_ratio gavinluo/applied_math 0.2453
query_ratio EduChat-Math 0.1895
query_ratio exam/coig_exam 0.1700
query_ratio exam/kaoyan 0.1447
query_ratio logi_qa/logi-qa 0.1412
query_ratio human_value/coig_human 0.1340
query_ratio Haijian/Advanced-Math 0.1105
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0285
query_ratio coig/neo 0.0227
query_ratio xhs/xhs 0.0210
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0204
query_ratio stem_zh/phy 0.0191
query_ratio stem_zh/chem 0.0190
query_ratio stem_zh/med 0.0171
query_ratio human_value/100poison 0.0167
query_ratio stem_zh/bio 0.0159
qtanda_diff meta-math/GSM8K_zh 0.3413
qtanda_diff gavinluo/applied_math 0.4074
qtanda_diff EduChat-Math 0.4519
qtanda_diff exam/kaoyan 0.4654
qtanda_diff exam/coig_exam 0.4727
qtanda_diff logi_qa/logi-qa 0.5312
qtanda_diff Haijian/Advanced-Math 0.5575
qtanda_diff human_value/coig_human 0.5738
qtanda_diff stem_zh/chem 0.8074
qtanda_diff stem_zh/phy 0.8133
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8259
qtanda_diff stem_zh/med 0.8462
qtanda_diff stem_zh/bio 0.8584
qtanda_diff coig/neo 0.8586
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8678
qtanda_diff human_value/100poison 0.8860
qtanda_diff xhs/xhs 0.9089
"""

raw_data14B = """
think_ratio logi_qa/logi-qa 0.3868
think_ratio EduChat-Math 0.3813
think_ratio meta-math/GSM8K_zh 0.3772
think_ratio exam/kaoyan 0.3753
think_ratio gavinluo/applied_math 0.3697
think_ratio Haijian/Advanced-Math 0.3544
think_ratio exam/coig_exam 0.3298
think_ratio human_value/coig_human 0.2958
think_ratio stem_zh/phy 0.1475
think_ratio stem_zh/chem 0.1474
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1359
think_ratio stem_zh/med 0.1079
think_ratio coig/neo 0.1036
think_ratio stem_zh/bio 0.1005
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0925
think_ratio human_value/100poison 0.0774
think_ratio xhs/xhs 0.0621
query_ratio meta-math/GSM8K_zh 0.3016
query_ratio gavinluo/applied_math 0.2524
query_ratio EduChat-Math 0.2070
query_ratio exam/coig_exam 0.1835
query_ratio logi_qa/logi-qa 0.1611
query_ratio exam/kaoyan 0.1597
query_ratio human_value/coig_human 0.1553
query_ratio Haijian/Advanced-Math 0.0965
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0420
query_ratio coig/neo 0.0272
query_ratio stem_zh/med 0.0250
query_ratio stem_zh/chem 0.0249
query_ratio stem_zh/phy 0.0242
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0234
query_ratio stem_zh/bio 0.0213
query_ratio xhs/xhs 0.0199
query_ratio human_value/100poison 0.0166
qtanda_diff meta-math/GSM8K_zh 0.3216
qtanda_diff gavinluo/applied_math 0.3765
qtanda_diff EduChat-Math 0.4000
qtanda_diff logi_qa/logi-qa 0.4443
qtanda_diff exam/kaoyan 0.4516
qtanda_diff exam/coig_exam 0.4788
qtanda_diff Haijian/Advanced-Math 0.5383
qtanda_diff human_value/coig_human 0.5506
qtanda_diff stem_zh/chem 0.8246
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8263
qtanda_diff stem_zh/phy 0.8274
qtanda_diff stem_zh/med 0.8658
qtanda_diff coig/neo 0.8666
qtanda_diff stem_zh/bio 0.8778
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8795
qtanda_diff human_value/100poison 0.9050
qtanda_diff xhs/xhs 0.9178
"""
raw_data = raw_data1_5B
model_name = "1_5B"
# raw_data = raw_data7B
# model_name = "7B"
# raw_data = raw_data14B
# model_name = "14B"
# 解析数据
data_dict = {}
for line in raw_data.strip().split('\n'):
    parts = line.split()
    metric = parts[0].replace("qtanda_diff", "qt&a_diff")  # 统一键名
    dataset = parts[1]
    value = float(parts[2])
    
    if dataset not in data_dict:
        data_dict[dataset] = {}
    data_dict[dataset][metric] = value

# 按think_ratio排序
sorted_datasets = sorted(data_dict.keys(), 
                        key=lambda x: 1 -  data_dict[x]['qt&a_diff'], 
                        reverse=True)

# 构建排序后的数据
data = {
    "think_ratio": [data_dict[ds]["think_ratio"] for ds in sorted_datasets],
    "query_ratio": [data_dict[ds]["query_ratio"] for ds in sorted_datasets],
    "qt&a_diff": [data_dict[ds]["qt&a_diff"] for ds in sorted_datasets]
}

# 绘图参数
values = np.array(list(data.values()))
x = np.arange(len(sorted_datasets))
bar_width = 0.8
colors = ['red', 'orange', 'gainsboro']

# 创建图形
fig, ax = plt.subplots(figsize=(18, 6))
bottom = np.zeros(len(sorted_datasets))

# 绘制堆叠柱状图
for i, (label, value) in enumerate(data.items()):
    ax.bar(x, value, bar_width, label=label, bottom=bottom, color=colors[i])
    bottom += value

# 设置坐标轴
ax.set_xticks(x)
ax.set_xticklabels([str(i+1) for i in range(len(sorted_datasets))])
ax.set_xlabel('数据集标号')
ax.set_ylabel('贡献比重')
if model_name == "1_5B":
    ax.set_title(f'不同数据集的query、think贡献占比，模型：DeepSeek-R1-Distill-Qwen-1.5B')
else:
    ax.set_title(f'不同数据集的query、think贡献占比，模型：DeepSeek-R1-Distill-Qwen-{model_name}')
ax.legend(loc='upper left')

# 右侧图例
text_x = 1.02
text_y = 0.98
line_spacing = 0.035
dataset_text = "\n".join([f"{i+1}: {name}" for i, name in enumerate(sorted_datasets)])

props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
            edgecolor='black', pad=0.6)
plt.text(text_x, text_y, dataset_text, 
        transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='left',
        bbox=props)

plt.subplots_adjust(right=0.55)  # 调整右侧空白
# plt.show()
# 保存图片
plt.savefig(f'/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/data/{model_name}_results/{model_name}_q_t_a_ratio_pic.png', dpi=300, bbox_inches='tight')

# plt.show()


"""
1_5B
think_ratio meta-math/GSM8K_zh 0.3937
think_ratio gavinluo/applied_math 0.3872
think_ratio exam/kaoyan 0.3814
think_ratio EduChat-Math 0.3745
think_ratio Haijian/Advanced-Math 0.3514
think_ratio exam/coig_exam 0.3428
think_ratio logi_qa/logi-qa 0.3083
think_ratio human_value/coig_human 0.2924
think_ratio stem_zh/chem 0.1693
think_ratio stem_zh/phy 0.1626
think_ratio stem_zh/med 0.1327
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1322
think_ratio stem_zh/bio 0.1217
think_ratio coig/neo 0.1082
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0985
think_ratio human_value/100poison 0.0863
think_ratio xhs/xhs 0.0679
query_ratio meta-math/GSM8K_zh 0.2957
query_ratio gavinluo/applied_math 0.2481
query_ratio EduChat-Math 0.1972
query_ratio exam/coig_exam 0.1616
query_ratio logi_qa/logi-qa 0.1468
query_ratio exam/kaoyan 0.1464
query_ratio human_value/coig_human 0.1315
query_ratio Haijian/Advanced-Math 0.1135
query_ratio xhs/xhs 0.0272
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0243
query_ratio coig/neo 0.0233
query_ratio stem_zh/phy 0.0218
query_ratio stem_zh/chem 0.0214
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0203
query_ratio stem_zh/med 0.0195
query_ratio stem_zh/bio 0.0187
query_ratio human_value/100poison 0.0156
qtanda_diff meta-math/GSM8K_zh 0.3103
qtanda_diff gavinluo/applied_math 0.3648
qtanda_diff EduChat-Math 0.4223
qtanda_diff exam/kaoyan 0.4655
qtanda_diff exam/coig_exam 0.4882
qtanda_diff Haijian/Advanced-Math 0.5289
qtanda_diff logi_qa/logi-qa 0.5455
qtanda_diff human_value/coig_human 0.5849
qtanda_diff stem_zh/chem 0.8080
qtanda_diff stem_zh/phy 0.8145
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8461
qtanda_diff stem_zh/med 0.8470
qtanda_diff stem_zh/bio 0.8582
qtanda_diff coig/neo 0.8667
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8801
qtanda_diff human_value/100poison 0.8984
qtanda_diff xhs/xhs 0.9039


"""


"""
7B
think_ratio exam/kaoyan 0.3770
think_ratio meta-math/GSM8K_zh 0.3689
think_ratio exam/coig_exam 0.3480
think_ratio EduChat-Math 0.3477
think_ratio gavinluo/applied_math 0.3477
think_ratio Haijian/Advanced-Math 0.3321
think_ratio logi_qa/logi-qa 0.3280
think_ratio human_value/coig_human 0.2950
think_ratio stem_zh/chem 0.1735
think_ratio stem_zh/phy 0.1666
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1495
think_ratio stem_zh/med 0.1344
think_ratio stem_zh/bio 0.1243
think_ratio coig/neo 0.1166
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.1100
think_ratio human_value/100poison 0.0967
think_ratio xhs/xhs 0.0702
query_ratio meta-math/GSM8K_zh 0.2908
query_ratio gavinluo/applied_math 0.2453
query_ratio EduChat-Math 0.1895
query_ratio exam/coig_exam 0.1700
query_ratio exam/kaoyan 0.1447
query_ratio logi_qa/logi-qa 0.1412
query_ratio human_value/coig_human 0.1340
query_ratio Haijian/Advanced-Math 0.1105
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0285
query_ratio coig/neo 0.0227
query_ratio xhs/xhs 0.0210
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0204
query_ratio stem_zh/phy 0.0191
query_ratio stem_zh/chem 0.0190
query_ratio stem_zh/med 0.0171
query_ratio human_value/100poison 0.0167
query_ratio stem_zh/bio 0.0159
qtanda_diff meta-math/GSM8K_zh 0.3413
qtanda_diff gavinluo/applied_math 0.4074
qtanda_diff EduChat-Math 0.4519
qtanda_diff exam/kaoyan 0.4654
qtanda_diff exam/coig_exam 0.4727
qtanda_diff logi_qa/logi-qa 0.5312
qtanda_diff Haijian/Advanced-Math 0.5575
qtanda_diff human_value/coig_human 0.5738
qtanda_diff stem_zh/chem 0.8074
qtanda_diff stem_zh/phy 0.8133
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8259
qtanda_diff stem_zh/med 0.8462
qtanda_diff stem_zh/bio 0.8584
qtanda_diff coig/neo 0.8586
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8678
qtanda_diff human_value/100poison 0.8860
qtanda_diff xhs/xhs 0.9089

"""


"""
14B
think_ratio logi_qa/logi-qa 0.3868
think_ratio EduChat-Math 0.3813
think_ratio meta-math/GSM8K_zh 0.3772
think_ratio exam/kaoyan 0.3753
think_ratio gavinluo/applied_math 0.3697
think_ratio Haijian/Advanced-Math 0.3544
think_ratio exam/coig_exam 0.3298
think_ratio human_value/coig_human 0.2958
think_ratio stem_zh/phy 0.1475
think_ratio stem_zh/chem 0.1474
think_ratio ruozhiba/ruozhiba_ruozhiba 0.1359
think_ratio stem_zh/med 0.1079
think_ratio coig/neo 0.1036
think_ratio stem_zh/bio 0.1005
think_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0925
think_ratio human_value/100poison 0.0774
think_ratio xhs/xhs 0.0621
query_ratio meta-math/GSM8K_zh 0.3016
query_ratio gavinluo/applied_math 0.2524
query_ratio EduChat-Math 0.2070
query_ratio exam/coig_exam 0.1835
query_ratio logi_qa/logi-qa 0.1611
query_ratio exam/kaoyan 0.1597
query_ratio human_value/coig_human 0.1553
query_ratio Haijian/Advanced-Math 0.0965
query_ratio ruozhiba/ruozhiba_ruozhiba 0.0420
query_ratio coig/neo 0.0272
query_ratio stem_zh/med 0.0250
query_ratio stem_zh/chem 0.0249
query_ratio stem_zh/phy 0.0242
query_ratio zhihu/zhihu_score9.0-10_clean_v10 0.0234
query_ratio stem_zh/bio 0.0213
query_ratio xhs/xhs 0.0199
query_ratio human_value/100poison 0.0166
qtanda_diff meta-math/GSM8K_zh 0.3216
qtanda_diff gavinluo/applied_math 0.3765
qtanda_diff EduChat-Math 0.4000
qtanda_diff logi_qa/logi-qa 0.4443
qtanda_diff exam/kaoyan 0.4516
qtanda_diff exam/coig_exam 0.4788
qtanda_diff Haijian/Advanced-Math 0.5383
qtanda_diff human_value/coig_human 0.5506
qtanda_diff stem_zh/chem 0.8246
qtanda_diff ruozhiba/ruozhiba_ruozhiba 0.8263
qtanda_diff stem_zh/phy 0.8274
qtanda_diff stem_zh/med 0.8658
qtanda_diff coig/neo 0.8666
qtanda_diff stem_zh/bio 0.8778
qtanda_diff zhihu/zhihu_score9.0-10_clean_v10 0.8795
qtanda_diff human_value/100poison 0.9050
qtanda_diff xhs/xhs 0.9178

"""
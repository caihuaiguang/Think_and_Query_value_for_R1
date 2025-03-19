# Introduction
This repository implements a Shapley value-based approach to quantitatively evaluate the contributions of query (q) and think (t) in generating answer (a). 

# Method

```
think_value = [loss(a|q) - loss(a|q,t) + loss(a|∅) - loss(a|t)] / 2
query_value = [loss(a|t) - loss(a|q,t) + loss(a|∅) - loss(a|q)] / 2
think_ratio = think_value/loss(a|∅)
query_ratio = query_value/loss(a|∅)
```

# Dataset

https://huggingface.co/datasets/caihuaiguang/Think_and_Query_value_for_R1

# Result

![Think Ratio Distribution](/data/7B_results/think_ratio_all_overlapping.png)

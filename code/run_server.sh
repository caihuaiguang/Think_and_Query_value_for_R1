export PATH="/data/minimax-dialogue/users/kant/envs/TorchS/bin:$PATH"
which python

# /home/shuishengmu/miniconda3/envs/flashattenv
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type a_qt
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type a_q
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type ta_q
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type ta_empty
# torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_seek.py --type q
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type a_t
torchrun --nproc_per_node=8 /data/minimax-dialogue/users/shuishengmu/doc_qa/super_filter_think/thinking_value/code/ppl_1.py --type a_empty

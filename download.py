import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 这个镜像网站可能也可以换掉
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co/'   # 这个镜像网站可能也可以换掉

from huggingface_hub import snapshot_download

# snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#                   local_dir_use_symlinks=False,
#                   local_dir="/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
snapshot_download(repo_id="Qwen/Qwen3-0.6B",
                  local_dir_use_symlinks=False,
                  local_dir="/root/autodl-tmp/models/Qwen/Qwen3-0.6B")

snapshot_download(repo_id="Qwen/Qwen3-1.7B",
                  local_dir_use_symlinks=False,
                  local_dir="/root/autodl-tmp/models/Qwen/Qwen3-1.7B")

snapshot_download(repo_id="Qwen/Qwen3-4B",
                  local_dir_use_symlinks=False,
                  local_dir="/root/autodl-tmp/models/Qwen/Qwen3-4B")

                  


# snapshot_download(repo_id="Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT",
#                   repo_type="dataset",
#                   local_dir_use_symlinks=False,
#                   local_dir="/root/Think_and_Query_value_for_R1/data/")

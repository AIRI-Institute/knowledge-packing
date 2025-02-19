export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

python ./generate_dataset.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --data_path /path/to/data
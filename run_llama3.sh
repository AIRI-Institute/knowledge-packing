export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

for seed in 42 1337 10824
do
    for unk in 1 10 50 100 500 3000
    do
        python ./lora_train_llama.py --path "meta-llama/Meta-Llama-3.1-8B-Instruct" --unknown $unk --high_known 1 --rank 1 --seed $seed --data_path /path/to/data --paraphrase
        python ./lora_train_llama.py --path "meta-llama/Meta-Llama-3.1-8B-Instruct" --unknown $unk --high_known 10 --rank 1 --seed $seed --data_path /path/to/data --paraphrase
        python ./lora_train_llama.py --path "meta-llama/Meta-Llama-3.1-8B-Instruct" --unknown $unk --high_known 1 --rank 1 --seed $seed --data_path /path/to/data
        python ./lora_train_llama.py --path "meta-llama/Meta-Llama-3.1-8B-Instruct" --unknown $unk --high_known 10 --rank 1 --seed $seed --data_path /path/to/data
    done
done
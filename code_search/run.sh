# conda install pyzmq -y
echo "runing run.sh"
lang=java
current_time=$(date "+%Y%m%d%H%M%S")

code_length=256 
nl_length=128

epoch=30
batch_size=64
lr=2e-4

param=lr_${lr}_${lang}_bs_${batch_size}_e_${epoch}_clen_${code_length}_nlen_${nl_length}
output_dir=./saved_models/pre_trained/$param/${current_time}
echo ${output_dir}
mkdir -p ${output_dir}


function train () {
# --debug 
echo "============TRAINING============"
CUDA_VISIBLE_DEVICES=0 python run.py  --eval_frequency  100 --num_train_epochs ${epoch} \
    --output_dir ${output_dir}  \
    --config_name=microsoft/graphcodebert-base  \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --do_test \
    --train_data_file=dataset/$lang/train_add_ast.jsonl \
    --eval_data_file=dataset/$lang/test.jsonl  \
    --test_data_file=dataset/$lang/valid.jsonl \
    --codebase_file=dataset/$lang/codebase_add_ast.jsonl \
    --code_length ${code_length} \
    --data_flow_length 0 \
    --nl_length ${nl_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size  ${batch_size} \
    --learning_rate ${lr} \
    --seed 3407 2>&1| tee ${output_dir}/train.log
}

 train  
# Examples:
# 1. train_10k
# nohup bash train_nmt.sh 2 LayerNorm absolute 32 3e-4 512 lr_test> log/train_chinesenmt_default_log.txt 2>&1 &
# 2. rnn
# # 核心修改：rnn_baseline 后面加空格，再跟 > 重定向符号
# nohup bash train_nmt.sh 2 LayerNorm absolute 32 3e-4 512 teacher_forcing gru_attention general beam_search rnn_baseline > log/train_chinesenmt_default_log.txt 2>&1 &
gpu_id=${1}  # GPU ID
norm_type=${2}  # LayerNorm or RMSNorm
pos_encoding_type=${3} # absolute or relative
batch_size=${4} # 16, 32, 64, 128
lr=${5} # 1e-4, 3e-4, 5e-4
d_model=${6} # 256, 512, 1024
teacher=${7} # teacher_forcing or no_teacher_forcing
model_type=${8} # gru_attention or transformer
attn_model_type=${9}  # general, dot, concat
decode_mode=${10} # greedy or beam_search
addition=${11} # lr_test, pos_test, norm_test, scale_test

# 定义标志位变量，初始为空
TEACHER_FORCING_FLAG=""
if [ "${teacher}" = "teacher_forcing" ]; then
    # 仅当使用教师强制时，赋值标志位（无多余空格）
    TEACHER_FORCING_FLAG="--use_teacher_forcing"
fi

BEAM_SEARCH_FLAG=""
if [ "${decode_mode}" = "beam_search" ]; then
    # 仅当使用束搜索时，赋值标志位（无多余空格）
    BEAM_SEARCH_FLAG="--beam_search"
fi

cd /home/yiyu/code/homework/NLP/ChineseNMT

export HYDRA_FULL_ERROR=1 
export PYTHONPATH=/home/yiyu/code/homework/NLP/ChineseNMT
export HF_ENDPOINT=https://hf-mirror.com
model_path="/home/yiyu/code/homework/NLP/NMT/experiment/${model_type}/${addition}/model_${batch_size}bs_${lr}lr_${d_model}dm_${norm_type}_${pos_encoding_type}.pth"
output_path="/home/yiyu/code/homework/NLP/NMT/experiment/${model_type}/${addition}/output_${batch_size}bs_${lr}lr_${d_model}dm_${norm_type}_${pos_encoding_type}.txt"
log_path="/home/yiyu/code/homework/NLP/NMT/experiment/log/${model_type}/${addition}/train_chinesenmt_${batch_size}bs_${lr}lr_${d_model}dm_${norm_type}_${pos_encoding_type}_log.txt"
tb_log_path="/home/yiyu/code/homework/NLP/NMT/experiment/${model_type}/tb_log/${addition}/tb_log_${batch_size}bs_${lr}lr_${d_model}dm_${norm_type}_${pos_encoding_type}"

# ========== 核心修改：修复命令行换行和变量引用格式 ==========
python main.py \
    --cuda_visible_devices="${gpu_id}" \
    --norm_type="${norm_type}" \
    --pos_encoding_type="${pos_encoding_type}" \
    --batch_size="${batch_size}" \
    --lr="${lr}" \
    --d_model="${d_model}" \
    --model_type="${model_type}" \
    --attn_model_type="${attn_model_type}" \
    --model_path="${model_path}" \
    --output_path="${output_path}" \
    --log_path="${log_path}" \
    --tb_log_dir="${tb_log_path}" \
    ${TEACHER_FORCING_FLAG} \
    ${BEAM_SEARCH_FLAG}

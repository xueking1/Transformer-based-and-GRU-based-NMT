import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 32
epoch_num = 1
early_stop = 5
lr = 3e-4

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True
norm_type = 'RMSNorm'  # 'LayerNorm' or 'RMSNorm'
pos_encoding_type = 'absolute'  # 'absolute' or 'relative'
use_teacher_forcing = False  # 是否在GRU注意力解码器中使用教师强制

data_dir = './data'
train_data_path = './data/json/train_10k.jsonl'
dev_data_path = './data/json/valid.jsonl'
test_data_path = './data/json/test.jsonl'
model_path = './experiment/model_10k.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')

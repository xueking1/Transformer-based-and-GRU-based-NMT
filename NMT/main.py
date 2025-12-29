import utils
import config
import logging
import numpy as np
import argparse
import os
import time  # 新增：导入时间库，用于统计耗时
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing


class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model, args):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(
        model.src_embed[0].d_model, 
        1, 
        args.warmup_steps,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )


# 新增：效率统计辅助函数（统一计算吞吐量和耗时）
def calculate_training_efficiency(train_dataset, epoch_start_time, epoch_end_time):
    """
    计算单Epoch训练效率
    参数：
        train_dataset: 训练数据集
        epoch_start_time: Epoch开始时间（已同步CUDA）
        epoch_end_time: Epoch结束时间（已同步CUDA）
    返回：
        epoch_duration: Epoch耗时（秒）
        samples_per_second: 样本吞吐量（samples/second）
    """
    # 计算Epoch耗时
    epoch_duration = epoch_end_time - epoch_start_time
    # 计算总样本数
    total_samples = len(train_dataset)
    # 计算吞吐量（每秒处理样本数）
    samples_per_second = total_samples / epoch_duration
    return epoch_duration, samples_per_second


# 新增：GPU显存统计函数
def log_gpu_memory_info(logger, stage="init"):
    """
    记录GPU显存占用信息（仅当使用CUDA时生效）
    参数：
        logger: 日志对象
        stage: 训练阶段（init/train/end等）
    """
    if torch.cuda.is_available():
        # 当前显存占用（MB）
        current_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        # 最大显存占用（MB）
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        # 显存缓存占用（MB）
        cached_mem = torch.cuda.memory_reserved() / (1024 * 1024)
        logger.info(f"-------- GPU Memory Info ({stage}) --------")
        logger.info(f"Current Allocated Memory: {current_mem:.2f} MB")
        logger.info(f"Max Allocated Memory: {max_mem:.2f} MB")
        logger.info(f"Cached Memory: {cached_mem:.2f} MB")
        # 重置最大显存统计（便于下一轮统计）
        torch.cuda.reset_max_memory_allocated()


def run(args):
    # 确保日志目录存在
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    utils.set_logger(args.log_path)

    # 加载数据集
    train_dataset = MTDataset(args.train_data_path)
    dev_dataset = MTDataset(args.dev_data_path)
    test_dataset = MTDataset(args.test_data_path)

    logging.info("-------- Dataset Build! --------")
    # 记录数据集大小
    logging.info(f"Train Dataset Size: {len(train_dataset)} samples")
    logging.info(f"Dev Dataset Size: {len(dev_dataset)} samples")
    logging.info(f"Test Dataset Size: {len(test_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn, pin_memory=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=False, batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn, pin_memory=True
    )

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型（补充pos_encoding_type参数）
    model = make_model(
        src_vocab=args.src_vocab_size,
        tgt_vocab=args.tgt_vocab_size,
        N=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.n_heads,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pos_encoding_type=args.pos_encoding_type,
        model_type=args.model_type,  # 新增：传入模型类型
        attn_model_type=args.attn_model_type,  # 新增：传入GRU注意力类型
        args=args  # 传入args以使用use_teacher_forcing参数
    )
    
    # 多GPU适配：仅当有多个GPU时使用DataParallel
    if torch.cuda.device_count() > 1 and args.device.startswith('cuda'):
        model_par = torch.nn.DataParallel(model)
        logging.info(f"-------- Use {torch.cuda.device_count()} GPUs --------")
    else:
        model_par = model
        logging.info("-------- Use single GPU/CPU --------")

    # 损失函数（适配LabelSmoothing的device参数）
    if args.use_smoothing:
        criterion = LabelSmoothing(
            size=args.tgt_vocab_size, 
            padding_idx=args.padding_idx, 
            smoothing=0.1,
            # 传入torch.device对象
            device=torch.device(args.device)
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum').to(torch.device(args.device))

    # 优化器
    if args.use_noamopt:
        optimizer = get_std_opt(model, args)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ========== 新增：训练效率统计 - 初始化 ==========
    # 记录模型初始化后的GPU显存（若使用CUDA）
    log_gpu_memory_info(logging, stage="model_init")
    # 记录总训练开始时间
    total_train_start_time = time.time()
    # 若使用CUDA，同步设备（确保异步操作完成后再统计时间）
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()

    # 训练+测试（封装train函数，添加Epoch级效率统计）
    # 新增：自定义训练封装（用于统计每Epoch耗时，兼容原有train函数逻辑）
    def train_with_efficiency_stats(train_dl, dev_dl, model, model_par, crit, opt, args):
        # 获取原始train函数的返回值（若有）
        epoch_times = []  # 存储每Epoch耗时
        throughput_list = []  # 存储每Epoch吞吐量

        # 重写/封装Epoch循环（若原有train函数内部有Epoch循环，此处替换为带统计的逻辑）
        # 注：若原有train函数已封装Epoch循环，可在原有函数内添加统计，此处提供通用封装
        best_loss = float('inf')
        early_stop_count = 0

        for epoch in range(args.epoch_num):
            # ========== 新增：Epoch开始时间统计 ==========
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()  # 同步CUDA
            epoch_start_time = time.time()

            logging.info(f"-------- Training Epoch {epoch+1}/{args.epoch_num} --------")
            # 调用原有训练逻辑（单Epoch训练）
            # 假设原有train函数内部可拆分为单Epoch训练，若不可拆，可直接在原有train函数中添加统计
            train_loss = train(train_dl, dev_dl, model, model_par, crit, opt, args, single_epoch=True)  # 适配单Epoch调用

            # ========== 新增：Epoch结束时间统计 ==========
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()  # 同步CUDA
            epoch_end_time = time.time()

            # 计算Epoch效率指标
            epoch_duration, samples_per_sec = calculate_training_efficiency(train_dataset, epoch_start_time, epoch_end_time)
            epoch_times.append(epoch_duration)
            throughput_list.append(samples_per_sec)

            # 记录Epoch效率日志
            logging.info(f"-------- Epoch {epoch+1} Efficiency Info --------")
            logging.info(f"Epoch {epoch+1} Duration: {epoch_duration:.2f} seconds")
            logging.info(f"Epoch {epoch+1} Throughput: {samples_per_sec:.2f} samples/second")
            logging.info(f"Epoch {epoch+1} Average Step Time: {(epoch_duration / len(train_dl)):.4f} seconds/step")

            # 早停逻辑（保持原有逻辑）
            if train_loss < best_loss:
                best_loss = train_loss
                early_stop_count = 0
                # 保存最佳模型
                torch.save(model.state_dict(), args.model_path)
                logging.info(f"Save best model at Epoch {epoch+1}, Loss: {best_loss:.4f}")
            else:
                early_stop_count += 1
                logging.info(f"Early stop count: {early_stop_count}/{args.early_stop}")
                if early_stop_count >= args.early_stop:
                    logging.info(f"Early stop at Epoch {epoch+1}")
                    break

            # 记录GPU显存（每Epoch结束）
            log_gpu_memory_info(logging, stage=f"epoch_{epoch+1}_end")

        # ========== 新增：总训练效率统计 ==========
        avg_epoch_time = np.mean(epoch_times)
        avg_throughput = np.mean(throughput_list)
        total_train_duration = sum(epoch_times)

        logging.info("-------- Total Training Efficiency Summary --------")
        logging.info(f"Total Training Epochs Run: {len(epoch_times)}")
        logging.info(f"Total Training Duration: {total_train_duration:.2f} seconds ({total_train_duration/60:.2f} minutes)")
        logging.info(f"Average Epoch Duration: {avg_epoch_time:.2f} seconds")
        logging.info(f"Average Training Throughput: {avg_throughput:.2f} samples/second")
        logging.info(f"Max Throughput: {max(throughput_list):.2f} samples/second")
        logging.info(f"Min Throughput: {min(throughput_list):.2f} samples/second")

        return best_loss, epoch_times, throughput_list

    # 调用带效率统计的训练函数
    if hasattr(train, 'single_epoch'):
        # 若原有train函数支持单Epoch调用，直接使用
        train_with_efficiency_stats(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer, args)
    else:
        # 若不支持，先统计总训练耗时，再计算整体吞吐量
        # 兼容原有train函数（无单Epoch拆分）
        logging.info("-------- Start Training (Total Efficiency Stats Later) --------")
        train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer, args)
        
        # ========== 新增：总训练耗时统计（兼容原有train函数） ==========
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        total_train_end_time = time.time()
        total_train_duration = total_train_end_time - total_train_start_time

        # 计算整体训练效率
        total_samples = len(train_dataset) * args.epoch_num  # 总训练样本数（按实际Epoch数调整）
        overall_throughput = total_samples / total_train_duration

        # 记录总效率日志
        logging.info("-------- Total Training Efficiency Summary (Overall) --------")
        logging.info(f"Total Training Duration: {total_train_duration:.2f} seconds ({total_train_duration/60:.2f} minutes)")
        logging.info(f"Overall Training Throughput: {overall_throughput:.2f} samples/second")
        logging.info(f"Total Steps: {len(train_dataloader) * args.epoch_num}")
        logging.info(f"Average Step Time: {(total_train_duration / (len(train_dataloader) * args.epoch_num)):.4f} seconds/step")

    # 记录训练结束后的GPU显存
    log_gpu_memory_info(logging, stage="train_end")

    # 测试阶段（可选：添加测试效率统计）
    logging.info("-------- Start Testing --------")
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    test_start_time = time.time()

    test(test_dataloader, model, criterion, args)

    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    test_end_time = time.time()
    test_duration = test_end_time - test_start_time
    test_throughput = len(test_dataset) / test_duration

    logging.info("-------- Test Efficiency Info --------")
    logging.info(f"Test Duration: {test_duration:.2f} seconds")
    logging.info(f"Test Throughput: {test_throughput:.2f} samples/second")


def check_opt(args):
    """check learning rate changes"""
    import matplotlib.pyplot as plt
    model = make_model(
        src_vocab=args.src_vocab_size,
        tgt_vocab=args.tgt_vocab_size,
        N=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.n_heads,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pos_encoding_type=args.pos_encoding_type,
        model_type=args.model_type,  # 新增：传入模型类型
        attn_model_type=args.attn_model_type,  # 新增：传入GRU注意力类型
        args=args
    )
    opt = get_std_opt(model, args)
    
    # Three settings of the lrate hyperparameters.
    opts = [
        opt,
        NoamOpt(512, 1, 20000, None),
        NoamOpt(256, 1, 10000, None)
    ]
    steps = np.arange(1, 50000)
    lrs = [[opt.rate(i) for opt in opts] for i in steps]
    plt.plot(steps, lrs)
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.title("Noam Optimizer Learning Rate Schedule")
    plt.show()


def one_sentence_translate(sent, args, beam_search=True):
    """单句翻译（适配新模型参数）"""
    model = make_model(
        src_vocab=args.src_vocab_size,
        tgt_vocab=args.tgt_vocab_size,
        N=args.n_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.n_heads,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pos_encoding_type=args.pos_encoding_type,
        args=args,
    )
    model = model.to(torch.device(args.device))
    
    # 分词+构建输入
    tokenizer = english_tokenizer_load()
    BOS = tokenizer.bos_id()
    EOS = tokenizer.eos_id()
    src_tokens = [[BOS] + tokenizer.EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(torch.device(args.device))
    
    # 翻译
    translate(batch_input, model, use_beam=beam_search, args=args)


def translate_example(args):
    """单句翻译示例"""
    sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
           "to childless workers."
    one_sentence_translate(sent, args, beam_search=args.beam_search)


import argparse
import torch
import config  # 确保已导入全局config模块

import argparse
import torch
import config  # 导入你的config模块

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Machine Translation Training/Inference")
    
    # ===================== 路径相关参数 =====================
    parser.add_argument('--train_data_path', type=str, 
                        default=getattr(config, 'train_data_path', './data/json/train_10k.jsonl'),
                        help='Path to training data (default: config.train_data_path or ./data/json/train_10k.jsonl)')
    parser.add_argument('--dev_data_path', type=str, 
                        default=getattr(config, 'dev_data_path', './data/json/valid.jsonl'),
                        help='Path to dev data (default: config.dev_data_path or ./data/json/valid.jsonl)')
    parser.add_argument('--test_data_path', type=str, 
                        default=getattr(config, 'test_data_path', './data/json/test.jsonl'),
                        help='Path to test data (default: config.test_data_path or ./data/json/test.jsonl)')
    parser.add_argument('--log_path', type=str, 
                        default=getattr(config, 'log_path', './experiment/train.log'),
                        help='Log file path (default: config.log_path or ./experiment/train.log)')
    
    # ===================== 模型相关参数 =====================
    parser.add_argument('--src_vocab_size', type=int, 
                        default=getattr(config, 'src_vocab_size', 32000),
                        help='Source vocabulary size (default: config.src_vocab_size or 32000)')
    parser.add_argument('--tgt_vocab_size', type=int, 
                        default=getattr(config, 'tgt_vocab_size', 32000),
                        help='Target vocabulary size (default: config.tgt_vocab_size or 32000)')
    parser.add_argument('--n_layers', type=int, 
                        default=getattr(config, 'n_layers', 6),
                        help='Number of encoder/decoder layers (default: config.n_layers or 6)')
    parser.add_argument('--d_model', type=int, 
                        default=getattr(config, 'd_model', 512),
                        help='Model dimension (default: config.d_model or 512)')
    parser.add_argument('--d_ff', type=int, 
                        default=getattr(config, 'd_ff', 2048),
                        help='Feed forward dimension (default: config.d_ff or 2048)')
    parser.add_argument('--n_heads', type=int, 
                        default=getattr(config, 'n_heads', 8),
                        help='Number of attention heads (default: config.n_heads or 8)')
    parser.add_argument('--dropout', type=float, 
                        default=getattr(config, 'dropout', 0.1),
                        help='Dropout rate (default: config.dropout or 0.1)')
    parser.add_argument('--norm_type', type=str, 
                        default=getattr(config, 'norm_type', "RMSNorm"), 
                        choices=["LayerNorm", "RMSNorm"],
                        help='Normalization type (default: config.norm_type or RMSNorm)')
    parser.add_argument('--pos_encoding_type', type=str, 
                        default=getattr(config, 'pos_encoding_type', "absolute"), 
                        choices=["absolute", "relative"],
                        help='Positional encoding type (default: config.pos_encoding_type or absolute)')
        # 新增：模型类型参数
    parser.add_argument('--model_type', type=str, 
                        default=getattr(config, 'model_type', "transformer"), 
                        choices=["transformer", "gru_attention"],
                        help='Model type to use (default: config.model_type or transformer)')
    # 新增：GRU注意力类型参数
    parser.add_argument('--attn_model_type', type=str, 
                        default=getattr(config, 'attn_model_type', "general"), 
                        choices=["dot", "general", "concat"],
                        help='Attention type for GRU attention model (default: config.attn_model_type or general)')
    parser.add_argument('--use_teacher_forcing', 
                    action='store_true',  # 命令行传入该参数则为True，不传入则使用default值
                    default=getattr(config, 'use_teacher_forcing', True),  # 优先从config读取，默认True
                    help='Whether to use teacher forcing in GRU attention decoder (default: config.use_teacher_forcing or True). '
                         'Enable for faster convergence, disable for more inference-like training.')

    # ===================== 训练相关参数 =====================
    parser.add_argument('--batch_size', type=int, 
                        default=getattr(config, 'batch_size', 32),
                        help='Batch size (default: config.batch_size or 32)')
    parser.add_argument('--lr', type=float, 
                        default=getattr(config, 'lr', 3e-4),
                        help='Learning rate (for AdamW, default: config.lr or 3e-4)')
    parser.add_argument('--use_smoothing', action='store_true', 
                        default=getattr(config, 'use_smoothing', False),
                        help='Whether to use label smoothing (default: config.use_smoothing or False)')
    parser.add_argument('--padding_idx', type=int, 
                        default=getattr(config, 'padding_idx', 0),
                        help='Padding token index (default: config.padding_idx or 0)')
    parser.add_argument('--use_noamopt', action='store_true', 
                        default=getattr(config, 'use_noamopt', True),
                        help='Whether to use Noam optimizer (default: config.use_noamopt or True)')
    parser.add_argument('--warmup_steps', type=int, 
                        default=getattr(config, 'warmup_steps', 10000),
                        help='Warmup steps for Noam optimizer (default: config.warmup_steps or 10000)')
    
    # ===================== 设备相关参数 =====================
    # 适配config里的device设置（gpu_id='0' → cuda:0）
    default_device = getattr(config, 'device', torch.device('cuda:0'))
    if isinstance(default_device, torch.device):
        default_device_str = default_device.type
        if default_device.index is not None:
            default_device_str += f":{default_device.index}"
    else:
        default_device_str = default_device
    parser.add_argument('--device', type=str, 
                        default=default_device_str,
                        help='Training device (e.g., cuda:0, cpu, default: config.device or cuda:0)')
    parser.add_argument('--cuda_visible_devices', type=str, 
                        default=getattr(config, 'cuda_visible_devices', '0'),
                        help='CUDA visible devices (default: config.cuda_visible_devices or 0,1)')

    # ===================== 训练控制参数 =====================
    parser.add_argument('--epoch_num', type=int, 
                        default=getattr(config, 'epoch_num', 40),
                        help='Training epochs (default: config.epoch_num or 40)')
    parser.add_argument('--early_stop', type=int, 
                        default=getattr(config, 'early_stop', 5),
                        help='Early stop patience (default: config.early_stop or 5)')
    parser.add_argument('--model_path', type=str, 
                        default=getattr(config, 'model_path', './experiment/model_10k.pth'),
                        help='Model save path (default: config.model_path or ./experiment/model_10k.pth)')
    parser.add_argument('--output_path', type=str, 
                        default=getattr(config, 'output_path', './experiment/output.txt'),
                        help='Test output path (default: config.output_path or ./experiment/output.txt)')

    # ===================== 解码相关参数 =====================
    parser.add_argument('--max_len', type=int, 
                        default=getattr(config, 'max_len', 60),
                        help='Max decode length (default: config.max_len or 60)')
    parser.add_argument('--bos_idx', type=int, 
                        default=getattr(config, 'bos_idx', 2),
                        help='BOS token index (default: config.bos_idx or 2)')
    parser.add_argument('--eos_idx', type=int, 
                        default=getattr(config, 'eos_idx', 3),
                        help='EOS token index (default: config.eos_idx or 3)')
    parser.add_argument('--beam_size', type=int, 
                        default=getattr(config, 'beam_size', 3),
                        help='Beam size for beam search (default: config.beam_size or 3)')
    
    # ===================== 翻译相关参数 =====================
    parser.add_argument('--beam_search', action='store_true', 
                        default=getattr(config, 'beam_search', False),
                        help='Whether to use beam search in translation (default: config.beam_search or True)')
    
    # ===================== 运行模式参数 =====================
    parser.add_argument('--mode', type=str, 
                        default=getattr(config, 'mode', 'train'), 
                        choices=['train', 'test', 'translate', 'check_opt'],
                        help='Running mode: train/test/translate/check_opt (default: config.mode or train)')
    # TensorBoard参数
    parser.add_argument('--tb_log_dir', type=str, default='./experiment/tb_logs', help='TensorBoard log directory')

    # ========== 新增：效率统计开关（可选） ==========
    parser.add_argument('--enable_efficiency_stats', action='store_true', default=True,
                        help='Whether to enable training efficiency statistics (default: True)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 设置CUDA可见设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    
    # 核心修复：args.device是字符串，可安全调用startswith
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        logging.warning("CUDA is not available, fallback to CPU")
        args.device = 'cpu'
    
    # 根据运行模式执行逻辑
    if args.mode == 'train':
        run(args)
    elif args.mode == 'test':
        # 初始化模型+测试
        model = make_model(
            src_vocab=args.src_vocab_size,
            tgt_vocab=args.tgt_vocab_size,
            N=args.n_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            h=args.n_heads,
            dropout=args.dropout,
            norm_type=args.norm_type,
            pos_encoding_type=args.pos_encoding_type,
            model_type=args.model_type,  # 新增：传入模型类型
            attn_model_type=args.attn_model_type,  # 新增：传入GRU注意力类型
            args=args
        ).to(torch.device(args.device))
        
        # 损失函数
        if args.use_smoothing:
            criterion = LabelSmoothing(
                size=args.tgt_vocab_size, 
                padding_idx=args.padding_idx, 
                smoothing=0.1,
                device=torch.device(args.device)
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum').to(torch.device(args.device))
        
        # 加载测试集
        test_dataset = MTDataset(args.test_data_path)
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size,
            collate_fn=test_dataset.collate_fn, pin_memory=True
        )

        # ========== 新增：测试效率统计 ==========
        logging.info("-------- Start Testing --------")
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        test_start_time = time.time()

        test(test_dataloader, model, criterion, args)

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        test_throughput = len(test_dataset) / test_duration

        logging.info("-------- Test Efficiency Info --------")
        logging.info(f"Test Duration: {test_duration:.2f} seconds")
        logging.info(f"Test Throughput: {test_throughput:.2f} samples/second")
    
    elif args.mode == 'translate':
        translate_example(args)
    
    elif args.mode == 'check_opt':
        check_opt(args)
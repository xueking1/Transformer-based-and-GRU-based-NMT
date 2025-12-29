import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm

# 不再依赖全局config，改为通过args传参
# import config  
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import chinese_tokenizer_load


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


# -------------------------- 修改1：增加args参数，替换所有config为args --------------------------
def train(train_data, dev_data, model, model_par, criterion, optimizer, args):
    """训练并保存模型（适配args参数，移除全局config依赖）"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = args.early_stop  # 替换config.early_stop
    # 处理device_id：从args.device提取（如cuda:0 → [0]）
    device_id = [int(args.device.split(':')[1])] if ':' in args.device and args.device.startswith('cuda') else [0]
    
    for epoch in range(1, args.epoch_num + 1):  # 替换config.epoch_num
        # 模型训练
        model.train()
        train_loss = run_epoch(
            train_data, model_par,
            MultiGPULossCompute(
                model.generator, criterion, 
                device_id,  # 替换config.device_id
                optimizer,
                chunk_size=5,
                args=args  # 传入args
            )
        )
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        
        # 模型验证
        model.eval()
        dev_loss = run_epoch(
            dev_data, model_par,
            MultiGPULossCompute(
                model.generator, criterion, 
                device_id,  # 替换config.device_id
                None
            )
        )
        bleu_score = evaluate(dev_data, model, mode='dev', use_beam=args.beam_search, args=args)  # 传入args
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            # 确保模型保存目录存在
            import os
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            torch.save(model.state_dict(), args.model_path)  # 替换config.model_path
            best_bleu_score = bleu_score
            early_stop = args.early_stop  # 替换config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""
    # -------------------------- 修改2：增加args参数，替换config.use_noamopt --------------------------
    def __init__(self, generator, criterion, opt=None, args=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.args = args  # 保存args参数

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            # 替换config.use_noamopt为self.args.use_noamopt
            if self.args and self.args.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""
    # -------------------------- 修改3：增加args参数，替换config.use_noamopt --------------------------
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5, args=None):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        self.args = args  # 保存args参数

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            # 替换config.use_noamopt为self.args.use_noamopt
            if self.args and self.args.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


# -------------------------- 修改4：增加args参数，替换所有config为args --------------------------
def evaluate(data, model, mode='dev', use_beam=True, args=None):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam: 
                decode_result, _ = beam_search(model, src, src_mask, args.max_len,
                                            args.padding_idx, args.bos_idx, args.eos_idx,
                                            args.beam_size, args.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=args.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(args.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


# -------------------------- 修改5：增加args参数，替换所有config为args --------------------------
def test(data, model, criterion, args):
    """测试模型（适配args参数）"""
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))  # 替换config.model_path，增加map_location
        model_par = torch.nn.DataParallel(model)
        model.eval()
        
        # 处理device_id
        device_id = [int(args.device.split(':')[1])] if ':' in args.device and args.device.startswith('cuda') else [0]
        
        # 开始预测
        test_loss = run_epoch(
            data, model_par,
            MultiGPULossCompute(
                model.generator, criterion, 
                device_id,  # 替换config.device_id
                None,
                args=args  # 传入args
            )
        )
        bleu_score = evaluate(data, model, 'test', use_beam=args.beam_search, args=args)  # 传入args
        print(f'Test loss: {test_loss:.4f},  Bleu Score: {bleu_score:.2f}')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


# -------------------------- 修改6：增加args参数，替换所有config为args --------------------------
def translate(src, model, use_beam=True, args=None):
    """用训练好的模型进行预测单句，打印模型翻译结果（适配args参数）"""
    sp_chn = chinese_tokenizer_load()
    with torch.no_grad():
        # 加载模型（增加map_location适配不同设备）
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))  # 替换config.model_path
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(
                model, src, src_mask, 
                args.max_len,  # 替换config.max_len
                args.padding_idx,  # 替换config.padding_idx
                args.bos_idx,  # 替换config.bos_idx
                args.eos_idx,  # 替换config.eos_idx
                args.beam_size,  # 替换config.beam_size
                args.device  # 替换config.device
            )
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(
                model, src, src_mask,
                max_len=args.max_len  # 替换config.max_len
            )
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        print("翻译结果：", translation[0])
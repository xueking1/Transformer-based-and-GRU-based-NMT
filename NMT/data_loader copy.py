import torch
import json
import numpy as np
import jieba
import sentencepiece as spm  # 保留英文加载使用
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from typing import List, Tuple
import config

DEVICE = config.device


# ===================== 核心新增：模拟SentencePiece接口的JiebaTokenizer类 =====================
class JiebaTokenizer:
    """模拟SentencePieceProcessor接口的Jieba分词器，保证方法名/返回值完全兼容"""
    def __init__(self, vocab_path: str = "./tokenizer/chn.vocab"):
        # 启用paddle模式（匹配你提供的jieba调用范例）
        jieba.enable_paddle()
        # 加载词汇表
        self.word2id = self._load_vocab(vocab_path)
        self.id2word = {v: k for k, v in self.word2id.items()}
        # 固定特殊标记ID（与原有SentencePiece一致）
        self._pad_id = 0
        self._unk_id = 1
        self._bos_id = 2
        self._eos_id = 3

    def _load_vocab(self, vocab_path: str) -> dict:
        """加载jieba词汇表（格式：词\tID\t词频）"""
        word2id = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    idx = int(parts[1])
                    word2id[word] = idx
        return word2id

    # 以下方法完全匹配SentencePieceProcessor的接口
    def EncodeAsIds(self, text: str) -> List[int]:
        """核心方法：分词并转ID，替代原有sp_chn.EncodeAsIds"""
        # jieba paddle模式分词（匹配调用范例）
        seg_words = list(jieba.cut(text, use_paddle=True))
        # 转ID，未知词用UNK_ID
        ids = [self.word2id.get(word, self._unk_id) for word in seg_words]
        return ids

    def pad_id(self) -> int:
        return self._pad_id

    def unk_id(self) -> int:
        return self._unk_id

    def bos_id(self) -> int:
        return self._bos_id

    def eos_id(self) -> int:
        return self._eos_id

    # 可选：模拟其他常用方法（如需解码可添加）
    def DecodeIds(self, ids: List[int]) -> str:
        """解码ID序列为文本（可选，匹配spm的DecodeIds）"""
        words = [self.id2word.get(idx, '<unk>') for idx in ids]
        return ''.join(words)


# ===================== 仿照原有写法的加载函数 =====================
def chinese_tokenizer_load():
    """
    仿照原有chinese_tokenizer_load写法，返回兼容接口的JiebaTokenizer对象
    替换原有加载spm模型的逻辑，但函数名/返回值接口完全一致
    """
    # 原逻辑：加载./tokenizer/chn.model
    # 新逻辑：初始化JiebaTokenizer，加载./tokenizer/chn.vocab
    jieba_tokenizer = JiebaTokenizer(vocab_path="./tokenizer/chn.vocab")
    return jieba_tokenizer


# ===================== 原有代码（仅恢复chinese_tokenizer_load导入，其余无改动） =====================
def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        # 恢复原有调用方式，无需修改！
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    
    def len_argsort(self, seq: List[str]) -> List[int]:
        """补充依赖方法：按句子长度排序并返回下标（若类中已有可删除）"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def _auto_load_json_or_jsonl(self, data_path: str) -> List[List[str]]:
        """
        精简版：自动识别JSON/JSONL格式并读取，仅保留核心读取功能
        :param data_path: 文件路径
        :return: 二维列表 [[英文句子, 中文句子], ...]
        """
        # 第一步：尝试按JSON格式读取（二维列表）
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(item, list) and len(item) >= 2 for item in data):
                return [[item[0].strip(), item[1].strip()] for item in data]
        except (json.JSONDecodeError, ValueError):
            pass

        # 第二步：降级为JSONL格式读取（兼容列表/字典）
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json_obj = json.loads(line)
                if isinstance(json_obj, list) and len(json_obj) >= 2:
                    data.append([json_obj[0].strip(), json_obj[1].strip()])
                elif isinstance(json_obj, dict) and "en" in json_obj and "zh" in json_obj:  # 适配你实际的zh键
                    data.append([json_obj["en"].strip(), json_obj["zh"].strip()])
        
        return data

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = self._auto_load_json_or_jsonl(data_path)
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        # 原有调用方式完全保留，无需修改！
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
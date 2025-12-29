import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = config.device


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    集成绝对/相对位置编码的统一类（优化后健壮版本）
    :param d_model: 模型维度（绝对位置编码）/注意力头维度d_k（相对位置编码）
    :param dropout: dropout概率（仅绝对位置编码使用）
    :param max_len: 最大序列长度（相对位置编码的最大偏移范围）
    :param pos_encoding_type: 编码类型，可选 "absolute"（绝对）/ "relative"（相对）
    :param device: 初始化设备（None则动态适配）
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, pos_encoding_type="absolute", device=None):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding_type = pos_encoding_type
        self.d_model = d_model
        self.max_len = max_len  # 相对位置编码的最大正负偏移范围
        self.dropout = nn.Dropout(p=dropout) if pos_encoding_type == "absolute" else None

        # 绝对位置编码初始化（兼容原有逻辑）
        if pos_encoding_type == "absolute":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)
        
        # 相对位置编码初始化（优化：明确embedding词典大小，添加注释）
        elif pos_encoding_type == "relative":
            # 相对位置索引范围：0 ~ 2*max_len（对应 -max_len ~ +max_len）
            self.rel_pos_vocab_size = 2 * self.max_len + 1
            self.rel_pos_embed = nn.Embedding(self.rel_pos_vocab_size, d_model)
            # 初始化嵌入层参数（保持原有逻辑）
            nn.init.normal_(self.rel_pos_embed.weight, std=0.02)
            # 设备适配（优化：严谨的设备迁移）
            if device is not None:
                self.rel_pos_embed = self.rel_pos_embed.to(device)

    def get_rel_pos_index(self, q_len, k_len, device):
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)

        rel_pos = k_pos - q_pos
        rel_pos = torch.clamp(rel_pos, -self.max_len, self.max_len)
        rel_pos_idx = rel_pos + self.max_len

        return rel_pos_idx

    def forward(self, x, k_len=None):
        """
        :param x: [B, L, D]
        :param k_len: key 序列长度（None 表示自注意力）
        :return:
            - absolute: [B, L, D]
            - relative: [B, L, D]
        """
        device = x.device
        B, q_len, D = x.shape

        # ========= 绝对位置编码 =========
        if self.pos_encoding_type == "absolute":
            pe = self.pe[:, :q_len].to(device)
            x = x + Variable(pe, requires_grad=False)
            return self.dropout(x)

        # ========= 相对位置编码 =========
        elif self.pos_encoding_type == "relative":
            if k_len is None:
                k_len = q_len  # 自注意力

            # [q_len, k_len]
            rel_pos_idx = self.get_rel_pos_index(q_len, k_len, device)

            # [q_len, k_len, d_model]
            rel_pos_emb = self.rel_pos_embed(rel_pos_idx)

            # === 核心：压缩 key 维度 ===
            # [q_len, d_model]
            rel_pos_emb = rel_pos_emb.mean(dim=1)

            # === 扩展 batch 维度 ===
            # [B, q_len, d_model]
            rel_pos_emb = rel_pos_emb.unsqueeze(0).expand(B, -1, -1)

            return rel_pos_emb


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 返回Layer Norm的结果
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self, size, dropout, norm_type="LayerNorm"):
        super(SublayerConnection, self).__init__()
        # 【修改2】根据norm_type选择归一化方式
        self.norm = nn.RMSNorm(size) if norm_type == "RMSNorm" else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    # 【修改3】添加norm_type参数，默认使用LayerNorm
    def __init__(self, layer, N, norm_type="LayerNorm"):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # 【修改4】根据norm_type选择归一化方式，删除多余的self.rmsnorm
        self.norm = nn.RMSNorm(layer.size) if norm_type == "RMSNorm" else LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    # 【修改5】添加norm_type参数，默认使用LayerNorm
    def __init__(self, size, self_attn, feed_forward, dropout, norm_type="LayerNorm"):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 【修改6】传递norm_type参数给SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    # 【修改7】添加norm_type参数，默认使用LayerNorm
    def __init__(self, layer, N, norm_type="LayerNorm"):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # 【修改8】根据norm_type选择归一化方式
        self.norm = nn.RMSNorm(layer.size) if norm_type == "RMSNorm" else LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    # 【修改9】添加norm_type参数，默认使用LayerNorm
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, norm_type="LayerNorm"):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 【修改10】传递norm_type参数给SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


# 新增：GRU专属Generator（适配hidden_size*2输入维度）
class GRUGenerator(nn.Module):
    def __init__(self, hidden_size, vocab):
        super(GRUGenerator, self).__init__()
        # GRU解码器输出是rnn_output(hidden_size) + context(hidden_size)，拼接后维度为hidden_size*2
        self.proj = nn.Linear(hidden_size * 2, vocab)

    def forward(self, x):
        # 保持与Transformer Generator一致的输出格式（log_softmax）
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(
    src_vocab,     # 源词汇表大小
    tgt_vocab,     # 目标词汇表大小
    N=6,           # Transformer的编码器和解码器层数
    d_model=512,   # 模型的维度，即词嵌入的维度
    d_ff=2048,     # 前馈神经网络中隐藏层的大小
    h=8,           # 多头注意力机制中的头数
    dropout=0.1,   # dropout概率，用于防止过拟合
    norm_type="LayerNorm",
    pos_encoding_type="absolute",
    model_type="transformer",  # 新增：切换模型类型
    attn_model_type="general",       # 新增：GRU注意力模型的注意力类型
    args=None
):
    c = copy.deepcopy
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if 'DEVICE' not in locals() else DEVICE

    # 分支1：构建原有Transformer模型
    if model_type == "transformer":
        # 实例化Attention对象
        attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        # 实例化FeedForward对象
        ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
        # 实例化PositionalEncoding对象
        position = PositionalEncoding(d_model, dropout, pos_encoding_type=pos_encoding_type).to(DEVICE)
        # 实例化Transformer模型对象
        model = Transformer(
            # 传递norm_type参数给EncoderLayer和Encoder
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, norm_type).to(DEVICE), N, norm_type).to(DEVICE),
            # 传递norm_type参数给DecoderLayer和Decoder
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, norm_type).to(DEVICE), N, norm_type).to(DEVICE),
            nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
            Generator(d_model, tgt_vocab).to(DEVICE)
        ).to(DEVICE)

    # 分支2：构建GRU注意力模型（整合EncoderRNN+AttentionDecoderRNN）
    elif model_type == "gru_attention":
        # 初始化编码器：src_vocab对应input_size，d_model对应hidden_size，N对应n_layers
        encoder = EncoderRNN(
            input_size=src_vocab,
            hidden_size=d_model,
            n_layers=N
        ).to(DEVICE)
        # 初始化解码器：attn_model对应注意力类型，d_model对应hidden_size，tgt_vocab对应output_size
        decoder = AttentionDecoderRNN(
            attention_model=attn_model_type,
            hidden_size=d_model,
            output_size=tgt_vocab,
            n_layers=N,
            dropout_p=dropout
        ).to(DEVICE)
        # 封装为统一接口的GRU注意力模型（传入tgt_vocab用于初始化generator）
        model = GRUAttentionModel(encoder, decoder, tgt_vocab).to(DEVICE)

    else:
        raise ValueError(f"不支持的模型类型：{model_type}，可选值为'transformer'/'gru_attention'")

    # 统一参数初始化（和原有逻辑一致：Glorot / fan_avg）
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(DEVICE)


import torch
from data_loader import subsequent_mask  # 确保导入后续掩码函数

import torch
from data_loader import subsequent_mask  # 确保导入后续掩码函数

def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """
    批量贪心解码函数（Greedy Decoding）
    功能：为一批源语言序列（src）生成对应的目标语言序列，采用贪心策略，返回格式与束搜索collect_hypothesis_and_scores对齐
    参数说明：
        model: 训练完成的NMT模型（包含encoder、decoder、generator）
        src: 源语言输入张量，形状 [batch_size, src_seq_len]（长整型）
        src_mask: 源语言掩码张量，形状 [batch_size, 1, src_seq_len]
        max_len: 目标序列最大生成长度，默认64
        start_symbol: BOS标记token id，默认2
        end_symbol: EOS标记token id，默认3
    返回值：
        all_hyp: list[list[list[int]]]，三维列表（与束搜索结果格式对齐），
                外层=batch_size（样本数），中层=1（贪心解码仅1个候选，对应束搜索n_best=1），内层=token id序列
        all_scores: list[list[float]]，二维列表（与束搜索得分格式对齐），
                    外层=batch_size（样本数），中层=1（对应唯一候选，得分默认设为0.0，可自定义）
    """
    # 获取批次大小
    batch_size, src_seq_len = src.size()
    # 初始化停止标记列表和完成计数（用于提前终止）
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    # ========== 对齐collect_hypothesis_and_scores：初始化空列表用于合并 ==========
    all_hyp, all_scores = [], []  # 与束搜索函数初始化方式完全一致

    # 编码器编码源序列
    memory = model.encode(src, src_mask)
    # 初始化目标序列（长整型，匹配模型输入类型）
    tgt = torch.LongTensor(batch_size, 1).fill_(start_symbol).type_as(src)

    # 初始化每个样本的贪心生成序列（临时存储，后续按格式合并）
    single_hyp_list = [[] for _ in range(batch_size)]

    # 循环解码
    for s in range(max_len):
        # 构建目标序列掩码
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src)
        # 解码器解码（移除Variable，兼容新版PyTorch）
        out = model.decode(memory, src_mask, tgt, tgt_mask)
        # 预测下一个token的概率分布并贪心选择
        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)
        
        # 更新目标序列
        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()

        # 遍历样本处理预测结果
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    # 存储Python int类型的token id
                    single_hyp_list[i].append(int(pred[i].item()))
            
            # 所有样本完成，提前跳出内层循环
            if count == batch_size:
                break
        
        # 所有样本完成，提前跳出外层解码循环
        if count == batch_size:
            break

    # ========== 核心：按collect_hypothesis_and_scores的合并方式组装结果 ==========
    for inst_idx in range(batch_size):
        # 1. 处理当前样本的候选序列：贪心解码仅1个候选，包装为[token_seq]（中层列表，对齐束搜索n_best格式）
        hyp = [single_hyp_list[inst_idx]]  # 与束搜索hyps格式一致（中层列表长度=1）
        # 用 += [hyp] 方式合并，与collect_hypothesis_and_scores的all_hyp += [hyps] 完全对齐
        all_hyp += [hyp]

    # 返回与束搜索格式一致的结果
    return all_hyp

def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """传入一个训练好的模型，对指定数据进行预测"""
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                        src_mask,
                        Variable(ys),
                        Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


class Attention(nn.Module):
    """Attention层（矩阵批量运算版，无Python串行循环，大幅提升速度）"""
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
            self.other = nn.Parameter(torch.randn(1, self.hidden_size))
        # dot方法无需额外层

    def forward(self, hidden, encoder_outputs):
        """
        批量矩阵运算：无串行循环
        - hidden: [batch_size, hidden_size]
        - encoder_outputs: [src_seq_len, batch_size, hidden_size]
        - return: attention_weights [batch_size, 1, src_seq_len]
        """
        batch_size = hidden.size(0)
        src_seq_len = encoder_outputs.size(0)

        # ========= 核心修改：去掉for循环，矩阵批量计算energies =========
        if self.method == 'dot':
            # 步骤1：调整encoder_outputs形状 → [batch_size, hidden_size, src_seq_len]
            encoder_outputs_t = encoder_outputs.transpose(0, 1).transpose(1, 2)  # [B, H, S_src]
            # 步骤2：hidden扩展维度 → [batch_size, hidden_size, 1]
            hidden_exp = hidden.unsqueeze(2)  # [B, H, 1]
            # 步骤3：矩阵乘法一次性计算所有时间步分数 → [B, S_src, 1] → [B, S_src]
            energies = torch.bmm(encoder_outputs_t.transpose(1, 2), hidden_exp).squeeze(2)  # [B, S_src]

        elif self.method == 'general':
            # 步骤1：批量处理所有encoder输出（无需逐时间步）→ [S_src, B, H] → [B, S_src, H]
            encoder_outputs_batch = encoder_outputs.transpose(0, 1)  # [B, S_src, H]
            # 步骤2：线性层批量映射 → [B, S_src, H]
            encoder_outputs_proj = self.attention(encoder_outputs_batch)  # 批量运算，无循环
            # 步骤3：矩阵乘法一次性计算分数 → [B, S_src]
            hidden_exp = hidden.unsqueeze(2)  # [B, H, 1]
            energies = torch.bmm(encoder_outputs_proj, hidden_exp).squeeze(2)  # [B, S_src]

        elif self.method == 'concat':
            # 步骤1：hidden扩展维度 → [B, 1, H]，重复src_seq_len次 → [B, S_src, H]
            hidden_exp = hidden.unsqueeze(1).repeat(1, src_seq_len, 1)  # [B, S_src, H]
            # 步骤2：encoder_outputs调整形状 → [B, S_src, H]
            encoder_outputs_batch = encoder_outputs.transpose(0, 1)  # [B, S_src, H]
            # 步骤3：批量拼接 → [B, S_src, 2H]
            concat_input = torch.cat((hidden_exp, encoder_outputs_batch), dim=2)  # [B, S_src, 2H]
            # 步骤4：批量计算 → [B, S_src, H]
            energy_proj = torch.tanh(self.attention(concat_input))  # [B, S_src, H]
            # 步骤5：矩阵乘法一次性计算分数 → [B, S_src]
            other_exp = self.other.repeat(batch_size, 1, 1)  # [B, 1, H]
            energies = torch.bmm(other_exp, energy_proj.transpose(1, 2)).squeeze(1)  # [B, S_src]
        else:
            raise ValueError(f"不支持的注意力方法：{self.method}")

        # 无需逐时间步填充，直接得到完整energies，效率大幅提升
        attention_weights = F.softmax(energies, dim=1).unsqueeze(1)  # [B, 1, S_src]
        return attention_weights

    def _score(self, hidden, encoder_output):
        """
        修正：替换dot()，使用高维兼容的运算方式计算注意力分数
        - hidden: [batch_size, hidden_size]
        - encoder_output: [batch_size, hidden_size]
        - return: [batch_size]（每个样本的注意力分数）
        """
        if self.method == 'dot':
            # 替换 dot()：元素相乘后按hidden维度求和，等价于批量版dot
            # 结果 [batch_size]，兼容2D张量
            return torch.sum(hidden * encoder_output, dim=1)

        elif self.method == 'general':
            # 先通过线性层处理encoder输出，再元素相乘求和
            energy = self.attention(encoder_output)  # [batch_size, hidden_size]
            return torch.sum(hidden * energy, dim=1)  # [batch_size]

        elif self.method == 'concat':
            # 拼接hidden和encoder输出，再计算分数
            energy = self.attention(torch.cat((hidden, encoder_output), 1))  # [batch_size, hidden_size]
            energy = energy.t()  # [hidden_size, batch_size]
            # 与self.other做矩阵乘法后转置，得到 [batch_size]
            return torch.sum(self.other * energy, dim=0)

        else:
            raise ValueError(f"不支持的注意力方法：{self.method}，可选 dot/general/concat")

class EncoderRNN(nn.Module):
    """Recurrent neural network that encodes a given input sequence（支持批量输入）."""

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        # word_inputs: [src_seq_len, batch_size]（批量输入格式）
        embedded = self.embedding(word_inputs)  # [src_seq_len, batch_size, hidden_size]
        output, hidden = self.gru(embedded, hidden)  # 直接批量处理，无需拆分样本
        # output: [src_seq_len, batch_size, hidden_size]；hidden: [n_layers, batch_size, hidden_size]
        return output, hidden

    def init_hidden(self, batch_size=1):
        # 支持批量隐藏状态初始化
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=next(self.parameters()).device)
        return hidden


class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded inputs using attention."""

    def __init__(self, attention_model, hidden_size, output_size, n_layers=1, dropout_p=.1):
        super(AttentionDecoderRNN, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attention_model is not None:
            self.attention = Attention(attention_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """Run forward propagation one step at a time.
        
        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden state and context vector.
        
        Args:
            word_input: torch Variable representing the word input constituent
            last_context: torch Variable representing the previous context
            last_hidden: torch Variable representing the previous hidden state output
            encoder_outputs: torch Variable containing the encoder output values
            
        Return:
            output: torch Variable representing the predicted word constituent 
            context: torch Variable representing the context value
            hidden: torch Variable representing the hidden state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        # Run through RNN
        word_embedded = self.embedding(word_input)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        # rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # 修改后：对 last_hidden 调用 .contiguous()
        if last_hidden is not None:
            last_hidden = last_hidden.contiguous()  # 转为连续张量，解决报错
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # Calculate attention
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Predict output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = torch.cat((rnn_output, context), 1)
        # output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attention_weights


class GRUAttentionModel(nn.Module):
    """
    GRU + Attention 编解码模型（Transformer 风格接口）
    - encode(): 仅编码
    - decode(): 仅解码（支持教师强制 / 自回归）
    """

    def __init__(self, encoder, decoder, tgt_vocab):
        super(GRUAttentionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab = tgt_vocab
        self.hidden_size = encoder.hidden_size
        self.n_layers = encoder.n_layers

        # GRU 专属 Generator（2H -> vocab）
        self.generator = GRUGenerator(
            hidden_size=self.hidden_size,
            vocab=tgt_vocab
        ).to(next(encoder.parameters()).device)

        # ===== 为了接口对齐 Transformer =====
        class DummyEmbedding(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.d_model = hidden_size
            def forward(self, x):
                return x

        self.src_embed = nn.Sequential(DummyEmbedding(self.hidden_size))
        self.d_model = self.hidden_size

    # ======================================================
    # Encoder
    # ======================================================
    def encode(self, src, src_mask=None):
        """
        Args:
            src: [B, src_len]
        Returns:
            dict: 合并后的单一返回值，键名明确对应原始张量
        """
        batch_size = src.size(0)

        # GRU Encoder 需要 [src_len, B]
        src = src.transpose(0, 1)

        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(src, encoder_hidden)

        # 合并为字典（单一返回值），键名自定义（清晰易懂）
        combined_tensor = torch.cat([encoder_outputs, encoder_hidden], dim=0)
        return combined_tensor.permute(1, 0, 2)  # [B, src_len + n_layers, H]

    # ======================================================
    # Decoder（Transformer 风格）
    # ======================================================
    def decode(
        self,
        combined_tensor=None,
        tgt=None,
        tgt_mask=None,
        max_len=None,
        use_teacher_forcing=True,
        bos_token_idx=0,
        eos_token_idx=1,
    ):
        """
        Args:
            encoder_outputs: [src_len, B, H]
            encoder_hidden:  [n_layers, B, H]
            tgt:             [B, T]（教师强制时必需）
        Returns:
            outputs: [B, T, 2H]
        """
        combined_tensor = combined_tensor.permute(1, 0, 2)  # [src_len + n_layers, B, H]
        encoder_outputs=combined_tensor[:-self.n_layers]
        encoder_hidden = combined_tensor[-self.n_layers:]
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(1)

        if use_teacher_forcing:
            assert tgt is not None, "教师强制模式必须提供 tgt"
            max_len = tgt.size(1)
        else:
            assert max_len is not None, "非教师强制模式必须指定 max_len"

        # ===== 初始化 Decoder 状态 =====
        decoder_input = torch.full(
            (1, batch_size),
            bos_token_idx,
            dtype=torch.long,
            device=device
        )  # [1, B]

        decoder_context = torch.zeros(
            batch_size, self.hidden_size, device=device
        )  # [B, H]

        decoder_hidden = encoder_hidden

        outputs = []
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # ===== 时间步循环 =====
        for t in range(max_len):
            output_step, decoder_context, decoder_hidden, attn = self.decoder(
                decoder_input,
                decoder_context,
                decoder_hidden,
                encoder_outputs
            )
            # output_step: [B, 2H]
            outputs.append(output_step)

            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(0)
            else:
                probs = self.generator(output_step)  # [B, V]
                next_tokens = probs.argmax(dim=1)
                decoder_input = next_tokens.unsqueeze(0)

                finished_mask |= (next_tokens == eos_token_idx)
                if torch.all(finished_mask):
                    break
        outputs = torch.stack(outputs, dim=1)  # [B, T, 2H]
        return outputs

    # ======================================================
    # forward（thin wrapper）
    # ======================================================
    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        use_teacher_forcing=True,
        **kwargs
    ):
        combined_tensor = self.encode(src, src_mask)

        return self.decode(
            combined_tensor,
            tgt=tgt,
            tgt_mask=tgt_mask,
            use_teacher_forcing=use_teacher_forcing,
            **kwargs
        )